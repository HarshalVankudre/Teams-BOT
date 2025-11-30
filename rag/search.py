"""
RAG Search - Semantic retrieval and response generation.
Searches Pinecone for documents and machinery.
Hybrid RAG routes structured queries to PostgreSQL via orchestrator.
"""
import time
import json
import os
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
import pinecone
from .config import config


def model_supports_reasoning(model_name: str) -> bool:
    """Check if a model supports the reasoning.effort parameter.
    Only OpenAI's o-series models (o1, o3, etc.) support reasoning."""
    if not model_name:
        return False
    model_lower = model_name.lower()
    # o1, o1-mini, o1-pro, o3, o3-mini etc. support reasoning
    return model_lower.startswith('o1') or model_lower.startswith('o3')
from .vector_store import PineconeStore
from .embeddings import EmbeddingService

# Hybrid RAG with PostgreSQL
try:
    from .hybrid_orchestrator import HybridOrchestrator, QueryType
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    print("[WARNING] Hybrid orchestrator not available.")

# Tavily for supplementary web search
try:
    from tavily import AsyncTavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    print("[WARNING] tavily-python not installed. Web search disabled.")


class RAGSearch:
    """
    RAG Search with hybrid routing.
    PostgreSQL handles structured queries via orchestrator.
    Pinecone handles semantic queries.
    """

    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.vector_store = PineconeStore()
        self.embedding_service = EmbeddingService()

        # Model settings from config
        self.model = config.response_model
        self.reasoning_effort = config.response_reasoning

        if not self.model or not self.reasoning_effort:
            raise ValueError("OPENAI_MODEL and REASONING_EFFORT must be set in .env")

        print(f"[RAG] Model: {self.model}, Reasoning: {self.reasoning_effort}")

        # Tavily for supplementary web search
        self.enable_web_search = config.enable_web_search and TAVILY_AVAILABLE and config.tavily_api_key
        self.tavily_client = None
        if self.enable_web_search:
            self.tavily_client = AsyncTavilyClient(api_key=config.tavily_api_key)
            print("[RAG] Tavily: Enabled")
        else:
            print("[RAG] Tavily: Disabled")

        # Pinecone index
        self.pc = pinecone.Pinecone(api_key=config.pinecone_api_key)
        self.index = self.pc.Index(host=config.pinecone_host)
        self.machinery_namespace = config.pinecone_machinery_namespace
        self.documents_namespace = config.pinecone_namespace

        # Hybrid orchestrator (PostgreSQL + Pinecone routing)
        self.use_hybrid_rag = os.getenv("USE_HYBRID_RAG", "false").lower() == "true"
        self.hybrid_orchestrator = None
        if self.use_hybrid_rag and HYBRID_AVAILABLE:
            try:
                self.hybrid_orchestrator = HybridOrchestrator(
                    openai_client=self.client,
                    verbose=True
                )
                print("[RAG] Hybrid RAG: Enabled (PostgreSQL + Pinecone)")
            except Exception as e:
                print(f"[RAG] Hybrid RAG initialization failed: {e}")
                self.use_hybrid_rag = False
        else:
            print("[RAG] Hybrid RAG: Disabled")

    async def search_pinecone(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search both document and machinery namespaces in Pinecone"""
        query_embedding = await self.embedding_service.embed_query(query)

        # Build Pinecone filter
        pinecone_filter = None
        if filters:
            pinecone_filter = {}
            for key, value in filters.items():
                if isinstance(value, dict):
                    pinecone_filter[key] = value
                elif isinstance(value, list):
                    pinecone_filter[key] = {"$in": value}
                else:
                    pinecone_filter[key] = {"$eq": value}

        all_results = []

        # Search documents namespace
        try:
            doc_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=self.documents_namespace,
                include_metadata=True,
                filter=pinecone_filter
            )
            for match in doc_results.matches:
                all_results.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata,
                    "namespace": "documents",
                    "content": match.metadata.get("content", ""),
                    "title": match.metadata.get("title", ""),
                    "source_file": match.metadata.get("source_file", "")
                })
        except Exception as e:
            print(f"[Search] Documents error: {e}")

        # Search machinery namespace
        try:
            machinery_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=self.machinery_namespace,
                include_metadata=True,
                filter=pinecone_filter
            )
            for match in machinery_results.matches:
                metadata = match.metadata or {}
                all_results.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": metadata,
                    "namespace": "machinery",
                    "content": metadata.get("inhalt", ""),
                    "title": metadata.get("titel", ""),
                    "source_file": "machinery-database"
                })
        except Exception as e:
            print(f"[Search] Machinery error: {e}")

        # Sort by score
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results

    async def tavily_search(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """Supplementary web search via Tavily"""
        if not self.enable_web_search or not self.tavily_client:
            return []

        max_results = max_results or config.web_search_max_results

        try:
            response = await self.tavily_client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
                include_answer=False,
                include_raw_content=False
            )

            return [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0),
                    "source": "web"
                }
                for item in response.get("results", [])
            ]
        except Exception as e:
            print(f"[Tavily] Error: {e}")
            return []

    def _build_context(self, search_results: List[Dict], web_results: List[Dict]) -> str:
        """Build context from search results"""
        context_parts = []
        sources = []

        # Internal results
        for i, result in enumerate(search_results):
            metadata = result.get("metadata", {})
            namespace = result.get("namespace", "documents")

            if namespace == "machinery":
                content = self._format_machinery_content(metadata)
                title = result.get("title", f"Maschine {i + 1}")
                source_file = "Maschinendatenbank"
            else:
                content = metadata.get("content", "")
                title = metadata.get("title", f"Dokument {i + 1}")
                source_file = metadata.get("source_file", "Unknown")

            score = result.get("score", 0)

            context_parts.append(f"""
### Quelle {i + 1}: {title}
**Herkunft:** {source_file} ({namespace})
**Relevanz:** {score:.2%}

{content}
""")

            sources.append({
                "title": title,
                "source_file": source_file,
                "score": score,
                "namespace": namespace
            })

        internal_context = "\n---\n".join(context_parts) if context_parts else ""

        # Web results
        web_context = ""
        web_sources = []
        if web_results:
            web_parts = []
            for i, result in enumerate(web_results):
                web_parts.append(f"""
### Web-Quelle {i + 1}: {result.get('title', 'Untitled')}
**URL:** {result.get('url', '')}

{result.get('content', '')}
""")
                web_sources.append({
                    "title": result.get("title", ""),
                    "source_file": result.get("url", ""),
                    "score": result.get("score", 0),
                    "namespace": "web"
                })
            web_context = "\n---\n".join(web_parts)

        # Combine contexts
        if internal_context and web_context:
            full_context = f"""## INTERNE DATEN (PRIORITÄT):
{internal_context}

## ERGÄNZENDE WEB-INFORMATIONEN:
{web_context}"""
        elif internal_context:
            full_context = f"""## INTERNE DATEN:
{internal_context}"""
        elif web_context:
            full_context = f"""## WEB-INFORMATIONEN:
{web_context}"""
        else:
            full_context = "Keine relevanten Informationen gefunden."

        return full_context, sources + web_sources

    def _format_machinery_content(self, metadata: Dict) -> str:
        """Format machinery metadata as content"""
        lines = []
        if metadata.get("hersteller"):
            lines.append(f"Hersteller: {metadata['hersteller']}")
        if metadata.get("geraetegruppe"):
            lines.append(f"Typ: {metadata['geraetegruppe']}")
        if metadata.get("kategorie"):
            lines.append(f"Kategorie: {metadata['kategorie']}")
        if metadata.get("seriennummer"):
            lines.append(f"Seriennummer: {metadata['seriennummer']}")
        if metadata.get("inventarnummer"):
            lines.append(f"Inventarnummer: {metadata['inventarnummer']}")
        if metadata.get("motor_leistung_kw"):
            lines.append(f"Motorleistung: {metadata['motor_leistung_kw']} kW")
        if metadata.get("gewicht_kg"):
            lines.append(f"Gewicht: {metadata['gewicht_kg']} kg")
        if metadata.get("inhalt"):
            lines.append(f"\n{metadata['inhalt']}")
        return "\n".join(lines)

    async def _generate_natural_response(
        self,
        query: str,
        structured_data: str,
        query_type: str,
        raw_results: list = None
    ) -> str:
        """Generate natural language response from PostgreSQL structured data using LLM"""
        try:
            # Build context about the data
            result_count = len(raw_results) if raw_results else 0

            prompt = f"""Du bist der RÜKO AI-Assistent. Formuliere eine natürliche, ausführliche Antwort
basierend auf den Datenbankdaten.

FRAGE: {query}

ABFRAGETYP: {query_type}

DATENBANK-ERGEBNISSE ({result_count} Datensätze):
{structured_data}

ANWEISUNGEN:
1. Antworte in vollständigen deutschen Sätzen, nicht nur mit Zahlen
2. Bei Zählungen: "Wir haben X Geräte im Bestand" statt nur "X"
3. Bei Vergleichen (comparison):
   - Zeige beide Gruppen mit Anzahl, Durchschnittsgewicht und Durchschnittsleistung
   - Erkläre die Unterschiede: "Kettenbagger sind im Durchschnitt X kg schwerer als Mobilbagger"
   - Nenne typische Einsatzgebiete je Typ wenn sinnvoll

WICHTIG - Bei vielen Ergebnissen (mehr als 5):
- Zeige nur die TOP 3 relevantesten Ergebnisse mit Details (Hersteller, Modell, Gewicht, Leistung)
- Dann schreibe: "...und X weitere Ergebnisse"
- NICHT alle Ergebnisse einzeln auflisten!

Bei wenigen Ergebnissen (5 oder weniger):
- Zeige alle Ergebnisse mit Details

4. Strukturiere die Antwort übersichtlich und kompakt
5. Wenn keine Ergebnisse: Erkläre was gesucht wurde und dass nichts gefunden wurde

WICHTIG - Am Ende JEDER Antwort füge einen Abschnitt "💡 **Weiterführende Optionen:**" hinzu mit:
- 2-3 passende Folgefragen die der Nutzer stellen könnte (als Aufzählung)
- Filter-Vorschläge wenn relevant (z.B. "Nach Hersteller filtern", "Nur Geräte über 10t")
- "Mehr Details anzeigen" wenn es weitere Ergebnisse gibt

Beispiel:
💡 **Weiterführende Optionen:**
• "Zeige mir nur die Liebherr Bagger"
• "Welche davon haben über 100 kW Leistung?"
• "Mehr Ergebnisse anzeigen"

Antworte jetzt:"""

            response_params = {
                "model": self.model,
                "input": [
                    {"role": "user", "content": prompt}
                ],
                "max_output_tokens": 2000
            }

            # Add reasoning for supported models
            if (self.reasoning_effort and
                self.reasoning_effort.lower() != "none" and
                model_supports_reasoning(self.model)):
                response_params["reasoning"] = {"effort": self.reasoning_effort}

            response = await self.client.responses.create(**response_params)
            return response.output_text.strip() if response.output_text else structured_data

        except Exception as e:
            print(f"[RAG] Natural response generation failed: {e}")
            # Fall back to structured data if LLM fails
            return structured_data

    async def search_and_generate(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
        system_instructions: Optional[str] = None,
        previous_response_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry point: Route query and generate response.
        Hybrid RAG routes structured queries to PostgreSQL.
        """
        top_k = top_k or config.search_top_k
        start_time = time.time()

        # HYBRID RAG: Route to PostgreSQL for structured queries
        if self.use_hybrid_rag and self.hybrid_orchestrator:
            try:
                hybrid_result = await self.hybrid_orchestrator.query(query)
                print(f"[Hybrid] Type: {hybrid_result.query_type.value}, Source: {hybrid_result.source}")

                # PostgreSQL handled the query completely
                if hybrid_result.source == "postgres" and hybrid_result.answer:
                    # Generate natural language response from structured data
                    natural_response = await self._generate_natural_response(
                        query=query,
                        structured_data=hybrid_result.answer,
                        query_type=hybrid_result.query_type.value,
                        raw_results=hybrid_result.raw_results
                    )

                    # No web search for structured database queries
                    web_context = ""
                    web_sources = []

                    return {
                        "response": natural_response + web_context,
                        "sources": [{"title": "PostgreSQL", "source_file": "database", "score": hybrid_result.confidence, "namespace": "postgres"}] + web_sources,
                        "chunks_used": len(hybrid_result.raw_results or []),
                        "response_id": None,
                        "web_results_used": len(web_sources),
                        "query_type": hybrid_result.query_type.value
                    }

                # HYBRID: Apply filters to Pinecone search
                if hybrid_result.source == "hybrid" and hybrid_result.structured_filters:
                    pinecone_filters = {}
                    for key, value in hybrid_result.structured_filters.items():
                        if key == "kategorie" and value:
                            pinecone_filters["kategorie"] = {"$eq": value.lower()}
                        elif key == "hersteller" and value:
                            pinecone_filters["hersteller"] = {"$eq": value}
                        elif key == "features" and isinstance(value, list):
                            for feature in value:
                                pinecone_filters[feature.lower()] = {"$eq": True}

                    semantic_query = hybrid_result.semantic_query or query
                    filters = pinecone_filters if pinecone_filters else filters

            except Exception as e:
                print(f"[Hybrid] Error: {e}")

        # Semantic search via Pinecone
        search_results = await self.search_pinecone(query, top_k=top_k, filters=filters)

        # Supplementary web search
        web_results = []
        if self.enable_web_search:
            web_results = await self.tavily_search(query)

        # Build context
        full_context, all_sources = self._build_context(search_results, web_results)

        # Generate response
        if not system_instructions:
            system_instructions = """Du bist der RÜKO AI-Assistent mit Zugriff auf interne Datenbanken.

PRIORITÄT: Interne Daten immer zuerst, Web-Informationen nur ergänzend.

REGELN:
1. Zitiere Quellen: "Laut [Quelle]..."
2. Strukturiere Antworten übersichtlich
3. Antworte in der Sprache der Frage

Bei fehlenden internen Daten: "In den internen Datenbanken wurde nichts gefunden." """

        try:
            response_params = {
                "model": self.model,
                "input": [
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": f"""Beantworte basierend auf dem Kontext:

{full_context}

Frage: {query}"""}
                ],
                "max_output_tokens": 4000
            }

            # Only add reasoning parameter for models that support it (o1, o3 series)
            if (self.reasoning_effort and
                self.reasoning_effort.lower() != "none" and
                model_supports_reasoning(self.model)):
                response_params["reasoning"] = {"effort": self.reasoning_effort}

            if previous_response_id:
                response_params["previous_response_id"] = previous_response_id
                response_params["store"] = True

            response = await self.client.responses.create(**response_params)

            print(f"[RAG] Response generated in {time.time() - start_time:.2f}s")

            return {
                "response": response.output_text,
                "sources": all_sources,
                "chunks_used": len(search_results),
                "response_id": response.id,
                "web_results_used": len(web_results)
            }

        except Exception as e:
            print(f"[RAG] Error: {e}")
            return {
                "response": f"Fehler: {str(e)}",
                "sources": all_sources,
                "chunks_used": len(search_results),
                "response_id": None,
                "web_results_used": len(web_results)
            }

    async def search(self, query: str, top_k: int = None, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Simple search interface for backward compatibility"""
        top_k = top_k or config.search_top_k
        return await self.search_pinecone(query, top_k=top_k, filters=filters)
