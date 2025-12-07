"""
RAG Search - Main entry point for the search pipeline.
Routes queries through the multi-agent system or falls back to direct search.
"""
import time
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
    return model_lower.startswith('o1') or model_lower.startswith('o3')


# Import agent system
try:
    from .agents import AgentSystem, create_agent_system
    AGENT_SYSTEM_AVAILABLE = True
except ImportError as e:
    AGENT_SYSTEM_AVAILABLE = False
    print(f"[WARNING] Agent system not available: {e}")

# Import embeddings for fallback
from .vector_store import PineconeStore
from .embeddings import EmbeddingService


class RAGSearch:
    """
    RAG Search with multi-agent routing.

    Primary mode: Agent System (orchestrator -> sub-agents -> reviewer)
    Fallback mode: Direct Pinecone search
    """

    def __init__(self, redis_client=None):
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.vector_store = PineconeStore()
        self.embedding_service = EmbeddingService()
        self.redis_client = redis_client

        # Model settings from config
        self.model = config.response_model
        self.reasoning_effort = config.response_reasoning

        if not self.model or not self.reasoning_effort:
            raise ValueError("OPENAI_MODEL and REASONING_EFFORT must be set in .env")

        print(f"[RAG] Model: {self.model}, Reasoning: {self.reasoning_effort}")

        # Pinecone direct access (for fallback)
        self.pc = pinecone.Pinecone(api_key=config.pinecone_api_key)
        self.index = self.pc.Index(host=config.pinecone_host)
        self.machinery_namespace = config.pinecone_machinery_namespace
        self.documents_namespace = config.pinecone_namespace

        # Agent System
        self.use_agent_system = config.use_agent_system and AGENT_SYSTEM_AVAILABLE
        self.agent_system = None

        if self.use_agent_system:
            try:
                self.agent_system = create_agent_system(
                    verbose=config.agent_verbose,
                    enable_web_search=config.enable_web_search,
                    parallel_execution=config.agent_parallel_execution,
                    redis_client=redis_client
                )
                print("[RAG] Agent System: Enabled")
            except Exception as e:
                print(f"[RAG] Agent System initialization failed: {e}")
                self.use_agent_system = False
        else:
            print("[RAG] Agent System: Disabled (using direct search)")

    async def search_and_generate(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
        system_instructions: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        thread_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry point: Process query through agent system or fallback.

        Args:
            query: User's question
            top_k: Number of results (for fallback mode)
            filters: Search filters (for fallback mode)
            system_instructions: Custom system prompt (for fallback mode)
            previous_response_id: For conversation continuity
            user_id: User identifier
            user_name: User display name
            thread_key: Conversation thread key

        Returns:
            Dict with response, sources, and metadata
        """
        top_k = top_k or config.search_top_k
        start_time = time.time()

        # PRIMARY: Use Agent System
        if self.use_agent_system and self.agent_system:
            try:
                result = await self.agent_system.process(
                    user_query=query,
                    user_id=user_id,
                    user_name=user_name,
                    thread_key=thread_key
                )

                print(f"[RAG] Agent System response in {result.execution_time_ms}ms")
                print(f"[RAG] Agents used: {result.agents_used}")

                response_dict = {
                    "response": result.response,
                    "sources": result.sources,
                    "chunks_used": len(result.sources),
                    "response_id": None,  # Agent system manages its own context
                    "web_results_used": result.metadata.get("web_results_count", 0),
                    "query_type": result.query_intent or "agent",
                    "agents_used": result.agents_used,
                    "execution_time_ms": result.execution_time_ms
                }

                # Include file export if present
                if result.file_export:
                    response_dict["file_export"] = result.file_export
                    print(f"[RAG] File export included: {result.file_export.get('file_name')}")

                return response_dict

            except Exception as e:
                print(f"[RAG] Agent System error: {e}, falling back to direct search")
                # Fall through to direct search

        # FALLBACK: Direct Pinecone search
        return await self._fallback_search(
            query=query,
            top_k=top_k,
            filters=filters,
            system_instructions=system_instructions,
            previous_response_id=previous_response_id
        )

    async def _fallback_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        system_instructions: Optional[str],
        previous_response_id: Optional[str]
    ) -> Dict[str, Any]:
        """Fallback to direct Pinecone search without agent system"""
        print("[RAG] Using fallback direct search...")

        # Search Pinecone
        search_results = await self.search_pinecone(query, top_k=top_k, filters=filters)

        # Build context
        full_context, all_sources = self._build_context(search_results, [])

        # Generate response
        if not system_instructions:
            system_instructions = self._get_default_instructions()

        try:
            response_params = {
                "model": self.model,
                "input": [
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": f"""Beantworte basierend auf dem Kontext:

{full_context}

Frage: {query}"""}
                ],
                "max_output_tokens": 2000
            }

            # Add reasoning for supported models
            if (self.reasoning_effort and
                self.reasoning_effort.lower() != "none" and
                model_supports_reasoning(self.model)):
                response_params["reasoning"] = {"effort": self.reasoning_effort}

            if previous_response_id:
                response_params["previous_response_id"] = previous_response_id
                response_params["store"] = True

            response = await self.client.responses.create(**response_params)

            return {
                "response": response.output_text,
                "sources": all_sources,
                "chunks_used": len(search_results),
                "response_id": response.id,
                "web_results_used": 0,
                "query_type": "fallback"
            }

        except Exception as e:
            print(f"[RAG] Fallback error: {e}")
            return {
                "response": f"Fehler: {str(e)}",
                "sources": all_sources,
                "chunks_used": len(search_results),
                "response_id": None,
                "web_results_used": 0,
                "query_type": "error"
            }

    async def search_pinecone(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Direct Pinecone search across namespaces"""
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

    def _build_context(self, search_results: List[Dict], web_results: List[Dict]) -> tuple:
        """Build context from search results"""
        context_parts = []
        sources = []

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

        if internal_context:
            full_context = f"""## INTERNE DATEN:
{internal_context}"""
        else:
            full_context = "Keine relevanten Informationen gefunden."

        return full_context, sources

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

    def _get_default_instructions(self) -> str:
        """Get default system instructions"""
        return """Du bist der RÜKO AI-Assistent mit Zugriff auf interne Datenbanken.

PRIORITÄT: Interne Daten immer zuerst.

REGELN:
1. Zitiere Quellen: "Laut [Quelle]..."
2. Strukturiere Antworten übersichtlich
3. Antworte in der Sprache der Frage

Bei fehlenden internen Daten: "In den internen Datenbanken wurde nichts gefunden." """

    async def search(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Simple search interface for backward compatibility"""
        top_k = top_k or config.search_top_k
        return await self.search_pinecone(query, top_k=top_k, filters=filters)
