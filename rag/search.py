"""
RAG Search - Main entry point for the search pipeline.
Routes queries through the single agent or falls back to direct search.
"""
import time
import os
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
import pinecone

from .config import config


def model_supports_reasoning(model_name: str) -> bool:
    """Check if a model supports the reasoning.effort parameter.
    o-series and gpt-5 models support reasoning in the Responses API."""
    if not model_name:
        return False
    model_lower = model_name.lower()
    return (
        model_lower.startswith("o1")
        or model_lower.startswith("o3")
        or model_lower.startswith("gpt-5")
    )


# Import single agent (prefer clean agent when available)
SINGLE_AGENT_AVAILABLE = False
CLEAN_AGENT_AVAILABLE = False

try:
    from .single_agent_clean import CleanSingleAgent, create_clean_agent
    CLEAN_AGENT_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Clean agent not available: {e}")

try:
    from .single_agent import SingleAgent, create_single_agent
    SINGLE_AGENT_AVAILABLE = True
except ImportError as e:
    if not CLEAN_AGENT_AVAILABLE:
        print(f"[WARNING] Single agent not available: {e}")

# Import embeddings for fallback
from .vector_store import PineconeStore
from .embeddings import EmbeddingService


class RAGSearch:
    """
    RAG Search with single agent routing.

    Primary mode: Single Agent with tool calls
    Fallback mode: Direct Pinecone search
    """

    def __init__(self, redis_client=None):
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.vector_store = PineconeStore()
        self.embedding_service = EmbeddingService()
        self.redis_client = redis_client

        # Provider selection (SingleAgent uses provider directly).
        self.provider = (config.llm_provider or "openai").lower()

        # Fallback responses use OpenAI Responses API.
        # Keep separate from the main provider model to avoid invalid model names.
        self.model = config.openai_model
        self.reasoning_effort = config.openai_reasoning or "none"

        if not self.model:
            if config.use_agent_system and SINGLE_AGENT_AVAILABLE:
                print("[RAG] OpenAI fallback disabled (OPENAI_MODEL missing).")
            else:
                raise ValueError("OPENAI_MODEL must be set in .env for fallback mode")

        print(f"[RAG] Provider: {self.provider} | Agent model: {config.response_model} | Fallback model: {self.model}")

        # Pinecone direct access (for fallback)
        self.pc = pinecone.Pinecone(api_key=config.pinecone_api_key)
        self.index = self.pc.Index(host=config.pinecone_host)
        self.machinery_namespace = config.pinecone_machinery_namespace
        self.documents_namespace = config.pinecone_namespace

        # Single Agent - prefer clean agent for stability
        self.use_single_agent = config.use_agent_system and (CLEAN_AGENT_AVAILABLE or SINGLE_AGENT_AVAILABLE)
        self.agent = None

        if self.use_single_agent:
            try:
                # Use clean agent by default (simpler, more reliable)
                if config.use_clean_agent and CLEAN_AGENT_AVAILABLE:
                    self.agent = create_clean_agent(
                        verbose=config.agent_verbose,
                        pinecone_service=self.vector_store,
                    )
                    print("[RAG] Clean Agent: Enabled (simplified architecture)")
                elif SINGLE_AGENT_AVAILABLE:
                    self.agent = create_single_agent(
                        verbose=config.agent_verbose,
                        pinecone_service=self.vector_store,
                    )
                    print("[RAG] Single Agent: Enabled (full features)")
                else:
                    raise ImportError("No agent implementation available")
            except Exception as e:
                print(f"[RAG] Agent initialization failed: {e}")
                self.use_single_agent = False
        else:
            print("[RAG] Agent: Disabled (using direct search)")

        # LangGraph agent (new)
        self.langgraph_agent = None
        if config.use_langgraph_agent:
            try:
                from rag.langgraph_agent import get_langgraph_agent
                self.langgraph_agent = get_langgraph_agent()
                print("[RAG] LangGraph Agent: Enabled")
            except Exception as e:
                print(f"[RAG] LangGraph Agent initialization failed: {e}")

    async def _get_conversation_history(self, thread_key: str) -> List[Dict]:
        """Get full conversation history from Redis for context."""
        if not self.redis_client or not thread_key:
            return []
        try:
            import json
            redis_start = time.time()
            history_key = f"chat_history:{thread_key}"
            history_json = await self.redis_client.get(history_key)
            redis_ms = (time.time() - redis_start) * 1000
            print(f"⏱️  [redis:get_history] {redis_ms:.0f}ms")

            if history_json:
                history = json.loads(history_json)
                max_messages = max(2, int(config.conversation_max_messages))
                return history[-max_messages:]
        except Exception as e:
            print(f"[RAG] Error getting history: {e}")
        return []

    async def _store_conversation_turn(self, thread_key: str, user_msg: str, assistant_msg: str):
        """Store conversation turn in Redis for full session context."""
        if not self.redis_client or not thread_key:
            return
        try:
            import json
            redis_start = time.time()
            history_key = f"chat_history:{thread_key}"
            history = await self._get_conversation_history(thread_key)

            # Add new turn
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": assistant_msg})

            # Keep last N messages for full session context
            max_messages = max(2, int(config.conversation_max_messages))
            history = history[-max_messages:]

            # Store with configured TTL
            await self.redis_client.setex(history_key, config.conversation_ttl_hours * 3600, json.dumps(history))
            redis_ms = (time.time() - redis_start) * 1000
            print(f"⏱️  [redis:store_history] {redis_ms:.0f}ms")
        except Exception as e:
            print(f"[RAG] Error storing history: {e}")

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
        Main entry point: Process query through single agent or fallback.

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

        # Priority 1: LangGraph agent
        if self.langgraph_agent and config.use_langgraph_agent:
            try:
                # Get conversation history for context
                conversation_history = await self._get_conversation_history(thread_key)

                if conversation_history:
                    print(f"[RAG] LangGraph using {len(conversation_history)} messages from history")

                result = await self.langgraph_agent.process(
                    user_query=query,
                    thread_key=thread_key,
                    conversation_history=conversation_history
                )

                print(f"[RAG] LangGraph response in {result.execution_time_ms}ms")
                print(f"[RAG] Tools used: {result.tools_used}")

                # Store this turn in history
                await self._store_conversation_turn(thread_key, query, result.response)

                return {
                    "response": result.response,
                    "sources": result.sources or [],
                    "chunks_used": len(result.sources) if result.sources else getattr(result, 'sql_results_count', 0),
                    "response_id": None,
                    "web_results_used": 0,
                    "query_type": "langgraph_agent",
                    "agents_used": result.tools_used,
                    "execution_time_ms": result.execution_time_ms,
                    "agent": "langgraph"
                }
            except Exception as e:
                print(f"[RAG] LangGraph agent error: {e}, falling back to SingleAgent")
                # Fall through to single agent

        # Priority 2: Use Single Agent (legacy)
        if self.use_single_agent and self.agent:
            try:
                # Get conversation history for context
                conversation_history = await self._get_conversation_history(thread_key)
                
                if conversation_history:
                    print(f"[RAG] Using {len(conversation_history)} messages from history")
                
                result = await self.agent.process(
                    user_query=query,
                    conversation_history=conversation_history,
                    system_instructions=system_instructions,
                    thread_key=thread_key,
                )

                print(f"[RAG] Single Agent response in {result.execution_time_ms}ms")
                print(f"[RAG] Tools used: {result.tools_used}")

                # Store this turn in history
                await self._store_conversation_turn(thread_key, query, result.response)

                return {
                    "response": result.response,
                    "sources": result.sources,
                    "chunks_used": len(result.sources) if result.sources else result.sql_results_count,
                    "response_id": None,
                    "web_results_used": 0,
                    "query_type": "single_agent",
                    "agents_used": result.tools_used,
                    "execution_time_ms": result.execution_time_ms,
                    "logs": getattr(result, "logs", [])
                }

            except Exception as e:
                print(f"[RAG] Single Agent error: {e}, falling back to direct search")
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

        if not self.model:
            return {
                "response": "Fallback ist deaktiviert, da OPENAI_MODEL nicht konfiguriert ist.",
                "sources": [],
                "chunks_used": 0,
                "response_id": None,
                "web_results_used": 0,
                "query_type": "error"
            }

        # Search Pinecone
        search_results = await self.search_pinecone(query, top_k=top_k, filters=filters)

        if not search_results:
            return {
                "response": (
                    "Ich kann nur mit internen Daten antworten. "
                    "In den internen Datenbanken wurde keine Information gefunden. "
                    "Gibt es einen Hersteller, Maschinentyp oder weitere Kriterien?"
                ),
                "sources": [],
                "chunks_used": 0,
                "response_id": None,
                "web_results_used": 0,
                "query_type": "fallback"
            }

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
                "max_output_tokens": config.fallback_max_output_tokens
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
        """Direct Pinecone search across namespaces in parallel."""
        import asyncio

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

        # Helper function to search a single namespace in executor
        def _search_namespace_sync(namespace: str, content_key: str, title_key: str, source_default: str):
            """Search a single namespace (sync, for executor)."""
            try:
                results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    namespace=namespace,
                    include_metadata=True,
                    filter=pinecone_filter
                )
                formatted = []
                for match in results.matches:
                    metadata = match.metadata or {}
                    formatted.append({
                        "id": match.id,
                        "score": match.score,
                        "metadata": metadata,
                        "namespace": "documents" if namespace == self.documents_namespace else "machinery",
                        "content": metadata.get(content_key, ""),
                        "title": metadata.get(title_key, ""),
                        "source_file": metadata.get("source_file", source_default)
                    })
                return formatted
            except Exception as e:
                print(f"[Search] {namespace} error: {e}")
                return []

        # Run both searches in parallel using executor (Pinecone client is sync)
        loop = asyncio.get_event_loop()

        doc_task = loop.run_in_executor(
            None,
            _search_namespace_sync,
            self.documents_namespace, "content", "title", "Unknown"
        )
        machinery_task = loop.run_in_executor(
            None,
            _search_namespace_sync,
            self.machinery_namespace, "inhalt", "titel", "machinery-database"
        )

        # Wait for both searches to complete
        doc_results, machinery_results = await asyncio.gather(
            doc_task, machinery_task,
            return_exceptions=True
        )

        # Handle any exceptions from gather
        if isinstance(doc_results, Exception):
            print(f"[Search] Documents parallel error: {doc_results}")
            doc_results = []
        if isinstance(machinery_results, Exception):
            print(f"[Search] Machinery parallel error: {machinery_results}")
            machinery_results = []

        # Combine and sort by score
        all_results = doc_results + machinery_results
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
### Dokument {i + 1}: {title}
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
        return """Du bist der RUEKO AI-Assistent mit Zugriff auf interne Daten (Pinecone).

REGELN:
1. Antworte ausschliesslich auf Basis des internen Kontexts.
2. Nenne Quellen nur, wenn der Nutzer explizit danach fragt.
3. Keine externen Informationen oder Annahmen.
4. Wenn keine internen Daten vorhanden sind: sage das klar und stelle eine Rueckfrage.
5. Antworte in der Sprache der Frage."""

    async def search(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Simple search interface for backward compatibility"""
        top_k = top_k or config.search_top_k
        return await self.search_pinecone(query, top_k=top_k, filters=filters)
