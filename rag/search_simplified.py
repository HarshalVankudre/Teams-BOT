"""
RAG Search - Simplified entry point.
Routes queries through LangGraph agent or falls back to direct Pinecone search.
"""
import time
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
import pinecone

from .config import config
from .vector_store import PineconeStore
from .embeddings import EmbeddingService


class RAGSearch:
    """
    RAG Search with LangGraph agent.

    Primary: LangGraph ReAct agent with tools
    Fallback: Direct Pinecone search
    """

    def __init__(self, redis_client=None):
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.vector_store = PineconeStore()
        self.embedding_service = EmbeddingService()
        self.redis_client = redis_client
        
        # Fallback model
        self.model = config.openai_model

        # Pinecone direct access (for fallback)
        self.pc = pinecone.Pinecone(api_key=config.pinecone_api_key)
        self.index = self.pc.Index(host=config.pinecone_host)
        self.machinery_namespace = config.pinecone_machinery_namespace
        self.documents_namespace = config.pinecone_namespace

        # LangGraph agent
        self.langgraph_agent = None
        if config.use_langgraph_agent:
            try:
                from rag.langgraph_agent import get_langgraph_agent
                self.langgraph_agent = get_langgraph_agent()
                print("[RAG] LangGraph Agent: Enabled")
            except Exception as e:
                print(f"[RAG] LangGraph Agent init failed: {e}")

    async def _get_conversation_history(self, thread_key: str) -> List[Dict]:
        """Get conversation history from Redis."""
        if not self.redis_client or not thread_key:
            return []
        try:
            import json
            history_key = f"chat_history:{thread_key}"
            history_json = await self.redis_client.get(history_key)
            if history_json:
                history = json.loads(history_json)
                max_messages = max(2, int(config.conversation_max_messages))
                return history[-max_messages:]
        except Exception as e:
            print(f"[RAG] Error getting history: {e}")
        return []

    async def _store_conversation_turn(self, thread_key: str, user_msg: str, assistant_msg: str):
        """Store conversation turn in Redis."""
        if not self.redis_client or not thread_key:
            return
        try:
            import json
            history_key = f"chat_history:{thread_key}"
            history = await self._get_conversation_history(thread_key)
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": assistant_msg})
            max_messages = max(2, int(config.conversation_max_messages))
            history = history[-max_messages:]
            await self.redis_client.setex(
                history_key, 
                config.conversation_ttl_hours * 3600, 
                json.dumps(history)
            )
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
        """Main entry point: Process query through LangGraph or fallback."""
        top_k = top_k or config.search_top_k
        start_time = time.time()

        # Try LangGraph agent first
        if self.langgraph_agent:
            try:
                conversation_history = await self._get_conversation_history(thread_key)
                
                result = await self.langgraph_agent.process(
                    user_query=query,
                    thread_key=thread_key or "default",
                    conversation_history=conversation_history
                )
                
                response_text = result.get("response", "")
                
                # Store conversation turn
                if thread_key and response_text:
                    await self._store_conversation_turn(thread_key, query, response_text)
                
                return {
                    "response": response_text,
                    "sources": result.get("sources", []),
                    "chunks_used": 0,
                    "response_id": None,
                    "web_results_used": 0,
                    "query_type": "langgraph",
                    "tools_used": result.get("tools_used", []),
                    "processing_time": time.time() - start_time
                }
            except Exception as e:
                print(f"[RAG] LangGraph error: {e}, falling back to Pinecone")

        # Fallback: Direct Pinecone search + OpenAI
        return await self._fallback_search(
            query, top_k, system_instructions, previous_response_id
        )

    async def _fallback_search(
        self,
        query: str,
        top_k: int,
        system_instructions: Optional[str],
        previous_response_id: Optional[str]
    ) -> Dict[str, Any]:
        """Fallback to direct Pinecone search."""
        search_results = await self.search_pinecone(query, top_k=top_k)
        context, all_sources = self._build_context(search_results, [])
        
        instructions = system_instructions or self._get_default_instructions()
        
        try:
            response_params = {
                "model": self.model,
                "instructions": instructions,
                "input": f"Context:\n{context}\n\nFrage: {query}",
            }
            
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
                "query_type": "error"
            }

    async def search_pinecone(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Direct Pinecone search across namespaces."""
        import asyncio
        
        query_embedding = await self.embedding_service.embed_query(query)
        
        def _search_namespace(namespace: str, content_key: str, title_key: str):
            try:
                results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    namespace=namespace,
                    include_metadata=True
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
                        "source_file": metadata.get("source_file", "Unknown")
                    })
                return formatted
            except Exception as e:
                print(f"[Search] {namespace} error: {e}")
                return []

        loop = asyncio.get_event_loop()
        doc_task = loop.run_in_executor(
            None, _search_namespace, 
            self.documents_namespace, "content", "title"
        )
        machinery_task = loop.run_in_executor(
            None, _search_namespace,
            self.machinery_namespace, "inhalt", "titel"
        )
        
        doc_results, machinery_results = await asyncio.gather(
            doc_task, machinery_task, return_exceptions=True
        )
        
        if isinstance(doc_results, Exception):
            doc_results = []
        if isinstance(machinery_results, Exception):
            machinery_results = []
        
        all_results = doc_results + machinery_results
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results

    def _build_context(self, search_results: List[Dict], web_results: List[Dict]) -> tuple:
        """Build context from search results."""
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
            
            context_parts.append(f"### {title}\n{content}")
            sources.append({
                "title": title,
                "source_file": source_file,
                "score": result.get("score", 0),
                "namespace": namespace
            })
        
        return "\n---\n".join(context_parts), sources

    def _format_machinery_content(self, metadata: Dict) -> str:
        """Format machinery metadata."""
        lines = []
        for key in ["hersteller", "geraetegruppe", "seriennummer", "inventarnummer"]:
            if metadata.get(key):
                lines.append(f"{key}: {metadata[key]}")
        if metadata.get("inhalt"):
            lines.append(metadata["inhalt"])
        return "\n".join(lines)

    def _get_default_instructions(self) -> str:
        """Default system instructions for fallback."""
        return """Du bist der RUEKO AI-Assistent.
Antworte auf Basis des Kontexts. Wenn keine Daten vorhanden: sage das klar.
Antworte auf Deutsch."""

    async def search(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Simple search interface."""
        top_k = top_k or config.search_top_k
        return await self.search_pinecone(query, top_k=top_k, filters=filters)
