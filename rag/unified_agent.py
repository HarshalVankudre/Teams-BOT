"""
Unified single-agent responder that replaces the slower multi-agent orchestration.
Uses one LLM call with tool-calling for SQL, semantic (Pinecone) search, and optional web search,
then responds in a second turn. Designed to cut latency while preserving answer quality.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, date
from decimal import Decimal

import pinecone
from openai import AsyncOpenAI

from .config import config
from .embeddings import EmbeddingService
from .postgres import PostgresService
from .schema import ORCHESTRATOR_SCHEMA, SQL_SPECIAL_RULES

try:
    from tavily import TavilyClient
except ImportError:  # pragma: no cover - optional dependency
    TavilyClient = None


class UnifiedAgent:
    """Single-agent pipeline with minimal tool calls."""

    def __init__(self, redis_client=None):
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.model = config.response_model
        self.reasoning_effort = config.response_reasoning
        self.embedding_service = EmbeddingService()
        self.pg = PostgresService()
        self.redis_client = redis_client
        self._local_history: Dict[str, List[Dict[str, str]]] = {}
        self._thread_focus: Dict[str, Dict[str, Any]] = {}
        self._current_thread_key: Optional[str] = None

        # Pinecone setup
        self.pinecone_available = False
        try:
            self.pc = pinecone.Pinecone(api_key=config.pinecone_api_key)
            self.index = self.pc.Index(host=config.pinecone_host)
            self.documents_namespace = config.pinecone_namespace
            self.machinery_namespace = config.pinecone_machinery_namespace
            self.pinecone_available = True
        except Exception as e:  # pragma: no cover - runtime safety
            print(f"[UnifiedAgent] Pinecone unavailable: {e}")

        # Tavily setup
        self.tavily = TavilyClient(api_key=config.tavily_api_key) if (
            config.enable_web_search and TavilyClient and config.tavily_api_key
        ) else None

        self.default_sql_limit = 200
        self._dangerous_sql_patterns = (
            r"\bdrop\b",
            r"\bdelete\b",
            r"\btruncate\b",
            r"\binsert\b",
            r"\bupdate\b",
            r"\balter\b",
            r"\bcreate\b",
            r";.*;",
        )

        self.tools = self._build_tools()

    def _build_tools(self) -> List[Dict[str, Any]]:
        """Tool definitions for chat.completions."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_sql",
                    "description": "Fuehre eine SQL-SELECT-Abfrage gegen die Tabelle geraete aus. Nur SELECT erlaubt.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {"type": "string", "description": "Vollstaendige SQL-SELECT-Abfrage"},
                            "limit": {"type": "integer", "description": "Optionale Limit-Grenze"},
                        },
                        "required": ["sql"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "semantic_search",
                    "description": "Pinecone-Suche in Dokumenten und Maschinendaten.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Suchtext"},
                            "top_k": {"type": "integer", "description": "Anzahl Treffer pro Namespace (Default 5)"},
                            "namespace": {
                                "type": "string",
                                "enum": ["documents", "machinery", "both"],
                                "description": "Namespace-Auswahl",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

        if self.tavily:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Internetrecherche via Tavily, nur wenn interne Daten nicht reichen.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "max_results": {"type": "integer", "description": "Max Ergebnisse (Default 3)"},
                            },
                            "required": ["query"],
                        },
                    },
                }
            )

        return tools

    def _build_system_prompt(self, history: str, focus: Optional[Dict[str, Any]]) -> str:
        """Construct concise system prompt."""
        focus_text = ""
        if focus:
            parts = []
            if focus.get("machine_id"):
                parts.append(f"ID {focus['machine_id']}")
            if focus.get("inventarnummer"):
                parts.append(f"Inventarnummer {focus['inventarnummer']}")
            if focus.get("seriennummer"):
                parts.append(f"Seriennummer {focus['seriennummer']}")
            if focus.get("bohle"):
                parts.append(f"Bohle {focus['bohle']}")
            if focus.get("einbaubreite_max"):
                parts.append(f"max Einbaubreite {focus['einbaubreite_max']}")
            if parts:
                focus_text = "Aktueller Kontext: " + ", ".join(parts) + "."

        return f"""Du bist der Single-Agent fuer das RAoKO Teams-Bot Backend.
Arbeite schnell: maximal eine Tool-Runde, dann antworte.
Bevorzuge SQL fuer strukturierte/zaehlende Fragen, Pinecone fuer semantische Dokumente,
Web-Suche nur falls interne Daten fehlen.

Schema (geraete):
{ORCHESTRATOR_SCHEMA}

SQL-Hinweise:
{SQL_SPECIAL_RULES}

Kontext:
{history}
{focus_text}

Regeln:
- Folgefragen ohne neue IDs/Modelle MÜSSEN den zuletzt genannten Kontext nutzen (z.B. zuletzt genannte Maschinen-ID, Bohle, Einbaubreite). Kein Nachfragen, wenn der Kontext klar aus der letzten Antwort hervorgeht.
- Bei Fragen wie "maximale Einbaubreite?" nach einer Maschine/Bohle: Nutze die zuletzt genannte Maschine/Bohle aus der Konversation.
- Bei "Liste mit allen passenden Verbreiterungen": Filtere auf die zuletzt genannte Bohle/Maschine; gib kompakte Liste.
- Gib kurze, strukturierte Antworten in der Sprache der Frage.
- Nenne Quellen (SQL/Pinecone/Web) wenn moeglich.
- Fuehre keine destruktiven SQL-Befehle aus (nur SELECT)."""

    async def run(
        self,
        query: str,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        thread_key: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Entry point to answer a query with a single tool-enabled LLM call."""
        start = time.time()
        history = await self._load_history(thread_key)
        full_history = history + (conversation_history or [])
        history_text = self._render_history(full_history)
        focus = self._thread_focus.get(thread_key or "", {})

        # track current thread for tool handlers
        self._current_thread_key = thread_key

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt(history_text, focus)},
            {"role": "user", "content": query},
        ]

        sources: List[Dict[str, Any]] = []
        tools_used: List[str] = []
        web_results_used = 0

        # Single round of tool calls followed by final answer
        response_message = await self._chat(messages, allow_tools=True)

        if response_message.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": response_message.content or "",
                    "tool_calls": response_message.tool_calls,
                }
            )

            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                tools_used.append(tool_name)

                tool_result, tool_sources = await self._handle_tool(tool_name, args)
                sources.extend(tool_sources)
                if tool_name == "web_search":
                    web_results_used += len(tool_sources)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result, ensure_ascii=False)[:4000],
                    }
                )

            # Final answer using tool results
            response_message = await self._chat(messages, allow_tools=False)

        final_text = response_message.content or "Keine Antwort verfuegbar."

        if self.redis_client and thread_key:
            await self._store_history(thread_key, query, final_text)

        execution_ms = int((time.time() - start) * 1000)

        query_type = self._infer_query_type(tools_used)
        agents_used = ["single_agent"] + tools_used

        return {
            "response": final_text,
            "sources": sources,
            "chunks_used": len([s for s in sources if s.get("namespace") != "sql"]),
            "response_id": None,
            "web_results_used": web_results_used,
            "query_type": query_type,
            "agents_used": agents_used,
            "execution_time_ms": execution_ms,
        }

    async def _chat(self, messages: List[Dict[str, Any]], allow_tools: bool) -> Any:
        """Call OpenAI chat with optional tools."""
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        if allow_tools:
            request["tools"] = self.tools
            request["tool_choice"] = "auto"

        if self.reasoning_effort and self.reasoning_effort.lower() != "none" and self._supports_reasoning():
            request["reasoning"] = {"effort": self.reasoning_effort}

        resp = await self.client.chat.completions.create(**request)
        return resp.choices[0].message

    def _supports_reasoning(self) -> bool:
        """Check if model supports the 'reasoning' parameter on chat completions."""
        if not self.model:
            return False
        model_lower = self.model.lower()
        return model_lower.startswith("o1") or model_lower.startswith("o3")

    async def _handle_tool(self, name: str, args: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Dispatch tool calls."""
        if name == "execute_sql":
            return await self._tool_execute_sql(args)
        if name == "semantic_search":
            return await self._tool_semantic_search(args)
        if name == "web_search":
            return await self._tool_web_search(args)

        return {"error": f"Unknown tool {name}"}, []

    async def _tool_execute_sql(self, args: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Validate and run SQL."""
        sql = args.get("sql", "")
        limit = args.get("limit")

        if not sql:
            return {"error": "Kein SQL erhalten."}, []
        if not self.pg.available:
            return {"error": "PostgreSQL nicht verfuegbar."}, []

        lowered = sql.lower()
        if not lowered.strip().startswith("select"):
            return {"error": "Nur SELECT-Abfragen sind erlaubt."}, []

        for pattern in self._dangerous_sql_patterns:
            if re.search(pattern, lowered):
                return {"error": f"Unsicheres SQL-Muster erkannt: {pattern}"}, []

        safe_sql = self._enforce_limit(sql, limit)
        rows = self.pg.execute_query(safe_sql)
        row_count = len(rows)
        truncated_rows = rows[:50]  # prevent huge payloads
        serialized_rows = [self._make_jsonable(row) for row in truncated_rows]

        # Fallback: if no rows, try numeric lookup across id/seriennummer/inventarnummer
        if row_count == 0:
            num_match = re.search(r"\b(\d{5,})\b", safe_sql)
            if num_match:
                num = num_match.group(1)
                alt_sql = (
                    "SELECT id, bezeichnung, hersteller, inventarnummer, seriennummer, "
                    "prop_bohle, prop_einbaubreite_max "
                    "FROM geraete "
                    f"WHERE seriennummer = '{num}' OR inventarnummer = '{num}' OR id::text = '{num}' "
                    f"LIMIT {self.default_sql_limit}"
                )
                alt_rows = self.pg.execute_query(alt_sql)
                if alt_rows:
                    rows = alt_rows
                    row_count = len(rows)
                    serialized_rows = [self._make_jsonable(r) for r in rows[:50]]
                    safe_sql = alt_sql

        payload = {
            "type": "sql_result",
            "sql": safe_sql,
            "row_count": row_count,
            "rows": serialized_rows,
        }

        # Update thread focus from first row if available
        if serialized_rows:
            first = serialized_rows[0]
            focus_update = {
                "machine_id": first.get("id"),
                "inventarnummer": first.get("inventarnummer"),
                "seriennummer": first.get("seriennummer"),
                "bohle": first.get("prop_bohle") or first.get("bohle"),
                "einbaubreite_max": first.get("prop_einbaubreite_max"),
            }
            thread_key = self._current_thread_key
            if thread_key:
                existing = self._thread_focus.get(thread_key, {})
                existing.update({k: v for k, v in focus_update.items() if v})
                self._thread_focus[thread_key] = existing

        sources = [
            {
                "title": "PostgreSQL Abfrage",
                "source_file": "postgres",
                "score": 1.0,
                "namespace": "sql",
                "sql": safe_sql[:200],
            }
        ]

        return payload, sources

    def _enforce_limit(self, sql: str, limit: Optional[int]) -> str:
        """Append a LIMIT to large selects if none present."""
        if re.search(r"\blimit\s+\d+", sql, flags=re.IGNORECASE):
            return sql
        if re.search(r"\bcount\s*\(", sql, flags=re.IGNORECASE):
            return sql

        applied_limit = limit or self.default_sql_limit
        stripped = sql.rstrip(" ;")
        return f"{stripped} LIMIT {applied_limit}"

    async def _tool_semantic_search(self, args: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Run Pinecone search across namespaces."""
        query = args.get("query", "")
        top_k = int(args.get("top_k") or config.search_top_k)
        namespace = args.get("namespace", "both")

        if not query:
            return {"error": "Keine Suchanfrage erhalten."}, []
        if not self.pinecone_available:
            return {"error": "Pinecone nicht verfuegbar."}, []

        embedding = await self.embedding_service.embed_query(query)
        namespaces = self._resolve_namespaces(namespace)

        all_results: List[Dict[str, Any]] = []
        sources: List[Dict[str, Any]] = []

        for ns in namespaces:
            try:
                result = self.index.query(
                    vector=embedding,
                    top_k=top_k,
                    namespace=ns,
                    include_metadata=True,
                )
                for match in result.matches:
                    meta = match.metadata or {}
                    entry = {
                        "id": match.id,
                        "score": match.score,
                        "namespace": ns,
                        "title": meta.get("title") or meta.get("geraetegruppe") or "Dokument",
                        "content": meta.get("content") or meta.get("inhalt") or "",
                        "source_file": meta.get("source_file", "pinecone"),
                    }
                    all_results.append(entry)
                    sources.append(
                        {
                            "title": entry["title"],
                            "source_file": entry["source_file"],
                            "score": match.score,
                            "namespace": ns,
                        }
                    )
            except Exception as e:  # pragma: no cover - runtime safety
                print(f"[UnifiedAgent] Pinecone query failed: {e}")

        payload = {
            "type": "semantic_results",
            "query": query,
            "top_k": top_k,
            "results": all_results[:top_k * len(namespaces)],
        }
        return payload, sources

    def _resolve_namespaces(self, namespace: str) -> List[str]:
        """Map namespace shorthand to actual namespaces."""
        if namespace == "documents":
            return [self.documents_namespace]
        if namespace == "machinery":
            return [self.machinery_namespace]
        return [self.documents_namespace, self.machinery_namespace]

    def _make_jsonable(self, obj: Any) -> Any:
        """Convert common non-JSON types to serializable representations."""
        if isinstance(obj, dict):
            return {k: self._make_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_jsonable(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._make_jsonable(v) for v in obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        return obj

    async def _tool_web_search(self, args: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Optional Tavily web search."""
        query = args.get("query", "")
        max_results = int(args.get("max_results") or config.web_search_max_results)

        if not query:
            return {"error": "Keine Web-Suchanfrage erhalten."}, []
        if not self.tavily:
            return {"error": "Websuche ist deaktiviert."}, []

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self.tavily.search(query=query, max_results=max_results),
            )
        except Exception as e:  # pragma: no cover - network/runtime safety
            return {"error": f"Websuche fehlgeschlagen: {e}"}, []

        hits = result.get("results", []) if isinstance(result, dict) else []
        payload_hits = [
            {"title": r.get("title"), "url": r.get("url"), "content": r.get("content")}
            for r in hits
        ]

        sources = [
            {
                "title": r.get("title", "Web Result"),
                "source_file": r.get("url", ""),
                "score": r.get("score", 0),
                "namespace": "web",
            }
            for r in hits
        ]

        return {"type": "web_results", "results": payload_hits}, sources

    def _render_history(self, history: List[Dict[str, str]]) -> str:
        """Compact history for the prompt."""
        if not history:
            return "Keine vorherige Konversation."
        lines = []
        for turn in history[-10:]:
            role = turn.get("role", "user")
            content = (turn.get("content") or "")[:500]
            lines.append(f"- {role}: {content}")
        return "\n".join(lines)

    async def _load_history(self, thread_key: Optional[str]) -> List[Dict[str, Any]]:
        """Load conversation from Redis, fallback to local memory."""
        if not thread_key:
            return []

        if not self.redis_client:
            return self._local_history.get(thread_key, [])

        try:
            history_key = f"history:{thread_key}"
            raw = await self.redis_client.get(history_key)
            if raw:
                return json.loads(raw)
        except Exception as e:  # pragma: no cover - redis safety
            print(f"[UnifiedAgent] Redis load error: {e}")
        return self._local_history.get(thread_key, [])

    async def _store_history(self, thread_key: str, user_message: str, ai_response: str) -> None:
        """Persist conversation to Redis, always keep local cache."""
        if not thread_key:
            return

        # Local cache for context continuity when Redis is absent
        history = self._local_history.get(thread_key, [])
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": ai_response[:4000]})
        if len(history) > 20:
            history = history[-20:]
        self._local_history[thread_key] = history

        if not self.redis_client:
            return

        try:
            history_key = f"history:{thread_key}"
            raw = await self.redis_client.get(history_key)
            persisted = json.loads(raw) if raw else []
            persisted.extend(history[-2:])  # append latest turn

            if len(persisted) > 20:
                persisted = persisted[-20:]

            ttl_hours = config.conversation_ttl_hours if hasattr(config, "conversation_ttl_hours") else 24
            await self.redis_client.setex(history_key, ttl_hours * 3600, json.dumps(persisted, ensure_ascii=False))
        except Exception as e:  # pragma: no cover - redis safety
            print(f"[UnifiedAgent] Redis store error: {e}")

    def _infer_query_type(self, tools_used: List[str]) -> str:
        """Map tools used to a query type."""
        has_sql = "execute_sql" in tools_used
        has_sem = "semantic_search" in tools_used
        has_web = "web_search" in tools_used
        if has_sql and has_sem:
            return "hybrid_sql_semantic"
        if has_sql:
            return "structured_query"
        if has_sem:
            return "semantic_query"
        if has_web:
            return "external_search"
        return "direct_response"
