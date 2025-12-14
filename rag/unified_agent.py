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
from difflib import SequenceMatcher

import pinecone
from openai import AsyncOpenAI

from .config import config
from .embeddings import EmbeddingService
from .postgres import PostgresService
from .schema import (
    ORCHESTRATOR_SCHEMA,
    SQL_SPECIAL_RULES,
    CODE_NAME_FORMAT_COLUMNS,
    PROPERTY_COLUMNS,
    GERAETE_VIEW_BASE_COLUMNS,
)

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
        self._current_rewrite: Dict[str, Any] = {}
        self._current_search_queries: List[str] = []
        self._current_is_followup: bool = False

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

        self.default_sql_limit = config.unified_agent_default_sql_limit
        self.max_tool_rounds = max(1, int(config.unified_agent_max_tool_rounds or 1))
        self.retry_on_empty = bool(config.unified_agent_retry_on_empty)
        self.force_internal_first = bool(getattr(config, "unified_agent_force_internal_first", True))
        self.max_answer_words = int(getattr(config, "unified_agent_max_answer_words", 120) or 120)
        self.enable_query_rewrite = bool(getattr(config, "unified_agent_enable_query_rewrite", True))
        self.query_rewrite_model = getattr(config, "query_rewrite_model", "") or self.model
        self.query_rewrite_reasoning = getattr(config, "query_rewrite_reasoning", "") or "none"
        self.multi_query_retrieval = bool(getattr(config, "unified_agent_multi_query_retrieval", True))
        self.multi_query_max = max(1, int(getattr(config, "unified_agent_multi_query_max", 3) or 3))
        self.rewrite_history_turns = max(0, int(getattr(config, "unified_agent_rewrite_history_turns", 8) or 8))
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
                            "filters": {
                                "type": "object",
                                "description": "Optionaler Pinecone-Filter (Metadata), z.B. {\"source_file\": {\"$eq\": \"...\"}}",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "diagnose_data_sources",
                    "description": "Diagnostik: prueft Verfuegbarkeit von PostgreSQL/Pinecone/Websuche und liefert Basis-Stats.",
                    "parameters": {"type": "object", "properties": {}},
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

    def _build_system_prompt(
        self,
        history: str,
        focus: Optional[Dict[str, Any]],
        system_instructions: Optional[str],
    ) -> str:
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
            if focus.get("last_title") or focus.get("last_source_file"):
                title = str(focus.get("last_title") or "Quelle").strip()
                source = str(focus.get("last_source_file") or "").strip()
                if source:
                    parts.append(f"Letzte Quelle {title} ({source})")
                else:
                    parts.append(f"Letzte Quelle {title}")
            if parts:
                focus_text = "Aktueller Kontext: " + ", ".join(parts) + "."

        extra_sections: List[str] = []
        if getattr(config, "unified_agent_additional_instructions", ""):
            extra_sections.append(str(config.unified_agent_additional_instructions).strip())
        if system_instructions:
            extra_sections.append(str(system_instructions).strip())
        extra_text = "\n\n".join([s for s in extra_sections if s]).strip()
        extra_block = f"\n\nZusatzanweisungen:\n{extra_text}\n" if extra_text else ""

        return f"""Du bist der Single-Agent fuer das RAoKO Teams-Bot Backend.
 Arbeite schnell: maximal {self.max_tool_rounds} Tool-Runden, dann antworte.
 Bevorzuge SQL fuer strukturierte/zaehlende Fragen, Pinecone fuer semantische Dokumente,
 Web-Suche nur falls interne Daten fehlen.

Schema (geraete):
{ORCHESTRATOR_SCHEMA}

SQL-Hinweise:
{SQL_SPECIAL_RULES}

 Kontext:
 {history}
 {focus_text}
{extra_block}

 Regeln:
- Folgefragen ohne neue IDs/Modelle sollen den zuletzt genannten Kontext nutzen, WENN die Frage offensichtlich darauf referenziert (z.B. zuletzt genannte Maschinen-ID, Bohle, Einbaubreite). Bei Themenwechsel Kontext ignorieren und neu passend waehlen.
 - Bei Fragen wie "maximale Einbaubreite?" nach einer Maschine/Bohle: Nutze die zuletzt genannte Maschine/Bohle aus der Konversation.
  - Bei "Liste mit allen passenden Verbreiterungen": Filtere auf die zuletzt genannte Bohle/Maschine; gib kompakte Liste.
  - KEINE Spekulation: Wenn Verbreiterungs-Module/Spalten nicht in den Daten stehen, sage explizit, dass keine Verbreiterungen hinterlegt sind und nenne nur bekannte Grund-/Max-Einbaubreiten.
  - Wenn Tool-Ergebnisse leer sind oder Fehler liefern: versuche eine alternative Strategie (Filter lockern, ILIKE statt '=', *_search Spalten nutzen, andere Namespace, ggf. diagnose_data_sources) bevor du final aufgibst.
  - Antworte kurz und praezise (max. {self.max_answer_words} Woerter; 3-6 Stichpunkte oder 2-4 Saetze).
  - Nenne Quellen (SQL/Pinecone/Web) wenn moeglich.
  - Fuehre keine destruktiven SQL-Befehle aus (nur SELECT)."""

    async def run(
        self,
        query: str,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        thread_key: Optional[str] = None,
        system_instructions: Optional[str] = None,
        verbose: bool = False,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Entry point to answer a query with bounded tool-enabled LLM calls."""
        start = time.time()
        query, verbose_from_flag = self._strip_verbose_flag(query)
        verbose = bool(verbose or verbose_from_flag)
        trace: List[str] = []
        if verbose:
            trace.append("Verbose: Ausfuehrungsprotokoll (keine internen Gedanken/Chain-of-Thought).")
            trace.append(f"Model={self.model}, reasoning_effort={self.reasoning_effort}")
            trace.append(f"max_tool_rounds={self.max_tool_rounds}, force_internal_first={self.force_internal_first}")

        history = await self._load_history(thread_key)
        full_history = history + (conversation_history or [])
        history_text = self._render_history(full_history)
        focus = self._thread_focus.get(thread_key or "", {})
        is_recommendation = self._is_recommendation_intent(query)
        forced_recommendation_tool_attempt = False

        # track current thread for tool handlers
        self._current_thread_key = thread_key

        # Short-circuit known cases to avoid hallucinations
        short = self._short_circuit(query, focus)
        if short:
            if thread_key:
                await self._store_history(thread_key, query, short["response"])
            return short

        rewrite = await self._rewrite_query_for_retrieval(query, full_history, focus)
        tool_query = str(rewrite.get("standalone_question") or query).strip() or query
        search_queries = rewrite.get("search_queries") or [tool_query]
        if not isinstance(search_queries, list):
            search_queries = [tool_query]
        search_queries = [str(q).strip() for q in search_queries if str(q).strip()] or [tool_query]

        self._current_rewrite = rewrite
        self._current_search_queries = search_queries
        self._current_is_followup = bool(rewrite.get("is_followup"))

        rewrite_context = ""
        if self._current_is_followup or len(search_queries) > 1:
            rewrite_context = (
                "Retrieval-Hinweis (automatisch aus Verlauf/Fokus):\n"
                f"- Standalone: {tool_query}\n"
                f"- Search Queries: {', '.join(search_queries)}"
            )
            if verbose:
                trace.append("rewrite: contextualized retrieval query generated")

        prefetch_context = ""
        prefetch_sources: List[Dict[str, Any]] = []
        if is_recommendation:
            prefetch_context, prefetch_sources = await self._prefetch_recommendation_context(tool_query)
            if verbose and prefetch_context:
                trace.append(f"prefetch: semantic_search namespace=machinery (candidates={len(prefetch_sources)})")

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt(history_text, focus, system_instructions)},
            {"role": "system", "content": self._recommendation_rules() if is_recommendation else ""},
            {"role": "system", "content": prefetch_context},
            {"role": "system", "content": rewrite_context},
            {"role": "user", "content": query},
        ]
        messages = [m for m in messages if (m.get("content") or "").strip()]

        sources: List[Dict[str, Any]] = []
        tools_used: List[str] = []
        web_results_used = 0
        if prefetch_sources:
            sources.extend(prefetch_sources)
            tools_used.append("prefetch_semantic_machinery")

        final_text = "Keine Antwort verfuegbar."
        for round_idx in range(self.max_tool_rounds):
            response_message = await self._chat(messages, allow_tools=True)

            if not response_message.tool_calls:
                final_text = response_message.content or final_text
                if self.force_internal_first and round_idx == 0:
                    if is_recommendation and not forced_recommendation_tool_attempt:
                        forced_recommendation_tool_attempt = True
                        messages.append({"role": "system", "content": self._recommendation_force_tool_prompt(query)})
                        if verbose:
                            trace.append("forced: recommendation requires at least one internal tool call")
                        continue

                    forced_context, forced_sources = await self._force_internal_context(tool_query)
                    if forced_context:
                        sources.extend(forced_sources)
                        tools_used.append("forced_internal_lookup")
                        messages.append({"role": "system", "content": forced_context})
                        response_message = await self._chat(messages, allow_tools=False)
                        final_text = response_message.content or final_text
                        if verbose:
                            trace.append(f"forced: internal context added (sources={len(forced_sources)})")
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": response_message.content or "",
                    "tool_calls": response_message.tool_calls,
                }
            )

            tool_payloads: List[Dict[str, Any]] = []
            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name
                raw_args = tool_call.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except Exception:
                    args = {}

                tools_used.append(tool_name)

                tool_result, tool_sources = await self._handle_tool(tool_name, args)
                sources.extend(tool_sources)
                if tool_name == "web_search":
                    web_results_used += len(tool_sources)

                if isinstance(tool_result, dict):
                    tool_payloads.append({"tool": tool_name, **tool_result})
                else:
                    tool_payloads.append({"tool": tool_name, "type": "tool_result", "result": str(tool_result)[:500]})

                if verbose:
                    trace.append(self._format_tool_trace(tool_name, args, tool_result))

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result, ensure_ascii=False)[:4000],
                    }
                )

            last_round = round_idx >= (self.max_tool_rounds - 1)
            if last_round or not self._should_retry_tool_round(tool_payloads):
                if is_recommendation:
                    messages.append({"role": "system", "content": self._recommendation_output_prompt()})
                response_message = await self._chat(messages, allow_tools=False)
                final_text = response_message.content or final_text
                break

            messages.append({"role": "system", "content": self._retry_prompt(tool_payloads)})

        final_text = await self._postprocess_answer(final_text, query, is_recommendation)
        if verbose:
            final_text = self._append_verbose_section(final_text, trace)

        if thread_key:
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

    @staticmethod
    def _strip_verbose_flag(query: str) -> Tuple[str, bool]:
        if not query:
            return "", False
        pattern = re.compile(r"\s+(--verbose|--ausführlich|--ausfuhrlich)\s*$", flags=re.IGNORECASE)
        match = pattern.search(query)
        if not match:
            return query, False
        return query[: match.start()].rstrip(), True

    @staticmethod
    def _dedupe_preserve_order(items: List[str]) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for item in items:
            key = (item or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        candidate = str(text).strip()
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    async def _rewrite_query_for_retrieval(
        self,
        query: str,
        history: List[Dict[str, Any]],
        focus: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Rewrite a potentially context-dependent user query into a standalone query for retrieval."""
        base = (query or "").strip()
        if not base:
            return {"standalone_question": "", "search_queries": [], "is_followup": False}

        if not self.enable_query_rewrite:
            return {"standalone_question": base, "search_queries": [base], "is_followup": False}

        # Keep the rewrite prompt bounded to avoid latency/token bloat.
        history_messages = history
        if self.rewrite_history_turns > 0:
            max_messages = self.rewrite_history_turns * 2
            history_messages = history[-max_messages:]

        history_text = self._render_history(history_messages, max_turns=len(history_messages), max_chars=280)
        focus_json = json.dumps(focus or {}, ensure_ascii=False)[:2000]

        system = (
            "Du bist ein Query-Rewriter fuer einen Conversational-RAG Assistenten. "
            "Aufgabe: Formuliere die LETZTE Nutzerfrage als eigenstaendige, eindeutige Frage, "
            "indem du noetigen Kontext aus Verlauf/Fokus hinzufuegst (IDs, Inventarnummern, Dokumenttitel etc.). "
            "Erfinde nichts: Wenn Kontext nicht eindeutig ist, lasse die Frage weitgehend unveraendert "
            "und setze is_followup=false. "
            "Gib zusaetzlich 2-4 kurze Suchanfragen (synonyme/umformulierungen/Schlagwoerter) aus, "
            "damit semantische Suche auch bei anderen Worten trifft.\n\n"
            "Antworte AUSSCHLIESSLICH als gueltiges JSON-Objekt mit EXACT diesen Schluesseln:\n"
            "- standalone_question: string\n"
            "- search_queries: string[]\n"
            "- is_followup: boolean\n"
        )

        user = (
            "VERLAUF (kompakt):\n"
            f"{history_text}\n\n"
            "FOCUS_JSON:\n"
            f"{focus_json}\n\n"
            "LETZTE_NUTZERFRAGE (verbatim):\n"
            f"{base}\n"
        )

        try:
            msg = await self._chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user[:8000]},
                ],
                allow_tools=False,
                model=self.query_rewrite_model,
                reasoning_effort=self.query_rewrite_reasoning,
                response_format={"type": "json_object"},
                max_tokens=280,
            )
        except Exception as e:  # pragma: no cover - runtime safety
            print(f"[UnifiedAgent] Query rewrite failed: {e}")
            return {"standalone_question": base, "search_queries": [base], "is_followup": False}

        parsed = self._extract_json_object(getattr(msg, "content", "") or "")
        if not parsed:
            return {"standalone_question": base, "search_queries": [base], "is_followup": False}

        standalone = str(parsed.get("standalone_question") or "").strip() or base
        queries_raw = parsed.get("search_queries") or []
        if not isinstance(queries_raw, list):
            queries_raw = []
        queries = [str(q).strip() for q in queries_raw if str(q).strip()]

        # Ensure the standalone query is always included.
        queries = [standalone] + queries
        queries = self._dedupe_preserve_order(queries)[: max(1, self.multi_query_max)]

        is_followup = bool(parsed.get("is_followup"))
        return {
            "standalone_question": standalone,
            "search_queries": queries,
            "is_followup": is_followup,
        }

    @staticmethod
    def _safe_italic(text: str) -> str:
        return str(text).replace("*", "\\*")

    def _append_verbose_section(self, answer: str, trace: List[str]) -> str:
        if not trace:
            return answer
        lines = [f"*{self._safe_italic(t)}*" for t in trace if str(t).strip()]
        return (answer or "").rstrip() + "\n\n---\n" + "\n".join(lines)

    def _format_tool_trace(self, tool_name: str, args: Dict[str, Any], tool_result: Any) -> str:
        try:
            if tool_name == "execute_sql" and isinstance(tool_result, dict):
                sql = (tool_result.get("sql") or "").replace("\n", " ")
                if len(sql) > 180:
                    sql = sql[:180] + "…"
                row_count = tool_result.get("row_count")
                err = tool_result.get("error")
                fixes = tool_result.get("fixes") or []
                fix_text = f", fixes={len(fixes)}" if fixes else ""
                return f"tool execute_sql: row_count={row_count}{fix_text}, sql=\"{sql}\"" + (f", error={err}" if err else "")

            if tool_name == "semantic_search" and isinstance(tool_result, dict):
                q = (tool_result.get("query") or "")
                if len(q) > 80:
                    q = q[:80] + "…"
                results = tool_result.get("results") or []
                top_k = tool_result.get("top_k")
                return f"tool semantic_search: top_k={top_k}, results={len(results)}, query=\"{q}\""

            if tool_name == "web_search" and isinstance(tool_result, dict):
                results = tool_result.get("results") or []
                return f"tool web_search: results={len(results)}"

            if tool_name == "diagnose_data_sources" and isinstance(tool_result, dict):
                diag = tool_result.get("diagnostics") or {}
                pg = diag.get("postgres") or {}
                pc = diag.get("pinecone") or {}
                return f"tool diagnose_data_sources: postgres_available={pg.get('available')}, pinecone_available={pc.get('available')}"

        except Exception:
            pass

        return f"tool {tool_name}: executed"

    def _tool_payload_has_hits(self, payload: Dict[str, Any]) -> bool:
        """Heuristic: determine whether a tool payload contains usable results."""
        if not isinstance(payload, dict):
            return False
        if payload.get("error"):
            return False

        tool = payload.get("tool")
        if tool == "execute_sql":
            return int(payload.get("row_count") or 0) > 0
        if tool in ("semantic_search", "web_search"):
            return bool(payload.get("results"))
        return False

    def _should_retry_tool_round(self, tool_payloads: List[Dict[str, Any]]) -> bool:
        """Retry only when all retrieval tools returned empty/errored."""
        if not self.retry_on_empty:
            return False

        retrieval = [
            p for p in tool_payloads
            if isinstance(p, dict) and p.get("tool") in ("execute_sql", "semantic_search", "web_search")
        ]
        if not retrieval:
            return False

        return not any(self._tool_payload_has_hits(p) for p in retrieval)

    def _retry_prompt(self, tool_payloads: List[Dict[str, Any]]) -> str:
        """Short system nudge to force an adaptive second attempt."""
        reasons: List[str] = []
        for payload in tool_payloads:
            if not isinstance(payload, dict):
                continue
            tool = payload.get("tool") or "tool"
            if payload.get("error"):
                reasons.append(f"{tool}: {payload['error']}")
                continue
            if tool == "execute_sql" and int(payload.get("row_count") or 0) == 0:
                reasons.append("SQL: 0 Zeilen")
            if tool == "semantic_search" and not payload.get("results"):
                reasons.append("Pinecone: 0 Treffer")
            if tool == "web_search" and not payload.get("results"):
                reasons.append("Web: 0 Treffer")

        reason_text = "; ".join(reasons)[:500] if reasons else "keine Treffer"
        return (
            "Die letzten Tool-Ergebnisse waren leer oder fehlerhaft (" + reason_text + "). "
            "Bevor du final antwortest, MUSST du mindestens einen alternativen Tool-Aufruf versuchen: "
            "z.B. bei SQL Filter lockern/ILIKE/%...%/*_search nutzen; bei Pinecone Namespace=both/andere Keywords; "
            "falls Quellen evtl. offline: diagnose_data_sources."
        )

    async def _chat(
        self,
        messages: List[Dict[str, Any]],
        allow_tools: bool,
        *,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Call OpenAI chat with optional tools.

        Args:
            model: Override model name (defaults to response model).
            reasoning_effort: Override reasoning effort (defaults to response reasoning).
            response_format: Optional response format (e.g., {"type": "json_object"}).
            temperature: Optional sampling temperature.
            max_tokens: Optional max completion tokens.
        """
        request: Dict[str, Any] = {"model": model or self.model, "messages": messages}

        if allow_tools:
            request["tools"] = self.tools
            request["tool_choice"] = "auto"

        if response_format:
            request["response_format"] = response_format
        if temperature is not None:
            request["temperature"] = float(temperature)
        if max_tokens is not None:
            # Newer models (e.g., gpt-5*) expect `max_completion_tokens` instead of `max_tokens`.
            request["max_completion_tokens"] = int(max_tokens)

        effective_reasoning = (
            reasoning_effort if reasoning_effort is not None else self.reasoning_effort
        )
        use_reasoning = bool(effective_reasoning and str(effective_reasoning).lower() != "none")
        if use_reasoning:
            request["reasoning"] = {"effort": effective_reasoning}

        try:
            resp = await self.client.chat.completions.create(**request)
            return resp.choices[0].message
        except Exception as e:
            msg = str(e).lower()
            should_retry = False

            # Some models/endpoints may not accept the `reasoning` parameter; retry once without it.
            if use_reasoning and ("reasoning" in msg):
                request.pop("reasoning", None)
                should_retry = True

            # Some models/endpoints may not support response_format.
            if response_format and ("response_format" in msg or "json" in msg):
                request.pop("response_format", None)
                should_retry = True

            # Some models only support default temperature or do not accept `temperature` at all.
            if temperature is not None and ("temperature" in msg):
                request.pop("temperature", None)
                should_retry = True

            # Compatibility: some models still use `max_tokens` instead of `max_completion_tokens`.
            if max_tokens is not None and ("max_completion_tokens" in msg):
                request.pop("max_completion_tokens", None)
                request["max_tokens"] = int(max_tokens)
                should_retry = True

            if should_retry:
                resp = await self.client.chat.completions.create(**request)
                return resp.choices[0].message
            raise

    def _is_recommendation_intent(self, query: str) -> bool:
        q = (query or "").lower()
        keywords = (
            "empfehl",
            "vorschlag",
            "schlage",
            "schlag ",
            "top ",
            "beste",
            "besten",
            "perfekt",
            "geeignet",
            "recommend",
        )
        if any(k in q for k in keywords):
            return True
        if re.search(r"\btop\s*\d+\b", q):
            return True
        return False

    def _recommendation_rules(self) -> str:
        return (
            "Wenn du Maschinen/Geraete empfehlen sollst: "
            "Hole IMMER zuerst interne Kandidaten (Pinecone semantic_search fuer Kandidaten + Ranking; SQL fuer harte Filter/Seriennummern falls noetig). "
            "Nenne pro Empfehlung kurz die Maschine + Seriennummer/ID (falls vorhanden), "
            "leite den Maschinentyp aus `geraetegruppe` ab, "
            "und gib 1-2 konkrete Gruende warum genau diese Empfehlung besser ist als Alternativen "
            "(z.B. Einbaubreite/Arbeitsbreite passt, MIET/Verwendung, passende Bohle/Grundöffnung, Datenvollstaendigkeit). "
            "Keine zufaelligen Picks."
        )

    def _recommendation_force_tool_prompt(self, query: str) -> str:
        return (
            "Der Nutzer verlangt konkrete Empfehlungen (z.B. Top 3/5, 'beste', Seriennummern). "
            "Du MUSST jetzt mindestens eine interne Abfrage machen, bevor du antwortest. "
            "Schritt 1: Verwende `semantic_search` im Namespace 'machinery' (oder 'both') um Kandidaten zu finden. "
            "Nutze falls passend Filter (z.B. verwendung_code='MIET' fuer Mietmaschinen). "
            "Schritt 2 (falls Seriennummer/harte Kriterien benoetigt): Verwende `execute_sql` um Details zu verifizieren oder fehlende Felder nachzuladen. "
            "Nutze `geraetegruppe` um den Maschinentyp zu bestimmen (z.B. Asphaltfertiger, Walze, Bagger). "
            "Danach ranke die Treffer und begruende kurz pro Empfehlung."
        )

    def _recommendation_output_prompt(self) -> str:
        return (
            "Gib jetzt die Top-N Empfehlungen aus (wie gefragt). "
            "Format: kurze Liste mit 3-5 Eintraegen. Pro Eintrag: "
            "Maschinentyp (aus geraetegruppe), Hersteller+Bezeichnung, Seriennummer (oder 'keine Seriennummer hinterlegt'), "
            "und 1-2 Gruende mit konkreten Datenfeldern (z.B. prop_einbaubreite_max, prop_arbeitsbreite, verwendung_code). "
            f"Halte die Gesamtausgabe <= {self.max_answer_words} Woerter."
        )

    async def _postprocess_answer(self, text: str, query: str, is_recommendation: bool) -> str:
        """Enforce brevity and add missing recommendation rationales when needed."""
        if not text:
            return text

        words = len(re.findall(r"\S+", text))
        needs_shortening = words > self.max_answer_words

        rationale_missing = False
        if is_recommendation:
            rationale_missing = not re.search(r"(?i)\b(warum|grund|begruend)\b", text)

        if not (needs_shortening or rationale_missing):
            return text

        system = (
            f"Ueberarbeite die Antwort: max {self.max_answer_words} Woerter, "
            "kurz und praezise. "
        )
        if is_recommendation:
            system += (
                "Bei Empfehlungen MUSST du pro Empfehlung 1-2 Gruende nennen (warum best/Top), "
                "Maschinentyp aus `geraetegruppe` ableiten, und keine zufaelligen Picks. "
            )
        user = f"Frage: {query}\n\nAntwort:\n{text}"

        msg = await self._chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user[:8000]},
            ],
            allow_tools=False,
        )
        return msg.content or text

    async def _force_internal_context(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Ensure internal data is consulted before a general answer.
        If Pinecone isn't available, return a diagnostics hint instead.
        """
        sources: List[Dict[str, Any]] = []

        sem_payload, sem_sources = await self._tool_semantic_search(
            {"query": query, "top_k": int(config.search_top_k), "namespace": "both"}
        )
        sources.extend(sem_sources)

        if isinstance(sem_payload, dict) and not sem_payload.get("error") and sem_payload.get("results"):
            results = sem_payload.get("results", [])[: min(6, len(sem_payload.get("results", [])))]
            snippets = []
            for r in results:
                title = (r.get("title") or "Quelle").strip()
                ns = r.get("namespace") or ""
                content = (r.get("content") or "").strip().replace("\n", " ")
                if len(content) > 400:
                    content = content[:400] + "…"
                snippets.append(f"- [{ns}] {title}: {content}")

            context = (
                "Interne Daten (Pinecone) wurden abgefragt. Nutze diese Infos bevorzugt; "
                "wenn sie nicht passen, sag kurz warum (keine Treffer/zu allgemein) und frag minimal nach.\n"
                + "\n".join(snippets)
            )
            return context, sources

        diag_payload, diag_sources = await self._tool_diagnose_data_sources({})
        sources.extend(diag_sources)

        diag_text = ""
        if isinstance(sem_payload, dict) and sem_payload.get("error"):
            diag_text = f"Pinecone-Fehler: {sem_payload.get('error')}"

        context = (
            "Interne Daten konnten nicht sinnvoll genutzt werden. "
            f"{diag_text}\n"
            "Bevor du Allgemeinwissen nutzt, nenne das knapp und schlage 1 konkreten naechsten Schritt vor "
            "(z.B. Index/Namespace pruefen oder passende interne Quelle nennen)."
        )
        return context, sources

    def _extract_top_n(self, query: str) -> int:
        m = re.search(r"\btop\s*(\d+)\b", (query or "").lower())
        if m:
            return max(1, min(10, int(m.group(1))))
        m2 = re.search(r"\b(\d+)\b", query or "")
        if m2:
            n = int(m2.group(1))
            if 1 <= n <= 10:
                return n
        return 5

    def _infer_usage_filter(self, query: str) -> Optional[Dict[str, Any]]:
        q = (query or "").lower()
        if "miet" in q or "vermiet" in q:
            return {"verwendung_code": {"$eq": "MIET"}}
        if "verkauf" in q:
            return {"verwendung_code": {"$eq": "VK"}}
        return None

    async def _prefetch_recommendation_context(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        For recommendation queries, prefetch top machinery candidates via semantic search
        so the model ranks from internal data instead of guessing.
        """
        top_n = self._extract_top_n(query)
        top_k = max(12, top_n * 4)
        filters = self._infer_usage_filter(query)

        payload, sources = await self._tool_semantic_search(
            {"query": query, "top_k": top_k, "namespace": "machinery", "filters": filters}
        )

        if not isinstance(payload, dict) or payload.get("error") or not payload.get("results"):
            return "", sources

        results = payload.get("results", [])[:top_k]
        lines = []
        for i, r in enumerate(results[: min(len(results), max(8, top_n * 2))], start=1):
            meta = r.get("metadata") if isinstance(r, dict) else {}
            if not isinstance(meta, dict):
                meta = {}

            hersteller = (meta.get("hersteller") or "").strip()
            bezeichnung = (meta.get("bezeichnung") or meta.get("titel") or "").strip()
            serien = (meta.get("seriennummer") or "").strip()
            verwendung = (meta.get("verwendung_code") or meta.get("verwendung") or "").strip()
            geraetegruppe = (meta.get("geraetegruppe") or "").strip()
            mid = (meta.get("id") or r.get("id") or "").strip()

            # Pull a couple of relevant width-like fields if present
            width_bits = []
            width_keys = []
            for k in meta.keys():
                k_low = str(k).lower()
                if any(t in k_low for t in ("einbaubreite", "arbeitsbreite", "bohle")):
                    width_keys.append(k)
            for k in sorted(set(width_keys))[:4]:
                v = meta.get(k)
                if v not in (None, "", "nicht-vorhanden"):
                    width_bits.append(f"{k}={v}")
            width_text = ("; " + ", ".join(width_bits)) if width_bits else ""

            label = " | ".join([p for p in [geraetegruppe, verwendung] if p])
            ident = " / ".join([p for p in [hersteller, bezeichnung] if p]) or "Maschine"
            serien_text = f"SN {serien}" if serien and serien != "nicht-vorhanden" else "SN: nicht hinterlegt"
            lines.append(f"{i}. {ident} ({label}) [{serien_text}] id={mid}{width_text}")

        context = (
            "Interne Kandidaten aus Pinecone (machinery) fuer Empfehlungen. "
            "Ranke daraus die besten Top-N fuer die Anfrage und begruende pro Auswahl mit Datenfeldern:\n"
            + "\n".join(lines)
        )
        return context, sources

    def _supports_reasoning(self) -> bool:
        """Check if model supports the 'reasoning' parameter on chat completions."""
        if not self.model:
            return False
        model_lower = self.model.lower()
        return (
            model_lower.startswith("o1")
            or model_lower.startswith("o3")
            or model_lower.startswith("gpt-5")
        )

    async def _handle_tool(self, name: str, args: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Dispatch tool calls."""
        if name == "execute_sql":
            return await self._tool_execute_sql(args)
        if name == "semantic_search":
            return await self._tool_semantic_search(args)
        if name == "web_search":
            return await self._tool_web_search(args)
        if name == "diagnose_data_sources":
            return await self._tool_diagnose_data_sources(args)

        return {"error": f"Unknown tool {name}"}, []

    def _short_circuit(self, query: str, focus: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return a deterministic answer for certain follow-ups to reduce hallucination risk."""
        q_lower = (query or "").lower()
        if "verbreiter" in q_lower and focus:
            bohle = focus.get("bohle")
            max_width = focus.get("einbaubreite_max")
            machine = focus.get("machine_name") or focus.get("bezeichnung") or ""
            base_width = focus.get("einbaubreite_grund") or focus.get("prop_einbaubreite_grundbohle") or focus.get("prop_einbaubreite_mit_verbreiterungen")

            parts = []
            if bohle:
                parts.append(f"Bohle: {bohle}")
            if machine:
                parts.append(f"Maschine: {machine}")
            if base_width:
                parts.append(f"Grundbohle: {base_width}")
            if max_width:
                parts.append(f"Max. Einbaubreite: {max_width}")

            header = ", ".join(parts) if parts else "Verbreiterungen"
            body = (
                f"{header}\n\n"
                "Verbreiterungs-Module sind in den hinterlegten Daten nicht einzeln aufgeführt. "
                "Bitte Prüfen/Bestellen nach Herstellerangaben (Teilekatalog/Handbuch). "
                "Bekannt sind nur Grund- und Max-Einbaubreite; keine Modulgrößen im System."
            )
            return {
                "response": body,
                "sources": [],
                "chunks_used": 0,
                "response_id": None,
                "web_results_used": 0,
                "query_type": "direct_response",
                "agents_used": ["single_agent"],
                "execution_time_ms": 0,
            }

        return None

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

        # Auto-fix common 'CODE - Name' pitfalls using schema metadata (e.g., verwendung = 'Vermietung')
        sql, sql_fixes = self._auto_fix_code_name_filters(sql)
        sql, text_fixes = self._auto_fix_text_equals_filters(sql)
        sql_fixes.extend(text_fixes)
        sql, prop_fixes = self._auto_fix_unknown_prop_columns(sql)
        sql_fixes.extend(prop_fixes)
        safe_sql = self._enforce_limit(sql, limit)
        rows = self.pg.execute_query(safe_sql)
        pg_error = getattr(self.pg, "last_error", None)
        if pg_error and not rows:
            repaired_sql, repair_notes = self._auto_repair_missing_column(safe_sql, pg_error)
            if repaired_sql and repaired_sql != safe_sql:
                sql_fixes.extend(repair_notes)
                rows = self.pg.execute_query(repaired_sql)
                pg_error = getattr(self.pg, "last_error", None)
                safe_sql = repaired_sql

        if pg_error and not rows:
            return {
                "type": "sql_result",
                "sql": safe_sql,
                "row_count": 0,
                "rows": [],
                "error": f"SQL-Fehler: {pg_error}",
            }, [
                {
                    "title": "PostgreSQL Abfrage (Fehler)",
                    "source_file": "postgres",
                    "score": 1.0,
                    "namespace": "sql",
                    "sql": safe_sql[:200],
                    "notes": "; ".join(sql_fixes) if sql_fixes else None,
                }
            ]
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
            "fixes": sql_fixes,
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
                "notes": "; ".join(sql_fixes) if sql_fixes else None,
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

    def _auto_fix_code_name_filters(self, sql: str) -> Tuple[str, List[str]]:
        """
        Detect and soften brittle equality filters on CODE-Name columns (e.g., verwendung = 'Vermietung').
        Uses schema metadata to rewrite to code/name-aware predicates without hardcoded value maps.
        """
        fixes: List[str] = []
        patched = sql

        def escape(val: str) -> str:
            return val.replace("'", "''").strip()

        def make_predicate(col: str, raw_val: str) -> str:
            val = escape(raw_val)
            like_terms = [f"{val}%", f"%{val}%"]
            candidates = [col, f"{col}_code", f"{col}_name", f"{col}_full"]
            parts: List[str] = []
            for candidate in candidates:
                for term in like_terms:
                    parts.append(f"{candidate} ILIKE '{term}'")
            # de-duplicate while preserving order
            seen = []
            unique_parts = []
            for p in parts:
                if p not in seen:
                    seen.append(p)
                    unique_parts.append(p)
            return "(" + " OR ".join(unique_parts) + ")"

        for col in CODE_NAME_FORMAT_COLUMNS.keys():
            pattern = rf"\b{col}\b\s*=\s*(['\"]?)([^'\"\s][^;,\n)]*?)\1"

            def repl(match: re.Match) -> str:
                value = match.group(2)
                if not value:
                    return match.group(0)
                fixes.append(f"{col} -> code/name-aware match")
                return make_predicate(col, value)

            patched = re.sub(pattern, repl, patched, flags=re.IGNORECASE)

        return patched, fixes

    @staticmethod
    def _normalize_to_search_term(text: str) -> str:
        """Normalize for *_search columns (ASCII-ish, lowercase, no punctuation/spaces)."""
        if not text:
            return ""

        term = str(text).lower()
        replacements = {
            "ä": "ae",
            "ö": "oe",
            "ü": "ue",
            "ß": "ss",
            "Ä": "ae",
            "Ö": "oe",
            "Ü": "ue",
            # Backwards-compat for previously mis-encoded umlaut sequences seen in the repo
            "A": "ae",
            "A,": "ae",
            "A": "oe",
            "A-": "oe",
            "AŹ": "ue",
            "Ao": "ue",
            "AY": "ss",
        }
        for src, dst in replacements.items():
            term = term.replace(src, dst)
        term = re.sub(r"[^a-z0-9]+", "", term)
        return term

    def _auto_fix_text_equals_filters(self, sql: str) -> Tuple[str, List[str]]:
        """
        Detect brittle text equality filters (e.g., hersteller = 'Vögele') and rewrite to ILIKE
        with optional *_search column support. Keeps changes narrow to known text columns.
        """
        fixes: List[str] = []
        patched = sql

        columns = {
            "bezeichnung": "bezeichnung_search",
            "hersteller": "hersteller_search",
            "geraetegruppe": "geraetegruppe_search",
        }

        def escape(val: str) -> str:
            return val.replace("'", "''").strip()

        for col, search_col in columns.items():
            pattern = rf"\b{col}\b\s*=\s*(['\"])(.*?)\1"

            def repl(match: re.Match) -> str:
                raw_val = (match.group(2) or "").strip()
                if len(raw_val) < 2:
                    return match.group(0)

                raw_escaped = escape(raw_val)
                norm = self._normalize_to_search_term(raw_val)
                parts = [f"{col} ILIKE '%{raw_escaped}%'"]
                if norm:
                    parts.append(f"{search_col} ILIKE '%{norm}%'")
                fixes.append(f"{col} '=' -> ILIKE/%...% (+ {search_col})")
                return "(" + " OR ".join(parts) + ")"

            patched = re.sub(pattern, repl, patched, flags=re.IGNORECASE | re.DOTALL)

        return patched, fixes

    def _auto_fix_unknown_prop_columns(self, sql: str) -> Tuple[str, List[str]]:
        """Fix unknown prop_* column names by suggesting closest known prop columns."""
        fixes: List[str] = []
        patched = sql

        prop_tokens = sorted(set(re.findall(r"\bprop_[a-zA-Z0-9_]+\b", sql)))
        if not prop_tokens:
            return patched, fixes

        known = set(PROPERTY_COLUMNS)
        candidates = list(PROPERTY_COLUMNS)

        def normalize_prop_name(name: str) -> str:
            lowered = name.lower()
            return lowered[5:] if lowered.startswith("prop_") else lowered

        def candidate_subset(needle: str) -> List[str]:
            required: List[str] = []
            for token in ("einbau", "breite", "max", "grund", "verbreiter", "bohl", "arbeit"):
                if token in needle:
                    required.append(token)

            subset = candidates
            for token in required:
                subset = [c for c in subset if token in normalize_prop_name(c)]
            return subset or candidates

        def similarity(a: str, b: str) -> float:
            return SequenceMatcher(None, a, b).ratio()

        for prop in prop_tokens:
            if prop in known:
                continue

            needle = normalize_prop_name(prop)
            best: Optional[Tuple[str, float]] = None
            for cand in candidate_subset(needle):
                score = similarity(needle, normalize_prop_name(cand))
                if not best or score > best[1]:
                    best = (cand, score)

            if not best:
                continue

            best_cand, best_score = best
            if best_score < 0.72:
                continue

            patched = re.sub(rf"\b{re.escape(prop)}\b", best_cand, patched)
            fixes.append(f"{prop} -> {best_cand}")

        return patched, fixes

    def _auto_repair_missing_column(self, sql: str, error: str) -> Tuple[str, List[str]]:
        """
        If Postgres reports a missing column, try a schema-aware replacement.
        This prevents the model from hallucinating columns like `status`.
        """
        fixes: List[str] = []
        match = re.search(r'column\\s+\"(?P<col>[^\"]+)\"\\s+does not exist', error, flags=re.IGNORECASE)
        if not match:
            return sql, fixes

        missing = match.group("col")
        if not missing:
            return sql, fixes

        # Prefer domain-aware candidates for status-like questions
        status_like = {"status", "verfuegbarkeit", "mietstatus", "availability", "available"}
        if missing.lower() in status_like:
            replacement = "nuclos_state"
            if replacement in GERAETE_VIEW_BASE_COLUMNS:
                repaired = re.sub(rf"\\b{re.escape(missing)}\\b", replacement, sql)
                fixes.append(f"{missing} -> {replacement}")
                return repaired, fixes

        # Generic: pick closest known base column by string similarity
        best = None
        for cand in GERAETE_VIEW_BASE_COLUMNS:
            score = SequenceMatcher(None, missing.lower(), cand.lower()).ratio()
            if not best or score > best[1]:
                best = (cand, score)

        if best and best[1] >= 0.78:
            replacement = best[0]
            repaired = re.sub(rf"\\b{re.escape(missing)}\\b", replacement, sql)
            fixes.append(f"{missing} -> {replacement}")
            return repaired, fixes

        return sql, fixes

    async def _tool_semantic_search(self, args: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Run Pinecone search across namespaces."""
        query = str(args.get("query") or "").strip()
        top_k = int(args.get("top_k") or config.search_top_k)
        namespace = args.get("namespace", "both")
        filters = args.get("filters")
        pinecone_filter = filters if isinstance(filters, dict) else None

        if not query:
            return {"error": "Keine Suchanfrage erhalten."}, []
        if not self.pinecone_available:
            return {"error": "Pinecone nicht verfuegbar."}, []

        queries_to_run: List[str] = [query]

        # Allow internal callers to pass extra queries (not part of the public tool schema).
        extra_queries = args.get("queries")
        if isinstance(extra_queries, list):
            queries_to_run.extend([str(q).strip() for q in extra_queries if str(q).strip()])

        # Conversation-aware multi-query retrieval (query expansion) to improve recall on paraphrases.
        if self.multi_query_retrieval and self._current_search_queries:
            queries_to_run.extend([str(q).strip() for q in self._current_search_queries if str(q).strip()])

        queries_to_run = self._dedupe_preserve_order(queries_to_run)
        if self.multi_query_retrieval:
            queries_to_run = queries_to_run[: max(1, self.multi_query_max)]
        else:
            queries_to_run = queries_to_run[:1]

        try:
            if len(queries_to_run) == 1:
                embeddings = [await self.embedding_service.embed_query(queries_to_run[0])]
            else:
                embeddings = await self.embedding_service.embed_texts(queries_to_run)
        except Exception as e:  # pragma: no cover - runtime safety
            print(f"[UnifiedAgent] Embedding failed: {e}")
            queries_to_run = [query]
            embeddings = [await self.embedding_service.embed_query(query)]

        namespaces = self._resolve_namespaces(namespace)

        def _trim(v: Any, max_len: int = 240) -> Any:
            if isinstance(v, str) and len(v) > max_len:
                return v[:max_len] + "…"
            return v

        def _select_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(meta, dict):
                return {}

            selected: Dict[str, Any] = {}
            base_keys = [
                "id",
                "seriennummer",
                "inventarnummer",
                "bezeichnung",
                "hersteller",
                "hersteller_code",
                "geraetegruppe",
                "geraetegruppe_code",
                "kategorie",
                "verwendung",
                "verwendung_code",
                "einsatzgebiete",
                "typische_aufgaben",
                "geeignet_fuer",
                "source_file",
                "category",
                "importance",
                "doc_type",
                "machine_name",
                "serial_number",
                "inventory_number",
                "machine_type",
            ]
            for k in base_keys:
                if k in meta and k != "full_data_json":
                    selected[k] = _trim(meta.get(k))

            # Include a small set of width/working-width related fields if present (from machinery metadata)
            width_keys = []
            for k in meta.keys():
                k_low = str(k).lower()
                if any(t in k_low for t in ("einbaubreite", "arbeitsbreite", "bohle", "fraesbreite", "schnittbreite")):
                    if k != "full_data_json":
                        width_keys.append(k)
            for k in sorted(set(width_keys))[:20]:
                if k not in selected:
                    selected[k] = _trim(meta.get(k))

            return selected

        best_results: Dict[str, Dict[str, Any]] = {}
        multi_query_active = len(queries_to_run) > 1

        for q_text, embedding in zip(queries_to_run, embeddings):
            for ns in namespaces:
                try:
                    result = self.index.query(
                        vector=embedding,
                        top_k=top_k,
                        namespace=ns,
                        include_metadata=True,
                        filter=pinecone_filter,
                    )
                    for match in result.matches:
                        meta = match.metadata or {}
                        selected_meta = _select_metadata(meta)
                        entry = {
                            "id": match.id,
                            "score": match.score,
                            "namespace": ns,
                            "title": (
                                meta.get("title")
                                or meta.get("titel")
                                or meta.get("bezeichnung")
                                or meta.get("geraetegruppe")
                                or "Dokument"
                            ),
                            "content": meta.get("content") or meta.get("inhalt") or "",
                            "source_file": meta.get("source_file", "pinecone"),
                            "metadata": selected_meta,
                        }
                        if multi_query_active:
                            entry["matched_query"] = q_text

                        key = f"{ns}:{match.id}"
                        prev = best_results.get(key)
                        if not prev or float(match.score) > float(prev.get("score") or 0):
                            best_results[key] = entry
                except Exception as e:  # pragma: no cover - runtime safety
                    print(f"[UnifiedAgent] Pinecone query failed: {e}")

        all_results = sorted(best_results.values(), key=lambda r: float(r.get("score") or 0), reverse=True)
        limited_results = all_results[: top_k * len(namespaces)]

        sources: List[Dict[str, Any]] = [
            {
                "title": r.get("title", "Dokument"),
                "source_file": r.get("source_file", "pinecone"),
                "score": r.get("score", 0),
                "namespace": r.get("namespace", ""),
            }
            for r in limited_results
        ]

        # Update thread focus using the top semantic hit (helps follow-ups).
        thread_key = self._current_thread_key
        if thread_key and limited_results:
            top_hit = limited_results[0]
            meta = top_hit.get("metadata") if isinstance(top_hit, dict) else {}
            if not isinstance(meta, dict):
                meta = {}

            focus_update = {
                "machine_id": meta.get("id"),
                "inventarnummer": meta.get("inventarnummer") or meta.get("inventory_number"),
                "seriennummer": meta.get("seriennummer") or meta.get("serial_number"),
                "bohle": meta.get("prop_bohle") or meta.get("bohle"),
                "einbaubreite_max": meta.get("prop_einbaubreite_max"),
                "last_source_file": top_hit.get("source_file"),
                "last_title": top_hit.get("title"),
                "last_namespace": top_hit.get("namespace"),
            }
            existing = self._thread_focus.get(thread_key, {})
            existing.update({k: v for k, v in focus_update.items() if v})
            self._thread_focus[thread_key] = existing

        payload = {
            "type": "semantic_results",
            "query": query,
            "queries_used": queries_to_run,
            "top_k": top_k,
            "results": limited_results,
        }
        return payload, sources

    async def _tool_diagnose_data_sources(self, args: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Collect basic availability and stats for underlying data sources."""
        postgres_info: Dict[str, Any] = {
            "available": bool(getattr(self.pg, "available", False)),
            "host": getattr(getattr(self.pg, "config", None), "host", None),
            "database": getattr(getattr(self.pg, "config", None), "database", None),
        }
        if not postgres_info["available"]:
            password = getattr(getattr(self.pg, "config", None), "password", "") or ""
            if not password:
                postgres_info["hint"] = "POSTGRES_PASSWORD fehlt (oder psycopg2 nicht installiert)."
        else:
            try:
                rows = self.pg.execute_query("SELECT COUNT(*) as count FROM geraete")
                if rows and isinstance(rows[0], dict) and "count" in rows[0]:
                    postgres_info["geraete_count"] = rows[0]["count"]
            except Exception as e:  # pragma: no cover - runtime safety
                postgres_info["error"] = str(e)

        pinecone_info: Dict[str, Any] = {
            "available": bool(getattr(self, "pinecone_available", False)),
            "host_configured": bool(getattr(config, "pinecone_host", "")),
            "namespaces_configured": self._resolve_namespaces("both"),
        }
        if not pinecone_info["available"]:
            if not getattr(config, "pinecone_api_key", ""):
                pinecone_info["hint"] = "PINECONE_API_KEY fehlt."
            elif not getattr(config, "pinecone_host", ""):
                pinecone_info["hint"] = "PINECONE_HOST fehlt."
        else:
            try:
                stats = self.index.describe_index_stats()
                if isinstance(stats, dict):
                    pinecone_info["stats"] = stats
                else:
                    namespaces = getattr(stats, "namespaces", None)
                    total = getattr(stats, "total_vector_count", None)
                    dim = getattr(stats, "dimension", None)
                    pinecone_info["stats"] = {
                        "total_vector_count": total,
                        "dimension": dim,
                        "namespaces": self._make_jsonable(namespaces),
                    }
            except Exception as e:  # pragma: no cover - runtime safety
                pinecone_info["error"] = str(e)

        diagnostics = {
            "postgres": postgres_info,
            "pinecone": pinecone_info,
            "web_search_enabled": bool(self.tavily),
        }

        payload = {"type": "diagnostics", "diagnostics": diagnostics}
        sources = [
            {
                "title": "Diagnose Datenquellen",
                "source_file": "internal",
                "score": 1.0,
                "namespace": "diagnostics",
            }
        ]
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

    def _render_history(
        self,
        history: List[Dict[str, str]],
        *,
        max_turns: int = 10,
        max_chars: int = 500,
    ) -> str:
        """Compact conversation history for prompts."""
        if not history:
            return "Keine vorherige Konversation."

        safe_turns = max(1, int(max_turns or 1))
        safe_chars = max(50, int(max_chars or 500))

        lines: List[str] = []
        for turn in history[-safe_turns:]:
            role = turn.get("role", "user")
            content = (turn.get("content") or "")[:safe_chars]
            lines.append(f"- {role}: {content}")
        return "\n".join(lines)

    async def _load_history(self, thread_key: Optional[str]) -> List[Dict[str, Any]]:
        """Load conversation from Redis, fallback to local memory."""
        if not thread_key:
            return []

        if not self.redis_client:
            return self._local_history.get(thread_key, [])

        history_key = f"history:{thread_key}"
        try:
            raw = await self.redis_client.get(history_key)
            if raw:
                return json.loads(raw)

            # Redis is reachable but the key is missing (expired/reset): treat as a fresh thread.
            self._local_history.pop(thread_key, None)
            self._thread_focus.pop(thread_key, None)
            return []
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

    async def clear_thread(self, thread_key: str) -> None:
        """Clear cached conversation state for a specific thread."""
        if not thread_key:
            return

        self._local_history.pop(thread_key, None)
        self._thread_focus.pop(thread_key, None)

        if not self.redis_client:
            return

        try:
            await self.redis_client.delete(f"history:{thread_key}")
        except Exception as e:  # pragma: no cover - redis safety
            print(f"[UnifiedAgent] Redis clear_thread error: {e}")

    async def clear_all_threads(self) -> None:
        """Clear cached conversation state for all threads (Redis + local cache)."""
        self._local_history.clear()
        self._thread_focus.clear()

        if not self.redis_client:
            return

        try:
            cursor = 0
            while True:
                cursor, keys = await self.redis_client.scan(cursor, match="history:*", count=100)
                if keys:
                    await self.redis_client.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:  # pragma: no cover - redis safety
            print(f"[UnifiedAgent] Redis clear_all_threads error: {e}")

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
