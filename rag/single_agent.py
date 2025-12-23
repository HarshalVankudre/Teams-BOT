"""
Single Agent RAG System

A streamlined AI agent with tool-calling capabilities for equipment queries.
Replaces the multi-agent architecture with a single, efficient agent.

Tools:
    - execute_sql: Query PostgreSQL database
    - search_documents: Search Pinecone vector store
"""
import csv
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from .config import config
from .schema import SQL_AGENT_SCHEMA
from .postgres import PostgresService
from .sql_guard import SQLGuard, SQLIntent
from .answer_guard import AnswerGuard, AnswerContext


@dataclass
class AgentResult:
    """Result from the single agent."""
    response: str
    success: bool = True
    error: Optional[str] = None
    execution_time_ms: int = 0
    tools_used: List[str] = field(default_factory=list)
    sql_results_count: int = 0
    sources: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[Dict[str, Any]] = field(default_factory=list)


class SingleAgent:
    """
    Single AI agent with tool-calling for equipment queries.
    
    Uses GPT model with function calling to:
    1. Execute SQL queries against PostgreSQL
    2. Search documents in Pinecone (optional)
    
    Attributes:
        model: OpenAI model to use
        postgres: PostgreSQL service for database queries
        verbose: Enable detailed logging
    """
    
    SYSTEM_PROMPT = f"""RÜKO Baumaschinen-Assistent.

{SQL_AGENT_SCHEMA}

TOOLS:
- execute_sql: Datenbankabfragen
- search_documents: Dokumentensuche (Handbücher, Anleitungen)

DATENPRIORITÄT (WICHTIG):
- Interne Daten (SQL + interne Dokumente) haben immer Vorrang.
- Wenn interne Quellen vorhanden sind, nutze sie und zitiere sie.
- Wenn interne Daten fehlen: sag das explizit und stelle gezielte Rückfragen.

KONTEXT-BEWUSSTSEIN (SEHR WICHTIG):
Du siehst den gesamten Gesprächsverlauf. Nutze ihn IMMER!
- Bei "davon", "diese", "wie viele sind..." → vorherige Filter beibehalten
- Bei "zeige mir mehr" oder "Details" → auf vorherige Ergebnisse beziehen  
- Bei Folgefragen → ALLE vorherigen Kriterien kombinieren
Beispiel: "Mietmaschinen?" → "davon Bomag?" → "davon mit Klimaanlage?"
= WHERE e.verwendung_code = 'MIET' AND e.hersteller_name ILIKE '%bomag%' AND e.prop_klimaanlage IS TRUE

Du bist wie ein Assistent der sich an alles erinnert was besprochen wurde!

ANTWORTZIEL (KURZ & PRAEZISE):
- Beantworte nur die gestellte Frage, keine Extras.
- Wenn etwas unklar ist: genau eine Rueckfrage.
- Standard: 2-4 Saetze oder max. 5 Bulletpoints.
- Laengere Antworten nur auf ausdrueckliche Bitte.

SQL/DATENBANK (KOMPAKT):
- MAX 3 Saetze
- Zaehlen kurz: "45 Bomag-Mietmaschinen"
- Listen: max 5 Bulletpoints

DOKUMENTE (NUR AUF ANFRAGE):
- Erklaerungen nur detailliert, wenn der Nutzer es explizit verlangt.

EMPFEHLUNGEN (SEHR WICHTIG):
- Gib niemals eine Empfehlung nur basierend auf **einem** Merkmal. Du musst immer mehrere Faktoren gegeneinander abwägen und das begründen (nur Fakten aus SQL).

EMPFEHLUNGS-WORKFLOW (Best Practice):
1) Hard Constraints identifizieren (z.B. Breite, Gewicht, Reichweite, Miet/Verkauf, Verfügbarkeit).
2) Alle Kandidaten via SQL bestimmen (mindestens COUNT(*) + Candidate-Set mit WHERE ...).
3) Vergleich: Kandidaten nach mehreren Kriterien vergleichen und ranken (nicht nur das gefragte prop_*).
4) Entscheidung + Begründung: Nenne klar, warum Kandidat A vor B gewinnt (mit konkreten SQL-Werten).
5) Alternativen: Nenne 2-3 Alternativen + wann sie besser wären.

MEHRKRITERIEN-RANKING (SQL-Hilfe):
- Verfügbarkeit: nuclos_state = 'Released' zuerst (Locked nur als Fallback mit Hinweis).
- Nutzung: Wenn der Nutzer Miete will: verwendung_code = 'MIET' bevorzugen.
- Passgenauigkeit bei Zahlenanforderungen: z.B. fit_delta = COALESCE(prop_einbaubreite_max,0) - 3.0 (kleinste positive Differenz zuerst).
- Datenvollständigkeit (um "alle anderen Props" einzubeziehen, ohne zu raten):
  data_completeness = (SELECT COUNT(*) FROM jsonb_each(jsonb_strip_nulls(to_jsonb(e))))
  -> mehr befüllte Felder = bessere Vergleichsbasis; wenn wichtige Felder fehlen, sag das explizit.

WENN der Nutzer messbare Anforderungen nennt (z.B. "3m", "2,5m"):
- MUSST du mit SQL gegen die Equipment-Tabelle validieren (siehe Schema oben).
- Du musst die gesamte Kandidatenmenge berücksichtigen (mindestens via COUNT + Ranking-Query), nicht nur ein kleines Sample.

Für Fertiger/Asphalt-Einbau:
- Breite primär über prop_einbaubreite_max (in Metern) prüfen (z.B. >= 3.0).
- Zusätzlich vergleichen (falls vorhanden): prop_einbaubreite_grundbohle, prop_einbaustaerke, prop_motor_leistung, prop_gewicht, nuclos_state, verwendung_code, data_completeness.

FOLLOW-UPS:
- Bei "welche davon/diese/die alle": beziehe dich auf das zuletzt gelistete Resultset (Thread Context) und halte Filter/Kriterien konstant.

SQL: Haupttabelle ist die Equipment-Tabelle aus dem Schema oben. prop_* sind direkte Spalten (BOOLEAN/DOUBLE/TEXT).
HERSTELLER:
- Hersteller kA¶nnen als Name oder Code vorliegen. Bei Filtern Name + Code berA¼cksichtigen (z.B. hersteller_name ILIKE '%bomag%' OR hersteller_code = 'BOM').
Mietmaschinen: verwendung_code = 'MIET' (nur filtern, wenn der Nutzer explizit Miete will; sonst entweder alle verwendung_code zulassen oder Rueckfrage stellen)."""

    # Tool definitions for OpenAI function calling
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "execute_sql",
                "description": "Execute SQL query on the equipment database. Use for counts, lists, filters, aggregations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "SQL SELECT query to execute. Only SELECT allowed."
                        },
                        "purpose": {
                            "type": "string",
                            "description": "Brief description of what this query does."
                        }
                    },
                    "required": ["sql", "purpose"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Search documents for manuals, guides, specifications. Use when user asks about documentation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for documents."
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    def __init__(
        self,
        model: Optional[str] = None,
        verbose: bool = False,
        pinecone_service=None
    ):
        """
        Initialize the single agent.
        
        Args:
            model: OpenAI model to use (default from config)
            verbose: Enable detailed logging
            pinecone_service: Optional Pinecone service for document search
        """
        self.model = model or config.get_chat_model()
        self.verbose = verbose
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.postgres = PostgresService()
        self.pinecone = pinecone_service
        self.sql_guard = SQLGuard(
            equipment_table=self.postgres.equipment_table,
            column_resolver=self.postgres.get_column_info,
        )
        self.answer_guard = AnswerGuard()

        # Per-thread memory (helps follow-ups like "davon/diese/welche").
        # Keyed by thread_key to avoid cross-user leakage in shared agent instances.
        self._thread_state: Dict[str, Dict[str, Any]] = {}
        self._thread_state_ttl_seconds: int = max(60, int(config.conversation_ttl_hours) * 3600)

        # Lazy-loaded manufacturer lookup (sql_export/manufacturers.csv).
        self._manufacturer_lookup: Optional[List[Dict[str, str]]] = None
        
        self._log(f"Initialized with model: {self.model}")
        self._log(f"PostgreSQL available: {self.postgres.available}")

    def _get_thread_key(self, thread_key: Optional[str]) -> str:
        return thread_key or "default"

    def _load_manufacturer_lookup(self) -> List[Dict[str, str]]:
        if self._manufacturer_lookup is not None:
            return self._manufacturer_lookup

        lookup: List[Dict[str, str]] = []
        root = Path(__file__).resolve().parent.parent
        path = root / "sql_export" / "manufacturers.csv"
        if not path.exists():
            self._manufacturer_lookup = lookup
            return lookup

        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    name = (row.get("name") or "").strip()
                    code = (row.get("code") or "").strip()
                    full_name = (row.get("full_name") or "").strip()
                    if not name or not code:
                        continue
                    tokens = re.findall(r"[a-z0-9]+", name.lower())
                    if not tokens:
                        continue
                    lookup.append({
                        "name": name,
                        "code": code,
                        "full_name": full_name or f"{code} - {name}",
                        "tokens": tokens,
                    })
        except Exception:
            lookup = []

        self._manufacturer_lookup = lookup
        return lookup

    def _manufacturer_matches_for_query(self, user_query: Optional[str]) -> List[Dict[str, str]]:
        if not user_query:
            return []

        lookup = self._load_manufacturer_lookup()
        if not lookup:
            return []

        query_tokens = set(re.findall(r"[a-z0-9]+", user_query.lower()))
        if not query_tokens:
            return []

        matches: List[Dict[str, str]] = []
        seen_codes = set()
        for entry in lookup:
            tokens = entry.get("tokens") or []
            if tokens and all(token in query_tokens for token in tokens):
                code = entry.get("code")
                if code in seen_codes:
                    continue
                seen_codes.add(code)
                matches.append(entry)
        return matches

    def _manufacturer_hints_from_matches(
        self,
        matches: List[Dict[str, str]],
    ) -> Optional[str]:
        if not matches:
            return None

        hints = []
        for entry in matches:
            code = entry.get("code") or ""
            name = entry.get("name") or ""
            full_name = entry.get("full_name") or ""
            if not (code and name):
                continue
            hints.append(
                f"- {name} -> Code '{code}' (voll: '{full_name}')"
            )

        if not hints:
            return None

        return (
            "HERSTELLER-CODE HINWEISE (fuer SQL-Filter):\n"
            + "\n".join(hints)
            + "\nNutze diese Codes zusaetzlich zu hersteller_name (z.B. hersteller_code = 'BOM')."
        )

    def _manufacturer_hints_for_query(self, user_query: Optional[str]) -> Optional[str]:
        return self._manufacturer_hints_from_matches(
            self._manufacturer_matches_for_query(user_query)
        )

    def _prune_thread_state(self) -> None:
        now = time.time()
        expired_keys = [
            key
            for key, state in self._thread_state.items()
            if (now - float(state.get("updated_at", 0.0))) > self._thread_state_ttl_seconds
        ]
        for key in expired_keys:
            self._thread_state.pop(key, None)

    @staticmethod
    def _extract_width_m(text: str) -> Optional[float]:
        """
        Extract a width in meters from text like "3m", "3,0 m", "3.5m".
        Returns None if not found or invalid.
        """
        match = re.search(r"\b(\d+(?:[.,]\d+)?)\s*m\b", (text or "").lower())
        if not match:
            return None
        try:
            return float(match.group(1).replace(",", "."))
        except ValueError:
            return None

    @staticmethod
    def _minimize_sql_rows(rows: List[Dict[str, Any]], max_rows: int = 10) -> List[Dict[str, Any]]:
        if not rows:
            return []

        preferred_keys = [
            "id",
            "bezeichnung",
            "inventarnummer",
            "seriennummer",
            "hersteller_name",
            "geraetegruppe_name",
            "verwendung_code",
            "nuclos_state",
            "nuclos_process",
            "prop_einbaubreite_max",
            "prop_einbaubreite_grundbohle",
            "prop_einbaubreite_mit_verbreiterungen",
            "prop_arbeitsbreite",
            "prop_motor_leistung",
            "prop_gewicht",
            "prop_klimaanlage",
        ]

        minimized: List[Dict[str, Any]] = []
        for row in rows[:max_rows]:
            small = {k: row.get(k) for k in preferred_keys if k in row}
            if not small:
                small = {k: row.get(k) for k in list(row.keys())[:8]}
            minimized.append(small)
        return minimized

    @staticmethod
    def _extract_result_ids(rows: List[Dict[str, Any]], max_ids: int = 25) -> List[Any]:
        ids: List[Any] = []
        for row in rows:
            if "id" in row and row["id"] is not None:
                ids.append(row["id"])
            if len(ids) >= max_ids:
                break
        return ids

    def _format_thread_state(self, state: Dict[str, Any]) -> str:
        parts: List[str] = ["THREAD_CONTEXT (internal; do not reveal directly):"]

        if state.get("target_width_m") is not None:
            parts.append(f"- target_width_m: {state['target_width_m']}")

        if state.get("last_sql_purpose"):
            parts.append(f"- last_sql_purpose: {state['last_sql_purpose']}")
        if state.get("last_sql_row_count") is not None:
            parts.append(f"- last_sql_row_count: {state['last_sql_row_count']}")
        if state.get("last_sql_error"):
            parts.append(f"- last_sql_error: {state['last_sql_error']}")
        if state.get("last_result_ids"):
            parts.append(f"- last_result_ids: {state['last_result_ids']}")
        if state.get("last_sql_results_sample"):
            sample_json = json.dumps(state["last_sql_results_sample"], ensure_ascii=False, default=str)
            parts.append(f"- last_sql_results_sample: {sample_json}")

        parts.append("FOLLOW-UP RULES:")
        parts.append("- If the user says 'davon/diese/die/alle/welche davon': refer to last_result_ids and keep the same filters.")
        table = getattr(self.postgres, "equipment_table", None) or "<equipment_table>"
        parts.append(f"- If you need more fields for those rows, query with: SELECT ... FROM {table} WHERE id IN (...).")
        return "\n".join(parts)

    def _build_documents_context(self, results: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for i, r in enumerate(results[: config.search_top_k], 1):
            metadata = r.get("metadata") or {}
            title = r.get("title") or metadata.get("title") or f"Dokument {i}"
            source_file = r.get("source_file") or metadata.get("source_file") or "Unknown"
            score = r.get("score", 0)
            content = (r.get("content") or metadata.get("content") or "").strip()
            if len(content) > 1200:
                content = content[:1200] + "…"

            parts.append(
                f"### Quelle {i}: {title}\n"
                f"- Datei: {source_file}\n"
                f"- Relevanz: {score:.2%}\n\n"
                f"{content}"
            )

        joined = "\n\n---\n\n".join(parts) if parts else "Keine relevanten internen Dokumente gefunden."
        return (
            "## INTERNE DOKUMENTE (Pinecone Suche)\n"
            f"{joined}\n\n"
            "REGELN:\n"
            "- Nutze diese internen Quellen als Primärquelle.\n"
            "- Zitiere als: \"Laut [Quelle X: Titel – Datei] …\".\n"
            "- Wenn die Quellen die Frage nicht beantworten: sag das und frage gezielt nach fehlenden Details."
        )

    def _sources_from_document_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        for r in (results or [])[: config.search_top_k]:
            metadata = r.get("metadata") or {}
            sources.append({
                "title": r.get("title") or metadata.get("title") or "Dokument",
                "source_file": r.get("source_file") or metadata.get("source_file") or "Unknown",
                "score": r.get("score", 0),
                "namespace": r.get("namespace") or "documents",
                "id": r.get("id"),
            })
        return sources

    def _dedupe_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for s in sources or []:
            key = (s.get("id"), s.get("source_file"), s.get("title"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(s)
        return deduped

    def _log(self, message: str) -> None:
        """Log message if verbose mode enabled."""
        if self.verbose:
            print(f"[SingleAgent] {message}")

    async def process(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict]] = None,
        system_instructions: Optional[str] = None,
        thread_key: Optional[str] = None,
    ) -> AgentResult:
        """
        Process a user query using tool calls.
        
        Args:
            user_query: The user's question
            conversation_history: Previous messages for context
            
        Returns:
            AgentResult with response and metadata
        """
        start_time = time.time()
        tools_used = []
        sql_results_count = 0
        sql_errors: List[str] = []
        sources: List[Dict[str, Any]] = []
        execution_logs = []

        # Per-thread state helps follow-ups ("davon/diese/welche") without leaking across users.
        self._prune_thread_state()
        tk = self._get_thread_key(thread_key)
        thread_state = self._thread_state.get(tk, {})

        width_m = self._extract_width_m(user_query or "")
        if width_m is not None:
            thread_state["target_width_m"] = width_m
            thread_state["updated_at"] = time.time()
            self._thread_state[tk] = thread_state

        # Log initial query
        execution_logs.append({
            "event": "start",
            "query": user_query,
            "timestamp": time.time()
        })

        manufacturer_matches = self._manufacturer_matches_for_query(user_query)
        intent = self.sql_guard.extract_intent(
            user_query,
            thread_state=thread_state,
            manufacturer_matches=manufacturer_matches,
        )
        execution_logs.append({
            "event": "sql_intent",
            "intent": intent.to_dict(),
            "timestamp": time.time()
        })

        if intent.clarification and not intent.followup_ids:
            execution_time = int((time.time() - start_time) * 1000)
            execution_logs.append({
                "event": "clarification_required",
                "message": intent.clarification,
                "timestamp": time.time()
            })
            return AgentResult(
                response=intent.clarification,
                success=True,
                execution_time_ms=execution_time,
                tools_used=[],
                sql_results_count=0,
                sources=[],
                logs=execution_logs
            )
        
        # Build messages (allow caller to inject/override high-level instructions)
        system_prompt = self.SYSTEM_PROMPT
        if system_instructions:
            system_prompt = f"{system_instructions}\n\n---\n\n{self.SYSTEM_PROMPT}"

        messages = [{"role": "system", "content": system_prompt}]

        # Inject per-thread memory early so it is always available to the model.
        if thread_state:
            messages.append({"role": "system", "content": self._format_thread_state(thread_state)})

        manufacturer_hints = self._manufacturer_hints_from_matches(manufacturer_matches)
        if manufacturer_hints:
            messages.append({"role": "system", "content": manufacturer_hints})

        policy_message = self.sql_guard.build_policy_message(intent)
        if policy_message:
            messages.append({"role": "system", "content": policy_message})
        
        # Add conversation history if provided
        if conversation_history:
            max_messages = max(2, int(config.conversation_max_messages))
            for msg in conversation_history[-max_messages:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })

        # Deterministic internal-doc retrieval: avoid noise for structured SQL queries.
        should_prefetch_docs = (
            config.agent_prefetch_documents
            and self.pinecone
            and intent.prefers_documents
        )
        if should_prefetch_docs:
            try:
                doc_results = await self.pinecone.search(user_query, top_k=config.search_top_k)
                execution_logs.append({
                    "event": "prefetch_documents",
                    "count": len(doc_results),
                    "timestamp": time.time()
                })
                if doc_results:
                    tools_used.append("search_documents")
                    sources.extend(self._sources_from_document_results(doc_results))
                    messages.append({"role": "system", "content": self._build_documents_context(doc_results)})
            except Exception as e:
                execution_logs.append({
                    "event": "prefetch_documents_error",
                    "error": str(e),
                    "timestamp": time.time()
                })
                self._log(f"Prefetch documents error: {e}")
        elif config.agent_prefetch_documents and self.pinecone:
            execution_logs.append({
                "event": "prefetch_documents_skipped",
                "reason": "no_doc_signal",
                "timestamp": time.time()
            })
         
        messages.append({"role": "user", "content": user_query})
        
        self._log(f"Processing: {user_query[:100]}...")
        
        try:
            max_completion_tokens = max(1, int(config.agent_max_completion_tokens))
            max_tool_rounds = max(1, int(config.agent_max_tool_rounds))
            tool_rounds = 0

            # Initial call with tools
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.TOOLS,
                tool_choice="auto",
                max_completion_tokens=max_completion_tokens
            )
            
            message = response.choices[0].message
            
            # Handle tool calls if present
            while message.tool_calls:
                tool_rounds += 1
                if tool_rounds > max_tool_rounds:
                    self._log("Tool round limit reached; forcing final response.")
                    messages.append({
                        "role": "system",
                        "content": "Stop calling tools and answer with the best available information."
                    })
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=self.TOOLS,
                        tool_choice="none",
                        max_completion_tokens=max_completion_tokens
                    )
                    message = response.choices[0].message
                    break

                self._log(f"Tool calls: {len(message.tool_calls)}")
                
                # Log tool calls
                execution_logs.append({
                    "event": "tool_calls",
                    "tools": [{
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    } for tc in message.tool_calls],
                    "timestamp": time.time()
                })
                
                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })
                
                # Execute each tool call
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tools_used.append(tool_name)
                    
                    try:
                        args = json.loads(tool_call.function.arguments)
                        result = await self._execute_tool(
                            tool_name,
                            args,
                            intent=intent,
                            thread_key=tk,
                        )
                        
                        if tool_name == "execute_sql" and isinstance(result, dict):
                            sql_results_count += int(result.get("row_count") or 0)

                            # Preserve prior state if this SQL call failed; don't wipe last_result_ids on errors.
                            if result.get("error"):
                                sql_errors.append(str(result.get("error")))
                                state = self._thread_state.get(tk, {})
                                state.update({
                                    "last_sql_error": result.get("error"),
                                    "last_sql_purpose": result.get("purpose") or state.get("last_sql_purpose"),
                                    "last_sql": result.get("sql") or state.get("last_sql"),
                                })
                                if state.get("target_width_m") is None:
                                    state["target_width_m"] = self._extract_width_m(user_query or "")
                                state["updated_at"] = time.time()
                                self._thread_state[tk] = state
                            else:
                                # Persist last SQL results per thread for pronoun follow-ups ("davon/diese").
                                rows = result.get("results") or []
                                state = self._thread_state.get(tk, {})
                                state.update({
                                    "last_sql_error": None,
                                    "last_sql_purpose": result.get("purpose"),
                                    "last_sql": result.get("sql"),
                                    "last_sql_row_count": result.get("row_count"),
                                    "last_result_ids": self._extract_result_ids(rows, max_ids=25),
                                    "last_sql_results_sample": self._minimize_sql_rows(rows, max_rows=10),
                                })
                                if state.get("target_width_m") is None:
                                    state["target_width_m"] = self._extract_width_m(user_query or "")
                                state["updated_at"] = time.time()
                                self._thread_state[tk] = state

                        if tool_name == "search_documents" and isinstance(result, dict):
                            sources.extend(self._sources_from_document_results(result.get("results") or []))
                        
                        # Log tool output
                        execution_logs.append({
                            "event": "tool_result",
                            "tool": tool_name,
                            "output": str(result)[:1000] + "..." if len(str(result)) > 1000 else str(result),
                            "timestamp": time.time()
                        })

                        # Add tool response
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result, ensure_ascii=False, default=str)
                        })
                        
                    except Exception as e:
                        self._log(f"Tool error: {e}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps({"error": str(e)})
                        })
                
                # Get next response
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.TOOLS,
                    tool_choice="auto",
                    max_completion_tokens=max_completion_tokens
                )
                message = response.choices[0].message
            
            # Final response
            final_response = message.content or "Keine Antwort verfügbar."
            sources = self._dedupe_sources(sources)
            tools_used = list(dict.fromkeys(tools_used))
            guard_context = AnswerContext(
                query=user_query,
                tools_used=tools_used,
                sql_results_count=sql_results_count,
                sql_error=sql_errors[-1] if sql_errors else None,
                sources=sources,
                equipment_table=self.postgres.equipment_table,
                intent=intent,
            )
            guarded = self.answer_guard.apply(final_response, guard_context)
            final_response = guarded.response
            if guarded.issues:
                execution_logs.append({
                    "event": "response_guard",
                    "issues": guarded.issues,
                    "timestamp": time.time()
                })
            
            execution_time = int((time.time() - start_time) * 1000)
            self._log(f"Completed in {execution_time}ms, tools: {tools_used}")
            
            execution_logs.append({
                "event": "final_response",
                "content": final_response[:200] + "...",
                "timestamp": time.time()
            })

            return AgentResult(
                response=final_response,
                success=True,
                execution_time_ms=execution_time,
                tools_used=tools_used,
                sql_results_count=sql_results_count,
                sources=sources,
                logs=execution_logs
            )
            
        except Exception as e:
            self._log(f"Error: {e}")
            return AgentResult(
                response=f"Fehler bei der Verarbeitung: {str(e)}",
                success=False,
                error=str(e),
                execution_time_ms=int((time.time() - start_time) * 1000)
            )

    async def _execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        *,
        intent: Optional[SQLIntent] = None,
        thread_key: Optional[str] = None,
    ) -> Any:
        """
        Execute a tool and return results.
        
        Args:
            tool_name: Name of the tool to execute
            args: Tool arguments
            
        Returns:
            Tool execution result
        """
        if tool_name == "execute_sql":
            return await self._execute_sql(args, intent=intent, thread_key=thread_key)
        elif tool_name == "search_documents":
            return await self._search_documents(args)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    async def _execute_sql(
        self,
        args: Dict[str, Any],
        *,
        intent: Optional[SQLIntent] = None,
        thread_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute SQL query with safety checks.
        
        Args:
            args: Contains 'sql' and 'purpose'
            
        Returns:
            Query results or error
        """
        sql = args.get("sql", "")
        purpose = args.get("purpose", "")

        self._log(f"SQL [{purpose}]: {sql[:100]}...")

        if not self.postgres.available:
            return {
                "purpose": purpose,
                "sql": sql,
                "error": self.postgres.availability_error or "PostgreSQL unavailable",
            }

        if intent is None:
            intent = SQLIntent(
                query="",
                requires_sql=False,
                prefers_sql=False,
                prefers_documents=False,
            )

        validation = self.sql_guard.validate_sql(sql, intent)
        if not validation.ok:
            return {
                "purpose": purpose,
                "sql": sql,
                "error": "SQL validation failed",
                "validation_errors": validation.errors,
                "validation_warnings": validation.warnings,
                "normalized_sql": validation.normalized_sql,
                "referenced_tables": validation.referenced_tables,
                "referenced_columns": validation.referenced_columns,
                "limit_value": validation.limit_value,
            }

        prepared_sql, error = self.postgres.prepare_readonly_sql(sql, default_limit=10000)
        if error:
            return {
                "purpose": purpose,
                "sql": sql,
                "error": error,
            }

        try:
            results = self.postgres.execute_query(prepared_sql, raise_on_error=True)
        except Exception as e:
            return {
                "purpose": purpose,
                "sql": prepared_sql,
                "error": str(e),
            }
        
        return {
            "purpose": purpose,
            "sql": prepared_sql,
            "row_count": len(results),
            "results": results[:50] if len(results) > 50 else results,
            "truncated": len(results) > 50,
            "validation_warnings": validation.warnings,
            "referenced_tables": validation.referenced_tables,
            "referenced_columns": validation.referenced_columns,
            "limit_value": validation.limit_value,
        }

    async def _search_documents(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search documents in Pinecone.
        
        Args:
            args: Contains 'query'
            
        Returns:
            Search results or message if not available
        """
        query = args.get("query", "")
        self._log(f"Document search: {query}")
        
        if not self.pinecone:
            return {
                "message": "Dokumentensuche nicht verfügbar",
                "results": []
            }
        
        try:
            results = await self.pinecone.search(query, top_k=config.search_top_k)
            return {
                "query": query,
                "results": results
            }
        except Exception as e:
            return {"error": str(e)}


def create_single_agent(verbose: bool = False, pinecone_service=None) -> SingleAgent:
    """
    Factory function to create a configured SingleAgent.
    
    Args:
        verbose: Enable detailed logging
        
    Returns:
        Configured SingleAgent instance
    """
    return SingleAgent(verbose=verbose, pinecone_service=pinecone_service)
