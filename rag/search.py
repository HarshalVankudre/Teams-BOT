"""
RAG Search — main entry point for the search pipeline.

Routing:
  Advisory queries  → Gemini advisory agent (Google Search Grounding)
  Direct retrieval  → LangGraph ReAct agent (SQL + Pinecone)
  Fallback          → Gemini grounded in Pinecone search
"""
import logging
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import pinecone

from .config import config
from .project_memory import ProjectMemoryStore
from .project_planner import (
    MachineAllocation,
    PlanningConstraints,
    ProjectPhase,
    ProjectPlan,
    ProjectSpec,
)
from .vector_store import PineconeStore
from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class RAGSearch:
    """
    RAG Search with two-path routing:
      1. Gemini advisory agent  - project planning, machine recommendations
      2. LangGraph ReAct agent  - direct inventory / SQL queries
    """

    def __init__(self, redis_client=None):
        self.vector_store = PineconeStore()
        self.embedding_service = EmbeddingService()
        self.redis_client = redis_client
        self.fallback_model = config.fallback_model
        self._gemini_client = None

        if config.google_api_key:
            from google import genai

            self._gemini_client = genai.Client(api_key=config.google_api_key)

        logger.info(
            "RAG runtime ready: advisory_model=%s langgraph_model=%s fallback_model=%s",
            config.advisory_model,
            config.langgraph_model,
            self.fallback_model,
        )

        # Pinecone (grounded fallback retrieval)
        self.pc = pinecone.Pinecone(api_key=config.pinecone_api_key)
        self.index = self.pc.Index(host=config.pinecone_host)
        self.machinery_namespace = config.pinecone_machinery_namespace
        self.documents_namespace = config.pinecone_namespace

        # LangGraph agent (new)
        self.langgraph_agent = None
        if config.use_langgraph_agent:
            try:
                from rag.langgraph_agent import get_langgraph_agent
                self.langgraph_agent = get_langgraph_agent()
                logger.info("LangGraph agent enabled")
            except Exception as e:
                logger.warning("LangGraph agent initialization failed: %s", e)

        # Gemini advisory agent (with native Google Search Grounding)
        self.compound_agent = None
        if config.enable_compound_agent and config.google_api_key:
            try:
                from rag.compound_agent import get_compound_agent
                self.compound_agent = get_compound_agent()
                logger.info("Gemini advisory agent enabled (model=%s)", config.advisory_model)
            except Exception as e:
                logger.warning("Gemini advisory agent initialization failed: %s", e)

        # Track local advisory state with timestamps so stale sessions can be pruned.
        self._advisory_threads: Dict[str, float] = {}
        self._advisory_recommended_fallback: Dict[str, float] = {}
        self._history_fallback: Dict[str, Dict[str, Any]] = {}

        # Persistent compact project memory (max 5 entries/thread)
        self.project_memory_store = ProjectMemoryStore(
            redis_client=redis_client,
            max_memories=max(1, int(config.project_memory_max_items)),
            max_age_seconds=max(1, int(config.advisory_session_timeout_hours)) * 3600,
        )

        # Lazy postgres instance for deterministic SEMA matching
        self._postgres_service = None
        self.enable_expert_planner = bool(config.enable_expert_planner)

    # Advisory sessions need more history for multi-turn conversations
    ADVISORY_MAX_MESSAGES = max(4, int(config.advisory_history_max_messages))
    PLANNER_COMMANDS = {"/plan", "/projektplan", "/projectplan"}

    def _now_ts(self) -> float:
        return time.time()

    def _advisory_session_ttl_seconds(self) -> int:
        return max(1, int(config.advisory_session_timeout_hours)) * 3600

    def _is_recent_session_iso(self, timestamp: Optional[str]) -> bool:
        if not timestamp:
            return False
        try:
            parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        except Exception:
            return False

        now_local = datetime.now().astimezone()
        created_local = parsed.astimezone()
        if created_local.date() != now_local.date():
            return False

        return (now_local - created_local).total_seconds() <= self._advisory_session_ttl_seconds()

    def _is_recent_session_timestamp(self, timestamp: Optional[float]) -> bool:
        if timestamp is None:
            return False

        now_local = datetime.now().astimezone()
        created_local = datetime.fromtimestamp(float(timestamp), tz=timezone.utc).astimezone()
        if created_local.date() != now_local.date():
            return False

        return (now_local - created_local).total_seconds() <= self._advisory_session_ttl_seconds()

    def _get_fallback_history(self, thread_key: str) -> List[Dict[str, str]]:
        entry = self._history_fallback.get(thread_key)
        if not isinstance(entry, dict):
            return []

        updated_at = entry.get("updated_at")
        if not self._is_recent_session_timestamp(updated_at):
            self._history_fallback.pop(thread_key, None)
            return []

        messages = entry.get("messages")
        return messages if isinstance(messages, list) else []

    def _set_fallback_history(self, thread_key: str, messages: List[Dict[str, str]]) -> None:
        self._history_fallback[thread_key] = {
            "updated_at": self._now_ts(),
            "messages": messages,
        }

    def _is_recent_local_advisory_thread(self, thread_key: Optional[str]) -> bool:
        if not thread_key:
            return False

        timestamp = self._advisory_threads.get(thread_key)
        if self._is_recent_session_timestamp(timestamp):
            return True

        self._advisory_threads.pop(thread_key, None)
        return False

    def _is_recent_local_recommendation(self, thread_key: Optional[str]) -> bool:
        if not thread_key:
            return False

        timestamp = self._advisory_recommended_fallback.get(thread_key)
        if self._is_recent_session_timestamp(timestamp):
            return True

        self._advisory_recommended_fallback.pop(thread_key, None)
        return False

    def _touch_local_advisory_thread(self, thread_key: Optional[str]) -> None:
        if thread_key:
            self._advisory_threads[thread_key] = self._now_ts()

    def _prune_local_thread_state(self, thread_key: Optional[str]) -> None:
        if not thread_key:
            return

        self._is_recent_local_advisory_thread(thread_key)
        self._is_recent_local_recommendation(thread_key)
        self._get_fallback_history(thread_key)

    async def _get_conversation_history(self, thread_key: str, advisory: bool = False) -> List[Dict]:
        """Get conversation history from Redis for context.

        Args:
            thread_key: Conversation thread key
            advisory: If True, use higher message limit for 6-phase advisory flow
        """
        if not thread_key:
            return []
        if not self.redis_client:
            history = self._get_fallback_history(thread_key)
            if advisory:
                max_messages = self.ADVISORY_MAX_MESSAGES
            else:
                max_messages = max(2, int(config.conversation_max_messages))
            return history[-max_messages:]
        try:
            import json
            redis_start = time.time()
            history_key = f"chat_history:{thread_key}"
            history_json = await self.redis_client.get(history_key)
            redis_ms = (time.time() - redis_start) * 1000
            print(f"[redis:get_history] {redis_ms:.0f}ms")

            if history_json:
                history_payload = json.loads(history_json)
                history: List[Dict]
                if isinstance(history_payload, dict):
                    updated_at = history_payload.get("updated_at")
                    if advisory and not self._is_recent_session_timestamp(updated_at):
                        await self.redis_client.delete(history_key)
                        return []
                    history = history_payload.get("messages") or []
                elif isinstance(history_payload, list):
                    if advisory:
                        return []
                    history = history_payload
                else:
                    history = []
                if advisory:
                    max_messages = self.ADVISORY_MAX_MESSAGES
                else:
                    max_messages = max(2, int(config.conversation_max_messages))
                return history[-max_messages:]
        except Exception as e:
            print(f"[RAG] Error getting history: {e}")
        history = self._get_fallback_history(thread_key)
        if advisory:
            max_messages = self.ADVISORY_MAX_MESSAGES
        else:
            max_messages = max(2, int(config.conversation_max_messages))
        return history[-max_messages:]

    async def _store_conversation_turn(self, thread_key: str, user_msg: str, assistant_msg: str, advisory: bool = False):
        """Store conversation turn in Redis for full session context."""
        if not thread_key:
            return

        if advisory:
            max_messages = self.ADVISORY_MAX_MESSAGES
        else:
            max_messages = max(2, int(config.conversation_max_messages))

        # Always maintain in-memory fallback history.
        fallback_history = self._get_fallback_history(thread_key)
        fallback_history.append({"role": "user", "content": user_msg})
        fallback_history.append({"role": "assistant", "content": assistant_msg})
        self._set_fallback_history(thread_key, fallback_history[-max_messages:])

        if not self.redis_client:
            return

        try:
            import json
            redis_start = time.time()
            history_key = f"chat_history:{thread_key}"
            history = await self._get_conversation_history(thread_key, advisory=advisory)

            # Add new turn
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": assistant_msg})

            # Advisory sessions keep more messages than direct retrieval
            history = history[-max_messages:]

            # Store with configured TTL
            history_payload = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "messages": history,
            }
            await self.redis_client.setex(
                history_key,
                config.conversation_ttl_hours * 3600,
                json.dumps(history_payload),
            )
            redis_ms = (time.time() - redis_start) * 1000
            print(f"[redis:store_history] {redis_ms:.0f}ms")
        except Exception as e:
            print(f"[RAG] Error storing history: {e}")

    async def _get_recommendation_given(self, thread_key: str) -> bool:
        """Check if a final recommendation was already given for this thread."""
        if not thread_key:
            return False
        if self._is_recent_local_recommendation(thread_key):
            return True
        if not self.redis_client:
            return False
        try:
            import json

            key = f"advisory_recommended:{thread_key}"
            raw = await self.redis_client.get(key)
            if not raw:
                return False
            try:
                payload = json.loads(raw)
            except Exception:
                await self.redis_client.delete(key)
                return False
            if not isinstance(payload, dict) or not self._is_recent_session_iso(payload.get("created_at")):
                await self.redis_client.delete(key)
                return False
            return True
        except Exception:
            return False

    async def _set_recommendation_given(self, thread_key: str):
        """Mark that a final recommendation has been given for this thread."""
        if not thread_key:
            return
        self._advisory_recommended_fallback[thread_key] = self._now_ts()
        if not self.redis_client:
            return
        try:
            import json

            key = f"advisory_recommended:{thread_key}"
            payload = {"created_at": datetime.now(timezone.utc).isoformat()}
            await self.redis_client.setex(key, self._advisory_session_ttl_seconds(), json.dumps(payload))
        except Exception as e:
            print(f"[RAG] Error storing recommendation flag: {e}")

    async def _clear_recommendation_given(self, thread_key: str):
        """Clear final recommendation marker for a thread."""
        if not thread_key:
            return
        self._advisory_recommended_fallback.pop(thread_key, None)
        if not self.redis_client:
            return
        try:
            key = f"advisory_recommended:{thread_key}"
            await self.redis_client.delete(key)
        except Exception as e:
            print(f"[RAG] Error clearing recommendation flag: {e}")

    async def reset_thread_state(self, thread_key: Optional[str]):
        """Reset advisory state, history, and project memory for a thread."""
        if not thread_key:
            return

        self._advisory_threads.pop(thread_key, None)
        self._advisory_recommended_fallback.pop(thread_key, None)
        self._history_fallback.pop(thread_key, None)

        await self.project_memory_store.clear(thread_key)
        await self._clear_recommendation_given(thread_key)

        if not self.redis_client:
            return

        try:
            await self.redis_client.delete(f"chat_history:{thread_key}")
        except Exception as e:
            print(f"[RAG] Error clearing thread history: {e}")

    async def reset_all_thread_state(self):
        """Reset all known advisory state and cached histories."""
        self._advisory_threads.clear()
        self._advisory_recommended_fallback.clear()
        self._history_fallback.clear()
        self.project_memory_store._fallback_memories.clear()

        if not self.redis_client:
            return

        for pattern in ("chat_history:*", "advisory_recommended:*", "project_memories:*"):
            try:
                cursor = 0
                while True:
                    cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=100)
                    if keys:
                        await self.redis_client.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as e:
                print(f"[RAG] Error clearing keys for pattern {pattern}: {e}")

    def _looks_like_new_project_start(self, query: str) -> bool:
        """Heuristic: identify fresh project kickoff questions."""
        q = (query or "").lower().strip()
        if not q:
            return False

        kickoff_patterns = [
            r"\bneues?\s+projekt\b",
            r"\bnew\s+project\b",
            r"\bvon\s+vorne\b",
            r"\bstart\s+neu\b",
            r"\bwenn\s+ich\b.*\bbau\w*\b",
            r"\bich\s+(?:will|moechte)\b.*\bbau\w*\b",
            r"\bwas\s+brauch\w*\s+ich\s+fuer\b",
            r"\bwelche\s+maschinen\s+brauch\w*\s+ich\b",
            r"\bwelche\s+maschinen\s+fuer\b",
            r"\bstrasse\b.*\bbau\w*\b",
            r"\broad\b.*\bbuild\w*\b",
        ]
        return any(re.search(pattern, q) for pattern in kickoff_patterns)

    def _looks_like_followup_query(self, query: str) -> bool:
        """Heuristic: identify follow-up questions to an existing recommendation."""
        q = (query or "").lower().strip()
        if not q:
            return False

        followup_patterns = [
            r"\balternative\w*\b",
            r"\bdetails?\b",
            r"\bmehr\b",
            r"\bwarum\b",
            r"\bwelches?\s+modell\b",
            r"\bseriennummer\b",
            r"\binventarnummer\b",
            r"\bverfuegbar\w*\b",
            r"\bverfuegbarkeit\b",
            r"\bdazu\b",
            r"\bdavon\b",
            r"\bdiese\b",
            r"\bdie\s+genannten\b",
            r"\bund\s+wenn\b",
            r"\bwhat about\b",
        ]
        return any(re.search(pattern, q) for pattern in followup_patterns)

    def _normalize_command_token(self, token: str) -> str:
        return (
            (token or "")
            .lower()
            .replace("\u00fc", "ue")
            .replace("\u00f6", "oe")
            .replace("\u00e4", "ae")
            .replace("\u00df", "ss")
        )

    def _extract_planner_command(self, query: str) -> Tuple[str, bool]:
        """Extract /plan command payload and return (normalized_query, force_planner)."""
        raw = (query or "").strip()
        if not raw.startswith("/"):
            return raw, False

        parts = raw.split(maxsplit=1)
        command = self._normalize_command_token(parts[0])
        if command not in self.PLANNER_COMMANDS:
            return raw, False

        payload = parts[1].strip() if len(parts) > 1 else ""
        return payload, True

    def _project_planner_intent_score(self, query: str) -> Tuple[int, List[str]]:
        text = (query or "").strip().lower()
        if not text:
            return 0, []

        score = 0
        reasons: List[str] = []

        def add(points: int, label: str):
            nonlocal score
            score += points
            reasons.append(label)

        if re.search(r"\bprojekt\w*\b|\bproject\w*\b|\bpro+ject\w*\b", text):
            add(2, "project_term")

        if re.search(r"\bplanung\b|\bplan\w*\b|\broadmap\b|\bworkflow\b|\bablauf\b|\bphasen\w*\b", text):
            add(2, "planning_term")

        if re.search(r"\bbau\w*\b|\bbuild\w*\b|\bconstruction\b|\bconstruct\w*\b", text):
            add(2, "construction_term")

        if re.search(
            r"\b(welche|which|what)\s+maschinen?\b|\bmaschinen?\s+(brauch\w*|benoetig\w*|need\w*|required)\b",
            text,
        ):
            add(2, "machine_need_term")

        if re.search(r"\bstrasse\b|\broad\b|\basphalt\w*\b|\bbeton\w*\b|\bfundament\b|\bsite\b", text):
            add(1, "civil_context_term")

        if re.search(r"\bentire\b|\bfull\b|\bend-to-end\b|\bkomplett\b|\bgesamt\w*\b", text):
            add(1, "scope_term")

        if re.search(
            r"\b(database|db|inventory|bestand|available|verfuegbar|machinery)\b.*\b(based|basis|nur|only)\b|"
            r"\b(based|basis)\b.*\b(database|db|inventory|bestand)\b",
            text,
        ):
            add(1, "inventory_grounding_term")

        return score, reasons

    def _has_inventory_context(self, query: str) -> bool:
        text = (query or "").strip().lower()
        if not text:
            return False

        patterns = [
            r"\bmietpark\b",
            r"\bbestand\b",
            r"\bin\s+der\s+datenbank\b",
            r"\baus\s+(?:dem|unserem)\s+(?:mietpark|bestand)\b",
            r"\bbei\s+r[üu]ko\b",
            r"\bhaben\s+wir\b",
            r"\bwir\s+haben\b",
        ]
        return any(re.search(pattern, text) for pattern in patterns)

    def _contains_machine_identifier(self, query: str) -> bool:
        text = (query or "").strip()
        if not text:
            return False

        if re.search(r"\b(?:seriennummer|inventarnummer|sn|id)\b", text, flags=re.IGNORECASE):
            return True

        if re.search(r"\b(?:maschine|geraet|gerät)\b", text, flags=re.IGNORECASE):
            if re.search(r"\b\d{5,}\b", text):
                return True
            if re.search(r"\b[A-Z0-9-]{6,}\b", text):
                return True

        return False

    def _is_machine_lookup_query(self, query: str) -> bool:
        text = (query or "").strip().lower()
        if not text:
            return False

        if re.search(r"\b(?:diese|die)\s+maschine\b", text):
            return True

        if not self._contains_machine_identifier(text):
            return False

        property_patterns = [
            r"\beigenschaft\w*\b",
            r"\binfo\w*\b",
            r"\bdetails?\b",
            r"\bwelche\b.*\b(?:hoehe|breite|laenge|gewicht|grabtiefe|arbeitsbreite|bohlentyp|verwendung|status)\b",
            r"\bgib\b.*\b(?:info\w*|eigenschaft\w*)\b",
            r"\bohne\s+interpretation\b",
        ]
        return any(re.search(pattern, text) for pattern in property_patterns) or self._contains_machine_identifier(text)

    def _is_inventory_recommendation_query(self, query: str) -> bool:
        text = (query or "").strip().lower()
        if not text or not self._has_inventory_context(text):
            return False

        recommendation_patterns = [
            r"\bempfehl\w*\b",
            r"\bpassend\w*\b",
            r"\bwelche\s+maschine\b",
            r"\bwelchen?\b",
        ]
        return any(re.search(pattern, text) for pattern in recommendation_patterns)

    def _is_raw_machine_dump_query(self, query: str) -> bool:
        text = (query or "").strip().lower()
        if not text:
            return False

        patterns = [
            r"\balle\s+eigenschaft\w*\b",
            r"\balle\s+infos?\b",
            r"\balle\s+information\w*\b",
            r"\bin\s+einer\s+tabelle\b",
            r"\bohne\s+interpretation\b",
            r"\bnur\s+eigenschaft\s+und\s+wert\b",
        ]
        return self._contains_machine_identifier(text) and any(re.search(pattern, text) for pattern in patterns)

    def _extract_machine_identifier(self, query: str) -> Tuple[str, str]:
        text = (query or "").strip()
        if not text:
            return "", ""

        patterns = [
            ("id", r"\bid\b\s*[:#-]?\s*(\d{3,})\b"),
            ("serial", r"\bseriennummer\b\s*[:#-]?\s*([A-Za-z0-9-]{3,})\b"),
            ("inventory", r"\binventarnummer\b\s*[:#-]?\s*([A-Za-z0-9-]{3,})\b"),
        ]
        for kind, pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return kind, match.group(1).strip()

        machine_tail = re.search(r"\b(?:maschine|geraet|gerät)\b(.+)$", text, flags=re.IGNORECASE)
        if machine_tail:
            tokens = re.findall(r"[A-Za-z0-9-]{5,}", machine_tail.group(1))
            if tokens:
                numeric_tokens = [token for token in tokens if any(char.isdigit() for char in token)]
                candidate = numeric_tokens[-1] if numeric_tokens else tokens[-1]
                return "reference", candidate.strip()

        return "", ""

    def _extract_identifier_from_text(self, text: str) -> Tuple[str, str]:
        return self._extract_machine_identifier(text or "")

    def _resolve_machine_reference_from_history(self, history: List[Dict[str, Any]]) -> Tuple[str, str]:
        for message in reversed(history or []):
            content = (message.get("content") or "").strip()
            identifier_type, identifier_value = self._extract_identifier_from_text(content)
            if identifier_value:
                return identifier_type, identifier_value
        return "", ""

    def _fetch_machine_full_row(self, identifier_type: str, identifier_value: str) -> Optional[Dict[str, Any]]:
        postgres = self._get_postgres_service()
        if not postgres or not identifier_value:
            return None

        if identifier_type == "id":
            return postgres.get_equipment_by_id(identifier_value)

        table_name = postgres.equipment_table
        like_value = f"%{identifier_value}%"
        sql = f"""
            SELECT *
            FROM {table_name}
            WHERE CAST(id AS TEXT) = %s
               OR seriennummer = %s
               OR inventarnummer = %s
               OR seriennummer ILIKE %s
               OR inventarnummer ILIKE %s
               OR bezeichnung ILIKE %s
            ORDER BY
                CASE
                    WHEN seriennummer = %s THEN 0
                    WHEN inventarnummer = %s THEN 1
                    WHEN CAST(id AS TEXT) = %s THEN 2
                    WHEN bezeichnung ILIKE %s THEN 3
                    ELSE 4
                END,
                id
            LIMIT 1
        """
        params = [
            identifier_value,
            identifier_value,
            identifier_value,
            like_value,
            like_value,
            like_value,
            identifier_value,
            identifier_value,
            identifier_value,
            like_value,
        ]
        rows = postgres.execute_query(sql, params)
        return rows[0] if rows else None

    def _format_raw_machine_table(self, row: Dict[str, Any], *, strict: bool) -> str:
        core_fields = [
            ("bezeichnung", "Bezeichnung"),
            ("hersteller_name", "Hersteller"),
            ("geraetegruppe_name", "Geraetegruppe"),
            ("seriennummer", "Seriennummer"),
            ("inventarnummer", "Inventarnummer"),
            ("verwendung_code", "Verwendung"),
            ("nuclos_state", "Nuclos-Status"),
        ]

        rows: List[Tuple[str, str]] = []
        for field_name, label in core_fields:
            value = row.get(field_name)
            if value not in (None, ""):
                rows.append((label, str(value)))

        properties = row.get("properties_jsonb") or {}
        if isinstance(properties, dict):
            for key in sorted(properties):
                value = properties.get(key)
                if value not in (None, ""):
                    rows.append((str(key), str(value)))

        if not rows:
            return "Keine Datenbankwerte gefunden."

        table_lines = [
            "| Eigenschaft | Wert |",
            "|---|---|",
        ]
        table_lines.extend(
            f"| {str(label).replace('|', '/')} | {str(value).replace('|', '/')} |"
            for label, value in rows
        )
        table = "\n".join(table_lines)
        return table if strict else f"Datenbankwerte zur Maschine:\n\n{table}"

    def _normalize_property_query_text(self, text: str) -> str:
        return (
            (text or "")
            .lower()
            .replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
            .replace("ß", "ss")
        )

    def _extract_property_from_machine_row(self, row: Dict[str, Any], query: str) -> Optional[Dict[str, str]]:
        text = self._normalize_property_query_text(query)
        props = row.get("properties_jsonb") or {}
        if not isinstance(props, dict):
            props = {}

        direct_rules = [
            (
                ("hoehe",),
                "Hoehe",
                lambda: row.get("hoehe_mm_num") or props.get("prop_e1860_hoehe_mm"),
            ),
            (
                ("breite",),
                "Breite",
                lambda: row.get("breite_mm_num") or props.get("prop_e1330_breite_mm"),
            ),
            (
                ("laenge",),
                "Laenge",
                lambda: row.get("laenge_mm_num") or props.get("prop_e1990_laenge_mm"),
            ),
            (
                ("gewicht",),
                "Gewicht",
                lambda: row.get("gewicht_kg_num") or props.get("prop_e1730_gewicht_kg"),
            ),
            (
                ("verwendung", "nutzung"),
                "Verwendung",
                lambda: row.get("verwendung_name") or row.get("verwendung_code"),
            ),
            (
                ("bohlentyp", "bohle"),
                "Bohlentyp",
                lambda: props.get("prop_e2970_bohle_typ"),
            ),
            (
                ("hgt", "schotter"),
                "Einbau von HGT/Schotter",
                lambda: props.get("prop_e3070_einbau_von_hgt_schotter"),
            ),
            (
                ("grundbohle",),
                "Einbaubreite Grundbohle",
                lambda: row.get("einbaubreite_grundbohle_m_num") or props.get("prop_e1470_einbaubreite_grundbohle_m"),
            ),
            (
                ("verbreiterung", "verbreiterungen"),
                "Einbaubreite mit Verbreiterungen",
                lambda: row.get("einbaubreite_verbreiterungen_m_num") or props.get("prop_e1490_einbaubreite_mit_verbreiterungen_m"),
            ),
            (
                ("einbaubreite", "maximalbreite", "maximale breite"),
                "Einbaubreite max.",
                lambda: row.get("einbaubreite_max_m_num") or props.get("prop_e1480_einbaubreite_max_m"),
            ),
            (
                ("grabtiefe",),
                "Grabtiefe",
                lambda: row.get("grabtiefe_mm_num") or props.get("prop_e1740_grabtiefe_mm"),
            ),
        ]

        for keywords, label, getter in direct_rules:
            if any(keyword in text for keyword in keywords):
                value = getter()
                if value not in (None, ""):
                    return {"label": label, "value": str(value)}

        if re.search(r"\bverfuegbar\w*\b|\bverfuegbarkeit\b|\bstatus\b", text):
            state = row.get("nuclos_state")
            if state:
                return {
                    "label": "Nuclos-Status",
                    "value": f"{state} - Released ist nur Bestandsstatus im System, nicht die gesicherte Live-Verfuegbarkeit.",
                }

        if re.search(r"\b(?:welche|gib|nenn)\b.*\bseriennummer\b", text):
            serial = row.get("seriennummer")
            if serial:
                return {"label": "Seriennummer", "value": str(serial)}

        if re.search(r"\b(?:welche|gib|nenn)\b.*\binventarnummer\b", text):
            inventory = row.get("inventarnummer")
            if inventory:
                return {"label": "Inventarnummer", "value": str(inventory)}

        return None

    def _format_machine_property_response(
        self,
        row: Dict[str, Any],
        property_hit: Dict[str, str],
        *,
        strict: bool,
    ) -> str:
        serial = row.get("seriennummer") or row.get("inventarnummer") or row.get("id")
        reference = f"Seriennummer {serial}" if row.get("seriennummer") else f"Referenz {serial}"
        if strict:
            return (
                "| Eigenschaft | Wert |\n"
                "|---|---|\n"
                f"| {property_hit['label']} | {str(property_hit['value']).replace('|', '/')} |"
            )

        return (
            f"Die Maschine mit {reference} hat folgende Angabe in der Datenbank:\n\n"
            f"- {property_hit['label']}: {property_hit['value']}"
        )

    async def _try_direct_machine_property_response(
        self,
        query: str,
        thread_key: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if not self._is_machine_lookup_query(query):
            return None

        normalized_query = self._normalize_property_query_text(query)
        if not re.search(
            r"\b(hoehe|breite|laenge|gewicht|verwendung|nutzung|bohle|bohlentyp|hgt|schotter|grundbohle|verbreiter|einbaubreite|grabtiefe|seriennummer|inventarnummer|verfuegbar|verfuegbarkeit|status)\b",
            normalized_query,
        ):
            return None

        identifier_type, identifier_value = self._extract_machine_identifier(query)
        if not identifier_value and re.search(r"\b(?:diese|dieses|der|die)\s+(?:maschine|geraet|gerät|fertiger|bagger)\b", normalized_query):
            history = await self._get_conversation_history(thread_key) if thread_key else []
            identifier_type, identifier_value = self._resolve_machine_reference_from_history(history)

        if not identifier_value:
            return None

        row = self._fetch_machine_full_row(identifier_type, identifier_value)
        if not row:
            return None

        property_hit = self._extract_property_from_machine_row(row, query)
        if not property_hit:
            return None

        strict = bool(re.search(r"\bohne\s+interpretation\b|\bin\s+einer\s+tabelle\b|\bnur\s+eigenschaft\s+und\s+wert\b", query, flags=re.IGNORECASE))
        return {
            "response": self._format_machine_property_response(row, property_hit, strict=strict),
            "sources": [],
            "chunks_used": 0,
            "response_id": None,
            "web_results_used": 0,
            "query_type": "machine_property_db",
            "agents_used": ["postgres_direct_property"],
            "execution_time_ms": 0,
            "agent": "postgres_direct",
        }

    def _try_raw_machine_dump(self, query: str) -> Optional[Dict[str, Any]]:
        if not self._is_raw_machine_dump_query(query):
            return None

        identifier_type, identifier_value = self._extract_machine_identifier(query)
        if not identifier_value:
            return None

        machine = self._fetch_machine_full_row(identifier_type, identifier_value)
        if not machine:
            return None

        strict = bool(re.search(r"\bohne\s+interpretation\b|\bnur\s+eigenschaft\s+und\s+wert\b", query, flags=re.IGNORECASE))
        return {
            "response": self._format_raw_machine_table(machine, strict=strict),
            "sources": [],
            "chunks_used": 0,
            "response_id": None,
            "web_results_used": 0,
            "query_type": "machine_raw_db",
            "agents_used": ["postgres_direct"],
            "execution_time_ms": 0,
            "agent": "postgres_direct",
        }

    def _is_explicit_retrieval_only(self, query: str) -> bool:
        text = (query or "").strip().lower()
        if not text:
            return False

        if (
            self._is_machine_lookup_query(text)
            or self._is_inventory_recommendation_query(text)
            or self._is_raw_machine_dump_query(text)
        ):
            return True

        retrieval_patterns = [
            r"\bwie\s*viele?\b",
            r"\bhow\s+many\b",
            r"\bzeig\w*\b",
            r"\bshow\b",
            r"\bliste\b",
            r"\blist\b",
            r"\binfos?\s+(zu|zum|von|ueber|über|fuer)\b",
            r"\bn[aä]here?\s+infos?\b",
            r"\bseriennummer\b",
            r"\bserial\s*number\b",
            r"\binventarnummer\b",
            r"\binventory\s*number\b",
            r"\bverfuegbar\w*\b",
            r"\bavailable\b",
            r"\bcount\b",
            r"\beigenschaft\w*\b",
            r"\bdetails?\s+(zu|von|fuer)\b",
            r"\bohne\s+interpretation\b",
        ]
        has_retrieval = any(re.search(pattern, text) for pattern in retrieval_patterns)
        if not has_retrieval:
            return False

        planner_score, _ = self._project_planner_intent_score(text)
        return planner_score < 3

    def _is_advisory_query(self, query: str) -> bool:
        """Classify whether a query is advisory (needs web search) vs direct retrieval."""
        query_lower = query.lower().strip()

        if self._is_machine_lookup_query(query_lower):
            print("[RAG] Query classified as DIRECT RETRIEVAL (machine lookup override)")
            return False

        if self._is_inventory_recommendation_query(query_lower):
            print("[RAG] Query classified as DIRECT RETRIEVAL (inventory recommendation override)")
            return False

        planner_score, planner_reasons = self._project_planner_intent_score(query_lower)
        if planner_score >= 3 and not self._is_explicit_retrieval_only(query_lower):
            reason_text = ", ".join(planner_reasons[:3]) if planner_reasons else "score"
            print(f"[RAG] Query classified as ADVISORY (planner_intent: {planner_score}, {reason_text})")
            return True

        direct_patterns = [
            r"\bwie\s*viele?\b",
            r"\bhow\s+many\b",
            r"\bzeig\w*\b",
            r"\bshow\b",
            r"\bliste\b",
            r"\blist\b",
            r"\binfos?\s+(zu|zum|von|ueber|über|fuer)\b",
            r"\bn[aä]here?\s+infos?\b",
            r"\bwelche\s+maschinen\b.*\bhaben\s+wir\b",
            r"\bwelche\s+maschinen\b.*\b(im\s+bestand|verfuegbar|liste|anzahl)\b",
            r"\bwhich\s+machines?\b.*\b(we\s+have|in\s+inventory|available|list|count)\b",
            r"\bseriennummer\b",
            r"\bserial\s*number\b",
            r"\binventarnummer\b",
            r"\binventory\s*number\b",
            r"\bverfuegbar\w*\b",
            r"\bavailable\b",
            r"\banzahl\b",
            r"\bzahle?\b",
            r"\bcount\b",
            r"\balle\s+\w+fertiger\b",
            r"\balle\s+\w+bagger\b",
            r"\balle\s+\w+walze\b",
            r"\balle\s+\w+fraese\b",
            r"\bhersteller\b.*\bhaben\b",
            r"\bbestand\b",
            r"\bmiete?\b",
            r"\bverkauf\b",
            r"\bdetails?\s+(zu|von|fuer)\b",
        ]

        advisory_patterns = [
            r"\bwas\s+brauch\w*\b",
            r"\bwas\s+benoetig\w*\b",
            r"\bwelche\s+maschinen?\s+fuer\b",
            r"\bempfehl\w*\b",
            r"\bbauen\b",
            r"\bbuild\w*\b",
            r"\bprojekt\b",
            r"\bproject\w*\b",
            r"\bpro+ject\w*\b",
            r"\bstrasse\b",
            r"\broad\b",
            r"\basphalt\w*\b",
            r"\bbeton\w*\b",
            r"\bfundament\b",
            r"\baushub\b",
            r"\bexcavat\w*\b",
            r"\bverdicht\w*\b",
            r"\bfraes\w*arbeit\b",
            r"\bwie\s+(kann|soll)\b",
            r"\bhow\s+(can|should)\b",
            r"\btipps?\b",
            r"\bbest\w*\s*practice\b",
            r"\bvorgehensweise\b",
            r"\bplanung\b",
            r"\bplanning\b",
            r"\bwofuer\b",
            r"\bwozu\b",
            r"\bgeeignet\w*\b",
            r"\bworkflow\b",
            r"\bphases?\b",
            r"\bstep[-\s]*by[-\s]*step\b",
            r"\bcomplete\s+plan\b",
        ]

        direct_matches = [pattern for pattern in direct_patterns if re.search(pattern, query_lower)]
        advisory_matches = [pattern for pattern in advisory_patterns if re.search(pattern, query_lower)]

        # Resolve ambiguous "welche maschinen ..." by intent context.
        if re.search(r"\bwelche\s+maschinen\b", query_lower):
            inventory_hints = [
                r"\bhaben\s+wir\b",
                r"\bim\s+bestand\b",
                r"\bverfuegbar\w*\b",
                r"\bseriennummer\b",
                r"\binventarnummer\b",
                r"\banzahl\b",
                r"\bliste\b",
            ]
            project_hints = [
                r"\bbauen\b",
                r"\bprojekt\b",
                r"\bstrasse\b",
                r"\bweg\b",
                r"\bsanierung\b",
                r"\bneubau\b",
                r"\baushub\b",
                r"\bverdicht\w*\b",
                r"\bfuer\b",
                r"\bbrauch\w*\b",
                r"\bbenoetig\w*\b",
            ]

            has_inventory_hint = any(re.search(pattern, query_lower) for pattern in inventory_hints)
            has_project_hint = any(re.search(pattern, query_lower) for pattern in project_hints)

            if has_project_hint and not has_inventory_hint:
                advisory_matches.append("__welche_maschinen_project__")
            elif has_inventory_hint and not has_project_hint:
                direct_matches.append("__welche_maschinen_inventory__")
            elif has_project_hint and has_inventory_hint:
                advisory_matches.append("__welche_maschinen_mixed__")
            else:
                advisory_matches.append("__welche_maschinen_default__")

        if advisory_matches and not direct_matches:
            print(f"[RAG] Query classified as ADVISORY (matched: {advisory_matches[0]})")
            return True

        if direct_matches and not advisory_matches:
            print(f"[RAG] Query classified as DIRECT RETRIEVAL (matched: {direct_matches[0]})")
            return False

        if advisory_matches and direct_matches:
            # Ambiguous: both direct retrieval AND advisory signals present.
            # Only let advisory win if a STRONG planning-intent signal matched.
            # Weak signals (asphalt, strasse, beton, bauen...) appear in retrieval
            # queries too ("Wie viele Asphaltfertiger haben wir?") and must not
            # hijack the flow away from a fast SQL answer.
            strong_advisory_patterns = [
                r"\bwas\s+brauch\w*\b",
                r"\bwas\s+benoetig\w*\b",
                r"\bwelche\s+maschinen?\s+fuer\b",
                r"\bwelche\s+maschinen?\s+(brauch|benoetig|need|required)\w*\b",
                r"\bempfehl\w*\b",
                r"\bwofuer\b",
                r"\bwozu\b",
                r"\bgeeignet\w*\b",
                r"\bwie\s+(kann|soll)\b",
                r"\bhow\s+(can|should)\b",
                r"\btipps?\b",
                r"\bbest\w*\s*practice\b",
                r"\bvorgehensweise\b",
                r"\bworkflow\b",
                r"\bstep[-\s]*by[-\s]*step\b",
                r"\bcomplete\s+plan\b",
            ]
            has_strong_advisory = any(re.search(p, query_lower) for p in strong_advisory_patterns)
            if has_strong_advisory:
                print(
                    "[RAG] Query classified as ADVISORY "
                    f"(strong planning intent overrides direct: {advisory_matches[0]})"
                )
                return True
            print(
                "[RAG] Query classified as DIRECT RETRIEVAL "
                f"(weak advisory + direct, retrieval wins: {direct_matches[0]})"
            )
            return False

        print("[RAG] Query classified as DIRECT RETRIEVAL (default)")
        return False

    async def _process_compound_query(
        self,
        query: str,
        thread_key: Optional[str],
        conversation_history: Optional[List[Dict]] = None,
        force_planner: bool = False,
    ) -> Dict[str, Any]:
        """Run staged advisory flow with web research + deterministic SEMA matching."""
        start_time = time.time()
        history = conversation_history or []

        recommendation_given = await self._get_recommendation_given(thread_key) if thread_key else False
        project_memory = await self.project_memory_store.latest_memory(thread_key) if thread_key else None
        # project_memory provides context only - don't let it skip the MCQ flow

        restart_flow = (
            recommendation_given
            and self._looks_like_new_project_start(query)
            and not self._looks_like_followup_query(query)
        )
        if restart_flow:
            print("[RAG] New project kickoff detected - restarting MCQ advisory flow")
            if thread_key:
                await self.reset_thread_state(thread_key)
            history = []
            recommendation_given = False
            project_memory = None

        # Pass full history — the AI decides dynamically what to ask and when to recommend.
        history_for_compound = history
        print(f"[RAG] Advisory mode, recommendation_given={recommendation_given}")

        compound_result = await self.compound_agent.process(
            user_query=query,
            conversation_history=history_for_compound,
            recommendation_given=recommendation_given,
            project_memory=project_memory,
        )
        if compound_result.is_followup:
            compound_result.response = self._remove_question_lines(compound_result.response)

        machine_rows: List[Dict[str, Any]] = []
        unresolved_categories: List[str] = []
        db_tools_used: List[str] = []
        verification_notes = ""
        combined_response = compound_result.response
        categories: List[str] = []
        project_plan: Optional[ProjectPlan] = None

        planning_constraints = self._build_planning_constraints(query=query, force_planner=force_planner)
        project_spec = self._extract_project_spec(
            query=query,
            conversation_history=history_for_compound,
            compound_response=compound_result.response,
            project_memory=project_memory,
        )

        if compound_result.needs_db_lookup:
            categories = self._normalize_categories(compound_result.suggested_categories or [])
            if not categories:
                categories = self._normalize_categories(
                    self.compound_agent._extract_equipment_categories(
                        " ".join(
                            m.get("content", "")
                            for m in history_for_compound + [{"role": "assistant", "content": compound_result.response}]
                        )
                    )
                )
            if not categories:
                context_text = " ".join(
                    [
                        query,
                        compound_result.response,
                        *[m.get("content", "") for m in history_for_compound],
                    ]
                )
                categories = self._infer_categories_from_context(context_text)

            print(f"[RAG] Final recommendation categories: {categories}")

            match_result = await self._resolve_categories_in_sema(
                categories=categories,
                project_context=compound_result.response,
                require_released=planning_constraints.require_released_only,
            )
            machine_rows = match_result["machine_rows"]
            unresolved_categories = match_result["unresolved_categories"]
            db_tools_used.extend(match_result["tools_used"])

            machine_names = [
                row.get("machine", {}).get("bezeichnung", "").strip()
                for row in machine_rows
                if row.get("machine", {}).get("bezeichnung")
            ]
            if machine_names:
                verification_notes = await self._verify_with_manuals(machine_names, compound_result.response)
                if verification_notes:
                    db_tools_used.append("manual_verification")

            phases = self._build_project_phases(
                categories=categories,
                machine_rows=machine_rows,
                project_spec=project_spec,
            )
            allocations = self._build_machine_allocations(machine_rows=machine_rows)
            decision_log = self._build_decision_log(
                constraints=planning_constraints,
                categories=categories,
                unresolved_categories=unresolved_categories,
                machine_rows=machine_rows,
            )
            next_actions = self._build_next_actions(
                unresolved_categories=unresolved_categories,
                machine_rows=machine_rows,
            )

            project_plan = ProjectPlan(
                query_text=query,
                summary=self._extract_web_research_summary(compound_result.response),
                spec=project_spec,
                constraints=planning_constraints,
                phases=phases,
                allocations=allocations,
                unresolved_gaps=unresolved_categories,
                verification_notes=verification_notes,
                next_actions=next_actions,
                decision_log=decision_log,
                full_recommendation=compound_result.response,
            )

            combined_response = self._build_final_project_response(project_plan)

            if thread_key:
                await self._set_recommendation_given(thread_key)
                memory_summary = self._build_project_memory_summary(
                    initial_query=query,
                    compound_response=compound_result.response,
                    machine_rows=machine_rows,
                    unresolved_categories=unresolved_categories,
                    verification_notes=verification_notes,
                )
                await self.project_memory_store.add_memory(
                    thread_key=thread_key,
                    summary=memory_summary,
                    machine_rows=machine_rows,
                    meta={
                        "categories": categories,
                        "unresolved_categories": unresolved_categories,
                        "stored_via": "expert_planner_final",
                        "planner_version": "v1",
                        "constraints": asdict(planning_constraints),
                        "project_spec": asdict(project_spec),
                        "phase_count": len(phases),
                        "allocation_count": len(allocations),
                        "plan": project_plan.to_dict() if project_plan else {},
                    },
                )

        execution_time = int((time.time() - start_time) * 1000)
        query_type = "expert_project_planner" if (self.enable_expert_planner or force_planner) else "compound_advisory"

        return {
            "response": self._strip_cost_info(combined_response),
            "sources": [],
            "chunks_used": 0,
            "response_id": None,
            "web_results_used": len(compound_result.executed_tools),
            "query_type": query_type,
            "agents_used": ["compound"] + compound_result.executed_tools + db_tools_used,
            "execution_time_ms": execution_time,
            "agent": "compound",
        }


    def _build_planning_constraints(self, query: str, force_planner: bool) -> PlanningConstraints:
        query_lower = (query or "").lower()
        english_hints = [
            "project",
            "build",
            "road",
            "machine",
            "inventory",
            "available",
        ]
        german_hints = [
            "projekt",
            "maschinen",
            "bestand",
            "verfuegbar",
            "strasse",
            "bau",
        ]

        english_score = sum(1 for hint in english_hints if hint in query_lower)
        german_score = sum(1 for hint in german_hints if hint in query_lower)
        language = "en" if english_score > german_score else "de"

        return PlanningConstraints(
            db_only=True,
            require_released_only=True,
            availability_first=True,
            allow_external_alternatives=False,
            language=language,
        )

    def _parse_decimal(self, text: str) -> Optional[float]:
        value = (text or "").strip().replace(",", ".")
        try:
            return float(value)
        except Exception:
            return None

    def _collect_user_inputs(
        self,
        query: str,
        conversation_history: List[Dict[str, Any]],
    ) -> List[str]:
        """Collect user-only text snippets ordered by recency (newest first)."""
        user_messages: List[str] = []
        for msg in conversation_history:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "user":
                continue
            content = (msg.get("content") or "").strip()
            if content:
                user_messages.append(content)

        ordered: List[str] = []
        current = (query or "").strip()
        if current:
            ordered.append(current)
        ordered.extend(reversed(user_messages))
        return ordered

    def _extract_mcq_answers(self, text: str) -> Dict[int, str]:
        """Extract A-D answer letters from compact user replies like 'B, A, B, D' or '1) B'."""
        if not text:
            return {}

        answers: Dict[int, str] = {}
        numbered = re.findall(r"(?:^|[\s,;])([1-4])\s*[\)\.\:\-]?\s*([A-Da-d])\b", text)
        for index, letter in numbered:
            answers[int(index)] = letter.upper()
        if answers:
            return answers

        plain_letters = re.findall(r"\b([A-Da-d])\b", text)
        if 2 <= len(plain_letters) <= 6:
            for index, letter in enumerate(plain_letters[:4], start=1):
                answers[index] = letter.upper()
        return answers

    def _first_match_in_texts(
        self,
        texts: List[str],
        pattern_map: List[Tuple[str, str]],
    ) -> str:
        for text in texts:
            lowered = text.lower()
            for pattern, value in pattern_map:
                if re.search(pattern, lowered):
                    return value
        return ""

    def _extract_project_spec(
        self,
        query: str,
        conversation_history: List[Dict[str, Any]],
        compound_response: str,
        project_memory: Optional[Dict[str, Any]],
    ) -> ProjectSpec:
        # Keep signature stable; project spec extraction is intentionally user-input driven.
        _ = compound_response
        _ = project_memory

        user_texts = self._collect_user_inputs(query=query, conversation_history=conversation_history)
        user_context = " ".join(user_texts).strip()
        user_context_lower = user_context.lower()

        round1_project_map = {"A": "Neubau", "B": "Sanierung", "C": "Erweiterung", "D": "Sonstiges"}
        round1_construction_map = {"A": "Asphalt", "B": "Pflaster", "C": "Schotter", "D": "Beton"}
        round1_ground_map = {
            "A": "Ungebunden / Erdreich",
            "B": "Bestehende Tragschicht",
            "C": "Hart / felsig",
            "D": "Noch unklar",
        }
        round1_load_map = {
            "A": "Leichte Last",
            "B": "Mittlere Last (PKW)",
            "C": "Schwere Last (LKW)",
            "D": "Gemischt / unklar",
        }
        round2_space_map = {
            "A": "Enge Platzverhaeltnisse",
            "B": "Mittlere Platzverhaeltnisse",
            "C": "Grosszuegige Platzverhaeltnisse",
            "D": "Noch unklar",
        }
        round2_drainage_map = {
            "A": "Laengs-/Quergefaelle vorhanden",
            "B": "Zusaetzliche Entwaesserung noetig",
            "C": "Noch offen",
            "D": "Nicht relevant",
        }
        round2_delivery_map = {"A": "Miete", "B": "Kauf", "C": "Gemischt", "D": "Noch offen"}
        round3_timeline_map = {
            "A": "Dringend (1-2 Wochen)",
            "B": "Mittelfristig (1-3 Monate)",
            "C": "Langfristig (3+ Monate)",
            "D": "Noch unklar",
        }
        round3_experience_map = {
            "A": "Sehr erfahren (Profis)",
            "B": "Grundkenntnisse vorhanden",
            "C": "Wenig Erfahrung / Einweisung noetig",
            "D": "Noch unklar",
        }
        round3_special_map = {
            "A": "Laermschutz / Emissionsauflagen",
            "B": "Transportbeschraenkungen (enge Zufahrt, Gewichtslimit)",
            "C": "Vorhandene Maschinen einbinden",
            "D": "Keine besonderen Anforderungen",
        }

        project_type_mcq = ""
        construction_method_mcq = ""
        ground_condition_mcq = ""
        load_profile_mcq = ""
        space_constraints_mcq = ""
        drainage_requirements_mcq = ""
        delivery_preference_mcq = ""
        timeline_mcq = ""
        experience_mcq = ""
        special_req_mcq = ""

        for text in user_texts:
            answers = self._extract_mcq_answers(text)
            if not answers:
                continue

            if len(answers) >= 4:
                project_type_mcq = project_type_mcq or round1_project_map.get(answers.get(1, ""), "")
                construction_method_mcq = construction_method_mcq or round1_construction_map.get(answers.get(2, ""), "")
                ground_condition_mcq = ground_condition_mcq or round1_ground_map.get(answers.get(3, ""), "")
                load_profile_mcq = load_profile_mcq or round1_load_map.get(answers.get(4, ""), "")
            elif len(answers) == 3:
                if not space_constraints_mcq:
                    space_constraints_mcq = round2_space_map.get(answers.get(1, ""), "")
                    drainage_requirements_mcq = round2_drainage_map.get(answers.get(2, ""), "")
                    delivery_preference_mcq = round2_delivery_map.get(answers.get(3, ""), "")
                else:
                    timeline_mcq = timeline_mcq or round3_timeline_map.get(answers.get(1, ""), "")
                    experience_mcq = experience_mcq or round3_experience_map.get(answers.get(2, ""), "")
                    special_req_mcq = special_req_mcq or round3_special_map.get(answers.get(3, ""), "")

        project_type = project_type_mcq or self._first_match_in_texts(
            user_texts,
            [
                (r"\bsanier\w*\b", "Sanierung"),
                (r"\berweiter\w*\b", "Erweiterung"),
                (r"\bneubau\b", "Neubau"),
                (r"\bsonstig\w*\b|\bother\b", "Sonstiges"),
                (r"\broad\b|\bstrasse\b|\bfahrbahn\b", "Strassenbau"),
            ],
        )

        construction_method = construction_method_mcq or self._first_match_in_texts(
            user_texts,
            [
                (r"\basphalt\w*\b", "Asphalt"),
                (r"\bbeton\w*\b", "Beton"),
                (r"\bpflaster\w*\b", "Pflaster"),
                (r"\bschotter\w*\b", "Schotter"),
            ],
        )

        length_m: Optional[float] = None
        for text in user_texts:
            lowered = text.lower()
            km_match = re.search(r"(\d+(?:[.,]\d+)?)\s*km\b", lowered)
            if km_match:
                parsed = self._parse_decimal(km_match.group(1))
                if parsed is not None:
                    length_m = parsed * 1000.0
                    break

            meter_match = re.search(r"(?:laenge|length|strecke)[^\d]{0,12}(\d+(?:[.,]\d+)?)\s*m\b", lowered)
            if meter_match:
                length_m = self._parse_decimal(meter_match.group(1))
                break

        width_m: Optional[float] = None
        for text in user_texts:
            lowered = text.lower()
            width_match = re.search(r"(?:breite|width|einbaubreite)[^\d]{0,12}(\d+(?:[.,]\d+)?)\s*m\b", lowered)
            if width_match:
                width_m = self._parse_decimal(width_match.group(1))
                if width_m is not None:
                    break

        # Fallback for compact user wording like "Asphalteinbau von 3,5m".
        if width_m is None:
            for text in user_texts:
                lowered = text.lower()
                has_width_context = bool(re.search(r"asphalt|einbau|fertiger|fahrbahn|spur", lowered))
                if not has_width_context:
                    continue
                for candidate in re.finditer(r"(\d+(?:[.,]\d+)?)\s*m\b", lowered):
                    parsed = self._parse_decimal(candidate.group(1))
                    if parsed is None:
                        continue
                    if 1.0 <= parsed <= 12.0:
                        width_m = parsed
                        break
                if width_m is not None:
                    break

        load_profile = load_profile_mcq or self._first_match_in_texts(
            user_texts,
            [
                (r"\bgemischt\b|\bmixed\b|\bgemischt\s*/\s*unklar\b", "Gemischt / unklar"),
                (r"\blkw\b|\bschwer\w*\s+last\b|\bheavy\b", "Schwere Last (LKW)"),
                (r"\bpkw\b|\bcar\b", "Mittlere Last (PKW)"),
                (r"\bfussgaenger\b|\bfahrrad\b|\bpedestrian\b|\bcycle\b|\bleichte?\s+last\b", "Leichte Last"),
            ],
        )

        ground_condition = ground_condition_mcq or self._first_match_in_texts(
            user_texts,
            [
                (r"\btragschicht\b|\bunterbau\b|\bbase\s*layer\b", "Bestehende Tragschicht"),
                (r"\bungebunden\b|\berdreich\b|\bsoil\b", "Ungebunden / Erdreich"),
                (r"\bfels\w*\b|\brock\b|\bhart\b", "Hart / felsig"),
                (r"\bnoch\s*unklar\b", "Noch unklar"),
            ],
        )

        space_constraints = space_constraints_mcq or self._first_match_in_texts(
            user_texts,
            [
                (r"\bsehr\s*eng\w*\b|\beng\w*\b|\bnarrow\b|\bplatzmangel\b", "Enge Platzverhaeltnisse"),
                (r"\bplatz\w*[^.]{0,20}\bmittel\b|\bmittel\b(?!\s*last)", "Mittlere Platzverhaeltnisse"),
                (r"\bgrosszuegig\b|\bgro[ss]z[ue]gig\b|\bweitlaeufig\b|\bopen\s*site\b", "Grosszuegige Platzverhaeltnisse"),
                (r"\bplatz\w*[^.]{0,24}\bunklar\b|\bengstellen?\s+unklar\b", "Noch unklar"),
            ],
        )

        drainage_requirements = drainage_requirements_mcq or self._first_match_in_texts(
            user_texts,
            [
                (r"\bnicht\s+relevant\b", "Nicht relevant"),
                (r"\bentwaesser\w*\b|\bdraenage\b|\bdrainage\b", "Entwaesserung relevant"),
                (r"\bgefaelle\b|\bslope\b", "Gefaelleprofil relevant"),
                (r"\bentwaesser\w*[^.]{0,24}\boffen\b|\bdraenage[^.]{0,24}\boffen\b", "Noch offen"),
            ],
        )

        delivery_preference = delivery_preference_mcq or self._first_match_in_texts(
            user_texts,
            [
                (r"(?:bereitstellung|beschaffung|miete|kauf)[^.]{0,24}\bgemischt\b", "Gemischt"),
                (r"\bmiete?\b|\brental\b", "Miete"),
                (r"\bkauf\b|\bpurchase\b|\bbuy\b", "Kauf"),
                (r"(?:bereitstellung|beschaffung)[^.]{0,24}\boffen\b", "Noch offen"),
            ],
        )

        special_constraints: List[str] = []
        if timeline_mcq and timeline_mcq != "Noch unklar":
            special_constraints.append(f"Zeitrahmen: {timeline_mcq}")
        if experience_mcq and experience_mcq != "Noch unklar":
            special_constraints.append(f"Bediener: {experience_mcq}")
        if special_req_mcq and special_req_mcq != "Keine besonderen Anforderungen":
            special_constraints.append(special_req_mcq)
        if re.search(r"\bnacht\w*\b|\bnight\b", user_context_lower):
            special_constraints.append("Nachtbetrieb")
        if re.search(r"\bverkehr\b|\btraffic\b", user_context_lower):
            special_constraints.append("Verkehr unter Betrieb")
        if re.search(r"\bhang\b|\bslope\b", user_context_lower):
            special_constraints.append("Hang-/Neigungsverhaeltnisse")

        assumptions: List[str] = []
        open_questions: List[str] = []

        if not project_type:
            assumptions.append("Projektart wird vorlaeufig als allgemeiner Tief-/Strassenbau behandelt.")
            open_questions.append("Projektart final bestaetigen (Neubau/Sanierung/Erweiterung).")

        if not construction_method:
            assumptions.append("Bauweise wurde als Standard-Asphalt angenommen.")
            open_questions.append("Bauweise bestaetigen (Asphalt/Beton/Pflaster/Schotter).")

        if length_m is None:
            assumptions.append("Laenge ist unklar; Maschinenabfolge wird ohne Mengenkalkulation geplant.")
            open_questions.append("Projektlaenge angeben.")

        if not load_profile:
            assumptions.append("Belastung wurde konservativ als gemischte Nutzung bewertet.")
            open_questions.append("Belastungsprofil spezifizieren (PKW/LKW/Leichtverkehr).")

        return ProjectSpec(
            project_type=project_type,
            construction_method=construction_method,
            length_m=length_m,
            width_m=width_m,
            load_profile=load_profile,
            ground_condition=ground_condition,
            space_constraints=space_constraints,
            drainage_requirements=drainage_requirements,
            delivery_preference=delivery_preference,
            special_constraints=special_constraints,
            assumptions=assumptions,
            open_questions=open_questions,
        )

    def _phase_template_for_category(self, category: str) -> Tuple[str, str, str]:
        category_lower = (category or "").lower()
        if "fraese" in category_lower or "frase" in category_lower:
            return ("Bestandsabtrag", "Vorhandene Schichten abtragen und Profil herstellen.", "Abtragstiefe und Ebenheit pruefen.")
        if "bagger" in category_lower or "raupe" in category_lower:
            return ("Untergrundvorbereitung", "Aushub, Planum und Untergrundprofil herstellen.", "Tragfaehigkeit und Planumhoehen pruefen.")
        if "radlader" in category_lower or "dumper" in category_lower:
            return ("Materiallogistik", "Materialumschlag und Zulieferung sicherstellen.", "Materialfluss ohne Engpass pruefen.")
        if "fertiger" in category_lower:
            return ("Materialeinbau", "Deck- oder Tragschicht mit Sollgeometrie einbauen.", "Einbaubreite, Temperatur und Querneigung pruefen.")
        if "walze" in category_lower or "ruettel" in category_lower:
            return ("Verdichtung", "Geforderte Verdichtung und Oberflaechenqualitaet herstellen.", "Verdichtungsgrad und Oberflaechenbild pruefen.")
        return ("Ausfuehrung", "Arbeitsschritt gemaess Projektbedarf ausfuehren.", "Arbeitsergebnis vor Freigabe pruefen.")

    def _build_project_phases(
        self,
        categories: List[str],
        machine_rows: List[Dict[str, Any]],
        project_spec: ProjectSpec,
    ) -> List[ProjectPhase]:
        phase_map: Dict[str, ProjectPhase] = {}

        source_categories = categories or [
            row.get("requested_category", "")
            for row in machine_rows
            if row.get("requested_category")
        ]

        for category in source_categories:
            phase_name, objective, quality_check = self._phase_template_for_category(category)
            if phase_name not in phase_map:
                phase_map[phase_name] = ProjectPhase(
                    name=phase_name,
                    objective=objective,
                    categories_needed=[],
                    quality_check=quality_check,
                    dependency="",
                )
            if category not in phase_map[phase_name].categories_needed:
                phase_map[phase_name].categories_needed.append(category)

        if not phase_map:
            default_objective = "Grundplanung fuer das Projekt aus internen Daten erstellen."
            if project_spec.project_type:
                default_objective = f"{project_spec.project_type}: Ausfuehrungsplanung mit Bestandsgleichlauf erstellen."
            phase_map["Projektplanung"] = ProjectPhase(
                name="Projektplanung",
                objective=default_objective,
                categories_needed=[],
                quality_check="Projektannahmen vor Ausfuehrung bestaetigen.",
                dependency="",
            )

        preferred_order = [
            "Untergrundvorbereitung",
            "Bestandsabtrag",
            "Materiallogistik",
            "Materialeinbau",
            "Verdichtung",
            "Ausfuehrung",
            "Projektplanung",
        ]

        ordered_names = sorted(
            phase_map.keys(),
            key=lambda name: preferred_order.index(name) if name in preferred_order else len(preferred_order),
        )

        phases: List[ProjectPhase] = []
        previous_name = ""
        for name in ordered_names:
            phase = phase_map[name]
            phase.dependency = previous_name
            phases.append(phase)
            previous_name = name

        return phases

    def _build_machine_allocations(self, machine_rows: List[Dict[str, Any]]) -> List[MachineAllocation]:
        allocations: List[MachineAllocation] = []
        for row in machine_rows:
            requested_category = row.get("requested_category", "-")
            resolved_category = row.get("resolved_category", requested_category)
            phase_name, _, _ = self._phase_template_for_category(requested_category)
            primary_machine = row.get("machine") or {}
            backup_machine = row.get("backup_machine") or {}
            fallback_plan = row.get("fallback_plan") or ""
            if not fallback_plan:
                if backup_machine.get("bezeichnung"):
                    fallback_plan = f"Backup einsetzen: {backup_machine.get('bezeichnung')}."
                else:
                    fallback_plan = "Bei Ausfall: Kategorie sofort neu disponieren und Kapazitaet anpassen."

            allocations.append(
                MachineAllocation(
                    requested_category=requested_category,
                    resolved_category=resolved_category,
                    phase_name=phase_name,
                    purpose=self._goal_for_category(requested_category),
                    primary_machine=primary_machine,
                    backup_machine=backup_machine,
                    selection_reason=row.get("selection_reason")
                    or "Auswahl nach Released-Bestandsstatus und Kategorienpassung.",
                    constraint_check=row.get("constraint_check")
                    or "Geprueft: Released-Status als Bestandsstatus, Kategorienfit, eindeutige Zuordnung.",
                    fallback_plan=fallback_plan,
                    is_alternative=bool(row.get("is_alternative")),
                )
            )
        return allocations

    def _build_decision_log(
        self,
        constraints: PlanningConstraints,
        categories: List[str],
        unresolved_categories: List[str],
        machine_rows: List[Dict[str, Any]],
    ) -> List[str]:
        log: List[str] = []
        if constraints.require_released_only:
            log.append("Nur Maschinen mit Status 'Released' (Bestandsstatus, nicht Live-Verfuegbarkeit) wurden beruecksichtigt.")
        if constraints.availability_first:
            log.append("Priorisierung: Released-Bestandsstatus vor Feintuning, um Auswahlrisiken zu minimieren.")
        if categories:
            log.append(f"Abgeleitete Maschinenklassen: {', '.join(categories)}.")
        if machine_rows:
            log.append(f"Bestandszuordnung abgeschlossen fuer {len(machine_rows)} Maschinenpositionen.")
        if unresolved_categories:
            log.append(f"Offene Bedarfe ohne Released-Bestandsmatch: {', '.join(unresolved_categories)}.")
        return log

    def _build_next_actions(self, unresolved_categories: List[str], machine_rows: List[Dict[str, Any]]) -> List[str]:
        actions: List[str] = []
        if machine_rows:
            actions.append("Die echte Dispositionsverfuegbarkeit der zugeordneten Serien-/Inventarnummern mit der Disposition final bestaetigen.")
            actions.append("Einsatzfenster je Phase terminlich abstimmen und Uebergaben definieren.")
        if unresolved_categories:
            actions.append(
                "Offene Maschinenklassen mit Bauleitung klaeren und Alternativumfang im bestehenden Bestand abstimmen."
            )
        actions.append("Vor Baustart einen kurzen Realitaetscheck der Annahmen (Untergrund, Lastprofil, Bauweise) durchfuehren.")
        return actions

    def _normalize_categories(self, categories: List[str]) -> List[str]:
        cleaned: List[str] = []
        for raw in categories:
            cat = re.sub(r"\s+", " ", str(raw or "").strip())
            if not cat:
                continue
            if cat not in cleaned:
                cleaned.append(cat)
        return cleaned[:6]

    def _infer_categories_from_context(self, text: str) -> List[str]:
        """Fallback category inference when model output misses machine classes."""
        context = (text or "").lower()
        inferred: List[str] = []

        rules = [
            (["strasse", "asphalt", "pflaster", "wegbau", "fahrbahn"], ["Walze", "Radlader", "Raupe"]),
            (["aushub", "erdarbeiten", "graben", "fundament"], ["Bagger", "Radlader", "Walze"]),
            (["abbruch", "rueckbau"], ["Bagger", "Radlader", "Brecher"]),
            (["verdicht", "tragschicht"], ["Walze", "Ruettelplatte"]),
            (["fraes", "frase"], ["Fraese", "Walze"]),
        ]

        for keywords, categories in rules:
            if any(keyword in context for keyword in keywords):
                for category in categories:
                    if category not in inferred:
                        inferred.append(category)

        if not inferred:
            inferred = ["Walze", "Radlader"]

        return inferred[:6]

    def _get_postgres_service(self):
        if self._postgres_service is False:
            return None
        if self._postgres_service is not None:
            return self._postgres_service
        try:
            from .postgres import PostgresService
            service = PostgresService()
            if getattr(service, "available", False):
                self._postgres_service = service
                return service
            self._postgres_service = False
            print(f"[RAG] Postgres unavailable for SEMA lookup: {service.availability_error}")
            return None
        except Exception as e:
            print(f"[RAG] Could not initialize PostgresService: {e}")
            self._postgres_service = False
            return None

    def _sanitize_sql_text(self, value: str, max_len: int = 80) -> str:
        value = (
            (value or "")
            .replace("\u00e4", "ae")
            .replace("\u00f6", "oe")
            .replace("\u00fc", "ue")
            .replace("\u00c4", "Ae")
            .replace("\u00d6", "Oe")
            .replace("\u00dc", "Ue")
            .replace("\u00df", "ss")
        )
        safe = re.sub(r"[^A-Za-z0-9\-\s()/]", "", value)
        safe = safe.replace("'", "''").strip()
        return safe[:max_len]
    def _query_sema_category(
        self,
        category: str,
        limit: int = 1,
        only_released: bool = False,
    ) -> List[Dict[str, Any]]:
        postgres = self._get_postgres_service()
        if not postgres:
            return []

        safe_category = self._sanitize_sql_text(category)
        if not safe_category:
            return []

        table_name = postgres.equipment_table
        limit = max(1, min(limit, 5))
        filters = [f"geraetegruppe_name ILIKE '%{safe_category}%'"]
        if only_released:
            filters.append("nuclos_state = 'Released'")
        where_sql = " AND ".join(filters)
        sql = f"""
            SELECT
                id,
                bezeichnung,
                hersteller_name,
                geraetegruppe_name,
                seriennummer,
                inventarnummer,
                verwendung_code,
                nuclos_state
            FROM {table_name}
            WHERE {where_sql}
            ORDER BY
                CASE WHEN nuclos_state = 'Released' THEN 0 ELSE 1 END,
                CASE WHEN verwendung_code = 'MIET' THEN 0 ELSE 1 END,
                id
            LIMIT {limit}
        """
        try:
            return postgres.execute_query(sql)
        except Exception as e:
            print(f"[RAG] SEMA query failed for category '{category}': {e}")
            return []

    def _list_available_db_categories(self) -> List[str]:
        postgres = self._get_postgres_service()
        if not postgres:
            return []

        sql = f"""
            SELECT DISTINCT geraetegruppe_name
            FROM {postgres.equipment_table}
            WHERE geraetegruppe_name IS NOT NULL
            ORDER BY geraetegruppe_name
            LIMIT 250
        """
        try:
            rows = postgres.execute_query(sql)
            categories = []
            for row in rows:
                cat = (row.get("geraetegruppe_name") or "").strip()
                if cat and cat not in categories:
                    categories.append(cat)
            return categories
        except Exception as e:
            print(f"[RAG] Could not load DB categories: {e}")
            return []

    def _build_resolution_row(
        self,
        requested_category: str,
        resolved_category: str,
        machine: Dict[str, Any],
        is_alternative: bool,
        backup_machine: Optional[Dict[str, Any]] = None,
        selection_reason: str = "",
        constraint_check: str = "",
        fallback_plan: str = "",
    ) -> Dict[str, Any]:
        backup_payload = {
            "bezeichnung": (backup_machine or {}).get("bezeichnung", ""),
            "hersteller_name": (backup_machine or {}).get("hersteller_name", ""),
            "geraetegruppe_name": (backup_machine or {}).get("geraetegruppe_name", ""),
            "seriennummer": (backup_machine or {}).get("seriennummer", ""),
            "inventarnummer": (backup_machine or {}).get("inventarnummer", ""),
            "verwendung_code": (backup_machine or {}).get("verwendung_code", ""),
            "nuclos_state": (backup_machine or {}).get("nuclos_state", ""),
        }
        return {
            "requested_category": requested_category,
            "resolved_category": resolved_category,
            "is_alternative": is_alternative,
            "selection_reason": selection_reason,
            "constraint_check": constraint_check,
            "fallback_plan": fallback_plan,
            "machine": {
                "bezeichnung": machine.get("bezeichnung", ""),
                "hersteller_name": machine.get("hersteller_name", ""),
                "geraetegruppe_name": machine.get("geraetegruppe_name", ""),
                "seriennummer": machine.get("seriennummer", ""),
                "inventarnummer": machine.get("inventarnummer", ""),
                "verwendung_code": machine.get("verwendung_code", ""),
                "nuclos_state": machine.get("nuclos_state", ""),
            },
            "backup_machine": backup_payload,
        }

    def _machine_identity(self, machine: Dict[str, Any]) -> str:
        machine_id = machine.get("id")
        if machine_id is not None:
            return f"id:{machine_id}"
        inventory = (machine.get("inventarnummer") or "").strip()
        if inventory:
            return f"inv:{inventory}"
        serial = (machine.get("seriennummer") or "").strip()
        if serial:
            return f"sn:{serial}"
        model = (machine.get("bezeichnung") or "").strip()
        return f"model:{model}" if model else "unknown"

    def _pick_primary_and_backup(
        self,
        candidates: List[Dict[str, Any]],
        used_machine_ids: set,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if not candidates:
            return None, None

        primary: Optional[Dict[str, Any]] = None
        backup: Optional[Dict[str, Any]] = None

        for candidate in candidates:
            key = self._machine_identity(candidate)
            if key not in used_machine_ids:
                primary = candidate
                used_machine_ids.add(key)
                break

        if primary is None:
            primary = candidates[0]
            used_machine_ids.add(self._machine_identity(primary))

        primary_key = self._machine_identity(primary)
        for candidate in candidates:
            key = self._machine_identity(candidate)
            if key != primary_key:
                backup = candidate
                break

        return primary, backup

    async def _resolve_categories_in_sema(
        self,
        categories: List[str],
        project_context: str,
        require_released: bool = True,
    ) -> Dict[str, Any]:
        machine_rows: List[Dict[str, Any]] = []
        unresolved: List[str] = []
        tools_used: List[str] = ["sema_primary_lookup"]
        unresolved_reasons: Dict[str, str] = {}
        used_machine_ids: set = set()

        for category in categories:
            matches = self._query_sema_category(
                category,
                limit=5,
                only_released=require_released,
            )
            primary, backup = self._pick_primary_and_backup(matches, used_machine_ids)
            if primary:
                backup_name = (backup or {}).get("bezeichnung", "")
                fallback_plan = (
                    f"Backup innerhalb der Kategorie: {backup_name}."
                    if backup_name
                    else "Keine zweite Released-Option in derselben Kategorie; bei Bedarf neu disponieren."
                )
                machine_rows.append(
                    self._build_resolution_row(
                        requested_category=category,
                        resolved_category=category,
                        machine=primary,
                        is_alternative=False,
                        backup_machine=backup,
                        selection_reason="Direkter Kategorienmatch mit priorisiertem Released-Bestandsstatus.",
                        constraint_check="Geprueft: Released-Status als Bestandsstatus, Kategorie, eindeutige Maschinenzuordnung.",
                        fallback_plan=fallback_plan,
                    )
                )
            else:
                unresolved.append(category)
                if require_released:
                    blocked = self._query_sema_category(category, limit=1, only_released=False)
                    if blocked:
                        unresolved_reasons[category] = "kein Released-Bestandsstatus im aktuellen Bestand"

        if unresolved and self.compound_agent:
            available_categories = self._list_available_db_categories()
            if available_categories:
                alt_map = await self.compound_agent.suggest_alternative_categories(
                    missing_categories=unresolved,
                    project_context=project_context,
                    available_categories=available_categories,
                )
                tools_used.append("compound_alternative_research")
            else:
                alt_map = {}

            still_unresolved: List[str] = []
            for missing_category in unresolved:
                matched = False
                for alt_category in alt_map.get(missing_category, []):
                    alt_rows = self._query_sema_category(
                        alt_category,
                        limit=5,
                        only_released=require_released,
                    )
                    primary, backup = self._pick_primary_and_backup(alt_rows, used_machine_ids)
                    if not primary:
                        continue

                    backup_name = (backup or {}).get("bezeichnung", "")
                    fallback_plan = (
                        f"Alternative Backup-Maschine: {backup_name}."
                        if backup_name
                        else "Keine zweite Released-Alternative; Disposition fuer Ersatzklasse erforderlich."
                    )
                    machine_rows.append(
                        self._build_resolution_row(
                            requested_category=missing_category,
                            resolved_category=alt_category,
                            machine=primary,
                            is_alternative=True,
                            backup_machine=backup,
                            selection_reason="Alternative Kategorienzuordnung aus SEMA-Bestand mit Released-Status.",
                            constraint_check="Geprueft: Released-Status als Bestandsstatus, Ersatzkategorie, eindeutige Maschinenzuordnung.",
                            fallback_plan=fallback_plan,
                        )
                    )
                    matched = True
                    break
                if not matched:
                    still_unresolved.append(missing_category)

            unresolved = still_unresolved
            tools_used.append("sema_alternative_lookup")

        unresolved_with_reason = []
        for category in unresolved:
            reason = unresolved_reasons.get(category, "")
            if reason:
                unresolved_with_reason.append(f"{category} ({reason})")
            else:
                unresolved_with_reason.append(category)

        return {
            "machine_rows": machine_rows,
            "unresolved_categories": unresolved_with_reason,
            "tools_used": tools_used,
        }

    def _format_table_cell(self, value: Any) -> str:
        text = str(value or "-").replace("\n", " ").replace("|", "/").strip()
        return text if text else "-"

    def _format_usage(self, code: str) -> str:
        if code == "MIET":
            return "Miete"
        if code == "VK":
            return "Verkauf"
        return code or "-"

    def _format_availability(self, state: str) -> str:
        if state == "Released":
            return "Im Bestand (Released)"
        if state == "Locked":
            return "Gesperrt (Locked)"
        return state or "-"

    def _build_project_machine_table(self, allocations: List[MachineAllocation]) -> str:
        if not allocations:
            return "Keine passende Released-Bestandszuordnung gefunden."

        rows: List[str] = [
            "| # | Maschinenklasse | Maschine | Hersteller | Serien-Nr. | Nutzung | Status |",
            "|---|---|---|---|---|---|---|",
        ]

        for idx, allocation in enumerate(allocations, start=1):
            primary = allocation.primary_machine or {}
            alt_note = " (Alternative)" if allocation.is_alternative else ""
            rows.append(
                "| "
                f"{idx} | "
                f"{self._format_table_cell(allocation.requested_category)}{alt_note} | "
                f"{self._format_table_cell(primary.get('bezeichnung'))} | "
                f"{self._format_table_cell(primary.get('hersteller_name'))} | "
                f"{self._format_table_cell(primary.get('seriennummer'))} | "
                f"{self._format_table_cell(self._format_usage(primary.get('verwendung_code', '')))} | "
                f"{self._format_table_cell(self._format_availability(primary.get('nuclos_state', '')))} |"
            )

        return "\n".join(rows)

    def _goal_for_category(self, category: str) -> str:
        category_lower = (category or "").lower()
        goals = {
            "bagger": "Aushub und Materialabtrag",
            "fertiger": "Einbau oder Verteilung des Oberbaus",
            "fraese": "Abtrag bestehender Schichten",
            "frase": "Abtrag bestehender Schichten",
            "walze": "Verdichtung",
            "radlader": "Materialtransport und Verladung",
            "raupe": "Profilierung und Planum",
            "dumper": "Transport auf der Baustelle",
        }
        for key, goal in goals.items():
            if key in category_lower:
                return goal
        return "Arbeitsschritt laut Projektbedarf"

    def _build_project_workflow_table(
        self,
        phases: List[ProjectPhase],
        allocations: List[MachineAllocation],
    ) -> str:
        if not phases:
            return "Kein ausfuehrbarer Phasenablauf verfuegbar."

        rows: List[str] = [
            "| Schritt | Ziel | Maschinenklassen | Zuordnung | Qualitaetscheck | Abhaengigkeit |",
            "|---|---|---|---|---|---|",
        ]
        for idx, phase in enumerate(phases, start=1):
            phase_allocations = [a for a in allocations if a.phase_name == phase.name]
            machine_list = ", ".join(
                self._format_table_cell(a.primary_machine.get("bezeichnung")) for a in phase_allocations
            )
            category_list = ", ".join(phase.categories_needed) if phase.categories_needed else "-"
            dependency = phase.dependency or "Start"
            rows.append(
                "| "
                f"{idx} - {self._format_table_cell(phase.name)} | "
                f"{self._format_table_cell(phase.objective)} | "
                f"{self._format_table_cell(category_list)} | "
                f"{self._format_table_cell(machine_list or '-')} | "
                f"{self._format_table_cell(phase.quality_check)} | "
                f"{self._format_table_cell(dependency)} |"
            )
        return "\n".join(rows)

    def _format_project_dimensions(self, project_spec: ProjectSpec) -> str:
        parts: List[str] = []
        if project_spec.length_m is not None:
            parts.append(f"Laenge: {project_spec.length_m:.0f} m")
        if project_spec.width_m is not None:
            parts.append(f"Breite: {project_spec.width_m:.1f} m")
        return ", ".join(parts) if parts else "Abmessungen nicht eindeutig angegeben"

    def _build_project_spec_section(
        self,
        project_spec: ProjectSpec,
        constraints: PlanningConstraints,
    ) -> str:
        rows = [
            "| Kriterium | Wert |",
            "|---|---|",
            f"| Projektart | {self._format_table_cell(project_spec.project_type or 'Noch offen')} |",
            f"| Bauweise | {self._format_table_cell(project_spec.construction_method or 'Noch offen')} |",
            f"| Abmessungen | {self._format_table_cell(self._format_project_dimensions(project_spec))} |",
            f"| Lastprofil | {self._format_table_cell(project_spec.load_profile or 'Noch offen')} |",
            f"| Untergrund | {self._format_table_cell(project_spec.ground_condition or 'Noch offen')} |",
            f"| Platzverhaeltnisse | {self._format_table_cell(project_spec.space_constraints or 'Noch offen')} |",
            f"| Entwaesserung/Gefaelle | {self._format_table_cell(project_spec.drainage_requirements or 'Noch offen')} |",
            f"| Bereitstellung | {self._format_table_cell(project_spec.delivery_preference or 'Noch offen')} |",
        ]

        if project_spec.special_constraints:
            rows.append(
                f"| Besondere Randbedingungen | {self._format_table_cell(', '.join(project_spec.special_constraints))} |"
            )

        if project_spec.assumptions:
            rows.append("")
            rows.append("Getroffene Annahmen (nur fuer offene Punkte):")
            rows.extend(f"- {assumption}" for assumption in project_spec.assumptions[:3])

        return "\n".join(rows)

    def _extract_web_research_summary(self, compound_response: str) -> str:
        text = self._ascii_safe_text(compound_response or "", collapse_whitespace=False)
        text = re.sub(r"(?i)\[empfehlung_bereit\]", " ", text)

        lines: List[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith("---"):
                continue
            if line.count("|") >= 2:
                continue
            if re.match(r"^\d+[\)\.]\s+", line):
                continue
            lowered = line.lower()
            if lowered.startswith(("benoetigte", "empfohlener", "hinweis")):
                continue
            lines.append(line)
        summary = " ".join(lines)
        summary = re.sub(r"(?i)kurzfazit\s*:\s*", "", summary)
        summary = re.sub(r"\*+", "", summary)
        summary = self._ascii_safe_text(summary, collapse_whitespace=True)
        if summary:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]
            if sentences:
                summary = " ".join(sentences[:2])
        summary = self._compact_text(summary, 320)
        return summary if summary else "Web-Recherche und Validierung wurden durchgefuehrt."

    def _build_final_project_response(self, project_plan: ProjectPlan) -> str:
        sections = []

        # Show the full AI expert recommendation first.
        if project_plan.full_recommendation:
            sections.append(project_plan.full_recommendation)
        elif project_plan.summary:
            sections.append(project_plan.summary)

        # DB machine lookup results.
        if project_plan.allocations:
            sections.append("---\n**Verfügbare Maschinen im SEMA-Bestand:**")
            sections.append(self._build_project_machine_table(project_plan.allocations))
        else:
            sections.append("---\n**Hinweis:** Für die empfohlenen Maschinenklassen wurden keine freigegebenen Maschinen im aktuellen Bestand gefunden.")

        if project_plan.unresolved_gaps:
            missing = ", ".join(project_plan.unresolved_gaps)
            sections.append(f"**Nicht im Bestand vorhanden:** {missing}")

        if project_plan.next_actions:
            sections.append("**Nächste Schritte:**")
            sections.append("\n".join(f"- {action}" for action in project_plan.next_actions))

        return "\n\n".join(sections).strip()
    def _build_project_memory_summary(
        self,
        initial_query: str,
        compound_response: str,
        machine_rows: List[Dict[str, Any]],
        unresolved_categories: List[str],
        verification_notes: str,
    ) -> str:
        summary_lines = [
            f"Projektanfrage: {self._compact_text(initial_query, 220)}",
            f"Fachliche Empfehlung: {self._compact_text(compound_response, 700)}",
        ]

        if machine_rows:
            machine_notes = []
            for row in machine_rows[:6]:
                machine = row.get("machine", {})
                req = row.get("requested_category", "-")
                model = machine.get("bezeichnung", "-")
                serial = machine.get("seriennummer", "-")
                machine_notes.append(f"{req}: {model} (SN {serial})")
            summary_lines.append("SEMA-Zuordnung: " + "; ".join(machine_notes))

        if unresolved_categories:
            summary_lines.append("Offene Kategorien: " + ", ".join(unresolved_categories))

        if verification_notes:
            summary_lines.append("Validierung: " + self._compact_text(verification_notes, 280))

        return "\n".join(summary_lines)

    def _compact_text(self, text: str, max_chars: int) -> str:
        compact = re.sub(r"\s+", " ", text or "").strip()
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3].rstrip() + "..."

    def _ascii_safe_text(self, text: str, collapse_whitespace: bool = False) -> str:
        normalized = (
            str(text or "")
            .replace("\u2011", "-")
            .replace("\u2013", "-")
            .replace("\u2014", "-")
            .replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2026", "...")
            .replace("\u00a0", " ")
            .replace("\u202f", " ")
        )
        normalized = (
            normalized
            .replace("\u00e4", "ae")
            .replace("\u00f6", "oe")
            .replace("\u00fc", "ue")
            .replace("\u00c4", "Ae")
            .replace("\u00d6", "Oe")
            .replace("\u00dc", "Ue")
            .replace("\u00df", "ss")
        )
        normalized = normalized.encode("ascii", "ignore").decode("ascii")
        if collapse_whitespace:
            normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _remove_question_lines(self, text: str) -> str:
        filtered = []
        for line in (text or "").splitlines():
            if "?" in line:
                continue
            filtered.append(line)
        cleaned = "\n".join(filtered).strip()
        return cleaned or (text or "").strip()

    def _extract_machine_names(self, db_response: str) -> List[str]:
        """Extract machine names/models from DB response for manual verification."""
        names = []
        # Match bold machine names like **Super 1800-3** or **BW 213 D-5**
        bold_matches = re.findall(r'\*\*([A-Z][A-Za-z0-9\s\-\.]+)\*\*', db_response)
        for name in bold_matches:
            name = name.strip()
            # Filter out section headers (keep only machine-looking names)
            if len(name) < 40 and any(c.isdigit() for c in name):
                names.append(name)

        # Also match "bezeichnung: XYZ" patterns
        bez_matches = re.findall(r'[Bb]ezeichnung[:\s]+([A-Z][A-Za-z0-9\s\-\.]+)', db_response)
        for name in bez_matches:
            name = name.strip()
            if name not in names:
                names.append(name)

        print(f"[RAG] Extracted {len(names)} machine names from DB response: {names}")
        return names[:5]  # Max 5 machines to verify

    async def _verify_with_manuals(
        self, machine_names: List[str], project_requirements: str
    ) -> str:
        """Verify recommended machines against German manuals in Pinecone."""
        verified = []

        for name in machine_names:
            try:
                manual_results = await self.vector_store.search(
                    query=f"{name} technische Daten Spezifikationen Anleitung",
                    top_k=3,
                )

                if not manual_results:
                    print(f"[RAG] No manual found for: {name}")
                    continue

                best_score = max(r.get("score", 0) for r in manual_results)
                best_title = manual_results[0].get("metadata", {}).get("title", "")
                safe_title = self._ascii_safe_text(best_title, collapse_whitespace=True)[:160] or "unbekannt"

                if best_score >= 0.5:
                    verified.append(name)
                    print(f"[RAG] Manual verified: {name} (score: {best_score:.2f}, source: {safe_title})")
                else:
                    print(f"[RAG] Manual match too weak for: {name} (score: {best_score:.2f})")
            except Exception as e:
                print(f"[RAG] Manual verification error for {name}: {e}")

        if verified:
            return (
                "Die Spezifikationen folgender Maschinen wurden gegen die "
                f"technischen Handbuecher geprueft: {', '.join(verified)}."
            )

        return ""

    def _strip_cost_info(self, text: str) -> str:
        """Remove any cost/price/budget mentions from text."""
        result = text
        cost_patterns = [
            r'.*(?:kosten|preis|budget|euro|eur|preislich|preisrahmen|kostenschaetzung|kostenabschaetzung|kostenschatzung|kostenabschatzung).*\n?',
        ]
        for pattern in cost_patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)
        return result.strip()

    def _extract_generated_text(self, response: Any) -> str:
        """Safely extract plain text from a Gemini response."""
        try:
            text = response.text or ""
            if text:
                return self._strip_cost_info(text)
        except Exception:
            pass

        try:
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    text = "".join(
                        part.text for part in candidate.content.parts if hasattr(part, "text") and part.text
                    )
                    return self._strip_cost_info(text)
        except Exception:
            pass

        return ""

    async def search_and_generate(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
        system_instructions: Optional[str] = None,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        thread_key: Optional[str] = None,
        force_planner: bool = False,
        planner_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point for advisory and retrieval queries.

        Routing order:
        1. Advisory flow via Gemini compound agent.
        2. LangGraph DB retrieval.
        3. Gemini fallback grounded in Pinecone.
        """
        top_k = top_k or config.search_top_k

        query, command_forced_planner = self._extract_planner_command(query)
        force_planner = force_planner or command_forced_planner or (planner_mode or "").lower() in {
            "expert",
            "forced",
            "project",
        }

        if force_planner and not query.strip():
            return {
                "response": (
                    "Bitte beschreiben Sie Ihr Projekt direkt nach `/plan`, z. B.:\n"
                    "`/plan Wir bauen 2 km Asphaltstrasse fuer LKW-Verkehr auf bestehender Tragschicht.`"
                ),
                "sources": [],
                "chunks_used": 0,
                "response_id": None,
                "web_results_used": 0,
                "query_type": "expert_project_planner",
                "agents_used": ["planner_input_guard"],
                "execution_time_ms": 0,
                "agent": "compound",
            }

        if not force_planner:
            property_machine_result = await self._try_direct_machine_property_response(query, thread_key)
            if property_machine_result:
                await self._store_conversation_turn(thread_key, query, property_machine_result["response"])
                return property_machine_result

            raw_machine_result = self._try_raw_machine_dump(query)
            if raw_machine_result:
                await self._store_conversation_turn(thread_key, query, raw_machine_result["response"])
                return raw_machine_result

        # Priority 0: Advisory queries or advisory follow-up sessions.
        is_advisory_session = False
        if thread_key:
            self._prune_local_thread_state(thread_key)
            has_recommendation = await self._get_recommendation_given(thread_key)
            has_project_memory = bool(await self.project_memory_store.latest_memory(thread_key))
            is_advisory_session = (
                self._is_recent_local_advisory_thread(thread_key)
                or has_recommendation
                or has_project_memory
            )
        is_advisory_query = self._is_advisory_query(query) if self.compound_agent else False
        explicit_retrieval_query = self._is_explicit_retrieval_only(query)
        if force_planner and self.compound_agent:
            is_advisory_query = True
            logger.info("Planner mode forced -> advisory routing enabled")

        should_use_advisory = self.compound_agent and (
            is_advisory_query or (is_advisory_session and not explicit_retrieval_query)
        )

        if should_use_advisory:
            if is_advisory_session and not is_advisory_query:
                logger.info("Follow-up in advisory session -> staying in advisory flow")
            try:
                conversation_history = await self._get_conversation_history(thread_key, advisory=True)
                result = await self._process_compound_query(
                    query=query,
                    thread_key=thread_key,
                    conversation_history=conversation_history,
                    force_planner=force_planner,
                )
                if thread_key:
                    self._touch_local_advisory_thread(thread_key)
                await self._store_conversation_turn(thread_key, query, result["response"], advisory=True)
                logger.info("Compound response in %sms", result["execution_time_ms"])
                return result
            except Exception as e:
                logger.warning("Compound agent error, falling back to LangGraph: %s", e)
        elif is_advisory_session and explicit_retrieval_query:
            logger.info("Explicit retrieval query inside advisory session -> using LangGraph")

        # Priority 1: LangGraph direct retrieval.
        if self.langgraph_agent and config.use_langgraph_agent:
            try:
                conversation_history = await self._get_conversation_history(thread_key)
                if conversation_history:
                    logger.info("LangGraph using %s messages from history", len(conversation_history))

                result = await self.langgraph_agent.process(
                    user_query=query,
                    thread_key=thread_key,
                    conversation_history=conversation_history,
                )

                logger.info("LangGraph response in %sms", result.execution_time_ms)
                if result.tools_used:
                    logger.info("LangGraph tools used: %s", ", ".join(result.tools_used))
                await self._store_conversation_turn(thread_key, query, result.response)

                return {
                    "response": result.response,
                    "sources": result.sources or [],
                    "chunks_used": len(result.sources) if result.sources else getattr(result, "sql_results_count", 0),
                    "response_id": None,
                    "web_results_used": 0,
                    "query_type": "langgraph_agent",
                    "agents_used": result.tools_used,
                    "execution_time_ms": result.execution_time_ms,
                    "agent": "langgraph",
                }
            except Exception as e:
                logger.warning("LangGraph agent error, falling back to direct search: %s", e)

        # Priority 2: direct Pinecone fallback.
        return await self._fallback_search(
            query=query,
            top_k=top_k,
            filters=filters,
            system_instructions=system_instructions,
            thread_key=thread_key,
        )

    async def _fallback_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        system_instructions: Optional[str],
        thread_key: Optional[str],
    ) -> Dict[str, Any]:
        """Fallback to Gemini grounded only in Pinecone search results."""
        print("[RAG] Using grounded Gemini fallback...")

        if not self._gemini_client or not self.fallback_model:
            return {
                "response": "Fallback ist deaktiviert, da Gemini nicht konfiguriert ist.",
                "sources": [],
                "chunks_used": 0,
                "response_id": None,
                "web_results_used": 0,
                "query_type": "error",
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
                "query_type": "fallback",
            }

        # Build context
        full_context, all_sources = self._build_context(search_results, [])

        # Generate response
        if not system_instructions:
            system_instructions = self._get_default_instructions()

        try:
            from google.genai import types

            conversation_history = await self._get_conversation_history(thread_key)
            contents: List[types.Content] = []
            for message in conversation_history[-6:]:
                role = message.get("role", "")
                text = (message.get("content") or "").strip()
                if not text:
                    continue
                gemini_role = "user" if role == "user" else "model"
                contents.append(types.Content(role=gemini_role, parts=[types.Part(text=text)]))

            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=(
                                "Nutze ausschliesslich den folgenden internen Kontext. "
                                "Wenn der Kontext keine belastbare Antwort enthaelt, sage das klar.\n\n"
                                f"KONTEXT:\n{full_context}\n\n"
                                f"FRAGE:\n{query}"
                            )
                        )
                    ],
                )
            )

            response = await self._gemini_client.aio.models.generate_content(
                model=self.fallback_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instructions,
                    temperature=0,
                    max_output_tokens=config.fallback_max_output_tokens,
                ),
            )

            return {
                "response": self._extract_generated_text(response),
                "sources": all_sources,
                "chunks_used": len(search_results),
                "response_id": None,
                "web_results_used": 0,
                "query_type": "fallback",
            }

        except Exception as e:
            print(f"[RAG] Fallback error: {e}")
            return {
                "response": f"Fehler: {str(e)}",
                "sources": all_sources,
                "chunks_used": len(search_results),
                "response_id": None,
                "web_results_used": 0,
                "query_type": "error",
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
