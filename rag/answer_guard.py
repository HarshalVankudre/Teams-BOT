"""
Answer quality guardrails for concise, grounded responses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata
from typing import Any, Dict, List, Optional


_SMALLTALK_RE = re.compile(r"\b(hallo|hi|hello|hey|moin|servus|danke|thanks|guten\s*tag|gruss)\b")
_PROMPT_INJECTION_RE = re.compile(
    r"\b(ignore|ignoriere|override|bypass|system prompt|systemprompt)\b"
)
_SECRETS_RE = re.compile(
    r"\b(api key|apikey|secret|token|password|passwort|credentials?)\b"
)
_EXFIL_RE = re.compile(r"\b(show|zeige|gib|liste|reveal|leak)\b")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_HELP_QUESTION_RE = re.compile(
    r"\b(wie\s+funktioniert|was\s+bedeutet|was\s+ist|erklar|hilfe|help|"
    r"kannst\s+du|was\s+kannst|wozu|wofur|warum)\b"
)

# Patterns indicating "no info found" responses
_NO_INFO_RE = re.compile(
    r"\b(keine\s+informationen|nicht\s+gefunden|nichts\s+gefunden|"
    r"keine\s+daten|keine\s+ergebnisse|leider\s+nicht|"
    r"konnte\s+nicht\s+finden|habe\s+keine|finde\s+keine)\b",
    re.IGNORECASE
)

# Patterns that could have space variants (model numbers, codes)
_SPACE_VARIANT_RE = re.compile(r"\b[a-zA-Z]{1,4}\s*\d{2,5}\b")


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


@dataclass
class AnswerContext:
    query: str
    tools_used: List[str]
    sql_results_count: int
    sql_error: Optional[str]
    sources: List[Dict[str, Any]]
    equipment_table: Optional[str]
    intent: Optional[Any] = None
    has_conversation_history: bool = False


@dataclass
class GuardedResponse:
    response: str
    issues: List[str] = field(default_factory=list)


class AnswerGuard:
    def __init__(
        self,
        *,
        max_sentences: int = 4,
        max_bullets: int = 5,
        max_chars: int = 1200,
    ):
        self.max_sentences = max_sentences
        self.max_bullets = max_bullets
        self.max_chars = max_chars

    def apply(self, response: str, context: AnswerContext) -> GuardedResponse:
        issues: List[str] = []
        normalized_query = _normalize_text(context.query or "")

        if self._is_injection_request(normalized_query):
            issues.append("refusal")
            return GuardedResponse(
                response=(
                    "Ich kann keine internen Prompts oder sensiblen Daten offenlegen."
                ),
                issues=issues,
            )

        if self._is_secret_request(normalized_query):
            issues.append("refusal")
            return GuardedResponse(
                response=(
                    "Ich kann keine Zugangsdaten, Tokens oder Geheimnisse teilen."
                ),
                issues=issues,
            )

        if context.sql_error and context.sql_results_count == 0:
            issues.append("sql_error")
            error_preview = (context.sql_error or "").strip()
            if len(error_preview) > 160:
                error_preview = error_preview[:160] + "..."
            if "validation" in error_preview.lower():
                return GuardedResponse(
                    response=(
                        "Entschuldigung, diese Anfrage konnte ich leider nicht verarbeiten. "
                        "Koennten Sie Ihre Frage etwas anders formulieren?"
                    ),
                    issues=issues,
                )
            return GuardedResponse(
                response=(
                    "Entschuldigung, bei der Datenbankabfrage ist ein Problem aufgetreten. "
                    "Koennten Sie Ihre Frage etwas anders formulieren oder weitere Details nennen?"
                ),
                issues=issues,
            )

        if self._should_clarify(context, normalized_query):
            issues.append("clarification")
            clarification = getattr(context.intent, "clarification", None)
            return GuardedResponse(
                response=clarification or "Koennten Sie mir bitte etwas mehr Details zu Ihrer Anfrage geben?",
                issues=issues,
            )

        if self._needs_no_data_fallback(context, normalized_query):
            issues.append("no_data")
            return GuardedResponse(
                response=(
                    "Leider habe ich dazu keine Informationen in den internen Datenbanken gefunden. "
                    "Moechten Sie die Suche mit anderen Kriterien versuchen? "
                    "Zum Beispiel: Hersteller, Maschinentyp oder Einsatzgebiet?"
                ),
                issues=issues,
            )

        # Check for "no info" response when search variants could help
        if self._is_no_info_without_retry(response or "", context):
            issues.append("no_info_without_retry")
            return GuardedResponse(
                response=self._get_search_variant_suggestion(context),
                issues=issues,
            )

        trimmed = self._trim_response(response or "")
        if trimmed != (response or ""):
            issues.append("trimmed")

        with_sources = self._ensure_sources(trimmed, context)
        if with_sources != trimmed:
            issues.append("sources_appended")

        return GuardedResponse(response=with_sources, issues=issues)

    def _is_smalltalk(self, normalized_query: str) -> bool:
        return bool(_SMALLTALK_RE.search(normalized_query))

    def _is_injection_request(self, normalized_query: str) -> bool:
        return bool(_PROMPT_INJECTION_RE.search(normalized_query))

    def _is_secret_request(self, normalized_query: str) -> bool:
        return bool(_SECRETS_RE.search(normalized_query) and _EXFIL_RE.search(normalized_query))

    def _should_clarify(self, context: AnswerContext, normalized_query: str) -> bool:
        if self._is_smalltalk(normalized_query):
            return False
        clarification = getattr(context.intent, "clarification", None)
        if clarification:
            return True
        return False

    def _needs_no_data_fallback(self, context: AnswerContext, normalized_query: str) -> bool:
        if self._is_smalltalk(normalized_query):
            return False
        if context.sources or context.sql_results_count > 0:
            return False
        # Trust SQL execution even with 0 results - 0 is a valid answer
        if "execute_sql" in (context.tools_used or []):
            return False
        # General/help questions don't need data lookup
        if self._is_help_question(normalized_query):
            return False
        # If conversation history exists, trust LLM to use context for follow-up questions
        if context.has_conversation_history:
            return False
        return True

    def _is_help_question(self, normalized_query: str) -> bool:
        return bool(_HELP_QUESTION_RE.search(normalized_query))

    def _is_no_info_without_retry(self, response: str, context: AnswerContext) -> bool:
        """
        Detect when LLM says 'no info' for equipment queries that could have search variants.

        This catches cases where:
        1. Intent required/preferred SQL (equipment query)
        2. SQL was executed but returned 0 results
        3. Response says "keine Informationen" or similar
        4. Query contains model numbers that could have space variants (e.g., bw174 vs bw 174)
        """
        # Only check if intent required/preferred SQL
        intent = context.intent
        if not intent:
            return False
        requires_sql = getattr(intent, "requires_sql", False)
        prefers_sql = getattr(intent, "prefers_sql", False)
        if not (requires_sql or prefers_sql):
            return False

        # Only if SQL was called but returned 0 results
        if "execute_sql" not in (context.tools_used or []):
            return False
        if context.sql_results_count > 0:
            return False

        # Check if response indicates "no info found"
        if not _NO_INFO_RE.search(response or ""):
            return False

        # Check if query has model numbers that could have space variants
        query = context.query or ""
        if _SPACE_VARIANT_RE.search(query):
            return True

        return False

    def _get_search_variant_suggestion(self, context: AnswerContext) -> str:
        """Generate suggestion for search variants based on query."""
        query = context.query or ""

        # Find model number patterns and suggest variants
        matches = _SPACE_VARIANT_RE.findall(query)
        if matches:
            term = matches[0]
            # Check if it has space or not
            if " " in term:
                no_space = term.replace(" ", "")
                return (
                    f"Keine Treffer für '{term}'. Probieren Sie auch die Schreibweise "
                    f"ohne Leerzeichen: '{no_space}'. Oder beschreiben Sie die gesuchte "
                    f"Maschine (z.B. Hersteller, Typ, Einsatzgebiet)."
                )
            else:
                # Insert space between letters and numbers
                with_space = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", term)
                return (
                    f"Keine Treffer für '{term}'. Probieren Sie auch die Schreibweise "
                    f"mit Leerzeichen: '{with_space}'. Oder beschreiben Sie die gesuchte "
                    f"Maschine (z.B. Hersteller, Typ, Einsatzgebiet)."
                )

        return (
            "Keine passenden Maschinen gefunden. Versuchen Sie andere Suchbegriffe "
            "(z.B. Hersteller, Maschinentyp, Seriennummer) oder beschreiben Sie "
            "die gewünschte Funktion."
        )

    def _trim_response(self, response: str) -> str:
        text = (response or "").strip()
        if not text:
            return text

        bullet_lines = [line for line in text.splitlines() if line.strip().startswith(("-", "*"))]
        if bullet_lines and len(bullet_lines) > self.max_bullets:
            trimmed_lines = []
            kept = 0
            for line in text.splitlines():
                if line.strip().startswith(("-", "*")):
                    if kept >= self.max_bullets:
                        continue
                    kept += 1
                trimmed_lines.append(line)
            text = "\n".join(trimmed_lines).strip()
            text += "\n\nSoll ich Ihnen weitere Details zeigen?"

        sentences = _SENTENCE_SPLIT_RE.split(text)
        if len(sentences) > self.max_sentences:
            text = " ".join(sentences[: self.max_sentences]).strip()
            text += " Soll ich mehr Details zeigen?"

        if len(text) > self.max_chars:
            text = text[: self.max_chars].rstrip() + "..."
        return text

    def _ensure_sources(self, response: str, context: AnswerContext) -> str:
        if "quelle" in (response or "").lower():
            return response

        sources = []
        if context.sql_results_count > 0:
            table = context.equipment_table or "equipment_matrix"
            sources.append(f"Interne Datenbank ({table})")

        for source in (context.sources or [])[:2]:
            title = source.get("title") or "Dokument"
            source_file = source.get("source_file") or "Unknown"
            sources.append(f"{title} ({source_file})")

        if not sources:
            return response

        sources_line = "Quellen: " + "; ".join(sources)
        if response.endswith("\n"):
            return response + sources_line
        return response + "\n\n" + sources_line
