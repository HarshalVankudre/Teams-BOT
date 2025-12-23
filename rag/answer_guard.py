"""
Answer quality guardrails for concise, grounded responses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata
from typing import Any, Dict, List, Optional


_SMALLTALK_RE = re.compile(r"\b(hallo|hi|hello|hey|moin|servus|danke|thanks)\b")
_PROMPT_INJECTION_RE = re.compile(
    r"\b(ignore|ignoriere|override|bypass|system prompt|systemprompt)\b"
)
_SECRETS_RE = re.compile(
    r"\b(api key|apikey|secret|token|password|passwort|credentials?)\b"
)
_EXFIL_RE = re.compile(r"\b(show|zeige|gib|liste|reveal|leak)\b")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


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

        if context.sql_error:
            issues.append("sql_error")
            error_preview = (context.sql_error or "").strip()
            if len(error_preview) > 160:
                error_preview = error_preview[:160] + "..."
            if "validation" in error_preview.lower():
                return GuardedResponse(
                    response=(
                        "Die Abfrage passt nicht zu den geforderten Kriterien. "
                        "Bitte praezisieren Sie Ihre Anfrage."
                    ),
                    issues=issues,
                )
            return GuardedResponse(
                response=(
                    "Die Datenbankabfrage konnte nicht ausgefuehrt werden. "
                    f"Details: {error_preview} "
                    "Bitte praezisieren Sie Ihre Anfrage."
                ),
                issues=issues,
            )

        if self._should_clarify(context, normalized_query):
            issues.append("clarification")
            clarification = getattr(context.intent, "clarification", None)
            return GuardedResponse(
                response=clarification or "Bitte praezisieren Sie Ihre Anfrage.",
                issues=issues,
            )

        if self._needs_no_data_fallback(context, normalized_query):
            issues.append("no_data")
            return GuardedResponse(
                response=(
                    "In den internen Datenbanken wurde keine Information gefunden. "
                    "Gibt es einen Hersteller, Maschinentyp oder weitere Kriterien?"
                ),
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
        if context.tools_used and "execute_sql" in context.tools_used:
            return True
        if getattr(context.intent, "requires_sql", False):
            return True
        return False

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
            text += "\nWeitere Details auf Anfrage."

        sentences = _SENTENCE_SPLIT_RE.split(text)
        if len(sentences) > self.max_sentences:
            text = " ".join(sentences[: self.max_sentences]).strip()
            text += " Weitere Details auf Anfrage."

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
