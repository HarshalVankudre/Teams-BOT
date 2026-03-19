"""Advisory agent powered by Google Gemini 3 Flash Preview with Search Grounding."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from rag.config import config
from rag.prompts import ADVISORY_SYSTEM_PROMPT, FOLLOWUP_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

RECOMMENDATION_MARKER = "[EMPFEHLUNG_BEREIT]"


@dataclass
class CompoundResult:
    """Result from one advisory turn."""

    response: str
    executed_tools: List[str]
    execution_time_ms: int
    needs_db_lookup: bool = False
    suggested_categories: Optional[List[str]] = None
    suggested_specs: Optional[Dict[str, Any]] = None
    is_followup: bool = False


class CompoundAgent:
    """Gemini advisory agent with native Google Search Grounding."""

    def __init__(self):
        if not config.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required for the Gemini advisory agent")

        self.client = genai.Client(api_key=config.google_api_key)
        self.gemini_model = config.advisory_model
        logger.info("Gemini advisory agent initialized (model: %s)", self.gemini_model)

    async def process(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict]] = None,
        recommendation_given: bool = False,
        stage: str = "",
        project_memory: Optional[Dict[str, Any]] = None,
    ) -> CompoundResult:
        """Process one advisory turn using Gemini with Google Search Grounding."""
        _ = stage
        start_time = time.time()
        history = conversation_history or []
        is_followup = recommendation_given
        system_prompt = FOLLOWUP_SYSTEM_PROMPT if is_followup else ADVISORY_SYSTEM_PROMPT

        if project_memory:
            memory_summary = (project_memory.get("summary") or "").strip()[:1200]
            if memory_summary:
                system_prompt += f"\n\nBekannter Projektkontext:\n{memory_summary}"

        contents: List[types.Content] = []
        for message in history[-20:]:
            role = message.get("role", "")
            text = (message.get("content") or "").strip().replace(RECOMMENDATION_MARKER, "").strip()
            if not text:
                continue
            gemini_role = "user" if role == "user" else "model"
            contents.append(types.Content(role=gemini_role, parts=[types.Part(text=text)]))

        contents.append(types.Content(role="user", parts=[types.Part(text=user_query)]))

        try:
            response = await self.client.aio.models.generate_content(
                model=self.gemini_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
            response_text = self._strip_cost_info(self._extract_text(response))
            needs_db = RECOMMENDATION_MARKER in response_text
            response_text = response_text.replace(RECOMMENDATION_MARKER, "").strip()
            return CompoundResult(
                response=response_text,
                executed_tools=["google_search"] if self._search_was_used(response) else [],
                execution_time_ms=int((time.time() - start_time) * 1000),
                needs_db_lookup=needs_db,
                suggested_categories=self._extract_equipment_categories(response_text) if needs_db else None,
                is_followup=is_followup,
            )
        except Exception as exc:
            logger.error("Gemini advisory agent error: %s", exc, exc_info=True)
            raise

    async def suggest_alternative_categories(
        self,
        missing_categories: List[str],
        project_context: str,
        available_categories: List[str],
    ) -> Dict[str, List[str]]:
        """Suggest in-database alternatives for missing categories via Gemini + Search."""
        if not missing_categories:
            return {}

        available_preview = ", ".join(available_categories[:80])
        prompt = (
            "Es fehlen passende Kategorien in unserer Datenbank.\n"
            f"Fehlende Kategorien: {', '.join(missing_categories)}\n"
            f"Projektkontext: {project_context[:1000]}\n\n"
            "Verfuegbare Datenbank-Kategorien (nur diese sind erlaubt):\n"
            f"{available_preview}\n\n"
            "Recherchiere kurz und bestimme fachlich sinnvolle Ersatz-Kategorien. "
            "Antworte NUR als JSON:\n"
            '{"alternatives":[{"missing":"<Kategorie>","suggested":["<DB-Kategorie>","..."]}]}'
        )

        try:
            response = await self.client.aio.models.generate_content(
                model=self.gemini_model,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(
                    system_instruction=ADVISORY_SYSTEM_PROMPT,
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
            parsed = self._parse_alternative_mapping(
                self._extract_text(response),
                missing_categories,
                available_categories,
            )
            if parsed:
                return parsed
        except Exception as exc:
            logger.warning("Alternative category research failed: %s", exc)

        return self._fallback_alternative_mapping(missing_categories, available_categories)

    def _extract_text(self, response: Any) -> str:
        """Safely extract text from a Gemini response."""
        try:
            return response.text or ""
        except Exception:
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    return "".join(
                        part.text for part in candidate.content.parts if hasattr(part, "text") and part.text
                    )
        return ""

    def _search_was_used(self, response: Any) -> bool:
        """Check if Google Search grounding was actually triggered."""
        try:
            if not response.candidates:
                return False
            grounding = getattr(response.candidates[0], "grounding_metadata", None)
            if not grounding:
                return False
            return bool(
                getattr(grounding, "web_search_queries", None)
                or getattr(grounding, "search_entry_point", None)
            )
        except Exception:
            return False

    def _strip_cost_info(self, text: str) -> str:
        pattern = r".*(?:kosten|preis|budget|euro|eur|preisrahmen|kostenschaetzung|kostenabschaetzung).*\n?"
        return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    def _extract_equipment_categories(self, response: str) -> List[str]:
        """Extract equipment category names mentioned in recommendation text."""
        category_map = {
            "kettenfertiger": "Kettenfertiger",
            "radfertiger": "Radfertiger",
            "fertiger": "Fertiger",
            "kettenbagger": "Kettenbagger",
            "mobilbagger": "Mobilbagger",
            "minibagger": "Minibagger",
            "bagger": "Bagger",
            "tandemwalze": "Tandemwalze",
            "gummiradwalze": "Gummiradwalze",
            "walze": "Walze",
            "fraese": "Fraese",
            "frase": "Fraese",
            "kaltfraese": "Kaltfraese",
            "radlader": "Radlader",
            "raupe": "Raupe",
            "dumper": "Dumper",
            "ruettelplatte": "Ruettelplatte",
            "stampfer": "Stampfer",
            "brecher": "Brecher",
            "siebanlage": "Siebanlage",
        }
        lower = response.lower()
        found: List[str] = []
        for keyword, canonical in category_map.items():
            if keyword in lower and canonical not in found:
                found.append(canonical)
        return found

    def _parse_alternative_mapping(
        self,
        raw_text: str,
        missing_categories: List[str],
        available_categories: List[str],
    ) -> Dict[str, List[str]]:
        json_candidate = self._extract_json_object(raw_text)
        if not json_candidate:
            return {}
        try:
            parsed = json.loads(json_candidate)
        except Exception:
            return {}

        alternatives = parsed.get("alternatives", []) if isinstance(parsed, dict) else []
        if not isinstance(alternatives, list):
            return {}

        available_lookup = {category.lower(): category for category in available_categories}
        result: Dict[str, List[str]] = {category: [] for category in missing_categories}

        for row in alternatives:
            if not isinstance(row, dict):
                continue
            missing = (row.get("missing") or "").strip()
            if missing not in result:
                continue
            for suggestion in row.get("suggested") or []:
                key = str(suggestion).strip().lower()
                if key in available_lookup:
                    canonical = available_lookup[key]
                    if canonical not in result[missing]:
                        result[missing].append(canonical)
        return result

    def _extract_json_object(self, text: str) -> Optional[str]:
        clean = text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", clean, flags=re.IGNORECASE)
        if fenced:
            return fenced.group(1)
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            return clean[start : end + 1]
        return None

    def _fallback_alternative_mapping(
        self,
        missing_categories: List[str],
        available_categories: List[str],
    ) -> Dict[str, List[str]]:
        fallback = {
            "Kettenfertiger": ["Radfertiger", "Fertiger"],
            "Radfertiger": ["Kettenfertiger", "Fertiger"],
            "Kaltfraese": ["Fraese", "Walze"],
            "Kettenbagger": ["Mobilbagger", "Minibagger", "Bagger"],
            "Mobilbagger": ["Kettenbagger", "Minibagger", "Bagger"],
            "Tandemwalze": ["Walze", "Gummiradwalze"],
        }
        available = set(available_categories)
        return {
            missing: [candidate for candidate in fallback.get(missing, []) if candidate in available]
            for missing in missing_categories
        }


_compound_instance: Optional[CompoundAgent] = None


def get_compound_agent() -> CompoundAgent:
    """Get or create the singleton Gemini advisory agent."""
    global _compound_instance
    if _compound_instance is None:
        _compound_instance = CompoundAgent()
    return _compound_instance
