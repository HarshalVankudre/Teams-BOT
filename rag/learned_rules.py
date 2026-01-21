"""
Learned Rules Service

Automatically extracts actionable rules from user feedback and injects them
into the AI's system prompt to improve future responses.

Example:
- User feedback: "vielleicht nicht die ID's ausgeben, sondern direkt die Seriennummer"
- Extracted rule: "Bei Maschinenauflistungen Seriennummer statt ID anzeigen"
- Future queries automatically receive this rule in the system prompt
"""
import json
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExtractedRule:
    """A rule extracted from user feedback."""
    rule_text: str
    category: str  # 'output_format', 'data_display', 'behavior'
    keywords: List[str]
    confidence_score: float
    is_actionable: bool
    source_question: Optional[str] = None
    source_feedback: Optional[str] = None


# Extraction prompt template
EXTRACTION_PROMPT = """Analysiere dieses Nutzerfeedback und extrahiere EINE konkrete Regel.

Frage: {question}
AI-Antwort: {response}
Feedback: {feedback}

Antworte NUR als valides JSON (keine Markdown-Formatierung):
{{
  "rule_text": "Imperative Regel auf Deutsch (z.B. 'Bei Maschinenauflistungen Seriennummer statt ID anzeigen')",
  "category": "output_format|data_display|behavior",
  "keywords": ["keyword1", "keyword2"],
  "confidence_score": 0.0-1.0,
  "is_actionable": true/false
}}

Kategorien:
- output_format: Wie Daten formatiert werden (Listen, Tabellen, Zahlenformat)
- data_display: Welche Daten angezeigt werden (IDs vs Namen, Details)
- behavior: Allgemeines Verhalten (Kuerze, Detaillevel, Sprache)

Setze is_actionable=false wenn:
- Das Feedback nur Lob oder Kritik ohne konkrete Aenderung ist
- Das Feedback zu vage ist um eine Regel abzuleiten
- Das Feedback sich auf einen Fehler bezieht der bereits behoben wurde

Beispiele:
- "zu lang" -> {{"rule_text": "Antworten kuerzer halten", "category": "behavior", "keywords": ["antwort", "laenge"], "confidence_score": 0.7, "is_actionable": true}}
- "danke!" -> {{"rule_text": "", "category": "", "keywords": [], "confidence_score": 0.0, "is_actionable": false}}
- "nicht die ID's ausgeben" -> {{"rule_text": "Bei Maschinenauflistungen Seriennummer statt ID anzeigen", "category": "data_display", "keywords": ["maschine", "liste", "id", "seriennummer"], "confidence_score": 0.9, "is_actionable": true}}
"""


class LearnedRulesService:
    """
    Service for extracting rules from feedback and injecting them into prompts.

    Uses the AdminLogger's database connection for persistence.
    Uses gpt-4o-mini for cost-effective rule extraction.
    """

    # Use a smaller, cheaper model for rule extraction
    EXTRACTION_MODEL = "gpt-4o-mini"

    def __init__(self):
        self._client = None
        self._admin_logger = None

    def _get_client(self):
        """Lazy-load the OpenAI client for rule extraction."""
        if self._client is None:
            import os
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    def _get_admin_logger(self):
        """Lazy-load the admin logger."""
        if self._admin_logger is None:
            from .admin_logger import admin_logger
            self._admin_logger = admin_logger
        return self._admin_logger

    async def extract_rule_from_feedback(
        self,
        question: str,
        response: str,
        feedback: str
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to extract an actionable rule from user feedback.

        Uses gpt-4o-mini for cost-effective extraction.

        Args:
            question: The original user question
            response: The AI's response that received feedback
            feedback: The user's feedback text

        Returns:
            Dict with rule data if actionable, None otherwise
        """
        if not feedback or len(feedback.strip()) < 3:
            logger.debug("Feedback too short for rule extraction")
            return None

        try:
            client = self._get_client()

            # Build the extraction prompt
            prompt = EXTRACTION_PROMPT.format(
                question=question or "(keine Frage)",
                response=(response or "")[:500],  # Truncate long responses
                feedback=feedback
            )

            # Call gpt-4o-mini for cost-effective extraction
            response_obj = await client.chat.completions.create(
                model=self.EXTRACTION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3  # Lower temperature for more consistent JSON
            )

            content = response_obj.choices[0].message.content or ""

            # Parse the JSON response
            # Clean up potential markdown formatting
            content = content.strip()
            if content.startswith("```"):
                # Remove markdown code blocks
                lines = content.split("\n")
                content = "\n".join(
                    line for line in lines
                    if not line.startswith("```")
                )

            rule_data = json.loads(content)

            if not rule_data.get("is_actionable", False):
                logger.debug(f"Feedback not actionable: {feedback[:50]}...")
                return None

            # Add source info
            rule_data["source_question"] = question
            rule_data["source_feedback"] = feedback

            logger.info(f"Extracted rule (via {self.EXTRACTION_MODEL}): {rule_data.get('rule_text', '')[:50]}...")
            return rule_data

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse rule extraction response: {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting rule from feedback: {e}")
            return None

    def save_rule(self, rule: Dict[str, Any]) -> bool:
        """
        Save an extracted rule to the database.

        Args:
            rule: Dict containing rule_text, category, keywords, etc.

        Returns:
            True if saved successfully
        """
        admin_logger = self._get_admin_logger()
        if not admin_logger.available:
            logger.warning("Admin logger not available, cannot save rule")
            return False

        # Check for duplicates first
        if self.check_duplicate(rule.get("rule_text", "")):
            logger.info("Duplicate rule detected, skipping save")
            return False

        return admin_logger.save_learned_rule(rule)

    def check_duplicate(self, rule_text: str) -> bool:
        """
        Check if a similar rule already exists.

        Args:
            rule_text: The rule text to check

        Returns:
            True if a similar rule exists
        """
        if not rule_text:
            return True

        admin_logger = self._get_admin_logger()
        if not admin_logger.available:
            return False

        # Get existing rules and check for similarity
        existing_rules = admin_logger.get_active_rules()

        rule_lower = rule_text.lower().strip()
        for existing in existing_rules:
            existing_lower = existing.get("rule_text", "").lower().strip()

            # Exact match
            if rule_lower == existing_lower:
                return True

            # High similarity (simple substring check)
            if len(rule_lower) > 20 and len(existing_lower) > 20:
                # Check if one is a substring of the other
                if rule_lower in existing_lower or existing_lower in rule_lower:
                    return True

        return False

    def get_all_active_rules(self) -> List[Dict[str, Any]]:
        """
        Get all active learned rules from the database.

        Returns:
            List of rule dicts with rule_text, category, etc.
        """
        admin_logger = self._get_admin_logger()
        if not admin_logger.available:
            return []

        return admin_logger.get_active_rules()

    def build_rules_prompt_section(self) -> str:
        """
        Build the rules section to inject into the system prompt.

        Returns:
            Formatted string with all active rules, or empty string if none
        """
        rules = self.get_all_active_rules()

        if not rules:
            return ""

        rules_text = "\n".join([f"  {i+1}. {r['rule_text']}" for i, r in enumerate(rules)])

        return f"""
================================================================================
NUTZERPRAEFERENZEN (aus Feedback gelernt) - {len(rules)} aktive Regeln
================================================================================

PRAEFERENZEN:
{rules_text}

ANWENDUNGSLOGIK:
- Diese Praeferenzen sind STANDARDWERTE fuer allgemeine Anfragen
- EXPLIZITE Nutzerwuensche haben IMMER Vorrang vor diesen Praeferenzen
- Wende Praeferenzen nur an wenn sie zur Frage passen und kein Konflikt besteht
- Bei Unsicherheit: Folge der direkten Nutzeranfrage, nicht der Praeferenz

================================================================================

"""

    def increment_usage(self, rule_id: int) -> bool:
        """
        Increment the usage count for a rule.

        Args:
            rule_id: The rule's database ID

        Returns:
            True if updated successfully
        """
        admin_logger = self._get_admin_logger()
        if not admin_logger.available:
            return False

        return admin_logger.increment_rule_usage(rule_id)


# Global singleton instance
learned_rules_service = LearnedRulesService()
