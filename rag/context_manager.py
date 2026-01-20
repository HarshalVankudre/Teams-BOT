"""
Enhanced Context Manager

Manages conversation context and follow-up handling with explicit context injection.
Improves the agent's ability to maintain context across turns.
"""
import time
import re
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """Rich context for a conversation thread."""
    thread_key: str

    # Current turn context
    current_query: str = ""
    is_followup: bool = False
    followup_type: Optional[str] = None  # "filter", "detail", "count", "compare"
    referenced_entities: List[str] = field(default_factory=list)

    # Previous turn context
    last_query: str = ""
    last_response: str = ""
    last_sql: str = ""
    last_sql_purpose: str = ""
    last_result_ids: List[Any] = field(default_factory=list)
    last_result_count: int = 0
    last_results_sample: List[Dict[str, Any]] = field(default_factory=list)

    # Accumulated context
    target_width_m: Optional[float] = None
    active_filters: Dict[str, Any] = field(default_factory=dict)
    mentioned_manufacturers: List[str] = field(default_factory=list)
    mentioned_categories: List[str] = field(default_factory=list)

    # Metadata
    turn_count: int = 0
    last_updated: float = field(default_factory=time.time)

    def to_prompt_section(self) -> str:
        """Convert context to a prompt section for injection."""
        lines = ["CONVERSATION CONTEXT (use this information):"]

        if self.is_followup:
            lines.append(f"- This is a FOLLOW-UP query (type: {self.followup_type})")
            if self.last_result_ids:
                lines.append(f"- Previous results: {len(self.last_result_ids)} items")
                lines.append(f"- Result IDs for 'davon/diese' queries: {self.last_result_ids[:25]}")

        if self.last_sql_purpose:
            lines.append(f"- Last query was about: {self.last_sql_purpose}")

        if self.target_width_m is not None:
            lines.append(f"- Target width requirement: {self.target_width_m}m")

        if self.active_filters:
            filters_str = ", ".join(f"{k}={v}" for k, v in self.active_filters.items())
            lines.append(f"- Active filters: {filters_str}")

        if self.mentioned_manufacturers:
            lines.append(f"- Mentioned manufacturers: {', '.join(self.mentioned_manufacturers)}")

        if self.mentioned_categories:
            lines.append(f"- Mentioned categories: {', '.join(self.mentioned_categories)}")

        if self.last_results_sample:
            lines.append("- Sample of last results:")
            for i, item in enumerate(self.last_results_sample[:3], 1):
                name = item.get("bezeichnung") or item.get("id") or f"Item {i}"
                lines.append(f"  {i}. {name}")

        lines.append("")
        lines.append("FOLLOW-UP RULES:")
        lines.append("- 'davon/diese/welche davon' = filter the previous result set")
        lines.append("- 'zeige mehr/details' = expand on previous results")
        lines.append("- Preserve active filters unless user explicitly changes them")

        # Add specific guidance for comparison/recommendation follow-ups
        if self.followup_type == "compare":
            lines.append("")
            lines.append("COMPARISON/RECOMMENDATION REQUEST DETECTED:")
            lines.append("- User wants comparison or recommendation based on previous results")
            lines.append("- You MUST use execute_sql to get data for comparison")
            lines.append("- Compare the relevant attributes (e.g., Kette vs Mobil, manufacturers, specs)")
            lines.append("- Use the previous query context to build your comparison query")
        elif self.followup_type == "continuation":
            lines.append("")
            lines.append("CONTINUATION REQUEST DETECTED:")
            lines.append("- User wants to continue with previous results")
            lines.append("- Reference previous result IDs if available")
            lines.append("- Apply new criteria to the previous result set")

        return "\n".join(lines)


# Patterns for detecting follow-up types
FOLLOWUP_PATTERNS = {
    "filter": [
        r"\bdavon\b",
        r"\bdiese[nr]?\b",
        r"\bwelche\s+davon\b",
        r"\bdie\s+alle\b",
        r"\bmit\s+\w+\b.*\?$",  # "mit Klimaanlage?"
        r"\bohne\s+\w+",  # "ohne X"
        r"\bdie\s+maschine[n]?\b",  # "die Maschine(n)" - reference to previous
        r"\bdas\s+gerät\b",  # "das Gerät" - reference to previous
    ],
    "detail": [
        r"\bzeige?\s+(mir\s+)?mehr\b",
        r"\bdetails?\b",
        r"\bgenauer\b",
        r"\bwelche\s+eigenschaften\b",
        r"\bwas\s+kannst\s+du\s+mir\s+.*\s+sagen\b",
    ],
    "count": [
        r"\bwie\s+viele\s+davon\b",
        r"\banzahl\b.*\bdavon\b",
        r"\bwieviele\b",
    ],
    "compare": [
        r"\bvergleiche?\b.*\bdavon\b",
        r"\bwelche[rs]?\s+.*\s+(besser|optimal|am\s+besten)\b",
        r"\bunterschied\b",
        r"\bempfiehl\w*\b",  # "empfehlen", "empfiehlst" - recommendation request
        r"\b(kette|mobil|rad)\s+(oder|vs)\s+(kette|mobil|rad)\b",  # "Kette oder Mobil"
        r"\bwas\s+.*\b(besser|empfehlen)\b",  # "was ist besser", "was empfiehlst du"
        r"\bsollte?\s+ich\b",  # "sollte ich" - asking for advice
    ],
    "continuation": [
        r"\bich\s+(möchte|will|brauche)\s+(die|das|eine?n?)\b",  # "Ich möchte die/das/einen"
        r"\bmieten\b.*\b(die|das|eine?n?)\b",  # "mieten die/das/einen"
        r"\bkaufen\b.*\b(die|das|eine?n?)\b",  # "kaufen die/das/einen"
        r"\bdamit\b",  # "damit" - with it (reference to previous)
        r"\bdafür\b",  # "dafür" - for that (reference to previous)
    ],
}

# Patterns for extracting entities
MANUFACTURER_PATTERNS = [
    r"\b(bomag|voegele|vögele|hamm|dynapac|wirtgen|kleemann|caterpillar|cat|volvo|liebherr)\b",
]

CATEGORY_PATTERNS = [
    r"\b(kettenfertiger|radfertiger|walze[nr]?|fertiger|bagger|mobilbagger|kettenbagger|"
    r"radlader|dumper|kran|telekran|kaltfraese|kaltfräse)\b",
]


class ContextManager:
    """
    Manages conversation context across turns.

    Provides rich context injection and follow-up detection.
    """

    def __init__(self, ttl_seconds: int = 72 * 3600):
        """
        Initialize the context manager.

        Args:
            ttl_seconds: Time-to-live for context (default: 72 hours)
        """
        self._contexts: Dict[str, ConversationContext] = {}
        self._ttl_seconds = ttl_seconds

    def get_context(self, thread_key: str) -> ConversationContext:
        """Get or create context for a thread."""
        self._prune_expired()

        if thread_key not in self._contexts:
            self._contexts[thread_key] = ConversationContext(thread_key=thread_key)

        return self._contexts[thread_key]

    def update_context(
        self,
        thread_key: str,
        query: str,
        response: str = "",
        sql: str = "",
        sql_purpose: str = "",
        result_ids: Optional[List[Any]] = None,
        results_sample: Optional[List[Dict[str, Any]]] = None,
        result_count: int = 0
    ) -> ConversationContext:
        """
        Update context after a turn.

        Args:
            thread_key: Thread identifier
            query: User's query
            response: Assistant's response
            sql: SQL query executed (if any)
            sql_purpose: Purpose of the SQL query
            result_ids: IDs from SQL results
            results_sample: Sample of results
            result_count: Total result count

        Returns:
            Updated ConversationContext
        """
        ctx = self.get_context(thread_key)

        # Shift current to previous
        ctx.last_query = ctx.current_query
        ctx.last_response = response

        if sql:
            ctx.last_sql = sql
        if sql_purpose:
            ctx.last_sql_purpose = sql_purpose
        if result_ids is not None:
            ctx.last_result_ids = result_ids
        if results_sample is not None:
            ctx.last_results_sample = results_sample
        if result_count:
            ctx.last_result_count = result_count

        # Update current
        ctx.current_query = query
        ctx.turn_count += 1
        ctx.last_updated = time.time()

        # Detect follow-up
        ctx.is_followup, ctx.followup_type = self._detect_followup(query)

        # Extract entities
        ctx.referenced_entities = self._extract_entities(query)

        # Extract width requirement
        width = self._extract_width(query)
        if width is not None:
            ctx.target_width_m = width

        # Extract manufacturers
        manufacturers = self._extract_manufacturers(query)
        if manufacturers:
            ctx.mentioned_manufacturers = list(set(ctx.mentioned_manufacturers + manufacturers))

        # Extract categories
        categories = self._extract_categories(query)
        if categories:
            ctx.mentioned_categories = list(set(ctx.mentioned_categories + categories))

        # Update active filters from query
        self._update_filters(ctx, query)

        return ctx

    def clear_context(self, thread_key: str) -> None:
        """Clear context for a thread."""
        self._contexts.pop(thread_key, None)

    def _prune_expired(self) -> None:
        """Remove expired contexts."""
        now = time.time()
        expired = [
            key for key, ctx in self._contexts.items()
            if (now - ctx.last_updated) > self._ttl_seconds
        ]
        for key in expired:
            del self._contexts[key]

    def _detect_followup(self, query: str) -> tuple[bool, Optional[str]]:
        """Detect if query is a follow-up and what type."""
        query_lower = query.lower()

        # Find all matching types
        matching_types = []
        for followup_type, patterns in FOLLOWUP_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    matching_types.append(followup_type)
                    break  # Found match for this type, move to next type

        if not matching_types:
            return False, None

        # Priority order: compare > continuation > detail > count > filter
        # (more specific types should take precedence)
        priority = ["compare", "continuation", "detail", "count", "filter"]
        for ptype in priority:
            if ptype in matching_types:
                return True, ptype

        # Fallback to first match
        return True, matching_types[0]

    def _extract_entities(self, query: str) -> List[str]:
        """Extract entity references from query."""
        entities = []

        # ID references
        id_matches = re.findall(r"\b(?:id\s*)?(\d{4,})\b", query)
        entities.extend([f"id:{m}" for m in id_matches])

        # Serial number references
        serial_matches = re.findall(r"\b(?:seriennummer\s+)?([A-Z]{2,}\d+)\b", query)
        entities.extend([f"serial:{m}" for m in serial_matches])

        return entities

    @staticmethod
    def _extract_width(query: str) -> Optional[float]:
        """Extract width requirement from query."""
        match = re.search(r"\b(\d+(?:[.,]\d+)?)\s*m\b", query.lower())
        if match:
            try:
                return float(match.group(1).replace(",", "."))
            except ValueError:
                pass
        return None

    @staticmethod
    def _extract_manufacturers(query: str) -> List[str]:
        """Extract manufacturer mentions from query."""
        manufacturers = []
        query_lower = query.lower()

        for pattern in MANUFACTURER_PATTERNS:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            manufacturers.extend(matches)

        return list(set(manufacturers))

    @staticmethod
    def _extract_categories(query: str) -> List[str]:
        """Extract category mentions from query."""
        categories = []
        query_lower = query.lower()

        for pattern in CATEGORY_PATTERNS:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            categories.extend(matches)

        return list(set(categories))

    def _update_filters(self, ctx: ConversationContext, query: str) -> None:
        """Update active filters based on query."""
        query_lower = query.lower()

        # Rental filter
        if re.search(r"\bmiet", query_lower):
            ctx.active_filters["verwendung"] = "MIET"
        elif re.search(r"\bverkauf|vk\b", query_lower):
            ctx.active_filters["verwendung"] = "VK"

        # Availability filter
        if re.search(r"\bverfügbar|released|frei\b", query_lower):
            ctx.active_filters["nuclos_state"] = "Released"

        # AC filter
        if re.search(r"\bklima", query_lower):
            if re.search(r"\bohne\s+klima", query_lower):
                ctx.active_filters["klimaanlage"] = False
            else:
                ctx.active_filters["klimaanlage"] = True


# Global instance
context_manager = ContextManager()
