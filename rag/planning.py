"""
Planning module for the Enhanced Single Agent.

Provides lightweight query analysis and step planning before tool execution.
This helps the agent think through complex queries systematically.
"""
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class QueryPlan:
    """A plan for answering a user query."""
    query: str
    complexity: str  # "simple", "moderate", "complex"
    steps: List[Dict[str, str]] = field(default_factory=list)
    requires_multiple_tools: bool = False
    requires_calculation: bool = False
    requires_comparison: bool = False
    requires_aggregation: bool = False
    is_followup: bool = False
    context_needed: List[str] = field(default_factory=list)

    def to_prompt_section(self) -> str:
        """Convert plan to a prompt section for the agent."""
        if not self.steps:
            return ""

        lines = ["QUERY PLAN (follow these steps):"]
        for i, step in enumerate(self.steps, 1):
            action = step.get("action", "")
            reason = step.get("reason", "")
            lines.append(f"{i}. {action}" + (f" - {reason}" if reason else ""))

        if self.context_needed:
            lines.append(f"\nCONTEXT REQUIRED: {', '.join(self.context_needed)}")

        return "\n".join(lines)


PLANNING_PROMPT = """Analyze this user query and create a brief execution plan.

Query: {query}
Thread Context: {context}

Respond with JSON only:
{{
  "complexity": "simple|moderate|complex",
  "steps": [
    {{"action": "what to do", "tool": "execute_sql|search_documents|calculate|compare|none", "reason": "why"}}
  ],
  "requires_calculation": true/false,
  "requires_comparison": true/false,
  "requires_aggregation": true/false,
  "is_followup": true/false,
  "context_needed": ["list", "of", "context", "items"]
}}

Rules:
- Keep plans SHORT (1-4 steps max)
- "simple" = single tool call, direct answer
- "moderate" = 2-3 tool calls or light reasoning
- "complex" = multiple tools + calculations/comparisons
- is_followup = true if query references previous results ("davon", "diese", "welche davon")
- context_needed = what info from previous conversation is needed
"""


class QueryPlanner:
    """
    Lightweight query planner that analyzes queries before execution.

    Uses a small/fast model call to create execution plans for complex queries.
    Simple queries bypass planning entirely for cost efficiency.
    """

    # Patterns that indicate a simple query (no planning needed)
    SIMPLE_PATTERNS = [
        r"^wie\s+viele\b",  # "wie viele X" - simple count
        r"^zeige?\s+(mir\s+)?\d+\b",  # "zeige 5 X" - simple list
        r"^was\s+ist\b",  # "was ist X" - simple lookup
        r"^liste\b",  # "liste X" - simple list
    ]

    # Patterns that indicate complex query (planning recommended)
    COMPLEX_PATTERNS = [
        r"\bvergleich",  # comparison
        r"\bbest[en]?\b",  # best/recommendation
        r"\bempfehl",  # recommendation
        r"\boptimal",  # optimal choice
        r"\bunterschied",  # difference
        r"\bberechne",  # calculate
        r"\bkost",  # cost calculation
        r"\bdurchschnitt",  # average
        r"\bsumme\b",  # sum
        r"\bgesamt",  # total
        r"\balle\s+.+\s+die\b",  # "alle X die Y" - filtered aggregation
        r"\bwelche[rs]?\s+.+\s+(am\s+besten|optimal)",  # which is best
    ]

    def __init__(self, provider=None, model: str = ""):
        """
        Initialize the planner.

        Args:
            provider: LLM provider for planning calls (optional)
            model: Override model for planning (uses smaller model if empty)
        """
        self._provider = provider
        self._model = model

    def should_plan(self, query: str, thread_state: Optional[Dict] = None) -> bool:
        """
        Determine if a query needs planning.

        Simple queries skip planning for cost efficiency.
        Complex queries benefit from planning.
        """
        import re

        query_lower = query.lower().strip()

        # Skip planning for very short queries
        if len(query_lower) < 15:
            return False

        # Check for simple patterns - no planning needed
        for pattern in self.SIMPLE_PATTERNS:
            if re.search(pattern, query_lower):
                return False

        # Check for complex patterns - planning recommended
        for pattern in self.COMPLEX_PATTERNS:
            if re.search(pattern, query_lower):
                return True

        # Check for follow-ups with context - might need planning
        if thread_state and thread_state.get("last_result_ids"):
            followup_patterns = [r"\bdavon\b", r"\bdiese\b", r"\bwelche\s+davon\b"]
            for pattern in followup_patterns:
                if re.search(pattern, query_lower):
                    return True

        # Default: moderate-length queries get planning
        return len(query_lower) > 60

    def create_simple_plan(self, query: str, thread_state: Optional[Dict] = None) -> QueryPlan:
        """
        Create a simple plan without LLM call.

        Used for queries that don't need sophisticated planning.
        """
        import re

        query_lower = query.lower()

        plan = QueryPlan(query=query, complexity="simple")

        # Detect follow-up
        if re.search(r"\b(davon|diese|diesen|welche\s+davon)\b", query_lower):
            plan.is_followup = True
            if thread_state and thread_state.get("last_result_ids"):
                plan.context_needed = ["last_result_ids"]

        # Detect aggregation
        if re.search(r"\b(wie\s+viele|anzahl|count|zaehle)\b", query_lower):
            plan.requires_aggregation = True
            plan.steps = [{"action": "Count matching records", "tool": "execute_sql"}]
        else:
            plan.steps = [{"action": "Query database or search documents", "tool": "auto"}]

        return plan

    async def create_plan(
        self,
        query: str,
        thread_state: Optional[Dict] = None,
        use_llm: bool = True
    ) -> QueryPlan:
        """
        Create an execution plan for a query.

        Args:
            query: The user's query
            thread_state: Current thread context
            use_llm: Whether to use LLM for planning (False = rule-based only)

        Returns:
            QueryPlan with steps and metadata
        """
        # Check if planning is needed
        if not self.should_plan(query, thread_state):
            return self.create_simple_plan(query, thread_state)

        # If no provider or LLM disabled, use rule-based planning
        if not use_llm or not self._provider:
            return self._create_rule_based_plan(query, thread_state)

        # Use LLM for complex planning
        try:
            return await self._create_llm_plan(query, thread_state)
        except Exception as e:
            logger.warning(f"LLM planning failed, falling back to rules: {e}")
            return self._create_rule_based_plan(query, thread_state)

    def _create_rule_based_plan(
        self,
        query: str,
        thread_state: Optional[Dict] = None
    ) -> QueryPlan:
        """Create a plan using rules instead of LLM."""
        import re

        query_lower = query.lower()
        plan = QueryPlan(query=query, complexity="moderate")

        # Detect requirements
        plan.requires_comparison = bool(re.search(r"\b(vergleich|unterschied|besser|schlechter)\b", query_lower))
        plan.requires_calculation = bool(re.search(r"\b(berechne|kost|preis|summe|gesamt)\b", query_lower))
        plan.requires_aggregation = bool(re.search(r"\b(durchschnitt|average|summe|total|gesamt)\b", query_lower))
        plan.is_followup = bool(re.search(r"\b(davon|diese|diesen|welche\s+davon)\b", query_lower))

        if plan.is_followup and thread_state:
            plan.context_needed = ["last_result_ids", "last_sql_purpose"]

        # Build steps based on detected requirements
        steps = []

        if plan.requires_comparison:
            steps.append({"action": "Get data for all items to compare", "tool": "execute_sql"})
            steps.append({"action": "Compare items on relevant criteria", "tool": "compare"})
        elif plan.requires_calculation:
            steps.append({"action": "Get required data", "tool": "execute_sql"})
            steps.append({"action": "Perform calculation", "tool": "calculate"})
        elif plan.requires_aggregation:
            steps.append({"action": "Query with aggregation", "tool": "execute_sql"})
        else:
            steps.append({"action": "Query database", "tool": "execute_sql"})

        plan.steps = steps
        plan.requires_multiple_tools = len(steps) > 1

        if plan.requires_multiple_tools:
            plan.complexity = "complex"

        return plan

    async def _create_llm_plan(
        self,
        query: str,
        thread_state: Optional[Dict] = None
    ) -> QueryPlan:
        """Create a plan using LLM."""
        from .providers import ChatMessage

        context_summary = "None"
        if thread_state:
            parts = []
            if thread_state.get("last_sql_purpose"):
                parts.append(f"Last query: {thread_state['last_sql_purpose']}")
            if thread_state.get("last_sql_row_count") is not None:
                parts.append(f"Last results: {thread_state['last_sql_row_count']} rows")
            if thread_state.get("last_result_ids"):
                parts.append(f"Has {len(thread_state['last_result_ids'])} result IDs")
            if parts:
                context_summary = "; ".join(parts)

        prompt = PLANNING_PROMPT.format(query=query, context=context_summary)

        response = await self._provider.chat_completion(
            messages=[ChatMessage(role="user", content=prompt)],
            tools=None,
            max_tokens=300
        )

        content = response.content or "{}"

        # Parse JSON response
        try:
            # Clean up potential markdown
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            data = json.loads(content.strip())
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse planning response: {content[:100]}")
            return self._create_rule_based_plan(query, thread_state)

        plan = QueryPlan(
            query=query,
            complexity=data.get("complexity", "moderate"),
            steps=data.get("steps", []),
            requires_calculation=data.get("requires_calculation", False),
            requires_comparison=data.get("requires_comparison", False),
            requires_aggregation=data.get("requires_aggregation", False),
            is_followup=data.get("is_followup", False),
            context_needed=data.get("context_needed", []),
        )
        plan.requires_multiple_tools = len(plan.steps) > 1

        return plan
