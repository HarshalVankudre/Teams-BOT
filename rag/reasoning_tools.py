"""
Reasoning Tools Module

Provides tools for calculations, comparisons, and aggregations
that go beyond simple data retrieval.
"""
import re
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CalculationResult:
    """Result of a calculation."""
    expression: str
    result: Union[float, int, str]
    unit: Optional[str] = None
    breakdown: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class ComparisonResult:
    """Result of a comparison."""
    items: List[Dict[str, Any]]
    criteria: List[str]
    ranking: List[Dict[str, Any]] = field(default_factory=list)
    winner: Optional[Dict[str, Any]] = None
    summary: str = ""
    success: bool = True
    error: Optional[str] = None


@dataclass
class AggregationResult:
    """Result of an aggregation."""
    operation: str  # "sum", "avg", "count", "min", "max"
    field: str
    result: Union[float, int]
    group_by: Optional[str] = None
    groups: Optional[List[Dict[str, Any]]] = None
    success: bool = True
    error: Optional[str] = None


# Tool definitions for OpenAI function calling format
REASONING_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform calculations on data. Use for cost calculations, unit conversions, percentages, and math operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate (e.g., '100 * 3.5 + 50', 'sum([10, 20, 30])')"
                    },
                    "values": {
                        "type": "object",
                        "description": "Named values to use in expression (e.g., {'price': 100, 'quantity': 5})"
                    },
                    "unit": {
                        "type": "string",
                        "description": "Unit for the result (e.g., 'EUR', 'kg', 'm')"
                    },
                    "purpose": {
                        "type": "string",
                        "description": "What this calculation is for"
                    }
                },
                "required": ["expression", "purpose"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare",
            "description": "Compare multiple items across criteria. Use for recommendations, finding best options, and ranking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "List of items to compare (each with properties)"
                    },
                    "criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Criteria to compare on (e.g., ['prop_einbaubreite_max', 'prop_gewicht', 'nuclos_state'])"
                    },
                    "weights": {
                        "type": "object",
                        "description": "Optional weights for criteria (e.g., {'prop_einbaubreite_max': 2.0, 'prop_gewicht': 1.0})"
                    },
                    "requirements": {
                        "type": "object",
                        "description": "Hard requirements (e.g., {'prop_einbaubreite_max': {'min': 3.0}})"
                    },
                    "purpose": {
                        "type": "string",
                        "description": "What this comparison is for"
                    }
                },
                "required": ["items", "criteria", "purpose"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "aggregate",
            "description": "Perform aggregation operations on data. Use for totals, averages, counts by group.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Data to aggregate"
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["sum", "avg", "count", "min", "max"],
                        "description": "Aggregation operation"
                    },
                    "field": {
                        "type": "string",
                        "description": "Field to aggregate (e.g., 'prop_gewicht')"
                    },
                    "group_by": {
                        "type": "string",
                        "description": "Optional field to group by (e.g., 'hersteller_name')"
                    },
                    "purpose": {
                        "type": "string",
                        "description": "What this aggregation is for"
                    }
                },
                "required": ["data", "operation", "field", "purpose"]
            }
        }
    }
]


class ReasoningTools:
    """
    Tools for reasoning operations: calculations, comparisons, aggregations.
    """

    # Safe math operations for eval
    SAFE_MATH = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "pow": pow,
    }

    def calculate(
        self,
        expression: str,
        values: Optional[Dict[str, Any]] = None,
        unit: Optional[str] = None,
        purpose: str = ""
    ) -> CalculationResult:
        """
        Perform a calculation.

        Args:
            expression: Math expression (e.g., "price * quantity")
            values: Named values to substitute
            unit: Unit for result
            purpose: Description of calculation

        Returns:
            CalculationResult with computed value
        """
        try:
            # Build namespace with safe functions and values
            namespace = dict(self.SAFE_MATH)
            if values:
                namespace.update(values)

            # Sanitize expression (basic safety)
            sanitized = expression
            # Remove potentially dangerous patterns
            dangerous = ["import", "exec", "eval", "__", "open", "file"]
            for d in dangerous:
                if d in sanitized.lower():
                    return CalculationResult(
                        expression=expression,
                        result=0,
                        success=False,
                        error=f"Unsafe expression: contains '{d}'"
                    )

            # Evaluate expression
            result = eval(sanitized, {"__builtins__": {}}, namespace)

            # Format result
            if isinstance(result, float):
                result = round(result, 4)

            breakdown = None
            if values:
                breakdown = f"Values: {values} → {expression} = {result}"

            return CalculationResult(
                expression=expression,
                result=result,
                unit=unit,
                breakdown=breakdown,
                success=True
            )

        except Exception as e:
            return CalculationResult(
                expression=expression,
                result=0,
                success=False,
                error=str(e)
            )

    def compare(
        self,
        items: List[Dict[str, Any]],
        criteria: List[str],
        weights: Optional[Dict[str, float]] = None,
        requirements: Optional[Dict[str, Dict[str, Any]]] = None,
        purpose: str = ""
    ) -> ComparisonResult:
        """
        Compare items across multiple criteria.

        Args:
            items: List of items (dicts) to compare
            criteria: Fields to compare on
            weights: Optional weights for criteria (default: 1.0 each)
            requirements: Hard requirements (items not meeting these are excluded)
            purpose: Description of comparison

        Returns:
            ComparisonResult with ranking and winner
        """
        if not items:
            return ComparisonResult(
                items=[],
                criteria=criteria,
                success=False,
                error="No items to compare"
            )

        if not criteria:
            return ComparisonResult(
                items=items,
                criteria=[],
                success=False,
                error="No criteria specified"
            )

        weights = weights or {c: 1.0 for c in criteria}
        requirements = requirements or {}

        # Filter by requirements
        filtered_items = []
        for item in items:
            meets_requirements = True
            for field, req in requirements.items():
                value = item.get(field)
                if value is None:
                    meets_requirements = False
                    break
                if "min" in req and value < req["min"]:
                    meets_requirements = False
                    break
                if "max" in req and value > req["max"]:
                    meets_requirements = False
                    break
                if "equals" in req and value != req["equals"]:
                    meets_requirements = False
                    break

            if meets_requirements:
                filtered_items.append(item)

        if not filtered_items:
            return ComparisonResult(
                items=items,
                criteria=criteria,
                success=True,
                summary="No items meet the requirements"
            )

        # Score each item
        scored_items = []
        for item in filtered_items:
            score = 0.0
            score_details = {}

            for criterion in criteria:
                value = item.get(criterion)
                weight = weights.get(criterion, 1.0)

                if value is None:
                    continue

                # Normalize score based on type
                if isinstance(value, bool):
                    criterion_score = 1.0 if value else 0.0
                elif isinstance(value, (int, float)):
                    # Higher is better by default
                    criterion_score = float(value)
                elif isinstance(value, str):
                    # Special handling for known values
                    if value.lower() == "released":
                        criterion_score = 1.0
                    elif value.lower() in ("locked", "verkauft"):
                        criterion_score = 0.0
                    else:
                        criterion_score = 0.5
                else:
                    criterion_score = 0.0

                weighted_score = criterion_score * weight
                score += weighted_score
                score_details[criterion] = {"value": value, "score": weighted_score}

            scored_items.append({
                **item,
                "_score": score,
                "_score_details": score_details
            })

        # Sort by score descending
        ranked = sorted(scored_items, key=lambda x: x.get("_score", 0), reverse=True)

        # Generate summary
        winner = ranked[0] if ranked else None
        summary_parts = []
        if winner:
            name = winner.get("bezeichnung") or winner.get("id") or "Item 1"
            summary_parts.append(f"Best match: {name} (score: {winner.get('_score', 0):.2f})")
            if len(ranked) > 1:
                summary_parts.append(f"Compared {len(ranked)} items meeting requirements")

        return ComparisonResult(
            items=items,
            criteria=criteria,
            ranking=ranked[:10],  # Top 10
            winner=winner,
            summary=" | ".join(summary_parts),
            success=True
        )

    def aggregate(
        self,
        data: List[Dict[str, Any]],
        operation: str,
        field: str,
        group_by: Optional[str] = None,
        purpose: str = ""
    ) -> AggregationResult:
        """
        Perform aggregation on data.

        Args:
            data: List of data items
            operation: "sum", "avg", "count", "min", "max"
            field: Field to aggregate
            group_by: Optional grouping field
            purpose: Description of aggregation

        Returns:
            AggregationResult with computed values
        """
        if not data:
            return AggregationResult(
                operation=operation,
                field=field,
                result=0,
                success=False,
                error="No data to aggregate"
            )

        valid_ops = {"sum", "avg", "count", "min", "max"}
        if operation not in valid_ops:
            return AggregationResult(
                operation=operation,
                field=field,
                result=0,
                success=False,
                error=f"Invalid operation. Use: {valid_ops}"
            )

        def do_aggregate(items: List[Dict], op: str, fld: str) -> Union[int, float]:
            values = [
                item.get(fld) for item in items
                if item.get(fld) is not None and isinstance(item.get(fld), (int, float))
            ]

            if not values:
                return 0

            if op == "sum":
                return sum(values)
            elif op == "avg":
                return sum(values) / len(values)
            elif op == "count":
                return len(values)
            elif op == "min":
                return min(values)
            elif op == "max":
                return max(values)
            return 0

        if group_by:
            # Group data
            groups: Dict[str, List[Dict]] = {}
            for item in data:
                key = str(item.get(group_by, "Unknown"))
                if key not in groups:
                    groups[key] = []
                groups[key].append(item)

            # Aggregate each group
            group_results = []
            for key, items in sorted(groups.items()):
                value = do_aggregate(items, operation, field)
                group_results.append({
                    group_by: key,
                    f"{operation}_{field}": value,
                    "count": len(items)
                })

            # Total across all
            total = do_aggregate(data, operation, field)

            return AggregationResult(
                operation=operation,
                field=field,
                result=total,
                group_by=group_by,
                groups=group_results,
                success=True
            )
        else:
            result = do_aggregate(data, operation, field)
            return AggregationResult(
                operation=operation,
                field=field,
                result=result,
                success=True
            )


# Global instance
reasoning_tools = ReasoningTools()
