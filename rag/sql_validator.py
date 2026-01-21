"""
SQL Validator with Predicate Extraction.

Validates generated SQL against the ReducedSchema and extracts
predicate pairs (column, op, value) from the AST for value checking.

This complements sql_guard.py with deeper semantic validation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, TYPE_CHECKING

try:
    import sqlglot
    from sqlglot import exp
    SQLGLOT_AVAILABLE = True
except ImportError:
    SQLGLOT_AVAILABLE = False

if TYPE_CHECKING:
    from .schema_linker import ReducedSchema

logger = logging.getLogger(__name__)


@dataclass
class Predicate:
    """A filter predicate extracted from SQL."""
    column_id: str  # table.column
    op: str         # =, IN, ILIKE, >=, <=, BETWEEN, etc.
    value: str      # The literal value


@dataclass
class ValidationResult:
    """Result of SQL validation."""
    valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    predicates: List[Predicate] = field(default_factory=list)
    columns_used: List[str] = field(default_factory=list)
    tables_used: List[str] = field(default_factory=list)


class SQLValidator:
    """
    Validate generated SQL against reduced schema.

    Key features:
    - Extract predicate pairs (column, op, value) from AST
    - Check columns exist in ReducedSchema.allowed_columns
    - Check values against value_index (warnings, not hard errors)
    - GROUP BY completeness checking
    """

    # Single-table policy
    ALLOWED_TABLES = frozenset(["equipment_matrix"])

    def __init__(self, default_table: str = "equipment_matrix"):
        self.default_table = default_table

    def validate(
        self,
        sql: str,
        schema: "ReducedSchema",
        value_index: Optional[Any] = None,
    ) -> ValidationResult:
        """
        Validate SQL against reduced schema.

        Args:
            sql: SQL query to validate
            schema: ReducedSchema with allowed columns
            value_index: Optional ValueIndex for categorical column checking

        Returns:
            ValidationResult with issues, warnings, and extracted predicates
        """
        result = ValidationResult(valid=True)

        if not SQLGLOT_AVAILABLE:
            result.warnings.append("sqlglot not available, skipping AST validation")
            return result

        # 1. Parse SQL
        try:
            parsed = sqlglot.parse_one(sql, dialect="postgres")
        except Exception as e:
            result.valid = False
            result.issues.append(f"Syntax error: {e}")
            result.suggestions.append("Check SQL syntax and try again")
            return result

        # 2. Extract referenced tables
        tables = self._extract_tables(parsed)
        result.tables_used = list(tables)

        # 3. Validate tables against policy
        table_errors = self._validate_tables(tables)
        if table_errors:
            result.issues.extend(table_errors)

        # 4. Extract referenced columns (as table.column)
        columns = self._extract_qualified_columns(parsed)
        result.columns_used = list(columns)

        # 5. Check columns exist in schema
        for col_id in columns:
            if col_id not in schema.allowed_columns:
                result.issues.append(f"Unknown column: {col_id}")
                closest = self._find_closest(col_id, schema.allowed_columns)
                if closest:
                    result.suggestions.append(
                        f"Did you mean '{closest}' instead of '{col_id}'?"
                    )

        # 6. Extract predicate pairs from AST
        predicates = self._extract_predicates(parsed)
        result.predicates = predicates

        # 7. Check values against value_index (if provided)
        if value_index:
            for pred in predicates:
                self._check_predicate_value(pred, value_index, result)

        # 8. Check GROUP BY completeness
        group_by_issues = self._check_group_by(parsed)
        result.warnings.extend(group_by_issues)

        # Final validity
        if result.issues:
            result.valid = False

        return result

    def _extract_tables(self, parsed: exp.Expression) -> Set[str]:
        """Extract table names from parsed SQL."""
        tables = set()
        for node in parsed.walk():
            if isinstance(node, exp.Table):
                table_name = node.name
                if table_name:
                    tables.add(table_name.lower())
        return tables

    def _validate_tables(self, tables: Set[str]) -> List[str]:
        """Validate tables against single-table policy."""
        errors = []
        disallowed = tables - self.ALLOWED_TABLES
        if disallowed:
            errors.append(
                f"Query references disallowed tables: {disallowed}. "
                f"Only these tables are allowed: {self.ALLOWED_TABLES}"
            )
        return errors

    def _extract_qualified_columns(self, parsed: exp.Expression) -> Set[str]:
        """Extract columns as qualified table.column strings."""
        columns = set()
        for node in parsed.walk():
            if isinstance(node, exp.Column):
                col_id = self._qualify_column(node)
                if col_id:
                    columns.add(col_id)
        return columns

    def _qualify_column(self, col: exp.Column) -> str:
        """Convert Column AST node to qualified table.column string."""
        table = col.table or self.default_table
        name = col.name
        if not name:
            return ""
        return f"{table}.{name}"

    def _extract_predicates(self, parsed: exp.Expression) -> List[Predicate]:
        """
        Extract predicate pairs from SQL AST.

        Handles:
        - col = 'X'
        - col IN ('X', 'Y')
        - col ILIKE '%X%'
        - col >= N, col <= N
        - LOWER(col) = 'x'
        """
        predicates = []

        for node in parsed.walk():
            # Handle: col = 'value'
            if isinstance(node, exp.EQ):
                pred = self._extract_eq_predicate(node)
                if pred:
                    predicates.append(pred)

            # Handle: col IN ('a', 'b', 'c')
            elif isinstance(node, exp.In):
                preds = self._extract_in_predicate(node)
                predicates.extend(preds)

            # Handle: col ILIKE '%x%'
            elif isinstance(node, exp.ILike):
                pred = self._extract_like_predicate(node, "ILIKE")
                if pred:
                    predicates.append(pred)

            # Handle: col LIKE '%x%'
            elif isinstance(node, exp.Like):
                pred = self._extract_like_predicate(node, "LIKE")
                if pred:
                    predicates.append(pred)

            # Handle: col >= value
            elif isinstance(node, exp.GTE):
                pred = self._extract_comparison_predicate(node, ">=")
                if pred:
                    predicates.append(pred)

            # Handle: col <= value
            elif isinstance(node, exp.LTE):
                pred = self._extract_comparison_predicate(node, "<=")
                if pred:
                    predicates.append(pred)

            # Handle: col > value
            elif isinstance(node, exp.GT):
                pred = self._extract_comparison_predicate(node, ">")
                if pred:
                    predicates.append(pred)

            # Handle: col < value
            elif isinstance(node, exp.LT):
                pred = self._extract_comparison_predicate(node, "<")
                if pred:
                    predicates.append(pred)

        return predicates

    def _extract_eq_predicate(self, node: exp.EQ) -> Optional[Predicate]:
        """Extract column and value from equality predicate."""
        left, right = node.left, node.right

        # Handle LOWER(col) = 'value'
        if isinstance(left, exp.Lower) and isinstance(right, exp.Literal):
            inner = left.this
            if isinstance(inner, exp.Column):
                col_id = self._qualify_column(inner)
                return Predicate(
                    column_id=col_id,
                    op="LOWER=",
                    value=str(right.this)
                )

        # Check both orders: col = 'val' and 'val' = col
        if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
            col_id = self._qualify_column(left)
            return Predicate(
                column_id=col_id,
                op="=",
                value=str(right.this)
            )
        elif isinstance(right, exp.Column) and isinstance(left, exp.Literal):
            col_id = self._qualify_column(right)
            return Predicate(
                column_id=col_id,
                op="=",
                value=str(left.this)
            )

        return None

    def _extract_in_predicate(self, node: exp.In) -> List[Predicate]:
        """Extract predicates from IN clause."""
        predicates = []

        # Get column
        col_expr = node.this
        if not isinstance(col_expr, exp.Column):
            return predicates

        col_id = self._qualify_column(col_expr)

        # Get values
        for value_node in node.expressions:
            if isinstance(value_node, exp.Literal):
                predicates.append(Predicate(
                    column_id=col_id,
                    op="IN",
                    value=str(value_node.this)
                ))

        return predicates

    def _extract_like_predicate(
        self,
        node: exp.Expression,
        op: str
    ) -> Optional[Predicate]:
        """Extract predicate from LIKE/ILIKE clause."""
        # node.this is the column, node.expression is the pattern
        col_expr = node.this
        pattern_expr = node.expression

        if not isinstance(col_expr, exp.Column):
            return None

        col_id = self._qualify_column(col_expr)
        pattern = ""
        if isinstance(pattern_expr, exp.Literal):
            pattern = str(pattern_expr.this)

        return Predicate(
            column_id=col_id,
            op=op,
            value=pattern
        )

    def _extract_comparison_predicate(
        self,
        node: exp.Expression,
        op: str
    ) -> Optional[Predicate]:
        """Extract predicate from comparison (>=, <=, >, <)."""
        left, right = node.left, node.right

        if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
            col_id = self._qualify_column(left)
            return Predicate(
                column_id=col_id,
                op=op,
                value=str(right.this)
            )
        elif isinstance(right, exp.Column) and isinstance(left, exp.Literal):
            # Flip the operator for reversed order
            flipped_op = {">=": "<=", "<=": ">=", ">": "<", "<": ">"}.get(op, op)
            col_id = self._qualify_column(right)
            return Predicate(
                column_id=col_id,
                op=flipped_op,
                value=str(left.this)
            )

        return None

    def _check_predicate_value(
        self,
        pred: Predicate,
        value_index: Any,
        result: ValidationResult
    ) -> None:
        """Check if predicate value exists in value_index."""
        # Skip non-equality predicates
        if pred.op not in ("=", "IN", "LOWER="):
            return

        # Skip if not a categorical column
        if not hasattr(value_index, "categorical_columns"):
            return

        if pred.column_id not in value_index.categorical_columns:
            return

        # Look up value
        matches = value_index.find_matching_values(pred.column_id, pred.value)
        if not matches:
            # WARNING, not error - allow ILIKE fallback
            result.warnings.append(
                f"Value '{pred.value}' not found in {pred.column_id}. "
                f"Consider using ILIKE or check spelling."
            )
            close_vals = value_index.get_close_values(pred.column_id, pred.value, limit=3)
            if close_vals:
                result.suggestions.append(f"Similar values: {', '.join(close_vals)}")
        elif hasattr(matches[0], "match_type") and matches[0].match_type != "exact":
            # Found fuzzy match - suggest correction
            result.suggestions.append(
                f"'{pred.value}' → '{matches[0].value}' ({matches[0].match_type} match)"
            )

    def _check_group_by(self, parsed: exp.Expression) -> List[str]:
        """Check GROUP BY completeness."""
        warnings = []

        # Find SELECT expressions and GROUP BY
        select_cols = []
        group_by_cols = []
        has_aggregate = False

        for node in parsed.walk():
            if isinstance(node, exp.Select):
                for expr in node.expressions:
                    if isinstance(expr, exp.Column):
                        select_cols.append(self._qualify_column(expr))
                    elif isinstance(expr, (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)):
                        has_aggregate = True

            if isinstance(node, exp.Group):
                for expr in node.expressions:
                    if isinstance(expr, exp.Column):
                        group_by_cols.append(self._qualify_column(expr))

        # If there's an aggregate but no GROUP BY, that's potentially okay (aggregate over all)
        # If there's GROUP BY, check all non-aggregate SELECT columns are in GROUP BY
        if group_by_cols and select_cols:
            missing = set(select_cols) - set(group_by_cols)
            if missing:
                warnings.append(
                    f"Columns in SELECT but not in GROUP BY: {missing}. "
                    f"This may cause an error."
                )

        return warnings

    def _find_closest(self, col_id: str, allowed: Set[str]) -> Optional[str]:
        """Find closest matching column name."""
        if not allowed:
            return None

        col_name = col_id.split(".")[-1].lower()

        # Try exact suffix match first
        for candidate in allowed:
            if candidate.split(".")[-1].lower() == col_name:
                return candidate

        # Try prefix match for prop_ columns
        if col_name.startswith("prop_"):
            for candidate in allowed:
                cand_name = candidate.split(".")[-1].lower()
                if cand_name.startswith("prop_") and col_name[5:10] == cand_name[5:10]:
                    return candidate

        return None


# Global instance
sql_validator = SQLValidator()
