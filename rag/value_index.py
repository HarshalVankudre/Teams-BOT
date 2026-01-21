"""
Value Index for Categorical Column Matching.

Indexes distinct values in categorical columns and provides fuzzy matching
to help correct user input (e.g., "Bomag" -> "BOMAG").

Phase 4 of the semantic schema linking system.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .postgres import PostgresService

logger = logging.getLogger(__name__)


@dataclass
class ValueMatch:
    """A matched value from the index."""
    value: str
    confidence: float = 1.0
    match_type: str = "exact"  # exact, case_insensitive, fuzzy, partial


@dataclass
class ColumnValues:
    """Indexed values for a single column."""
    column_id: str  # table.column
    values: Set[str] = field(default_factory=set)
    values_lower: Dict[str, str] = field(default_factory=dict)  # lower -> original
    value_count: int = 0


class ValueIndex:
    """
    Index of categorical column values for fuzzy matching.

    Provides:
    - Exact and case-insensitive matching
    - Fuzzy matching (edit distance)
    - Partial matching (substring)
    - Value suggestions for autocomplete
    """

    # Categorical columns to index (table.column)
    CATEGORICAL_COLUMNS = frozenset([
        "equipment_matrix.geraetegruppe_name",
        "equipment_matrix.hersteller_name",
        "equipment_matrix.hersteller_code",
        "equipment_matrix.verwendung_code",
        "equipment_matrix.verwendung_name",
        "equipment_matrix.nuclos_state",
    ])

    def __init__(self):
        self._columns: Dict[str, ColumnValues] = {}
        self._initialized = False

    @property
    def categorical_columns(self) -> Set[str]:
        """Return set of indexed categorical column IDs."""
        return set(self._columns.keys())

    def initialize(self, postgres: "PostgresService") -> None:
        """
        Load distinct values for all categorical columns from database.

        Args:
            postgres: PostgresService for database access
        """
        if self._initialized:
            return

        if not postgres.available:
            logger.warning("[ValueIndex] PostgreSQL not available")
            self._initialized = True
            return

        table = postgres.equipment_table
        total_values = 0

        for col_id in self.CATEGORICAL_COLUMNS:
            parts = col_id.split(".", 1)
            if len(parts) != 2:
                continue

            col_table, col_name = parts
            if col_table != table.split(".")[-1]:
                # Skip if table doesn't match
                continue

            try:
                # Get distinct values
                results = postgres.execute_query(
                    f"SELECT DISTINCT {col_name} FROM {table} "
                    f"WHERE {col_name} IS NOT NULL "
                    f"ORDER BY {col_name} "
                    f"LIMIT 1000"
                )

                column_values = ColumnValues(column_id=col_id)
                for row in (results or []):
                    val = row.get(col_name)
                    if val:
                        val_str = str(val).strip()
                        if val_str:
                            column_values.values.add(val_str)
                            column_values.values_lower[val_str.lower()] = val_str

                column_values.value_count = len(column_values.values)
                self._columns[col_id] = column_values
                total_values += column_values.value_count

                logger.debug(
                    f"[ValueIndex] Indexed {column_values.value_count} values for {col_id}"
                )

            except Exception as e:
                logger.warning(f"[ValueIndex] Failed to index {col_id}: {e}")

        logger.info(
            f"[ValueIndex] Initialized with {total_values} values "
            f"across {len(self._columns)} columns"
        )
        self._initialized = True

    def find_matching_values(
        self,
        column_id: str,
        search_value: str,
        max_results: int = 5,
    ) -> List[ValueMatch]:
        """
        Find matching values for a search term.

        Args:
            column_id: Fully-qualified column ID (table.column)
            search_value: Value to search for
            max_results: Maximum number of results

        Returns:
            List of ValueMatch objects, ordered by confidence
        """
        if column_id not in self._columns:
            return []

        col_values = self._columns[column_id]
        search_lower = (search_value or "").strip().lower()
        if not search_lower:
            return []

        matches: List[ValueMatch] = []

        # 1. Exact match
        if search_value in col_values.values:
            matches.append(ValueMatch(
                value=search_value,
                confidence=1.0,
                match_type="exact"
            ))
            return matches[:max_results]

        # 2. Case-insensitive match
        if search_lower in col_values.values_lower:
            matches.append(ValueMatch(
                value=col_values.values_lower[search_lower],
                confidence=0.95,
                match_type="case_insensitive"
            ))
            return matches[:max_results]

        # 3. Partial match (substring)
        for val in col_values.values:
            val_lower = val.lower()
            if search_lower in val_lower:
                matches.append(ValueMatch(
                    value=val,
                    confidence=0.8,
                    match_type="partial"
                ))
            elif val_lower in search_lower:
                matches.append(ValueMatch(
                    value=val,
                    confidence=0.75,
                    match_type="partial"
                ))

        # 4. Fuzzy match (edit distance)
        if not matches:
            fuzzy_matches = self._fuzzy_match(search_lower, col_values.values)
            for val, score in fuzzy_matches[:max_results]:
                if score > 0.6:
                    matches.append(ValueMatch(
                        value=val,
                        confidence=score,
                        match_type="fuzzy"
                    ))

        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches[:max_results]

    def get_close_values(
        self,
        column_id: str,
        search_value: str,
        limit: int = 5,
    ) -> List[str]:
        """
        Get close/similar values for suggestion.

        Args:
            column_id: Fully-qualified column ID
            search_value: Value to find similar values for
            limit: Maximum number of suggestions

        Returns:
            List of similar values
        """
        matches = self.find_matching_values(column_id, search_value, max_results=limit)
        return [m.value for m in matches]

    def get_all_values(self, column_id: str) -> List[str]:
        """Get all indexed values for a column."""
        if column_id not in self._columns:
            return []
        return sorted(self._columns[column_id].values)

    def get_value_hints(
        self,
        column_id: str,
        query: str,
        limit: int = 5,
    ) -> List[ValueMatch]:
        """
        Get value hints for a column based on query context.

        Extracts relevant terms from query and finds matching values.

        Args:
            column_id: Column to get hints for
            query: User query for context
            limit: Maximum hints to return

        Returns:
            List of ValueMatch hints
        """
        if column_id not in self._columns:
            return []

        col_values = self._columns[column_id]
        query_lower = (query or "").lower()

        # Extract potential value terms from query
        # Look for terms that might match categorical values
        terms = set(re.findall(r'\b\w{3,}\b', query_lower))

        hints: List[ValueMatch] = []
        seen = set()

        for term in terms:
            matches = self.find_matching_values(column_id, term, max_results=3)
            for match in matches:
                if match.value not in seen:
                    hints.append(match)
                    seen.add(match.value)

        # Sort by confidence
        hints.sort(key=lambda m: m.confidence, reverse=True)
        return hints[:limit]

    def _fuzzy_match(
        self,
        search: str,
        candidates: Set[str],
    ) -> List[tuple]:
        """
        Fuzzy match using simple edit distance ratio.

        Returns list of (value, score) tuples.
        """
        results = []
        for val in candidates:
            val_lower = val.lower()
            score = self._similarity(search, val_lower)
            if score > 0.5:
                results.append((val, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @staticmethod
    def _similarity(s1: str, s2: str) -> float:
        """
        Calculate similarity ratio between two strings.

        Simple implementation based on longest common subsequence.
        """
        if not s1 or not s2:
            return 0.0

        # Quick check for identical
        if s1 == s2:
            return 1.0

        # Calculate longest common subsequence ratio
        m, n = len(s1), len(s2)
        if m > n:
            s1, s2 = s2, s1
            m, n = n, m

        # Simple LCS-based similarity
        dp = [[0] * (n + 1) for _ in range(2)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + 1
                else:
                    dp[i % 2][j] = max(dp[(i - 1) % 2][j], dp[i % 2][j - 1])

        lcs_len = dp[m % 2][n]
        return (2.0 * lcs_len) / (m + n)


# Global singleton
value_index = ValueIndex()
