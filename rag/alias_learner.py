"""
Alias Learner for Column/Value Mappings.

Learns column and value mappings from successful queries.
Separate from learned_rules.py which handles general feedback rules.

Phase 5 of the semantic schema linking system.

Tables (to be created in admin DB):
- learned_column_aliases: Maps user terms to column names
- learned_value_aliases: Maps user terms to DB values
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ColumnAlias:
    """A learned column alias."""
    id: int
    alias_text: str
    target_table: str
    target_column: str
    confidence: float
    use_count: int
    success_count: int
    failure_count: int


@dataclass
class ValueAlias:
    """A learned value alias."""
    id: int
    alias_text: str
    target_table: str
    target_column: str
    target_value: str
    confidence: float
    use_count: int
    success_count: int
    failure_count: int


class AliasLearner:
    """
    Learn column/value mappings from successful queries.

    GATING RULES (from design review):
    - Create aliases at low confidence (0.55)
    - Only USE aliases when:
      - use_count >= 3 AND
      - confidence >= 0.75 AND
      - failure_count < success_count
    """

    # Thresholds for using aliases
    MIN_USE_COUNT = 3
    MIN_CONFIDENCE = 0.75

    # Initial confidence for new aliases
    INITIAL_CONFIDENCE = 0.55

    def __init__(self):
        self._db = None
        self._initialized = False

    def _get_db(self):
        """Lazy-load admin database connection."""
        if self._db is None:
            try:
                from .admin_logger import admin_logger
                if admin_logger.available:
                    self._db = admin_logger
                    self._ensure_tables()
            except Exception as e:
                logger.warning(f"[AliasLearner] Could not connect to admin DB: {e}")
        return self._db

    def _ensure_tables(self) -> None:
        """Ensure alias tables exist in admin database."""
        if self._initialized:
            return

        db = self._get_db()
        if not db or not db.available:
            return

        try:
            # Create column aliases table
            db._execute_query("""
                CREATE TABLE IF NOT EXISTS learned_column_aliases (
                    id SERIAL PRIMARY KEY,
                    alias_text VARCHAR(255) NOT NULL,
                    target_table VARCHAR(255) NOT NULL,
                    target_column VARCHAR(255) NOT NULL,
                    confidence FLOAT DEFAULT 0.55,
                    use_count INT DEFAULT 1,
                    success_count INT DEFAULT 1,
                    failure_count INT DEFAULT 0,
                    source VARCHAR(50) DEFAULT 'auto',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(alias_text, target_table, target_column)
                )
            """)

            # Create value aliases table
            db._execute_query("""
                CREATE TABLE IF NOT EXISTS learned_value_aliases (
                    id SERIAL PRIMARY KEY,
                    alias_text VARCHAR(255) NOT NULL,
                    target_table VARCHAR(255) NOT NULL,
                    target_column VARCHAR(255) NOT NULL,
                    target_value VARCHAR(500) NOT NULL,
                    confidence FLOAT DEFAULT 0.55,
                    use_count INT DEFAULT 1,
                    success_count INT DEFAULT 1,
                    failure_count INT DEFAULT 0,
                    source VARCHAR(50) DEFAULT 'auto',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(alias_text, target_table, target_column, target_value)
                )
            """)

            # Create indexes
            db._execute_query("""
                CREATE INDEX IF NOT EXISTS idx_col_aliases_text
                ON learned_column_aliases(LOWER(alias_text))
            """)
            db._execute_query("""
                CREATE INDEX IF NOT EXISTS idx_val_aliases_text
                ON learned_value_aliases(LOWER(alias_text))
            """)

            self._initialized = True
            logger.info("[AliasLearner] Tables initialized")

        except Exception as e:
            logger.warning(f"[AliasLearner] Could not create tables: {e}")

    def get_usable_column_aliases(self, term: str) -> List[ColumnAlias]:
        """
        Get column aliases that pass gating threshold.

        Args:
            term: User term to look up

        Returns:
            List of ColumnAlias that meet the gating criteria
        """
        db = self._get_db()
        if not db or not db.available:
            return []

        try:
            results = db._execute_query("""
                SELECT id, alias_text, target_table, target_column,
                       confidence, use_count, success_count, failure_count
                FROM learned_column_aliases
                WHERE LOWER(alias_text) = LOWER(%s)
                  AND use_count >= %s
                  AND confidence >= %s
                  AND failure_count < success_count
                ORDER BY confidence DESC
                LIMIT 3
            """, [term, self.MIN_USE_COUNT, self.MIN_CONFIDENCE])

            return [
                ColumnAlias(
                    id=row["id"],
                    alias_text=row["alias_text"],
                    target_table=row["target_table"],
                    target_column=row["target_column"],
                    confidence=row["confidence"],
                    use_count=row["use_count"],
                    success_count=row["success_count"],
                    failure_count=row["failure_count"],
                )
                for row in (results or [])
            ]
        except Exception as e:
            logger.warning(f"[AliasLearner] Error getting column aliases: {e}")
            return []

    def get_usable_value_aliases(self, term: str) -> List[ValueAlias]:
        """
        Get value aliases that pass gating threshold.

        Args:
            term: User term to look up

        Returns:
            List of ValueAlias that meet the gating criteria
        """
        db = self._get_db()
        if not db or not db.available:
            return []

        try:
            results = db._execute_query("""
                SELECT id, alias_text, target_table, target_column, target_value,
                       confidence, use_count, success_count, failure_count
                FROM learned_value_aliases
                WHERE LOWER(alias_text) = LOWER(%s)
                  AND use_count >= %s
                  AND confidence >= %s
                  AND failure_count < success_count
                ORDER BY confidence DESC
                LIMIT 3
            """, [term, self.MIN_USE_COUNT, self.MIN_CONFIDENCE])

            return [
                ValueAlias(
                    id=row["id"],
                    alias_text=row["alias_text"],
                    target_table=row["target_table"],
                    target_column=row["target_column"],
                    target_value=row["target_value"],
                    confidence=row["confidence"],
                    use_count=row["use_count"],
                    success_count=row["success_count"],
                    failure_count=row["failure_count"],
                )
                for row in (results or [])
            ]
        except Exception as e:
            logger.warning(f"[AliasLearner] Error getting value aliases: {e}")
            return []

    def learn_column_alias(
        self,
        alias_text: str,
        target_table: str,
        target_column: str,
        source: str = "auto"
    ) -> bool:
        """
        Learn a new column alias (or increment existing).

        Args:
            alias_text: User term
            target_table: Target table name
            target_column: Target column name
            source: Source of learning (auto, manual, feedback)

        Returns:
            True if saved/updated successfully
        """
        db = self._get_db()
        if not db or not db.available:
            return False

        try:
            # Upsert: insert or update if exists
            db._execute_query("""
                INSERT INTO learned_column_aliases
                    (alias_text, target_table, target_column, source)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (alias_text, target_table, target_column)
                DO UPDATE SET
                    use_count = learned_column_aliases.use_count + 1,
                    updated_at = NOW()
            """, [alias_text, target_table, target_column, source])

            logger.info(
                f"[AliasLearner] Learned column alias: "
                f"{alias_text} -> {target_table}.{target_column}"
            )
            return True

        except Exception as e:
            logger.warning(f"[AliasLearner] Error learning column alias: {e}")
            return False

    def learn_value_alias(
        self,
        alias_text: str,
        target_table: str,
        target_column: str,
        target_value: str,
        source: str = "auto"
    ) -> bool:
        """
        Learn a new value alias (or increment existing).

        Args:
            alias_text: User term
            target_table: Target table name
            target_column: Target column name
            target_value: Actual DB value
            source: Source of learning

        Returns:
            True if saved/updated successfully
        """
        db = self._get_db()
        if not db or not db.available:
            return False

        try:
            db._execute_query("""
                INSERT INTO learned_value_aliases
                    (alias_text, target_table, target_column, target_value, source)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (alias_text, target_table, target_column, target_value)
                DO UPDATE SET
                    use_count = learned_value_aliases.use_count + 1,
                    updated_at = NOW()
            """, [alias_text, target_table, target_column, target_value, source])

            logger.info(
                f"[AliasLearner] Learned value alias: "
                f"{alias_text} -> {target_table}.{target_column}={target_value}"
            )
            return True

        except Exception as e:
            logger.warning(f"[AliasLearner] Error learning value alias: {e}")
            return False

    def record_column_outcome(self, alias_id: int, success: bool) -> bool:
        """
        Update column alias based on query outcome.

        Args:
            alias_id: Database ID of the alias
            success: Whether the query succeeded

        Returns:
            True if updated successfully
        """
        db = self._get_db()
        if not db or not db.available:
            return False

        try:
            if success:
                db._execute_query("""
                    UPDATE learned_column_aliases
                    SET success_count = success_count + 1,
                        use_count = use_count + 1,
                        confidence = LEAST(0.95, confidence + 0.05 * (1 - confidence)),
                        updated_at = NOW()
                    WHERE id = %s
                """, [alias_id])
            else:
                db._execute_query("""
                    UPDATE learned_column_aliases
                    SET failure_count = failure_count + 1,
                        use_count = use_count + 1,
                        confidence = GREATEST(0.1, confidence - 0.1),
                        updated_at = NOW()
                    WHERE id = %s
                """, [alias_id])

            return True

        except Exception as e:
            logger.warning(f"[AliasLearner] Error recording outcome: {e}")
            return False

    def record_value_outcome(self, alias_id: int, success: bool) -> bool:
        """
        Update value alias based on query outcome.

        Args:
            alias_id: Database ID of the alias
            success: Whether the query succeeded

        Returns:
            True if updated successfully
        """
        db = self._get_db()
        if not db or not db.available:
            return False

        try:
            if success:
                db._execute_query("""
                    UPDATE learned_value_aliases
                    SET success_count = success_count + 1,
                        use_count = use_count + 1,
                        confidence = LEAST(0.95, confidence + 0.05 * (1 - confidence)),
                        updated_at = NOW()
                    WHERE id = %s
                """, [alias_id])
            else:
                db._execute_query("""
                    UPDATE learned_value_aliases
                    SET failure_count = failure_count + 1,
                        use_count = use_count + 1,
                        confidence = GREATEST(0.1, confidence - 0.1),
                        updated_at = NOW()
                    WHERE id = %s
                """, [alias_id])

            return True

        except Exception as e:
            logger.warning(f"[AliasLearner] Error recording outcome: {e}")
            return False

    def get_all_column_aliases(self, limit: int = 100) -> List[ColumnAlias]:
        """Get all column aliases (for admin dashboard)."""
        db = self._get_db()
        if not db or not db.available:
            return []

        try:
            results = db._execute_query("""
                SELECT id, alias_text, target_table, target_column,
                       confidence, use_count, success_count, failure_count
                FROM learned_column_aliases
                ORDER BY use_count DESC
                LIMIT %s
            """, [limit])

            return [
                ColumnAlias(
                    id=row["id"],
                    alias_text=row["alias_text"],
                    target_table=row["target_table"],
                    target_column=row["target_column"],
                    confidence=row["confidence"],
                    use_count=row["use_count"],
                    success_count=row["success_count"],
                    failure_count=row["failure_count"],
                )
                for row in (results or [])
            ]
        except Exception as e:
            logger.warning(f"[AliasLearner] Error getting all aliases: {e}")
            return []

    def get_all_value_aliases(self, limit: int = 100) -> List[ValueAlias]:
        """Get all value aliases (for admin dashboard)."""
        db = self._get_db()
        if not db or not db.available:
            return []

        try:
            results = db._execute_query("""
                SELECT id, alias_text, target_table, target_column, target_value,
                       confidence, use_count, success_count, failure_count
                FROM learned_value_aliases
                ORDER BY use_count DESC
                LIMIT %s
            """, [limit])

            return [
                ValueAlias(
                    id=row["id"],
                    alias_text=row["alias_text"],
                    target_table=row["target_table"],
                    target_column=row["target_column"],
                    target_value=row["target_value"],
                    confidence=row["confidence"],
                    use_count=row["use_count"],
                    success_count=row["success_count"],
                    failure_count=row["failure_count"],
                )
                for row in (results or [])
            ]
        except Exception as e:
            logger.warning(f"[AliasLearner] Error getting all value aliases: {e}")
            return []


# Global singleton
alias_learner = AliasLearner()
