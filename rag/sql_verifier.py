"""
SQL Verification Module

Provides self-verification for SQL queries before execution.
Catches common errors and suggests corrections.
"""
import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of SQL verification."""
    is_valid: bool = True
    confidence: float = 1.0  # 0.0-1.0
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    corrected_sql: Optional[str] = None
    should_retry: bool = False


# Common SQL mistakes and their fixes
SQL_PATTERNS = {
    # Wrong column names
    r"\bkostenstelle\s*=": {
        "issue": "kostenstelle column doesn't exist, use ibs_nuclet_geraete_kostenstelle",
        "fix": lambda sql: re.sub(
            r"\bkostenstelle\s*=\s*'([^']+)'",
            r"ibs_nuclet_geraete_kostenstelle ILIKE '%\1%'",
            sql
        )
    },
    r"\bkostenstelle_code\b": {
        "issue": "kostenstelle_code doesn't exist, use ibs_nuclet_geraete_kostenstelle",
        "fix": lambda sql: re.sub(
            r"\bkostenstelle_code\s*=\s*'([^']+)'",
            r"ibs_nuclet_geraete_kostenstelle ILIKE '\1 -%'",
            sql
        )
    },
    # Category mistakes
    r"geraetegruppe_name\s+ilike\s+'%fertiger%'\s+and\s+prop_e2100": {
        "issue": "Don't combine geraetegruppe with prop_e2100 for Kettenfertiger/Radfertiger",
        "suggestion": "Use geraetegruppe_name = 'Kettenfertiger' or 'Radfertiger' directly"
    },
    # Missing quotes
    r"=\s+Released\b(?!')": {
        "issue": "String value 'Released' needs quotes",
        "fix": lambda sql: re.sub(r"=\s+Released\b", "= 'Released'", sql)
    },
    r"=\s+MIET\b(?!')": {
        "issue": "String value 'MIET' needs quotes",
        "fix": lambda sql: re.sub(r"=\s+MIET\b", "= 'MIET'", sql)
    },
}


class SQLVerifier:
    """
    Verifies SQL queries before execution.

    Performs both pattern-based checks and optional LLM-based verification
    for complex queries.
    """

    def __init__(
        self,
        equipment_table: str,
        column_resolver: Optional[callable] = None,
        provider=None,
        use_llm_verification: bool = False
    ):
        self.equipment_table = equipment_table
        self._column_resolver = column_resolver
        self._provider = provider
        self._use_llm = use_llm_verification
        self._cached_columns: Optional[Dict[str, str]] = None

    def _get_columns(self) -> Dict[str, str]:
        """Get available columns."""
        if self._cached_columns is not None:
            return self._cached_columns
        if self._column_resolver:
            try:
                self._cached_columns = self._column_resolver() or {}
            except Exception:
                self._cached_columns = {}
        else:
            self._cached_columns = {}
        return self._cached_columns

    def verify(
        self,
        sql: str,
        purpose: str = "",
        user_query: str = ""
    ) -> VerificationResult:
        """
        Verify a SQL query before execution.

        Args:
            sql: The SQL query to verify
            purpose: What the query is supposed to do
            user_query: The original user question

        Returns:
            VerificationResult with validation status and suggestions
        """
        result = VerificationResult()
        sql_lower = sql.lower()

        # Check for common patterns
        for pattern, info in SQL_PATTERNS.items():
            if re.search(pattern, sql, re.IGNORECASE):
                result.issues.append(info.get("issue", "Pattern match issue"))
                if "suggestion" in info:
                    result.suggestions.append(info["suggestion"])
                if "fix" in info:
                    try:
                        fixed = info["fix"](sql)
                        if fixed != sql:
                            result.corrected_sql = fixed
                            result.should_retry = True
                    except Exception as e:
                        logger.debug(f"Fix failed: {e}")

        # Check for unsafe patterns
        unsafe_patterns = [
            (r"\bdelete\b", "DELETE not allowed"),
            (r"\bupdate\b", "UPDATE not allowed"),
            (r"\binsert\b", "INSERT not allowed"),
            (r"\bdrop\b", "DROP not allowed"),
            (r"\btruncate\b", "TRUNCATE not allowed"),
            (r"\balter\b", "ALTER not allowed"),
        ]
        for pattern, msg in unsafe_patterns:
            if re.search(pattern, sql_lower):
                result.is_valid = False
                result.issues.append(msg)
                result.confidence = 0.0

        # Check column references
        columns = self._get_columns()
        if columns:
            referenced = self._extract_column_refs(sql)
            unknown = []
            for col in referenced:
                if col.lower() not in {c.lower() for c in columns.keys()}:
                    # Skip known SQL keywords and functions
                    if col.lower() not in {"count", "sum", "avg", "max", "min", "coalesce", "nullif", "cast", "as"}:
                        unknown.append(col)

            if unknown:
                result.issues.append(f"Unknown columns: {', '.join(unknown[:5])}")
                result.confidence = max(0.5, result.confidence - 0.2 * len(unknown))

        # Check for missing LIMIT on potentially large queries
        if "limit" not in sql_lower and "count(" not in sql_lower:
            if "select" in sql_lower and "from" in sql_lower:
                result.suggestions.append("Consider adding LIMIT to prevent large result sets")

        # Verify table reference
        if self.equipment_table:
            table_name = self.equipment_table.split(".")[-1].lower()
            if table_name not in sql_lower and "equipment" not in sql_lower:
                result.issues.append(f"Query doesn't reference expected table {self.equipment_table}")
                result.confidence *= 0.8

        # Calculate final validity
        if result.issues and not result.corrected_sql:
            result.confidence = max(0.3, result.confidence - 0.15 * len(result.issues))

        if result.confidence < 0.5:
            result.is_valid = False

        return result

    def _extract_column_refs(self, sql: str) -> List[str]:
        """Extract column references from SQL."""
        # Simple extraction - get words that look like column names
        tokens = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", sql)

        # Filter out SQL keywords
        keywords = {
            "select", "from", "where", "and", "or", "not", "in", "is", "null",
            "true", "false", "like", "ilike", "between", "case", "when", "then",
            "else", "end", "as", "on", "join", "left", "right", "inner", "outer",
            "group", "by", "order", "asc", "desc", "limit", "offset", "having",
            "distinct", "count", "sum", "avg", "max", "min", "coalesce", "nullif",
            "cast", "numeric", "integer", "text", "boolean", "double", "precision",
            "fetch", "first", "rows", "only", "with", "union", "all", "exists"
        }

        return [t for t in tokens if t.lower() not in keywords]

    async def verify_with_llm(
        self,
        sql: str,
        purpose: str,
        user_query: str
    ) -> VerificationResult:
        """
        Use LLM to verify SQL query correctness.

        Only used for complex queries where pattern matching isn't enough.
        """
        if not self._provider:
            return self.verify(sql, purpose, user_query)

        # First do pattern-based verification
        result = self.verify(sql, purpose, user_query)

        # If pattern check found serious issues, don't bother with LLM
        if not result.is_valid:
            return result

        # Use LLM for semantic verification
        from .providers import ChatMessage

        prompt = f"""Verify this SQL query for a SEMA equipment database.

User Question: {user_query}
Query Purpose: {purpose}
SQL: {sql}

Check for:
1. Does the SQL answer the user's question?
2. Are column names correct? (Use hersteller_name not hersteller, verwendung_code not verwendung)
3. For equipment categories (Kettenfertiger, Radfertiger, etc.), is geraetegruppe_name used correctly?
4. Are string comparisons using proper operators (= for exact, ILIKE for partial)?

Respond with JSON only:
{{"is_valid": true/false, "issues": ["issue1", "issue2"], "suggestions": ["suggestion1"]}}"""

        try:
            response = await self._provider.chat_completion(
                messages=[ChatMessage(role="user", content=prompt)],
                tools=None,
                max_tokens=200
            )

            import json
            content = response.content or "{}"
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()

            data = json.loads(content)

            if not data.get("is_valid", True):
                result.is_valid = False
                result.confidence = 0.4

            result.issues.extend(data.get("issues", []))
            result.suggestions.extend(data.get("suggestions", []))

        except Exception as e:
            logger.debug(f"LLM verification failed: {e}")

        return result
