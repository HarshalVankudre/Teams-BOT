"""
SQL Generator SubAgent

Generates and executes SQL queries against the PostgreSQL database.
Uses a fast, non-reasoning model for efficiency.

Features:
    - Automatic SQL generation from natural language
    - Query validation and sanitization
    - Fallback query generation on errors
    - Query explanation and optimization suggestions
    - Schema introspection tools
    - Result formatting and aggregation

Author: RÜKO GmbH Baumaschinen
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Final, TypeAlias

from openai import AsyncOpenAI

from .interface import (
    AgentCapability,
    AgentContext,
    AgentMetadata,
    AgentResponse,
    AgentType,
    RetryConfig,
    RetryStrategy,
    SubAgentBase,
    register_subagent,
    tool,
)
from ...config import config
from ...postgres import PostgresService, postgres_service
from ...schema import SQL_AGENT_SCHEMA, SQL_SPECIAL_RULES

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Type aliases
SQLResult: TypeAlias = list[dict[str, Any]]
ColumnStats: TypeAlias = dict[str, dict[str, Any]]


class QueryType(str, Enum):
    """Types of SQL queries."""
    
    SELECT = "select"
    COUNT = "count"
    AGGREGATE = "aggregate"
    UNKNOWN = "unknown"


class SQLValidationError(Exception):
    """Raised when SQL validation fails."""
    
    def __init__(self, query: str, reason: str):
        self.query = query
        self.reason = reason
        super().__init__(f"Invalid SQL: {reason}")


@dataclass(frozen=True, slots=True)
class SQLAgentConfig:
    """Configuration for the SQL agent."""
    
    # Model settings
    default_model: str = "gpt-4o-mini"
    max_retries: int = 2
    
    # Query limits
    max_results_default: int = 1000
    max_query_length: int = 5000
    
    # Validation
    allowed_operations: frozenset[str] = frozenset({"select"})
    dangerous_patterns: tuple[str, ...] = (
        r"\bdrop\b",
        r"\bdelete\b",
        r"\btruncate\b",
        r"\bupdate\b",
        r"\binsert\b",
        r"\balter\b",
        r"\bcreate\b",
        r"--",
        r";.*;",
    )


DEFAULT_CONFIG: Final[SQLAgentConfig] = SQLAgentConfig()


@dataclass(slots=True)
class QueryResult:
    """Result of a SQL query execution."""
    
    success: bool
    query: str
    results: SQLResult
    row_count: int
    execution_time_ms: float
    error: str | None = None
    explanation: str | None = None
    is_fallback: bool = False
    query_type: QueryType = QueryType.UNKNOWN
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "query": self.query,
            "results": self.results,
            "row_count": self.row_count,
            "execution_time_ms": self.execution_time_ms,
            "error": self.error,
            "explanation": self.explanation,
            "is_fallback": self.is_fallback,
            "query_type": self.query_type.value,
        }


# Agent metadata
SQL_AGENT_METADATA: Final[AgentMetadata] = AgentMetadata(
    agent_id="sql",
    name="SQL Generator Agent",
    description="Generates and executes SQL queries against the equipment database",
    detailed_description="""Generiert und führt SQL-Abfragen gegen die PostgreSQL-Datenbank aus.
Verwende diesen Agenten für:
- Zählung von Geräten (z.B. "Wie viele Bagger haben wir?")
- Filterung nach Eigenschaften (z.B. "Bagger über 15 Tonnen")
- Aggregationen (z.B. "Durchschnittsgewicht der Mobilbagger")
- Vergleiche zwischen Gerätegruppen
- Suche nach Seriennummer oder Inventarnummer
- Statistische Analysen (Min, Max, Durchschnitt)
- Gruppierungen und Sortierungen""",
    capabilities=[AgentCapability.DATABASE_QUERY, AgentCapability.DATA_ANALYSIS],
    uses_reasoning=False,
    default_model=DEFAULT_CONFIG.default_model,
    parameters={
        "sql_task": {
            "type": "string",
            "description": "Beschreibung der SQL-Aufgabe, z.B. 'Zähle alle Bagger im Bestand'"
        },
        "sql_filters": {
            "type": "object",
            "description": "Optionale Filter wie {'hersteller': 'Liebherr', 'min_gewicht': 10000}"
        },
        "limit": {
            "type": "integer",
            "description": "Maximale Anzahl der Ergebnisse (optional)"
        }
    },
    example_queries=[
        "Wie viele Bagger haben wir?",
        "Liste alle Kettenbagger von Liebherr",
        "Welche Geräte haben GPS?",
        "Durchschnittsgewicht der Mobilbagger",
        "Vergleich Kettenbagger vs Mobilbagger",
        "Top 10 schwerste Maschinen",
        "Alle Geräte mit Klimaanlage sortiert nach Gewicht",
    ],
    priority=10,
)


@register_subagent()
class SQLGeneratorAgent(SubAgentBase):
    """
    Generates SQL queries based on task descriptions and executes them.
    
    Uses tool-calling to generate valid SQL for the equipment database.
    Includes automatic retry logic and fallback query generation.
    
    Attributes:
        client: AsyncOpenAI client for LLM calls
        model: Model to use for SQL generation
        postgres: PostgreSQL service for query execution
        agent_config: Configuration settings
    """

    METADATA = SQL_AGENT_METADATA

    # System prompt uses schema from centralized schema.py
    _SYSTEM_PROMPT: Final[str] = f"""Du bist ein SQL-Generator für eine PostgreSQL-Datenbank mit Baumaschinen.

{SQL_AGENT_SCHEMA}

REGELN:
1. Generiere NUR SELECT-Abfragen
2. Verwende IMMER korrekte JSONB-Syntax
3. Bei numerischen Vergleichen: Prüfe IMMER auf gültiges Zahlenformat mit Regex
4. KEIN LIMIT bei Zähl- oder Auflistungsanfragen ("alle", "welche", "wie viele", "liste")
   - Nur bei "zeige einige/ein paar/Beispiele" ein kleines LIMIT (10-20) verwenden
5. Bei Boolean-Eigenschaften: 'true', 'false', oder 'nicht-vorhanden'
6. Verwende ILIKE für case-insensitive Textsuche
7. Bei AVG/SUM: Filtere 'nicht-vorhanden' und leere Werte aus
8. VERWENDE geraetegruppe ILIKE '%bagger%' für Gerätetypen, NICHT kategorie!
9. FÜGE KEINE Filter für 'verwendung' hinzu, außer explizit angefragt!

{SQL_SPECIAL_RULES}

KRITISCH:
- Für Gerätetypen (Bagger, Walzen, etc.): IMMER geraetegruppe verwenden!
- kategorie ist oft NULL und unvollständig - NUR als Fallback!
- Keine unnötigen Filter hinzufügen!

WICHTIG:
- Gib das SQL als einzeiligen String zurück
- Keine Kommentare im SQL
- Keine Erklärungen, nur das SQL"""

    # OpenAI tool definitions
    _OPENAI_TOOLS: Final[list[dict[str, Any]]] = [
        {
            "type": "function",
            "function": {
                "name": "execute_sql",
                "description": "Execute a SQL query against the geraete table",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "The PostgreSQL SELECT query to execute"
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Brief explanation of what this query does"
                        }
                    },
                    "required": ["sql"]
                }
            }
        }
    ]

    __slots__ = ("client", "model", "postgres", "agent_config", "_query_cache")

    def __init__(
        self,
        openai_client: AsyncOpenAI | None = None,
        model: str | None = None,
        postgres: PostgresService | None = None,
        verbose: bool = False,
        agent_config: SQLAgentConfig | None = None,
    ) -> None:
        """
        Initialize the SQL generator agent.
        
        Args:
            openai_client: Optional pre-configured OpenAI client
            model: Model to use for SQL generation
            postgres: PostgreSQL service instance
            verbose: Enable verbose logging
            agent_config: Optional custom configuration
        """
        super().__init__(verbose=verbose)
        
        self._agent_type = AgentType.SQL_GENERATOR
        self.agent_config = agent_config or DEFAULT_CONFIG
        
        self.client = openai_client or AsyncOpenAI(api_key=config.openai_api_key)
        self.model = model or config.chunking_model or self.agent_config.default_model
        self.postgres = postgres or postgres_service
        
        # Simple query cache for repeated queries
        self._query_cache: dict[str, QueryResult] = {}
        
        logger.debug(
            "Initialized SQLGeneratorAgent with model=%s",
            self.model,
        )

    # ==================== TOOLS ====================

    @tool(
        name="execute_sql",
        description="Execute a SQL query against the equipment database",
        parameters={
            "sql": {"type": "string", "description": "The SQL SELECT query to execute"}
        },
        required=["sql"],
        retry_config=RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            max_attempts=2,
            base_delay_seconds=0.5,
        ),
        tags=["database", "query"],
    )
    async def execute_sql_tool(self, sql: str) -> dict[str, Any]:
        """
        Execute a SQL query and return results.
        
        Args:
            sql: The SQL query to execute
            
        Returns:
            Dict with success status, results, and row count
        """
        try:
            # Validate query first
            self._validate_query(sql)
            
            start_time = time.perf_counter()
            results = self.postgres.execute_dynamic_sql(sql)
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return {
                "success": True,
                "results": results,
                "row_count": len(results),
                "execution_time_ms": execution_time,
            }
        except SQLValidationError as e:
            logger.warning("SQL validation failed: %s", e.reason)
            return {
                "success": False,
                "error": f"Validation error: {e.reason}",
                "results": [],
                "row_count": 0,
            }
        except Exception as e:
            logger.error("SQL execution error: %s", str(e))
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "row_count": 0,
            }

    @tool(
        name="get_schema_info",
        description="Get database schema information for query building",
        parameters={},
        tags=["database", "schema"],
    )
    async def get_schema_info_tool(self) -> dict[str, Any]:
        """Return schema information for the database."""
        return {
            "schema": SQL_AGENT_SCHEMA,
            "table": "geraete",
            "primary_key": "inventarnummer",
        }

    @tool(
        name="get_column_stats",
        description="Get statistics for a specific column (min, max, distinct values)",
        parameters={
            "column_name": {"type": "string", "description": "Name of the column"},
            "is_jsonb": {"type": "boolean", "description": "Whether column is in JSONB properties"}
        },
        required=["column_name"],
        tags=["database", "statistics"],
    )
    async def get_column_stats_tool(
        self,
        column_name: str,
        is_jsonb: bool = True,
    ) -> dict[str, Any]:
        """
        Get statistics for a database column.
        
        Args:
            column_name: Name of the column to analyze
            is_jsonb: Whether the column is in the JSONB properties field
            
        Returns:
            Dict with column statistics
        """
        try:
            if is_jsonb:
                # JSONB column
                sql = f"""
                SELECT 
                    COUNT(*) as total_count,
                    COUNT(DISTINCT properties->>'{column_name}') as distinct_count,
                    COUNT(*) FILTER (WHERE properties->>'{column_name}' IS NOT NULL 
                        AND properties->>'{column_name}' != 'nicht-vorhanden') as non_null_count
                FROM geraete
                """
            else:
                # Regular column
                sql = f"""
                SELECT 
                    COUNT(*) as total_count,
                    COUNT(DISTINCT {column_name}) as distinct_count,
                    COUNT(*) FILTER (WHERE {column_name} IS NOT NULL) as non_null_count
                FROM geraete
                """
            
            results = self.postgres.execute_dynamic_sql(sql)
            
            if results:
                return {
                    "success": True,
                    "column": column_name,
                    "is_jsonb": is_jsonb,
                    "stats": results[0],
                }
            
            return {"success": False, "error": "No results", "column": column_name}
            
        except Exception as e:
            return {"success": False, "error": str(e), "column": column_name}

    @tool(
        name="get_distinct_values",
        description="Get distinct values for a column (useful for filters)",
        parameters={
            "column_name": {"type": "string", "description": "Name of the column"},
            "is_jsonb": {"type": "boolean", "description": "Whether column is in JSONB properties"},
            "limit": {"type": "integer", "description": "Max number of values to return"}
        },
        required=["column_name"],
        tags=["database", "exploration"],
    )
    async def get_distinct_values_tool(
        self,
        column_name: str,
        is_jsonb: bool = True,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        Get distinct values for a column.
        
        Args:
            column_name: Name of the column
            is_jsonb: Whether it's a JSONB property
            limit: Maximum values to return
            
        Returns:
            Dict with distinct values and counts
        """
        try:
            if is_jsonb:
                sql = f"""
                SELECT 
                    properties->>'{column_name}' as value,
                    COUNT(*) as count
                FROM geraete
                WHERE properties->>'{column_name}' IS NOT NULL
                    AND properties->>'{column_name}' != 'nicht-vorhanden'
                    AND properties->>'{column_name}' != ''
                GROUP BY properties->>'{column_name}'
                ORDER BY count DESC
                LIMIT {limit}
                """
            else:
                sql = f"""
                SELECT 
                    {column_name}::text as value,
                    COUNT(*) as count
                FROM geraete
                WHERE {column_name} IS NOT NULL
                GROUP BY {column_name}
                ORDER BY count DESC
                LIMIT {limit}
                """
            
            results = self.postgres.execute_dynamic_sql(sql)
            
            return {
                "success": True,
                "column": column_name,
                "values": results,
                "count": len(results),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "column": column_name, "values": []}

    @tool(
        name="explain_query",
        description="Get execution plan for a SQL query (for optimization)",
        parameters={
            "sql": {"type": "string", "description": "The SQL query to explain"}
        },
        required=["sql"],
        tags=["database", "optimization"],
    )
    async def explain_query_tool(self, sql: str) -> dict[str, Any]:
        """
        Get the execution plan for a query.
        
        Args:
            sql: The SQL query to explain
            
        Returns:
            Dict with execution plan details
        """
        try:
            self._validate_query(sql)
            explain_sql = f"EXPLAIN (FORMAT JSON) {sql}"
            results = self.postgres.execute_dynamic_sql(explain_sql)
            
            return {
                "success": True,
                "plan": results,
                "query": sql,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "query": sql}

    @tool(
        name="count_by_group",
        description="Count records grouped by a column",
        parameters={
            "group_column": {"type": "string", "description": "Column to group by"},
            "is_jsonb": {"type": "boolean", "description": "Whether column is in JSONB"},
            "filter_column": {"type": "string", "description": "Optional filter column"},
            "filter_value": {"type": "string", "description": "Optional filter value"}
        },
        required=["group_column"],
        tags=["database", "aggregation"],
    )
    async def count_by_group_tool(
        self,
        group_column: str,
        is_jsonb: bool = True,
        filter_column: str | None = None,
        filter_value: str | None = None,
    ) -> dict[str, Any]:
        """
        Count records grouped by a column.
        
        Args:
            group_column: Column to group by
            is_jsonb: Whether it's a JSONB property
            filter_column: Optional column to filter on
            filter_value: Optional value to filter by
            
        Returns:
            Dict with grouped counts
        """
        try:
            if is_jsonb:
                group_expr = f"properties->>'{group_column}'"
            else:
                group_expr = group_column
            
            where_clause = ""
            if filter_column and filter_value:
                if is_jsonb:
                    where_clause = f"WHERE properties->>'{filter_column}' ILIKE '%{filter_value}%'"
                else:
                    where_clause = f"WHERE {filter_column} ILIKE '%{filter_value}%'"
            
            sql = f"""
            SELECT 
                {group_expr} as group_value,
                COUNT(*) as count
            FROM geraete
            {where_clause}
            GROUP BY {group_expr}
            HAVING {group_expr} IS NOT NULL AND {group_expr} != ''
            ORDER BY count DESC
            """
            
            results = self.postgres.execute_dynamic_sql(sql)
            
            return {
                "success": True,
                "group_column": group_column,
                "results": results,
                "total_groups": len(results),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "group_column": group_column}

    @tool(
        name="numeric_stats",
        description="Get numeric statistics (min, max, avg, sum) for a column",
        parameters={
            "column_name": {"type": "string", "description": "Numeric column name"},
            "filter_type": {"type": "string", "description": "Optional: filter by geraetegruppe"}
        },
        required=["column_name"],
        tags=["database", "statistics"],
    )
    async def numeric_stats_tool(
        self,
        column_name: str,
        filter_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Get numeric statistics for a column.
        
        Args:
            column_name: Name of the numeric column
            filter_type: Optional filter by equipment type
            
        Returns:
            Dict with numeric statistics
        """
        try:
            where_clause = ""
            if filter_type:
                where_clause = f"AND properties->>'geraetegruppe' ILIKE '%{filter_type}%'"
            
            sql = f"""
            SELECT 
                COUNT(*) as count,
                MIN((properties->>'{column_name}')::numeric) as min_value,
                MAX((properties->>'{column_name}')::numeric) as max_value,
                AVG((properties->>'{column_name}')::numeric)::numeric(10,2) as avg_value,
                SUM((properties->>'{column_name}')::numeric)::numeric(10,2) as sum_value
            FROM geraete
            WHERE properties->>'{column_name}' IS NOT NULL
                AND properties->>'{column_name}' != 'nicht-vorhanden'
                AND properties->>'{column_name}' ~ '^[0-9]+\\.?[0-9]*$'
                {where_clause}
            """
            
            results = self.postgres.execute_dynamic_sql(sql)
            
            if results:
                return {
                    "success": True,
                    "column": column_name,
                    "filter": filter_type,
                    "stats": results[0],
                }
            
            return {"success": False, "error": "No numeric data found", "column": column_name}
            
        except Exception as e:
            return {"success": False, "error": str(e), "column": column_name}

    # ==================== VALIDATION ====================

    def _validate_query(self, sql: str) -> None:
        """
        Validate a SQL query for safety.
        
        Args:
            sql: The SQL query to validate
            
        Raises:
            SQLValidationError: If validation fails
        """
        sql_lower = sql.lower().strip()
        
        # Check length
        if len(sql) > self.agent_config.max_query_length:
            raise SQLValidationError(sql, "Query exceeds maximum length")
        
        # Check it starts with SELECT
        if not sql_lower.startswith("select"):
            raise SQLValidationError(sql, "Only SELECT queries are allowed")
        
        # Check for dangerous patterns
        for pattern in self.agent_config.dangerous_patterns:
            if re.search(pattern, sql_lower, re.IGNORECASE):
                raise SQLValidationError(sql, f"Dangerous pattern detected: {pattern}")

    def _determine_query_type(self, sql: str) -> QueryType:
        """Determine the type of SQL query."""
        sql_lower = sql.lower()
        
        if "count(" in sql_lower:
            return QueryType.COUNT
        elif any(agg in sql_lower for agg in ["avg(", "sum(", "min(", "max("]):
            return QueryType.AGGREGATE
        elif sql_lower.startswith("select"):
            return QueryType.SELECT
        return QueryType.UNKNOWN

    # ==================== MAIN EXECUTION ====================

    async def _execute(self, context: AgentContext) -> AgentResponse:
        """
        Generate and execute SQL based on the task description from orchestrator.
        
        Args:
            context: Agent context with metadata from orchestrator
            
        Returns:
            AgentResponse with query results
        """
        # Extract task from orchestrator instruction
        task_description = context.metadata.get("sql_task")
        filters = context.metadata.get("sql_filters", {})
        limit = context.metadata.get("limit")

        if not task_description:
            return AgentResponse.error_response(
                error="No task description provided by orchestrator",
                agent_type=self._agent_type,
            )

        self.log(f"Generating SQL for: {task_description[:50]}...")
        logger.info("SQL generation task: %s", task_description[:100])

        # Build the prompt
        prompt = self._build_prompt(task_description, filters, limit)
        
        # Generate SQL via LLM
        query_result = await self._generate_and_execute_sql(prompt, task_description)
        
        if not query_result.success:
            # Try fallback
            fallback_result = await self._try_fallback_query(
                task_description,
                query_result.error or "Unknown error",
            )
            
            if fallback_result and fallback_result.success:
                query_result = fallback_result
            else:
                return AgentResponse.error_response(
                    error=f"SQL execution failed: {query_result.error}",
                    agent_type=self._agent_type,
                )

        # Store results in context for reviewer
        context.sql_results = query_result.results

        return AgentResponse.success_response(
            data=query_result.to_dict(),
            agent_type=self._agent_type,
            reasoning=query_result.explanation,
            tool_calls=[{"name": "execute_sql", "sql": query_result.query}],
            sources=[{
                "type": "postgresql",
                "query": query_result.query,
                "row_count": query_result.row_count,
            }],
        )

    def _build_prompt(
        self,
        task_description: str,
        filters: dict[str, Any],
        limit: int | None,
    ) -> str:
        """Build the prompt for SQL generation."""
        prompt = f"Generiere eine SQL-Abfrage für folgende Anfrage:\n\n{task_description}"

        if filters:
            prompt += f"\n\nZusätzliche Filter: {json.dumps(filters, ensure_ascii=False)}"
        
        if limit:
            prompt += f"\n\nMaximale Ergebnisse: {limit}"

        return prompt

    async def _generate_and_execute_sql(
        self,
        prompt: str,
        task_description: str,
    ) -> QueryResult:
        """
        Generate SQL using LLM and execute it.
        
        Args:
            prompt: The prompt for SQL generation
            task_description: Original task description
            
        Returns:
            QueryResult with execution details
        """
        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self._OPENAI_TOOLS,
                tool_choice={"type": "function", "function": {"name": "execute_sql"}},
            )

            message = response.choices[0].message

            if not message.tool_calls:
                return QueryResult(
                    success=False,
                    query="",
                    results=[],
                    row_count=0,
                    execution_time_ms=0,
                    error="No SQL generated",
                )

            # Extract SQL from tool call
            tool_call = message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            sql = args.get("sql", "")
            explanation = args.get("explanation", "")

            if not sql:
                return QueryResult(
                    success=False,
                    query="",
                    results=[],
                    row_count=0,
                    execution_time_ms=0,
                    error="Empty SQL query",
                )

            self.log(f"Generated SQL: {sql[:100]}...")
            logger.debug("Generated SQL: %s", sql)

            # Execute the SQL
            result = await self.call_tool("execute_sql", sql=sql)

            if not result.success:
                return QueryResult(
                    success=False,
                    query=sql,
                    results=[],
                    row_count=0,
                    execution_time_ms=result.duration_ms,
                    error=result.error,
                    explanation=explanation,
                )

            return QueryResult(
                success=True,
                query=sql,
                results=result.data.get("results", []),
                row_count=result.data.get("row_count", 0),
                execution_time_ms=result.data.get("execution_time_ms", 0),
                explanation=explanation,
                query_type=self._determine_query_type(sql),
            )

        except json.JSONDecodeError as e:
            logger.error("JSON decode error: %s", e)
            return QueryResult(
                success=False,
                query="",
                results=[],
                row_count=0,
                execution_time_ms=0,
                error=f"Invalid response format: {e}",
            )
        except Exception as e:
            logger.error("SQL generation error: %s", e)
            return QueryResult(
                success=False,
                query="",
                results=[],
                row_count=0,
                execution_time_ms=0,
                error=str(e),
            )

    async def _try_fallback_query(
        self,
        task_description: str,
        error: str,
    ) -> QueryResult | None:
        """
        Try to generate a simpler fallback query after an error.
        
        Args:
            task_description: Original task description
            error: Error from the failed query
            
        Returns:
            QueryResult if fallback succeeds, None otherwise
        """
        self.log(f"Attempting fallback query due to: {error}")
        logger.info("Attempting fallback query for: %s", task_description[:50])

        fallback_prompt = f"""Die vorherige SQL-Abfrage ist fehlgeschlagen mit: {error}

Generiere eine EINFACHERE Abfrage für: {task_description}

WICHTIG:
- Verwende eine vereinfachte Version
- Vermeide komplexe JSONB-Operationen wenn möglich
- Stelle sicher, dass alle Spalten existieren
- Verwende keine Subqueries"""

        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": fallback_prompt},
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self._OPENAI_TOOLS,
                tool_choice={"type": "function", "function": {"name": "execute_sql"}},
            )

            message = response.choices[0].message
            
            if not message.tool_calls:
                return None

            args = json.loads(message.tool_calls[0].function.arguments)
            sql = args.get("sql", "")

            if not sql:
                return None

            result = await self.call_tool("execute_sql", sql=sql)
            
            if result.success:
                self.log(f"Fallback successful: {result.data.get('row_count', 0)} rows")
                return QueryResult(
                    success=True,
                    query=sql,
                    results=result.data.get("results", []),
                    row_count=result.data.get("row_count", 0),
                    execution_time_ms=result.data.get("execution_time_ms", 0),
                    explanation="Fallback query",
                    is_fallback=True,
                    query_type=self._determine_query_type(sql),
                )

        except Exception as e:
            logger.warning("Fallback query failed: %s", e)
            self.log(f"Fallback also failed: {e}")

        return None

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"SQLGeneratorAgent(model={self.model!r}, "
            f"tools={len(self._tools)})"
        )
