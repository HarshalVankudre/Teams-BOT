"""
SQL Generator SubAgent

Generates and executes SQL queries against the PostgreSQL database.
Uses a fast, non-reasoning model for efficiency.
Supports both OpenAI and Ollama providers.
"""
import json
from typing import List, Dict, Any, Optional, Union
from openai import AsyncOpenAI

from .interface import (
    SubAgentBase,
    tool,
    register_subagent,
    AgentMetadata,
    AgentCapability,
    AgentContext,
    AgentResponse,
    AgentType
)
from ...config import config
from ...postgres import postgres_service, PostgresService
from ...schema import SQL_AGENT_SCHEMA
from ...providers import create_llm_client, AsyncOllamaClient


# Agent metadata
SQL_AGENT_METADATA = AgentMetadata(
    agent_id="sql",
    name="SQL Generator Agent",
    description="Generates and executes SQL queries against the equipment database",
    detailed_description="""Generiert und führt SQL-Abfragen gegen die PostgreSQL-Datenbank aus.
Verwende diesen Agenten für:
- Zählung von Geräten (z.B. "Wie viele Bagger haben wir?")
- Filterung nach Eigenschaften (z.B. "Bagger über 15 Tonnen")
- Aggregationen (z.B. "Durchschnittsgewicht der Mobilbagger")
- Vergleiche zwischen Gerätegruppen
- Suche nach Seriennummer oder Inventarnummer""",
    capabilities=[AgentCapability.DATABASE_QUERY, AgentCapability.DATA_ANALYSIS],
    uses_reasoning=False,
    default_model="gpt-4o-mini",
    parameters={
        "task_description": {
            "type": "string",
            "description": "Beschreibung der SQL-Aufgabe, z.B. 'Zähle alle Bagger im Bestand'"
        },
        "filters": {
            "type": "object",
            "description": "Optionale Filter wie {'hersteller': 'Liebherr', 'min_gewicht': 10000}"
        }
    },
    example_queries=[
        "Wie viele Bagger haben wir?",
        "Liste alle Kettenbagger von Liebherr",
        "Welche Geräte haben GPS?",
        "Durchschnittsgewicht der Mobilbagger",
        "Vergleich Kettenbagger vs Mobilbagger"
    ],
    priority=10
)


@register_subagent()
class SQLGeneratorAgent(SubAgentBase):
    """
    Generates SQL queries based on task descriptions and executes them.
    Uses tool-calling to generate valid SQL for the equipment database.
    """

    METADATA = SQL_AGENT_METADATA

    # System prompt uses schema from centralized schema.py
    SYSTEM_PROMPT = f"""Du bist ein SQL-Generator für eine PostgreSQL-Datenbank mit Baumaschinen.

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

KRITISCH:
- Für Gerätetypen (Bagger, Walzen, etc.): IMMER geraetegruppe verwenden!
- kategorie ist oft NULL und unvollständig - NUR als Fallback!
- Keine unnötigen Filter hinzufügen!

WICHTIG:
- Gib das SQL als einzeiligen String zurück
- Keine Kommentare im SQL
- Keine Erklärungen, nur das SQL"""

    # OpenAI tool definition for SQL execution
    OPENAI_TOOLS = [
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

    def __init__(
        self,
        openai_client: Optional[Union[AsyncOpenAI, AsyncOllamaClient]] = None,
        model: Optional[str] = None,
        postgres: Optional[PostgresService] = None,
        verbose: bool = False
    ):
        super().__init__(verbose=verbose)
        self._agent_type = AgentType.SQL_GENERATOR

        # Create client if not provided
        if openai_client is not None:
            self.client = openai_client
        else:
            self.client = create_llm_client(
                provider=config.llm_provider,
                api_key=config.openai_api_key,
                base_url=config.ollama_base_url if config.is_ollama() else None,
                model=config.ollama_model if config.is_ollama() else None
            )

        # Use configured model or fall back to config
        if config.is_ollama():
            self.model = model or config.get_chat_model()
        else:
            self.model = model or config.chunking_model or "gpt-4o-mini"

        self.postgres = postgres or postgres_service

        # Detect if using Ollama
        self._is_ollama = isinstance(self.client, AsyncOllamaClient) or config.is_ollama()

    # ==================== TOOLS ====================

    @tool(
        name="execute_sql",
        description="Execute a SQL query against the equipment database",
        parameters={
            "sql": {"type": "string", "description": "The SQL SELECT query to execute"}
        },
        required=["sql"]
    )
    async def execute_sql_tool(self, sql: str) -> Dict[str, Any]:
        """Execute a SQL query and return results"""
        try:
            results = self.postgres.execute_dynamic_sql(sql)
            return {
                "success": True,
                "results": results,
                "row_count": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "row_count": 0
            }

    @tool(
        name="get_schema_info",
        description="Get database schema information",
        parameters={}
    )
    async def get_schema_info_tool(self) -> Dict[str, Any]:
        """Return schema information for the database"""
        return {
            "schema": SQL_AGENT_SCHEMA
        }

    # ==================== MAIN EXECUTION ====================

    async def _execute(self, context: AgentContext) -> AgentResponse:
        """
        Generate and execute SQL based on the task description.
        """
        # Extract task from context (set by orchestrator)
        task_description = context.metadata.get("sql_task", context.user_query)
        filters = context.metadata.get("sql_filters", {})

        # Build the prompt
        prompt = f"Generiere eine SQL-Abfrage für folgende Anfrage:\n\n{task_description}"

        if filters:
            prompt += f"\n\nZusätzliche Filter: {json.dumps(filters, ensure_ascii=False)}"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        self.log(f"Generating SQL for: {task_description[:50]}...")

        # Call LLM to generate SQL
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.OPENAI_TOOLS,
            tool_choice={"type": "function", "function": {"name": "execute_sql"}}
        )

        message = response.choices[0].message

        if not message.tool_calls:
            return AgentResponse.error_response(
                error="No SQL generated",
                agent_type=self._agent_type
            )

        # Extract SQL from tool call
        tool_call = message.tool_calls[0]
        try:
            args = json.loads(tool_call.function.arguments)
            sql = args.get("sql", "")
            explanation = args.get("explanation", "")
        except json.JSONDecodeError:
            return AgentResponse.error_response(
                error="Invalid SQL response format",
                agent_type=self._agent_type
            )

        if not sql:
            return AgentResponse.error_response(
                error="Empty SQL query",
                agent_type=self._agent_type
            )

        self.log(f"SQL: {sql[:100]}...")

        # Execute the SQL using our tool
        result = await self.execute_sql_tool(sql)

        if not result["success"]:
            self.log(f"SQL execution error: {result.get('error')}")

            # Try fallback query
            fallback_result = await self._try_fallback_query(task_description, result.get("error", ""))
            if fallback_result:
                context.sql_results = fallback_result["results"]
                return AgentResponse.success_response(
                    data=fallback_result,
                    agent_type=self._agent_type,
                    reasoning=f"Fallback query used after error: {result.get('error')}"
                )

            return AgentResponse.error_response(
                error=f"SQL execution failed: {result.get('error')}",
                agent_type=self._agent_type
            )

        results = result["results"]
        self.log(f"Results: {len(results)} rows")

        # Store results in context for reviewer
        context.sql_results = results

        return AgentResponse.success_response(
            data={
                "sql": sql,
                "results": results,
                "row_count": len(results),
                "explanation": explanation
            },
            agent_type=self._agent_type,
            reasoning=explanation,
            tool_calls=[{"name": "execute_sql", "sql": sql}],
            sources=[{"type": "postgresql", "query": sql, "row_count": len(results)}]
        )

    async def _try_fallback_query(
        self,
        task_description: str,
        error: str
    ) -> Optional[Dict[str, Any]]:
        """Try to generate a simpler fallback query after an error"""
        self.log(f"Attempting fallback query due to: {error}")

        fallback_prompt = f"""Die vorherige SQL-Abfrage ist fehlgeschlagen mit: {error}

Generiere eine EINFACHERE Abfrage für: {task_description}

WICHTIG:
- Verwende eine vereinfachte Version
- Vermeide komplexe JSONB-Operationen wenn möglich
- Stelle sicher, dass alle Spalten existieren"""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": fallback_prompt}
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.OPENAI_TOOLS,
                tool_choice={"type": "function", "function": {"name": "execute_sql"}}
            )

            message = response.choices[0].message
            if message.tool_calls:
                args = json.loads(message.tool_calls[0].function.arguments)
                sql = args.get("sql", "")

                if sql:
                    result = await self.execute_sql_tool(sql)
                    if result["success"]:
                        self.log(f"Fallback successful: {result['row_count']} rows")
                        return {
                            "sql": sql,
                            "results": result["results"],
                            "row_count": result["row_count"],
                            "explanation": "Fallback query",
                            "is_fallback": True
                        }
        except Exception as e:
            self.log(f"Fallback also failed: {str(e)}")

        return None
