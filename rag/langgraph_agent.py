"""LangGraph ReAct agent for RÜKO equipment queries."""

import re
import time
import logging
from typing import List, Optional
from dataclasses import dataclass

from langchain_core.tools import tool

from rag.config import config
from rag.postgres import PostgresService
from rag.vector_store import PineconeStore
from rag.schema_linker import SchemaLinker

logger = logging.getLogger(__name__)

# Initialize services (singleton pattern matches existing codebase)
_postgres: Optional[PostgresService] = None
_pinecone: Optional[PineconeStore] = None
_schema_linker: Optional[SchemaLinker] = None


def _get_postgres() -> PostgresService:
    global _postgres
    if _postgres is None:
        _postgres = PostgresService()
    return _postgres


def _get_pinecone() -> PineconeStore:
    global _pinecone
    if _pinecone is None:
        _pinecone = PineconeStore()
    return _pinecone


def _get_schema_linker() -> SchemaLinker:
    global _schema_linker
    if _schema_linker is None:
        _schema_linker = SchemaLinker()
    return _schema_linker


@tool
def execute_sql(sql: str, purpose: str) -> dict:
    """Execute a read-only SQL query against the equipment database.

    Use this tool to query sema_matrix.equipment_matrix for equipment data.
    Only SELECT queries are allowed. Results limited to 50 rows.

    Args:
        sql: The SELECT query to execute. Use proper German column names.
        purpose: Brief description of what this query is for.

    Returns:
        Dict with row_count, results (max 50 rows), and result_ids for follow-ups.
    """
    postgres = _get_postgres()

    # Validate and prepare SQL
    prepared, error = postgres.prepare_readonly_sql(sql)
    if error:
        return {"error": error, "sql": sql}

    try:
        results = postgres.execute_query(prepared)
        result_ids = [r.get("id") for r in results if r.get("id")]

        return {
            "purpose": purpose,
            "sql": prepared,
            "row_count": len(results),
            "results": results[:50],
            "result_ids": result_ids[:100]  # Store for follow-ups
        }
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        return {"error": str(e), "sql": prepared}


@tool
async def search_documents(query: str, top_k: int = 10) -> dict:
    """Search equipment manuals, documentation, and technical specifications.

    Use this for questions about operating instructions, maintenance,
    technical details not in the database, or general equipment information.

    Args:
        query: Search query in German (e.g., "Kettenfertiger Wartung")
        top_k: Number of results to return (default 10, max 20)

    Returns:
        Dict with matches containing title, content snippet, and source.
    """
    pinecone = _get_pinecone()
    top_k = min(top_k, 20)

    try:
        results = await pinecone.search(query, top_k=top_k)

        matches = []
        for r in results:
            matches.append({
                "title": r.get("metadata", {}).get("title", "Untitled"),
                "content": r.get("metadata", {}).get("content", "")[:500],
                "source": r.get("metadata", {}).get("source_file", "unknown"),
                "score": r.get("score", 0)
            })

        return {
            "query": query,
            "match_count": len(matches),
            "matches": matches
        }
    except Exception as e:
        logger.error(f"Document search error: {e}")
        return {"error": str(e), "query": query}


@tool
def find_columns(keyword: str) -> dict:
    """Find relevant database columns by semantic search.

    Use this when you need to find the correct column name for a property
    like width, weight, power, etc. Returns matching column names with units.

    Args:
        keyword: German keyword like 'breite' (width), 'gewicht' (weight),
                'leistung' (power), 'tiefe' (depth)

    Returns:
        Dict with matching columns including column_name, display_name, and unit.
    """
    schema_linker = _get_schema_linker()

    try:
        reduced_schema = schema_linker.get_reduced_schema(keyword, top_k=15)

        columns = []
        for col_id, col_info in reduced_schema.column_info.items():
            columns.append({
                "column_name": col_info.column_name,
                "display_name": col_info.display_name,
                "unit": col_info.unit or "",
                "description": col_info.description or ""
            })

        return {
            "keyword": keyword,
            "found": len(columns),
            "columns": columns
        }
    except Exception as e:
        logger.error(f"Column search error: {e}")
        return {"error": str(e), "keyword": keyword}


@tool
def explore_column(column_name: str) -> dict:
    """Show distinct values in a database column.

    Use this to understand what values exist in a column before writing queries.
    Helpful for categories, manufacturers, or status fields.

    Args:
        column_name: The column to explore (e.g., 'hersteller_name', 'geraetegruppe_name')

    Returns:
        Dict with distinct values (max 50).
    """
    postgres = _get_postgres()

    # Validate column name format (PostgreSQL identifier)
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', column_name):
        return {"error": f"Invalid column name: {column_name}"}

    sql = f"""
        SELECT DISTINCT "{column_name}"
        FROM sema_matrix.equipment_matrix
        WHERE "{column_name}" IS NOT NULL
        ORDER BY "{column_name}"
        LIMIT 50
    """

    try:
        results = postgres.execute_query(sql)
        values = [r.get(column_name) for r in results if r.get(column_name)]

        return {
            "column": column_name,
            "distinct_count": len(values),
            "values": values
        }
    except Exception as e:
        logger.error(f"Column exploration error: {e}")
        return {"error": str(e), "column": column_name}


# System prompt for the ReAct agent
SYSTEM_PROMPT = """Du bist der RÜKO Baumaschinen-Assistent.

DATENBANK-INFORMATIONEN:
- Tabelle: sema_matrix.equipment_matrix (~2400 Maschinen)
- Kategorien (geraetegruppe_name): Kettenfertiger, Radfertiger, Kettenbagger, Mobilbagger, Radlader, etc.
- Hersteller (hersteller_name): Voegele, Bomag, Caterpillar, Liebherr, etc.
- verwendung_code: 'MIET' (Vermietung), 'VK' (Verkauf)
- Zahlenformat: Komma als Dezimaltrennzeichen (z.B. "3,5" = 3.5)
- Für numerische Vergleiche: CAST(NULLIF(REPLACE(regexp_replace(col, '[^0-9,]', '', 'g'), ',', '.'), '') AS NUMERIC)

WERKZEUGE:
1. execute_sql - SQL-Abfragen für Bestandsdaten
2. search_documents - Technische Dokumentation durchsuchen
3. find_columns - Spalten für Eigenschaften finden (Breite, Gewicht, etc.)
4. explore_column - Mögliche Werte einer Spalte anzeigen

WICHTIGE REGELN:
- Antworte IMMER auf Deutsch
- Nutze find_columns ZUERST wenn du unsicher über Spaltennamen bist
- Bei "davon", "diese", "welche" - beziehe dich auf vorherige Ergebnisse
- Halte Antworten kurz und präzise
- Liste Maschinen mit: Bezeichnung, Hersteller, relevante Eigenschaften
"""


@dataclass
class AgentResult:
    """Result from LangGraph agent processing."""
    response: str
    tools_used: List[str]
    execution_time_ms: int
    token_usage: Optional[dict] = None
    sources: Optional[List[str]] = None


class LangGraphAgent:
    """LangGraph ReAct agent for equipment queries."""

    def __init__(self, redis_url: Optional[str] = None):
        """Initialize the LangGraph agent.

        Args:
            redis_url: Redis connection URL for checkpointing.
                      If None, uses in-memory checkpointer.
        """
        from langgraph.prebuilt import create_react_agent
        from langchain_openai import ChatOpenAI

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config.openai_model,
            temperature=0,
            api_key=config.openai_api_key
        )

        # Collect tools
        self.tools = [execute_sql, search_documents, find_columns, explore_column]

        # Add web search if enabled
        if hasattr(config, 'enable_web_search') and config.enable_web_search:
            try:
                from langchain_community.tools.tavily_search import TavilySearchResults
                web_search = TavilySearchResults(
                    max_results=5,
                    description="Search the web for current information about equipment, manufacturers, or recommendations."
                )
                self.tools.append(web_search)
            except Exception as e:
                logger.warning(f"Web search not available: {e}")

        # Set up checkpointer
        self.checkpointer = None
        if redis_url:
            try:
                from langgraph.checkpoint.redis import RedisSaver
                self.checkpointer = RedisSaver.from_conn_string(redis_url)
                logger.info("Using Redis checkpointer for LangGraph")
            except Exception as e:
                logger.warning(f"Redis checkpointer failed, using memory: {e}")

        if self.checkpointer is None:
            from langgraph.checkpoint.memory import MemorySaver
            self.checkpointer = MemorySaver()
            logger.info("Using in-memory checkpointer for LangGraph")

        # Create ReAct agent
        self.graph = create_react_agent(
            model=self.llm,
            tools=self.tools,
            checkpointer=self.checkpointer,
            state_modifier=SYSTEM_PROMPT
        )

        logger.info(f"LangGraph agent initialized with {len(self.tools)} tools")

    async def process(
        self,
        user_query: str,
        thread_key: str,
        conversation_history: Optional[List[dict]] = None
    ) -> AgentResult:
        """Process a user query through the ReAct agent.

        Args:
            user_query: The user's question in German
            thread_key: Unique thread identifier for state persistence
            conversation_history: Optional prior messages (used if no checkpoint exists)

        Returns:
            AgentResult with response and metadata
        """
        start_time = time.time()

        # Build messages
        messages = []

        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[-6:]:  # Last 3 exchanges
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append((role, content))

        # Add current query
        messages.append(("user", user_query))

        # Configure with thread_id for checkpointing
        run_config = {"configurable": {"thread_id": thread_key}}

        try:
            # Invoke the graph
            result = await self.graph.ainvoke(
                {"messages": messages},
                config=run_config
            )

            # Extract response from last message
            final_message = result["messages"][-1]
            response = final_message.content if hasattr(final_message, "content") else str(final_message)

            # Extract tools used
            tools_used = self._extract_tools_used(result["messages"])

            execution_time = int((time.time() - start_time) * 1000)

            return AgentResult(
                response=response,
                tools_used=tools_used,
                execution_time_ms=execution_time
            )

        except Exception as e:
            logger.error(f"LangGraph agent error: {e}", exc_info=True)
            raise

    def _extract_tools_used(self, messages: list) -> List[str]:
        """Extract unique tool names from message history."""
        tools = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    if name and name not in tools:
                        tools.append(name)
        return tools


# Singleton instance
_agent_instance: Optional[LangGraphAgent] = None


def get_langgraph_agent() -> LangGraphAgent:
    """Get or create the singleton LangGraph agent instance."""
    global _agent_instance
    if _agent_instance is None:
        redis_url = config.redis_url if hasattr(config, "redis_url") else None
        _agent_instance = LangGraphAgent(redis_url=redis_url)
    return _agent_instance
