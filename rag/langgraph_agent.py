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
