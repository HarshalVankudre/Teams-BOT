# LangGraph Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace CleanSingleAgent with a LangGraph ReAct agent for simpler, more maintainable code.

**Architecture:** ReAct agent using `create_react_agent()` with 4 core tools (execute_sql, search_documents, find_columns, explore_column). State persisted via LangGraph's RedisSaver checkpointer. Fallback chain preserved: LangGraph → CleanSingleAgent → Legacy.

**Tech Stack:** LangGraph 0.2+, LangChain 0.3+, langchain-openai, Redis

**Worktree:** `.worktrees/langgraph-migration` (branch: `feature/langgraph-migration`)

---

## Task 1: Add Dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Add LangGraph and LangChain dependencies**

Add these lines to `requirements.txt`:

```
# LangGraph agent
langgraph>=0.2.0
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-community>=0.3.0
```

**Step 2: Verify dependencies can be resolved**

Run: `pip install -r requirements.txt --dry-run`
Expected: No conflicts, all packages resolvable

**Step 3: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: Successfully installed langgraph, langchain, langchain-openai, langchain-community

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "deps: add langgraph and langchain dependencies"
```

---

## Task 2: Add Config Variables

**Files:**
- Modify: `rag/config.py`

**Step 1: Read current config.py**

Read `rag/config.py` to understand the existing pattern for environment variables.

**Step 2: Add USE_LANGGRAPH_AGENT config**

Add to the RAGConfig class (after USE_CLEAN_AGENT):

```python
# LangGraph agent toggle
USE_LANGGRAPH_AGENT: bool = os.getenv("USE_LANGGRAPH_AGENT", "true").lower() == "true"
```

**Step 3: Add REDIS_URL config if not present**

Check if REDIS_URL exists. If not, add:

```python
# Redis connection for LangGraph checkpointing
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
```

**Step 4: Verify config loads**

Run: `python -c "from rag.config import config; print(config.USE_LANGGRAPH_AGENT)"`
Expected: `True`

**Step 5: Commit**

```bash
git add rag/config.py
git commit -m "config: add USE_LANGGRAPH_AGENT and REDIS_URL settings"
```

---

## Task 3: Create LangGraph Agent - Tool Definitions

**Files:**
- Create: `rag/langgraph_agent.py`

**Step 1: Create file with imports and tool definitions**

Create `rag/langgraph_agent.py`:

```python
"""LangGraph ReAct agent for RÜKO equipment queries."""

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
def search_documents(query: str, top_k: int = 10) -> dict:
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
        results = pinecone.search(query, top_k=top_k)

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

    # Validate column name (prevent injection)
    if not column_name.replace("_", "").replace("prop", "").replace("e", "").isalnum():
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
```

**Step 2: Verify imports work**

Run: `python -c "from rag.langgraph_agent import execute_sql, search_documents, find_columns, explore_column; print('Tools loaded')"`
Expected: `Tools loaded`

**Step 3: Commit**

```bash
git add rag/langgraph_agent.py
git commit -m "feat: add LangGraph tool definitions"
```

---

## Task 4: Create LangGraph Agent - ReAct Graph

**Files:**
- Modify: `rag/langgraph_agent.py`

**Step 1: Add system prompt and agent creation**

Append to `rag/langgraph_agent.py`:

```python
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
            model=config.OPENAI_MODEL,
            temperature=0,
            api_key=config.OPENAI_API_KEY
        )

        # Collect tools
        self.tools = [execute_sql, search_documents, find_columns, explore_column]

        # Add web search if enabled
        if config.ENABLE_WEB_SEARCH:
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
        redis_url = config.REDIS_URL if hasattr(config, "REDIS_URL") else None
        _agent_instance = LangGraphAgent(redis_url=redis_url)
    return _agent_instance
```

**Step 2: Verify agent can be instantiated**

Run: `python -c "from rag.langgraph_agent import LangGraphAgent; print('Agent class loaded')"`
Expected: `Agent class loaded`

**Step 3: Commit**

```bash
git add rag/langgraph_agent.py
git commit -m "feat: add LangGraph ReAct agent with checkpointing"
```

---

## Task 5: Integrate with RAGSearch

**Files:**
- Modify: `rag/search.py`

**Step 1: Read current search.py structure**

Read `rag/search.py` to find the `search_and_generate` method and understand the routing logic.

**Step 2: Add LangGraph import and initialization**

Near the top of the class `__init__`, add:

```python
# LangGraph agent (new)
self.langgraph_agent = None
if config.USE_LANGGRAPH_AGENT:
    try:
        from rag.langgraph_agent import get_langgraph_agent
        self.langgraph_agent = get_langgraph_agent()
        logger.info("LangGraph agent initialized")
    except Exception as e:
        logger.warning(f"LangGraph agent init failed: {e}")
```

**Step 3: Add LangGraph routing in search_and_generate**

At the start of the agent routing section (before CleanSingleAgent), add:

```python
# Priority 1: LangGraph agent
if self.langgraph_agent and config.USE_LANGGRAPH_AGENT:
    try:
        result = await self.langgraph_agent.process(
            user_query=query,
            thread_key=thread_key,
            conversation_history=conversation_history
        )
        return {
            "response": result.response,
            "sources": result.sources or [],
            "tools_used": result.tools_used,
            "execution_time_ms": result.execution_time_ms,
            "agent": "langgraph"
        }
    except Exception as e:
        logger.warning(f"LangGraph agent failed: {e}, falling back")
```

**Step 4: Verify search module loads**

Run: `python -c "from rag.search import RAGSearch; print('RAGSearch loaded')"`
Expected: `RAGSearch loaded`

**Step 5: Commit**

```bash
git add rag/search.py
git commit -m "feat: integrate LangGraph agent into RAGSearch"
```

---

## Task 6: Manual Integration Test

**Files:**
- None (testing only)

**Step 1: Set environment for testing**

Ensure `.env` has required variables:
- `OPENAI_API_KEY`
- `POSTGRES_*` credentials
- `PINECONE_API_KEY`
- `USE_LANGGRAPH_AGENT=true`

**Step 2: Run interactive test**

Run: `python cli_tester.py`

Test queries:
1. `Wie viele Kettenfertiger haben wir?`
2. `Welche davon sind von Voegele?`
3. `Zeige alle Bomag Maschinen zur Vermietung`

**Step 3: Verify agent identifier**

Check that response metadata shows `"agent": "langgraph"`

**Step 4: Test fallback**

Set `USE_LANGGRAPH_AGENT=false` and verify CleanSingleAgent handles queries.

---

## Task 7: Add Logging and Error Handling

**Files:**
- Modify: `rag/langgraph_agent.py`

**Step 1: Add verbose logging option**

Add after the imports:

```python
VERBOSE = config.AGENT_VERBOSE if hasattr(config, "AGENT_VERBOSE") else False

def _log_tool_call(tool_name: str, args: dict, result: dict):
    """Log tool calls when verbose mode is enabled."""
    if VERBOSE:
        logger.info(f"Tool: {tool_name}")
        logger.info(f"  Args: {args}")
        logger.info(f"  Result rows: {result.get('row_count', result.get('match_count', 'N/A'))}")
```

**Step 2: Add error recovery in process method**

Wrap the graph invocation with better error handling:

```python
try:
    result = await self.graph.ainvoke(...)
except Exception as e:
    if "rate_limit" in str(e).lower():
        logger.warning("Rate limited, waiting 5s...")
        await asyncio.sleep(5)
        result = await self.graph.ainvoke(...)  # Retry once
    else:
        raise
```

**Step 3: Commit**

```bash
git add rag/langgraph_agent.py
git commit -m "feat: add verbose logging and rate limit handling"
```

---

## Task 8: Update Documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update architecture section**

Add LangGraph to the request flow:

```markdown
### Request Flow
```
Teams → Bot Framework → FastAPI (app.py) → RAGSearch → LangGraphAgent → Tools
                                                              ├── execute_sql (PostgreSQL)
                                                              ├── search_documents (Pinecone)
                                                              ├── find_columns (SchemaLinker)
                                                              └── explore_column (DB introspection)
```
```

**Step 2: Add LangGraph agent description**

Add to Key Components:

```markdown
**rag/langgraph_agent.py (LangGraphAgent)** - RECOMMENDED: LangGraph ReAct agent using `create_react_agent()`. Automatic tool loop, Redis checkpointing for conversation state. Default when USE_LANGGRAPH_AGENT=true.
```

**Step 3: Update configuration section**

Add to key optional toggles:

```markdown
- `USE_LANGGRAPH_AGENT=true` - Use LangGraph ReAct agent (recommended, default)
```

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add LangGraph agent to CLAUDE.md"
```

---

## Task 9: Final Validation

**Files:**
- None (testing only)

**Step 1: Run batch tests**

Run: `python cli_tester.py --batch`
Expected: All test cases pass

**Step 2: Verify conversation memory**

Run interactive session with follow-ups:
1. Query: `Zeige mir alle Bagger`
2. Follow-up: `Welche davon sind von Caterpillar?`
3. Verify the follow-up uses previous context

**Step 3: Test fallback chain**

1. Set `USE_LANGGRAPH_AGENT=false` → verify CleanSingleAgent works
2. Set `USE_CLEAN_AGENT=false` → verify legacy SingleAgent works
3. Restore both to `true`

**Step 4: Create final commit**

```bash
git add -A
git commit -m "feat: complete LangGraph migration

- LangGraph ReAct agent with 4 core tools
- Redis checkpointing for conversation state
- Fallback chain: LangGraph → CleanSingleAgent → Legacy
- Verbose logging and error handling

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

| Task | Description | Estimated Steps |
|------|-------------|-----------------|
| 1 | Add dependencies | 4 |
| 2 | Add config variables | 5 |
| 3 | Create tool definitions | 3 |
| 4 | Create ReAct agent | 3 |
| 5 | Integrate with RAGSearch | 5 |
| 6 | Manual integration test | 4 |
| 7 | Add logging/error handling | 3 |
| 8 | Update documentation | 4 |
| 9 | Final validation | 4 |

**Total: 9 tasks, ~35 steps**

---

## Rollback

If issues arise:

```bash
# Immediate rollback (no code changes)
export USE_LANGGRAPH_AGENT=false

# Full revert
git checkout main
git branch -D feature/langgraph-migration
```
