# LangGraph Migration Design

**Date:** 2026-01-23
**Status:** Approved
**Goal:** Replace CleanSingleAgent with LangGraph ReAct agent for simpler, more maintainable code

---

## Summary

Migrate the Teams bot agent from a custom tool loop (`CleanSingleAgent`) to LangGraph's `create_react_agent()` pattern. This provides:
- Explicit state management via graph nodes
- Built-in Redis checkpointing for conversation memory
- Native LangChain tool integration
- Cleaner separation of concerns

---

## Architecture Overview

```
Teams → app.py → RAGSearch → LangGraphAgent → Tools
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
              execute_sql    search_docs    search_web
                    │              │              │
              PostgreSQL      Pinecone        Tavily
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| LangGraph Agent | `rag/langgraph_agent.py` (new) | ReAct agent with tools |
| RAGSearch | `rag/search.py` (modified) | Routes to LangGraph agent |
| Config | `rag/config.py` (modified) | `USE_LANGGRAPH_AGENT` toggle |
| Legacy Fallback | `rag/single_agent.py` | Kept as safety net |

---

## State Definition

```python
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
    thread_key: str                          # User/conversation ID
    last_result_ids: List[int]               # For follow-up queries
```

The `messages` field uses LangGraph's `add_messages` reducer for automatic conversation management.

---

## Tools

### Core Tools (always enabled)

| Tool | Purpose | Validation |
|------|---------|------------|
| `execute_sql` | Run SELECT queries | `PostgresService.prepare_readonly_sql()`, 50-row limit |
| `search_documents` | Pinecone vector search | Parallel namespace search (docs + machinery) |
| `find_columns` | Semantic column lookup | `SchemaLinker.get_reduced_schema()` |
| `explore_column` | Show distinct values | Direct SQL with LIMIT 50 |

### Optional Tools

| Tool | Purpose | Condition |
|------|---------|-----------|
| `search_web` | Tavily web search | `ENABLE_WEB_SEARCH=true` |

### Tool Implementation Pattern

```python
from langchain_core.tools import tool

@tool
def execute_sql(sql: str, purpose: str) -> dict:
    """Execute a read-only SQL query against the equipment database.

    Args:
        sql: The SELECT query to execute
        purpose: Brief description of what this query is for

    Returns:
        Dict with row_count and results (max 50 rows)
    """
    prepared, error = postgres.prepare_readonly_sql(sql)
    if error:
        return {"error": error}
    results = postgres.execute_query(prepared)
    # Store IDs for follow-up queries
    return {
        "purpose": purpose,
        "row_count": len(results),
        "results": results[:50],
        "result_ids": [r.get("id") for r in results if r.get("id")]
    }

@tool
def search_documents(query: str, top_k: int = 10) -> dict:
    """Search equipment manuals and documentation.

    Args:
        query: Search query in German
        top_k: Number of results to return (default 10)

    Returns:
        Dict with matches containing title, content snippet, and source
    """
    results = pinecone_store.search(query, top_k=top_k)
    return {"matches": results}

@tool
def find_columns(keyword: str) -> dict:
    """Find relevant database columns by semantic search.

    Args:
        keyword: German keyword like 'breite' (width) or 'gewicht' (weight)

    Returns:
        Dict with matching column names, display names, and units
    """
    reduced_schema = schema_linker.get_reduced_schema(keyword, top_k=15)
    return {
        "keyword": keyword,
        "found": len(reduced_schema.column_info),
        "columns": [
            {
                "column_name": col.column_name,
                "display_name": col.display_name,
                "unit": col.unit
            }
            for col in reduced_schema.column_info.values()
        ]
    }

@tool
def explore_column(column_name: str) -> dict:
    """Show distinct values in a database column.

    Args:
        column_name: The column to explore (e.g., 'hersteller_name')

    Returns:
        Dict with distinct values (max 50)
    """
    sql = f"SELECT DISTINCT {column_name} FROM sema_matrix.equipment_matrix WHERE {column_name} IS NOT NULL LIMIT 50"
    results = postgres.execute_query(sql)
    return {
        "column": column_name,
        "distinct_values": [r[column_name] for r in results]
    }
```

---

## ReAct Graph Creation

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis import RedisSaver

SYSTEM_PROMPT = """Du bist der RÜKO Baumaschinen-Assistent.

WICHTIGE REGELN:
- Antworte IMMER auf Deutsch
- Nutze die verfügbaren Tools um Daten abzufragen
- Tabelle: sema_matrix.equipment_matrix
- Kategorien in geraetegruppe_name: Kettenfertiger, Radfertiger, Kettenbagger, etc.
- verwendung_code: 'MIET' (Vermietung), 'VK' (Verkauf)
- Zahlenformat in DB: Komma als Dezimaltrennzeichen (z.B. "3,5" = 3.5)

FOLLOW-UP FRAGEN:
- Bei "davon", "diese", "welche" beziehe dich auf vorherige Ergebnisse
- Nutze WHERE id IN (...) mit den letzten Ergebnis-IDs

ANTWORTFORMAT:
- Kurz und präzise
- Liste Maschinen mit: Bezeichnung, Hersteller, relevante Eigenschaften
- Bei Empfehlungen: Vor-/Nachteile nennen
"""

def create_agent(redis_url: str):
    """Create the LangGraph ReAct agent with Redis checkpointing."""

    # LLM setup
    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        temperature=0
    )

    # Tools list
    tools = [execute_sql, search_documents, find_columns, explore_column]
    if config.ENABLE_WEB_SEARCH:
        tools.append(search_web)

    # Redis checkpointer for state persistence
    checkpointer = RedisSaver.from_conn_string(redis_url)

    # Create ReAct agent
    graph = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=checkpointer,
        state_modifier=SYSTEM_PROMPT
    )

    return graph
```

---

## Agent Invocation

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class AgentResult:
    response: str
    tools_used: List[str]
    execution_time_ms: int
    token_usage: Optional[dict] = None

class LangGraphAgent:
    def __init__(self, redis_url: str):
        self.graph = create_agent(redis_url)

    async def process(
        self,
        user_query: str,
        thread_key: str
    ) -> AgentResult:
        """Process a user query through the ReAct agent."""

        import time
        start = time.time()

        # Thread ID maps to existing thread_key format
        config = {"configurable": {"thread_id": thread_key}}

        # Invoke graph - memory loaded automatically from Redis
        result = await self.graph.ainvoke(
            {"messages": [("user", user_query)]},
            config=config
        )

        # Extract final response
        final_message = result["messages"][-1]
        tools_used = self._extract_tools_used(result["messages"])

        execution_time = int((time.time() - start) * 1000)

        return AgentResult(
            response=final_message.content,
            tools_used=tools_used,
            execution_time_ms=execution_time
        )

    def _extract_tools_used(self, messages: list) -> List[str]:
        """Extract tool names from message history."""
        tools = []
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc['name'] not in tools:
                        tools.append(tc['name'])
        return tools
```

---

## RAGSearch Integration

```python
# In rag/search.py

class RAGSearch:
    def __init__(self):
        # Initialize based on config
        if config.USE_LANGGRAPH_AGENT:
            from rag.langgraph_agent import LangGraphAgent
            self.langgraph_agent = LangGraphAgent(config.REDIS_URL)

        if config.USE_CLEAN_AGENT:
            self.clean_agent = CleanSingleAgent(...)

        # Legacy always available as final fallback
        self.legacy_agent = SingleAgent(...)

    async def search_and_generate(
        self,
        query: str,
        thread_key: str,
        ...
    ) -> dict:
        """Route query through agent priority chain."""

        # Priority 1: LangGraph agent
        if config.USE_LANGGRAPH_AGENT:
            try:
                result = await self.langgraph_agent.process(query, thread_key)
                return self._format_result(result)
            except Exception as e:
                logger.warning(f"LangGraph agent failed: {e}, falling back")

        # Priority 2: Clean agent
        if config.USE_CLEAN_AGENT:
            try:
                result = await self.clean_agent.process(query, thread_key)
                return self._format_result(result)
            except Exception as e:
                logger.warning(f"Clean agent failed: {e}, falling back")

        # Priority 3: Legacy agent
        return await self._run_legacy_agent(query, thread_key)
```

---

## Configuration

### New Environment Variables

```bash
# Agent selection (priority order)
USE_LANGGRAPH_AGENT=true   # Try LangGraph first (new)
USE_CLEAN_AGENT=true       # Fallback to CleanSingleAgent
USE_AGENT_SYSTEM=true      # Fallback to legacy SingleAgent

# Redis for LangGraph checkpointing
REDIS_URL=redis://localhost:6379
```

### Config Updates (rag/config.py)

```python
# Add to RAGConfig class
USE_LANGGRAPH_AGENT: bool = os.getenv("USE_LANGGRAPH_AGENT", "true").lower() == "true"
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
```

---

## Dependencies

### New (requirements.txt)

```
langgraph>=0.2.0
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-community>=0.3.0
redis>=5.0.0
```

---

## File Changes Summary

### Create

| File | Lines (est.) | Purpose |
|------|--------------|---------|
| `rag/langgraph_agent.py` | ~300 | Main LangGraph agent |

### Modify

| File | Changes |
|------|---------|
| `rag/search.py` | Add LangGraph routing (~30 lines) |
| `rag/config.py` | Add config vars (~5 lines) |
| `requirements.txt` | Add 4 dependencies |

### Unchanged

- `app.py` - Uses RAGSearch interface
- `rag/postgres.py` - Called by tool
- `rag/vector_store.py` - Called by tool
- `rag/schema_linker.py` - Called by tool
- `rag/single_agent.py` - Legacy fallback
- `rag/single_agent_clean.py` - Fallback

### Remove (Phase 2)

- `rag/providers/base.py`
- `rag/providers/openai_provider.py`
- `rag/providers/cerebras_provider.py`
- `rag/providers/__init__.py`

---

## Implementation Phases

### Phase 1: Core Implementation

1. Add LangGraph dependencies to `requirements.txt`
2. Create `rag/langgraph_agent.py`:
   - State definition
   - Tool implementations
   - ReAct graph creation
   - System prompt
3. Add `USE_LANGGRAPH_AGENT` to `rag/config.py`
4. Wire up routing in `rag/search.py`

### Phase 2: Testing & Validation

1. Interactive testing: `python cli_tester.py`
2. Batch tests: `python cli_tester.py --batch`
3. Test cases:
   - Simple counts: "Wie viele Bagger haben wir?"
   - Filters: "Zeige alle Bomag Kettenfertiger"
   - Follow-ups: "Welche davon sind zur Vermietung?"
   - Documents: "Kettenfertiger Betriebsanleitung"
   - Recommendations: "Welchen Fertiger empfehlen Sie?"

### Phase 3: Cleanup

1. Remove `rag/providers/` directory
2. Update CLAUDE.md documentation
3. Set `USE_LANGGRAPH_AGENT=true` as default

---

## Rollback Strategy

Immediate rollback without code changes:

```bash
# Revert to CleanSingleAgent
USE_LANGGRAPH_AGENT=false

# Revert to legacy SingleAgent
USE_LANGGRAPH_AGENT=false
USE_CLEAN_AGENT=false
```

---

## Benefits

| Aspect | Before (CleanSingleAgent) | After (LangGraph) |
|--------|---------------------------|-------------------|
| Tool loop | Manual while loop | Automatic ReAct graph |
| State management | Manual `_thread_state` dict | Built-in MemorySaver |
| Conversation memory | Custom Redis code | LangGraph checkpointing |
| Provider abstraction | Custom wrapper classes | LangChain ChatOpenAI |
| Debugging | Print statements | LangGraph tracing |
| Code complexity | ~737 lines | ~300 lines |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| LangGraph API changes | Pin versions in requirements.txt |
| Redis checkpointer issues | Fallback chain to CleanSingleAgent |
| Tool compatibility | Reuse existing service classes |
| Performance regression | Benchmark with batch tests |
