# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Teams-BOT is a Microsoft Teams bot backend powered by FastAPI that provides an LLM-backed assistant with tool access to structured (PostgreSQL) and unstructured (Pinecone) data sources. The bot responds to equipment queries in German for RÜKO Baumaschinen.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (port 8001)
python app.py

# Run with uvicorn (port 8000, with reload)
uvicorn app:app --reload --port 8000

# Interactive CLI testing (no Teams required)
python cli_tester.py

# Run CLI tester with specific query
python cli_tester.py "Wie viele Bomag Maschinen haben wir?"

# Run batch tests
python cli_tester.py --batch

# Direct SQL test
python cli_tester.py --sql "SELECT COUNT(*) FROM sema_matrix.equipment_matrix"

# Document search test
python cli_tester.py --search "Kettenfertiger Anleitung"
```

## Architecture

### Request Flow
```
Teams → Bot Framework → FastAPI (app.py) → RAGSearch → LangGraphAgent → Tools
                                                              ├── execute_sql (PostgreSQL)
                                                              ├── search_documents (Pinecone)
                                                              ├── find_columns (SchemaLinker)
                                                              └── explore_column (DB introspection)
```

### Key Components

**app.py** - FastAPI entry point, handles Teams webhook at `/api/messages`, manages Redis conversation state, typing indicators, and Bot Framework authentication.

**rag/search.py (RAGSearch)** - Main orchestrator. Routes queries to Agent, manages conversation history from Redis, falls back to direct Pinecone search if agent fails.

**rag/langgraph_agent.py (LangGraphAgent)** - RECOMMENDED: LangGraph ReAct agent using `create_react_agent()`. Automatic tool loop, Redis checkpointing for conversation state. Default when USE_LANGGRAPH_AGENT=true.

**rag/single_agent_clean.py (CleanSingleAgent)** - RECOMMENDED: Simplified AI agent with tool-calling. Uses a single clear system prompt, lets LLM naturally choose tools, and avoids aggressive post-processing. This is the default agent (USE_CLEAN_AGENT=true).

**rag/single_agent.py (SingleAgent)** - LEGACY: Full-featured agent with planning, SQL verification, answer guards, etc. Over-engineered with 2700+ lines. Use only if USE_CLEAN_AGENT=false.

**rag/postgres.py (PostgresService)** - PostgreSQL interface with safety checks. `prepare_readonly_sql()` validates queries (SELECT-only, no dangerous keywords). Equipment table is `sema_matrix.equipment_matrix` with ~2400 records.

**rag/schema.py** - Database schema documentation. Defines core columns (id, bezeichnung, hersteller_name, geraetegruppe_name, verwendung_code, etc.) and property columns (prop_e####_name_unit format).

**rag/config.py (RAGConfig)** - Central configuration from environment variables. All model/feature settings flow through here.

### Data Flow for Queries

1. User message arrives at `/api/messages`
2. `RAGSearch.search_and_generate()` retrieves conversation history from Redis
3. `SingleAgent.process()` builds prompt with system instructions, thread context, and reduced schema
4. Agent may call `execute_sql` (→ PostgresService) or `search_documents` (→ PineconeStore)
5. Tool results are processed, response is formatted
6. Conversation turn is stored in Redis for follow-ups

### Thread State Management

Follow-up queries rely on `_thread_state` dict in SingleAgent, keyed by `thread_key = "{user_id}:{conversation_id}"`. Stores:
- `last_result_ids`: Previous SQL result IDs for WHERE IN clauses
- `last_sql_purpose`: What the last query was trying to do
- `target_width_m`: Extracted numeric constraints from user queries

## Configuration

All settings via environment variables. Required:
- `OPENAI_API_KEY`, `OPENAI_MODEL`, `REASONING_EFFORT`
- `BOT_APP_ID`, `BOT_APP_PASSWORD`, `AZURE_TENANT_ID`
- `PINECONE_API_KEY`, `PINECONE_HOST`
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_SCHEMA`, `POSTGRES_EQUIPMENT_TABLE`, `POSTGRES_USER`, `POSTGRES_PASSWORD`

Key optional toggles:
- `USE_LANGGRAPH_AGENT=true` - Use LangGraph ReAct agent (recommended, default)
- `USE_AGENT_SYSTEM=true` - Use internal RAG system
- `USE_CLEAN_AGENT=true` - Use simplified agent (recommended, default)
- `AGENT_VERBOSE=true` - Detailed logging
- `AGENT_ENABLE_PLANNING=false` - Query planning (off by default)
- `AGENT_ENABLE_SQL_VERIFICATION=false` - SQL auto-correction (off by default)
- `AGENT_ENABLE_REASONING_TOOLS=false` - Extra reasoning tools (off by default)

## Database Schema Notes

- Equipment table: `sema_matrix.equipment_matrix`
- Categories in `geraetegruppe_name`: Kettenfertiger, Radfertiger, Kettenbagger, Mobilbagger, etc.
- Usage filter: `verwendung_code = 'MIET'` (rental) or `'VK'` (sale)
- Property columns are TEXT with German number formatting (commas as decimals)
- Numeric comparison requires: `CAST(NULLIF(REPLACE(regexp_replace(col, '[^0-9,]', '', 'g'), ',', '.'), '') AS NUMERIC)`

## Testing

Use `cli_tester.py` for all testing:
- Interactive mode: `python cli_tester.py` then type queries
- Commands in interactive: `/sql`, `/search`, `/stats`, `/schema`, `/batch`, `/clear`, `/export`
- Batch mode runs predefined test cases and reports pass/fail

## Code Patterns

- SQL validation always goes through `PostgresService.prepare_readonly_sql()`
- Clean agent uses single system prompt with tool_choice="auto" (lets LLM decide)
- Follow-up context stored as `last_result_ids` for WHERE IN clauses
- Trust LLM reasoning instead of code-level workarounds

## Architecture Principles (Lessons Learned)

1. **Single System Prompt** - Multiple injected system messages create conflicting instructions
2. **Let LLM Choose Tools** - Forced tool_choice prevents natural reasoning
3. **Minimal Post-Processing** - Aggressive response cleaning corrupts valid data
4. **Simple State** - Only track what's needed (last_result_ids for follow-ups)
5. **Trust Tool Results** - Don't re-query/validate unless truly necessary
