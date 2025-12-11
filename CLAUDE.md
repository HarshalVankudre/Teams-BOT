# Teams-BOT Notes (Unified Agent)

FastAPI backend that connects Microsoft Teams to OpenAI via a single unified agent. The agent can call three tools: SQL (PostgreSQL equipment DB), Pinecone semantic search, and optional Tavily web search. Multi-agent orchestration has been removed for speed and simplicity.

## Development Commands
- Install deps: `python -m venv .venv && .\.venv\Scripts\activate && pip install -r requirements.txt`
- Run API: `python app.py` (port 8001) or `uvicorn app:app --reload --port 8000`
- CLI smoke test: `python cli_test.py`
- RAG regression: `python tests/simple_test.py -f tests/TEST_15_ESSENTIAL_QUESTIONS.txt`
- Unified-agent demo: `python tests/test_agent.py`

## RAG Flow (Production)
1. Teams -> FastAPI webhook (`/api/messages`)
2. Unified agent (`rag/unified_agent.py`) builds a single LLM call with tools: `execute_sql`, `semantic_search`, optional `web_search`
3. Tool results are combined and answered in a follow-up turn; Redis stores short conversation context.
4. `rag/search.py` orchestrates the unified agent and falls back to direct Pinecone search on errors.

## Key Files
- `rag/unified_agent.py` — single-agent logic and tool handlers
- `rag/search.py` — entrypoint for unified agent + Pinecone fallback
- `rag/postgres.py` — PostgreSQL access
- `rag/vector_store.py` — Pinecone access
- `app.py` — FastAPI app, Teams webhook, streaming responses
- `cli_test.py` — interactive CLI tester for the unified agent

## Environment Variables
- Required: `OPENAI_API_KEY`, `OPENAI_MODEL`, `REASONING_EFFORT`
- RAG: `USE_CUSTOM_RAG=true`, `USE_SINGLE_AGENT=true`
- Pinecone: `PINECONE_API_KEY`, `PINECONE_HOST`, `PINECONE_NAMESPACE`, `PINECONE_MACHINERY_NAMESPACE`
- PostgreSQL: `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
- Redis: `REDIS_URL`, `CONVERSATION_TTL_HOURS`
- Tavily (optional): `TAVILY_API_KEY`, `ENABLE_WEB_SEARCH`
