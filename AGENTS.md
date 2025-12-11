# Repository Guidelines

Contributor quickstart for the Teams-BOT single-agent FastAPI backend. Keep changes small, documented, and tested before raising a PR.

## Project Structure & Module Organization
- `app.py` hosts the FastAPI app, Teams webhook, and dependency wiring; `commands.py` holds chat command handlers; `cli_test.py` is a lightweight manual runner.
- `rag/` contains retrieval + response code (`search.py`, `embeddings.py`, `postgres.py`, `vector_store.py`, `unified_agent.py`). The unified agent in `unified_agent.py` is the only flow; `search.py` routes through it and falls back to direct Pinecone search if needed.
- `scripts/` includes ingestion/indexing utilities (e.g., `index_documents.py`, `index_machinery.py`).
- `tests/` stores LLM-driven regression scripts (`simple_test.py`, `test_agent.py`) and prompt files such as `TEST_15_ESSENTIAL_QUESTIONS.txt`.
- `docs/`, `SETUP_GUIDE.md`, and `.env.example` provide setup and operational notes.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && .\.venv\Scripts\activate` (Windows) then `pip install -r requirements.txt`.
- Run API: `python app.py` (port 8001) or `uvicorn app:app --reload --port 8000`.
- CLI smoke test: `python cli_test.py` to exercise the bot loop without Teams.
- Single-agent RAG path (default): ensure `USE_SINGLE_AGENT=true` (default) and run `python tests/simple_test.py -f tests/TEST_15_ESSENTIAL_QUESTIONS.txt` (requires OpenAI/Pinecone/PostgreSQL).

## Coding Style & Naming Conventions
- Follow PEP 8, 4-space indent, type hints, and concise docstrings on public functions; keep async I/O for networked calls.
- Classes use PascalCase; functions/vars use snake_case; constants are UPPER_SNAKE.
- Prefer small, composable functions; keep agent-specific logic inside `rag/agents/*` and shared data access in `rag/postgres.py` or `rag/vector_store.py`.
- Document non-obvious control flow (agent selection, retries) with short comments; avoid noisy logging.

## Testing Guidelines
- Tests depend on live services; set `OPENAI_API_KEY`, `PINECONE_*`, `POSTGRES_*`, and bot credentials in `.env`. Use staging resources; never point to production.
- Add new scenario files under `tests/` and reuse `simple_test.py` for semantic scoring; include IDs and expected key facts in prompt files.
- For deterministic checks, stub external calls where possible; otherwise note expected variance in the test description.

## Commit & Pull Request Guidelines
- Commits: imperative, scoped messages (e.g., `Add Pinecone batch upsert helper`). Group refactors and features separately.
- PRs: include a short summary, linked issue/ticket, config changes (.env keys, ports), and screenshots or curl output for user-visible changes.
- List the commands you ran (tests, lint, manual scripts) in the PR body; call out any failing or skipped checks and why.

## Security & Configuration Tips
- Do not commit secrets or personal data; keep `.env` local and use `.env.example` for new keys.
- Validate new agents for safe tool use (no unbounded SQL or external calls without guards). Sanitize user inputs passed to database or vector queries.
- Large data exports and backups belong outside the repo; prefer reproducible scripts in `scripts/` and document required buckets/paths.
