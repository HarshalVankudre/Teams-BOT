<p align="center">
  <img src="outline.png" alt="Teams-BOT" width="120" />
</p>

<h1 align="center">Teams-BOT</h1>

<p align="center">
  A production-ready FastAPI backend for a Microsoft Teams bot powered by OpenAI + RAG (PostgreSQL + Pinecone).
</p>

<p align="center">
  <a href="https://python.org"><img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://fastapi.tiangolo.com"><img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white"></a>
  <a href="https://openai.com"><img alt="OpenAI" src="https://img.shields.io/badge/OpenAI-API-412991?style=flat-square&logo=openai&logoColor=white"></a>
  <a href="https://cloud.google.com/run"><img alt="Cloud Run" src="https://img.shields.io/badge/Google_Cloud_Run-4285F4?style=flat-square&logo=google-cloud&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"></a>
</p>

---

## What this is

Teams-BOT connects **Microsoft Teams** (via **Bot Framework**) to an LLM-backed assistant with **tool access**:

- **Structured data**: PostgreSQL (`geraete` view) via safe, SELECT-only SQL
- **Unstructured/internal docs**: Pinecone semantic search (documents + machinery namespaces)
- **Optional web**: Tavily search for supplemental, external info
- **Multi-turn conversations**: Redis-backed history with per-user isolation
- **Better follow-ups**: conversation-aware query rewriting + multi-query semantic retrieval (no hardcoded synonym lists)

> For setup details and operational notes, see [`SETUP_GUIDE.md`](SETUP_GUIDE.md).

---

## Quickstart (local)

### 1) Install

```bash
python -m venv .venv
# Windows:
#   .\.venv\Scripts\activate
# macOS/Linux:
#   source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Configure

Copy `.env.example` -> `.env` and fill in required values.

Minimum required keys:

```env
OPENAI_API_KEY=
OPENAI_MODEL=
REASONING_EFFORT=
BOT_APP_ID=
BOT_APP_PASSWORD=
AZURE_TENANT_ID=
PINECONE_API_KEY=
PINECONE_HOST=
POSTGRES_HOST=
POSTGRES_DB=
POSTGRES_USER=
POSTGRES_PASSWORD=
```

### 3) Run

```bash
# Default: http://localhost:8001
python app.py

# Or:
uvicorn app:app --reload --port 8000
```

Notes:
- `python app.py` listens on `8001` by default.
- Docker/Cloud Run listen on `$PORT` (defaults to `8080` in the `Dockerfile`).

### 4) Smoke-test without Teams

```bash
python cli_test.py
```

---

## Architecture (high-level)

```text
Microsoft Teams
  |
  v
Bot Framework (Azure)
  |  POST /api/messages
  v
FastAPI (app.py)
  |
  v
Unified Agent (rag/unified_agent.py)
  |-- execute_sql         -> PostgreSQL (geraete view)
  |-- semantic_search     -> Pinecone (documents/machinery)
  `-- web_search (opt.)   -> Tavily
```

The unified agent does (typically) **one tool round** + **one answer round** to keep latency low while preserving accuracy.

---

## Conversation & follow-ups

- Threads are isolated by `thread_key = "{user_id}:{conversation_id}"` (safe in group chats).
- Redis stores conversation state with TTL (`CONVERSATION_TTL_HOURS`):
  - `history:{thread_key}`: unified-agent chat history (for follow-ups)
  - `conversation:{thread_key}`: Responses API continuity (fallback mode only)

**Reset**
- In Teams: send `/reset` (or `/zuruecksetzen`)
- Or via API: `POST /api/reset-conversation` with `{ "thread_key": "..." }` or `{}` for all

---

## Configuration notes

All config is via environment variables (`.env.example`).

Useful toggles:

```env
# Use the internal RAG system (recommended)
USE_CUSTOM_RAG=true
USE_SINGLE_AGENT=true

# Improve follow-up understanding + paraphrase retrieval
UNIFIED_AGENT_ENABLE_QUERY_REWRITE=true
UNIFIED_AGENT_MULTI_QUERY_RETRIEVAL=true
UNIFIED_AGENT_MULTI_QUERY_MAX=3
```

---

## Deployment (Cloud Run)

```bash
gcloud run deploy teams-bot \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

Set environment variables (example):

```bash
gcloud run services update teams-bot \
  --region us-central1 \
  --set-env-vars="OPENAI_MODEL=gpt-5,REASONING_EFFORT=medium,USE_CUSTOM_RAG=true,USE_SINGLE_AGENT=true"
```

> Recommended: store secrets in Secret Manager and reference them from Cloud Run instead of pasting keys into CLI history.

---

## Testing

Most tests are **live-service** checks and require environment variables (`.env`) for OpenAI/Pinecone/Postgres (and optionally Redis).

```bash
python tests/test_agent.py
python tests/simple_test.py -f tests/TEST_15_ESSENTIAL_QUESTIONS.txt
```

---

## Endpoints

| Method | Endpoint | Purpose |
|---:|---|---|
| `GET` | `/` | Liveness |
| `GET` | `/health` | Health + Redis status |
| `POST` | `/api/messages` | Teams webhook (Bot Framework) |
| `POST` | `/api/reset-conversation` | Reset thread/all history |

---

## Repo layout

```text
app.py                 FastAPI app + Teams webhook
commands.py            Slash-command handlers (Teams)
cli_test.py            Local interactive runner (no Teams)
rag/                   Unified agent + retrieval + SQL tooling
scripts/               Ingestion / indexing utilities
tests/                 Live-service regression scripts (LLM-driven)
```

---

## Security

- Never commit secrets: keep `.env` local and use `.env.example` as a template.
- Prefer Secret Manager for production credentials (Cloud Run).

---

## License

MIT. See `LICENSE`.
