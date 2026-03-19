# Teams Equipment Assistant

Backend for a Microsoft Teams bot that answers German-language equipment questions for RUEKO. The system combines structured inventory lookups, document retrieval, and advisory planning in one service.

## What it does

- Queries PostgreSQL for equipment inventory and machine details
- Searches Pinecone for manuals and technical documents
- Uses Gemini-based advisory routing for project-style recommendations
- Keeps per-thread conversation state in Redis when available

## Stack

- FastAPI
- LangGraph
- Google Gemini
- OpenAI embeddings
- PostgreSQL
- Pinecone
- Redis

## Key files

- `app.py` - FastAPI app and Teams webhook entry point
- `commands.py` - command handling
- `rag/search.py` - main routing and orchestration
- `rag/langgraph_tools.py` - retrieval and SQL tools
- `rag/compound_agent.py` - advisory agent
- `admin_dashboard/dashboard.py` - admin dashboard

## Required environment variables

- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_HOST`

## Common optional environment variables

- `REDIS_URL`
- `BOT_APP_ID`
- `BOT_APP_PASSWORD`
- `AZURE_TENANT_ID`
- `PINECONE_NAMESPACE`
- `PINECONE_MACHINERY_NAMESPACE`
- `GEMINI_MODEL`

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with the required variables, then start the API:

```bash
uvicorn app:app --reload
```

## Useful commands

```bash
python -m pytest tests -q
python cli_tester.py
```

## API endpoints

- `GET /health` - service and Redis status
- `POST /api/messages` - Teams message webhook
- `POST /api/reset-conversation` - clear one thread or all stored thread state
