# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A FastAPI backend that connects Microsoft Teams to OpenAI using a **Multi-Agent System**. The bot receives messages from Teams via Bot Framework, routes them through specialized AI agents, and returns intelligent responses with full multi-turn conversation support.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (port 8001)
python app.py

# Run with uvicorn directly (port 8000)
uvicorn app:app --host 0.0.0.0 --port 8000

# Test health endpoint
curl http://localhost:8000/health

# Reset all conversations
curl -X POST http://localhost:8000/api/reset-conversation -H "Content-Type: application/json" -d "{}"

# Interactive update/deploy tool
python update_bot.py
```

## Deployment

```bash
# Deploy to Azure App Service (Windows)
deploy.bat

# Manual Azure deployment
az webapp deployment source config-zip --resource-group Teams --name rueko-teams-bot --src deploy.zip
```

## Architecture

### Multi-Agent System (NEW)

The bot uses a sophisticated multi-agent architecture located in `rag/agents/`:

```
User Query (Teams)
       ↓
┌─────────────────────────────────────┐
│     ORCHESTRATOR AGENT              │
│     (Reasoning Model)               │
│     - Analyzes query intent         │
│     - Loads conversation context    │
│     - Decides which agents to call  │
└─────────────────────────────────────┘
       ↓ (parallel execution)
┌───────────────┬───────────────┬───────────────┐
│  SQL AGENT    │ PINECONE      │ WEB SEARCH    │
│  (Non-reason) │ AGENT         │ AGENT         │
│               │ (Non-reason)  │ (Optional)    │
│  PostgreSQL   │ Semantic      │ Tavily API    │
│  queries      │ search        │               │
└───────────────┴───────────────┴───────────────┘
       ↓
┌─────────────────────────────────────┐
│     REVIEWER AGENT                  │
│     (Reasoning Model)               │
│     - Reviews all data              │
│     - Formats response              │
│     - Smart display logic           │
│     - Adds follow-up suggestions    │
└─────────────────────────────────────┘
       ↓
Final Response to Teams
```

### Agent Files

- `rag/agents/base.py` - Base agent class and shared types
- `rag/agents/orchestrator.py` - Main reasoning agent that routes queries
- `rag/agents/sql_agent.py` - SQL generation and PostgreSQL queries
- `rag/agents/pinecone_agent.py` - Semantic search in Pinecone
- `rag/agents/web_search_agent.py` - Tavily web search (supplementary)
- `rag/agents/reviewer_agent.py` - Response formatting and review
- `rag/agents/agent_system.py` - Coordinator that orchestrates all agents

### Core Components

- **app.py**: FastAPI application with Bot Framework integration
- **rag/search.py**: Main RAG entry point that routes to Agent System
- **rag/config.py**: Configuration management from environment variables
- **rag/postgres.py**: PostgreSQL service for equipment database
- **rag/embeddings.py**: OpenAI embedding service

### Message Flow

1. Teams sends POST to `/api/messages` via Bot Framework
2. Bot extracts user message, removes @mentions if present
3. **Orchestrator Agent** analyzes query and decides which tools to use
4. **Sub-agents** execute in parallel (SQL, Pinecone, Web Search)
5. **Reviewer Agent** formats the final response
6. Response sent back to Teams via Bot Framework REST API

### Query Routing Logic

| Query Type | Example | Agent Used |
|------------|---------|------------|
| Counting | "Wie viele Bagger?" | SQL Agent |
| Filtering | "Geräte mit Klimaanlage" | SQL Agent |
| Comparison | "Kettenbagger vs Mobilbagger" | SQL Agent |
| Lookup | "Zeige CAT 320" | SQL Agent |
| Recommendations | "Beste Maschine für 9m Straße" | Pinecone Agent |
| Scenarios | "Was für enge Baustellen?" | Pinecone Agent |
| External Info | "Aktuelle Preise" | Web Search Agent |

### Environment Variables

Required in `.env`:
- `OPENAI_API_KEY` - OpenAI API key
- `OPENAI_MODEL` - Model to use (e.g., gpt-4o, o1, o3)
- `REASONING_EFFORT` - Reasoning effort: none, low, medium, high
- `BOT_APP_ID` - Azure Bot Service App ID
- `BOT_APP_PASSWORD` - Azure Bot Service App Password
- `PINECONE_API_KEY` - Pinecone API key
- `PINECONE_HOST` - Pinecone index host URL
- `POSTGRES_*` - PostgreSQL connection settings

Optional:
- `USE_AGENT_SYSTEM` - Enable multi-agent system (default: true)
- `AGENT_PARALLEL_EXECUTION` - Run agents in parallel (default: true)
- `AGENT_VERBOSE` - Enable verbose logging (default: false)
- `ENABLE_WEB_SEARCH` - Enable Tavily web search (default: true)
- `TAVILY_API_KEY` - Tavily API key for web search
- `REDIS_URL` - Redis URL for conversation persistence
- `CONVERSATION_TTL_HOURS` - Conversation TTL (default: 24)

See `.env.example` for all configuration options.

## API Endpoints

- `GET /` - Health check, returns status
- `GET /health` - Detailed health with model info and active conversation count
- `POST /api/messages` - Bot Framework webhook for Teams messages
- `POST /api/reset-conversation` - Reset conversation history (body: `{"thread_key": "..."}` or `{}` for all)

## Database Schema

### PostgreSQL - Equipment Database (geraete)

**Schema updated 2024-12-07**: All properties extracted to direct columns for faster queries.

**Identification & Classification:**
- `id`: BIGINT PRIMARY KEY (SEMA primaryKey)
- `bezeichnung`: Model name (e.g., "CAT 320", "BW 174 AP-5 AM")
- `hersteller`: Manufacturer (Caterpillar, Liebherr, Bomag, etc.)
- `geraetegruppe`: Equipment type - MOST IMPORTANT! (Kettenbagger, Mobilbagger, Tandemwalze, etc.)
- `kategorie`: Category (bagger, lader, verdichter, fertiger, fraese, kran)
- `verwendung`: Usage (Vermietung, Verkauf, Fuhrpark)
- `seriennummer`, `inventarnummer`

**Property Columns (prop_*) - 171 columns for all technical specs:**
- `prop_breite`, `prop_hoehe`, `prop_laenge`, `prop_gewicht` - Dimensions with units (e.g., "1400 mm")
- `prop_motor_leistung` - Motor power (e.g., "129 kW")
- `prop_klimaanlage`, `prop_oszillation` - Features (values: "Ja" or "Nein")
- `prop_abgasstufe_eu`, `prop_motor_hersteller` - Text values
- ... and 160+ more prop_ columns

**Important Notes:**
- German umlauts converted to ASCII: ae, oe, ue, ss (e.g., "Hoehe" not "Hohe")
- Boolean properties use TEXT: `prop_klimaanlage = 'Ja'` (not true/false)
- Values include units: "1400 mm", "620 kg", "129 kW"

**SQL Examples:**
```sql
-- Equipment with air conditioning
SELECT * FROM geraete WHERE prop_klimaanlage = 'Ja'

-- Rollers with oscillation
SELECT * FROM geraete WHERE geraetegruppe ILIKE '%walze%' AND prop_oszillation = 'Ja'
```

### Pinecone Namespaces

- `machinery-data`: Equipment embeddings with metadata
- `rueko-documents`: Company documents and policies

## Teams App Manifest

`manifest.json` defines the Teams app configuration. Bot ID matches the Azure Bot Service registration. App package (`rueko-bot-app.zip`) contains manifest + icons for sideloading.
