# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A FastAPI backend that connects Microsoft Teams to OpenAI using the **Responses API**. The bot receives messages from Teams via Bot Framework, forwards them to OpenAI's Responses API, and returns responses with full multi-turn conversation support.

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

### Core Components

- **app.py**: Single-file FastAPI application containing:
  - Bot Framework message endpoint (`/api/messages`)
  - OpenAI Responses API integration (synchronous, no polling required)
  - Conversation reset endpoint (`/api/reset-conversation`)
  - OAuth token acquisition for Bot Framework authentication
  - Per-user conversation tracking via `previous_response_id` (in-memory dict `conversation_responses`)

### Message Flow

1. Teams sends POST to `/api/messages` via Bot Framework
2. Bot extracts user message, removes @mentions if present
3. Creates response using OpenAI Responses API with `previous_response_id` for context
4. Stores response ID for multi-turn conversation continuity
5. Sends response back to Teams using Bot Framework REST API

### OpenAI Responses API

The bot uses OpenAI's new Responses API instead of the legacy Assistants API:
- **Synchronous**: No polling required, responses are returned directly
- **Multi-turn**: Uses `previous_response_id` with `store=True` for conversation context
- **Configurable**: Model and system instructions set via environment variables
- **Simpler**: No thread/run management required

### Authentication

- Single-tenant Azure AD app (tenant ID hardcoded in `get_bot_token()`)
- Bot Framework OAuth scope: `https://api.botframework.com/.default`

### Environment Variables

Required in `.env`:
- `OPENAI_API_KEY` - OpenAI API key
- `BOT_APP_ID` - Azure Bot Service App ID
- `BOT_APP_PASSWORD` - Azure Bot Service App Password

Optional:
- `OPENAI_MODEL` - Model to use (default: `gpt-4o`)
- `SYSTEM_INSTRUCTIONS` - Custom system prompt for the AI assistant

## API Endpoints

- `GET /` - Health check, returns status
- `GET /health` - Detailed health with model info and active conversation count
- `POST /api/messages` - Bot Framework webhook for Teams messages
- `POST /api/reset-conversation` - Reset conversation history (body: `{"thread_key": "..."}` or `{}` for all)

## Teams App Manifest

`manifest.json` defines the Teams app configuration. Bot ID matches the Azure Bot Service registration. App package (`rueko-bot-app.zip`) contains manifest + icons for sideloading.
