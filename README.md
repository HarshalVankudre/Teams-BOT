# Teams-BOT

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--5-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![Microsoft Teams](https://img.shields.io/badge/Microsoft_Teams-6264A7?style=for-the-badge&logo=microsoft-teams&logoColor=white)](https://teams.microsoft.com)
[![Google Cloud](https://img.shields.io/badge/Google_Cloud_Run-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)](https://cloud.google.com/run)
[![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

A production-grade FastAPI backend that connects Microsoft Teams to OpenAI using a **unified single-agent pipeline**. The bot receives messages from Teams via Bot Framework, routes through one agent that can call tools (SQL, Pinecone, optional web search), and returns intelligent responses with full multi-turn conversation support.

> Note: The former multi-agent orchestrator/sub-agent system has been retired in favor of the faster unified agent.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### Unified Agent

- Single GPT-5/4o chat model with tool-calling: SQL (PostgreSQL), semantic search (Pinecone), optional web search (Tavily).
- One LLM turn to decide and call tools, one turn to answer; minimizes latency versus legacy multi-agent orchestration.
- Redis-backed conversation context for short-term memory across turns.
- Automatic safety on SQL (SELECT-only, limit enforcement) and Pinecone namespaces for documents vs. machinery data.

### Equipment Database

- PostgreSQL database with **2,395+ construction equipment records** (Baumaschinen)
- Tracks manufacturers: Caterpillar, Liebherr, Bomag, Vogele, Hamm, Wirtgen, Kubota, Volvo, Hitachi, Komatsu
- Equipment types: excavators, rollers, pavers, milling machines
- Detailed specifications stored in JSONB: weight, power, dimensions, features, availability status

### Enterprise Features

| Feature | Description |
|---------|-------------|
| **OpenAI Responses API** | Latest synchronous API with streaming - no polling required |
| **Multi-Turn Conversations** | Full context via `previous_response_id` for natural dialogue |
| **Redis Persistence** | Conversation storage with configurable TTL expiration |
| **Per-User Isolation** | Individual conversation threads even in group chats |
| **Typing Indicators** | Continuous feedback during AI processing |
| **Feedback Tracking** | Analytics and conversation logging |
| **Graceful Fallback** | In-memory storage when Redis is unavailable |

### Cloud-Native Design

- Docker containerization
- Google Cloud Run deployment
- Azure Bot Framework integration
- Connection pooling and token caching
- HTTP client pooling for high performance

---

## Architecture

```
                         +-------------------+
                         |  Microsoft Teams  |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         |   Bot Framework   |
                         +---------+---------+
                                   |
                                   v
+-------------------------------------------------------------------+
|                        FastAPI Backend (app.py)                   |
|   - Bot token caching                                             |
|   - Redis conversation state (optional)                           |
|   - HTTP connection pooling                                       |
+-------------------------------------------------------------------+
                                   |
                                   v
+-------------------------------------------------------------------+
|                        Unified Agent (LLM)                        |
|   - Single chat completion with tool-calling                      |
|   - Tools: execute_sql, semantic_search, optional web_search      |
|   - One tool round + one answer round                             |
+-------------------------------------------------------------------+
    |                  |                         |
    v                  v                         v
PostgreSQL         Pinecone                Tavily (optional)
(structured)       (documents +            (external web)
                   machinery namespaces)
```

### Message Flow

1. Teams sends POST to `/api/messages` via Bot Framework
2. Bot extracts user message and removes @mentions
3. Orchestrator Agent analyzes query intent using GPT-5
4. Orchestrator plans which sub-agents to invoke
5. Sub-agents execute in parallel (SQL, Pinecone, Web Search)
6. Reviewer Agent synthesizes all results into coherent response
7. Response sent back to Teams with conversation context preserved

---

## Tech Stack

| Category | Technology |
|----------|------------|
| **Backend Framework** | FastAPI + Uvicorn |
| **AI Models** | OpenAI GPT-5 / GPT-4o (Responses API) |
| **Database** | PostgreSQL (psycopg2) |
| **Vector Database** | Pinecone |
| **Session Storage** | Redis |
| **Web Search** | Tavily API |
| **Bot Framework** | Microsoft Bot Framework |
| **Document Processing** | PyMuPDF, python-docx, openpyxl |
| **Containerization** | Docker |
| **Cloud Platform** | Google Cloud Run |
| **Authentication** | Azure AD (Single-tenant) |

---

## Installation

### Prerequisites

- Python 3.11+
- PostgreSQL database with equipment data
- Pinecone account and index
- Redis instance (optional, falls back to in-memory)
- Azure Bot Service registration
- OpenAI API key

### Local Setup

1. **Clone the repository**

```bash
git clone https://github.com/HarshalVankudre/Teams-BOT.git
cd Teams-BOT
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a `.env` file in the project root (see [Configuration](#configuration) section).

5. **Run the application**

```bash
python app.py
```

The server starts on `http://localhost:8001`

---

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

#### Required - OpenAI

```env
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_MODEL=gpt-5
REASONING_EFFORT=medium
```

#### Required - Azure Bot Service

```env
BOT_APP_ID=your-bot-app-id
BOT_APP_PASSWORD=your-bot-app-password
AZURE_TENANT_ID=your-azure-tenant-id
```

#### Required - Database

```env
POSTGRES_HOST=your-postgres-host
POSTGRES_PORT=5432
POSTGRES_DB=your-database-name
POSTGRES_USER=your-username
POSTGRES_PASSWORD=your-password
```

#### Required - Pinecone

```env
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_HOST=your-pinecone-host
PINECONE_NAMESPACE=your-namespace
```

#### Optional - Redis

```env
REDIS_URL=redis://localhost:6379
CONVERSATION_TTL_HOURS=24
```

#### Optional - Web Search

```env
TAVILY_API_KEY=your-tavily-api-key
```

#### Optional - RAG / Unified Agent

```env
USE_CUSTOM_RAG=true
USE_SINGLE_AGENT=true
```

---

## Usage

### Running Locally

```bash
# Start the server (port 8001)
python app.py

# Or use uvicorn directly (port 8000)
uvicorn app:app --host 0.0.0.0 --port 8000
```

### CLI Testing

```bash
# Interactive testing without Teams
python cli_test.py
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Reset Conversations

```bash
# Reset all conversations
curl -X POST http://localhost:8000/api/reset-conversation \
  -H "Content-Type: application/json" \
  -d "{}"

# Reset specific conversation
curl -X POST http://localhost:8000/api/reset-conversation \
  -H "Content-Type: application/json" \
  -d '{"thread_key": "user123:conversation456"}'
```

### Local Testing with Teams

For local development, use [ngrok](https://ngrok.com) to expose your local server:

```bash
ngrok http 8000
# Use the ngrok URL as your messaging endpoint in Azure Bot Service
```

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check - returns status |
| `GET` | `/health` | Detailed health with model info, Redis status, and active conversation count |
| `POST` | `/api/messages` | Bot Framework webhook for Teams messages |
| `POST` | `/api/reset-conversation` | Reset conversation history |

### Health Check Response

```json
{
  "status": "healthy",
  "model": "gpt-5",
  "redis": "connected",
  "active_conversations": 42
}
```

### Reset Conversation Request

```json
{
  "thread_key": "user123:conversation456"
}
```

Or empty body `{}` to reset all conversations.

---

## Deployment

### Google Cloud Run (Recommended)

1. **Build and deploy**

```bash
gcloud run deploy teams-bot \
  --source . \
  --region=us-central1 \
  --allow-unauthenticated
```

2. **Set environment variables**

```bash
gcloud run services update teams-bot \
  --set-env-vars="OPENAI_API_KEY=sk-xxx,OPENAI_MODEL=gpt-5,REASONING_EFFORT=medium"
```

### Docker

1. **Build the image**

```bash
docker build -t teams-bot .
```

2. **Run locally**

```bash
docker run -p 8080:8080 --env-file .env teams-bot
```

### Azure App Service

```bash
# Deploy using the deployment script (Windows)
deploy.bat

# Or manual deployment
az webapp deployment source config-zip \
  --resource-group Teams \
  --name rueko-teams-bot \
  --src deploy.zip
```

### Configure Bot Messaging Endpoint

1. Go to **Azure Portal** > **Bot Services** > Your Bot
2. Navigate to **Settings** > **Configuration**
3. Set **Messaging endpoint** to: `https://your-app-url/api/messages`
4. Save the configuration

---

## Project Structure

```
teams-bot/
|-- app.py                    # Main FastAPI application
|-- commands.py               # Bot command handlers
|-- cli_test.py               # CLI testing tool
|-- requirements.txt          # Python dependencies
|-- Dockerfile                # Container configuration
|-- .env                      # Environment variables (not committed)
|
|-- rag/                      # RAG and unified agent
|   |-- __init__.py
|   |-- config.py             # Configuration management
|   |-- search.py             # RAG search coordinator
|   |-- embeddings.py         # OpenAI embeddings
|   |-- vector_store.py       # Pinecone integration
|   |-- postgres.py           # PostgreSQL queries
|   |-- chunker.py            # Document chunking
|   |-- processor.py          # Document processing
|   |-- feedback.py           # Feedback tracking
|   |-- unified_agent.py      # Single-agent responder with tools
|
|-- scripts/                  # Utility scripts
|   |-- index_documents.py    # Document indexing to Pinecone
|   |-- index_machinery.py    # Equipment data indexing
|
|-- tests/                    # Test files
    |-- simple_test.py
    |-- debug_queries.py
    |-- test_agent.py
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for public functions
- Add tests for new features

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [OpenAI](https://openai.com) for GPT models and Responses API
- [FastAPI](https://fastapi.tiangolo.com) for the excellent web framework
- [Pinecone](https://pinecone.io) for vector database services
- [Microsoft Bot Framework](https://dev.botframework.com) for Teams integration
- [Google Cloud](https://cloud.google.com) for Cloud Run hosting

---

<p align="center">
  Built with dedication for RUKO construction equipment management
</p>
