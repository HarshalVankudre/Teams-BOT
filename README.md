# RUEKO Teams Bot

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--5-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![Microsoft Teams](https://img.shields.io/badge/Microsoft_Teams-6264A7?style=for-the-badge&logo=microsoft-teams&logoColor=white)](https://teams.microsoft.com)
[![Google Cloud](https://img.shields.io/badge/Google_Cloud_Run-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)](https://cloud.google.com/run)
[![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)

A production-grade FastAPI backend that connects Microsoft Teams to OpenAI using a sophisticated **Multi-Agent Architecture**. The RUEKO AI Assistant processes messages from Teams through specialized AI agents and returns intelligent responses with full multi-turn conversation support for construction equipment management.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Bot Commands](#bot-commands)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### Multi-Agent Intelligence

| Agent | Model | Description |
|-------|-------|-------------|
| **Orchestrator Agent** | GPT-5 (Reasoning) | Analyzes queries using advanced reasoning and routes to appropriate sub-agents |
| **SQL Agent** | GPT-5 (Non-reasoning) | Generates and executes SQL queries against PostgreSQL equipment database |
| **Pinecone Agent** | GPT-5 (Non-reasoning) | Semantic search for recommendations and scenario-based queries |
| **Web Search Agent** | Tavily API | External information retrieval for supplementary data |
| **Reviewer Agent** | GPT-5 (Reasoning) | Synthesizes results from all agents into coherent, formatted responses |

**Key Capabilities:**
- Modular agent system with self-registration via decorators
- Parallel execution of independent agents for optimal performance
- Dynamic agent discovery from registry
- Smart query routing based on intent analysis

### Equipment Database

- PostgreSQL database with **2,395+ construction equipment records** (Baumaschinen)
- Tracks manufacturers: Caterpillar, Liebherr, Bomag, Vogele, Hamm, Wirtgen, Kubota, Volvo, Hitachi, Komatsu
- Equipment types: excavators (Bagger), rollers (Walzen), pavers (Fertiger), milling machines (Fraesen)
- Detailed specifications stored in JSONB: weight, power, dimensions, features, availability status

### Enterprise Features

| Feature | Description |
|---------|-------------|
| **OpenAI Responses API** | Latest synchronous API with streaming - no polling required |
| **Multi-Turn Conversations** | Full context preservation via `previous_response_id` for natural dialogue |
| **Redis Persistence** | Conversation storage with configurable TTL expiration (default: 24 hours) |
| **Per-User Isolation** | Individual conversation threads even in group chats |
| **Typing Indicators** | Continuous visual feedback during AI processing (every 2.5 seconds) |
| **Feedback Tracking** | Analytics and conversation logging for improvement |
| **Graceful Fallback** | In-memory storage when Redis is unavailable |

### Cloud-Native Design

- Docker containerization with optimized layer caching
- Google Cloud Run deployment ready
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
+---------------------------------------------------------------------------------+
|                           FastAPI Backend (app.py)                              |
|                                                                                 |
|  +----------------+    +-----------------+    +------------------+              |
|  | Token Caching  |    | Redis Session   |    | HTTP Connection  |              |
|  | (Bot Framework)|    | Management      |    | Pooling          |              |
|  +----------------+    +-----------------+    +------------------+              |
+---------------------------------------------------------------------------------+
                                              |
                                              v
+---------------------------------------------------------------------------------+
|                           Multi-Agent System                                    |
|                                                                                 |
|  +-------------------+                                                          |
|  |    Orchestrator   |  <-- GPT-5 Reasoning Model                               |
|  |       Agent       |      Analyzes queries, plans execution                   |
|  +---------+---------+                                                          |
|            |                                                                    |
|            +------------+------------+-------------+                            |
|            |            |            |             |                            |
|            v            v            v             v                            |
|  +---------+---+ +------+-----+ +----+------+ +----+---------+                  |
|  |  SQL Agent  | |  Pinecone  | | Web Search| |   Reviewer   |                  |
|  |             | |   Agent    | |   Agent   | |    Agent     |                  |
|  +------+------+ +------+-----+ +-----+-----+ +------+-------+                  |
|         |               |             |              |                          |
|         v               v             v              v                          |
|  +------+------+ +------+-----+ +-----+-----+ +------+-------+                  |
|  | PostgreSQL  | |  Pinecone  | |  Tavily   | |  Synthesizes |                  |
|  |  Database   | | Vector DB  | |   API     | |   Response   |                  |
|  +-------------+ +------------+ +-----------+ +--------------+                  |
+---------------------------------------------------------------------------------+
```

### Message Flow

1. Teams sends POST to `/api/messages` via Bot Framework
2. Bot extracts user message and removes @mentions
3. Commands (starting with `/`) are routed to command handler
4. Regular messages: Orchestrator Agent analyzes query intent using GPT-5
5. Orchestrator plans which sub-agents to invoke
6. Sub-agents execute in parallel (SQL, Pinecone, Web Search)
7. Reviewer Agent synthesizes all results into coherent response
8. Response sent back to Teams with conversation context preserved

### Query Routing Logic

| Query Type | Example | Agent Used |
|------------|---------|------------|
| Counting | "Wie viele Bagger haben wir?" | SQL Agent |
| Filtering | "Geraete mit Klimaanlage" | SQL Agent |
| Comparison | "Kettenbagger vs Mobilbagger" | SQL Agent |
| Lookup | "Zeige CAT 320" | SQL Agent |
| Recommendations | "Beste Maschine fuer 9m Strasse" | Pinecone Agent |
| Scenarios | "Was fuer enge Baustellen?" | Pinecone Agent |
| External Info | "Aktuelle Preise" | Web Search Agent |

---

## Tech Stack

| Category | Technology |
|----------|------------|
| **Backend Framework** | FastAPI 0.115 + Uvicorn |
| **AI Models** | OpenAI GPT-5 / GPT-4o (Responses API) |
| **Database** | PostgreSQL (psycopg2-binary) |
| **Vector Database** | Pinecone |
| **Session Storage** | Redis |
| **Web Search** | Tavily API |
| **Bot Framework** | Microsoft Bot Framework |
| **Document Processing** | PyMuPDF, python-docx, openpyxl, pandas |
| **Containerization** | Docker |
| **Cloud Platform** | Google Cloud Run |
| **Authentication** | Azure AD (Single-tenant) |

---

## Prerequisites

Before you begin, ensure you have the following:

- **Python 3.11+** installed
- **PostgreSQL database** with equipment data schema
- **Pinecone account** and configured index
- **Redis instance** (optional, falls back to in-memory storage)
- **Azure Bot Service** registration with App ID and Password
- **OpenAI API key** with access to GPT-5 or GPT-4o models
- **Tavily API key** (optional, for web search functionality)

---

## Installation

### Local Setup

1. **Clone the repository**

```bash
git clone https://github.com/HarshalVankudre/Teams-BOT.git
cd Teams-BOT
```

2. **Create virtual environment**

```bash
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Create environment configuration**

```bash
cp .env.example .env
```

Edit the `.env` file with your credentials (see [Configuration](#configuration) section).

5. **Set up directory structure** (for document processing)

```
teams-bot/
├── documents/
│   ├── original/        # Place source files here
│   ├── enriched/        # Processed PDFs
│   └── processed/       # Converted formats
└── logs/
    ├── pdf_enrichment.log
    ├── document_upload.log
    └── file_processing.log
```

6. **Run the application**

```bash
python app.py
```

The server starts on `http://localhost:8001`

---

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

#### Required - OpenAI

```env
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_MODEL=gpt-5-nano                    # gpt-5, gpt-5-mini, gpt-5-nano, gpt-4o
REASONING_EFFORT=medium                     # minimal, low, medium, high
```

#### Required - Azure Bot Service

```env
BOT_APP_ID=your-bot-app-id
BOT_APP_PASSWORD=your-bot-app-password
AZURE_TENANT_ID=your-azure-tenant-id
```

#### Required - PostgreSQL Database

```env
POSTGRES_HOST=your-postgres-host
POSTGRES_PORT=5432
POSTGRES_DB=sema
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your-password
```

#### Required - Pinecone Vector Database

```env
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_HOST=https://your-index.pinecone.io
PINECONE_NAMESPACE=rueko-documents
PINECONE_MACHINERY_NAMESPACE=machinery-data
```

#### Optional - Redis (Conversation Persistence)

```env
REDIS_URL=redis://localhost:6379
CONVERSATION_TTL_HOURS=24
```

#### Optional - Web Search (Tavily)

```env
TAVILY_API_KEY=tvly-your-api-key
ENABLE_WEB_SEARCH=true
WEB_SEARCH_MAX_RESULTS=3
```

#### Optional - Agent System

```env
USE_CUSTOM_RAG=true
USE_AGENT_SYSTEM=true
AGENT_PARALLEL_EXECUTION=true
AGENT_VERBOSE=false
```

#### Advanced Configuration

```env
CHUNKING_MODEL=gpt-5-nano
CHUNKING_REASONING=minimal
EMBEDDING_MODEL=text-embedding-3-large
SEARCH_TOP_K=5
RERANK_TOP_N=3
VECTOR_STORE_ID=vs_...
```

---

## Bot Commands

The RUEKO Teams Bot supports both German and English commands:

### Document Management

| Command | Alias | Description |
|---------|-------|-------------|
| `/liste` | `/list` | Display all documents in the knowledge database with filename, size, upload date |
| `/suchen <term>` | `/search <term>` | Search documents by filename (e.g., `/suchen urlaub`) |
| `/hochladen` | `/upload` | Information about document uploading (admin feature) |
| `/loeschen` | `/delete` | Information about document deletion (admin feature) |

### Conversation Management

| Command | Alias | Description |
|---------|-------|-------------|
| `/zuruecksetzen` | `/reset` | Clear conversation history and start fresh |
| `/feedback <text>` | `/rueckmeldung <text>` | Submit feedback about the bot's response |

### Information

| Command | Alias | Description |
|---------|-------|-------------|
| `/hilfe` | `/help` | Display comprehensive help with all commands and capabilities |
| `/status` | - | Show system information: AI model, document count, feature status |

### Supported File Formats

The bot can process: PDF, DOCX, XLSX, JSON, CSV, TXT

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

Test the bot without Teams integration:

```bash
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
# Example: https://your-subdomain.ngrok-free.dev/api/messages
```

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Basic health check - returns status confirmation |
| `GET` | `/health` | Detailed health with model info, Redis status, and active conversation count |
| `POST` | `/api/messages` | Bot Framework webhook for Teams messages |
| `POST` | `/api/reset-conversation` | Reset conversation history |

### Health Check Response

```json
{
  "status": "healthy",
  "model": "gpt-5-nano",
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

Or send an empty body `{}` to reset all conversations.

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
  --set-env-vars="OPENAI_API_KEY=sk-xxx,OPENAI_MODEL=gpt-5-nano,REASONING_EFFORT=medium"
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

The container exposes port 8080 and uses `uvicorn` to run the application.

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
Teams-BOT/
|-- app.py                    # Main FastAPI application
|-- commands.py               # Bot command handlers (/hilfe, /status, etc.)
|-- cli_test.py               # CLI testing tool for local development
|-- requirements.txt          # Python dependencies
|-- Dockerfile                # Container configuration (Python 3.11-slim)
|-- manifest.json             # Teams app manifest
|-- .env.example              # Environment variables template
|-- startup.sh                # Container startup script
|-- color.png                 # Teams app color icon
|-- outline.png               # Teams app outline icon
|
|-- rag/                      # RAG and Agent System
|   |-- __init__.py           # Package initialization
|   |-- config.py             # Configuration management
|   |-- search.py             # RAG search coordinator
|   |-- embeddings.py         # OpenAI embeddings service
|   |-- vector_store.py       # Pinecone integration
|   |-- postgres.py           # PostgreSQL equipment queries
|   |-- chunker.py            # Document chunking
|   |-- processor.py          # Document processing
|   |-- schema.py             # Data schema definitions
|   |-- feedback.py           # Feedback tracking system
|   |
|   |-- agents/               # Multi-Agent System
|       |-- __init__.py       # Agent exports
|       |-- base.py           # Base agent class and context
|       |-- registry.py       # Agent registry with decorators
|       |-- agent_system.py   # Main coordinator
|       |-- orchestrator.py   # Query analysis and routing (GPT-5)
|       |-- reviewer_agent.py # Response synthesis
|       |-- subagents/        # Specialized sub-agents
|
|-- scripts/                  # Utility scripts
|   |-- index_documents.py    # Document indexing to Pinecone
|   |-- index_machinery.py    # Equipment data indexing
|
|-- tests/                    # Test files
|   |-- simple_test.py
|   |-- debug_queries.py
|   |-- test_agent.py
|
|-- docs/                     # Additional documentation
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
- Use type hints for function parameters and return values
- Write docstrings for public functions and classes
- Add tests for new features

---

## License

This project is proprietary software developed for RUEKO GmbH.

---

## Acknowledgments

- [OpenAI](https://openai.com) for GPT models and Responses API
- [FastAPI](https://fastapi.tiangolo.com) for the excellent web framework
- [Pinecone](https://pinecone.io) for vector database services
- [Microsoft Bot Framework](https://dev.botframework.com) for Teams integration
- [Google Cloud](https://cloud.google.com) for Cloud Run hosting
- [Tavily](https://tavily.com) for web search API

---

<p align="center">
  <strong>RUEKO AI Assistant</strong> - Powered by OpenAI<br>
  Built for RUEKO GmbH construction equipment management
</p>
