# :robot: Teams-BOT | Microsoft Teams AI Assistant

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![Microsoft Teams](https://img.shields.io/badge/Microsoft%20Teams-6264A7?style=for-the-badge&logo=microsoft-teams&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-0078D4?style=for-the-badge&logo=microsoft-azure&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white)

**A powerful FastAPI backend that seamlessly connects Microsoft Teams to OpenAI's cutting-edge Responses API**

*Intelligent conversations. Enterprise-ready. Lightning fast.*

[Features](#-features) | [Quick Start](#-quick-start) | [Architecture](#-architecture) | [Deployment](#-deployment) | [API Reference](#-api-reference)

---

</div>

## :sparkles: Features

| Feature | Description |
|---------|-------------|
| :brain: **OpenAI Responses API** | Uses the latest synchronous Responses API - no polling, instant responses |
| :speech_balloon: **Multi-Turn Conversations** | Full conversation context via `previous_response_id` for natural dialogue |
| :file_folder: **RAG Integration** | Custom Retrieval-Augmented Generation with Pinecone vector database |
| :mag: **Document Search** | Built-in file search with OpenAI's vector store for enterprise knowledge |
| :zap: **Streaming Responses** | Real-time streaming for responsive user experience |
| :floppy_disk: **Redis Persistence** | Persistent conversation storage with automatic TTL expiration |
| :arrows_counterclockwise: **Graceful Fallback** | In-memory storage when Redis is unavailable |
| :keyboard: **Typing Indicators** | Continuous typing feedback during AI processing |
| :shield: **Enterprise Auth** | Single-tenant Azure AD with Bot Framework OAuth |
| :rocket: **Azure Ready** | One-command deployment to Azure App Service |

---

## :building_construction: Architecture

```
+----------------+     +------------------+     +----------------+
|                |     |                  |     |                |
| Microsoft Teams| --> |  FastAPI Backend | --> |  OpenAI API    |
|                |     |                  |     |  Responses API |
+----------------+     +------------------+     +----------------+
                              |
                              v
                    +------------------+
                    |                  |
                    |  Redis / Memory  |
                    |  (Conversations) |
                    |                  |
                    +------------------+
                              |
                              v
                    +------------------+
                    |                  |
                    |  Pinecone        |
                    |  (Vector Store)  |
                    |                  |
                    +------------------+
```

### Message Flow

1. **Receive**: Teams sends message to `/api/messages` via Bot Framework
2. **Process**: Bot extracts user message, removes @mentions
3. **Search**: Optional RAG search in Pinecone or OpenAI vector store
4. **Generate**: OpenAI Responses API generates contextual response
5. **Store**: Response ID saved for multi-turn conversation continuity
6. **Reply**: Response sent back to Teams via Bot Framework REST API

---

## :rocket: Quick Start

### Prerequisites

- Python 3.11+
- Azure Bot Service registration
- OpenAI API key
- Redis (optional, for persistent storage)
- Pinecone account (optional, for custom RAG)

### Installation

```bash
# Clone the repository
git clone https://github.com/HarshalVankudre/Teams-BOT.git
cd Teams-BOT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.template .env
# Edit .env with your credentials
```

### Configuration

Create a `.env` file with the following variables:

```env
# Required
OPENAI_API_KEY=sk-your-openai-api-key
BOT_APP_ID=your-azure-bot-app-id
BOT_APP_PASSWORD=your-azure-bot-password

# Optional - Model Configuration
OPENAI_MODEL=gpt-4o                    # Default: gpt-4o
REASONING_EFFORT=low                    # none, low, medium, high
VECTOR_STORE_ID=vs_your_vector_store   # OpenAI vector store ID

# Optional - Redis for persistent storage
REDIS_URL=redis://localhost:6379
CONVERSATION_TTL_HOURS=24

# Optional - Custom RAG with Pinecone
USE_CUSTOM_RAG=true
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX=your-index-name

# Optional - Custom system instructions
SYSTEM_INSTRUCTIONS="You are a helpful AI assistant..."
```

### Run Locally

```bash
# Start the server (port 8001)
python app.py

# Or with uvicorn (port 8000)
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Test the Bot

```bash
# Health check
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","model":"gpt-4o","redis":"connected","active_conversations":0}
```

For local testing with Teams, use [ngrok](https://ngrok.com):

```bash
ngrok http 8000
# Use the ngrok URL as your messaging endpoint in Azure Bot Service
```

---

## :cloud: Deployment

### Azure App Service (Recommended)

```bash
# Login to Azure
az login

# Create App Service
az webapp up \
  --name your-teams-bot \
  --resource-group YourResourceGroup \
  --runtime PYTHON:3.11 \
  --sku B1

# Configure environment variables
az webapp config appsettings set \
  --name your-teams-bot \
  --resource-group YourResourceGroup \
  --settings \
    OPENAI_API_KEY="your-key" \
    BOT_APP_ID="your-bot-app-id" \
    BOT_APP_PASSWORD="your-password" \
    OPENAI_MODEL="gpt-4o"
```

### Docker Deployment

```bash
# Build the image
docker build -t teams-bot .

# Run the container
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  -e BOT_APP_ID=your-id \
  -e BOT_APP_PASSWORD=your-password \
  teams-bot
```

### Azure Container Apps

```bash
# Create Container Apps environment
az containerapp env create \
  --name teams-bot-env \
  --resource-group YourResourceGroup \
  --location eastus

# Deploy the container
az containerapp create \
  --name teams-bot \
  --resource-group YourResourceGroup \
  --environment teams-bot-env \
  --image ghcr.io/your-username/teams-bot:latest \
  --target-port 8000 \
  --ingress external \
  --env-vars \
    OPENAI_API_KEY=your-key \
    BOT_APP_ID=your-id \
    BOT_APP_PASSWORD=your-password
```

### Configure Bot Messaging Endpoint

1. Go to **Azure Portal** > **Bot Services** > Your Bot
2. Navigate to **Settings** > **Configuration**
3. Set **Messaging endpoint** to: `https://your-app-url/api/messages`
4. Save the configuration

---

## :books: API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Basic health check |
| `GET` | `/health` | Detailed health with model info and Redis status |
| `POST` | `/api/messages` | Bot Framework webhook for Teams messages |
| `POST` | `/api/reset-conversation` | Reset conversation history |

### Reset Conversation

```bash
# Reset specific conversation
curl -X POST http://localhost:8000/api/reset-conversation \
  -H "Content-Type: application/json" \
  -d '{"thread_key": "user123:conversation456"}'

# Reset all conversations
curl -X POST http://localhost:8000/api/reset-conversation \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Health Check Response

```json
{
  "status": "healthy",
  "model": "gpt-4o",
  "redis": "connected",
  "active_conversations": 15
}
```

---

## :file_folder: Project Structure

```
Teams-BOT/
|-- app.py                 # Main FastAPI application
|-- commands.py            # Bot command handlers
|-- requirements.txt       # Python dependencies
|-- Dockerfile             # Container configuration
|-- startup.sh             # Azure startup script
|-- manifest.json          # Teams app manifest
|-- .env.template          # Environment template
|-- CLAUDE.md              # Claude Code instructions
|-- SETUP_GUIDE.md         # Detailed setup guide
|-- rag/                   # Custom RAG implementation
|   |-- __init__.py
|   |-- search.py          # Pinecone search integration
|   |-- chunker.py         # Document chunking
|   |-- embeddings.py      # Embedding generation
|   |-- processor.py       # Document processor
|   |-- vector_store.py    # Vector store operations
|   +-- config.py          # RAG configuration
|-- scripts/
|   +-- index_documents.py # Document indexing script
+-- docs/                  # Documentation files
```

---

## :gear: Advanced Configuration

### Reasoning Models

The bot supports OpenAI's reasoning models with configurable effort levels:

```env
OPENAI_MODEL=gpt-5.1
REASONING_EFFORT=medium  # none, low, medium, high
```

### Custom System Instructions

Customize the AI's behavior with system instructions:

```env
SYSTEM_INSTRUCTIONS="You are a helpful enterprise assistant. Always cite your sources and provide concise answers."
```

### Redis Configuration

For production deployments, configure Redis for persistent conversation storage:

```env
REDIS_URL=redis://username:password@your-redis-host:6379
CONVERSATION_TTL_HOURS=48  # Conversations expire after 48 hours
```

---

## :handshake: Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## :page_facing_up: License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## :email: Support

For questions or support, please open an issue on GitHub or contact the maintainer.

---

<div align="center">

**Built with :heart: using FastAPI and OpenAI**

*Empowering enterprise conversations with AI*

</div>
