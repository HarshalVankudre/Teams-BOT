"""
Microsoft Teams Bot Backend with OpenAI Responses API

Improvements:
- Token caching for Bot Framework authentication
- HTTP connection pooling for better performance
- Redis for persistent conversation storage (per-user isolation)
- Graceful fallback to in-memory storage if Redis unavailable
- Custom RAG with Pinecone vector database
- File export with temporary download links
"""
import os
import asyncio
import uuid
import base64
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import httpx
import redis.asyncio as redis
from dotenv import load_dotenv
from commands import handle_command
import time
import io

# Custom RAG imports
from rag.search import RAGSearch
from rag.feedback import feedback_service

# Load environment variables from .env file
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BOT_APP_ID = os.getenv("BOT_APP_ID", "")
BOT_APP_PASSWORD = os.getenv("BOT_APP_PASSWORD", "")
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID", "")  # Required for single-tenant apps

# Model configuration (REQUIRED - from .env, no hardcoded defaults)
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
REASONING_EFFORT = os.getenv("REASONING_EFFORT")
if not OPENAI_MODEL or not REASONING_EFFORT:
    raise ValueError("OPENAI_MODEL and REASONING_EFFORT must be set in .env file")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID", "")
# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CONVERSATION_TTL_HOURS = int(os.getenv("CONVERSATION_TTL_HOURS", "24"))

# RAG configuration
USE_CUSTOM_RAG = os.getenv("USE_CUSTOM_RAG", "true").lower() == "true"
USE_SINGLE_AGENT = os.getenv("USE_SINGLE_AGENT", "true").lower() == "true"

SYSTEM_INSTRUCTIONS = os.getenv("SYSTEM_INSTRUCTIONS", """Du bist RÜKO GPT mit Zugriff auf interne Datenbanken und ergänzende Web-Suche.

DATENPRIORITÄT (WICHTIG):
1. INTERNE DATEN haben IMMER Vorrang:
   - Dokumentendatenbank (rueko-documents): Unternehmensrichtlinien, Anleitungen, Prozesse
   - Maschinendatenbank (machinery-data): 2.395 Baumaschinen mit technischen Daten
2. Web-Suche (Tavily) ist NUR ERGÄNZEND für aktuelle Preise, externe Spezifikationen
3. Bei Widersprüchen: Interne Daten sind IMMER maßgeblich

KERNREGELN:
1. Durchsuche und nutze IMMER zuerst die internen Datenbanken
2. Zitiere Quellen: "Laut [interner Quelle]..." oder "Laut [Web-Quelle]..."
3. Web-Informationen nur als Ergänzung, nie als Hauptquelle

MASCHINENFRAGEN:
- Gib ALLE verfügbaren technischen Details aus der internen Datenbank
- Seriennummer/Inventarnummer: Zeige alle gespeicherten Eigenschaften
- Empfehlungen: Basierend auf Arbeitsbreite, Leistung, Einsatzgebiet aus internen Daten
- Verfügbarkeit: "Vermietung" oder "Verkauf" aus Maschinendatenbank

ANTWORTLÄNGE:
- Einfache Fragen: 2-4 Sätze
- Maschinendaten: Vollständige strukturierte Liste
- Komplexe Fragen: max. 500 Wörter, mit Aufzählungen

FORMAT:
- Direkte Antwort zuerst
- Technische Daten in übersichtlichen Listen
- Quellenangabe am Ende (intern vs. web kennzeichnen)

WENN KEINE INTERNEN DATEN:
"In den internen Datenbanken wurde keine Information gefunden." + ggf. Web-Ergänzung

Erfinde NIEMALS Informationen. Interne Daten = Wahrheit.""")

# Debug output
print(f"Bot App ID loaded: {BOT_APP_ID[:10]}..." if BOT_APP_ID else "Bot App ID NOT loaded!")
print(f"Bot Password loaded: {'Yes' if BOT_APP_PASSWORD else 'No'}")
print(f"OpenAI API Key loaded: {'Yes' if OPENAI_API_KEY else 'No'}")
print(f"Model: {OPENAI_MODEL}")
print(f"Reasoning Effort: {REASONING_EFFORT}")
print(f"Vector Store ID: {VECTOR_STORE_ID}")
print(f"Redis URL: {REDIS_URL[:50]}..." if len(REDIS_URL) > 50 else f"Redis URL: {REDIS_URL}")
print(f"Conversation TTL: {CONVERSATION_TTL_HOURS} hours")
print(f"Custom RAG (Pinecone): {'Enabled' if USE_CUSTOM_RAG else 'Disabled'}")
print(f"Unified Agent: {'Enabled' if USE_SINGLE_AGENT else 'Disabled'}")

# Initialize async OpenAI client for streaming
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Initialize custom RAG search (if enabled) - Redis client added after startup
rag_search = None  # Initialized in lifespan after Redis is available
if USE_CUSTOM_RAG:
    print("Custom RAG will be initialized with Pinecone")

# Token caching for Bot Framework authentication
@dataclass
class TokenCache:
    token: str
    expires_at: datetime

token_cache: TokenCache | None = None

# Fallback in-memory storage (used when Redis unavailable)
conversation_responses: dict[str, str] = {}

# Temporary file storage for downloads (files expire after 10 minutes)
@dataclass
class TempFile:
    file_data: bytes
    file_name: str
    mime_type: str
    created_at: datetime

temp_files: dict[str, TempFile] = {}
FILE_EXPIRY_MINUTES = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown"""
    # Startup: Initialize Redis and HTTP client
    print("Starting up...")

    # Initialize Redis connection pool
    try:
        app.state.redis_pool = redis.ConnectionPool.from_url(
            REDIS_URL,
            max_connections=20,
            decode_responses=True
        )
        # Test connection
        r = redis.Redis(connection_pool=app.state.redis_pool)
        await r.ping()
        app.state.redis_available = True
        print(f"[OK] Redis connected")
    except Exception as e:
        print(f"[WARN] Redis unavailable ({e}), using in-memory storage")
        app.state.redis_pool = None
        app.state.redis_available = False

    # Initialize shared HTTP client for connection pooling
    app.state.http_client = httpx.AsyncClient(
        timeout=30.0,
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
    )
    print("[OK] HTTP client pool initialized")

    # Initialize RAG search with Redis client for conversation context
    global rag_search
    if USE_CUSTOM_RAG:
        redis_client = None
        if app.state.redis_available and app.state.redis_pool:
            redis_client = redis.Redis(connection_pool=app.state.redis_pool)
        rag_search = RAGSearch(redis_client=redis_client)
        print(f"[OK] RAG Search initialized (Unified Agent: {USE_SINGLE_AGENT})")

    yield  # Application runs here

    # Shutdown: Cleanup resources
    print("Shutting down...")
    await app.state.http_client.aclose()
    if app.state.redis_pool:
        await app.state.redis_pool.disconnect()
    print("[OK] Resources cleaned up")


app = FastAPI(title="Teams Bot - OpenAI Responses API", lifespan=lifespan)


# Redis helper functions
async def get_redis(request: Request) -> redis.Redis | None:
    """Get Redis client from app state"""
    if hasattr(request.app.state, 'redis_pool') and request.app.state.redis_pool:
        return redis.Redis(connection_pool=request.app.state.redis_pool)
    return None


async def store_conversation_id(request: Request, thread_key: str, response_id: str):
    """Store conversation ID in Redis with TTL, fallback to memory"""
    r = await get_redis(request)
    if r:
        try:
            await r.setex(
                f"conversation:{thread_key}",
                CONVERSATION_TTL_HOURS * 3600,
                response_id
            )
            return
        except Exception as e:
            print(f"Redis store error: {e}")
    # Fallback to in-memory
    conversation_responses[thread_key] = response_id


async def get_conversation_id(request: Request, thread_key: str) -> str | None:
    """Get conversation ID from Redis, fallback to memory"""
    r = await get_redis(request)
    if r:
        try:
            result = await r.get(f"conversation:{thread_key}")
            if result:
                return result
        except Exception as e:
            print(f"Redis get error: {e}")
    # Fallback to in-memory
    return conversation_responses.get(thread_key)


async def delete_conversation_id(request: Request, thread_key: str):
    """Delete conversation ID from Redis, fallback to memory"""
    r = await get_redis(request)
    if r:
        try:
            await r.delete(f"conversation:{thread_key}")
        except Exception as e:
            print(f"Redis delete error: {e}")
    # Also clear from memory
    conversation_responses.pop(thread_key, None)


async def clear_all_conversations(request: Request):
    """Clear all conversations from Redis, fallback to memory"""
    r = await get_redis(request)
    if r:
        try:
            # Get all conversation keys and delete them
            cursor = 0
            while True:
                cursor, keys = await r.scan(cursor, match="conversation:*", count=100)
                if keys:
                    await r.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            print(f"Redis clear error: {e}")
    # Also clear in-memory
    conversation_responses.clear()


@app.get("/")
async def root():
    return {"status": "ok", "message": "Teams Bot is running"}


@app.get("/health")
async def health(request: Request):
    """Health check with Redis status"""
    redis_status = "unavailable"
    active_conversations = len(conversation_responses)

    r = await get_redis(request)
    if r:
        try:
            await r.ping()
            redis_status = "connected"
            # Count Redis keys for active conversations
            cursor = 0
            count = 0
            while True:
                cursor, keys = await r.scan(cursor, match="conversation:*", count=100)
                count += len(keys)
                if cursor == 0:
                    break
            active_conversations = count
        except Exception:
            redis_status = "error"

    return {
        "status": "healthy",
        "model": OPENAI_MODEL,
        "redis": redis_status,
        "active_conversations": active_conversations
    }


@app.post("/api/reset-conversation")
async def reset_conversation(request: Request):
    """Reset a specific user's conversation history"""
    try:
        body = await request.json()
        thread_key = body.get("thread_key")

        if thread_key:
            await delete_conversation_id(request, thread_key)
            return {"status": "ok", "message": f"Conversation {thread_key} reset"}
        else:
            # Reset all conversations
            await clear_all_conversations(request)
            return {"status": "ok", "message": "All conversations reset"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/download/{file_id}")
async def download_file(file_id: str):
    """Download a temporary file by ID"""
    # Clean up expired files first
    now = datetime.utcnow()
    expired = [fid for fid, f in temp_files.items()
               if (now - f.created_at).total_seconds() > FILE_EXPIRY_MINUTES * 60]
    for fid in expired:
        del temp_files[fid]

    # Check if file exists
    if file_id not in temp_files:
        return Response(
            status_code=404,
            content="File not found or expired"
        )

    temp_file = temp_files[file_id]

    # Return file as download
    return StreamingResponse(
        io.BytesIO(temp_file.file_data),
        media_type=temp_file.mime_type,
        headers={
            "Content-Disposition": f'attachment; filename="{temp_file.file_name}"',
            "Content-Length": str(len(temp_file.file_data))
        }
    )


def store_temp_file(file_data: bytes, file_name: str, mime_type: str) -> str:
    """Store a file temporarily and return its ID"""
    file_id = str(uuid.uuid4())
    temp_files[file_id] = TempFile(
        file_data=file_data,
        file_name=file_name,
        mime_type=mime_type,
        created_at=datetime.utcnow()
    )
    print(f"Stored temp file: {file_id} ({file_name}, {len(file_data)} bytes)")
    return file_id


@app.post("/api/messages")
async def messages(request: Request):
    """Handle incoming messages from Microsoft Teams/Bot Framework"""
    try:
        body = await request.json()
        
        # Log incoming activity
        activity_type = body.get("type", "")
        print(f"Received activity: {activity_type}")
        
        if activity_type == "message":
            # Extract message details
            user_message = body.get("text", "")
            conversation_id = body.get("conversation", {}).get("id", "")
            service_url = body.get("serviceUrl", "")

            # Get user information
            user_info = body.get("from", {})
            user_id = user_info.get("id", "unknown")
            user_name = user_info.get("name", "Unknown User")
            user_email = user_info.get("email", "")  # This might be available

            # Remove bot mention from message if present
            if body.get("entities"):
                for entity in body["entities"]:
                    if entity.get("type") == "mention":
                        mentioned_text = entity.get("text", "")
                        user_message = user_message.replace(mentioned_text, "").strip()

            print(f"User: {user_name} (ID: {user_id})")
            if user_email:
                print(f"Email: {user_email}")
            print(f"Message: {user_message}")
            print(f"Conversation ID: {conversation_id}")

            # Check if message is a command
            if user_message.strip().startswith("/"):
                print(f"Command detected: {user_message}")

                # Helper function to send replies for commands
                async def send_command_reply(body, message):
                    await send_reply(
                        request=request,
                        service_url=service_url,
                        conversation_id=conversation_id,
                        activity_id=body.get("id"),
                        reply_to_id=body.get("id"),
                        recipient=body.get("from"),
                        from_bot=body.get("recipient"),
                        message=message
                    )

                # Route to command handler
                await handle_command(body, user_message, send_command_reply)

            else:
                # Regular conversation - AI response
                # Create unique thread key per user (combines user ID with conversation ID)
                # This ensures each user has their own thread even in group chats
                thread_key = f"{user_id}:{conversation_id}"

                # Start continuous typing indicator (for reasoning models that take longer)
                typing_manager = TypingIndicatorManager(
                    request=request,
                    service_url=service_url,
                    conversation_id=conversation_id,
                    from_bot=body.get("recipient")
                )
                typing_manager.start()

                # Track response time
                start_time = time.time()

                try:
                    # Get response from unified agent (returns dict with 'response' and optional 'file_export')
                    result = await get_assistant_response_streaming(
                        request, thread_key, user_message,
                        user_id=user_id, user_name=user_name
                    )
                finally:
                    # Always stop typing indicator when done
                    typing_manager.stop()

                # Extract response text and optional file export
                assistant_response = result.get("response", "")
                file_export = result.get("file_export")

                # Calculate response time in milliseconds
                response_time_ms = int((time.time() - start_time) * 1000)

                # Store conversation in feedback database
                try:
                    data_source = "unified_agent" if USE_CUSTOM_RAG else "openai_file_search"
                    feedback_service.save_conversation(
                        user_id=user_id,
                        user_message=user_message,
                        ai_response=assistant_response,
                        user_name=user_name,
                        user_email=user_email,
                        conversation_thread_id=thread_key,
                        response_time_ms=response_time_ms,
                        query_type=None,  # Could be determined by RAG system
                        data_source=data_source
                    )
                except Exception as fb_error:
                    print(f"[Feedback] Error storing conversation: {fb_error}")

                # Send reply back to Teams (with optional file attachment)
                await send_reply(
                    request=request,
                    service_url=service_url,
                    conversation_id=conversation_id,
                    activity_id=body.get("id"),
                    reply_to_id=body.get("id"),
                    recipient=body.get("from"),
                    from_bot=body.get("recipient"),
                    message=assistant_response,
                    file_export=file_export
                )

        elif activity_type == "conversationUpdate":
            # Handle when bot is added to conversation
            members_added = body.get("membersAdded", [])
            for member in members_added:
                if member.get("id") != body.get("recipient", {}).get("id"):
                    # A user was added, send welcome message
                    service_url = body.get("serviceUrl", "")
                    conversation_id = body.get("conversation", {}).get("id", "")

                    await send_reply(
                        request=request,
                        service_url=service_url,
                        conversation_id=conversation_id,
                        activity_id=body.get("id"),
                        reply_to_id=body.get("id"),
                        recipient=body.get("from"),
                        from_bot=body.get("recipient"),
                        message="Hallo! Ich bin RÜKO GPT. Wie kann ich Ihnen helfen?"
                    )
        
        return Response(status_code=200)
    
    except Exception as e:
        print(f"Error processing message: {e}")
        return Response(status_code=500, content=str(e))


async def send_typing_indicator(request: Request, service_url: str, conversation_id: str, from_bot: dict):
    """Send typing indicator to Teams to show bot is processing"""
    try:
        token = await get_bot_token(request)

        typing_activity = {
            "type": "typing",
            "from": from_bot,
            "conversation": {"id": conversation_id}
        }

        url = f"{service_url}v3/conversations/{conversation_id}/activities"

        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        # Use pooled HTTP client
        http_client = request.app.state.http_client
        await http_client.post(url, json=typing_activity, headers=headers)
        print("Typing indicator sent")
    except Exception as e:
        print(f"Failed to send typing indicator: {e}")


class TypingIndicatorManager:
    """Manages continuous typing indicators for long-running operations"""

    def __init__(self, request: Request, service_url: str, conversation_id: str, from_bot: dict):
        self.request = request
        self.service_url = service_url
        self.conversation_id = conversation_id
        self.from_bot = from_bot
        self._task = None
        self._stop = False

    async def _send_typing_loop(self):
        """Send typing indicators every 2.5 seconds (Teams expires after ~3s)"""
        while not self._stop:
            await send_typing_indicator(
                self.request, self.service_url, self.conversation_id, self.from_bot
            )
            await asyncio.sleep(2.5)

    def start(self):
        """Start sending typing indicators"""
        self._stop = False
        self._task = asyncio.create_task(self._send_typing_loop())

    def stop(self):
        """Stop sending typing indicators"""
        self._stop = True
        if self._task:
            self._task.cancel()
            self._task = None


async def get_custom_rag_response(
    request: Request,
    thread_key: str,
    user_message: str,
    user_id: str = None,
    user_name: str = None
) -> dict:
    """Get response using the unified agent with conversation continuity.

    Returns:
        dict with 'response' (str) and optionally 'file_export' (dict)
    """
    print(f"Using unified agent (USE_CUSTOM_RAG={USE_CUSTOM_RAG}, USE_SINGLE_AGENT={USE_SINGLE_AGENT})...")

    try:
        # Get previous response ID for conversation continuity (fallback mode)
        previous_response_id = await get_conversation_id(request, thread_key)
        if previous_response_id:
            print(f"Continuing conversation for {thread_key}")

        # Search and generate response using unified agent
        result = await rag_search.search_and_generate(
            query=user_message,
            system_instructions=SYSTEM_INSTRUCTIONS,
            previous_response_id=previous_response_id,
            user_id=user_id,
            user_name=user_name,
            thread_key=thread_key
        )

        response = result["response"]

        # Log agent info
        agents_used = result.get("agents_used", [])
        query_type = result.get("query_type", "unknown")
        execution_time = result.get("execution_time_ms", 0)

        if agents_used:
            print(f"Agents used: {agents_used}")
        print(f"Query type: {query_type}, Execution time: {execution_time}ms")

        # Store response ID for conversation continuity (if using fallback)
        response_id = result.get("response_id")
        if response_id:
            await store_conversation_id(request, thread_key, response_id)
            print(f"Response ID stored for {thread_key}: {response_id}")

        # Add sources if available (only for non-agent responses or minimal sources)
        sources = result.get("sources", [])
        if sources and query_type == "fallback":
            response += "\n\n---\n**Quellen:**"
            for source in sources[:3]:
                score_pct = source.get("score", 0)
                if isinstance(score_pct, float) and score_pct <= 1:
                    score_pct = score_pct * 100
                response += f"\n- {source.get('title', 'Unbekannt')} ({source.get('source_file', '')}) [{score_pct:.0f}%]"

        print(f"Response generated using {result.get('chunks_used', 0)} sources")

        # Return dict with response and optional file export
        result_dict = {"response": response}

        # Check if file export is included
        file_export = result.get("file_export")
        if file_export:
            print(f"File export included: {file_export.get('file_name')}")
            result_dict["file_export"] = file_export

        return result_dict

    except Exception as e:
        print(f"Unified agent error: {e}")
        import traceback
        traceback.print_exc()
        return {"response": f"Fehler bei der Verarbeitung: {str(e)}"}


async def get_assistant_response_streaming(
    request: Request,
    thread_key: str,
    user_message: str,
    user_id: str = None,
    user_name: str = None
) -> dict:
    """Get response from the unified agent or fallback to OpenAI Responses API.

    Returns:
        dict with 'response' (str) and optionally 'file_export' (dict)
    """

    # Use unified agent / Custom RAG if enabled
    if USE_CUSTOM_RAG and rag_search:
        return await get_custom_rag_response(
            request, thread_key, user_message,
            user_id=user_id, user_name=user_name
        )

    # Fallback to OpenAI file_search
    try:
        # Check if we have a previous response ID for this conversation (from Redis or memory)
        previous_response_id = await get_conversation_id(request, thread_key)

        if previous_response_id:
            print(f"Continuing conversation for {thread_key} with previous_response_id: {previous_response_id}")
        else:
            print(f"Starting new conversation for {thread_key}")

        # Create streaming response using the Responses API with file_search tool
        print("Starting streaming response...")

        stream = await client.responses.create(
            model=OPENAI_MODEL,
            reasoning={"effort": REASONING_EFFORT},  # GPT-5.1 adaptive reasoning
            instructions=SYSTEM_INSTRUCTIONS,
            input=user_message,
            previous_response_id=previous_response_id,
            store=True,  # Required for multi-turn conversations - keeps context
            max_output_tokens=800,  # Reduced for concise responses
            tools=[{
                "type": "file_search",
                "vector_store_ids": [VECTOR_STORE_ID],
                "max_num_results": 20,  # Get more results for better context
                "ranking_options": {
                    "ranker": "auto",
                    "score_threshold": 0.0  # Include all results, let model decide
                }
            }],
            stream=True,  # Enable streaming
        )

        # Accumulate the response text from stream events
        accumulated_text = ""
        response_id = None
        file_search_triggered = False

        async for event in stream:
            # Capture response ID for conversation continuity
            if hasattr(event, 'response') and event.response and hasattr(event.response, 'id'):
                response_id = event.response.id

            # Handle different event types
            if event.type == "response.output_text.delta":
                # Accumulate text deltas
                if hasattr(event, 'delta'):
                    accumulated_text += event.delta
                    # Print progress indicator
                    print(".", end="", flush=True)

            elif event.type == "response.file_search_call.in_progress":
                print("\n  File search in progress...")
                file_search_triggered = True

            elif event.type == "response.file_search_call.done":
                print("  File search completed!")

            elif event.type == "response.completed":
                # Get final response ID from completed event
                if hasattr(event, 'response') and event.response:
                    response_id = event.response.id
                    print(f"\nStreaming completed. Response ID: {response_id}")

        print()  # New line after dots

        # Store the response ID for future turns (in Redis or memory)
        if response_id:
            await store_conversation_id(request, thread_key, response_id)
            print(f"Response ID stored for {thread_key}: {response_id}")

        if file_search_triggered:
            print("  File search was used for this response")

        if accumulated_text:
            return {"response": accumulated_text}

        return {"response": "Keine Antwort erhalten."}

    except Exception as e:
        print(f"Error getting streaming response: {e}")
        # If the error is related to previous_response_id, try without it
        if "previous_response_id" in str(e).lower():
            print(f"Retrying without previous_response_id for {thread_key}")
            try:
                # Clear the stored response and retry
                await delete_conversation_id(request, thread_key)
                stream = await client.responses.create(
                    model=OPENAI_MODEL,
                    reasoning={"effort": REASONING_EFFORT},  # GPT-5.1 adaptive reasoning
                    instructions=SYSTEM_INSTRUCTIONS,
                    input=user_message,
                    store=True,
                    max_output_tokens=800,
                    tools=[{
                        "type": "file_search",
                        "vector_store_ids": [VECTOR_STORE_ID],
                        "max_num_results": 20,
                        "ranking_options": {
                            "ranker": "auto",
                            "score_threshold": 0.0
                        }
                    }],
                    stream=True,
                )

                accumulated_text = ""
                response_id = None
                async for event in stream:
                    if hasattr(event, 'response') and event.response and hasattr(event.response, 'id'):
                        response_id = event.response.id
                    if event.type == "response.output_text.delta" and hasattr(event, 'delta'):
                        accumulated_text += event.delta
                    elif event.type == "response.completed" and hasattr(event, 'response'):
                        response_id = event.response.id

                if response_id:
                    await store_conversation_id(request, thread_key, response_id)

                if accumulated_text:
                    return {"response": accumulated_text}

            except Exception as retry_error:
                print(f"Retry also failed: {retry_error}")
                return {"response": f"Fehler: {str(retry_error)}"}
        return {"response": f"Fehler: {str(e)}"}


async def get_bot_token(request: Request = None) -> str:
    """Get OAuth token for Bot Framework with caching"""
    global token_cache

    if not BOT_APP_ID or not BOT_APP_PASSWORD:
        print("Warning: BOT_APP_ID or BOT_APP_PASSWORD not set")
        return ""

    # Return cached token if still valid (with 5-minute buffer)
    if token_cache and datetime.utcnow() < token_cache.expires_at - timedelta(minutes=5):
        return token_cache.token

    # For SingleTenant apps, we need to authenticate against the tenant
    # but use the Bot Framework scope
    if not AZURE_TENANT_ID:
        print("Warning: AZURE_TENANT_ID not set for single-tenant app")
        return ""
    tenant_id = AZURE_TENANT_ID

    # Use pooled HTTP client if available, otherwise create new one
    http_client = request.app.state.http_client if request and hasattr(request.app.state, 'http_client') else None

    try:
        if http_client:
            response = await http_client.post(
                f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": BOT_APP_ID,
                    "client_secret": BOT_APP_PASSWORD,
                    "scope": "https://api.botframework.com/.default"
                }
            )
        else:
            async with httpx.AsyncClient() as temp_client:
                response = await temp_client.post(
                    f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
                    data={
                        "grant_type": "client_credentials",
                        "client_id": BOT_APP_ID,
                        "client_secret": BOT_APP_PASSWORD,
                        "scope": "https://api.botframework.com/.default"
                    }
                )

        if response.status_code == 200:
            data = response.json()
            # Cache the token
            token_cache = TokenCache(
                token=data.get("access_token", ""),
                expires_at=datetime.utcnow() + timedelta(seconds=data.get("expires_in", 3600))
            )
            print(f"✅ Bot token cached (expires in {data.get('expires_in', 3600)}s)")
            return token_cache.token
        else:
            print(f"Failed to get token: {response.text}")
            return ""
    except Exception as e:
        print(f"Error getting bot token: {e}")
        return ""


async def send_reply(
    request: Request,
    service_url: str,
    conversation_id: str,
    activity_id: str,
    reply_to_id: str,
    recipient: dict,
    from_bot: dict,
    message: str,
    file_export: dict = None
):
    """Send reply back to Teams with optional file attachment.

    Args:
        file_export: Optional dict with file_name, mime_type, base64_data
    """
    try:
        token = await get_bot_token(request)
        http_client = request.app.state.http_client

        # If we have a file export, store it and add download link
        if file_export:
            file_name = file_export.get("file_name", "export.xlsx")
            mime_type = file_export.get("mime_type", "application/octet-stream")
            base64_data = file_export.get("base64_data", "")

            if base64_data:
                # Decode and store the file
                file_bytes = base64.b64decode(base64_data)
                file_id = store_temp_file(file_bytes, file_name, mime_type)

                # Generate download URL
                # Use the Cloud Run service URL
                download_url = f"https://teams-bot-942547788950.us-central1.run.app/api/download/{file_id}"

                # Add download link to message
                file_size_kb = len(file_bytes) / 1024
                message += f"\n\n📥 **[{file_name} herunterladen]({download_url})** ({file_size_kb:.1f} KB)\n*Link gültig für {FILE_EXPIRY_MINUTES} Minuten*"

        # Construct reply activity
        reply_activity = {
            "type": "message",
            "from": from_bot,
            "conversation": {"id": conversation_id},
            "recipient": recipient,
            "text": message,
            "replyToId": reply_to_id
        }

        # Send to Bot Framework
        url = f"{service_url}v3/conversations/{conversation_id}/activities/{activity_id}"

        headers = {
            "Content-Type": "application/json"
        }

        if token:
            headers["Authorization"] = f"Bearer {token}"

        response = await http_client.post(
            url,
            json=reply_activity,
            headers=headers
        )

        if response.status_code not in [200, 201]:
            print(f"Failed to send reply: {response.status_code} - {response.text}")
        else:
            print(f"Reply sent successfully" + (" with download link" if file_export else ""))

    except Exception as e:
        print(f"Error sending reply: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
