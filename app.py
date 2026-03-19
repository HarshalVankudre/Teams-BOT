"""
Microsoft Teams Bot Backend for the RUEKO equipment assistant

Features:
- Token caching for Bot Framework authentication
- HTTP connection pooling for better performance
- Redis for persistent conversation storage (per-user isolation)
- Graceful fallback to in-memory storage if Redis unavailable
- Gemini advisory routing + LangGraph retrieval + Pinecone fallback
- Rate limiting for API endpoints
"""
import os
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from fastapi import FastAPI, Request, Response
import httpx
import redis.asyncio as redis
from dotenv import load_dotenv
from commands import handle_command
import time

# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False
    print("[WARN] slowapi not installed. Rate limiting disabled.")

# RAG imports
from rag.search import RAGSearch
from rag.config import config
from rag.feedback import feedback_service
from rag.admin_logger import admin_logger

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
BOT_APP_ID = os.getenv("BOT_APP_ID", "")
BOT_APP_PASSWORD = os.getenv("BOT_APP_PASSWORD", "")
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID", "")  # Required for single-tenant apps

# Model configuration
MODEL_NAME = config.langgraph_model
ADVISORY_MODEL = config.advisory_model
FALLBACK_MODEL = config.fallback_model
# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CONVERSATION_TTL_HOURS = int(os.getenv("CONVERSATION_TTL_HOURS", "24"))

# Agent configuration
AGENT_VERBOSE = os.getenv("AGENT_VERBOSE", "false").lower() == "true"
SYSTEM_INSTRUCTIONS = config.system_instructions

logger.info(
    "Teams bot config loaded: advisory=%s retrieval=%s fallback=%s redis=%s ttl_hours=%s verbose=%s",
    ADVISORY_MODEL,
    MODEL_NAME,
    FALLBACK_MODEL,
    bool(REDIS_URL),
    CONVERSATION_TTL_HOURS,
    AGENT_VERBOSE,
)

# Initialize RAG search - Redis client added after startup
rag_search = None  # Initialized in lifespan after Redis is available
logger.info("RAG search will be initialized during startup")

# Token caching for Bot Framework authentication
@dataclass
class TokenCache:
    token: str
    expires_at: datetime

token_cache: TokenCache | None = None
token_cache_lock = asyncio.Lock()  # Thread-safe token refresh

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
    redis_client = None
    if app.state.redis_available and app.state.redis_pool:
        redis_client = redis.Redis(connection_pool=app.state.redis_pool)
    rag_search = RAGSearch(redis_client=redis_client)
    print("[OK] RAG Search initialized")

    yield  # Application runs here

    # Shutdown: Cleanup resources
    print("Shutting down...")
    await app.state.http_client.aclose()
    if app.state.redis_pool:
        await app.state.redis_pool.disconnect()
    print("[OK] Resources cleaned up")


app = FastAPI(title="Teams Bot - RAG API", lifespan=lifespan)

# Rate limiting setup
if RATE_LIMITING_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    print("[OK] Rate limiting enabled (30 requests/minute per IP)")


# Redis helper functions
async def get_redis(request: Request) -> redis.Redis | None:
    """Get Redis client from app state"""
    if hasattr(request.app.state, 'redis_pool') and request.app.state.redis_pool:
        return redis.Redis(connection_pool=request.app.state.redis_pool)
    return None


@app.get("/")
async def root():
    return {"status": "ok", "message": "Teams Bot is running"}





@app.get("/health")
async def health(request: Request):
    """Health check with Redis status"""
    redis_status = "unavailable"
    active_conversations = 0

    r = await get_redis(request)
    if r:
        try:
            await r.ping()
            redis_status = "connected"
            # Count Redis history keys for active conversations
            cursor = 0
            count = 0
            while True:
                cursor, keys = await r.scan(cursor, match="chat_history:*", count=100)
                count += len(keys)
                if cursor == 0:
                    break
            active_conversations = count
        except Exception:
            redis_status = "error"

    return {
        "status": "healthy",
        "retrieval_model": MODEL_NAME,
        "advisory_model": ADVISORY_MODEL,
        "fallback_model": FALLBACK_MODEL,
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
            if rag_search:
                await rag_search.reset_thread_state(thread_key)
            return {"status": "ok", "message": f"Conversation {thread_key} reset"}
        else:
            if rag_search:
                await rag_search.reset_all_thread_state()
            return {"status": "ok", "message": "All conversations reset"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/messages")
async def messages(request: Request):
    """Handle incoming messages from Microsoft Teams/Bot Framework"""
    # Rate limiting check (if available) - 30 requests per minute per IP
    if RATE_LIMITING_AVAILABLE:
        try:
            limiter = request.app.state.limiter
            # Note: slowapi's limit decorator doesn't work well with async,
            # so we use the underlying check directly
            limit_value = "30/minute"
        except Exception as rl_err:
            print(f"[App] Rate limit check error: {rl_err}")

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

            # Create unique thread key per user (combines user ID with conversation ID)
            # This ensures each user has their own thread even in group chats
            thread_key = f"{user_id}:{conversation_id}"

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

                command_name = user_message.strip().split(maxsplit=1)[0].lower()
                if command_name in {"/zuruecksetzen", "/zurücksetzen", "/reset"} and rag_search:
                    await rag_search.reset_thread_state(thread_key)

                # Route to command handler
                await handle_command(body, user_message, send_command_reply)

            else:
                # Regular conversation - AI response
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
                    # Get response from Agent System
                    assistant_response, result_metadata = await get_assistant_response_streaming(
                        request, thread_key, user_message,
                        user_id=user_id, user_name=user_name
                    )
                finally:
                    # Always stop typing indicator when done
                    typing_manager.stop()

                # Calculate response time in milliseconds
                response_time_ms = int((time.time() - start_time) * 1000)

                # Store conversation in feedback database
                try:
                    feedback_service.save_conversation(
                        user_id=user_id,
                        user_message=user_message,
                        ai_response=assistant_response,
                        user_name=user_name,
                        user_email=user_email,
                        conversation_thread_id=thread_key,
                        response_time_ms=response_time_ms,
                        query_type=(result_metadata or {}).get("query_type"),
                        data_source=(result_metadata or {}).get("agent", "assistant")
                    )
                except Exception as fb_error:
                    print(f"[Feedback] Error storing conversation: {fb_error}")

                # Store in admin dashboard database
                try:
                    admin_logger.log_conversation(
                        thread_id=thread_key,
                        ms_user_id=user_id,
                        user_message=user_message,
                        assistant_response=assistant_response,
                        user_name=user_name,
                        user_email=user_email,
                        response_time_ms=response_time_ms,
                        logs=result_metadata.get("logs") if result_metadata else None
                    )
                except Exception as admin_error:
                    print(f"[AdminLogger] Error: {admin_error}")

                # Send reply back to Teams
                await send_reply(
                    request=request,
                    service_url=service_url,
                    conversation_id=conversation_id,
                    activity_id=body.get("id"),
                    reply_to_id=body.get("id"),
                    recipient=body.get("from"),
                    from_bot=body.get("recipient"),
                    message=assistant_response
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
                        message=config.welcome_message
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
) -> tuple[str, dict]:
    """Get response from the configured RAG assistant."""
    logger.info("Processing assistant response for thread=%s", thread_key)

    try:
        # Search and generate response using agent system
        result = await rag_search.search_and_generate(
            query=user_message,
            system_instructions=SYSTEM_INSTRUCTIONS,
            user_id=user_id,
            user_name=user_name,
            thread_key=thread_key
        )

        response = result["response"]

        # Log agent system info
        agents_used = result.get("agents_used", [])
        query_type = result.get("query_type", "unknown")
        execution_time = result.get("execution_time_ms", 0)

        if agents_used:
            print(f"Agents used: {agents_used}")
        print(f"Query type: {query_type}, Execution time: {execution_time}ms")

        # Add sources if available (only for non-agent responses or minimal sources)
        sources = result.get("sources", [])
        if sources and query_type == "fallback":
            response += "\n\n---\n**Quellen:**"
            for source in sources[:2]:
                score_pct = source.get("score", 0)
                if isinstance(score_pct, float) and score_pct <= 1:
                    score_pct = score_pct * 100
                response += f"\n- {source.get('title', 'Unbekannt')} ({source.get('source_file', '')}) [{score_pct:.0f}%]"

        print(f"Response generated using {result.get('chunks_used', 0)} sources")
        return response, result

    except Exception as e:
        print(f"Agent System error: {e}")
        import traceback
        traceback.print_exc()
        return f"Fehler bei der Verarbeitung: {str(e)}", {}


async def get_assistant_response_streaming(
    request: Request,
    thread_key: str,
    user_message: str,
    user_id: str = None,
    user_name: str = None
) -> tuple[str, dict]:
    """Get response from the configured assistant runtime."""
    return await get_custom_rag_response(
        request, thread_key, user_message,
        user_id=user_id, user_name=user_name
    )


async def get_bot_token(request: Request = None) -> str:
    """Get OAuth token for Bot Framework with caching and thread safety."""
    global token_cache

    if not BOT_APP_ID or not BOT_APP_PASSWORD:
        print("Warning: BOT_APP_ID or BOT_APP_PASSWORD not set")
        return ""

    # Fast path: check cache without lock
    if token_cache and datetime.utcnow() < token_cache.expires_at - timedelta(minutes=5):
        return token_cache.token

    # Acquire lock for token refresh (prevents multiple concurrent refresh attempts)
    async with token_cache_lock:
        # Double-check after acquiring lock (another coroutine may have refreshed)
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
                print(f"[App] Bot token refreshed (expires in {data.get('expires_in', 3600)}s)")
                return token_cache.token
            else:
                print(f"[App] Failed to get token: {response.text}")
                return ""
        except Exception as e:
            print(f"[App] Error getting bot token: {e}")
            return ""


async def send_reply(
    request: Request,
    service_url: str,
    conversation_id: str,
    activity_id: str,
    reply_to_id: str,
    recipient: dict,
    from_bot: dict,
    message: str
):
    """Send reply back to Teams"""
    try:
        token = await get_bot_token(request)

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

        # Use pooled HTTP client
        http_client = request.app.state.http_client
        response = await http_client.post(
            url,
            json=reply_activity,
            headers=headers
        )

        if response.status_code not in [200, 201]:
            print(f"Failed to send reply: {response.status_code} - {response.text}")
        else:
            print(f"Reply sent successfully")

    except Exception as e:
        print(f"Error sending reply: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
