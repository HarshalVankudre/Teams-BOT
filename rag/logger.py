"""
Centralized logging configuration for Teams-BOT.

Provides structured logging with:
- Log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Request correlation IDs
- Consistent formatting
- Optional JSON output for production
"""
import logging
import sys
import os
import uuid
from contextvars import ContextVar
from typing import Optional
from functools import wraps
from datetime import datetime

# Context variable for request correlation ID
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    """Get current request correlation ID."""
    return request_id_var.get() or "no-request-id"


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set request correlation ID for current context."""
    rid = request_id or str(uuid.uuid4())[:8]
    request_id_var.set(rid)
    return rid


def clear_request_id():
    """Clear the request ID for the current context."""
    request_id_var.set("")


class CorrelationFilter(logging.Filter):
    """Add correlation ID to log records."""

    def filter(self, record):
        record.request_id = get_request_id()
        return True


class TeamsBotFormatter(logging.Formatter):
    """Custom formatter with correlation ID and component."""

    # Color codes for terminal output
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
        "RESET": "\033[0m",       # Reset
    }

    def __init__(self, include_timestamp: bool = True, use_colors: bool = True):
        self.include_timestamp = include_timestamp
        self.use_colors = use_colors and sys.stdout.isatty()
        super().__init__()

    def format(self, record):
        # Get component from logger name (e.g., "teamsbot.postgres" -> "PostgreSQL")
        component_map = {
            "teamsbot.app": "App",
            "teamsbot.postgres": "PostgreSQL",
            "teamsbot.search": "RAG",
            "teamsbot.agent": "Agent",
            "teamsbot.pinecone": "Pinecone",
            "teamsbot.redis": "Redis",
            "teamsbot.admin": "Admin",
            "teamsbot.feedback": "Feedback",
            "teamsbot.embeddings": "Embeddings",
            "teamsbot.guard": "Guard",
            "teamsbot.cache": "Cache",
        }
        component = component_map.get(record.name, record.name.split(".")[-1].title())

        # Level indicator
        level_indicators = {
            "DEBUG": "DBG",
            "INFO": "INF",
            "WARNING": "WRN",
            "ERROR": "ERR",
            "CRITICAL": "CRT",
        }
        indicator = level_indicators.get(record.levelname, "???")

        # Build message parts
        parts = []

        if self.include_timestamp:
            parts.append(datetime.now().strftime("%H:%M:%S.%f")[:-3])

        # Add colored level indicator
        if self.use_colors:
            color = self.COLORS.get(record.levelname, "")
            reset = self.COLORS["RESET"]
            parts.append(f"{color}[{indicator}]{reset}")
        else:
            parts.append(f"[{indicator}]")

        parts.append(f"[{component}]")

        if record.request_id:
            parts.append(f"[{record.request_id}]")

        parts.append(record.getMessage())

        # Add exception info if present
        if record.exc_info:
            parts.append("\n" + self.formatException(record.exc_info))

        return " ".join(parts)


class JsonFormatter(logging.Formatter):
    """JSON formatter for production log aggregation."""

    def format(self, record):
        import json

        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "request_id") and record.request_id:
            log_entry["request_id"] = record.request_id

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add any extra fields
        for key in ["user_id", "thread_key", "sql", "duration_ms", "tool"]:
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(
    level: str = None,
    json_output: bool = None,
    include_timestamp: bool = True,
) -> logging.Logger:
    """
    Set up application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Default from env LOG_LEVEL or INFO.
        json_output: Use JSON format. Default from env LOG_JSON or False.
        include_timestamp: Include timestamp in console output

    Returns:
        Root logger for teamsbot
    """
    level = level or os.getenv("LOG_LEVEL", "INFO").upper()

    if json_output is None:
        json_output = os.getenv("LOG_JSON", "false").lower() == "true"

    numeric_level = getattr(logging, level, logging.INFO)

    # Create root logger for application
    logger = logging.getLogger("teamsbot")
    logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(TeamsBotFormatter(
            include_timestamp=include_timestamp,
            use_colors=True
        ))

    # Add correlation filter
    handler.addFilter(CorrelationFilter())

    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific component.

    Args:
        name: Component name (e.g., "postgres", "agent", "search")

    Returns:
        Logger instance
    """
    return logging.getLogger(f"teamsbot.{name}")


# Pre-configured loggers for common components
app_logger = get_logger("app")
postgres_logger = get_logger("postgres")
search_logger = get_logger("search")
agent_logger = get_logger("agent")
pinecone_logger = get_logger("pinecone")
redis_logger = get_logger("redis")
admin_logger = get_logger("admin")
feedback_logger = get_logger("feedback")
embeddings_logger = get_logger("embeddings")
guard_logger = get_logger("guard")
cache_logger = get_logger("cache")


# Initialize logging on import
_root_logger = setup_logging()


# Convenience functions for structured logging
def log_sql_execution(sql: str, duration_ms: int, row_count: int, error: str = None):
    """Log SQL execution with structured data."""
    extra = {
        "sql": sql[:200] + "..." if len(sql) > 200 else sql,
        "duration_ms": duration_ms,
    }
    if error:
        postgres_logger.error(f"SQL failed ({duration_ms}ms): {error}", extra=extra)
    else:
        postgres_logger.info(f"SQL executed: {row_count} rows in {duration_ms}ms", extra=extra)


def log_tool_call(tool_name: str, duration_ms: int, success: bool, result_summary: str = ""):
    """Log tool execution."""
    extra = {"tool": tool_name, "duration_ms": duration_ms}
    if success:
        agent_logger.info(f"Tool {tool_name}: {result_summary} ({duration_ms}ms)", extra=extra)
    else:
        agent_logger.warning(f"Tool {tool_name} failed ({duration_ms}ms): {result_summary}", extra=extra)


def log_request_start(thread_key: str, user_message: str):
    """Log incoming request."""
    extra = {"thread_key": thread_key}
    preview = user_message[:100] + "..." if len(user_message) > 100 else user_message
    app_logger.info(f"Request: {preview}", extra=extra)


def log_request_end(thread_key: str, duration_ms: int, success: bool):
    """Log request completion."""
    extra = {"thread_key": thread_key, "duration_ms": duration_ms}
    if success:
        app_logger.info(f"Response sent ({duration_ms}ms)", extra=extra)
    else:
        app_logger.warning(f"Request failed ({duration_ms}ms)", extra=extra)
