"""RAG Pipeline Configuration."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

from .prompts import (  # noqa: E402
    DEFAULT_TEAMS_SYSTEM_INSTRUCTIONS,
    DEFAULT_TEAMS_WELCOME_MESSAGE,
)


@dataclass
class RAGConfig:
    """Central configuration for the RAG pipeline."""

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    langgraph_model: str = os.getenv("LANGGRAPH_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    fallback_model: str = os.getenv("FALLBACK_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    enable_compound_agent: bool = os.getenv("ENABLE_COMPOUND_AGENT", "true").lower() == "true"
    enable_expert_planner: bool = os.getenv("ENABLE_EXPERT_PLANNER", "true").lower() == "true"

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))

    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_host: str = os.getenv("PINECONE_HOST", "")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "rueko-documents")
    pinecone_machinery_namespace: str = os.getenv("PINECONE_MACHINERY_NAMESPACE", "machinery-data")

    redis_url: str = os.getenv("REDIS_URL", "")
    conversation_ttl_hours: int = int(os.getenv("CONVERSATION_TTL_HOURS", "24"))
    conversation_max_messages: int = int(os.getenv("CONVERSATION_MAX_MESSAGES", "6"))
    advisory_history_max_messages: int = int(os.getenv("ADVISORY_HISTORY_MAX_MESSAGES", "12"))
    advisory_session_timeout_hours: int = int(os.getenv("ADVISORY_SESSION_TIMEOUT_HOURS", "12"))
    project_memory_max_items: int = int(os.getenv("PROJECT_MEMORY_MAX_ITEMS", "5"))

    use_langgraph_agent: bool = os.getenv("USE_LANGGRAPH_AGENT", "true").lower() == "true"
    agent_verbose: bool = os.getenv("AGENT_VERBOSE", "true").lower() == "true"
    agent_max_tool_rounds: int = int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "6"))

    search_top_k: int = int(os.getenv("SEARCH_TOP_K", "5"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    system_instructions: str = os.getenv("SYSTEM_INSTRUCTIONS", DEFAULT_TEAMS_SYSTEM_INSTRUCTIONS)
    welcome_message: str = os.getenv("WELCOME_MESSAGE", DEFAULT_TEAMS_WELCOME_MESSAGE)

    @property
    def advisory_model(self) -> str:
        return self.gemini_model

    @property
    def fallback_max_output_tokens(self) -> int:
        return int(os.getenv("FALLBACK_MAX_OUTPUT_TOKENS", "4096"))

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required for embeddings and the current Pinecone index")
        if not self.google_api_key:
            errors.append("GOOGLE_API_KEY is required for Gemini chat and LangGraph")
        if not self.pinecone_api_key:
            errors.append("PINECONE_API_KEY is required")
        if not self.pinecone_host:
            errors.append("PINECONE_HOST is required")
        return errors


config = RAGConfig()

_errors = config.validate()
if _errors:
    print("[CONFIG WARNING] Missing configuration:")
    for err in _errors:
        print(f"  - {err}")
