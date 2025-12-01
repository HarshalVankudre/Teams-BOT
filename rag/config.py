"""
RAG Pipeline Configuration
All model settings are loaded from environment variables - no hardcoded defaults.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline - all from .env"""

    # OpenAI Settings (REQUIRED - from .env)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Model Settings (REQUIRED - from .env, no hardcoded defaults)
    response_model: str = os.getenv("OPENAI_MODEL")  # e.g., gpt-5, gpt-4o
    response_reasoning: str = os.getenv("REASONING_EFFORT")  # none, low, medium, high

    # Chunking Model (from .env with fallback)
    chunking_model: str = os.getenv("CHUNKING_MODEL", "gpt-4o-mini")
    chunking_reasoning: str = os.getenv("CHUNKING_REASONING", "none")

    # Embedding Model
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    embedding_dimensions: int = 3072  # Full dimensions for text-embedding-3-large

    # Pinecone Settings
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_host: str = os.getenv("PINECONE_HOST", "")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "rueko-documents")
    pinecone_machinery_namespace: str = os.getenv("PINECONE_MACHINERY_NAMESPACE", "machinery-data")

    # Tavily Web Search Settings (supplementary - internal data is prioritized)
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    enable_web_search: bool = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"
    web_search_max_results: int = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3"))

    # Search Settings
    search_top_k: int = int(os.getenv("SEARCH_TOP_K", "5"))  # Results per namespace
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "3"))

    # Chunk Settings
    max_chunk_tokens: int = int(os.getenv("MAX_CHUNK_TOKENS", "500"))
    min_chunk_tokens: int = int(os.getenv("MIN_CHUNK_TOKENS", "50"))

    # Agent System Settings
    use_agent_system: bool = os.getenv("USE_AGENT_SYSTEM", "true").lower() == "true"
    agent_parallel_execution: bool = os.getenv("AGENT_PARALLEL_EXECUTION", "true").lower() == "true"
    agent_verbose: bool = os.getenv("AGENT_VERBOSE", "false").lower() == "true"

    # Conversation Settings
    conversation_ttl_hours: int = int(os.getenv("CONVERSATION_TTL_HOURS", "24"))

    def validate(self):
        """Validate required configuration"""
        errors = []
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required")
        if not self.response_model:
            errors.append("OPENAI_MODEL is required in .env")
        if not self.response_reasoning:
            errors.append("REASONING_EFFORT is required in .env")
        if not self.pinecone_api_key:
            errors.append("PINECONE_API_KEY is required")
        if not self.pinecone_host:
            errors.append("PINECONE_HOST is required")
        if self.enable_web_search and not self.tavily_api_key:
            errors.append("TAVILY_API_KEY is required when ENABLE_WEB_SEARCH=true")
        return errors


# Global config instance
config = RAGConfig()

# Validate on import and warn about missing config
_errors = config.validate()
if _errors:
    print(f"[CONFIG WARNING] Missing required configuration:")
    for err in _errors:
        print(f"  - {err}")
