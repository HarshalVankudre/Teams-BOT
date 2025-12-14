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

    # Query Rewriting (conversation-aware retrieval; defaults to chunking model)
    query_rewrite_model: str = os.getenv("QUERY_REWRITE_MODEL") or chunking_model
    query_rewrite_reasoning: str = os.getenv("QUERY_REWRITE_REASONING") or chunking_reasoning

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

    # Agent Settings
    # Unified single-agent mode (preferred for performance)
    use_single_agent: bool = os.getenv("USE_SINGLE_AGENT", "true").lower() == "true"

    # Conversation Settings
    conversation_ttl_hours: int = int(os.getenv("CONVERSATION_TTL_HOURS", "24"))

    # Unified Agent Settings (behavior tuning)
    unified_agent_max_tool_rounds: int = int(os.getenv("UNIFIED_AGENT_MAX_TOOL_ROUNDS", "2"))
    unified_agent_retry_on_empty: bool = os.getenv("UNIFIED_AGENT_RETRY_ON_EMPTY", "true").lower() == "true"
    unified_agent_default_sql_limit: int = int(os.getenv("UNIFIED_AGENT_DEFAULT_SQL_LIMIT", "200"))
    unified_agent_additional_instructions: str = os.getenv("UNIFIED_AGENT_ADDITIONAL_INSTRUCTIONS", "")
    unified_agent_force_internal_first: bool = os.getenv("UNIFIED_AGENT_FORCE_INTERNAL_FIRST", "true").lower() == "true"
    unified_agent_max_answer_words: int = int(os.getenv("UNIFIED_AGENT_MAX_ANSWER_WORDS", "120"))
    unified_agent_enable_query_rewrite: bool = os.getenv("UNIFIED_AGENT_ENABLE_QUERY_REWRITE", "true").lower() == "true"
    unified_agent_multi_query_retrieval: bool = os.getenv("UNIFIED_AGENT_MULTI_QUERY_RETRIEVAL", "true").lower() == "true"
    unified_agent_multi_query_max: int = int(os.getenv("UNIFIED_AGENT_MULTI_QUERY_MAX", "3"))
    unified_agent_rewrite_history_turns: int = int(os.getenv("UNIFIED_AGENT_REWRITE_HISTORY_TURNS", "8"))

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
