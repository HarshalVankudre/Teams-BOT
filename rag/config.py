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

    # LLM Provider: "openai" or "gemini"
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")

    # OpenAI Settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "")  # e.g., gpt-5, gpt-4o
    openai_reasoning: str = os.getenv("REASONING_EFFORT", "")  # none, low, medium, high

    # Gemini Settings
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    gemini_reasoning: str = os.getenv("GEMINI_REASONING", "low")  # none, low, medium, high

    # Resolved model (based on provider)
    @property
    def response_model(self) -> str:
        if self.llm_provider == "gemini":
            return self.gemini_model
        return self.openai_model

    @property
    def response_reasoning(self) -> str:
        return self.openai_reasoning

    # Chunking Model (from .env with fallback)
    chunking_model: str = os.getenv("CHUNKING_MODEL", "gpt-4o-mini")
    chunking_reasoning: str = os.getenv("CHUNKING_REASONING", "none")
    chunking_max_output_tokens: int = int(os.getenv("CHUNKING_MAX_OUTPUT_TOKENS", "6000"))

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
    web_search_max_results: int = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))

    # Search Settings
    search_top_k: int = int(os.getenv("SEARCH_TOP_K", "10"))  # Results per namespace
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "5"))
    fallback_max_output_tokens: int = int(os.getenv("FALLBACK_MAX_OUTPUT_TOKENS", "1200"))

    # Chunk Settings
    max_chunk_tokens: int = int(os.getenv("MAX_CHUNK_TOKENS", "500"))
    min_chunk_tokens: int = int(os.getenv("MIN_CHUNK_TOKENS", "50"))

    # Agent System Settings
    use_agent_system: bool = os.getenv("USE_AGENT_SYSTEM", "true").lower() == "true"
    agent_parallel_execution: bool = os.getenv("AGENT_PARALLEL_EXECUTION", "true").lower() == "true"
    agent_verbose: bool = os.getenv("AGENT_VERBOSE", "false").lower() == "true"
    agent_prefetch_documents: bool = os.getenv("AGENT_PREFETCH_DOCUMENTS", "true").lower() == "true"
    agent_max_completion_tokens: int = int(os.getenv("AGENT_MAX_COMPLETION_TOKENS", "1200"))
    agent_max_tool_rounds: int = int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "6"))

    # Enhanced Agent Features (all configurable, all off by default for safety)
    agent_enable_planning: bool = os.getenv("AGENT_ENABLE_PLANNING", "true").lower() == "true"
    agent_enable_sql_verification: bool = os.getenv("AGENT_ENABLE_SQL_VERIFICATION", "true").lower() == "true"
    agent_enable_reasoning_tools: bool = os.getenv("AGENT_ENABLE_REASONING_TOOLS", "true").lower() == "true"
    agent_planning_model: str = os.getenv("AGENT_PLANNING_MODEL", "")  # Empty = use main model
    agent_verification_model: str = os.getenv("AGENT_VERIFICATION_MODEL", "")  # Empty = use main model

    # Conversation Settings
    conversation_ttl_hours: int = int(os.getenv("CONVERSATION_TTL_HOURS", "72"))
    conversation_max_messages: int = int(os.getenv("CONVERSATION_MAX_MESSAGES", "40"))

    def validate(self):
        """Validate required configuration"""
        errors = []

        # Validate provider-specific settings
        if self.llm_provider == "openai":
            if not self.openai_api_key:
                errors.append("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
            if not self.openai_model:
                errors.append("OPENAI_MODEL is required when LLM_PROVIDER=openai")
        elif self.llm_provider == "gemini":
            if not self.gemini_api_key:
                errors.append("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
            if not self.gemini_model:
                errors.append("GEMINI_MODEL is required when LLM_PROVIDER=gemini")
            # OpenAI still needed for embeddings
            if not self.openai_api_key:
                errors.append("OPENAI_API_KEY is required for embeddings")
        else:
            errors.append(f"Invalid LLM_PROVIDER: {self.llm_provider}. Use 'openai' or 'gemini'")

        # Pinecone is always required
        if not self.pinecone_api_key:
            errors.append("PINECONE_API_KEY is required")
        if not self.pinecone_host:
            errors.append("PINECONE_HOST is required")

        # Web search validation
        if self.enable_web_search and not self.tavily_api_key:
            errors.append("TAVILY_API_KEY is required when ENABLE_WEB_SEARCH=true")

        return errors

    def is_gemini(self) -> bool:
        """Check if using Gemini provider"""
        return self.llm_provider.lower() == "gemini"


# Global config instance
config = RAGConfig()

# Validate on import and warn about missing config
_errors = config.validate()
if _errors:
    print(f"[CONFIG WARNING] Missing required configuration:")
    for err in _errors:
        print(f"  - {err}")
