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

    # LLM Provider Settings
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "ollama"

    # OpenAI Settings (REQUIRED for embeddings, optional for chat if using Ollama)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Ollama Settings (when LLM_PROVIDER=ollama)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:14b")

    # Model Settings (used when LLM_PROVIDER=openai)
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

        # OpenAI API key is always required (for embeddings)
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required (used for embeddings)")

        # Validate provider-specific settings
        if self.llm_provider == "openai":
            if not self.response_model:
                errors.append("OPENAI_MODEL is required when LLM_PROVIDER=openai")
            if not self.response_reasoning:
                errors.append("REASONING_EFFORT is required when LLM_PROVIDER=openai")
        elif self.llm_provider == "ollama":
            if not self.ollama_model:
                errors.append("OLLAMA_MODEL is required when LLM_PROVIDER=ollama")
        else:
            errors.append(f"Invalid LLM_PROVIDER: {self.llm_provider}. Use 'openai' or 'ollama'")

        # Pinecone is always required
        if not self.pinecone_api_key:
            errors.append("PINECONE_API_KEY is required")
        if not self.pinecone_host:
            errors.append("PINECONE_HOST is required")

        # Web search validation
        if self.enable_web_search and not self.tavily_api_key:
            errors.append("TAVILY_API_KEY is required when ENABLE_WEB_SEARCH=true")

        return errors

    def get_chat_model(self) -> str:
        """Get the chat model based on provider"""
        if self.llm_provider == "ollama":
            return self.ollama_model
        return self.response_model

    def is_ollama(self) -> bool:
        """Check if using Ollama provider"""
        return self.llm_provider.lower() == "ollama"


# Global config instance
config = RAGConfig()

# Validate on import and warn about missing config
_errors = config.validate()
if _errors:
    print(f"[CONFIG WARNING] Missing required configuration:")
    for err in _errors:
        print(f"  - {err}")
