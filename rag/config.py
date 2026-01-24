"""
RAG Pipeline Configuration (Simplified)
All model settings are loaded from environment variables.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline - all from .env"""

    # LLM Provider Selection
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")

    # OpenAI Settings (for embeddings and fallback)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))

    # Groq Settings (for LangGraph agent)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")

    @property
    def response_model(self) -> str:
        """Get the model to use for responses"""
        if self.groq_api_key:
            return self.groq_model
        return self.openai_model

    @property
    def response_reasoning(self) -> str:
        """Get reasoning effort (for compatibility with OpenAI o-series models)"""
        return os.getenv("REASONING_EFFORT", "none")

    # Legacy provider support (for app.py compatibility)
    @property
    def cerebras_model(self) -> str:
        """Legacy: Cerebras model (not used with Groq)"""
        return os.getenv("CEREBRAS_MODEL", "")

    @property
    def openai_reasoning(self) -> str:
        """OpenAI reasoning effort setting"""
        return os.getenv("REASONING_EFFORT", "none")

    @property
    def use_agent_system(self) -> bool:
        """Whether to use the agent system"""
        return os.getenv("USE_AGENT_SYSTEM", "true").lower() == "true"

    @property
    def use_clean_agent(self) -> bool:
        """Whether to use the clean agent (legacy)"""
        return os.getenv("USE_CLEAN_AGENT", "false").lower() == "true"

    @property
    def fallback_max_output_tokens(self) -> int:
        """Max output tokens for fallback responses"""
        return int(os.getenv("FALLBACK_MAX_OUTPUT_TOKENS", "4096"))

    # Embedding Model
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    embedding_dimensions: int = 3072

    # Pinecone Settings
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_host: str = os.getenv("PINECONE_HOST", "")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "rueko-documents")
    pinecone_machinery_namespace: str = os.getenv("PINECONE_MACHINERY_NAMESPACE", "machinery-data")

    # Tavily Web Search (optional)
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    enable_web_search: bool = os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true"
    web_search_max_results: int = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3"))

    # Search Settings
    search_top_k: int = int(os.getenv("SEARCH_TOP_K", "5"))

    # Redis connection for conversation history
    redis_url: str = os.getenv("REDIS_URL", "")

    # Agent Settings
    use_langgraph_agent: bool = os.getenv("USE_LANGGRAPH_AGENT", "true").lower() == "true"
    agent_verbose: bool = os.getenv("AGENT_VERBOSE", "true").lower() == "true"
    agent_max_tool_rounds: int = int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "6"))

    # Conversation Settings
    conversation_ttl_hours: int = int(os.getenv("CONVERSATION_TTL_HOURS", "24"))
    conversation_max_messages: int = int(os.getenv("CONVERSATION_MAX_MESSAGES", "6"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def validate(self):
        """Validate required configuration"""
        errors = []

        # OpenAI is required for embeddings
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required for embeddings")

        # Either Groq or OpenAI needed for LLM
        if not self.groq_api_key and not self.openai_api_key:
            errors.append("GROQ_API_KEY or OPENAI_API_KEY is required")

        # Pinecone is required
        if not self.pinecone_api_key:
            errors.append("PINECONE_API_KEY is required")
        if not self.pinecone_host:
            errors.append("PINECONE_HOST is required")

        # Web search validation (optional)
        if self.enable_web_search and not self.tavily_api_key:
            errors.append("TAVILY_API_KEY is required when ENABLE_WEB_SEARCH=true")

        return errors


# Global config instance
config = RAGConfig()

# Validate on import and warn about missing config
_errors = config.validate()
if _errors:
    print(f"[CONFIG WARNING] Missing configuration:")
    for err in _errors:
        print(f"  - {err}")
