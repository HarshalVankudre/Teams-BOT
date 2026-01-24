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
    """
    Configuration for the RAG pipeline - all from .env

    Enhanced Agent Features:
        AGENT_ENABLE_PLANNING: Enable query planning before execution (default: true)
        AGENT_ENABLE_SQL_VERIFICATION: Enable SQL verification/autocorrection (default: true)
        AGENT_ENABLE_REASONING_TOOLS: Enable calculate/compare/aggregate tools (default: true)
        AGENT_PLANNING_MODEL: Model for planning (empty = use main model)
        AGENT_VERIFICATION_MODEL: Model for SQL verification (empty = use main model)
    """

    # LLM Provider Selection
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")  # openai, cerebras

    # OpenAI Settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")  # Default to gpt-4o
    openai_reasoning: str = os.getenv("REASONING_EFFORT", "none")  # none for gpt-4o, only o1/o3 use this
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))  # Low for factual responses
    openai_max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
    openai_top_p: float = float(os.getenv("OPENAI_TOP_P", "0.95"))
    openai_frequency_penalty: float = float(os.getenv("OPENAI_FREQUENCY_PENALTY", "0.0"))
    openai_presence_penalty: float = float(os.getenv("OPENAI_PRESENCE_PENALTY", "0.0"))

    # Cerebras Settings
    cerebras_api_key: str = os.getenv("CEREBRAS_API_KEY", "")
    cerebras_model: str = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")  # e.g., gpt-oss-120b, llama-3.3-70b
    cerebras_reasoning: str = os.getenv("CEREBRAS_REASONING", "low")  # low, medium, high (gpt-oss-120b only)

    # Groq Settings (for LangGraph agent)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")

    @property
    def response_model(self) -> str:
        provider = (self.llm_provider or "openai").lower()
        if provider == "cerebras":
            return self.cerebras_model
        return self.openai_model

    @property
    def response_reasoning(self) -> str:
        provider = (self.llm_provider or "openai").lower()
        if provider == "cerebras":
            return self.cerebras_reasoning
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

    # Redis connection for LangGraph checkpointing
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # Agent System Settings
    use_agent_system: bool = os.getenv("USE_AGENT_SYSTEM", "true").lower() == "true"
    use_langgraph_agent: bool = os.getenv("USE_LANGGRAPH_AGENT", "true").lower() == "true"
    agent_parallel_execution: bool = os.getenv("AGENT_PARALLEL_EXECUTION", "true").lower() == "true"
    agent_verbose: bool = os.getenv("AGENT_VERBOSE", "true").lower() == "true"  # Enable for debugging
    agent_prefetch_documents: bool = os.getenv("AGENT_PREFETCH_DOCUMENTS", "false").lower() == "true"  # Disable unless needed
    agent_max_completion_tokens: int = int(os.getenv("AGENT_MAX_COMPLETION_TOKENS", "4096"))  # Match OpenAI max tokens
    agent_max_tool_rounds: int = int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "6"))

    # Schema Linker Settings
    schema_linker_top_k: int = int(os.getenv("SCHEMA_LINKER_TOP_K", "40"))  # More columns visible to model

    # Enhanced Agent Features (GPT-4o optimized defaults)
    agent_enable_planning: bool = os.getenv("AGENT_ENABLE_PLANNING", "false").lower() == "true"  # GPT-4o doesn't need extra planning
    agent_enable_sql_verification: bool = os.getenv("AGENT_ENABLE_SQL_VERIFICATION", "true").lower() == "true"  # Keep SQL verification on
    agent_enable_reasoning_tools: bool = os.getenv("AGENT_ENABLE_REASONING_TOOLS", "false").lower() == "true"
    agent_planning_model: str = os.getenv("AGENT_PLANNING_MODEL", "")  # Empty = use main model
    agent_verification_model: str = os.getenv("AGENT_VERIFICATION_MODEL", "")  # Empty = use main model

    # Use clean/simplified agent architecture (recommended for stability)
    use_clean_agent: bool = os.getenv("USE_CLEAN_AGENT", "true").lower() == "true"

    # Prompt optimization
    agent_compact_prompt: bool = os.getenv("AGENT_COMPACT_PROMPT", "true").lower() == "true"  # Reduce prompt size

    # Conversation Settings (GPT-4o has large context, can keep more history)
    conversation_ttl_hours: int = int(os.getenv("CONVERSATION_TTL_HOURS", "24"))
    conversation_max_messages: int = int(os.getenv("CONVERSATION_MAX_MESSAGES", "6"))

    # Logging Settings
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_json: bool = os.getenv("LOG_JSON", "false").lower() == "true"

    def validate(self):
        """Validate required configuration"""
        errors = []

        # Validate based on selected provider
        if self.llm_provider == "cerebras":
            if not self.cerebras_api_key:
                errors.append("CEREBRAS_API_KEY is required when LLM_PROVIDER=cerebras")
            if not self.cerebras_model:
                errors.append("CEREBRAS_MODEL is required when LLM_PROVIDER=cerebras")
        else:
            # OpenAI is default
            if not self.openai_api_key:
                errors.append("OPENAI_API_KEY is required")
            if not self.openai_model:
                errors.append("OPENAI_MODEL is required")

        # Pinecone is always required
        if not self.pinecone_api_key:
            errors.append("PINECONE_API_KEY is required")
        if not self.pinecone_host:
            errors.append("PINECONE_HOST is required")

        # Web search validation
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
