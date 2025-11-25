"""
RAG Pipeline Configuration
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline"""

    # OpenAI Settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Chunking Model (GPT-5-nano - fast & cheap for structuring data)
    chunking_model: str = "gpt-5-nano"
    chunking_reasoning: str = "none"

    # Response Model (from env var, default to gpt-4o-mini for speed)
    response_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    response_reasoning: str = os.getenv("REASONING_EFFORT", "none")

    # Embedding Model
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072  # Full dimensions for text-embedding-3-large

    # Pinecone Settings
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_host: str = os.getenv("PINECONE_HOST", "https://teams-328h5t9.svc.aped-4627-b74a.pinecone.io")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "rueko-documents")

    # Search Settings
    search_top_k: int = 3  # Minimal for fastest response
    rerank_top_n: int = 2

    # Chunk Settings
    max_chunk_tokens: int = 500
    min_chunk_tokens: int = 50


# Global config instance
config = RAGConfig()
