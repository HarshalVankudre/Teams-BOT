"""
Custom RAG Pipeline for Teams Bot
- Hybrid routing: PostgreSQL (structured) + Pinecone (semantic)
- OpenAI text-embedding-3-large embeddings
- AI-powered SQL generation for equipment queries
- Multi-source search with Tavily web integration
"""

from .config import RAGConfig
from .chunker import SemanticChunker
from .embeddings import EmbeddingService
from .vector_store import PineconeStore
from .processor import DocumentProcessor
from .search import RAGSearch

__all__ = [
    "RAGConfig",
    "SemanticChunker",
    "EmbeddingService",
    "PineconeStore",
    "DocumentProcessor",
    "RAGSearch"
]
