"""
Custom RAG Pipeline for Teams Bot
- Semantic chunking with GPT-5.1
- OpenAI text-embedding-3-large embeddings
- Pinecone vector database
- Rich metadata support
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
