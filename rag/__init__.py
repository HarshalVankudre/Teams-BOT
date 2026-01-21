"""
Custom RAG Pipeline for Teams Bot
- Hybrid routing: PostgreSQL (structured) + Pinecone (semantic)
- OpenAI text-embedding-3-large embeddings
- AI-powered SQL generation for equipment queries
- Multi-source search with Tavily web integration
- Semantic schema linking for text-to-SQL (Phase 1-5)
"""

from .config import RAGConfig
from .embeddings import EmbeddingService
from .vector_store import PineconeStore
from .search import RAGSearch

# Semantic schema linking components (Phase 1-5)
from .schema_linker import schema_linker, SchemaLinker, ReducedSchema, ColumnID, ColumnMetadata
from .sql_validator import sql_validator, SQLValidator, ValidationResult, Predicate
from .value_index import value_index, ValueIndex, ValueMatch
from .alias_learner import alias_learner, AliasLearner, ColumnAlias, ValueAlias

__all__ = [
    "RAGConfig",
    "EmbeddingService",
    "PineconeStore",
    "RAGSearch",
    # Schema linking
    "schema_linker",
    "SchemaLinker",
    "ReducedSchema",
    "ColumnID",
    "ColumnMetadata",
    # SQL validation
    "sql_validator",
    "SQLValidator",
    "ValidationResult",
    "Predicate",
    # Value index
    "value_index",
    "ValueIndex",
    "ValueMatch",
    # Alias learning
    "alias_learner",
    "AliasLearner",
    "ColumnAlias",
    "ValueAlias",
]
