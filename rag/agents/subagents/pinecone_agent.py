"""
Pinecone Search SubAgent

Performs semantic search in Pinecone vector store.
Used for recommendations, conceptual queries, and document retrieval.

Features:
    - Multi-namespace search (machinery data, company documents)
    - Similarity search for finding related equipment
    - Hybrid search combining semantic and metadata filters
    - Batch search for multiple queries
    - Reranking support for improved relevance

Author: RÜKO GmbH Baumaschinen
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Final, TypeAlias

import pinecone
from openai import AsyncOpenAI

from .interface import (
    AgentCapability,
    AgentContext,
    AgentMetadata,
    AgentResponse,
    AgentType,
    RetryConfig,
    RetryStrategy,
    SubAgentBase,
    register_subagent,
    tool,
)
from ...config import config
from ...embeddings import EmbeddingService

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Type aliases
EmbeddingVector: TypeAlias = list[float]
SearchResult: TypeAlias = dict[str, Any]
SearchResults: TypeAlias = list[SearchResult]
PineconeFilter: TypeAlias = dict[str, Any]


class SearchNamespace(str, Enum):
    """Available Pinecone namespaces."""
    
    MACHINERY = "machinery-data"
    DOCUMENTS = "rueko-documents"
    BOTH = "both"


class ResultType(str, Enum):
    """Types of search results."""
    
    MACHINERY = "machinery"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PineconeAgentConfig:
    """Configuration for the Pinecone agent."""
    
    # Search defaults
    default_top_k: int = 5
    max_top_k: int = 50
    min_score_threshold: float = 0.5
    
    # Batch settings
    max_batch_size: int = 10
    
    # Result formatting
    max_content_length: int = 500
    include_raw_metadata: bool = True


DEFAULT_CONFIG: Final[PineconeAgentConfig] = PineconeAgentConfig()


@dataclass(slots=True)
class SearchOptions:
    """Options for semantic search."""
    
    namespace: SearchNamespace = SearchNamespace.BOTH
    top_k: int = 5
    min_score: float = 0.0
    filters: PineconeFilter | None = None
    include_metadata: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "namespace": self.namespace.value,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "filters": self.filters,
            "include_metadata": self.include_metadata,
        }


@dataclass(slots=True)
class FormattedResult:
    """A formatted search result."""
    
    result_type: ResultType
    id: str
    score: float
    title: str
    content: str
    metadata: dict[str, Any]
    namespace: str
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.result_type.value,
            "id": self.id,
            "score": self.score,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
            "namespace": self.namespace,
        }


# Agent metadata
PINECONE_AGENT_METADATA: Final[AgentMetadata] = AgentMetadata(
    agent_id="pinecone",
    name="Pinecone Semantic Search Agent",
    description="Performs semantic/similarity search in the vector database",
    detailed_description="""Führt semantische Suche in der Pinecone Vektordatenbank durch.
Verwende diesen Agenten für:
- Empfehlungen und ähnliche Geräte finden
- Konzeptuelle Suchen ("Geräte für Straßenbau")
- Dokumentensuche in Unternehmensunterlagen
- Wenn die Anfrage nicht strukturiert/präzise ist
- Wenn du ähnliche Produkte oder Alternativen finden willst
- Wissensfragen zu Unternehmensrichtlinien und Prozessen
- Technische Dokumentation und Anleitungen""",
    capabilities=[AgentCapability.SEMANTIC_SEARCH],
    uses_reasoning=False,
    parameters={
        "pinecone_query": {
            "type": "string",
            "description": "Die Suchanfrage für semantische Suche"
        },
        "pinecone_namespace": {
            "type": "string",
            "enum": ["machinery-data", "rueko-documents", "both"],
            "description": "Welcher Namespace durchsucht werden soll"
        },
        "pinecone_top_k": {
            "type": "integer",
            "description": "Anzahl der gewünschten Ergebnisse (Standard: 5)"
        },
        "pinecone_filters": {
            "type": "object",
            "description": "Optionale Metadaten-Filter"
        }
    },
    example_queries=[
        "Empfehle mir einen Bagger für schwere Einsätze",
        "Welche Geräte eignen sich für den Straßenbau?",
        "Ähnliche Maschinen wie der CAT 320",
        "Dokumente zu Wartungsintervallen",
        "Technische Spezifikationen für Walzen",
        "Was sind die Richtlinien für Gerätewartung?",
        "Finde Alternativen zu Liebherr Baggern",
    ],
    priority=8,
)


@register_subagent()
class PineconeSearchAgent(SubAgentBase):
    """
    Performs semantic search across Pinecone namespaces.
    
    Searches both machinery data and company documents using
    vector similarity search with optional metadata filtering.
    
    Attributes:
        client: AsyncOpenAI client for embeddings
        embedding_service: Service for generating embeddings
        pc: Pinecone client
        index: Pinecone index instance
        agent_config: Configuration settings
    """

    METADATA = PINECONE_AGENT_METADATA

    __slots__ = (
        "client",
        "embedding_service",
        "pc",
        "index",
        "machinery_namespace",
        "documents_namespace",
        "agent_config",
    )

    def __init__(
        self,
        openai_client: AsyncOpenAI | None = None,
        embedding_service: EmbeddingService | None = None,
        verbose: bool = False,
        agent_config: PineconeAgentConfig | None = None,
    ) -> None:
        """
        Initialize the Pinecone search agent.
        
        Args:
            openai_client: Optional pre-configured OpenAI client
            embedding_service: Optional embedding service
            verbose: Enable verbose logging
            agent_config: Optional custom configuration
        """
        super().__init__(verbose=verbose)
        
        self._agent_type = AgentType.PINECONE_SEARCH
        self.agent_config = agent_config or DEFAULT_CONFIG

        self.client = openai_client or AsyncOpenAI(api_key=config.openai_api_key)
        self.embedding_service = embedding_service or EmbeddingService()

        # Initialize Pinecone
        self.pc = pinecone.Pinecone(api_key=config.pinecone_api_key)
        self.index = self.pc.Index(host=config.pinecone_host)

        # Namespaces
        self.machinery_namespace = config.pinecone_machinery_namespace
        self.documents_namespace = config.pinecone_namespace
        
        logger.debug(
            "Initialized PineconeSearchAgent with namespaces: machinery=%s, docs=%s",
            self.machinery_namespace,
            self.documents_namespace,
        )

    # ==================== TOOLS ====================

    @tool(
        name="search_machinery",
        description="Search machinery database for similar equipment",
        parameters={
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "description": "Number of results (default: 5)"},
            "manufacturer": {"type": "string", "description": "Filter by manufacturer (optional)"},
            "equipment_type": {"type": "string", "description": "Filter by equipment type (optional)"}
        },
        required=["query"],
        retry_config=RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            max_attempts=3,
            base_delay_seconds=0.5,
        ),
        tags=["search", "machinery"],
    )
    async def search_machinery_tool(
        self,
        query: str,
        top_k: int = 5,
        manufacturer: str | None = None,
        equipment_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Search the machinery namespace with optional filters.
        
        Args:
            query: Search query
            top_k: Number of results
            manufacturer: Optional manufacturer filter
            equipment_type: Optional equipment type filter
            
        Returns:
            Dict with search results
        """
        try:
            query_embedding = await self.embedding_service.embed_query(query)
            
            # Build filter
            pinecone_filter = self._build_machinery_filter(manufacturer, equipment_type)
            
            results = await self._search_namespace(
                query_embedding,
                self.machinery_namespace,
                top_k,
                pinecone_filter,
            )
            
            formatted = [self._format_result(r) for r in results]
            
            return {
                "success": True,
                "results": [f.to_dict() for f in formatted],
                "count": len(formatted),
                "query": query,
            }
        except Exception as e:
            logger.error("Machinery search error: %s", e)
            return {"success": False, "error": str(e), "results": [], "count": 0}

    @tool(
        name="search_documents",
        description="Search company documents, policies, and knowledge base",
        parameters={
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "description": "Number of results (default: 5)"},
            "category": {"type": "string", "description": "Filter by document category (optional)"},
            "source_file": {"type": "string", "description": "Filter by source file (optional)"}
        },
        required=["query"],
        retry_config=RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            max_attempts=3,
            base_delay_seconds=0.5,
        ),
        tags=["search", "documents"],
    )
    async def search_documents_tool(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
        source_file: str | None = None,
    ) -> dict[str, Any]:
        """
        Search the documents namespace with optional filters.
        
        Args:
            query: Search query
            top_k: Number of results
            category: Optional category filter
            source_file: Optional source file filter
            
        Returns:
            Dict with search results
        """
        try:
            query_embedding = await self.embedding_service.embed_query(query)
            
            # Build filter
            pinecone_filter = self._build_document_filter(category, source_file)
            
            results = await self._search_namespace(
                query_embedding,
                self.documents_namespace,
                top_k,
                pinecone_filter,
            )
            
            formatted = [self._format_result(r) for r in results]
            
            return {
                "success": True,
                "results": [f.to_dict() for f in formatted],
                "count": len(formatted),
                "query": query,
            }
        except Exception as e:
            logger.error("Document search error: %s", e)
            return {"success": False, "error": str(e), "results": [], "count": 0}

    @tool(
        name="find_similar",
        description="Find items similar to a given equipment ID",
        parameters={
            "equipment_id": {"type": "string", "description": "ID of equipment to find similar items for"},
            "top_k": {"type": "integer", "description": "Number of similar items (default: 5)"},
            "same_type_only": {"type": "boolean", "description": "Only return same equipment type"}
        },
        required=["equipment_id"],
        tags=["search", "similarity"],
    )
    async def find_similar_tool(
        self,
        equipment_id: str,
        top_k: int = 5,
        same_type_only: bool = False,
    ) -> dict[str, Any]:
        """
        Find equipment similar to a given ID.
        
        Args:
            equipment_id: ID of the reference equipment
            top_k: Number of similar items
            same_type_only: Only return same equipment type
            
        Returns:
            Dict with similar items
        """
        try:
            # Fetch the vector for the given ID
            fetch_result = self.index.fetch(
                ids=[equipment_id],
                namespace=self.machinery_namespace,
            )

            if not fetch_result.vectors or equipment_id not in fetch_result.vectors:
                return {
                    "success": False,
                    "error": f"Equipment {equipment_id} not found",
                    "results": [],
                }

            vector_data = fetch_result.vectors[equipment_id]
            vector = vector_data.values
            
            # Build filter for same type if requested
            pinecone_filter = None
            if same_type_only and vector_data.metadata:
                equipment_type = vector_data.metadata.get("geraetegruppe")
                if equipment_type:
                    pinecone_filter = {"geraetegruppe": {"$eq": equipment_type}}

            # Search for similar (add 1 to exclude original)
            results = await self._search_namespace(
                vector,
                self.machinery_namespace,
                top_k + 1,
                pinecone_filter,
            )

            # Filter out the original
            results = [r for r in results if r.get("id") != equipment_id][:top_k]
            formatted = [self._format_result(r) for r in results]

            return {
                "success": True,
                "results": [f.to_dict() for f in formatted],
                "count": len(formatted),
                "reference_id": equipment_id,
            }
        except Exception as e:
            logger.error("Find similar error: %s", e)
            return {"success": False, "error": str(e), "results": []}

    @tool(
        name="hybrid_search",
        description="Perform hybrid search combining semantic and keyword matching",
        parameters={
            "query": {"type": "string", "description": "Search query"},
            "keywords": {"type": "array", "items": {"type": "string"}, "description": "Keywords to boost"},
            "top_k": {"type": "integer", "description": "Number of results"},
            "namespace": {"type": "string", "description": "Namespace to search"}
        },
        required=["query"],
        tags=["search", "hybrid"],
    )
    async def hybrid_search_tool(
        self,
        query: str,
        keywords: list[str] | None = None,
        top_k: int = 5,
        namespace: str = "both",
    ) -> dict[str, Any]:
        """
        Perform hybrid search with keyword boosting.
        
        Args:
            query: Search query
            keywords: Keywords to boost in results
            top_k: Number of results
            namespace: Namespace to search
            
        Returns:
            Dict with search results
        """
        try:
            # Build enhanced query with keywords
            enhanced_query = query
            if keywords:
                enhanced_query = f"{query} {' '.join(keywords)}"
            
            query_embedding = await self.embedding_service.embed_query(enhanced_query)
            
            all_results: SearchResults = []
            
            # Search based on namespace
            ns_enum = SearchNamespace(namespace) if namespace != "both" else SearchNamespace.BOTH
            
            if ns_enum in (SearchNamespace.MACHINERY, SearchNamespace.BOTH):
                machinery_results = await self._search_namespace(
                    query_embedding,
                    self.machinery_namespace,
                    top_k,
                )
                all_results.extend(machinery_results)
            
            if ns_enum in (SearchNamespace.DOCUMENTS, SearchNamespace.BOTH):
                doc_results = await self._search_namespace(
                    query_embedding,
                    self.documents_namespace,
                    top_k,
                )
                all_results.extend(doc_results)
            
            # Rerank by keyword relevance if keywords provided
            if keywords:
                all_results = self._rerank_by_keywords(all_results, keywords)
            
            # Sort by score and limit
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            all_results = all_results[:top_k]
            
            formatted = [self._format_result(r) for r in all_results]
            
            return {
                "success": True,
                "results": [f.to_dict() for f in formatted],
                "count": len(formatted),
                "query": query,
                "keywords": keywords,
            }
        except Exception as e:
            logger.error("Hybrid search error: %s", e)
            return {"success": False, "error": str(e), "results": []}

    @tool(
        name="batch_search",
        description="Search for multiple queries in parallel",
        parameters={
            "queries": {"type": "array", "items": {"type": "string"}, "description": "List of queries"},
            "top_k_per_query": {"type": "integer", "description": "Results per query"},
            "namespace": {"type": "string", "description": "Namespace to search"}
        },
        required=["queries"],
        tags=["search", "batch"],
    )
    async def batch_search_tool(
        self,
        queries: list[str],
        top_k_per_query: int = 3,
        namespace: str = "both",
    ) -> dict[str, Any]:
        """
        Search for multiple queries in parallel.
        
        Args:
            queries: List of search queries
            top_k_per_query: Results per query
            namespace: Namespace to search
            
        Returns:
            Dict with results for each query
        """
        if len(queries) > self.agent_config.max_batch_size:
            return {
                "success": False,
                "error": f"Max {self.agent_config.max_batch_size} queries per batch",
                "results": {},
            }
        
        try:
            # Generate all embeddings in parallel
            embedding_tasks = [
                self.embedding_service.embed_query(q) for q in queries
            ]
            embeddings = await asyncio.gather(*embedding_tasks)
            
            # Search all queries in parallel
            ns_enum = SearchNamespace(namespace) if namespace != "both" else SearchNamespace.BOTH
            target_namespace = (
                self.machinery_namespace 
                if ns_enum == SearchNamespace.MACHINERY 
                else self.documents_namespace
            )
            
            search_tasks = []
            for embedding in embeddings:
                if ns_enum == SearchNamespace.BOTH:
                    # Search both namespaces
                    search_tasks.append(self._search_namespace(
                        embedding, self.machinery_namespace, top_k_per_query
                    ))
                    search_tasks.append(self._search_namespace(
                        embedding, self.documents_namespace, top_k_per_query
                    ))
                else:
                    search_tasks.append(self._search_namespace(
                        embedding, target_namespace, top_k_per_query
                    ))
            
            all_results = await asyncio.gather(*search_tasks)
            
            # Organize results by query
            results_by_query: dict[str, list[dict]] = {}
            
            if ns_enum == SearchNamespace.BOTH:
                # Results come in pairs (machinery, documents)
                for i, query in enumerate(queries):
                    machinery_idx = i * 2
                    docs_idx = i * 2 + 1
                    
                    combined = all_results[machinery_idx] + all_results[docs_idx]
                    combined.sort(key=lambda x: x.get("score", 0), reverse=True)
                    
                    formatted = [self._format_result(r).to_dict() for r in combined[:top_k_per_query]]
                    results_by_query[query] = formatted
            else:
                for i, query in enumerate(queries):
                    formatted = [self._format_result(r).to_dict() for r in all_results[i]]
                    results_by_query[query] = formatted
            
            return {
                "success": True,
                "results": results_by_query,
                "query_count": len(queries),
            }
        except Exception as e:
            logger.error("Batch search error: %s", e)
            return {"success": False, "error": str(e), "results": {}}

    @tool(
        name="get_by_id",
        description="Get a specific item by its ID",
        parameters={
            "item_id": {"type": "string", "description": "ID of the item to retrieve"},
            "namespace": {"type": "string", "description": "Namespace to search in"}
        },
        required=["item_id"],
        tags=["retrieval"],
    )
    async def get_by_id_tool(
        self,
        item_id: str,
        namespace: str = "machinery-data",
    ) -> dict[str, Any]:
        """
        Retrieve a specific item by ID.
        
        Args:
            item_id: ID of the item
            namespace: Namespace to search in
            
        Returns:
            Dict with the item data
        """
        try:
            target_ns = (
                self.machinery_namespace 
                if namespace == "machinery-data" 
                else self.documents_namespace
            )
            
            fetch_result = self.index.fetch(
                ids=[item_id],
                namespace=target_ns,
            )
            
            if not fetch_result.vectors or item_id not in fetch_result.vectors:
                return {
                    "success": False,
                    "error": f"Item {item_id} not found",
                    "item": None,
                }
            
            vector_data = fetch_result.vectors[item_id]
            
            result = {
                "id": item_id,
                "metadata": vector_data.metadata or {},
                "namespace": target_ns,
                "score": 1.0,
            }
            
            formatted = self._format_result(result)
            
            return {
                "success": True,
                "item": formatted.to_dict(),
            }
        except Exception as e:
            logger.error("Get by ID error: %s", e)
            return {"success": False, "error": str(e), "item": None}

    @tool(
        name="search_with_filter",
        description="Search with complex metadata filters",
        parameters={
            "query": {"type": "string", "description": "Search query"},
            "filters": {"type": "object", "description": "Metadata filters (Pinecone format)"},
            "top_k": {"type": "integer", "description": "Number of results"},
            "namespace": {"type": "string", "description": "Namespace to search"}
        },
        required=["query", "filters"],
        tags=["search", "filtered"],
    )
    async def search_with_filter_tool(
        self,
        query: str,
        filters: dict[str, Any],
        top_k: int = 5,
        namespace: str = "machinery-data",
    ) -> dict[str, Any]:
        """
        Search with complex metadata filters.
        
        Args:
            query: Search query
            filters: Pinecone-style metadata filters
            top_k: Number of results
            namespace: Namespace to search
            
        Returns:
            Dict with filtered results
        """
        try:
            query_embedding = await self.embedding_service.embed_query(query)
            
            target_ns = (
                self.machinery_namespace 
                if namespace == "machinery-data" 
                else self.documents_namespace
            )
            
            # Build Pinecone filter from provided filters
            pinecone_filter = self._build_filter(filters)
            
            results = await self._search_namespace(
                query_embedding,
                target_ns,
                top_k,
                pinecone_filter,
            )
            
            formatted = [self._format_result(r) for r in results]
            
            return {
                "success": True,
                "results": [f.to_dict() for f in formatted],
                "count": len(formatted),
                "query": query,
                "filters_applied": filters,
            }
        except Exception as e:
            logger.error("Filtered search error: %s", e)
            return {"success": False, "error": str(e), "results": []}

    # ==================== MAIN EXECUTION ====================

    async def _execute(self, context: AgentContext) -> AgentResponse:
        """
        Perform semantic search based on the orchestrator's instruction.
        
        Args:
            context: Agent context with metadata from orchestrator
            
        Returns:
            AgentResponse with search results
        """
        # Extract search parameters from orchestrator
        search_query = context.metadata.get("pinecone_query")
        namespace = context.metadata.get("pinecone_namespace", "both")
        top_k = context.metadata.get("pinecone_top_k", self.agent_config.default_top_k)
        filters = context.metadata.get("pinecone_filters")

        if not search_query:
            return AgentResponse.error_response(
                error="No search query provided by orchestrator",
                agent_type=self._agent_type,
            )

        self.log(f"Searching: {search_query[:50]}... in {namespace}")
        logger.info("Pinecone search: %s in %s", search_query[:50], namespace)

        # Generate embedding
        try:
            query_embedding = await self.embedding_service.embed_query(search_query)
        except Exception as e:
            logger.error("Embedding generation failed: %s", e)
            return AgentResponse.error_response(
                error=f"Embedding generation failed: {e}",
                agent_type=self._agent_type,
            )

        # Build filter if provided
        pinecone_filter = self._build_filter(filters) if filters else None

        all_results: SearchResults = []
        sources: list[dict[str, Any]] = []

        # Search based on namespace selection
        ns_enum = SearchNamespace(namespace) if namespace != "both" else SearchNamespace.BOTH

        if ns_enum in (SearchNamespace.MACHINERY, SearchNamespace.BOTH):
            machinery_results = await self._search_namespace(
                query_embedding,
                self.machinery_namespace,
                top_k,
                pinecone_filter,
            )
            all_results.extend(machinery_results)

        if ns_enum in (SearchNamespace.DOCUMENTS, SearchNamespace.BOTH):
            doc_results = await self._search_namespace(
                query_embedding,
                self.documents_namespace,
                top_k,
                pinecone_filter,
            )
            all_results.extend(doc_results)

        # Sort by score and deduplicate
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Format results
        formatted_results: list[dict[str, Any]] = []
        for result in all_results[:top_k * 2]:  # Return up to 2x top_k after merging
            formatted = self._format_result(result)
            formatted_results.append(formatted.to_dict())
            sources.append({
                "type": "pinecone",
                "namespace": result.get("namespace", "unknown"),
                "id": result.get("id", ""),
                "score": result.get("score", 0),
            })

        self.log(f"Found {len(formatted_results)} results")
        logger.info("Pinecone search returned %d results", len(formatted_results))

        # Store in context for reviewer
        context.pinecone_results = formatted_results

        return AgentResponse.success_response(
            data={
                "results": formatted_results,
                "result_count": len(formatted_results),
                "namespaces_searched": namespace,
                "query": search_query,
            },
            agent_type=self._agent_type,
            sources=sources,
        )

    # ==================== HELPER METHODS ====================

    async def _search_namespace(
        self,
        query_embedding: EmbeddingVector,
        namespace: str,
        top_k: int,
        pinecone_filter: PineconeFilter | None = None,
    ) -> SearchResults:
        """
        Search a specific Pinecone namespace.
        
        Args:
            query_embedding: Query vector
            namespace: Namespace to search
            top_k: Number of results
            pinecone_filter: Optional metadata filter
            
        Returns:
            List of search results
        """
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                filter=pinecone_filter,
            )

            formatted: SearchResults = []
            for match in results.matches:
                formatted.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata or {},
                    "namespace": namespace,
                })

            self.log(f"  {namespace}: {len(formatted)} results")
            return formatted

        except Exception as e:
            logger.error("Namespace %s search error: %s", namespace, e)
            self.log(f"  {namespace} error: {e}")
            return []

    def _build_filter(self, filters: dict[str, Any] | None) -> PineconeFilter | None:
        """
        Build Pinecone filter from provided filters.
        
        Args:
            filters: Raw filter dictionary
            
        Returns:
            Pinecone-compatible filter or None
        """
        if not filters:
            return None
            
        pinecone_filter: PineconeFilter = {}

        for key, value in filters.items():
            if isinstance(value, dict):
                # Already in Pinecone format (e.g., {"$gt": 1000})
                pinecone_filter[key] = value
            elif isinstance(value, list):
                # Multiple values -> $in filter
                pinecone_filter[key] = {"$in": value}
            else:
                # Single value -> $eq filter
                pinecone_filter[key] = {"$eq": value}

        return pinecone_filter if pinecone_filter else None

    def _build_machinery_filter(
        self,
        manufacturer: str | None,
        equipment_type: str | None,
    ) -> PineconeFilter | None:
        """Build filter for machinery search."""
        filters: PineconeFilter = {}
        
        if manufacturer:
            filters["hersteller"] = {"$eq": manufacturer}
        if equipment_type:
            filters["geraetegruppe"] = {"$eq": equipment_type}
        
        return filters if filters else None

    def _build_document_filter(
        self,
        category: str | None,
        source_file: str | None,
    ) -> PineconeFilter | None:
        """Build filter for document search."""
        filters: PineconeFilter = {}
        
        if category:
            filters["category"] = {"$eq": category}
        if source_file:
            filters["source_file"] = {"$eq": source_file}
        
        return filters if filters else None

    def _rerank_by_keywords(
        self,
        results: SearchResults,
        keywords: list[str],
    ) -> SearchResults:
        """
        Rerank results by keyword relevance.
        
        Args:
            results: Original results
            keywords: Keywords to boost
            
        Returns:
            Reranked results
        """
        def keyword_score(result: SearchResult) -> float:
            metadata = result.get("metadata", {})
            content = str(metadata).lower()
            
            matches = sum(1 for kw in keywords if kw.lower() in content)
            return result.get("score", 0) * (1 + 0.1 * matches)
        
        return sorted(results, key=keyword_score, reverse=True)

    def _format_result(self, result: SearchResult) -> FormattedResult:
        """
        Format a single search result for output.
        
        Args:
            result: Raw search result
            
        Returns:
            FormattedResult instance
        """
        metadata = result.get("metadata", {})
        namespace = result.get("namespace", "")

        if namespace == self.machinery_namespace:
            return self._format_machinery_result(result, metadata)
        else:
            return self._format_document_result(result, metadata)

    def _format_machinery_result(
        self,
        result: SearchResult,
        metadata: dict[str, Any],
    ) -> FormattedResult:
        """Format a machinery result."""
        title = metadata.get("titel") or metadata.get("bezeichnung") or "Unbekannt"
        
        return FormattedResult(
            result_type=ResultType.MACHINERY,
            id=result.get("id", ""),
            score=result.get("score", 0),
            title=title,
            content=self._format_machinery_content(metadata),
            metadata={
                "hersteller": metadata.get("hersteller", ""),
                "geraetegruppe": metadata.get("geraetegruppe", ""),
                "kategorie": metadata.get("kategorie", ""),
                "gewicht_kg": metadata.get("gewicht_kg", ""),
                "motor_leistung_kw": metadata.get("motor_leistung_kw", ""),
                "arbeitsbreite_mm": metadata.get("arbeitsbreite_mm", ""),
            },
            namespace=result.get("namespace", ""),
        )

    def _format_document_result(
        self,
        result: SearchResult,
        metadata: dict[str, Any],
    ) -> FormattedResult:
        """Format a document result."""
        return FormattedResult(
            result_type=ResultType.DOCUMENT,
            id=result.get("id", ""),
            score=result.get("score", 0),
            title=metadata.get("title", "Dokument"),
            content=metadata.get("content", "")[:self.agent_config.max_content_length],
            metadata={
                "source_file": metadata.get("source_file", ""),
                "category": metadata.get("category", ""),
            },
            namespace=result.get("namespace", ""),
        )

    def _format_machinery_content(self, metadata: dict[str, Any]) -> str:
        """Format machinery metadata as readable content."""
        lines: list[str] = []

        if metadata.get("hersteller"):
            lines.append(f"Hersteller: {metadata['hersteller']}")
        if metadata.get("bezeichnung"):
            lines.append(f"Modell: {metadata['bezeichnung']}")
        if metadata.get("geraetegruppe"):
            lines.append(f"Typ: {metadata['geraetegruppe']}")
        if metadata.get("kategorie"):
            lines.append(f"Kategorie: {metadata['kategorie']}")

        # Technical specs
        specs: list[str] = []
        if metadata.get("gewicht_kg"):
            specs.append(f"Gewicht: {metadata['gewicht_kg']} kg")
        if metadata.get("motor_leistung_kw"):
            specs.append(f"Leistung: {metadata['motor_leistung_kw']} kW")
        if metadata.get("arbeitsbreite_mm"):
            specs.append(f"Arbeitsbreite: {metadata['arbeitsbreite_mm']} mm")
        if metadata.get("grabtiefe_mm"):
            specs.append(f"Grabtiefe: {metadata['grabtiefe_mm']} mm")

        if specs:
            lines.append("Technische Daten: " + ", ".join(specs))

        # Features
        feature_map = {
            "klimaanlage": "Klimaanlage",
            "schnellwechsler": "Schnellwechsler",
            "gps": "GPS",
            "rueckfahrkamera": "Rückfahrkamera",
        }
        
        features = [
            name for key, name in feature_map.items()
            if metadata.get(key) == "true"
        ]

        if features:
            lines.append("Ausstattung: " + ", ".join(features))

        # Description
        if metadata.get("inhalt"):
            content = metadata["inhalt"][:self.agent_config.max_content_length]
            lines.append(f"\n{content}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"PineconeSearchAgent("
            f"machinery_ns={self.machinery_namespace!r}, "
            f"docs_ns={self.documents_namespace!r}, "
            f"tools={len(self._tools)})"
        )
