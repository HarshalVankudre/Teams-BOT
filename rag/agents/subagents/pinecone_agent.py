"""
Pinecone Search SubAgent

Performs semantic search in Pinecone vector store.
Used for recommendations, conceptual queries, and document retrieval.
"""
from typing import List, Dict, Any, Optional
import pinecone
from openai import AsyncOpenAI

from .interface import (
    SubAgentBase,
    tool,
    register_subagent,
    AgentMetadata,
    AgentCapability,
    AgentContext,
    AgentResponse,
    AgentType
)
from ...config import config
from ...embeddings import EmbeddingService


# Agent metadata
PINECONE_AGENT_METADATA = AgentMetadata(
    agent_id="pinecone",
    name="Pinecone Semantic Search Agent",
    description="Performs semantic/similarity search in the vector database",
    detailed_description="""Führt semantische Suche in der Pinecone Vektordatenbank durch.
Verwende diesen Agenten für:
- Empfehlungen und ähnliche Geräte finden
- Konzeptuelle Suchen ("Geräte für Straßenbau")
- Dokumentensuche in Unternehmensunterlagen
- Wenn die Anfrage nicht strukturiert/präzise ist
- Wenn du ähnliche Produkte oder Alternativen finden willst""",
    capabilities=[AgentCapability.SEMANTIC_SEARCH],
    uses_reasoning=False,
    parameters={
        "search_query": {
            "type": "string",
            "description": "Die Suchanfrage für semantische Suche"
        },
        "namespace": {
            "type": "string",
            "enum": ["machinery-data", "rueko-documents", "both"],
            "description": "Welcher Namespace durchsucht werden soll"
        },
        "top_k": {
            "type": "integer",
            "description": "Anzahl der gewünschten Ergebnisse (Standard: 5)"
        }
    },
    example_queries=[
        "Empfehle mir einen Bagger für schwere Einsätze",
        "Welche Geräte eignen sich für den Straßenbau?",
        "Ähnliche Maschinen wie der CAT 320",
        "Dokumente zu Wartungsintervallen",
        "Technische Spezifikationen für Walzen"
    ],
    priority=8
)


@register_subagent()
class PineconeSearchAgent(SubAgentBase):
    """
    Performs semantic search across Pinecone namespaces.
    Searches both machinery data and company documents.
    """

    METADATA = PINECONE_AGENT_METADATA

    def __init__(
        self,
        openai_client: Optional[AsyncOpenAI] = None,
        embedding_service: Optional[EmbeddingService] = None,
        verbose: bool = False
    ):
        super().__init__(verbose=verbose)
        self._agent_type = AgentType.PINECONE_SEARCH

        self.client = openai_client or AsyncOpenAI(api_key=config.openai_api_key)
        self.embedding_service = embedding_service or EmbeddingService()

        # Initialize Pinecone
        self.pc = pinecone.Pinecone(api_key=config.pinecone_api_key)
        self.index = self.pc.Index(host=config.pinecone_host)

        # Namespaces
        self.machinery_namespace = config.pinecone_machinery_namespace
        self.documents_namespace = config.pinecone_namespace

        # Search configuration
        self.default_top_k = config.search_top_k

    # ==================== TOOLS ====================

    @tool(
        name="search_machinery",
        description="Search machinery database for similar equipment",
        parameters={
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "description": "Number of results (default: 5)"}
        },
        required=["query"]
    )
    async def search_machinery_tool(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search the machinery namespace"""
        try:
            query_embedding = await self.embedding_service.embed_query(query)
            results = await self._search_namespace(
                query_embedding,
                self.machinery_namespace,
                top_k
            )
            return {"success": True, "results": results, "count": len(results)}
        except Exception as e:
            return {"success": False, "error": str(e), "results": [], "count": 0}

    @tool(
        name="search_documents",
        description="Search company documents and policies",
        parameters={
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "description": "Number of results (default: 5)"}
        },
        required=["query"]
    )
    async def search_documents_tool(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search the documents namespace"""
        try:
            query_embedding = await self.embedding_service.embed_query(query)
            results = await self._search_namespace(
                query_embedding,
                self.documents_namespace,
                top_k
            )
            return {"success": True, "results": results, "count": len(results)}
        except Exception as e:
            return {"success": False, "error": str(e), "results": [], "count": 0}

    @tool(
        name="find_similar",
        description="Find items similar to a given equipment ID",
        parameters={
            "equipment_id": {"type": "string", "description": "ID of the equipment to find similar items for"},
            "top_k": {"type": "integer", "description": "Number of similar items (default: 5)"}
        },
        required=["equipment_id"]
    )
    async def find_similar_tool(self, equipment_id: str, top_k: int = 5) -> Dict[str, Any]:
        """Find equipment similar to a given ID"""
        try:
            # First, get the vector for the given ID
            fetch_result = self.index.fetch(
                ids=[equipment_id],
                namespace=self.machinery_namespace
            )

            if not fetch_result.vectors or equipment_id not in fetch_result.vectors:
                return {"success": False, "error": f"Equipment {equipment_id} not found", "results": []}

            vector = fetch_result.vectors[equipment_id].values

            # Search for similar
            results = await self._search_namespace(
                vector,
                self.machinery_namespace,
                top_k + 1  # Add 1 to exclude the original
            )

            # Filter out the original
            results = [r for r in results if r.get("id") != equipment_id][:top_k]

            return {"success": True, "results": results, "count": len(results)}
        except Exception as e:
            return {"success": False, "error": str(e), "results": [], "count": 0}

    # ==================== MAIN EXECUTION ====================

    async def _execute(self, context: AgentContext) -> AgentResponse:
        """
        Perform semantic search based on the orchestrator's instruction.
        Sub-agents only execute specific instructions - they don't see the original query.
        """
        # Extract search parameters from orchestrator instruction ONLY (no fallback)
        search_query = context.metadata.get("pinecone_query")
        namespace = context.metadata.get("pinecone_namespace", "both")
        top_k = context.metadata.get("pinecone_top_k", self.default_top_k)
        filters = context.metadata.get("pinecone_filters", None)

        if not search_query:
            return AgentResponse.error_response(
                error="No search query provided by orchestrator",
                agent_type=self._agent_type
            )

        self.log(f"Searching: {search_query[:50]}... in {namespace}")

        # Generate embedding for the query
        try:
            query_embedding = await self.embedding_service.embed_query(search_query)
        except Exception as e:
            return AgentResponse.error_response(
                error=f"Embedding generation failed: {str(e)}",
                agent_type=self._agent_type
            )

        # Build Pinecone filter if provided
        pinecone_filter = self._build_filter(filters) if filters else None

        all_results = []
        sources = []

        # Search based on namespace selection
        if namespace in ["machinery-data", "both"]:
            machinery_results = await self._search_namespace(
                query_embedding,
                self.machinery_namespace,
                top_k,
                pinecone_filter
            )
            all_results.extend(machinery_results)

        if namespace in ["rueko-documents", "both"]:
            doc_results = await self._search_namespace(
                query_embedding,
                self.documents_namespace,
                top_k,
                pinecone_filter
            )
            all_results.extend(doc_results)

        # Sort by score and deduplicate
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Format results
        formatted_results = []
        for result in all_results[:top_k * 2]:  # Return up to 2x top_k after merging
            formatted = self._format_result(result)
            formatted_results.append(formatted)
            sources.append({
                "type": "pinecone",
                "namespace": result.get("namespace", "unknown"),
                "id": result.get("id", ""),
                "score": result.get("score", 0)
            })

        self.log(f"Found {len(formatted_results)} results")

        # Store in context for reviewer
        context.pinecone_results = formatted_results

        return AgentResponse.success_response(
            data={
                "results": formatted_results,
                "result_count": len(formatted_results),
                "namespaces_searched": namespace,
                "query": search_query
            },
            agent_type=self._agent_type,
            sources=sources
        )

    async def _search_namespace(
        self,
        query_embedding: List[float],
        namespace: str,
        top_k: int,
        pinecone_filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search a specific Pinecone namespace"""
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                filter=pinecone_filter
            )

            formatted = []
            for match in results.matches:
                formatted.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata or {},
                    "namespace": namespace
                })

            self.log(f"  {namespace}: {len(formatted)} results")
            return formatted

        except Exception as e:
            self.log(f"  {namespace} error: {str(e)}")
            return []

    def _build_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build Pinecone filter from provided filters"""
        pinecone_filter = {}

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

    def _format_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single search result for output"""
        metadata = result.get("metadata", {})
        namespace = result.get("namespace", "")

        if namespace == self.machinery_namespace:
            # Machinery data formatting
            return {
                "type": "machinery",
                "id": result.get("id", ""),
                "score": result.get("score", 0),
                "title": metadata.get("titel", metadata.get("bezeichnung", "Unbekannt")),
                "hersteller": metadata.get("hersteller", ""),
                "geraetegruppe": metadata.get("geraetegruppe", ""),
                "kategorie": metadata.get("kategorie", ""),
                "gewicht_kg": metadata.get("gewicht_kg", ""),
                "motor_leistung_kw": metadata.get("motor_leistung_kw", ""),
                "arbeitsbreite_mm": metadata.get("arbeitsbreite_mm", ""),
                "content": self._format_machinery_content(metadata),
                "raw_metadata": metadata
            }
        else:
            # Document formatting
            return {
                "type": "document",
                "id": result.get("id", ""),
                "score": result.get("score", 0),
                "title": metadata.get("title", "Dokument"),
                "source_file": metadata.get("source_file", ""),
                "category": metadata.get("category", ""),
                "content": metadata.get("content", ""),
                "raw_metadata": metadata
            }

    def _format_machinery_content(self, metadata: Dict) -> str:
        """Format machinery metadata as readable content"""
        lines = []

        if metadata.get("hersteller"):
            lines.append(f"Hersteller: {metadata['hersteller']}")
        if metadata.get("bezeichnung"):
            lines.append(f"Modell: {metadata['bezeichnung']}")
        if metadata.get("geraetegruppe"):
            lines.append(f"Typ: {metadata['geraetegruppe']}")
        if metadata.get("kategorie"):
            lines.append(f"Kategorie: {metadata['kategorie']}")

        # Technical specs
        specs = []
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
        features = []
        for feature in ["klimaanlage", "schnellwechsler", "gps", "rueckfahrkamera"]:
            if metadata.get(feature) == "true":
                feature_names = {
                    "klimaanlage": "Klimaanlage",
                    "schnellwechsler": "Schnellwechsler",
                    "gps": "GPS",
                    "rueckfahrkamera": "Rückfahrkamera"
                }
                features.append(feature_names.get(feature, feature))

        if features:
            lines.append("Ausstattung: " + ", ".join(features))

        # Description
        if metadata.get("inhalt"):
            lines.append(f"\n{metadata['inhalt'][:500]}")

        return "\n".join(lines)
