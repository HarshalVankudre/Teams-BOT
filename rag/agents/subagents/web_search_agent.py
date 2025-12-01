"""
Web Search SubAgent

Retrieves supplementary information from the web using Tavily API.
Used ONLY for external information not available in internal databases.
"""
from typing import List, Dict, Any, Optional

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

# Try to import Tavily
try:
    from tavily import AsyncTavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False


# Agent metadata
WEB_SEARCH_AGENT_METADATA = AgentMetadata(
    agent_id="web_search",
    name="Web Search Agent",
    description="Searches the web for external information using Tavily",
    detailed_description="""Sucht im Internet nach ergänzenden Informationen mittels Tavily API.
Verwende diesen Agenten NUR für:
- Aktuelle Marktpreise und Preisvergleiche
- Herstellerinformationen und technische Spezifikationen von außen
- Aktuelle Nachrichten zu Baumaschinen
- Informationen die NICHT in der internen Datenbank verfügbar sind

WICHTIG: Interne Daten (PostgreSQL, Pinecone) haben IMMER Vorrang!
Web-Suche nur als Ergänzung verwenden.""",
    capabilities=[AgentCapability.WEB_SEARCH],
    uses_reasoning=False,
    parameters={
        "search_query": {
            "type": "string",
            "description": "Die Suchanfrage für die Web-Suche"
        },
        "search_depth": {
            "type": "string",
            "enum": ["basic", "advanced"],
            "description": "Suchtiefe: 'basic' für schnelle Suche, 'advanced' für tiefere Recherche"
        },
        "max_results": {
            "type": "integer",
            "description": "Maximale Anzahl der Ergebnisse (Standard: 3)"
        }
    },
    example_queries=[
        "Aktueller Marktpreis für Liebherr Bagger",
        "Neueste Modelle von Caterpillar 2024",
        "Vergleich Dieselmotor vs Elektromotor Baumaschinen",
        "Abgasnorm Stage V Anforderungen"
    ],
    priority=3  # Lower priority - internal data comes first
)


@register_subagent()
class WebSearchAgent(SubAgentBase):
    """
    Performs web searches using Tavily API.
    Used for supplementary information only - internal data takes priority.
    """

    METADATA = WEB_SEARCH_AGENT_METADATA

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = None,
        verbose: bool = False
    ):
        super().__init__(verbose=verbose)
        self._agent_type = AgentType.WEB_SEARCH

        self.api_key = api_key or config.tavily_api_key
        self.max_results = max_results or config.web_search_max_results
        self.enabled = config.enable_web_search and TAVILY_AVAILABLE and bool(self.api_key)

        if self.enabled:
            self.client = AsyncTavilyClient(api_key=self.api_key)
            self.log("Tavily web search enabled")
        else:
            self.client = None
            self.log("Web search disabled (missing API key or tavily not installed)")

    # ==================== TOOLS ====================

    @tool(
        name="web_search",
        description="Search the web for information",
        parameters={
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "description": "Maximum results (default: 3)"},
            "search_depth": {"type": "string", "description": "basic or advanced"}
        },
        required=["query"]
    )
    async def web_search_tool(
        self,
        query: str,
        max_results: int = 3,
        search_depth: str = "basic"
    ) -> Dict[str, Any]:
        """Perform a web search"""
        if not self.enabled:
            return {
                "success": False,
                "error": "Web search is disabled",
                "results": []
            }

        try:
            response = await self.client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                include_answer=False,
                include_raw_content=False
            )

            results = [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0)
                }
                for item in response.get("results", [])
            ]

            return {
                "success": True,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": []
            }

    @tool(
        name="search_with_context",
        description="Search the web with additional context to improve results",
        parameters={
            "query": {"type": "string", "description": "Main search query"},
            "context_hint": {"type": "string", "description": "Additional context to refine search"},
            "max_results": {"type": "integer", "description": "Maximum results"}
        },
        required=["query"]
    )
    async def search_with_context_tool(
        self,
        query: str,
        context_hint: str = None,
        max_results: int = None
    ) -> Dict[str, Any]:
        """Convenience method for direct web search with optional context."""
        if not self.enabled:
            return {"success": False, "error": "Web search disabled", "results": []}

        search_query = f"{query} {context_hint}" if context_hint else query

        try:
            response = await self.client.search(
                query=search_query,
                search_depth="basic",
                max_results=max_results or self.max_results,
                include_answer=False,
                include_raw_content=False
            )

            results = [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0)
                }
                for item in response.get("results", [])
            ]

            return {"success": True, "results": results, "count": len(results)}
        except Exception as e:
            return {"success": False, "error": str(e), "results": []}

    # ==================== MAIN EXECUTION ====================

    async def _execute(self, context: AgentContext) -> AgentResponse:
        """
        Perform web search for the given query.
        """
        if not self.enabled:
            return AgentResponse.success_response(
                data={
                    "results": [],
                    "result_count": 0,
                    "message": "Web search is disabled"
                },
                agent_type=self._agent_type,
                reasoning="Web search not available - using internal data only"
            )

        # Extract search parameters
        search_query = context.metadata.get("web_query", context.user_query)
        max_results = context.metadata.get("web_max_results", self.max_results)
        search_depth = context.metadata.get("web_search_depth", "basic")

        self.log(f"Searching web: {search_query[:50]}...")

        # Use our tool
        result = await self.web_search_tool(
            query=search_query,
            max_results=max_results,
            search_depth=search_depth
        )

        if not result["success"]:
            return AgentResponse.error_response(
                error=f"Web search failed: {result.get('error')}",
                agent_type=self._agent_type
            )

        results = result["results"]
        sources = [
            {
                "type": "web",
                "title": r["title"],
                "url": r["url"],
                "score": r["score"]
            }
            for r in results
        ]

        self.log(f"Found {len(results)} web results")

        # Store in context for reviewer
        context.web_results = results

        return AgentResponse.success_response(
            data={
                "results": results,
                "result_count": len(results),
                "query": search_query
            },
            agent_type=self._agent_type,
            sources=sources
        )

    # ==================== UTILITY METHODS ====================

    def format_results_for_context(
        self,
        results: List[Dict[str, Any]],
        max_content_length: int = 500
    ) -> str:
        """
        Format web results as context for the reviewer agent.

        Args:
            results: Web search results
            max_content_length: Max characters per result content

        Returns:
            Formatted string for use in prompts
        """
        if not results:
            return "Keine Web-Ergebnisse gefunden."

        formatted_parts = []
        for i, result in enumerate(results, 1):
            content = result.get("content", "")[:max_content_length]
            formatted_parts.append(f"""
### Web-Quelle {i}: {result.get('title', 'Unbekannt')}
**URL:** {result.get('url', '')}
**Relevanz:** {result.get('score', 0):.2%}

{content}
""")

        return "\n---\n".join(formatted_parts)
