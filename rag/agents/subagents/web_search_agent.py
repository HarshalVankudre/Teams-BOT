"""
Web Search SubAgent

Retrieves supplementary information from the web using Tavily API.
Used ONLY for external information not available in internal databases.

Features:
    - Web search with configurable depth
    - URL content extraction
    - News search
    - Domain-specific search
    - Result caching
    - Rate limiting

Author: RÜKO GmbH Baumaschinen
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Final, TypeAlias

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

if TYPE_CHECKING:
    from collections.abc import MutableMapping

# Try to import Tavily
try:
    from tavily import AsyncTavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    AsyncTavilyClient = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

# Type aliases
WebResult: TypeAlias = dict[str, Any]
WebResults: TypeAlias = list[WebResult]


class SearchDepth(str, Enum):
    """Search depth options."""
    
    BASIC = "basic"
    ADVANCED = "advanced"


class SearchType(str, Enum):
    """Types of web searches."""
    
    GENERAL = "general"
    NEWS = "news"
    ACADEMIC = "academic"


class WebSearchError(Exception):
    """Raised when web search fails."""
    
    def __init__(self, message: str, query: str | None = None):
        self.query = query
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class WebSearchAgentConfig:
    """Configuration for the web search agent."""
    
    # Search defaults
    default_max_results: int = 3
    max_results_limit: int = 10
    default_search_depth: SearchDepth = SearchDepth.BASIC
    
    # Rate limiting
    requests_per_minute: int = 30
    rate_limit_window_seconds: int = 60
    
    # Caching
    enable_cache: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    max_cache_size: int = 100
    
    # Result formatting
    max_content_length: int = 500
    include_urls: bool = True


DEFAULT_CONFIG: Final[WebSearchAgentConfig] = WebSearchAgentConfig()


@dataclass(slots=True)
class CacheEntry:
    """A cached search result."""
    
    results: WebResults
    timestamp: float
    query: str
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if the cache entry has expired."""
        return time.time() - self.timestamp > ttl_seconds


class LRUCache:
    """Simple LRU cache for search results."""
    
    __slots__ = ("_cache", "_max_size", "_ttl_seconds")
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300) -> None:
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
    
    def _make_key(self, query: str, **params: Any) -> str:
        """Create a cache key from query and parameters."""
        key_str = f"{query}:{sorted(params.items())}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, query: str, **params: Any) -> WebResults | None:
        """Get a cached result if available and not expired."""
        key = self._make_key(query, **params)
        
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        
        if entry.is_expired(self._ttl_seconds):
            del self._cache[key]
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return entry.results
    
    def set(self, query: str, results: WebResults, **params: Any) -> None:
        """Cache a search result."""
        key = self._make_key(query, **params)
        
        # Remove oldest if at capacity
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        
        self._cache[key] = CacheEntry(
            results=results,
            timestamp=time.time(),
            query=query,
        )
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
    
    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._cache)


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    __slots__ = ("_requests", "_max_requests", "_window_seconds")
    
    def __init__(self, max_requests: int = 30, window_seconds: int = 60) -> None:
        self._requests: list[float] = []
        self._max_requests = max_requests
        self._window_seconds = window_seconds
    
    def _clean_old_requests(self) -> None:
        """Remove requests outside the current window."""
        cutoff = time.time() - self._window_seconds
        self._requests = [t for t in self._requests if t > cutoff]
    
    def can_make_request(self) -> bool:
        """Check if a request can be made without exceeding rate limit."""
        self._clean_old_requests()
        return len(self._requests) < self._max_requests
    
    def record_request(self) -> None:
        """Record that a request was made."""
        self._requests.append(time.time())
    
    async def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        while not self.can_make_request():
            # Wait until oldest request expires
            wait_time = self._requests[0] + self._window_seconds - time.time()
            if wait_time > 0:
                await asyncio.sleep(wait_time + 0.1)
            self._clean_old_requests()
    
    @property
    def requests_remaining(self) -> int:
        """Number of requests remaining in current window."""
        self._clean_old_requests()
        return max(0, self._max_requests - len(self._requests))


@dataclass(slots=True)
class FormattedWebResult:
    """A formatted web search result."""
    
    title: str
    url: str
    content: str
    score: float
    source_domain: str
    published_date: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "score": self.score,
            "source_domain": self.source_domain,
        }
        if self.published_date:
            result["published_date"] = self.published_date
        return result


# Agent metadata
WEB_SEARCH_AGENT_METADATA: Final[AgentMetadata] = AgentMetadata(
    agent_id="web_search",
    name="Web Search Agent",
    description="Searches the web for external information using Tavily",
    detailed_description="""Sucht im Internet nach ergänzenden Informationen mittels Tavily API.
Verwende diesen Agenten NUR für:
- Aktuelle Marktpreise und Preisvergleiche
- Herstellerinformationen und technische Spezifikationen von außen
- Aktuelle Nachrichten zu Baumaschinen
- Informationen die NICHT in der internen Datenbank verfügbar sind
- Branchentrends und Marktanalysen
- Regulatorische Informationen und Normen

WICHTIG: Interne Daten (PostgreSQL, Pinecone) haben IMMER Vorrang!
Web-Suche nur als Ergänzung verwenden.""",
    capabilities=[AgentCapability.WEB_SEARCH],
    uses_reasoning=False,
    parameters={
        "web_query": {
            "type": "string",
            "description": "Die Suchanfrage für die Web-Suche"
        },
        "web_search_depth": {
            "type": "string",
            "enum": ["basic", "advanced"],
            "description": "Suchtiefe: 'basic' für schnelle Suche, 'advanced' für tiefere Recherche"
        },
        "web_max_results": {
            "type": "integer",
            "description": "Maximale Anzahl der Ergebnisse (Standard: 3)"
        },
        "web_include_domains": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Nur diese Domains durchsuchen (optional)"
        },
        "web_exclude_domains": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Diese Domains ausschließen (optional)"
        }
    },
    example_queries=[
        "Aktueller Marktpreis für Liebherr Bagger",
        "Neueste Modelle von Caterpillar 2024",
        "Vergleich Dieselmotor vs Elektromotor Baumaschinen",
        "Abgasnorm Stage V Anforderungen",
        "Baumaschinenmarkt Deutschland Trends",
        "EU Regulierungen Baumaschinen Emissionen",
    ],
    priority=3,  # Lower priority - internal data comes first
)


@register_subagent()
class WebSearchAgent(SubAgentBase):
    """
    Performs web searches using Tavily API.
    
    Used for supplementary information only - internal data takes priority.
    Includes caching and rate limiting to optimize API usage.
    
    Attributes:
        api_key: Tavily API key
        enabled: Whether web search is available
        client: Tavily client instance
        agent_config: Configuration settings
    """

    METADATA = WEB_SEARCH_AGENT_METADATA

    __slots__ = (
        "api_key",
        "enabled",
        "client",
        "agent_config",
        "_cache",
        "_rate_limiter",
    )

    def __init__(
        self,
        api_key: str | None = None,
        verbose: bool = False,
        agent_config: WebSearchAgentConfig | None = None,
    ) -> None:
        """
        Initialize the web search agent.
        
        Args:
            api_key: Optional Tavily API key (uses config if not provided)
            verbose: Enable verbose logging
            agent_config: Optional custom configuration
        """
        super().__init__(verbose=verbose)
        
        self._agent_type = AgentType.WEB_SEARCH
        self.agent_config = agent_config or DEFAULT_CONFIG

        self.api_key = api_key or config.tavily_api_key
        self.enabled = (
            config.enable_web_search 
            and TAVILY_AVAILABLE 
            and bool(self.api_key)
        )

        if self.enabled:
            self.client = AsyncTavilyClient(api_key=self.api_key)
            self.log("Tavily web search enabled")
            logger.info("Web search agent initialized with Tavily")
        else:
            self.client = None
            self.log("Web search disabled (missing API key or tavily not installed)")
            logger.warning("Web search disabled")
        
        # Initialize cache and rate limiter
        self._cache = LRUCache(
            max_size=self.agent_config.max_cache_size,
            ttl_seconds=self.agent_config.cache_ttl_seconds,
        )
        self._rate_limiter = RateLimiter(
            max_requests=self.agent_config.requests_per_minute,
            window_seconds=self.agent_config.rate_limit_window_seconds,
        )

    # ==================== TOOLS ====================

    @tool(
        name="web_search",
        description="Search the web for information",
        parameters={
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "description": "Maximum results (default: 3)"},
            "search_depth": {"type": "string", "description": "basic or advanced"},
            "include_domains": {"type": "array", "items": {"type": "string"}, "description": "Only search these domains"},
            "exclude_domains": {"type": "array", "items": {"type": "string"}, "description": "Exclude these domains"}
        },
        required=["query"],
        retry_config=RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            max_attempts=3,
            base_delay_seconds=1.0,
        ),
        tags=["search", "web"],
    )
    async def web_search_tool(
        self,
        query: str,
        max_results: int = 3,
        search_depth: str = "basic",
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Perform a web search.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            search_depth: "basic" or "advanced"
            include_domains: Only search these domains
            exclude_domains: Exclude these domains
            
        Returns:
            Dict with search results
        """
        if not self.enabled:
            return {
                "success": False,
                "error": "Web search is disabled",
                "results": [],
            }

        # Check cache first
        if self.agent_config.enable_cache:
            cached = self._cache.get(
                query,
                max_results=max_results,
                search_depth=search_depth,
            )
            if cached is not None:
                logger.debug("Cache hit for query: %s", query[:50])
                return {
                    "success": True,
                    "results": cached,
                    "count": len(cached),
                    "cached": True,
                }

        # Wait for rate limit if needed
        await self._rate_limiter.wait_if_needed()

        try:
            self._rate_limiter.record_request()
            
            response = await self.client.search(
                query=query,
                search_depth=search_depth,
                max_results=min(max_results, self.agent_config.max_results_limit),
                include_answer=False,
                include_raw_content=False,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            )

            results = [
                self._format_result(item).to_dict()
                for item in response.get("results", [])
            ]

            # Cache the results
            if self.agent_config.enable_cache:
                self._cache.set(
                    query,
                    results,
                    max_results=max_results,
                    search_depth=search_depth,
                )

            return {
                "success": True,
                "results": results,
                "count": len(results),
                "cached": False,
            }
        except Exception as e:
            logger.error("Web search error: %s", e)
            return {
                "success": False,
                "error": str(e),
                "results": [],
            }

    @tool(
        name="search_news",
        description="Search for recent news articles",
        parameters={
            "query": {"type": "string", "description": "News search query"},
            "max_results": {"type": "integer", "description": "Maximum results"},
            "days_back": {"type": "integer", "description": "How many days back to search"}
        },
        required=["query"],
        tags=["search", "news"],
    )
    async def search_news_tool(
        self,
        query: str,
        max_results: int = 5,
        days_back: int = 7,
    ) -> dict[str, Any]:
        """
        Search for recent news articles.
        
        Args:
            query: News search query
            max_results: Maximum number of results
            days_back: How many days back to search
            
        Returns:
            Dict with news results
        """
        if not self.enabled:
            return {"success": False, "error": "Web search is disabled", "results": []}

        # Add time context to query
        enhanced_query = f"{query} news {datetime.now().year}"
        
        # Use news-focused domains
        news_domains = [
            "reuters.com",
            "bloomberg.com",
            "handelsblatt.com",
            "faz.net",
            "spiegel.de",
            "manager-magazin.de",
            "baumaschine.de",
            "lectura-specs.de",
        ]

        return await self.web_search_tool(
            query=enhanced_query,
            max_results=max_results,
            search_depth="advanced",
            include_domains=news_domains,
        )

    @tool(
        name="search_technical_specs",
        description="Search for technical specifications of equipment",
        parameters={
            "manufacturer": {"type": "string", "description": "Equipment manufacturer"},
            "model": {"type": "string", "description": "Equipment model"},
            "spec_type": {"type": "string", "description": "Type of spec (weight, power, dimensions, etc.)"}
        },
        required=["manufacturer", "model"],
        tags=["search", "technical"],
    )
    async def search_technical_specs_tool(
        self,
        manufacturer: str,
        model: str,
        spec_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Search for technical specifications of equipment.
        
        Args:
            manufacturer: Equipment manufacturer
            model: Equipment model
            spec_type: Optional specific type of specification
            
        Returns:
            Dict with technical specification results
        """
        if not self.enabled:
            return {"success": False, "error": "Web search is disabled", "results": []}

        query_parts = [manufacturer, model, "technische daten", "specifications"]
        if spec_type:
            query_parts.append(spec_type)
        
        query = " ".join(query_parts)
        
        # Use technical specification sources
        tech_domains = [
            "lectura-specs.de",
            "ritchiespecs.com",
            "mascus.de",
            "machinerytrader.com",
        ]

        return await self.web_search_tool(
            query=query,
            max_results=5,
            search_depth="advanced",
            include_domains=tech_domains,
        )

    @tool(
        name="search_market_prices",
        description="Search for current market prices of equipment",
        parameters={
            "equipment_type": {"type": "string", "description": "Type of equipment"},
            "manufacturer": {"type": "string", "description": "Optional manufacturer"},
            "condition": {"type": "string", "description": "new or used"}
        },
        required=["equipment_type"],
        tags=["search", "pricing"],
    )
    async def search_market_prices_tool(
        self,
        equipment_type: str,
        manufacturer: str | None = None,
        condition: str = "used",
    ) -> dict[str, Any]:
        """
        Search for current market prices.
        
        Args:
            equipment_type: Type of equipment
            manufacturer: Optional manufacturer filter
            condition: "new" or "used"
            
        Returns:
            Dict with pricing information
        """
        if not self.enabled:
            return {"success": False, "error": "Web search is disabled", "results": []}

        query_parts = [equipment_type]
        if manufacturer:
            query_parts.append(manufacturer)
        
        condition_term = "gebraucht" if condition == "used" else "neu"
        query_parts.extend(["preis", condition_term, str(datetime.now().year)])
        
        query = " ".join(query_parts)
        
        # Use marketplace domains
        marketplace_domains = [
            "mascus.de",
            "machinerytrader.com",
            "baupool.com",
            "truckscout24.de",
            "mobile.de",
        ]

        return await self.web_search_tool(
            query=query,
            max_results=5,
            search_depth="advanced",
            include_domains=marketplace_domains,
        )

    @tool(
        name="search_regulations",
        description="Search for regulations and compliance information",
        parameters={
            "topic": {"type": "string", "description": "Regulation topic"},
            "region": {"type": "string", "description": "Geographic region (EU, Germany, etc.)"}
        },
        required=["topic"],
        tags=["search", "compliance"],
    )
    async def search_regulations_tool(
        self,
        topic: str,
        region: str = "EU",
    ) -> dict[str, Any]:
        """
        Search for regulations and compliance information.
        
        Args:
            topic: Regulation topic
            region: Geographic region
            
        Returns:
            Dict with regulatory information
        """
        if not self.enabled:
            return {"success": False, "error": "Web search is disabled", "results": []}

        query = f"{topic} {region} regulation vorschrift baumaschinen"
        
        # Use official and regulatory sources
        regulatory_domains = [
            "eur-lex.europa.eu",
            "gesetze-im-internet.de",
            "baua.de",
            "umweltbundesamt.de",
            "vdma.org",
        ]

        return await self.web_search_tool(
            query=query,
            max_results=5,
            search_depth="advanced",
            include_domains=regulatory_domains,
        )

    @tool(
        name="search_with_context",
        description="Search the web with additional context to improve results",
        parameters={
            "query": {"type": "string", "description": "Main search query"},
            "context_hint": {"type": "string", "description": "Additional context"},
            "max_results": {"type": "integer", "description": "Maximum results"}
        },
        required=["query"],
        tags=["search", "contextual"],
    )
    async def search_with_context_tool(
        self,
        query: str,
        context_hint: str | None = None,
        max_results: int | None = None,
    ) -> dict[str, Any]:
        """
        Convenience method for direct web search with optional context.
        
        Args:
            query: Main search query
            context_hint: Additional context to refine search
            max_results: Maximum number of results
            
        Returns:
            Dict with search results
        """
        if not self.enabled:
            return {"success": False, "error": "Web search disabled", "results": []}

        search_query = f"{query} {context_hint}" if context_hint else query
        
        return await self.web_search_tool(
            query=search_query,
            max_results=max_results or self.agent_config.default_max_results,
            search_depth="basic",
        )

    @tool(
        name="get_cache_stats",
        description="Get statistics about the search cache",
        parameters={},
        tags=["utility"],
    )
    async def get_cache_stats_tool(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": self._cache.size,
            "max_size": self.agent_config.max_cache_size,
            "ttl_seconds": self.agent_config.cache_ttl_seconds,
            "cache_enabled": self.agent_config.enable_cache,
            "rate_limit_remaining": self._rate_limiter.requests_remaining,
        }

    @tool(
        name="clear_cache",
        description="Clear the search result cache",
        parameters={},
        tags=["utility"],
    )
    async def clear_cache_tool(self) -> dict[str, Any]:
        """Clear the search cache."""
        old_size = self._cache.size
        self._cache.clear()
        return {
            "success": True,
            "cleared_entries": old_size,
        }

    # ==================== MAIN EXECUTION ====================

    async def _execute(self, context: AgentContext) -> AgentResponse:
        """
        Perform web search based on orchestrator's instruction.
        
        Args:
            context: Agent context with metadata from orchestrator
            
        Returns:
            AgentResponse with search results
        """
        if not self.enabled:
            return AgentResponse.success_response(
                data={
                    "results": [],
                    "result_count": 0,
                    "message": "Web search is disabled",
                },
                agent_type=self._agent_type,
                reasoning="Web search not available - using internal data only",
            )

        # Extract search parameters from orchestrator
        search_query = context.metadata.get("web_query")
        max_results = context.metadata.get(
            "web_max_results",
            self.agent_config.default_max_results,
        )
        search_depth = context.metadata.get(
            "web_search_depth",
            self.agent_config.default_search_depth.value,
        )
        include_domains = context.metadata.get("web_include_domains")
        exclude_domains = context.metadata.get("web_exclude_domains")

        if not search_query:
            return AgentResponse.error_response(
                error="No search query provided by orchestrator",
                agent_type=self._agent_type,
            )

        self.log(f"Searching web: {search_query[:50]}...")
        logger.info("Web search: %s", search_query[:50])

        # Use our tool
        result = await self.web_search_tool(
            query=search_query,
            max_results=max_results,
            search_depth=search_depth,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )

        if not result["success"]:
            return AgentResponse.error_response(
                error=f"Web search failed: {result.get('error')}",
                agent_type=self._agent_type,
            )

        results = result["results"]
        sources = [
            {
                "type": "web",
                "title": r["title"],
                "url": r["url"],
                "score": r["score"],
                "domain": r.get("source_domain", ""),
            }
            for r in results
        ]

        self.log(f"Found {len(results)} web results")
        logger.info("Web search returned %d results", len(results))

        # Store in context for reviewer
        context.web_results = results

        return AgentResponse.success_response(
            data={
                "results": results,
                "result_count": len(results),
                "query": search_query,
                "cached": result.get("cached", False),
            },
            agent_type=self._agent_type,
            sources=sources,
        )

    # ==================== HELPER METHODS ====================

    def _format_result(self, item: dict[str, Any]) -> FormattedWebResult:
        """
        Format a single web search result.
        
        Args:
            item: Raw result from Tavily
            
        Returns:
            FormattedWebResult instance
        """
        url = item.get("url", "")
        domain = self._extract_domain(url)
        content = item.get("content", "")
        
        # Truncate content if needed
        if len(content) > self.agent_config.max_content_length:
            content = content[:self.agent_config.max_content_length] + "..."
        
        return FormattedWebResult(
            title=item.get("title", ""),
            url=url,
            content=content,
            score=item.get("score", 0),
            source_domain=domain,
            published_date=item.get("published_date"),
        )

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.replace("www.", "")
        except Exception:
            return ""

    def format_results_for_context(
        self,
        results: WebResults,
        max_content_length: int | None = None,
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

        max_len = max_content_length or self.agent_config.max_content_length
        
        formatted_parts: list[str] = []
        for i, result in enumerate(results, 1):
            content = result.get("content", "")[:max_len]
            score = result.get("score", 0)
            
            formatted_parts.append(f"""
### Web-Quelle {i}: {result.get('title', 'Unbekannt')}
**URL:** {result.get('url', '')}
**Relevanz:** {score:.2%}

{content}
""")

        return "\n---\n".join(formatted_parts)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"WebSearchAgent("
            f"enabled={self.enabled}, "
            f"cache_size={self._cache.size}, "
            f"tools={len(self._tools)})"
        )
