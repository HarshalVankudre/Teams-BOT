"""
SubAgent Interface

This module defines the interface that all subagents must implement.
Provides base classes, decorators, and utilities for building subagents.

To create a new subagent:
1. Create a new .py file in this folder
2. Inherit from SubAgentBase
3. Implement required methods
4. Define your tools using the @tool decorator
5. Use @register_subagent decorator to register

The orchestrator will automatically discover and use registered subagents.

Author: RÜKO GmbH Baumaschinen
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Final,
    ParamSpec,
    TypeAlias,
    TypeVar,
)

from ..base import AgentContext, AgentResponse, AgentType, BaseAgent
from ..registry import AgentCapability, AgentMetadata, AgentRegistry

if TYPE_CHECKING:
    from collections.abc import Coroutine

# Type aliases for clarity
ToolHandler: TypeAlias = Callable[..., Awaitable[dict[str, Any]]]
ToolParameters: TypeAlias = dict[str, dict[str, Any]]
OpenAIToolFormat: TypeAlias = dict[str, Any]

# Generic type variables
P = ParamSpec("P")
T = TypeVar("T")

logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    """Raised when a tool execution fails."""
    
    def __init__(self, tool_name: str, message: str, original_error: Exception | None = None):
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class ToolNotFoundError(Exception):
    """Raised when a requested tool is not found."""
    
    def __init__(self, tool_name: str, available_tools: list[str]):
        self.tool_name = tool_name
        self.available_tools = available_tools
        super().__init__(
            f"Unknown tool: '{tool_name}'. Available tools: {', '.join(available_tools)}"
        )


class ToolValidationError(Exception):
    """Raised when tool parameters fail validation."""
    
    def __init__(self, tool_name: str, param_name: str, message: str):
        self.tool_name = tool_name
        self.param_name = param_name
        super().__init__(f"Tool '{tool_name}' validation error for '{param_name}': {message}")


class RetryStrategy(Enum):
    """Retry strategies for tool execution."""
    
    NONE = auto()
    LINEAR = auto()
    EXPONENTIAL = auto()


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Configuration for retry behavior."""
    
    strategy: RetryStrategy = RetryStrategy.NONE
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number."""
        if self.strategy == RetryStrategy.NONE:
            return 0.0
        elif self.strategy == RetryStrategy.LINEAR:
            return min(self.base_delay_seconds * attempt, self.max_delay_seconds)
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            return min(self.base_delay_seconds * (2 ** attempt), self.max_delay_seconds)
        return 0.0


@dataclass(slots=True)
class ToolDefinition:
    """Definition of a tool that a subagent can use."""
    
    name: str
    description: str
    parameters: ToolParameters
    handler: ToolHandler
    required_params: list[str] = field(default_factory=list)
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    timeout_seconds: float | None = None
    tags: list[str] = field(default_factory=list)
    
    # Execution statistics
    _call_count: int = field(default=0, repr=False)
    _total_duration_ms: float = field(default=0.0, repr=False)
    _error_count: int = field(default=0, repr=False)

    def to_openai_tool(self) -> OpenAIToolFormat:
        """Convert to OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required_params,
                },
            },
        }
    
    def record_execution(self, duration_ms: float, success: bool) -> None:
        """Record execution statistics."""
        self._call_count += 1
        self._total_duration_ms += duration_ms
        if not success:
            self._error_count += 1
    
    @property
    def average_duration_ms(self) -> float:
        """Get average execution duration in milliseconds."""
        if self._call_count == 0:
            return 0.0
        return self._total_duration_ms / self._call_count
    
    @property
    def error_rate(self) -> float:
        """Get error rate as a fraction."""
        if self._call_count == 0:
            return 0.0
        return self._error_count / self._call_count
    
    @property
    def stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        return {
            "call_count": self._call_count,
            "total_duration_ms": self._total_duration_ms,
            "average_duration_ms": self.average_duration_ms,
            "error_count": self._error_count,
            "error_rate": self.error_rate,
        }


@dataclass(slots=True)
class ToolResult:
    """Result of a tool execution with metadata."""
    
    success: bool
    data: dict[str, Any]
    tool_name: str
    duration_ms: float
    error: str | None = None
    retries: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "tool_name": self.tool_name,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "retries": self.retries,
        }


def tool(
    name: str,
    description: str,
    parameters: ToolParameters,
    required: list[str] | None = None,
    retry_config: RetryConfig | None = None,
    timeout_seconds: float | None = None,
    tags: list[str] | None = None,
) -> Callable[[ToolHandler], ToolHandler]:
    """
    Decorator to define a tool for a subagent.

    Usage:
        @tool(
            name="search_database",
            description="Search the equipment database",
            parameters={
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results"}
            },
            required=["query"],
            timeout_seconds=30.0,
            tags=["database", "search"]
        )
        async def search_database(self, query: str, limit: int = 10):
            # Implementation
            pass

    Args:
        name: Tool name (used in OpenAI tool calls)
        description: Human-readable description
        parameters: OpenAI-style parameter definitions
        required: List of required parameter names
        retry_config: Optional retry configuration
        timeout_seconds: Optional execution timeout
        tags: Optional tags for categorization

    Returns:
        Decorated function with tool metadata attached.
    """
    def decorator(func: ToolHandler) -> ToolHandler:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return await func(*args, **kwargs)

        # Attach tool metadata to the function
        wrapper._tool_definition = ToolDefinition(  # type: ignore[attr-defined]
            name=name,
            description=description,
            parameters=parameters,
            handler=func,
            required_params=required or [],
            retry_config=retry_config or RetryConfig(),
            timeout_seconds=timeout_seconds,
            tags=tags or [],
        )
        wrapper._is_tool = True  # type: ignore[attr-defined]
        return wrapper

    return decorator


class SubAgentBase(BaseAgent, ABC):
    """
    Base class for all subagents.

    Subagents are specialized agents that handle specific types of queries
    (SQL, semantic search, web search, etc.). They are invoked by the
    orchestrator based on query analysis.

    Features:
        - Automatic tool discovery via @tool decorator
        - Built-in retry logic for tool execution
        - Execution statistics tracking
        - Timeout support for long-running tools
        - Batch tool execution support

    To create a new subagent:
        1. Inherit from this class
        2. Define METADATA class attribute with AgentMetadata
        3. Implement _execute() method
        4. Optionally define tools using @tool decorator
        5. Use @register_subagent() decorator on the class

    Example:
        @register_subagent()
        class MyCustomAgent(SubAgentBase):
            METADATA = AgentMetadata(
                agent_id="my_custom",
                name="My Custom Agent",
                ...
            )

            @tool(name="my_tool", description="...", parameters={...})
            async def my_tool(self, param1: str):
                return {"result": "..."}

            async def _execute(self, context: AgentContext) -> AgentResponse:
                result = await self.call_tool("my_tool", param1="test")
                return AgentResponse.success_response(data=result, ...)

    Attributes:
        METADATA: Agent metadata (must be defined by subclasses)
    """

    # Subclasses must define this
    METADATA: AgentMetadata | None = None
    
    # Class-level constants
    _DEFAULT_TOOL_TIMEOUT: Final[float] = 60.0

    __slots__ = ("_tools", "_execution_lock")

    def __init__(self, verbose: bool = False, **kwargs: Any) -> None:
        """
        Initialize the subagent.

        Args:
            verbose: Enable verbose logging
            **kwargs: Additional arguments passed to BaseAgent
        """
        super().__init__(verbose=verbose)
        self._tools: dict[str, ToolDefinition] = {}
        self._execution_lock = asyncio.Lock()
        self._discover_tools()

    def _discover_tools(self) -> None:
        """Discover all tools defined on this agent using @tool decorator."""
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
                
            attr = getattr(self, attr_name, None)
            if attr is None:
                continue
                
            if not (hasattr(attr, "_is_tool") and attr._is_tool):
                continue
                
            tool_def: ToolDefinition = attr._tool_definition
            # Bind the handler to self
            tool_def.handler = attr
            self._tools[tool_def.name] = tool_def
            
            logger.debug("Discovered tool: %s", tool_def.name)
            self.log(f"Discovered tool: {tool_def.name}")

    def get_tools(self) -> list[OpenAIToolFormat]:
        """Get all tools in OpenAI format."""
        return [tool_def.to_openai_tool() for tool_def in self._tools.values()]

    def get_tool_names(self) -> list[str]:
        """Get names of all available tools."""
        return list(self._tools.keys())
    
    def get_tools_by_tag(self, tag: str) -> list[ToolDefinition]:
        """Get all tools with a specific tag."""
        return [t for t in self._tools.values() if tag in t.tags]

    def get_tool_stats(self) -> dict[str, dict[str, Any]]:
        """Get execution statistics for all tools."""
        return {name: tool_def.stats for name, tool_def in self._tools.items()}

    async def call_tool(
        self,
        tool_name: str,
        *,
        timeout_override: float | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Call a tool by name with the given parameters.

        Args:
            tool_name: Name of the tool to call
            timeout_override: Override the tool's default timeout
            **kwargs: Parameters to pass to the tool

        Returns:
            ToolResult with execution details and data

        Raises:
            ToolNotFoundError: If the tool doesn't exist
            ToolExecutionError: If execution fails after all retries
        """
        if tool_name not in self._tools:
            raise ToolNotFoundError(tool_name, self.get_tool_names())

        tool_def = self._tools[tool_name]
        retry_config = tool_def.retry_config
        timeout = timeout_override or tool_def.timeout_seconds or self._DEFAULT_TOOL_TIMEOUT
        
        logger.info("Calling tool: %s with params: %s", tool_name, kwargs)
        self.log(f"Calling tool: {tool_name} with {kwargs}")

        last_error: Exception | None = None
        retries = 0
        
        for attempt in range(retry_config.max_attempts):
            start_time = time.perf_counter()
            
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    tool_def.handler(**kwargs),
                    timeout=timeout,
                )
                
                duration_ms = (time.perf_counter() - start_time) * 1000
                tool_def.record_execution(duration_ms, success=True)
                
                logger.debug(
                    "Tool %s completed in %.2fms",
                    tool_name,
                    duration_ms,
                )
                
                return ToolResult(
                    success=True,
                    data=result,
                    tool_name=tool_name,
                    duration_ms=duration_ms,
                    retries=retries,
                )
                
            except asyncio.TimeoutError as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                tool_def.record_execution(duration_ms, success=False)
                last_error = e
                
                logger.warning(
                    "Tool %s timed out after %.2fs (attempt %d/%d)",
                    tool_name,
                    timeout,
                    attempt + 1,
                    retry_config.max_attempts,
                )
                
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                tool_def.record_execution(duration_ms, success=False)
                last_error = e
                
                logger.warning(
                    "Tool %s error: %s (attempt %d/%d)",
                    tool_name,
                    str(e),
                    attempt + 1,
                    retry_config.max_attempts,
                )
            
            retries += 1
            
            # Wait before retry (if not last attempt)
            if attempt < retry_config.max_attempts - 1:
                delay = retry_config.get_delay(attempt)
                if delay > 0:
                    await asyncio.sleep(delay)

        # All retries exhausted
        error_msg = str(last_error) if last_error else "Unknown error"
        
        return ToolResult(
            success=False,
            data={},
            tool_name=tool_name,
            duration_ms=0.0,
            error=error_msg,
            retries=retries,
        )

    async def call_tools_parallel(
        self,
        tool_calls: list[tuple[str, dict[str, Any]]],
    ) -> list[ToolResult]:
        """
        Execute multiple tools in parallel.

        Args:
            tool_calls: List of (tool_name, kwargs) tuples

        Returns:
            List of ToolResults in the same order as input
        """
        tasks = [
            self.call_tool(name, **kwargs)
            for name, kwargs in tool_calls
        ]
        return await asyncio.gather(*tasks)

    async def call_tools_sequential(
        self,
        tool_calls: list[tuple[str, dict[str, Any]]],
        stop_on_error: bool = False,
    ) -> list[ToolResult]:
        """
        Execute multiple tools sequentially.

        Args:
            tool_calls: List of (tool_name, kwargs) tuples
            stop_on_error: If True, stop execution on first error

        Returns:
            List of ToolResults (may be shorter than input if stop_on_error=True)
        """
        results: list[ToolResult] = []
        
        for name, kwargs in tool_calls:
            result = await self.call_tool(name, **kwargs)
            results.append(result)
            
            if stop_on_error and not result.success:
                break
        
        return results

    @classmethod
    def get_metadata(cls) -> AgentMetadata:
        """Get the agent metadata."""
        if cls.METADATA is None:
            raise NotImplementedError(
                f"{cls.__name__} must define METADATA class attribute"
            )
        return cls.METADATA

    @abstractmethod
    async def _execute(self, context: AgentContext) -> AgentResponse:
        """
        Execute the agent's main logic.

        This method must be implemented by all subagents.

        Args:
            context: The agent context with query, history, and metadata

        Returns:
            AgentResponse with the results
        """
        pass

    def __repr__(self) -> str:
        """Return string representation."""
        class_name = self.__class__.__name__
        tool_count = len(self._tools)
        metadata_id = self.METADATA.agent_id if self.METADATA else "unknown"
        return f"{class_name}(agent_id={metadata_id!r}, tools={tool_count})"


def register_subagent(
    metadata: AgentMetadata | None = None,
) -> Callable[[type[SubAgentBase]], type[SubAgentBase]]:
    """
    Decorator to register a subagent with the registry.

    Can be used in two ways:
        1. With metadata parameter: @register_subagent(metadata)
        2. Without parameter (uses class METADATA): @register_subagent()

    Example:
        @register_subagent()
        class MyAgent(SubAgentBase):
            METADATA = AgentMetadata(...)
            ...

    Args:
        metadata: Optional AgentMetadata (uses class METADATA if not provided)

    Returns:
        Decorated class registered with AgentRegistry

    Raises:
        ValueError: If no metadata is provided or found on class
    """
    def decorator(cls: type[SubAgentBase]) -> type[SubAgentBase]:
        # Use provided metadata or get from class
        agent_metadata = metadata or cls.METADATA

        if agent_metadata is None:
            raise ValueError(
                f"{cls.__name__} must either pass metadata to @register_subagent() "
                "or define METADATA class attribute"
            )

        # Store metadata on class for easy access
        cls.METADATA = agent_metadata

        # Register with the global registry
        AgentRegistry.register(agent_metadata, cls)
        
        logger.info(
            "Registered subagent: %s (%s)",
            agent_metadata.name,
            agent_metadata.agent_id,
        )

        return cls

    # Handle @register_subagent() without arguments
    if metadata is None:
        return decorator

    # Handle @register_subagent without parentheses (metadata is actually the class)
    if isinstance(metadata, type):
        cls = metadata
        return decorator(cls)  # type: ignore[arg-type]

    return decorator


# Export commonly used items for convenience
__all__ = [
    # Base classes
    "SubAgentBase",
    "ToolDefinition",
    "ToolResult",
    # Decorators
    "tool",
    "register_subagent",
    # Configuration
    "RetryConfig",
    "RetryStrategy",
    # Exceptions
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolValidationError",
    # Re-exports from registry/base
    "AgentMetadata",
    "AgentCapability",
    "AgentContext",
    "AgentResponse",
    "AgentType",
]
