"""
SubAgent Interface

This module defines the interface that all subagents must implement.
To create a new subagent:
1. Create a new .py file in this folder
2. Inherit from SubAgentBase
3. Implement required methods
4. Define your tools using the @tool decorator
5. Use @register_subagent decorator to register

The orchestrator will automatically discover and use registered subagents.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Type
from dataclasses import dataclass, field
from functools import wraps

from ..base import BaseAgent, AgentContext, AgentResponse, AgentType
from ..registry import AgentMetadata, AgentCapability, AgentRegistry


@dataclass
class ToolDefinition:
    """Definition of a tool that a subagent can use"""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable
    required_params: List[str] = field(default_factory=list)

    def to_openai_tool(self) -> Dict[str, Any]:
        """Convert to OpenAI tool format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required_params
                }
            }
        }


def tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    required: List[str] = None
):
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
            required=["query"]
        )
        async def search_database(self, query: str, limit: int = 10):
            # Implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        # Attach tool metadata to the function
        wrapper._tool_definition = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=func,
            required_params=required or []
        )
        wrapper._is_tool = True
        return wrapper
    return decorator


class SubAgentBase(BaseAgent, ABC):
    """
    Base class for all subagents.

    Subagents are specialized agents that handle specific types of queries
    (SQL, semantic search, web search, etc.). They are invoked by the
    orchestrator based on query analysis.

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
                # Use self.call_tool("my_tool", param1="value")
                result = await self.call_tool("my_tool", param1="test")
                return AgentResponse.success_response(data=result, ...)
    """

    # Subclasses must define this
    METADATA: AgentMetadata = None

    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__(verbose=verbose)
        self._tools: Dict[str, ToolDefinition] = {}
        self._discover_tools()

    def _discover_tools(self):
        """Discover all tools defined on this agent using @tool decorator"""
        for attr_name in dir(self):
            if attr_name.startswith('_'):
                continue
            attr = getattr(self, attr_name, None)
            if attr and hasattr(attr, '_is_tool') and attr._is_tool:
                tool_def = attr._tool_definition
                # Bind the handler to self
                tool_def.handler = attr
                self._tools[tool_def.name] = tool_def
                self.log(f"Discovered tool: {tool_def.name}")

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get all tools in OpenAI format"""
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def get_tool_names(self) -> List[str]:
        """Get names of all available tools"""
        return list(self._tools.keys())

    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Call a tool by name with the given parameters.

        Args:
            tool_name: Name of the tool to call
            **kwargs: Parameters to pass to the tool

        Returns:
            Result from the tool handler
        """
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}. Available: {self.get_tool_names()}")

        tool = self._tools[tool_name]
        self.log(f"Calling tool: {tool_name} with {kwargs}")

        try:
            result = await tool.handler(**kwargs)
            return result
        except Exception as e:
            self.log(f"Tool {tool_name} error: {str(e)}")
            raise

    @classmethod
    def get_metadata(cls) -> AgentMetadata:
        """Get the agent metadata"""
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


def register_subagent(metadata: AgentMetadata = None):
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
    """
    def decorator(cls: Type[SubAgentBase]) -> Type[SubAgentBase]:
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

        return cls

    # Handle @register_subagent() without arguments
    if metadata is None:
        return decorator

    # Handle @register_subagent(metadata) with arguments
    if isinstance(metadata, type):
        # Called as @register_subagent without parentheses
        cls = metadata
        return decorator(cls)

    return decorator


# Export commonly used items for convenience
__all__ = [
    'SubAgentBase',
    'ToolDefinition',
    'tool',
    'register_subagent',
    'AgentMetadata',
    'AgentCapability',
    'AgentContext',
    'AgentResponse',
    'AgentType'
]
