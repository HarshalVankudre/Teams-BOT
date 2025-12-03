"""
Base Agent Class
Provides common functionality for all agents in the system.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from enum import Enum
import time

if TYPE_CHECKING:
    from .registry import AgentMetadata


class AgentType(Enum):
    """Types of agents in the system - kept for backward compatibility"""
    ORCHESTRATOR = "orchestrator"
    SQL_GENERATOR = "sql"
    PINECONE_SEARCH = "pinecone"
    WEB_SEARCH = "web_search"
    REVIEWER = "reviewer"


@dataclass
class AgentContext:
    """
    Context passed between agents.
    Contains conversation history, user info, and accumulated data.
    """
    user_query: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    thread_key: Optional[str] = None

    # Data accumulated from sub-agents
    sql_results: Optional[List[Dict[str, Any]]] = None
    pinecone_results: Optional[List[Dict[str, Any]]] = None
    web_results: Optional[List[Dict[str, Any]]] = None

    # Orchestrator analysis
    query_intent: Optional[str] = None
    required_tools: List[str] = field(default_factory=list)
    reasoning: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to context"""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata from context"""
        return self.metadata.get(key, default)


@dataclass
class AgentResponse:
    """
    Response from an agent.
    Contains the result data and metadata about the operation.
    """
    success: bool
    data: Any = None
    error: Optional[str] = None
    agent_type: Optional[AgentType] = None
    execution_time_ms: int = 0
    reasoning: Optional[str] = None

    # For sub-agents to indicate what was done
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def success_response(
        cls,
        data: Any,
        agent_type: AgentType,
        execution_time_ms: int = 0,
        reasoning: str = None,
        tool_calls: List[Dict] = None,
        sources: List[Dict] = None
    ) -> "AgentResponse":
        """Create a successful response"""
        return cls(
            success=True,
            data=data,
            agent_type=agent_type,
            execution_time_ms=execution_time_ms,
            reasoning=reasoning,
            tool_calls=tool_calls or [],
            sources=sources or []
        )

    @classmethod
    def error_response(
        cls,
        error: str,
        agent_type: AgentType,
        execution_time_ms: int = 0
    ) -> "AgentResponse":
        """Create an error response"""
        return cls(
            success=False,
            error=error,
            agent_type=agent_type,
            execution_time_ms=execution_time_ms
        )


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    Provides common functionality like timing, logging, and error handling.
    """

    # Class-level metadata - set by @register_agent decorator
    _metadata: Optional['AgentMetadata'] = None

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._agent_type: AgentType = AgentType.ORCHESTRATOR  # Override in subclass

    @property
    def agent_type(self) -> AgentType:
        return self._agent_type

    @property
    def agent_id(self) -> str:
        """Get agent ID from metadata or fall back to agent_type"""
        if self._metadata:
            return self._metadata.agent_id
        return self._agent_type.value

    @classmethod
    def get_metadata(cls) -> Optional['AgentMetadata']:
        """Get the metadata for this agent class"""
        return cls._metadata

    def log(self, message: str) -> None:
        """Log message if verbose mode is enabled"""
        if self.verbose:
            try:
                print(f"[{self.agent_id.upper()}] {message}")
            except UnicodeEncodeError:
                # Handle Windows encoding issues with German characters
                safe_msg = message.encode('ascii', errors='replace').decode('ascii')
                print(f"[{self.agent_id.upper()}] {safe_msg}")

    async def execute(self, context: AgentContext) -> AgentResponse:
        """
        Execute the agent with timing and error handling.
        Subclasses should override _execute() instead.
        """
        start_time = time.time()

        try:
            self.log(f"Processing: {context.user_query[:50]}...")
            response = await self._execute(context)
            response.execution_time_ms = int((time.time() - start_time) * 1000)
            self.log(f"Completed in {response.execution_time_ms}ms")
            return response

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            self.log(f"Error: {str(e)}")
            return AgentResponse.error_response(
                error=str(e),
                agent_type=self._agent_type,
                execution_time_ms=execution_time_ms
            )

    @abstractmethod
    async def _execute(self, context: AgentContext) -> AgentResponse:
        """
        Execute the agent's main logic.
        Must be implemented by subclasses.
        """
        pass


class ToolDefinition:
    """Helper class for defining tools that agents can use"""

    @staticmethod
    def create(
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required: List[str] = None
    ) -> Dict[str, Any]:
        """Create a tool definition in OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required or []
                }
            }
        }
