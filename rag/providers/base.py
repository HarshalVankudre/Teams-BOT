"""
Abstract base class for LLM providers.
Defines the interface that LLM providers must implement.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ChatMessage:
    """Unified message format across providers."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_call_id: Optional[str] = None  # For tool responses
    tool_calls: Optional[List[Dict[str, Any]]] = None  # For assistant messages with tool calls


@dataclass
class ToolCall:
    """Represents a tool/function call from the model."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool_call_id: str
    content: str  # JSON string of the result


@dataclass
class TokenUsage:
    """Token usage statistics from a chat completion."""
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0  # For models with reasoning (o1)
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Allow adding token usage together."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass
class ChatResponse:
    """Unified response format from chat completion."""
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    raw_response: Any = None  # Provider-specific response object
    usage: TokenUsage = field(default_factory=TokenUsage)  # Token usage stats


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Providers implement this interface to work with SingleAgent.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        """Current model being used."""
        pass

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        max_tokens: int = 1200,
    ) -> ChatResponse:
        """
        Send a chat completion request.

        Args:
            messages: List of ChatMessage objects
            tools: Tool definitions in OpenAI function calling format
            tool_choice: "auto", "none", or "required"
            max_tokens: Maximum tokens in response

        Returns:
            ChatResponse with content and/or tool calls
        """
        pass

    @abstractmethod
    def convert_tools(self, openai_tools: List[Dict[str, Any]]) -> Any:
        """
        Convert OpenAI tool format to provider-specific format.

        Args:
            openai_tools: Tools in OpenAI function calling format

        Returns:
            Provider-specific tool definitions
        """
        pass

    def messages_to_native(self, messages: List[ChatMessage]) -> Any:
        """
        Convert ChatMessage list to provider-specific format.
        Default implementation returns messages as-is.
        Override in subclasses for provider-specific conversion.
        """
        return messages
