"""
Ollama Client - Drop-in replacement for AsyncOpenAI

Provides an OpenAI-compatible interface for Ollama models.
Supports native tool calling with Qwen 3 models.
"""
import json
import httpx
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class FunctionCall:
    """Represents a function call from the model"""
    name: str
    arguments: str  # JSON string


@dataclass
class ToolCall:
    """Represents a tool call from the model"""
    id: str
    type: str = "function"
    function: FunctionCall = None


@dataclass
class Message:
    """OpenAI-compatible message response"""
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


@dataclass
class Choice:
    """OpenAI-compatible choice"""
    index: int
    message: Message
    finish_reason: str = "stop"


@dataclass
class ChatCompletionResponse:
    """OpenAI-compatible chat completion response"""
    id: str
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[Choice] = field(default_factory=list)


class OllamaChatCompletions:
    """Handles chat completions for Ollama"""

    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def create(
        self,
        model: str = None,
        messages: List[Dict[str, Any]] = None,
        tools: List[Dict[str, Any]] = None,
        tool_choice: Any = None,
        max_tokens: int = None,
        max_completion_tokens: int = None,
        temperature: float = None,
        reasoning: Dict[str, Any] = None,
        **kwargs
    ) -> ChatCompletionResponse:
        """
        Create a chat completion using Ollama API.
        Compatible with OpenAI's chat.completions.create() interface.
        """
        model = model or self.model

        # Build Ollama request
        request_body = {
            "model": model,
            "messages": self._convert_messages(messages or []),
            "stream": False,
        }

        # Add tools if provided (Qwen 3 supports native tool calling)
        if tools:
            request_body["tools"] = tools

        # Handle tool_choice
        if tool_choice:
            if tool_choice == "required":
                # Ollama doesn't have exact equivalent, but tools presence encourages use
                pass
            elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                # Specific function requested - Ollama will handle this
                pass

        # Add options
        options = {}
        if max_tokens:
            options["num_predict"] = max_tokens
        if max_completion_tokens:
            options["num_predict"] = max_completion_tokens
        if temperature is not None:
            options["temperature"] = temperature

        if options:
            request_body["options"] = options

        # Make request to Ollama
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=request_body
            )
            response.raise_for_status()
            data = response.json()

        return self._parse_response(data, model)

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI message format to Ollama format"""
        converted = []
        for msg in messages:
            converted_msg = {
                "role": msg["role"],
                "content": msg.get("content", "")
            }
            # Handle tool results
            if msg.get("tool_call_id"):
                converted_msg["role"] = "tool"
            converted.append(converted_msg)
        return converted

    def _parse_response(self, data: Dict[str, Any], model: str) -> ChatCompletionResponse:
        """Parse Ollama response into OpenAI-compatible format"""
        message_data = data.get("message", {})

        # Extract content (may contain <think> tags for Qwen 3)
        content = message_data.get("content", "")

        # Strip thinking tags if present (Qwen 3 reasoning)
        if "<think>" in content and "</think>" in content:
            # Extract content after thinking
            think_end = content.rfind("</think>")
            if think_end != -1:
                content = content[think_end + 8:].strip()

        # Parse tool calls if present
        tool_calls = None
        ollama_tool_calls = message_data.get("tool_calls", [])

        if ollama_tool_calls:
            tool_calls = []
            for i, tc in enumerate(ollama_tool_calls):
                func = tc.get("function", {})
                tool_calls.append(ToolCall(
                    id=f"call_{i}",
                    type="function",
                    function=FunctionCall(
                        name=func.get("name", ""),
                        arguments=json.dumps(func.get("arguments", {}))
                    )
                ))

        # Build response
        message = Message(
            role="assistant",
            content=content if content else None,
            tool_calls=tool_calls
        )

        choice = Choice(
            index=0,
            message=message,
            finish_reason="tool_calls" if tool_calls else "stop"
        )

        return ChatCompletionResponse(
            id=f"ollama-{hash(str(data))}",
            model=model,
            choices=[choice]
        )


class OllamaChat:
    """Chat namespace for Ollama client"""

    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.completions = OllamaChatCompletions(base_url, model, timeout)


class AsyncOllamaClient:
    """
    Async Ollama client with OpenAI-compatible interface.

    Usage:
        client = AsyncOllamaClient(base_url="http://localhost:11434", model="qwen3:14b")
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[...]
        )
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:14b",
        timeout: float = 120.0,
        **kwargs  # Accept and ignore extra kwargs for compatibility
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.chat = OllamaChat(base_url, model, timeout)

    async def health_check(self) -> bool:
        """Check if Ollama is running and responsive"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> List[str]:
        """List available models in Ollama"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
