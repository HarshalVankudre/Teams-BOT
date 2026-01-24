"""
Cerebras LLM Provider implementation.
Uses OpenAI-compatible API with Cerebras endpoint.
"""
import json
import asyncio
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI

from .base import LLMProvider, ChatMessage, ChatResponse, ToolCall, TokenUsage
from ..config import config

# Retry settings for rate limiting
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # seconds


class CerebrasProvider(LLMProvider):
    """Cerebras provider using OpenAI-compatible API."""

    # Cerebras API endpoint
    CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

    def __init__(
        self,
        model_override: str = None,
        reasoning_override: str = None,
        temperature_override: float = None,
        max_tokens_override: int = None
    ):
        self._client = AsyncOpenAI(
            api_key=config.cerebras_api_key,
            base_url=self.CEREBRAS_BASE_URL
        )
        self._model = model_override or config.cerebras_model
        self._reasoning = reasoning_override or config.cerebras_reasoning
        self._temperature = temperature_override
        self._max_tokens = max_tokens_override
        print(f"[CerebrasProvider] Initialized with model: {self._model}, reasoning: {self._reasoning}")

    @property
    def name(self) -> str:
        return "cerebras"

    @property
    def model(self) -> str:
        return self._model

    async def chat_completion(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        max_tokens: int = 1200,
    ) -> ChatResponse:
        """Send chat completion request to Cerebras."""

        # Convert ChatMessage to OpenAI format
        openai_messages = self._convert_messages(messages)

        # Build request params (Cerebras uses max_completion_tokens)
        params = {
            "model": self._model,
            "messages": openai_messages,
            "max_completion_tokens": self._max_tokens or max_tokens,
        }

        # Apply reasoning_effort for gpt-oss-120b model (low, medium, high)
        if self._reasoning and "gpt-oss" in self._model.lower():
            params["reasoning_effort"] = self._reasoning.lower()

        # Apply temperature if configured (range: 0 to 1.5)
        # Note: Don't use both temperature and top_p together
        if self._temperature is not None:
            params["temperature"] = min(max(self._temperature, 0), 1.5)

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        # Make request with retry for rate limiting
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = await self._client.chat.completions.create(**params)
                break
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                # Retry on rate limiting (503) or queue exceeded errors
                if "503" in error_str or "queue" in error_str or "too_many_requests" in error_str:
                    wait_time = RETRY_DELAY_BASE * (2 ** attempt)
                    print(f"[CerebrasProvider] Rate limited, retrying in {wait_time}s (attempt {attempt + 1}/{MAX_RETRIES})")
                    await asyncio.sleep(wait_time)
                else:
                    raise  # Re-raise non-retryable errors
        else:
            # All retries failed
            raise last_error

        message = response.choices[0].message

        # Parse tool calls if present
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}

                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args
                ))

        # Extract token usage
        usage = TokenUsage()
        if response.usage:
            usage.input_tokens = response.usage.prompt_tokens or 0
            usage.output_tokens = response.usage.completion_tokens or 0
            usage.total_tokens = response.usage.total_tokens or 0

        return ChatResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason or "stop",
            raw_response=response,
            usage=usage
        )

    def convert_tools(self, openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cerebras uses OpenAI-compatible tool format."""
        return openai_tools

    def _convert_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert ChatMessage list to OpenAI format."""
        openai_messages = []

        for msg in messages:
            if msg.role == "tool":
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content
                })
            elif msg.role == "assistant" and msg.tool_calls:
                openai_messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"]
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                })
            else:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        return openai_messages
