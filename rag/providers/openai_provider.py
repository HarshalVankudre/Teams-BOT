"""
OpenAI LLM Provider implementation.
Wraps the AsyncOpenAI client with the unified LLMProvider interface.
"""
import json
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI

from .base import LLMProvider, ChatMessage, ChatResponse, ToolCall, TokenUsage
from ..config import config


class OpenAIProvider(LLMProvider):
    """OpenAI provider using AsyncOpenAI client."""

    def __init__(self):
        self._client = AsyncOpenAI(api_key=config.openai_api_key)
        self._model = config.openai_model
        print(f"[OpenAIProvider] Initialized with model: {self._model}")

    @property
    def name(self) -> str:
        return "openai"

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
        """Send chat completion request to OpenAI."""

        # Convert ChatMessage to OpenAI format
        openai_messages = self._convert_messages(messages)

        # Build request params
        params = {
            "model": self._model,
            "messages": openai_messages,
            "max_completion_tokens": max_tokens,
        }

        # Apply reasoning effort if configured (for o1/o3/gpt-5 models)
        if config.openai_reasoning and config.openai_reasoning.lower() != "none":
            params["reasoning_effort"] = config.openai_reasoning.lower()

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        # Make request
        response = await self._client.chat.completions.create(**params)
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
            # Check for reasoning tokens (o1/o3 models)
            if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
                usage.reasoning_tokens = getattr(response.usage.completion_tokens_details, 'reasoning_tokens', 0) or 0

        return ChatResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason or "stop",
            raw_response=response,
            usage=usage
        )

    def convert_tools(self, openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """OpenAI tools are already in the correct format."""
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
