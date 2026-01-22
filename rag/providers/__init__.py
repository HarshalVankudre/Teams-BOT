"""
LLM Providers for the RAG pipeline.
OpenAI provider with unified interface.
"""
from .base import LLMProvider, ChatMessage, ToolCall, ToolResult, ChatResponse, TokenUsage
from .openai_provider import OpenAIProvider

__all__ = [
    "LLMProvider",
    "ChatMessage",
    "ChatResponse",
    "ToolCall",
    "ToolResult",
    "TokenUsage",
    "OpenAIProvider",
    "get_provider",
]


def get_provider(
    provider_name: str = None,
    model_override: str = None,
    reasoning_override: str = None,
    temperature_override: float = None,
    max_tokens_override: int = None
) -> LLMProvider:
    """
    Factory function to get the OpenAI LLM provider.

    Args:
        provider_name: Unused, kept for compatibility
        model_override: Optional model to use instead of config default
        reasoning_override: Optional reasoning effort to use instead of config default
        temperature_override: Optional temperature override
        max_tokens_override: Optional max tokens override

    Returns:
        Configured OpenAIProvider instance
    """
    return OpenAIProvider(
        model_override=model_override,
        reasoning_override=reasoning_override,
        temperature_override=temperature_override,
        max_tokens_override=max_tokens_override
    )
