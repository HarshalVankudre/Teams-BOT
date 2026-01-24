"""
LLM Providers for the RAG pipeline.
Supports OpenAI and Cerebras providers with unified interface.
"""
from .base import LLMProvider, ChatMessage, ToolCall, ToolResult, ChatResponse, TokenUsage
from .openai_provider import OpenAIProvider
from .cerebras_provider import CerebrasProvider
from ..config import config

__all__ = [
    "LLMProvider",
    "ChatMessage",
    "ChatResponse",
    "ToolCall",
    "ToolResult",
    "TokenUsage",
    "OpenAIProvider",
    "CerebrasProvider",
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
    Factory function to get the configured LLM provider.

    Args:
        provider_name: Provider to use ("openai" or "cerebras"). If None, uses LLM_PROVIDER env var.
        model_override: Optional model to use instead of config default
        reasoning_override: Optional reasoning effort to use instead of config default
        temperature_override: Optional temperature override
        max_tokens_override: Optional max tokens override

    Returns:
        Configured LLMProvider instance (OpenAIProvider or CerebrasProvider)
    """
    # Determine which provider to use
    selected_provider = provider_name or config.llm_provider

    if selected_provider == "cerebras":
        return CerebrasProvider(
            model_override=model_override,
            reasoning_override=reasoning_override,
            temperature_override=temperature_override,
            max_tokens_override=max_tokens_override
        )
    else:
        # Default to OpenAI
        return OpenAIProvider(
            model_override=model_override,
            reasoning_override=reasoning_override,
            temperature_override=temperature_override,
            max_tokens_override=max_tokens_override
        )
