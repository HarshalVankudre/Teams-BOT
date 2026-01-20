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


def get_provider(provider_name: str = None) -> LLMProvider:
    """
    Factory function to get the OpenAI LLM provider.

    Returns:
        Configured OpenAIProvider instance
    """
    return OpenAIProvider()
