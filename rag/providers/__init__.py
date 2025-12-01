"""
LLM Provider Factory

Provides a unified interface for creating LLM clients.
Supports OpenAI and Ollama providers.
"""
from typing import Union, Optional
from openai import AsyncOpenAI

from .ollama_client import AsyncOllamaClient


def create_llm_client(
    provider: str = "openai",
    api_key: str = None,
    base_url: str = None,
    model: str = None,
    **kwargs
) -> Union[AsyncOpenAI, AsyncOllamaClient]:
    """
    Create an LLM client based on the provider type.

    Args:
        provider: "openai" or "ollama"
        api_key: API key (required for OpenAI)
        base_url: Base URL (required for Ollama, optional for OpenAI)
        model: Model name (used by Ollama client)
        **kwargs: Additional arguments passed to the client

    Returns:
        AsyncOpenAI or AsyncOllamaClient instance
    """
    provider = provider.lower().strip()

    if provider == "openai":
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        return AsyncOpenAI(**client_kwargs, **kwargs)

    elif provider == "ollama":
        return AsyncOllamaClient(
            base_url=base_url or "http://localhost:11434",
            model=model or "qwen3:14b",
            **kwargs
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'openai' or 'ollama'.")


__all__ = ["create_llm_client", "AsyncOllamaClient"]
