"""LangGraph ReAct agent for equipment retrieval queries."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from rag.config import config
from rag.langgraph_tools import (
    get_langgraph_tools,
    set_shared_pinecone,
    set_shared_postgres,
)
from rag.prompts import LANGGRAPH_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from LangGraph agent processing."""

    response: str
    tools_used: List[str]
    execution_time_ms: int
    token_usage: Optional[dict] = None
    sources: Optional[List[str]] = None


class LangGraphAgent:
    """LangGraph ReAct agent for equipment queries."""

    def __init__(self, redis_url: Optional[str] = None):
        """Initialize the LangGraph agent."""
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.prebuilt import create_react_agent

        _ = redis_url
        self.llm = self._build_llm()
        self.tools = get_langgraph_tools()
        self.checkpointer = MemorySaver()
        self.graph = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=LANGGRAPH_SYSTEM_PROMPT,
            checkpointer=self.checkpointer,
        )
        logger.info("LangGraph agent initialized with %s tools", len(self.tools))

    def _build_llm(self):
        from langchain_google_genai import ChatGoogleGenerativeAI

        logger.info("LangGraph using Gemini model: %s", config.langgraph_model)
        return ChatGoogleGenerativeAI(
            model=config.langgraph_model,
            google_api_key=config.google_api_key,
            temperature=0,
        )

    async def process(
        self,
        user_query: str,
        thread_key: str,
        conversation_history: Optional[List[dict]] = None,
    ) -> AgentResult:
        """Process a user query through the ReAct agent."""
        _ = conversation_history
        start_time = time.time()
        run_config = {"configurable": {"thread_id": thread_key or "default"}}

        try:
            result = await self._ainvoke_with_retry(
                payload={"messages": [("user", user_query)]},
                run_config=run_config,
            )
            final_message = result["messages"][-1]
            response = self._extract_response_text(final_message)
            return AgentResult(
                response=response,
                tools_used=self._extract_tools_used(result["messages"]),
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
        except Exception as exc:
            logger.error("LangGraph agent error: %s", exc, exc_info=True)
            raise

    async def _ainvoke_with_retry(self, payload: dict, run_config: dict) -> dict:
        try:
            return await self.graph.ainvoke(payload, config=run_config)
        except Exception as exc:
            if "rate_limit" not in str(exc).lower() and "429" not in str(exc):
                raise
            logger.warning("LangGraph rate limited, retrying once after 5 seconds")
            await asyncio.sleep(5)
            return await self.graph.ainvoke(payload, config=run_config)

    @staticmethod
    def _extract_tools_used(messages: List[object]) -> List[str]:
        tools_used: List[str] = []
        for message in messages:
            tool_calls = getattr(message, "tool_calls", None) or []
            for tool_call in tool_calls:
                name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
                if name and name not in tools_used:
                    tools_used.append(name)
        return tools_used

    @staticmethod
    def _extract_response_text(message: object) -> str:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                else:
                    text = getattr(item, "text", None)
                    if text:
                        parts.append(str(text))
            joined = "\n".join(part.strip() for part in parts if part and str(part).strip())
            if joined:
                return joined
        return str(message)


_agent_instance: Optional[LangGraphAgent] = None


def get_langgraph_agent() -> LangGraphAgent:
    """Get or create the singleton LangGraph agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = LangGraphAgent(redis_url=config.redis_url)
    return _agent_instance


__all__ = [
    "AgentResult",
    "LangGraphAgent",
    "get_langgraph_agent",
    "set_shared_pinecone",
    "set_shared_postgres",
]
