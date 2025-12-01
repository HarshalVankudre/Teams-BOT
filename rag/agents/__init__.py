"""
Multi-Agent System for RUKO Teams Bot

Architecture:
- OrchestratorAgent: Reasoning LLM that analyzes queries and delegates to specialized agents
- SQLGeneratorAgent: Generates and executes SQL queries against PostgreSQL
- PineconeSearchAgent: Performs semantic search in Pinecone vector store
- WebSearchAgent: Retrieves supplementary information from the web via Tavily
- ReviewerAgent: Reviews data from sub-agents and generates natural language responses

Flow:
1. User query -> OrchestratorAgent (analyzes intent, context from Redis)
2. OrchestratorAgent -> Invokes one or more sub-agents
3. Sub-agents -> Return structured data
4. ReviewerAgent -> Formats final response with smart display logic

Agent Registry:
- Agents self-register with metadata describing their capabilities
- The orchestrator dynamically discovers available agents
- Easy to add new agents by creating a new file with @register_agent decorator
"""

# Registry system for dynamic agent discovery
from .registry import (
    AgentRegistry,
    AgentMetadata,
    AgentCapability,
    register_agent
)

# Base classes
from .base import BaseAgent, AgentResponse, AgentContext, AgentType

# Import agents to trigger their registration
from .orchestrator import OrchestratorAgent
from .sql_agent import SQLGeneratorAgent
from .pinecone_agent import PineconeSearchAgent
from .web_search_agent import WebSearchAgent
from .reviewer_agent import ReviewerAgent

# Agent system coordinator
from .agent_system import AgentSystem, AgentSystemConfig, AgentSystemResult, create_agent_system

__all__ = [
    # Registry
    "AgentRegistry",
    "AgentMetadata",
    "AgentCapability",
    "register_agent",

    # Base classes
    "BaseAgent",
    "AgentResponse",
    "AgentContext",
    "AgentType",

    # Agents
    "OrchestratorAgent",
    "SQLGeneratorAgent",
    "PineconeSearchAgent",
    "WebSearchAgent",
    "ReviewerAgent",

    # Agent system
    "AgentSystem",
    "AgentSystemConfig",
    "AgentSystemResult",
    "create_agent_system",
]
