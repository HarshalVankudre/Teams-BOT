"""
Multi-Agent System for RUKO Teams Bot

Architecture:
- OrchestratorAgent: Reasoning LLM that analyzes queries and delegates to specialized agents
- SubAgents: Specialized agents in the subagents/ folder (SQL, Pinecone, WebSearch, etc.)
- ReviewerAgent: Reviews data from sub-agents and generates natural language responses

Flow:
1. User query -> OrchestratorAgent (analyzes intent, context from Redis)
2. OrchestratorAgent -> Invokes one or more sub-agents
3. Sub-agents -> Return structured data
4. ReviewerAgent -> Formats final response with smart display logic

Agent Registry:
- Agents self-register with metadata describing their capabilities
- The orchestrator dynamically discovers available agents
- Easy to add new agents by creating a new file in subagents/ folder

To add a new subagent:
1. Create a new .py file in rag/agents/subagents/
2. Inherit from SubAgentBase
3. Define METADATA class attribute
4. Use @register_subagent() decorator
5. Implement _execute() method
6. Optionally define tools using @tool decorator

See rag/agents/subagents/_template.py for an example!
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

# Import top-level agents
from .orchestrator import OrchestratorAgent
from .reviewer_agent import ReviewerAgent

# Import subagents package - triggers auto-discovery
from .subagents import (
    SubAgentBase,
    ToolDefinition,
    tool,
    register_subagent,
    discover_agents,
    get_all_subagent_classes,
)

# Ensure subagents are discovered and registered
discover_agents()

# Import specific subagent classes for backward compatibility
from .subagents.sql_agent import SQLGeneratorAgent
from .subagents.pinecone_agent import PineconeSearchAgent
from .subagents.web_search_agent import WebSearchAgent

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

    # SubAgent interface
    "SubAgentBase",
    "ToolDefinition",
    "tool",
    "register_subagent",
    "discover_agents",
    "get_all_subagent_classes",

    # Top-level agents
    "OrchestratorAgent",
    "ReviewerAgent",

    # SubAgents (backward compatibility)
    "SQLGeneratorAgent",
    "PineconeSearchAgent",
    "WebSearchAgent",

    # Agent system
    "AgentSystem",
    "AgentSystemConfig",
    "AgentSystemResult",
    "create_agent_system",
]
