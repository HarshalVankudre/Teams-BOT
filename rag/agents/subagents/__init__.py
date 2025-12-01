"""
SubAgents Package

This package contains all specialized subagents that are invoked by the orchestrator.
All agents in this package are automatically discovered and registered.

To add a new agent:
1. Create a new .py file in this folder (e.g., my_agent.py)
2. Inherit from SubAgentBase
3. Define METADATA class attribute with AgentMetadata
4. Use @register_subagent() decorator
5. Implement _execute() method
6. Optionally define tools using @tool decorator

The orchestrator will automatically discover and use your agent!

Example (my_agent.py):
    from .interface import (
        SubAgentBase, tool, register_subagent,
        AgentMetadata, AgentCapability, AgentContext, AgentResponse, AgentType
    )

    @register_subagent()
    class MyCustomAgent(SubAgentBase):
        METADATA = AgentMetadata(
            agent_id="my_custom",
            name="My Custom Agent",
            description="Does something custom",
            detailed_description="Detailed description for orchestrator...",
            capabilities=[AgentCapability.DATA_ANALYSIS],
            uses_reasoning=False,
            parameters={
                "task_description": {
                    "type": "string",
                    "description": "What to do"
                }
            },
            example_queries=["Example query 1", "Example query 2"],
            priority=5
        )

        @tool(
            name="my_tool",
            description="Does something useful",
            parameters={"param1": {"type": "string", "description": "A parameter"}},
            required=["param1"]
        )
        async def my_tool(self, param1: str):
            return {"result": f"Processed: {param1}"}

        async def _execute(self, context: AgentContext) -> AgentResponse:
            task = context.metadata.get("task_description", context.user_query)
            result = await self.my_tool(param1=task)
            return AgentResponse.success_response(
                data=result,
                agent_type=AgentType.SQL_GENERATOR  # or appropriate type
            )
"""
import os
import importlib
import sys
from pathlib import Path

# Export interface components
from .interface import (
    SubAgentBase,
    ToolDefinition,
    tool,
    register_subagent,
    AgentMetadata,
    AgentCapability,
    AgentContext,
    AgentResponse,
    AgentType
)

# Track discovered agents
_discovered_agents = []


def discover_agents():
    """
    Automatically discover and import all agent modules in this package.
    This triggers their registration with the AgentRegistry.
    """
    global _discovered_agents

    if _discovered_agents:
        return _discovered_agents

    package_dir = Path(__file__).parent

    # Find all .py files that aren't __init__ or interface
    agent_modules = [
        f.stem for f in package_dir.glob("*.py")
        if f.stem not in ("__init__", "interface", "_template")
        and not f.stem.startswith("_")
    ]

    for module_name in agent_modules:
        try:
            # Import the module - this triggers @register_subagent decorators
            module = importlib.import_module(f".{module_name}", package=__name__)
            _discovered_agents.append(module_name)
        except Exception as e:
            print(f"[SUBAGENTS] Warning: Failed to import {module_name}: {e}")

    return _discovered_agents


def get_all_subagent_classes():
    """
    Get all registered subagent classes.

    Returns:
        Dict mapping agent_id to agent class
    """
    from ..registry import AgentRegistry

    # Ensure discovery has run
    discover_agents()

    # Get all agents from registry that are subagents
    all_agents = AgentRegistry.get_all_agents()
    subagent_classes = {}

    for agent_id, metadata in all_agents.items():
        # Skip orchestrator and reviewer (they're not subagents)
        if agent_id in ("orchestrator", "reviewer"):
            continue

        agent_class = AgentRegistry.get_agent_class(agent_id)
        if agent_class and issubclass(agent_class, SubAgentBase):
            subagent_classes[agent_id] = agent_class

    return subagent_classes


# Auto-discover on import
discover_agents()


__all__ = [
    # Interface components
    "SubAgentBase",
    "ToolDefinition",
    "tool",
    "register_subagent",
    "AgentMetadata",
    "AgentCapability",
    "AgentContext",
    "AgentResponse",
    "AgentType",
    # Discovery functions
    "discover_agents",
    "get_all_subagent_classes",
]
