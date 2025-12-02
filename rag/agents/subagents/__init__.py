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
                agent_type=AgentType.SQL_GENERATOR
            )

Author: RÜKO GmbH Baumaschinen
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypeAlias

if TYPE_CHECKING:
    from ..registry import AgentMetadata

# Export interface components
from .interface import (
    AgentCapability,
    AgentContext,
    AgentMetadata,
    AgentResponse,
    AgentType,
    RetryConfig,
    RetryStrategy,
    SubAgentBase,
    ToolDefinition,
    ToolExecutionError,
    ToolNotFoundError,
    ToolResult,
    ToolValidationError,
    register_subagent,
    tool,
)

logger = logging.getLogger(__name__)

# Type aliases
AgentClassMap: TypeAlias = dict[str, type["SubAgentBase"]]

# Files to exclude from discovery
_EXCLUDED_MODULES: Final[frozenset[str]] = frozenset({
    "__init__",
    "interface",
    "_template",
})


@dataclass(slots=True)
class DiscoveryResult:
    """Result of agent discovery process."""
    
    discovered_modules: list[str] = field(default_factory=list)
    failed_modules: dict[str, str] = field(default_factory=dict)
    
    @property
    def success_count(self) -> int:
        """Number of successfully discovered modules."""
        return len(self.discovered_modules)
    
    @property
    def failure_count(self) -> int:
        """Number of failed module imports."""
        return len(self.failed_modules)
    
    @property
    def total_count(self) -> int:
        """Total number of modules attempted."""
        return self.success_count + self.failure_count
    
    def __bool__(self) -> bool:
        """True if any modules were discovered."""
        return self.success_count > 0


class AgentDiscoveryError(Exception):
    """Raised when agent discovery fails critically."""
    
    def __init__(self, message: str, failed_modules: dict[str, str]):
        self.failed_modules = failed_modules
        super().__init__(message)


class AgentDiscovery:
    """
    Handles automatic discovery and registration of subagent modules.
    
    This class scans the subagents directory for Python modules and imports them,
    which triggers their @register_subagent decorators to register with the
    AgentRegistry.
    
    Attributes:
        package_dir: Path to the subagents package directory
        result: Result of the last discovery run
    """
    
    __slots__ = ("package_dir", "result", "_discovered")
    
    def __init__(self, package_dir: Path | None = None) -> None:
        """
        Initialize the agent discovery.
        
        Args:
            package_dir: Optional path to package directory (defaults to this file's parent)
        """
        self.package_dir = package_dir or Path(__file__).parent
        self.result = DiscoveryResult()
        self._discovered = False
    
    def _find_agent_modules(self) -> list[str]:
        """
        Find all potential agent module names in the package directory.
        
        Returns:
            List of module names (without .py extension)
        """
        modules = []
        
        for path in self.package_dir.glob("*.py"):
            module_name = path.stem
            
            # Skip excluded modules and private modules
            if module_name in _EXCLUDED_MODULES:
                continue
            if module_name.startswith("_"):
                continue
                
            modules.append(module_name)
        
        return sorted(modules)
    
    def _import_module(self, module_name: str, package_name: str) -> bool:
        """
        Import a single module by name.
        
        Args:
            module_name: Name of the module to import
            package_name: Parent package name for relative imports
            
        Returns:
            True if import succeeded, False otherwise
        """
        full_module_path = f".{module_name}"
        
        try:
            importlib.import_module(full_module_path, package=package_name)
            logger.debug("Successfully imported agent module: %s", module_name)
            return True
            
        except ImportError as e:
            error_msg = f"ImportError: {e}"
            logger.warning("Failed to import %s: %s", module_name, error_msg)
            self.result.failed_modules[module_name] = error_msg
            return False
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error("Unexpected error importing %s: %s", module_name, error_msg)
            self.result.failed_modules[module_name] = error_msg
            return False
    
    def discover(self, force: bool = False) -> DiscoveryResult:
        """
        Discover and import all agent modules.
        
        This method finds all Python files in the package directory (excluding
        special files like __init__.py and interface.py) and imports them.
        Importing triggers the @register_subagent decorators.
        
        Args:
            force: If True, re-run discovery even if already done
            
        Returns:
            DiscoveryResult with details of discovered/failed modules
        """
        if self._discovered and not force:
            return self.result
        
        # Reset result for fresh discovery
        self.result = DiscoveryResult()
        
        # Find all potential agent modules
        module_names = self._find_agent_modules()
        logger.info("Found %d potential agent modules", len(module_names))
        
        # Import each module
        for module_name in module_names:
            if self._import_module(module_name, __name__):
                self.result.discovered_modules.append(module_name)
        
        self._discovered = True
        
        logger.info(
            "Agent discovery complete: %d succeeded, %d failed",
            self.result.success_count,
            self.result.failure_count,
        )
        
        return self.result
    
    def reload_module(self, module_name: str) -> bool:
        """
        Reload a specific agent module.
        
        Useful for development when you want to reload a modified agent
        without restarting the application.
        
        Args:
            module_name: Name of the module to reload
            
        Returns:
            True if reload succeeded, False otherwise
        """
        full_module_name = f"{__name__}.{module_name}"
        
        if full_module_name not in sys.modules:
            logger.warning("Module %s not loaded, importing instead", module_name)
            return self._import_module(module_name, __name__)
        
        try:
            module = sys.modules[full_module_name]
            importlib.reload(module)
            logger.info("Reloaded agent module: %s", module_name)
            return True
            
        except Exception as e:
            logger.error("Failed to reload %s: %s", module_name, e)
            return False


# Global discovery instance
_discovery = AgentDiscovery()


def discover_agents(force: bool = False) -> list[str]:
    """
    Automatically discover and import all agent modules in this package.
    
    This triggers their registration with the AgentRegistry via the
    @register_subagent decorators.
    
    Args:
        force: If True, re-run discovery even if already done
        
    Returns:
        List of successfully discovered module names
    """
    result = _discovery.discover(force=force)
    return result.discovered_modules


def get_discovery_result() -> DiscoveryResult:
    """
    Get the detailed result of the last discovery run.
    
    Returns:
        DiscoveryResult with discovered/failed modules
    """
    return _discovery.result


def reload_agent(module_name: str) -> bool:
    """
    Reload a specific agent module.
    
    Args:
        module_name: Name of the module to reload (without .py)
        
    Returns:
        True if reload succeeded
    """
    return _discovery.reload_module(module_name)


def get_all_subagent_classes() -> AgentClassMap:
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
    subagent_classes: AgentClassMap = {}

    for agent_id, metadata in all_agents.items():
        # Skip orchestrator and reviewer (they're not subagents)
        if agent_id in ("orchestrator", "reviewer"):
            continue

        agent_class = AgentRegistry.get_agent_class(agent_id)
        if agent_class is not None and issubclass(agent_class, SubAgentBase):
            subagent_classes[agent_id] = agent_class

    return subagent_classes


def get_subagent_by_id(agent_id: str) -> type[SubAgentBase] | None:
    """
    Get a specific subagent class by its ID.
    
    Args:
        agent_id: The agent's unique identifier
        
    Returns:
        The agent class if found, None otherwise
    """
    from ..registry import AgentRegistry
    
    discover_agents()
    
    agent_class = AgentRegistry.get_agent_class(agent_id)
    if agent_class is not None and issubclass(agent_class, SubAgentBase):
        return agent_class
    return None


def get_subagents_by_capability(capability: AgentCapability) -> list[type[SubAgentBase]]:
    """
    Get all subagent classes with a specific capability.
    
    Args:
        capability: The capability to filter by
        
    Returns:
        List of agent classes with the capability
    """
    from ..registry import AgentRegistry
    
    discover_agents()
    
    result: list[type[SubAgentBase]] = []
    all_agents = AgentRegistry.get_all_agents()
    
    for agent_id, metadata in all_agents.items():
        if capability in metadata.capabilities:
            agent_class = AgentRegistry.get_agent_class(agent_id)
            if agent_class is not None and issubclass(agent_class, SubAgentBase):
                result.append(agent_class)
    
    return result


def list_available_agents() -> list[dict[str, str]]:
    """
    Get a summary list of all available agents.
    
    Returns:
        List of dicts with agent_id, name, and description
    """
    from ..registry import AgentRegistry
    
    discover_agents()
    
    agents = []
    for agent_id, metadata in AgentRegistry.get_all_agents().items():
        if agent_id in ("orchestrator", "reviewer"):
            continue
        agents.append({
            "agent_id": metadata.agent_id,
            "name": metadata.name,
            "description": metadata.description,
            "priority": metadata.priority,
        })
    
    return sorted(agents, key=lambda x: x["priority"], reverse=True)


# Auto-discover on import
discover_agents()


__all__ = [
    # Interface components
    "SubAgentBase",
    "ToolDefinition",
    "ToolResult",
    "tool",
    "register_subagent",
    # Configuration
    "RetryConfig",
    "RetryStrategy",
    # Exceptions
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolValidationError",
    # Re-exports
    "AgentMetadata",
    "AgentCapability",
    "AgentContext",
    "AgentResponse",
    "AgentType",
    # Discovery functions
    "discover_agents",
    "get_discovery_result",
    "reload_agent",
    "get_all_subagent_classes",
    "get_subagent_by_id",
    "get_subagents_by_capability",
    "list_available_agents",
    # Discovery classes
    "AgentDiscovery",
    "DiscoveryResult",
    "AgentDiscoveryError",
]
