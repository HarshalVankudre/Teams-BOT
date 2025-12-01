"""
Agent Registry System

Provides dynamic agent discovery and registration.
Each agent registers itself with metadata describing its capabilities,
allowing the orchestrator to intelligently select agents at runtime.
"""
from typing import Dict, List, Type, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class AgentCapability(Enum):
    """Categories of agent capabilities"""
    DATABASE_QUERY = "database_query"
    SEMANTIC_SEARCH = "semantic_search"
    WEB_SEARCH = "web_search"
    DATA_ANALYSIS = "data_analysis"
    RESPONSE_GENERATION = "response_generation"
    ORCHESTRATION = "orchestration"


@dataclass
class AgentMetadata:
    """Metadata describing an agent's capabilities and configuration"""

    # Unique identifier for the agent
    agent_id: str

    # Human-readable name
    name: str

    # Description of what the agent does (used by orchestrator)
    description: str

    # Detailed description for orchestrator's system prompt
    detailed_description: str

    # List of capabilities this agent provides
    capabilities: List[AgentCapability]

    # Whether this agent uses a reasoning model
    uses_reasoning: bool = False

    # Default model to use (can be overridden)
    default_model: Optional[str] = None

    # Parameters the agent accepts (for tool definition)
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Example queries this agent can handle
    example_queries: List[str] = field(default_factory=list)

    # Priority when multiple agents could handle a query (higher = preferred)
    priority: int = 0

    # Whether this agent can be invoked directly or only through orchestrator
    direct_invocation: bool = True


class AgentRegistry:
    """
    Central registry for all available agents.
    Agents register themselves on import, making the system extensible.
    """

    _instance: Optional['AgentRegistry'] = None
    _agents: Dict[str, 'AgentMetadata'] = {}
    _agent_classes: Dict[str, Type] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._agents = {}
            cls._instance._agent_classes = {}
        return cls._instance

    @classmethod
    def register(
        cls,
        metadata: AgentMetadata,
        agent_class: Type
    ) -> None:
        """Register an agent with its metadata"""
        registry = cls()
        registry._agents[metadata.agent_id] = metadata
        registry._agent_classes[metadata.agent_id] = agent_class

    @classmethod
    def get_metadata(cls, agent_id: str) -> Optional[AgentMetadata]:
        """Get metadata for a specific agent"""
        registry = cls()
        return registry._agents.get(agent_id)

    @classmethod
    def get_agent_class(cls, agent_id: str) -> Optional[Type]:
        """Get the class for a specific agent"""
        registry = cls()
        return registry._agent_classes.get(agent_id)

    @classmethod
    def get_all_agents(cls) -> Dict[str, AgentMetadata]:
        """Get all registered agents"""
        registry = cls()
        return registry._agents.copy()

    @classmethod
    def get_agents_by_capability(
        cls,
        capability: AgentCapability
    ) -> List[AgentMetadata]:
        """Get all agents that have a specific capability"""
        registry = cls()
        return [
            meta for meta in registry._agents.values()
            if capability in meta.capabilities
        ]

    @classmethod
    def get_invokable_agents(cls) -> List[AgentMetadata]:
        """Get all agents that can be directly invoked by orchestrator"""
        registry = cls()
        return [
            meta for meta in registry._agents.values()
            if meta.direct_invocation and meta.agent_id not in ['orchestrator', 'reviewer']
        ]

    @classmethod
    def generate_orchestrator_tools(cls) -> List[Dict[str, Any]]:
        """Generate tool definitions for the orchestrator based on registered agents"""
        tools = []

        for metadata in cls.get_invokable_agents():
            tool = {
                "type": "function",
                "function": {
                    "name": f"invoke_{metadata.agent_id}_agent",
                    "description": metadata.detailed_description,
                    "parameters": {
                        "type": "object",
                        "properties": metadata.parameters.copy(),
                        "required": list(metadata.parameters.keys()) if metadata.parameters else []
                    }
                }
            }
            tools.append(tool)

        # Always add clarification tool
        tools.append({
            "type": "function",
            "function": {
                "name": "request_clarification",
                "description": "Request clarification from user when query is ambiguous",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The clarifying question to ask the user"
                        },
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of suggested options for the user"
                        }
                    },
                    "required": ["question"]
                }
            }
        })

        return tools

    @classmethod
    def generate_agent_descriptions(cls) -> str:
        """Generate a description of all available agents for the orchestrator prompt"""
        descriptions = []

        for metadata in cls.get_invokable_agents():
            desc = f"""
## {metadata.name} (invoke_{metadata.agent_id}_agent)
{metadata.detailed_description}

Beispiel-Anfragen:
{chr(10).join(f'- {ex}' for ex in metadata.example_queries[:3])}
"""
            descriptions.append(desc)

        return "\n".join(descriptions)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered agents (useful for testing)"""
        registry = cls()
        registry._agents.clear()
        registry._agent_classes.clear()


# Decorator for easy agent registration
def register_agent(metadata: AgentMetadata):
    """Decorator to register an agent class with metadata"""
    def decorator(cls):
        AgentRegistry.register(metadata, cls)
        return cls
    return decorator
