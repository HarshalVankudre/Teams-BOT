"""
TEMPLATE: Custom SubAgent

Copy this file to create a new subagent. Replace all TEMPLATE/template
placeholders with your agent's name and logic.

Steps to create a new agent:
    1. Copy this file: cp _template.py my_agent.py
    2. Rename the class and update METADATA
    3. Implement your tools using @tool decorator
    4. Implement _execute() method
    5. The agent will be auto-discovered on restart!

No need to modify any other files - the orchestrator will
automatically discover and use your new agent.

Author: RÜKO GmbH Baumaschinen
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from openai import AsyncOpenAI

from .interface import (
    AgentCapability,
    AgentContext,
    AgentMetadata,
    AgentResponse,
    AgentType,
    RetryConfig,
    RetryStrategy,
    SubAgentBase,
    ToolResult,
    register_subagent,
    tool,
)
from ...config import config

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True, slots=True)
class TemplateAgentConfig:
    """
    Configuration for the template agent.
    
    Define your agent's configurable parameters here.
    Use frozen=True to make it immutable after creation.
    """
    
    default_limit: int = 100
    max_retries: int = 3
    timeout_seconds: float = 30.0
    # Add your configuration parameters here


DEFAULT_CONFIG: Final[TemplateAgentConfig] = TemplateAgentConfig()


# =============================================================================
# AGENT METADATA
# =============================================================================
# This metadata tells the orchestrator what your agent does and when to use it.

TEMPLATE_AGENT_METADATA: Final[AgentMetadata] = AgentMetadata(
    # Unique identifier - used internally and in tool names (invoke_AGENT_ID_agent)
    agent_id="template",

    # Human-readable name
    name="Template Agent",

    # Short description (shown in agent list)
    description="A template agent for demonstration purposes",

    # Detailed description for the orchestrator - be specific about when to use this agent!
    # This is critical for proper routing by the orchestrator.
    detailed_description="""TEMPLATE: Beschreibung für den Orchestrator.
Verwende diesen Agenten für:
- Fall 1: Beschreibung
- Fall 2: Beschreibung
- Fall 3: Beschreibung

NICHT verwenden für:
- Fall A: Beschreibung (verwende stattdessen XYZ)""",

    # Capabilities - helps orchestrator categorize the agent
    # Options: DATABASE_QUERY, SEMANTIC_SEARCH, WEB_SEARCH, DATA_ANALYSIS, RESPONSE_GENERATION
    capabilities=[AgentCapability.DATA_ANALYSIS],

    # Does this agent use a reasoning model (o1, o3, gpt-5)?
    uses_reasoning=False,

    # Default model to use
    default_model="gpt-4o-mini",

    # Parameters the orchestrator can pass to this agent
    # These become available in context.metadata
    parameters={
        "task_description": {
            "type": "string",
            "description": "Beschreibung der Aufgabe"
        },
        "optional_param": {
            "type": "integer",
            "description": "Ein optionaler Parameter"
        },
        # Add more parameters as needed
    },

    # Example queries this agent handles - helps orchestrator learn when to use it
    example_queries=[
        "Example query 1 that this agent handles",
        "Example query 2 that this agent handles",
        "Example query 3 that this agent handles",
    ],

    # Priority (higher = preferred when multiple agents could handle a query)
    priority=5,
)


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

@register_subagent()  # This decorator registers the agent with the registry
class TemplateAgent(SubAgentBase):
    """
    Template agent implementation.

    This agent demonstrates the pattern for creating new subagents.
    Replace this with your actual implementation.
    
    Attributes:
        client: AsyncOpenAI client for LLM calls
        agent_config: Configuration settings
        custom_param: Your custom parameter
    
    Example:
        >>> agent = TemplateAgent(verbose=True)
        >>> context = AgentContext(
        ...     user_query="Do something",
        ...     metadata={"task_description": "Process this"}
        ... )
        >>> response = await agent.execute(context)
    """

    # Link to metadata (required)
    METADATA = TEMPLATE_AGENT_METADATA

    # Class-level constants
    _DEFAULT_TIMEOUT: Final[float] = 30.0

    # Define slots for memory efficiency
    __slots__ = ("client", "agent_config", "custom_param")

    def __init__(
        self,
        openai_client: AsyncOpenAI | None = None,
        verbose: bool = False,
        agent_config: TemplateAgentConfig | None = None,
        custom_param: str | None = None,
    ) -> None:
        """
        Initialize the agent.

        Args:
            openai_client: OpenAI client (optional, will create one if not provided)
            verbose: Enable verbose logging
            agent_config: Optional custom configuration
            custom_param: Your custom parameter
        """
        super().__init__(verbose=verbose)

        # Set agent type (for response tracking)
        # Choose from: SQL_GENERATOR, PINECONE_SEARCH, WEB_SEARCH, etc.
        self._agent_type = AgentType.SQL_GENERATOR

        # Initialize configuration
        self.agent_config = agent_config or DEFAULT_CONFIG

        # Initialize OpenAI client if needed
        self.client = openai_client or AsyncOpenAI(api_key=config.openai_api_key)

        # Initialize your custom resources
        self.custom_param = custom_param
        
        logger.debug(
            "Initialized %s with custom_param=%s",
            self.__class__.__name__,
            custom_param,
        )

    # =========================================================================
    # TOOLS
    # =========================================================================
    # Tools are functions that can be called during execution.
    # Use the @tool decorator to define them.

    @tool(
        name="process_data",
        description="Process some data and return results",
        parameters={
            "input_data": {"type": "string", "description": "The data to process"},
            "options": {"type": "object", "description": "Processing options"}
        },
        required=["input_data"],  # Required parameters
        retry_config=RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            max_attempts=3,
            base_delay_seconds=0.5,
        ),
        timeout_seconds=30.0,
        tags=["processing", "data"],
    )
    async def process_data_tool(
        self,
        input_data: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Example tool that processes data.

        This tool can be called via:
            result = await self.call_tool("process_data", input_data="...")
        
        Or directly:
            result = await self.process_data_tool(input_data="...")

        Args:
            input_data: The data to process
            options: Optional processing options

        Returns:
            Dict with processing results
        """
        self.log(f"Processing data: {input_data[:50]}...")
        logger.debug("Processing data with options: %s", options)

        # Your processing logic here
        result = {
            "processed": True,
            "input_length": len(input_data),
            "options_used": options or {},
        }

        return result

    @tool(
        name="fetch_external_data",
        description="Fetch data from an external source",
        parameters={
            "source_id": {"type": "string", "description": "ID of the source"},
            "limit": {"type": "integer", "description": "Max items to fetch"}
        },
        required=["source_id"],
        tags=["fetch", "external"],
    )
    async def fetch_external_data_tool(
        self,
        source_id: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Example tool that fetches external data.
        
        Args:
            source_id: ID of the data source
            limit: Maximum items to fetch
            
        Returns:
            Dict with fetched items
        """
        self.log(f"Fetching from source: {source_id}")
        logger.info("Fetching %d items from source %s", limit, source_id)

        # Simulate fetching
        items = [{"id": i, "source": source_id} for i in range(limit)]

        return {
            "success": True,
            "items": items,
            "count": len(items),
        }

    @tool(
        name="validate_input",
        description="Validate input data before processing",
        parameters={
            "data": {"type": "string", "description": "Data to validate"},
            "strict": {"type": "boolean", "description": "Use strict validation"}
        },
        required=["data"],
        tags=["validation"],
    )
    async def validate_input_tool(
        self,
        data: str,
        strict: bool = False,
    ) -> dict[str, Any]:
        """
        Validate input data.
        
        Args:
            data: Data to validate
            strict: Whether to use strict validation
            
        Returns:
            Dict with validation results
        """
        is_valid = bool(data and len(data) > 0)
        
        if strict:
            is_valid = is_valid and len(data) >= 10
        
        return {
            "valid": is_valid,
            "length": len(data) if data else 0,
            "mode": "strict" if strict else "normal",
        }

    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================

    async def _execute(self, context: AgentContext) -> AgentResponse:
        """
        Main execution method - called when the orchestrator invokes this agent.

        This method should:
            1. Extract parameters from context.metadata (set by orchestrator)
            2. Perform the agent's main logic (using tools if needed)
            3. Store results in context for the reviewer (optional)
            4. Return an AgentResponse

        Args:
            context: AgentContext with user_query, metadata, conversation_history, etc.

        Returns:
            AgentResponse with success/error status and data
        """
        # 1. Extract parameters from context
        # The orchestrator passes parameters via context.metadata
        task_description = context.metadata.get("task_description")
        optional_param = context.metadata.get("optional_param", self.agent_config.default_limit)

        # Validate required parameters
        if not task_description:
            return AgentResponse.error_response(
                error="No task description provided by orchestrator",
                agent_type=self._agent_type,
            )

        self.log(f"Executing for task: {task_description[:50]}...")
        logger.info("Template agent executing: %s", task_description[:50])

        try:
            # 2. Validate input first (using our tool)
            validation = await self.call_tool("validate_input", data=task_description)
            
            if not validation.success or not validation.data.get("valid"):
                return AgentResponse.error_response(
                    error="Input validation failed",
                    agent_type=self._agent_type,
                )

            # 3. Perform main logic using tools
            process_result = await self.call_tool(
                "process_data",
                input_data=task_description,
                options={"limit": optional_param},
            )

            if not process_result.success:
                return AgentResponse.error_response(
                    error=f"Processing failed: {process_result.error}",
                    agent_type=self._agent_type,
                )

            # 4. Optionally store results in context for reviewer
            # The reviewer agent can access these to format the final response
            # context.sql_results = [...]  # For SQL-like results
            # context.pinecone_results = [...]  # For search results
            # context.web_results = [...]  # For web results

            # 5. Return success response
            return AgentResponse.success_response(
                data={
                    "task": task_description,
                    "result": process_result.data,
                    "param_used": optional_param,
                    "validation": validation.data,
                },
                agent_type=self._agent_type,
                reasoning=f"Processed task with param {optional_param}",
                tool_calls=[
                    {"name": "validate_input", "data": task_description[:50]},
                    {"name": "process_data", "input": task_description[:50]},
                ],
                sources=[{"type": "template_agent", "task": task_description[:50]}],
            )

        except Exception as e:
            logger.error("Template agent error: %s", e, exc_info=True)
            self.log(f"Error: {e}")

            # Return error response
            return AgentResponse.error_response(
                error=f"Template agent failed: {e}",
                agent_type=self._agent_type,
            )

    # =========================================================================
    # HELPER METHODS (OPTIONAL)
    # =========================================================================
    # Add any helper methods your agent needs

    def _validate_config(self) -> bool:
        """Example helper method for configuration validation."""
        return self.agent_config is not None

    async def _call_external_api(self, endpoint: str) -> dict[str, Any]:
        """
        Example helper method for API calls.
        
        Args:
            endpoint: API endpoint to call
            
        Returns:
            Response data
        """
        # Your API logic here
        return {"endpoint": endpoint, "status": "ok"}

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"custom_param={self.custom_param!r}, "
            f"tools={len(self._tools)})"
        )
