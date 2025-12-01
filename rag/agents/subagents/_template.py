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
"""
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI

from .interface import (
    SubAgentBase,
    tool,
    register_subagent,
    AgentMetadata,
    AgentCapability,
    AgentContext,
    AgentResponse,
    AgentType
)
from ...config import config


# =============================================================================
# AGENT METADATA
# =============================================================================
# This metadata tells the orchestrator what your agent does and when to use it.

TEMPLATE_AGENT_METADATA = AgentMetadata(
    # Unique identifier - used internally and in tool names (invoke_AGENT_ID_agent)
    agent_id="template",

    # Human-readable name
    name="Template Agent",

    # Short description
    description="A template agent for demonstration purposes",

    # Detailed description for the orchestrator - be specific about when to use this agent!
    detailed_description="""TEMPLATE: Beschreibung für den Orchestrator.
Verwende diesen Agenten für:
- Fall 1: Beschreibung
- Fall 2: Beschreibung
- Fall 3: Beschreibung

NICHT verwenden für:
- Fall A: Beschreibung (verwende stattdessen XYZ)""",

    # Capabilities - helps orchestrator categorize the agent
    capabilities=[AgentCapability.DATA_ANALYSIS],  # Options: DATABASE_QUERY, SEMANTIC_SEARCH, WEB_SEARCH, DATA_ANALYSIS, RESPONSE_GENERATION

    # Does this agent use a reasoning model (o1, o3, gpt-5)?
    uses_reasoning=False,

    # Default model to use
    default_model="gpt-4o-mini",

    # Parameters the orchestrator can pass to this agent
    parameters={
        "task_description": {
            "type": "string",
            "description": "Beschreibung der Aufgabe"
        },
        "optional_param": {
            "type": "integer",
            "description": "Ein optionaler Parameter"
        }
    },

    # Example queries this agent handles - helps orchestrator learn when to use it
    example_queries=[
        "Example query 1 that this agent handles",
        "Example query 2 that this agent handles",
        "Example query 3 that this agent handles"
    ],

    # Priority (higher = preferred when multiple agents could handle a query)
    priority=5
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
    """

    # Link to metadata (required)
    METADATA = TEMPLATE_AGENT_METADATA

    def __init__(
        self,
        openai_client: Optional[AsyncOpenAI] = None,
        verbose: bool = False,
        # Add any custom constructor parameters here
        custom_param: str = None
    ):
        """
        Initialize the agent.

        Args:
            openai_client: OpenAI client (optional, will create one if not provided)
            verbose: Enable verbose logging
            custom_param: Your custom parameter
        """
        super().__init__(verbose=verbose)

        # Set agent type (for response tracking)
        self._agent_type = AgentType.SQL_GENERATOR  # Choose appropriate type

        # Initialize OpenAI client if needed
        self.client = openai_client or AsyncOpenAI(api_key=config.openai_api_key)

        # Initialize your custom resources
        self.custom_param = custom_param

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
        required=["input_data"]  # Required parameters
    )
    async def process_data_tool(
        self,
        input_data: str,
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Example tool that processes data.

        This tool can be called via self.call_tool("process_data", input_data="...")
        or directly as self.process_data_tool(input_data="...")

        Args:
            input_data: The data to process
            options: Optional processing options

        Returns:
            Dict with processing results
        """
        self.log(f"Processing data: {input_data[:50]}...")

        # Your processing logic here
        result = {
            "processed": True,
            "input_length": len(input_data),
            "options_used": options or {}
        }

        return result

    @tool(
        name="fetch_external_data",
        description="Fetch data from an external source",
        parameters={
            "source_id": {"type": "string", "description": "ID of the source"},
            "limit": {"type": "integer", "description": "Max items to fetch"}
        },
        required=["source_id"]
    )
    async def fetch_external_data_tool(
        self,
        source_id: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Example tool that fetches external data.
        """
        self.log(f"Fetching from source: {source_id}")

        # Simulate fetching
        items = [{"id": i, "source": source_id} for i in range(limit)]

        return {
            "success": True,
            "items": items,
            "count": len(items)
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
        task_description = context.metadata.get("task_description", context.user_query)
        optional_param = context.metadata.get("optional_param", 100)

        self.log(f"Executing for task: {task_description[:50]}...")

        try:
            # 2. Perform main logic
            # You can use your tools here
            process_result = await self.process_data_tool(
                input_data=task_description,
                options={"limit": optional_param}
            )

            # 3. Optionally store results in context for reviewer
            # The reviewer agent can access these to format the final response
            # context.sql_results = [...]  # For SQL-like results
            # context.pinecone_results = [...]  # For search results
            # context.web_results = [...]  # For web results

            # 4. Return success response
            return AgentResponse.success_response(
                data={
                    "task": task_description,
                    "result": process_result,
                    "param_used": optional_param
                },
                agent_type=self._agent_type,
                reasoning=f"Processed task with param {optional_param}",
                tool_calls=[{"name": "process_data", "input": task_description[:50]}],
                sources=[{"type": "template_agent", "task": task_description[:50]}]
            )

        except Exception as e:
            self.log(f"Error: {str(e)}")

            # Return error response
            return AgentResponse.error_response(
                error=f"Template agent failed: {str(e)}",
                agent_type=self._agent_type
            )

    # =========================================================================
    # HELPER METHODS (OPTIONAL)
    # =========================================================================
    # Add any helper methods your agent needs

    def _validate_input(self, data: str) -> bool:
        """Example helper method for input validation."""
        return bool(data and len(data) > 0)

    async def _call_external_api(self, endpoint: str) -> Dict[str, Any]:
        """Example helper method for API calls."""
        # Your API logic here
        return {"endpoint": endpoint, "status": "ok"}
