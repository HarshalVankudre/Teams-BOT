"""
Orchestrator Agent (Reasoning Model)
Analyzes user queries, understands context, and decides which sub-agents to invoke.
Uses a reasoning model to think through the query requirements.
Dynamically discovers available agents from the registry.
"""
import json
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

from .base import BaseAgent, AgentContext, AgentResponse, AgentType
from .registry import AgentMetadata, AgentCapability, AgentRegistry, register_agent
from ..config import config
from ..schema import ORCHESTRATOR_SCHEMA


# Agent metadata for registration
ORCHESTRATOR_AGENT_METADATA = AgentMetadata(
    agent_id="orchestrator",
    name="Orchestrator Agent",
    description="Analyzes queries and routes to appropriate sub-agents",
    detailed_description="""Analysiert Benutzeranfragen und entscheidet welche Sub-Agenten aufgerufen werden.
Verwendet ein Reasoning-Modell um die Anforderungen der Anfrage zu verstehen.""",
    capabilities=[AgentCapability.ORCHESTRATION],
    uses_reasoning=True,
    default_model="gpt-5",
    parameters={},
    example_queries=[],
    priority=100,
    direct_invocation=False  # Orchestrator is called by agent system, not by itself
)


@register_agent(ORCHESTRATOR_AGENT_METADATA)
class OrchestratorAgent(BaseAgent):
    """
    Main orchestrator that analyzes queries and routes to appropriate sub-agents.
    Uses a reasoning model to understand query intent and requirements.
    Dynamically discovers available agents from the registry.
    """

    # Database schema imported from centralized schema.py
    DATABASE_SCHEMA = ORCHESTRATOR_SCHEMA

    # Base system prompt template - agent descriptions are injected dynamically
    SYSTEM_PROMPT_TEMPLATE = """Du bist der Orchestrator-Agent für das RÜKO Baumaschinen-System.

DEINE AUFGABE:
Analysiere die Benutzeranfrage und entscheide, welche Agenten aufgerufen werden müssen.

{database_schema}

VERFÜGBARE AGENTEN:
{agent_descriptions}

ENTSCHEIDUNGSLOGIK:

- Zählen/Filtern/Auflisten → SQL-Agent (EINE Abfrage pro Anfrage-Teil)
- Konkrete Eigenschaften (Gewicht, Leistung, Ausstattung) → SQL-Agent
- "Wie viele", "Welche", "Liste alle" → SQL-Agent
- "Empfehlung", "eignet sich", "beste für" → Pinecone-Agent
- Szenario-Beschreibungen ohne exakte Werte → Pinecone-Agent
- Externe Informationen (Marktpreise, News) → Web-Such-Agent (nur wenn nötig!)

WICHTIG FÜR MULTI-TEIL ANFRAGEN:
Wenn eine Anfrage mehrere Teile hat (z.B. "Wie viele X und welche Y"):
- Rufe den SQL-Agent MEHRMALS auf - einmal pro Teil!
- Teil 1: "Wie viele Liebherr" → invoke_sql_agent für Zählung
- Teil 2: "welche Caterpillar" → invoke_sql_agent für Auflistung
- NICHT versuchen, alles in einer Abfrage zu kombinieren!

KONTEXT-BEWUSSTSEIN:
{conversation_context}

REGELN:
- Rufe IMMER mindestens einen Agenten auf
- Bevorzuge interne Daten (SQL/Pinecone) vor Web-Suche
- Bei Multi-Teil-Anfragen: MEHRERE separate Agenten-Aufrufe
- Wenn unklar: Frage nach (request_clarification)"""

    def __init__(
        self,
        openai_client: Optional[AsyncOpenAI] = None,
        model: Optional[str] = None,
        reasoning_effort: str = "medium",
        verbose: bool = False
    ):
        super().__init__(verbose=verbose)
        self._agent_type = AgentType.ORCHESTRATOR
        self.client = openai_client or AsyncOpenAI(api_key=config.openai_api_key)

        # Use configured model or fall back to config
        self.model = model or config.response_model
        self.reasoning_effort = reasoning_effort

        # Cache for tools - regenerated when agents change
        self._cached_tools: Optional[List[Dict]] = None
        self._cached_descriptions: Optional[str] = None

    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """Get tool definitions from the agent registry"""
        if self._cached_tools is None:
            self._cached_tools = AgentRegistry.generate_orchestrator_tools()
            self.log(f"Loaded {len(self._cached_tools)} tools from registry")
        return self._cached_tools

    def _get_agent_descriptions(self) -> str:
        """Get agent descriptions from the registry for the system prompt"""
        if self._cached_descriptions is None:
            self._cached_descriptions = AgentRegistry.generate_agent_descriptions()
        return self._cached_descriptions

    def refresh_agent_cache(self) -> None:
        """Refresh the cached tools and descriptions from registry"""
        self._cached_tools = None
        self._cached_descriptions = None
        self.log("Agent cache refreshed")

    def _build_conversation_context(self, context: AgentContext) -> str:
        """Build conversation context string for the system prompt"""
        if not context.conversation_history:
            return "Keine vorherige Konversation."

        history_lines = []
        for entry in context.conversation_history[-5:]:  # Last 5 exchanges
            role = entry.get("role", "unknown")
            content = entry.get("content", "")[:200]  # Truncate long messages
            history_lines.append(f"- {role}: {content}")

        return "Letzte Nachrichten:\n" + "\n".join(history_lines)

    def _supports_reasoning(self) -> bool:
        """Check if the model supports reasoning parameter in responses API"""
        if not self.model:
            return False
        model_lower = self.model.lower()
        # Only o1 and o3 series support reasoning in chat completions
        # GPT-5 uses responses API which is handled separately
        return model_lower.startswith('o1') or model_lower.startswith('o3')

    def _uses_max_completion_tokens(self) -> bool:
        """Check if model uses max_completion_tokens instead of max_tokens"""
        if not self.model:
            return False
        model_lower = self.model.lower()
        return (model_lower.startswith('gpt-5') or
                model_lower.startswith('o1') or
                model_lower.startswith('o3'))

    async def _execute(self, context: AgentContext) -> AgentResponse:
        """
        Analyze the query and decide which agents to invoke.
        Returns a plan of agent invocations.
        """
        # Get dynamically registered tools and descriptions
        available_tools = self._get_available_tools()
        agent_descriptions = self._get_agent_descriptions()

        # Build system prompt with conversation context, schema, and agent descriptions
        conversation_context = self._build_conversation_context(context)
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            database_schema=self.DATABASE_SCHEMA,
            agent_descriptions=agent_descriptions,
            conversation_context=conversation_context
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analysiere diese Anfrage und entscheide welche Agenten aufgerufen werden sollen:\n\n{context.user_query}"}
        ]

        # Build request parameters
        request_params = {
            "model": self.model,
            "messages": messages,
            "tools": available_tools,
            "tool_choice": "required"  # Force tool selection
        }

        # Add reasoning for supported models
        if self._supports_reasoning() and self.reasoning_effort != "none":
            request_params["reasoning"] = {"effort": self.reasoning_effort}

        self.log(f"Using model: {self.model}")

        # Call the LLM
        response = await self.client.chat.completions.create(**request_params)
        message = response.choices[0].message

        # Extract tool calls (agent invocations)
        agent_plan = []
        reasoning = ""

        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                agent_plan.append({
                    "agent": tool_name,
                    "arguments": tool_args,
                    "tool_call_id": tool_call.id
                })

                self.log(f"Plan: {tool_name} with {tool_args}")

        # Extract reasoning if present
        if message.content:
            reasoning = message.content

        # Update context with orchestrator analysis
        context.query_intent = self._determine_intent(agent_plan)
        context.required_tools = [p["agent"] for p in agent_plan]
        context.reasoning = reasoning

        return AgentResponse.success_response(
            data={
                "agent_plan": agent_plan,
                "reasoning": reasoning,
                "query_intent": context.query_intent
            },
            agent_type=self._agent_type,
            reasoning=reasoning,
            tool_calls=[{"name": p["agent"], "args": p["arguments"]} for p in agent_plan]
        )

    def _determine_intent(self, agent_plan: List[Dict]) -> str:
        """Determine the overall intent based on the planned agents"""
        agents = [p["agent"] for p in agent_plan]

        if "request_clarification" in agents:
            return "clarification_needed"

        # Check for known agent patterns
        has_sql = any("sql" in a for a in agents)
        has_pinecone = any("pinecone" in a for a in agents)
        has_web = any("web" in a for a in agents)

        if has_sql and has_pinecone:
            return "hybrid_query"
        elif has_sql:
            return "structured_query"
        elif has_pinecone:
            return "semantic_query"
        elif has_web:
            return "external_search"
        else:
            return "unknown"

    def get_registered_agents(self) -> Dict[str, AgentMetadata]:
        """Get all registered agents (useful for debugging/introspection)"""
        return AgentRegistry.get_all_agents()

    def get_invokable_agents(self) -> List[AgentMetadata]:
        """Get all agents that can be invoked by this orchestrator"""
        return AgentRegistry.get_invokable_agents()
