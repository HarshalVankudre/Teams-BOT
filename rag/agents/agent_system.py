"""
Agent System Coordinator
Main entry point that orchestrates all agents to process user queries.
Handles conversation context from Redis and coordinates the multi-agent flow.
Uses the agent registry for dynamic agent discovery.
"""
import asyncio
import time
from typing import Dict, Any, Optional, List, Type
from dataclasses import dataclass, field
from openai import AsyncOpenAI

from .base import AgentContext, AgentResponse, BaseAgent
from .registry import AgentRegistry, AgentMetadata
from ..config import config as rag_config


@dataclass
class AgentSystemConfig:
    """Configuration for the agent system"""
    # Model settings
    orchestrator_model: Optional[str] = None  # Uses config.response_model if None
    sql_model: Optional[str] = None  # Uses fast model for SQL generation
    reviewer_model: Optional[str] = None  # Uses config.response_model if None

    # Reasoning settings
    orchestrator_reasoning: str = "medium"
    reviewer_reasoning: str = "medium"

    # Behavior settings
    enable_web_search: bool = True
    parallel_execution: bool = True  # Execute independent agents in parallel
    max_agent_iterations: int = 3  # Max times orchestrator can refine

    # Verbose logging
    verbose: bool = False


@dataclass
class AgentSystemResult:
    """Result from the agent system"""
    response: str
    success: bool = True
    error: Optional[str] = None
    execution_time_ms: int = 0
    agents_used: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    query_intent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentSystem:
    """
    Main coordinator that orchestrates all agents.
    Dynamically discovers available agents from the registry.

    Flow:
    1. Load conversation context from Redis
    2. Orchestrator analyzes query and decides which agents to invoke
    3. Sub-agents execute in parallel where possible
    4. Reviewer generates final response
    5. Store updated context in Redis
    """

    def __init__(
        self,
        openai_client: Optional[AsyncOpenAI] = None,
        config: Optional[AgentSystemConfig] = None,
        redis_client=None
    ):
        self.openai_client = openai_client or AsyncOpenAI(api_key=rag_config.openai_api_key)
        self.config = config or AgentSystemConfig()
        self.redis_client = redis_client

        # Agent instances - dynamically populated from registry
        self._agents: Dict[str, BaseAgent] = {}

        # Initialize agents from registry
        self._init_agents()

    def _init_agents(self):
        """Initialize all agents from the registry"""
        # Import orchestrator and reviewer (top-level agents)
        from . import orchestrator, reviewer_agent

        # Import subagents - this triggers auto-discovery and registration
        from .subagents import discover_agents
        discover_agents()

        # Get all registered agents
        all_agents = AgentRegistry.get_all_agents()
        self._log(f"Found {len(all_agents)} registered agents")

        # Initialize each agent with appropriate configuration
        for agent_id, metadata in all_agents.items():
            agent_class = AgentRegistry.get_agent_class(agent_id)
            if agent_class is None:
                continue

            agent_instance = self._create_agent_instance(agent_id, agent_class, metadata)
            if agent_instance:
                self._agents[agent_id] = agent_instance
                self._log(f"  Initialized: {metadata.name}")

        self._log("Agent system initialized")

    def _create_agent_instance(
        self,
        agent_id: str,
        agent_class: Type[BaseAgent],
        metadata: AgentMetadata
    ) -> Optional[BaseAgent]:
        """Create an instance of an agent with appropriate configuration"""
        try:
            # Build kwargs based on what the agent accepts
            kwargs = {"verbose": self.config.verbose}

            # Add OpenAI client if agent uses it
            if hasattr(agent_class, '__init__'):
                import inspect
                sig = inspect.signature(agent_class.__init__)
                params = sig.parameters

                if 'openai_client' in params:
                    kwargs['openai_client'] = self.openai_client

                # Set model based on agent type
                if 'model' in params:
                    if agent_id == 'orchestrator':
                        kwargs['model'] = self.config.orchestrator_model
                    elif agent_id == 'reviewer':
                        kwargs['model'] = self.config.reviewer_model
                    elif agent_id == 'sql':
                        kwargs['model'] = self.config.sql_model
                    elif metadata.default_model:
                        kwargs['model'] = metadata.default_model

                # Set reasoning effort
                if 'reasoning_effort' in params:
                    if agent_id == 'orchestrator':
                        kwargs['reasoning_effort'] = self.config.orchestrator_reasoning
                    elif agent_id == 'reviewer':
                        kwargs['reasoning_effort'] = self.config.reviewer_reasoning

            return agent_class(**kwargs)

        except Exception as e:
            self._log(f"Failed to create agent {agent_id}: {e}")
            return None

    def _log(self, message: str):
        """Log message if verbose"""
        if self.config.verbose:
            print(f"[AGENT_SYSTEM] {message}")

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent instance by ID"""
        return self._agents.get(agent_id)

    def get_registered_agents(self) -> Dict[str, AgentMetadata]:
        """Get all registered agent metadata"""
        return AgentRegistry.get_all_agents()

    async def process(
        self,
        user_query: str,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        thread_key: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> AgentSystemResult:
        """
        Process a user query through the multi-agent system.

        Args:
            user_query: The user's question/request
            user_id: User identifier
            user_name: User display name
            thread_key: Conversation thread key for Redis
            conversation_history: Previous conversation messages

        Returns:
            AgentSystemResult with the response and metadata
        """
        start_time = time.time()
        agents_used = []
        all_sources = []

        try:
            # 1. Load conversation context from Redis if available
            redis_history = []
            if self.redis_client and thread_key:
                redis_history = await self._load_conversation_context(thread_key)

            # Merge histories (Redis + provided)
            full_history = redis_history + (conversation_history or [])

            # 2. Create agent context
            context = AgentContext(
                user_query=user_query,
                conversation_history=full_history,
                user_id=user_id,
                user_name=user_name,
                thread_key=thread_key
            )

            self._log(f"Processing: {user_query[:50]}...")

            # 3. Orchestrator analyzes and plans
            orchestrator = self.get_agent('orchestrator')
            if not orchestrator:
                return AgentSystemResult(
                    response="System nicht korrekt initialisiert - Orchestrator fehlt.",
                    success=False,
                    error="Orchestrator not found",
                    execution_time_ms=int((time.time() - start_time) * 1000)
                )

            orchestrator_result = await orchestrator.execute(context)
            agents_used.append("orchestrator")

            if not orchestrator_result.success:
                return AgentSystemResult(
                    response=f"Entschuldigung, ich konnte die Anfrage nicht verarbeiten: {orchestrator_result.error}",
                    success=False,
                    error=orchestrator_result.error,
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    agents_used=agents_used
                )

            # Get the agent plan
            agent_plan = orchestrator_result.data.get("agent_plan", [])
            context.query_intent = orchestrator_result.data.get("query_intent")
            context.reasoning = orchestrator_result.data.get("reasoning")

            self._log(f"Plan: {[p['agent'] for p in agent_plan]}")

            # 4. Handle clarification requests
            for plan_item in agent_plan:
                if plan_item["agent"] == "request_clarification":
                    question = plan_item["arguments"].get("question", "")
                    options = plan_item["arguments"].get("options", [])

                    response = f"{question}"
                    if options:
                        response += "\n\nOptionen:\n"
                        for i, opt in enumerate(options, 1):
                            response += f"{i}. {opt}\n"

                    return AgentSystemResult(
                        response=response,
                        success=True,
                        execution_time_ms=int((time.time() - start_time) * 1000),
                        agents_used=agents_used,
                        query_intent="clarification_needed"
                    )

            # 5. Execute sub-agents based on plan
            if self.config.parallel_execution:
                await self._execute_agents_parallel(context, agent_plan, agents_used, all_sources)
            else:
                await self._execute_agents_sequential(context, agent_plan, agents_used, all_sources)

            # 6. Reviewer generates final response
            reviewer = self.get_agent('reviewer')
            if not reviewer:
                return AgentSystemResult(
                    response="System nicht korrekt initialisiert - Reviewer fehlt.",
                    success=False,
                    error="Reviewer not found",
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    agents_used=agents_used
                )

            self._log("Generating final response...")
            reviewer_result = await reviewer.execute(context)
            agents_used.append("reviewer")

            if not reviewer_result.success:
                return AgentSystemResult(
                    response="Entschuldigung, ich konnte keine Antwort generieren.",
                    success=False,
                    error=reviewer_result.error,
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    agents_used=agents_used
                )

            response_text = reviewer_result.data.get("response", "Keine Antwort verfügbar.")

            # 7. Store updated context in Redis
            if self.redis_client and thread_key:
                await self._store_conversation_context(thread_key, user_query, response_text)

            execution_time = int((time.time() - start_time) * 1000)
            self._log(f"Completed in {execution_time}ms")

            return AgentSystemResult(
                response=response_text,
                success=True,
                execution_time_ms=execution_time,
                agents_used=agents_used,
                sources=all_sources,
                query_intent=context.query_intent,
                metadata={
                    "orchestrator_reasoning": context.reasoning,
                    "sql_results_count": len(context.sql_results) if context.sql_results else 0,
                    "pinecone_results_count": len(context.pinecone_results) if context.pinecone_results else 0,
                    "web_results_count": len(context.web_results) if context.web_results else 0
                }
            )

        except Exception as e:
            self._log(f"Error: {str(e)}")
            return AgentSystemResult(
                response=f"Ein Fehler ist aufgetreten: {str(e)}",
                success=False,
                error=str(e),
                execution_time_ms=int((time.time() - start_time) * 1000),
                agents_used=agents_used
            )

    def _extract_agent_id_from_tool(self, tool_name: str) -> Optional[str]:
        """Extract agent ID from tool name like 'invoke_sql_agent' -> 'sql'"""
        if tool_name.startswith("invoke_") and tool_name.endswith("_agent"):
            return tool_name[7:-6]  # Remove 'invoke_' prefix and '_agent' suffix
        return None

    async def _execute_agents_parallel(
        self,
        context: AgentContext,
        agent_plan: List[Dict],
        agents_used: List[str],
        all_sources: List[Dict]
    ):
        """Execute independent agents in parallel"""
        import copy

        tasks = []
        sql_results_with_context = []  # Collect SQL results with query context

        for i, plan_item in enumerate(agent_plan):
            tool_name = plan_item["agent"]
            args = plan_item["arguments"]

            # Extract agent ID from tool name
            agent_id = self._extract_agent_id_from_tool(tool_name)
            if not agent_id:
                continue

            agent = self.get_agent(agent_id)
            if not agent:
                self._log(f"Agent not found: {agent_id}")
                continue

            # Skip web search if disabled
            if agent_id == "web_search" and not self.config.enable_web_search:
                continue

            # Create a copy of context for each agent to avoid metadata conflicts
            agent_context = copy.copy(context)
            agent_context.metadata = context.metadata.copy()

            # Set up context metadata based on agent type
            self._setup_agent_context(agent_context, agent_id, args)

            # Store task description for context
            task_desc = args.get("task_description", args.get("search_query", ""))
            tasks.append((agent_id, agent_context, agent.execute(agent_context), i, task_desc))

        # Execute all tasks in parallel
        if tasks:
            self._log(f"Executing {len(tasks)} agents in parallel...")
            results = await asyncio.gather(*[t[2] for t in tasks], return_exceptions=True)

            for (agent_id, agent_context, _, idx, task_desc), result in zip(tasks, results):
                agents_used.append(agent_id)

                if isinstance(result, Exception):
                    self._log(f"{agent_id} error: {str(result)}")
                elif isinstance(result, AgentResponse) and result.success:
                    all_sources.extend(result.sources)

                    # Collect results with query context
                    if agent_id == "sql" and agent_context.sql_results:
                        # Add query context to results
                        sql_results_with_context.append({
                            "_query_context": task_desc,
                            "_result_count": len(agent_context.sql_results),
                            "_results": agent_context.sql_results
                        })
                    if agent_id == "pinecone" and agent_context.pinecone_results:
                        if context.pinecone_results is None:
                            context.pinecone_results = []
                        context.pinecone_results.extend(agent_context.pinecone_results)
                    if agent_id == "web_search" and agent_context.web_results:
                        if context.web_results is None:
                            context.web_results = []
                        context.web_results.extend(agent_context.web_results)

        # Store SQL results with context for reviewer
        if sql_results_with_context:
            context.sql_results = sql_results_with_context

    async def _execute_agents_sequential(
        self,
        context: AgentContext,
        agent_plan: List[Dict],
        agents_used: List[str],
        all_sources: List[Dict]
    ):
        """Execute agents sequentially"""
        import copy

        sql_results_with_context = []

        for plan_item in agent_plan:
            tool_name = plan_item["agent"]
            args = plan_item["arguments"]

            # Extract agent ID from tool name
            agent_id = self._extract_agent_id_from_tool(tool_name)
            if not agent_id:
                continue

            agent = self.get_agent(agent_id)
            if not agent:
                self._log(f"Agent not found: {agent_id}")
                continue

            # Skip web search if disabled
            if agent_id == "web_search" and not self.config.enable_web_search:
                continue

            # Create a copy of context for this agent
            agent_context = copy.copy(context)
            agent_context.metadata = context.metadata.copy()

            # Set up context metadata based on agent type
            self._setup_agent_context(agent_context, agent_id, args)

            # Store task description for context
            task_desc = args.get("task_description", args.get("search_query", ""))

            result = await agent.execute(agent_context)
            agents_used.append(agent_id)

            if result.success:
                all_sources.extend(result.sources)

                # Collect results with query context
                if agent_id == "sql" and agent_context.sql_results:
                    sql_results_with_context.append({
                        "_query_context": task_desc,
                        "_result_count": len(agent_context.sql_results),
                        "_results": agent_context.sql_results
                    })
                if agent_id == "pinecone" and agent_context.pinecone_results:
                    if context.pinecone_results is None:
                        context.pinecone_results = []
                    context.pinecone_results.extend(agent_context.pinecone_results)
                if agent_id == "web_search" and agent_context.web_results:
                    if context.web_results is None:
                        context.web_results = []
                    context.web_results.extend(agent_context.web_results)

        # Store SQL results with context for reviewer
        if sql_results_with_context:
            context.sql_results = sql_results_with_context

    def _setup_agent_context(
        self,
        context: AgentContext,
        agent_id: str,
        args: Dict[str, Any]
    ):
        """Set up context metadata for a specific agent"""
        if agent_id == "sql":
            context.metadata["sql_task"] = args.get("task_description", context.user_query)
            context.metadata["sql_filters"] = args.get("filters", {})

        elif agent_id == "pinecone":
            context.metadata["pinecone_query"] = args.get("search_query", context.user_query)
            context.metadata["pinecone_namespace"] = args.get("namespace", "both")
            context.metadata["pinecone_top_k"] = args.get("top_k", 5)

        elif agent_id == "web_search":
            context.metadata["web_query"] = args.get("search_query", context.user_query)
            context.metadata["web_search_depth"] = args.get("search_depth", "basic")
            context.metadata["web_max_results"] = args.get("max_results", 3)

    async def _load_conversation_context(self, thread_key: str) -> List[Dict]:
        """Load conversation history from Redis"""
        if not self.redis_client:
            return []

        try:
            history_key = f"history:{thread_key}"
            history_data = await self.redis_client.get(history_key)

            if history_data:
                import json
                return json.loads(history_data)
        except Exception as e:
            self._log(f"Redis load error: {str(e)}")

        return []

    async def _store_conversation_context(
        self,
        thread_key: str,
        user_message: str,
        ai_response: str
    ):
        """Store conversation turn in Redis"""
        if not self.redis_client:
            return

        try:
            import json

            history_key = f"history:{thread_key}"

            # Load existing history
            history_data = await self.redis_client.get(history_key)
            history = json.loads(history_data) if history_data else []

            # Add new turn
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": ai_response[:1000]})  # Truncate long responses

            # Keep last 10 turns (20 messages)
            if len(history) > 20:
                history = history[-20:]

            # Store with TTL from config
            ttl_hours = rag_config.conversation_ttl_hours if hasattr(rag_config, 'conversation_ttl_hours') else 24
            await self.redis_client.setex(
                history_key,
                ttl_hours * 3600,
                json.dumps(history, ensure_ascii=False)
            )

        except Exception as e:
            self._log(f"Redis store error: {str(e)}")


# Factory function for easy instantiation
def create_agent_system(
    verbose: bool = None,
    enable_web_search: bool = None,
    parallel_execution: bool = None,
    redis_client=None
) -> AgentSystem:
    """
    Create an agent system with default configuration from environment.

    Args:
        verbose: Enable detailed logging (default: from config)
        enable_web_search: Enable Tavily web search (default: from config)
        parallel_execution: Execute independent agents in parallel (default: from config)
        redis_client: Optional Redis client for conversation persistence

    Returns:
        Configured AgentSystem instance
    """
    # Use config values as defaults
    if verbose is None:
        verbose = rag_config.agent_verbose
    if enable_web_search is None:
        enable_web_search = rag_config.enable_web_search
    if parallel_execution is None:
        parallel_execution = rag_config.agent_parallel_execution

    system_config = AgentSystemConfig(
        orchestrator_model=rag_config.response_model,
        sql_model=rag_config.chunking_model or "gpt-4o-mini",
        reviewer_model=rag_config.response_model,
        orchestrator_reasoning=rag_config.response_reasoning or "medium",
        reviewer_reasoning=rag_config.response_reasoning or "medium",
        enable_web_search=enable_web_search,
        parallel_execution=parallel_execution,
        verbose=verbose
    )

    client = AsyncOpenAI(api_key=rag_config.openai_api_key)

    return AgentSystem(
        openai_client=client,
        config=system_config,
        redis_client=redis_client
    )
