# RAG Module Documentation

This document describes each Python file in the `rag/` folder.

---

## Core Components

### config.py
Central configuration management for the RAG pipeline. Loads all settings from environment variables including LLM provider configurations (OpenAI, Cerebras, Groq), API keys, model selection, and feature toggles for planning, SQL verification, and reasoning tools. All components reference this singleton config.

### search.py
Main orchestration entry point that routes queries through available agents. Prioritizes LangGraph ReAct agent, falls back to CleanSingleAgent/SingleAgent on failure, and ultimately to direct Pinecone search. Manages conversation history via Redis and handles response formatting for Teams.

### langgraph_agent.py
LangGraph ReAct agent implementation using `create_react_agent()`. Provides tools for SQL execution, document search, column discovery, and database exploration. Features automatic tool looping, Redis checkpointing for conversation state, and improved numeric handling for German text columns.

### single_agent_clean.py
Simplified AI agent (recommended default). Fixes hallucination issues with: single clear system prompt, natural tool selection (tool_choice="auto"), trusted tool results without aggressive post-processing, and simple thread state for follow-ups. Clean architecture under 750 lines.

### single_agent.py
Legacy full-featured agent with 2700+ lines. Includes query planning, SQL verification, answer guards, reasoning tools, and extensive post-processing. Over-engineered and less effective than simplified alternatives. Use only if `USE_CLEAN_AGENT=false`.

---

## Database & Storage

### postgres.py
PostgreSQL interface with safety guardrails. Provides read-only access with query validation (SELECT-only), connection pooling, and schema introspection. Validates queries via `prepare_readonly_sql()` to prevent SQL injection and dangerous operations. Equipment table: `public.equipment_matrix`.

### vector_store.py
Pinecone vector database integration for document storage and retrieval. Handles embeddings via EmbeddingService, supports batch upserts, and provides semantic search across equipment documentation. Uses configurable namespaces for documents and machinery data.

### embeddings.py
OpenAI embeddings service using text-embedding-3-large model. Implements automatic batching respecting OpenAI limits (2048 texts per batch, 250K tokens per request). Provides `embed_text()` for single queries and `embed_texts()` for batch processing.

---

## Schema & Columns

### schema.py
Static schema definition documenting the SEMA equipment database structure (~2400 records). Defines core columns (id, bezeichnung, hersteller_name, geraetegruppe_name, verwendung_code) and equipment categories. Property columns are loaded dynamically by ColumnCatalog.

### schema_linker.py
Semantic column resolution for text-to-SQL conversion. Maps user queries to relevant database columns using semantic retrieval. Provides reduced-schema approach to improve LLM token efficiency instead of sending full schema. Optional vector-based retrieval via Pinecone.

### column_catalog.py
Database column discovery and caching service. Loads all columns at startup with rich metadata (names, units, data types, descriptions). Computes statistics (null ratios, distinct values) and caches globally. Enables AI to understand which columns contain useful data.

### value_index.py
Categorical column value matching system. Indexes distinct values in columns enabling fuzzy matching to correct user input variations (e.g., "Bomag" -> "BOMAG"). Supports exact, case-insensitive, fuzzy, and partial matching strategies.

---

## SQL Safety & Validation

### sql_guard.py
SQL intent extraction and validation layer. Extracts user intent (count, filter, group, followup, document) from queries using regex patterns. Detects dangerous operations, context references for follow-ups, and supports German/English intent recognition.

### sql_validator.py
Semantic SQL validation against reduced schema. Validates generated SQL queries and extracts predicate tuples (column, operator, value) from AST. Complements sql_guard.py with deeper semantic correctness checking before execution.

### sql_verifier.py
Self-verification system for SQL query correction. Pre-execution verification catching common mistakes (wrong column names, missing quotes, category filtering errors). Suggests corrections and indicates when retry is needed for improved success rate.

---

## Response Quality

### answer_guard.py
Response quality guardrails validating outputs for quality, factuality, and safety. Detects small-talk, prompt injection attempts, secrets exposure, and hallucinations. Ensures grounded answers with scoring and improvement suggestions for concise German responses.

### planning.py
Query planning service for complex queries. Creates execution plans identifying complexity level, required steps, tool dependencies, and needed context. Optional feature helping agents think systematically before executing multi-step queries.

### reasoning_tools.py
Calculation and comparison tools for complex operations beyond simple data retrieval. Provides CalculationResult, ComparisonResult, and AggregationResult with breakdown explanations. Useful for queries requiring arithmetic or data analysis.

---

## Context & Memory

### context_manager.py
Conversation context management across turns. Tracks thread state, follow-up type detection, active filters, and mentioned entities. Injects rich context into prompts maintaining conversational continuity for multi-turn equipment queries.

### result_cache.py
Query result caching for follow-up questions. Caches SQL and Pinecone results in context to avoid redundant tool calls. Stores results with FIFO eviction and formats cached data as context summaries for prompt injection.

### cache.py
Redis caching utilities with TTL and automatic key generation. Provides `cached()` decorator for function results, gracefully falls back when Redis unavailable. Supports async functions with hashlib for handling long cache keys.

---

## Learning & Feedback

### feedback.py
Conversation logging to separate database for analysis. Stores conversations and user feedback in optional PostgreSQL database. Configurable via FEEDBACK_* environment variables, falls back to main POSTGRES config if not specified.

### learned_rules.py
Extracts and injects learned rules from user feedback. Automatically identifies actionable rules from feedback and injects into system prompt for continuous improvement. Categories: output_format, data_display, behavior rules.

### alias_learner.py
Learns column and value mappings from successful queries. Part of Phase 5 semantic schema linking, stores learned aliases in admin database with confidence scoring and usage tracking. Improves query understanding over time.

---

## Logging & Admin

### logger.py
Centralized logging configuration with correlation IDs and consistent formatting. Supports structured JSON output, debug/info/warning/error/critical levels, and context variables for request tracking across async operations.

### admin_logger.py
Admin logging service for Teams bot monitoring. Logs conversations to optional admin database for debugging and analytics. Configurable via ADMIN_* environment variables with fallback to main POSTGRES config.

---

## Module Entry

### __init__.py
RAG module public API exposing core components: RAGConfig, EmbeddingService, PineconeStore, RAGSearch, and Phase 1-5 semantic schema linking components (SchemaLinker, SQLValidator, ValueIndex, AliasLearner).
