<div align="center">

# Teams Equipment Assistant

### AI-Powered Microsoft Teams Bot for Construction Equipment Queries

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Pinecone](https://img.shields.io/badge/Pinecone-000000?style=for-the-badge&logo=pinecone&logoColor=white)](https://www.pinecone.io/)
[![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io/)

*A production-grade RAG system combining structured database queries with semantic document search*

</div>

---

## Overview

A Microsoft Teams bot backend that serves as an intelligent assistant for **RÃœKO Baumaschinen**, a construction equipment rental company. The bot answers German-language queries about machinery inventory by combining:

- **Structured Data** â€” SQL queries across 2,400+ equipment records with 100+ properties
- **Unstructured Data** â€” Semantic search through technical manuals and documentation
- **Conversational Memory** â€” Multi-turn conversations with intelligent follow-ups

---

## Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                      Microsoft Teams                             â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                           â”‚
                           â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                    FastAPI Backend                               â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”‚
â”‚  â”‚ Bot Framework â”‚  â”‚ OAuth Cache  â”‚  â”‚ Typing Indicators â”‚    â”‚
â”‚  â”‚    Webhook    â”‚  â”‚ (Thread-safe)â”‚  â”‚   (Real-time)     â”‚    â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
           â”‚
           â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                   RAG Orchestrator                               â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”   â”‚
â”‚  â”‚              LangGraph ReAct Agent                       â”‚   â”‚
â”‚  â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”   â”‚   â”‚
â”‚  â”‚  â”‚ execute_sql â”‚ â”‚  lookup_    â”‚ â”‚ search_documentsâ”‚   â”‚   â”‚
â”‚  â”‚  â”‚             â”‚ â”‚  equipment  â”‚ â”‚                 â”‚   â”‚   â”‚
â”‚  â”‚  â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”˜ â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”˜ â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜   â”‚   â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
             â”‚               â”‚                 â”‚
             â–¼               â–¼                 â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚   PostgreSQL   â”‚  â”‚    Redis     â”‚  â”‚     Pinecone     â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”‚  â”‚ â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”‚  â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”‚
â”‚  â”‚ 2,400+   â”‚  â”‚  â”‚ â”‚ Session  â”‚ â”‚  â”‚  â”‚ Equipment  â”‚  â”‚
â”‚  â”‚Equipment â”‚  â”‚  â”‚ â”‚ History  â”‚ â”‚  â”‚  â”‚  Manuals   â”‚  â”‚
â”‚  â”‚ Records  â”‚  â”‚  â”‚ â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â”‚  â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## Key Features

### Autonomous AI Agent
Built on **LangGraph's ReAct pattern**, the agent autonomously decides which tools to use based on query context. No rigid decision trees â€” the LLM naturally reasons about whether to query the database, search documents, or look up specific equipment.

### Hybrid Data Retrieval
Combines structured PostgreSQL queries with semantic Pinecone search. Equipment specs come from the database; operating manuals and technical guides come from vector search. Results are synthesized into coherent German responses.

### Conversation Intelligence
Redis-backed per-user conversation history enables natural follow-ups:
> **User:** "Zeige Kettenfertiger mit 3m Einbaubreite"
> **Bot:** *Returns 5 machines*
> **User:** "Welche davon sind zur Miete?"
> **Bot:** *Filters previous results by rental status*

### Security-First SQL
Every query passes through validation:
- Only SELECT statements allowed
- Dangerous keywords blocked (DROP, DELETE, ALTER)
- Results limited to 50 rows
- Identifier validation prevents injection

### Provider Flexibility
The runtime uses **Google Gemini** for advisory, LangGraph retrieval, and grounded fallback responses, while **OpenAI embeddings** remain in place for the existing Pinecone index.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **API Framework** | FastAPI with async/await |
| **Agent Framework** | LangGraph ReAct |
| **LLM Providers** | Google Gemini for chat/runtime, OpenAI embeddings for Pinecone |
| **Structured Data** | PostgreSQL with connection pooling |
| **Vector Search** | Pinecone + text-embedding-3-large |
| **Session State** | Redis with 24-hour TTL |
| **Bot Platform** | Microsoft Bot Framework |
| **Monitoring** | Flask Admin Dashboard |

---

## Technical Highlights

**German Number Handling**
Equipment specs use German decimal notation (1,5m). The system automatically handles CAST/REPLACE conversions for numeric comparisons in SQL.

**Concurrent Search**
Parallel Pinecone namespace queries using asyncio.gather() â€” documents and machinery data searched simultaneously.

**Graceful Degradation**
If Gemini advisory routing or LangGraph retrieval fails, the bot falls back to direct Pinecone search. The system still responds.

**Real-time UX**
Continuous typing indicators every 2.5 seconds keep Teams showing "Bot is typing..." during long reasoning operations.

**Thread Isolation**
Unique thread keys (`{user_id}:{conversation_id}`) prevent conversation bleed in group chats.

---

## Project Structure

```
teams-bot-dev/
|-- app.py                # FastAPI entry point, Teams webhook
|-- cli_tester.py         # Local testing without Teams
|-- rag/
|   |-- langgraph_agent.py # Thin LangGraph runtime wrapper
|   |-- langgraph_tools.py # Safe SQL and document tools
|   |-- prompts.py         # Centralized prompts and bot copy
|   |-- search.py          # RAG orchestrator
|   |-- postgres.py        # SQL safety wrapper
|   |-- vector_store.py    # Pinecone integration
|   `-- config.py          # Centralized configuration
`-- admin_dashboard/      # Flask monitoring UI
```

---

## Sample Interaction

```
User: Wie viele Bomag Walzen haben wir?

Agent: [Calling execute_sql]
       SELECT COUNT(*) FROM equipment_matrix
       WHERE hersteller_name ILIKE '%bomag%'
       AND geraetegruppe_name ILIKE '%walze%'

Bot:  Wir haben 47 Bomag Walzen im Bestand.
      Davon sind 32 zur Miete und 15 zum Verkauf verfÃ¼gbar.

      MÃ¶chten Sie Details zu bestimmten Modellen?
```

---

<div align="center">

**Built with modern AI engineering practices**

*LangGraph Agents â€¢ RAG Architecture â€¢ Production-Grade Error Handling*

</div>
