#!/usr/bin/env python3
"""
SEMA Integration Example
Shows how to integrate the query orchestrator into your existing app

This is a minimal example - adapt to your framework (FastAPI, Flask, etc.)
"""

import os
from openai import OpenAI
from pinecone import Pinecone

# Import the orchestrator
from query_orchestrator import QueryOrchestrator

# ============================================================
# CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "sema"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "sema-geraete")


# ============================================================
# INITIALIZE CLIENTS
# ============================================================

def create_orchestrator() -> QueryOrchestrator:
    """Create and configure the query orchestrator"""
    
    # OpenAI client (required)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Pinecone index (optional - for semantic search)
    pinecone_index = None
    if PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    
    # Create orchestrator
    orchestrator = QueryOrchestrator(
        db_config=DB_CONFIG,
        openai_client=openai_client,
        pinecone_index=pinecone_index,
        verbose=False  # Set True for debugging
    )
    
    return orchestrator


# ============================================================
# YOUR EXISTING CHAT FUNCTION - MODIFIED
# ============================================================

# Initialize once at startup
orchestrator = create_orchestrator()


def handle_chat(user_message: str) -> str:
    """
    Your existing chat handler - now with smart query routing
    
    This replaces your current RAG logic that only uses Pinecone
    """
    
    # The orchestrator automatically:
    # 1. Classifies the query using LLM
    # 2. Routes to PostgreSQL (counts, filters) or Pinecone (recommendations)
    # 3. Generates a natural language response
    
    result = orchestrator.query(user_message)
    
    return result.answer


def handle_chat_with_metadata(user_message: str) -> dict:
    """
    Extended version that returns metadata for debugging/logging
    """
    
    result = orchestrator.query(user_message)
    
    return {
        "answer": result.answer,
        "query_type": result.query_type.value,
        "source": result.source,  # "postgres" or "pinecone"
        "sql_query": result.sql_query,
        "result_count": len(result.raw_results) if result.raw_results else 0
    }


# ============================================================
# FASTAPI INTEGRATION EXAMPLE
# ============================================================

"""
# If you're using FastAPI, here's how to integrate:

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
orchestrator = create_orchestrator()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    query_type: str
    source: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = orchestrator.query(request.message)
    return ChatResponse(
        answer=result.answer,
        query_type=result.query_type.value,
        source=result.source
    )
"""


# ============================================================
# NEXT.JS API ROUTE EXAMPLE (Python Backend)
# ============================================================

"""
# If you're using Next.js with a Python backend (e.g., FastAPI):

# pages/api/chat.ts (Next.js frontend)
export async function POST(request: Request) {
    const { message } = await request.json();
    
    const response = await fetch('http://your-python-backend:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
    });
    
    return response.json();
}
"""


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("Testing Query Orchestrator Integration")
    print("=" * 50)
    
    test_queries = [
        # Aggregation → PostgreSQL
        "Wie viele Bagger haben wir?",
        
        # Filter → PostgreSQL  
        "Zeige alle Geräte mit Klimaanlage",
        
        # Semantic → Pinecone (or fallback to PostgreSQL)
        "Was empfiehlst du für Straßenbau?",
    ]
    
    for query in test_queries:
        print(f"\n❓ {query}")
        response = handle_chat_with_metadata(query)
        print(f"📊 Type: {response['query_type']} | Source: {response['source']}")
        print(f"💬 {response['answer'][:200]}...")
