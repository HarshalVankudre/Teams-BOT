"""Quick test for a single query"""
import asyncio
import sys
import os

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from rag.agents import create_agent_system


async def test_query(query: str):
    print("=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)

    agent_system = create_agent_system(verbose=True)

    result = await agent_system.process(
        user_query=query,
        user_id='test_user',
        user_name='Test User'
    )

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Agents Used: {result.agents_used}")
    print(f"Query Intent: {result.query_intent}")
    print(f"Execution Time: {result.execution_time_ms}ms")
    print(f"\nRESPONSE:\n{result.response}")

    if result.metadata:
        print(f"\nMETADATA: {result.metadata}")


if __name__ == "__main__":
    query = "Wie viele Bagger haben wir im Bestand?"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

    asyncio.run(test_query(query))
