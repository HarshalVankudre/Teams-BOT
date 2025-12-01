"""
Test the Multi-Agent System.
Verifies the orchestrator -> sub-agents -> reviewer flow works correctly.
"""
import asyncio
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from rag.agents import AgentSystem, create_agent_system


async def test_agent_system():
    """Test the multi-agent system with sample queries"""

    print("=" * 60)
    print("MULTI-AGENT SYSTEM TEST")
    print("=" * 60)

    # Create agent system with verbose logging
    agent_system = create_agent_system(verbose=True)

    # Test queries covering different agent types
    test_queries = [
        # SQL Agent queries
        ("Wie viele Bagger haben wir?", "SQL - Count"),
        ("Zeige mir den Caterpillar 320", "SQL - Lookup"),
        ("Vergleiche Kettenbagger und Mobilbagger", "SQL - Comparison"),
        ("Welche Geräte haben Klimaanlage?", "SQL - Filter"),

        # Pinecone Agent queries
        ("Was eignet sich für eine 9m breite Straße?", "Pinecone - Recommendation"),
        ("Beste Maschine für enge Baustellen?", "Pinecone - Scenario"),
    ]

    for query, expected_type in test_queries:
        print(f"\n{'=' * 60}")
        print(f"QUERY: {query}")
        print(f"EXPECTED: {expected_type}")
        print("=" * 60)

        try:
            result = await agent_system.process(
                user_query=query,
                user_id="test_user",
                user_name="Test User"
            )

            print(f"\nSUCCESS: {result.success}")
            print(f"AGENTS USED: {result.agents_used}")
            print(f"QUERY INTENT: {result.query_intent}")
            print(f"EXECUTION TIME: {result.execution_time_ms}ms")
            print(f"\nRESPONSE:\n{result.response[:500]}...")

            if result.metadata:
                print(f"\nMETADATA:")
                for key, value in result.metadata.items():
                    print(f"  {key}: {value}")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

        print()


async def test_single_query(query: str):
    """Test a single query for debugging"""
    print(f"Testing: {query}")
    print("=" * 60)

    agent_system = create_agent_system(verbose=True)
    result = await agent_system.process(user_query=query)

    print(f"\nAgents: {result.agents_used}")
    print(f"Intent: {result.query_intent}")
    print(f"\nResponse:\n{result.response}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Test single query from command line
        query = " ".join(sys.argv[1:])
        asyncio.run(test_single_query(query))
    else:
        # Run full test suite
        asyncio.run(test_agent_system())
