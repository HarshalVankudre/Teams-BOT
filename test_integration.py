"""Integration test for semantic column resolution."""
import asyncio
import sys

async def test_integration():
    print("=" * 60)
    print("INTEGRATION TEST - Semantic Column Resolution")
    print("=" * 60)

    from rag.single_agent import create_single_agent
    from rag.vector_store import PineconeStore

    # Initialize agent
    print("\nInitializing agent...")
    try:
        pinecone = PineconeStore()
        pinecone_available = pinecone.available
    except Exception as e:
        print(f"Pinecone not available: {e}")
        pinecone = None
        pinecone_available = False

    agent = create_single_agent(verbose=True, pinecone_service=pinecone)
    print(f"Agent initialized with {agent.model}")
    print(f"PostgreSQL available: {agent.postgres.available}")
    print(f"Pinecone available: {pinecone_available}")

    if not agent.postgres.available:
        print("\nERROR: PostgreSQL not available, cannot run integration tests")
        return False

    # Test queries with thread context
    queries = [
        "Ich brauche einen Bagger mit einer Grabtiefe von 3 metern",
        "Ich möchte die Maschine gerne mieten. Empfiehlst du mir Kette oder Mobil?",
        "Ich muss damit in einen Garten, durchfahrtsbreite 3m",
    ]

    thread_key = "integration_test"
    history = []
    all_passed = True

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print("=" * 60)

        try:
            result = await agent.process(
                query,
                conversation_history=history,
                thread_key=thread_key
            )

            print(f"\nTools used: {result.tools_used}")
            print(f"SQL results: {result.sql_results_count}")
            print(f"Tokens: {result.total_tokens}")
            print(f"\nResponse preview:")
            print("-" * 40)
            # Show first 500 chars of response
            preview = result.response[:500] + "..." if len(result.response) > 500 else result.response
            print(preview)

            # Add to history for next query
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": result.response})

            # Check for issues
            issues = []
            if "weiß nicht" in result.response.lower() or "kann nicht" in result.response.lower():
                issues.append("Response indicates uncertainty")
            if not result.tools_used and "execute_sql" not in result.tools_used:
                if i == 1:  # First query should definitely use SQL
                    issues.append("First query did not use SQL")

            if issues:
                print(f"\n[WARNINGS] {issues}")
            else:
                print("\n[OK] Query processed successfully")

        except Exception as e:
            print(f"\n[ERROR] Query {i} failed: {e}")
            all_passed = False

    print(f"\n{'='*60}")
    print("INTEGRATION TEST COMPLETE")
    print("=" * 60)

    return all_passed

if __name__ == "__main__":
    success = asyncio.run(test_integration())
    sys.exit(0 if success else 1)
