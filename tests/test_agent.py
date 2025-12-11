"""
Test the unified single-agent pipeline.
"""
import asyncio
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from rag.search import RAGSearch


async def test_unified_agent():
    """Run sample queries through the unified agent."""

    print("=" * 60)
    print("UNIFIED AGENT TEST")
    print("=" * 60)

    rag = RAGSearch()

    test_queries = [
        "Wie viele Bagger haben wir?",
        "Zeige mir den Caterpillar 320",
        "Welche Geräte haben Klimaanlage?",
        "Was eignet sich für eine 9m breite Straße?",
        "Beste Maschine für enge Baustellen?",
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"QUERY: {query}")
        print("=" * 60)

        try:
            result = await rag.search_and_generate(
                query=query,
                user_id="test_user",
                user_name="Test User"
            )

            print(f"\nAGENTS USED: {result.get('agents_used')}")
            print(f"QUERY TYPE: {result.get('query_type')}")
            print(f"EXECUTION TIME: {result.get('execution_time_ms')}ms")
            print(f"\nRESPONSE:\n{(result.get('response') or '')[:500]}...")

            sources = result.get("sources") or []
            if sources:
                print("\nSOURCES:")
                for src in sources[:3]:
                    print(f"  - {src.get('title')} ({src.get('namespace')})")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        async def _single():
            res = await RAGSearch().search_and_generate(query=query)
            print(res.get("response"))
        asyncio.run(_single())
    else:
        asyncio.run(test_unified_agent())
