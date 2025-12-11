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

from rag.search import RAGSearch


async def test_query(query: str):
    print("=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)

    rag = RAGSearch()
    result = await rag.search_and_generate(
        query=query,
        user_id='test_user',
        user_name='Test User'
    )

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Agents Used: {result.get('agents_used')}")
    print(f"Execution Time: {result.get('execution_time_ms')}ms")
    print(f"Query Type: {result.get('query_type')}")
    print(f"\nRESPONSE:\n{result.get('response')}")

    if result.get("sources"):
        print(f"\nSOURCES: {len(result['sources'])}")
        for src in result["sources"][:3]:
            print(f" - {src.get('title')} ({src.get('namespace')})")


if __name__ == "__main__":
    query = "Wie viele Bagger haben wir im Bestand?"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

    asyncio.run(test_query(query))
