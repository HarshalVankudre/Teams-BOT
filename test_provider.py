#!/usr/bin/env python3
"""
Test script for OpenAI LLM provider.
Run queries against the single agent.

Usage:
    # Test with default query
    python test_provider.py

    # Run specific query
    python test_provider.py --query "Welche Kettenfertiger gibt es?"

    # Interactive mode
    python test_provider.py -i
"""
import asyncio
import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()


async def test_query(query: str, verbose: bool = True):
    """Test the agent with a query."""
    from rag.config import RAGConfig
    config = RAGConfig()

    # Validate config
    errors = config.validate()
    if errors:
        print("\n[ERROR] Configuration errors:")
        for err in errors:
            print(f"  - {err}")
        return None

    # Import and create agent
    from rag.single_agent import create_single_agent
    from rag.vector_store import PineconeStore

    print(f"\n{'='*60}")
    print(f"Model: {config.response_model}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    try:
        pinecone = PineconeStore()
        agent = create_single_agent(verbose=verbose, pinecone_service=pinecone)

        start_time = time.time()
        result = await agent.process(
            user_query=query,
            conversation_history=[],
            thread_key="test_provider"
        )
        elapsed = time.time() - start_time

        print(f"\n[Response] ({elapsed:.2f}s, {result.execution_time_ms}ms internal)")
        print("-" * 40)
        print(result.response)
        print("-" * 40)
        print(f"Tools used: {result.tools_used}")
        print(f"SQL results: {result.sql_results_count}")
        print(f"Sources: {len(result.sources)}")
        print(f"Success: {result.success}")
        if result.error:
            print(f"Error: {result.error}")

        return result

    except Exception as e:
        print(f"\n[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def interactive_mode(verbose: bool = True):
    """Interactive query mode."""
    print("\nInteractive mode - OpenAI Provider")
    print("Type 'quit' or 'exit' to stop")
    print("-" * 40)

    while True:
        try:
            query = input("\nQuery> ").strip()
            if not query:
                continue
            if query.lower() in ["quit", "exit", "q"]:
                break

            await test_query(query, verbose=verbose)

        except KeyboardInterrupt:
            print("\nInterrupted")
            break
        except EOFError:
            break

    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="Test OpenAI LLM provider")
    parser.add_argument("--query", "-q", type=str,
                        default="Welche Mietmaschinen von Bomag gibt es?",
                        help="Query to test")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive query mode")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    if args.interactive:
        asyncio.run(interactive_mode(verbose=args.verbose))
    else:
        asyncio.run(test_query(args.query, verbose=args.verbose))


if __name__ == "__main__":
    main()
