#!/usr/bin/env python3
"""
Document Indexing Script
Index documents into Pinecone vector database

Usage:
    python scripts/index_documents.py --file path/to/document.pdf
    python scripts/index_documents.py --directory path/to/docs --recursive
    python scripts/index_documents.py --stats
"""
import asyncio
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import DocumentProcessor, PineconeStore, RAGSearch


async def index_file(file_path: str, context: str = None):
    """Index a single file"""
    processor = DocumentProcessor()
    result = await processor.process_and_index(file_path, additional_context=context)

    if result.get("status") == "success":
        print(f"\n✅ Successfully indexed: {result['file']}")
        print(f"   Type: {result['type']}")
        print(f"   Chunks created: {result['chunks_created']}")
        print(f"   Chunks indexed: {result['chunks_indexed']}")
    else:
        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")

    return result


async def index_directory(directory_path: str, recursive: bool = True):
    """Index all files in a directory"""
    processor = DocumentProcessor()
    result = await processor.process_directory(directory_path, recursive=recursive)

    print(f"\n📁 Directory indexing complete")
    print(f"   Total files: {result['total_files']}")
    print(f"   Successful: {result['successful']}")
    print(f"   Failed: {result['failed']}")

    return result


async def show_stats():
    """Show Pinecone index statistics"""
    store = PineconeStore()
    stats = await store.get_stats()

    print("\n📊 Pinecone Index Statistics")
    print(f"   Total vectors: {stats.get('total_vectors', 'N/A')}")
    print(f"   Dimension: {stats.get('dimension', 'N/A')}")

    namespaces = stats.get('namespaces', {})
    if namespaces:
        print("   Namespaces:")
        for ns, info in namespaces.items():
            print(f"      - {ns}: {info.get('vector_count', 0)} vectors")


async def test_search(query: str, top_k: int = 5):
    """Test search functionality"""
    search = RAGSearch()
    results = await search.search(query, top_k=top_k)

    print(f"\n🔍 Search results for: '{query}'")
    print(f"   Found {len(results)} results\n")

    for i, result in enumerate(results):
        metadata = result.get("metadata", {})
        print(f"{i + 1}. {metadata.get('title', 'Untitled')}")
        print(f"   Score: {result.get('score', 0):.2%}")
        print(f"   Source: {metadata.get('source_file', 'Unknown')}")
        print(f"   Category: {metadata.get('category', 'N/A')}")
        print()


async def test_rag(query: str):
    """Test full RAG pipeline"""
    search = RAGSearch()
    result = await search.search_and_generate(query)

    print(f"\n🤖 RAG Response for: '{query}'")
    print("-" * 50)
    print(result["response"])
    print("-" * 50)
    print(f"\nSources used: {result['chunks_used']}")

    for source in result["sources"][:3]:
        print(f"  - {source['title']} ({source['source_file']}) [{source['score']:.2%}]")


async def delete_source(source_file: str):
    """Delete all chunks from a source file"""
    store = PineconeStore()
    result = await store.delete_by_source(source_file)

    if result.get("status") == "deleted":
        print(f"\n🗑️ Deleted chunks from: {source_file}")
    else:
        print(f"\n❌ Error: {result.get('message', 'Unknown error')}")


def main():
    parser = argparse.ArgumentParser(
        description="Index documents into Pinecone vector database"
    )

    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to a single file to index"
    )
    parser.add_argument(
        "--directory", "-d",
        type=str,
        help="Path to a directory to index"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively index subdirectories"
    )
    parser.add_argument(
        "--context", "-c",
        type=str,
        help="Additional context about the document(s)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show Pinecone index statistics"
    )
    parser.add_argument(
        "--search", "-s",
        type=str,
        help="Test search with a query"
    )
    parser.add_argument(
        "--rag",
        type=str,
        help="Test full RAG pipeline with a question"
    )
    parser.add_argument(
        "--delete",
        type=str,
        help="Delete chunks from a source file"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results for search (default: 5)"
    )

    args = parser.parse_args()

    # Run the appropriate command
    if args.file:
        asyncio.run(index_file(args.file, args.context))
    elif args.directory:
        asyncio.run(index_directory(args.directory, args.recursive))
    elif args.stats:
        asyncio.run(show_stats())
    elif args.search:
        asyncio.run(test_search(args.search, args.top_k))
    elif args.rag:
        asyncio.run(test_rag(args.rag))
    elif args.delete:
        asyncio.run(delete_source(args.delete))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
