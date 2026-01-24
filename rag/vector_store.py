"""
Pinecone Vector Store Integration
Uses the Pinecone cloud database for vector storage and retrieval
"""
import time
from typing import List, Dict, Any, Optional
import pinecone
from .config import config
from .embeddings import EmbeddingService


class PineconeStore:
    """
    Pinecone vector database integration for storing and retrieving document chunks.
    Uses OpenAI text-embedding-3-large for embeddings.
    """

    def __init__(self):
        self.pc = pinecone.Pinecone(api_key=config.pinecone_api_key)
        self.index = self.pc.Index(host=config.pinecone_host)
        self.namespace = config.pinecone_namespace
        self.embedding_service = EmbeddingService()

    async def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Upsert document chunks into Pinecone.

        Args:
            chunks: List of chunk dictionaries with content and metadata
            batch_size: Number of vectors to upsert per batch

        Returns:
            Dictionary with upsert statistics
        """
        if not chunks:
            return {"upserted_count": 0}

        # Extract content for embedding
        contents = [chunk["content"] for chunk in chunks]

        # Generate embeddings
        print(f"Generating embeddings for {len(contents)} chunks...")
        embeddings = await self.embedding_service.embed_texts(contents)

        # Prepare vectors for upsert
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            # Prepare metadata (Pinecone has limits on metadata size)
            metadata = self._prepare_metadata(chunk)

            vectors.append({
                "id": chunk["id"],
                "values": embedding,
                "metadata": metadata
            })

        # Upsert in batches
        total_upserted = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)
            total_upserted += len(batch)
            print(f"Upserted batch {i // batch_size + 1}: {len(batch)} vectors")

        return {"upserted_count": total_upserted}

    async def search(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        search_all_namespaces: bool = False  # Only search docs namespace for speed
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using a text query.

        Args:
            query: The search query text
            top_k: Number of results to return
            filters: Metadata filters to apply
            include_metadata: Whether to include metadata in results
            search_all_namespaces: If True, search both documents and machinery namespaces (slower)

        Returns:
            List of matching chunks with scores
        """
        search_start = time.time()
        top_k = top_k or config.search_top_k

        # Generate query embedding
        embed_start = time.time()
        query_embedding = await self.embedding_service.embed_query(query)
        embed_ms = (time.time() - embed_start) * 1000
        print(f"⏱️  [pinecone:embedding] {embed_ms:.0f}ms")

        # Build filter if provided
        pinecone_filter = self._build_filter(filters) if filters else None

        all_results = []

        # Determine which namespaces to search
        namespaces_to_search = [self.namespace]
        if search_all_namespaces:
            machinery_ns = config.pinecone_machinery_namespace
            if machinery_ns and machinery_ns != self.namespace:
                namespaces_to_search.append(machinery_ns)

        # Search each namespace
        for ns in namespaces_to_search:
            try:
                query_start = time.time()
                results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    namespace=ns,
                    include_metadata=include_metadata,
                    filter=pinecone_filter
                )
                query_ms = (time.time() - query_start) * 1000
                print(f"⏱️  [pinecone:query:{ns}] {query_ms:.0f}ms ({len(results.matches)} matches)")

                # Format results from this namespace
                for match in results.matches:
                    result = {
                        "id": match.id,
                        "score": match.score,
                        "namespace": ns,
                    }
                    if include_metadata and match.metadata:
                        result["metadata"] = match.metadata
                        # Handle different metadata field names per namespace
                        if ns == config.pinecone_machinery_namespace:
                            result["content"] = match.metadata.get("inhalt", match.metadata.get("content", ""))
                            result["title"] = match.metadata.get("titel", match.metadata.get("title", ""))
                            result["source_file"] = "machinery-database"
                        else:
                            result["content"] = match.metadata.get("content", "")
                            result["title"] = match.metadata.get("title", "")
                            result["source_file"] = match.metadata.get("source_file", "")
                    all_results.append(result)
            except Exception as e:
                print(f"[PineconeStore] Error searching namespace '{ns}': {e}")
                # Continue with other namespaces instead of failing completely

        # Sort by score descending and limit to top_k
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        total_ms = (time.time() - search_start) * 1000
        print(f"⏱️  [pinecone:search:total] {total_ms:.0f}ms (embed={embed_ms:.0f}ms)")

        return all_results[:top_k]

    async def search_by_category(
        self,
        query: str,
        category: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search within a specific category"""
        return await self.search(
            query=query,
            top_k=top_k,
            filters={"category": {"$eq": category}}
        )

    async def search_by_source(
        self,
        query: str,
        source_file: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search within a specific source file"""
        return await self.search(
            query=query,
            top_k=top_k,
            filters={"source_file": {"$eq": source_file}}
        )

    async def search_high_importance(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search only high importance chunks"""
        return await self.search(
            query=query,
            top_k=top_k,
            filters={"importance": {"$eq": "high"}}
        )

    async def delete_by_source(self, source_file: str) -> Dict[str, Any]:
        """Delete all chunks from a specific source file"""
        # Note: Pinecone delete by metadata requires specific index configuration
        # This is a placeholder - actual implementation depends on your Pinecone setup
        try:
            self.index.delete(
                filter={"source_file": {"$eq": source_file}},
                namespace=self.namespace
            )
            return {"status": "deleted", "source_file": source_file}
        except Exception as e:
            print(f"Delete error: {e}")
            return {"status": "error", "message": str(e)}

    async def delete_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """Delete chunks by their IDs"""
        try:
            self.index.delete(ids=ids, namespace=self.namespace)
            return {"status": "deleted", "count": len(ids)}
        except Exception as e:
            print(f"Delete error: {e}")
            return {"status": "error", "message": str(e)}

    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "namespaces": stats.namespaces,
                "dimension": stats.dimension
            }
        except Exception as e:
            return {"error": str(e)}

    def _prepare_metadata(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare metadata for Pinecone.
        Pinecone has limits on metadata size, so we need to be selective.
        """
        # Store content truncated for retrieval display
        content = chunk.get("content", "")
        truncated_content = content[:1000] if len(content) > 1000 else content

        metadata = {
            # Core fields
            "content": truncated_content,
            "title": chunk.get("title", "")[:200],
            "summary": chunk.get("summary", "")[:500],

            # Categorization
            "category": chunk.get("category", "other"),
            "importance": chunk.get("importance", "medium"),
            "doc_type": chunk.get("doc_type", "text"),
            "chunk_type": chunk.get("chunk_type", "semantic"),

            # Source tracking
            "source_file": chunk.get("source_file", ""),
            "chunk_index": chunk.get("chunk_index", 0),
            "total_chunks": chunk.get("total_chunks", 1),

            # Searchable text fields
            "keywords": ",".join(chunk.get("keywords", [])[:10]),
            "language": chunk.get("language", "de"),
            "target_audience": chunk.get("target_audience", "all"),

            # Machine identifiers for hybrid search
            "machine_name": chunk.get("machine_name", ""),
            "serial_number": chunk.get("serial_number", ""),
            "inventory_number": chunk.get("inventory_number", ""),
            "machine_type": chunk.get("machine_type", ""),

            # Timestamps
            "indexed_at": chunk.get("indexed_at", "")
        }

        # Add entities if present (flattened)
        entities = chunk.get("entities", {})
        if entities:
            metadata["entities_people"] = ",".join(entities.get("people", [])[:5])
            metadata["entities_departments"] = ",".join(entities.get("departments", [])[:5])

        return metadata

    def _build_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build Pinecone filter from simple filter dict"""
        pinecone_filter = {}

        for key, value in filters.items():
            if isinstance(value, dict):
                # Already in Pinecone format
                pinecone_filter[key] = value
            elif isinstance(value, list):
                # IN filter
                pinecone_filter[key] = {"$in": value}
            else:
                # Equality filter
                pinecone_filter[key] = {"$eq": value}

        return pinecone_filter
