"""
RAG Search - Retrieval and Response Generation
"""
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from openai import AsyncOpenAI
from .config import config
from .vector_store import PineconeStore


class RAGSearch:
    """
    RAG Search: Retrieve relevant chunks and generate responses.
    Uses Pinecone for retrieval and GPT-5.1 for generation.
    """

    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.vector_store = PineconeStore()
        self.model = config.response_model  # gpt-5.1 for answering
        self.reasoning_effort = config.response_reasoning  # low reasoning

    def detect_identifiers(self, query: str) -> Dict[str, List[str]]:
        """
        Detect machine identifiers in a query for metadata filtering.

        Patterns detected:
        - Serial numbers: JG002579, NG004113, 0096 V08100445X2
        - Inventory numbers: 300582, 20205 (5-6 digit numbers)
        - Machine types: MF 2500 CS, 300.9D, V8 X2

        Returns:
            Dict with lists of detected identifiers by type
        """
        detected = {
            "serial_numbers": [],
            "inventory_numbers": [],
            "machine_names": []
        }

        # Serial number patterns (letters + numbers, e.g., JG002579, NG004113)
        serial_patterns = [
            r'\b[A-Z]{2}\d{6}\b',  # JG002579, NG004113
            r'\b\d{4}\s*[A-Z]\d{8}[A-Z]\d?\b',  # 0096 V08100445X2
            r'\b[A-Z]{2,3}\d{5,7}\b',  # Broader pattern
        ]
        for pattern in serial_patterns:
            matches = re.findall(pattern, query.upper())
            detected["serial_numbers"].extend(matches)

        # Inventory numbers (5-6 digit standalone numbers)
        inv_pattern = r'\b\d{5,6}\b'
        inv_matches = re.findall(inv_pattern, query)
        # Filter out years (1900-2099) and other obvious non-inventory numbers
        for match in inv_matches:
            if not (1900 <= int(match) <= 2099):
                detected["inventory_numbers"].append(match)

        # Machine type patterns (letters + numbers + optional suffix)
        machine_patterns = [
            r'\b(MF\s*\d{4}\s*[A-Z]{0,3})\b',  # MF 2500 CS
            r'\b(\d{3}\.\d[A-Z]?)\b',  # 300.9D
            r'\b([A-Z]\d+\s*[A-Z]\d*)\b',  # V8 X2
            r'\b(Big-Ski|Big Ski)\b',  # Special names
        ]
        for pattern in machine_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            detected["machine_names"].extend([m.strip() for m in matches if isinstance(m, str)])

        # Deduplicate
        for key in detected:
            detected[key] = list(set(detected[key]))

        return detected

    async def smart_search(
        self,
        query: str,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Smart hybrid search that combines:
        1. Metadata filtering for exact identifier matches
        2. Semantic search for concept matching
        3. Merges and deduplicates results

        Args:
            query: The search query
            top_k: Number of results to return

        Returns:
            Combined list of matching chunks
        """
        top_k = top_k or config.search_top_k
        all_results = []
        seen_ids = set()

        # Step 1: Detect identifiers in query
        identifiers = self.detect_identifiers(query)
        has_identifiers = any(identifiers.values())

        # Step 2: If identifiers found, do metadata filter searches first
        if has_identifiers:
            print(f"Detected identifiers: {identifiers}")

            # Search by serial number
            for serial in identifiers["serial_numbers"]:
                filter_results = await self.vector_store.search(
                    query=query,
                    top_k=5,
                    filters={"serial_number": {"$eq": serial}}
                )
                for r in filter_results:
                    if r.get("id") not in seen_ids:
                        r["match_type"] = "serial_number_exact"
                        all_results.append(r)
                        seen_ids.add(r.get("id"))

            # Search by inventory number
            for inv in identifiers["inventory_numbers"]:
                filter_results = await self.vector_store.search(
                    query=query,
                    top_k=5,
                    filters={"inventory_number": {"$eq": inv}}
                )
                for r in filter_results:
                    if r.get("id") not in seen_ids:
                        r["match_type"] = "inventory_number_exact"
                        all_results.append(r)
                        seen_ids.add(r.get("id"))

            # Search by machine name (partial match via semantic, but boost exact)
            for name in identifiers["machine_names"]:
                filter_results = await self.vector_store.search(
                    query=name,  # Search specifically for this name
                    top_k=5,
                    filters=None  # Semantic search for machine name
                )
                for r in filter_results:
                    if r.get("id") not in seen_ids:
                        r["match_type"] = "machine_name_semantic"
                        all_results.append(r)
                        seen_ids.add(r.get("id"))

        # Step 3: Always do semantic search as fallback/supplement
        semantic_results = await self.vector_store.search(
            query=query,
            top_k=top_k,
            filters=None
        )
        for r in semantic_results:
            if r.get("id") not in seen_ids:
                r["match_type"] = "semantic"
                all_results.append(r)
                seen_ids.add(r.get("id"))

        # Step 4: Sort by score and return top_k
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Prioritize exact matches by boosting their position
        exact_matches = [r for r in all_results if r.get("match_type", "").endswith("_exact")]
        other_matches = [r for r in all_results if not r.get("match_type", "").endswith("_exact")]

        # Return exact matches first, then others, limited to top_k
        final_results = (exact_matches + other_matches)[:top_k]

        if has_identifiers:
            print(f"Hybrid search: {len(exact_matches)} exact matches, {len(other_matches)} semantic matches")

        return final_results

    async def search(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks.

        Args:
            query: The search query
            top_k: Number of results to return
            filters: Metadata filters

        Returns:
            List of matching chunks with scores
        """
        top_k = top_k or config.search_top_k
        return await self.vector_store.search(
            query=query,
            top_k=top_k,
            filters=filters
        )

    async def search_and_generate(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
        system_instructions: Optional[str] = None,
        previous_response_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for relevant chunks and generate a response.

        Args:
            query: The user's question
            top_k: Number of chunks to retrieve
            filters: Metadata filters
            system_instructions: Custom system prompt
            previous_response_id: Previous response ID for conversation continuity

        Returns:
            Dictionary with response, sources, and response_id
        """
        top_k = top_k or config.search_top_k
        total_start = time.time()

        # Use smart hybrid search for better identifier matching
        search_start = time.time()
        search_results = await self.smart_search(query, top_k=top_k)
        search_time = time.time() - search_start
        print(f"[TIMING] Search: {search_time:.2f}s")

        if not search_results:
            return {
                "response": "Keine relevanten Informationen in den Dokumenten gefunden.",
                "sources": [],
                "chunks_used": 0,
                "response_id": None
            }

        # Build context from search results
        context_parts = []
        sources = []

        for i, result in enumerate(search_results):
            metadata = result.get("metadata", {})
            content = metadata.get("content", "")
            title = metadata.get("title", f"Chunk {i + 1}")
            source_file = metadata.get("source_file", "Unknown")
            score = result.get("score", 0)

            context_parts.append(f"""
### Quelle {i + 1}: {title}
**Datei:** {source_file}
**Relevanz:** {score:.2%}

{content}
""")

            sources.append({
                "title": title,
                "source_file": source_file,
                "score": score,
                "category": metadata.get("category", ""),
                "chunk_id": result.get("id", "")
            })

        context = "\n---\n".join(context_parts)

        # Default system instructions
        if not system_instructions:
            system_instructions = """Du bist ein präziser Dokumentenassistent.

REGELN:
1. Beantworte Fragen NUR basierend auf dem bereitgestellten Kontext
2. Zitiere die Quellen in deiner Antwort
3. Wenn die Information nicht im Kontext ist, sage es klar
4. Strukturiere deine Antwort übersichtlich
5. Verwende die gleiche Sprache wie die Frage"""

        # Generate response - use Responses API for reasoning models, Chat API for others
        llm_start = time.time()

        if "gpt-5" in self.model:
            # Responses API with reasoning for gpt-5 models
            response = await self.client.responses.create(
                model=self.model,
                reasoning={"effort": self.reasoning_effort},
                input=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": f"""Basierend auf dem folgenden Kontext, beantworte die Frage:

## Kontext aus Dokumenten:
{context}

## Frage:
{query}

Beantworte die Frage und zitiere die relevanten Quellen."""}
                ],
                max_output_tokens=400
            )
            response_text = response.output_text
            response_id = response.id
        else:
            # Chat Completions API for non-reasoning models
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": f"""Basierend auf dem folgenden Kontext, beantworte die Frage:

## Kontext aus Dokumenten:
{context}

## Frage:
{query}

Beantworte die Frage und zitiere die relevanten Quellen."""}
                ],
                max_tokens=400,
                temperature=0.3
            )
            response_text = response.choices[0].message.content
            response_id = response.id

        llm_time = time.time() - llm_start
        total_time = time.time() - total_start
        print(f"[TIMING] LLM: {llm_time:.2f}s | Total: {total_time:.2f}s")

        return {
            "response": response_text,
            "sources": sources,
            "chunks_used": len(search_results),
            "response_id": response_id
        }

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        categories: Optional[List[str]] = None,
        importance_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search with multiple filters.

        Args:
            query: The search query
            top_k: Number of results
            categories: Filter by categories
            importance_filter: Filter by importance level

        Returns:
            List of matching chunks
        """
        filters = {}

        if categories:
            filters["category"] = {"$in": categories}

        if importance_filter:
            filters["importance"] = {"$eq": importance_filter}

        return await self.search(query, top_k=top_k, filters=filters if filters else None)

    async def get_related_chunks(
        self,
        chunk_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find chunks related to a specific chunk.

        This is useful for "see also" functionality.
        """
        # First, get the content of the source chunk
        # Note: This requires fetching by ID which Pinecone supports via query
        # For now, we'll use keyword matching from metadata

        # This is a placeholder - in production, you'd fetch the chunk
        # and use its embedding to find similar ones
        return []

    async def search_with_reranking(
        self,
        query: str,
        initial_top_k: int = 20,
        final_top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search with two-stage retrieval:
        1. Initial broad search
        2. Re-rank with GPT for better relevance

        Args:
            query: The search query
            initial_top_k: Number of results for initial search
            final_top_k: Number of results after reranking

        Returns:
            Re-ranked list of chunks
        """
        # Initial search
        initial_results = await self.search(query, top_k=initial_top_k)

        if not initial_results or len(initial_results) <= final_top_k:
            return initial_results

        # Prepare candidates for reranking
        candidates = []
        for i, result in enumerate(initial_results):
            metadata = result.get("metadata", {})
            candidates.append({
                "id": i,
                "title": metadata.get("title", ""),
                "content": metadata.get("content", "")[:500],  # Truncate for efficiency
                "original_score": result.get("score", 0)
            })

        # Use GPT to rerank
        rerank_prompt = f"""Gegeben die Frage und die Kandidaten-Dokumente, ordne die Dokumente nach Relevanz.

Frage: {query}

Kandidaten:
{chr(10).join([f"{c['id']}: {c['title']} - {c['content'][:200]}..." for c in candidates])}

Gib die IDs der {final_top_k} relevantesten Dokumente zurück, sortiert nach Relevanz (höchste zuerst).
Format: [id1, id2, id3, ...]
Nur die JSON-Liste, keine Erklärung."""

        try:
            # Build rerank params - only include reasoning for gpt-5 models
            rerank_params = {
                "model": self.model,
                "input": rerank_prompt,
                "max_output_tokens": 100
            }
            if "gpt-5" in self.model:
                rerank_params["reasoning"] = {"effort": "low"}

            response = await self.client.responses.create(**rerank_params)

            # Parse the response
            import json
            ranked_ids = json.loads(response.output_text)

            # Return reranked results
            reranked = []
            for rank, idx in enumerate(ranked_ids[:final_top_k]):
                if idx < len(initial_results):
                    result = initial_results[idx].copy()
                    result["rerank_position"] = rank + 1
                    reranked.append(result)

            return reranked

        except Exception as e:
            print(f"Reranking failed: {e}, returning original results")
            return initial_results[:final_top_k]

    def format_sources_for_display(
        self,
        sources: List[Dict[str, Any]],
        max_sources: int = 5
    ) -> str:
        """Format sources for display in Teams message"""
        if not sources:
            return ""

        lines = ["\n\n---\n**Quellen:**"]
        for i, source in enumerate(sources[:max_sources]):
            score_pct = source.get("score", 0) * 100
            lines.append(
                f"- {source.get('title', 'Unbekannt')} "
                f"({source.get('source_file', '')}) "
                f"[{score_pct:.0f}%]"
            )

        return "\n".join(lines)
