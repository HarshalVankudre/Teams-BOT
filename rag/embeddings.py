"""
OpenAI Embeddings Service using text-embedding-3-large
"""
from openai import AsyncOpenAI
from typing import List
from .config import config


class EmbeddingService:
    """Generate embeddings using OpenAI text-embedding-3-large"""

    # Max tokens per embedding request (OpenAI limit is 300k, use 250k for safety)
    MAX_TOKENS_PER_REQUEST = 250000
    # Approximate chars per token (conservative estimate)
    CHARS_PER_TOKEN = 3

    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.model = config.embedding_model
        self.dimensions = config.embedding_dimensions

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text"""
        return len(text) // self.CHARS_PER_TOKEN + 1

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
            encoding_format="float"
        )
        return response.data[0].embedding

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts with automatic batching"""
        if not texts:
            return []

        # Batch texts by token count to stay under limit
        batches = []
        current_batch = []
        current_tokens = 0

        for text in texts:
            text_tokens = self._estimate_tokens(text)

            # If single text exceeds limit, truncate it
            if text_tokens > self.MAX_TOKENS_PER_REQUEST:
                max_chars = self.MAX_TOKENS_PER_REQUEST * self.CHARS_PER_TOKEN
                text = text[:max_chars]
                text_tokens = self._estimate_tokens(text)

            # If adding this text would exceed limit, start new batch
            if current_tokens + text_tokens > self.MAX_TOKENS_PER_REQUEST:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [text]
                current_tokens = text_tokens
            else:
                current_batch.append(text)
                current_tokens += text_tokens

        # Don't forget last batch
        if current_batch:
            batches.append(current_batch)

        print(f"Embedding {len(texts)} texts in {len(batches)} batches...")

        # Process each batch
        all_embeddings = []
        for i, batch in enumerate(batches):
            print(f"  Batch {i+1}/{len(batches)}: {len(batch)} texts...")
            response = await self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions,
                encoding_format="float"
            )
            # Sort by index to maintain order within batch
            batch_embeddings = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([e.embedding for e in batch_embeddings])

        return all_embeddings

    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a search query"""
        return await self.embed_text(query)

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate embeddings for documents (alias for embed_texts)"""
        return await self.embed_texts(documents)
