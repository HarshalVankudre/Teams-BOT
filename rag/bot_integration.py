"""
Bot Integration for Custom RAG
Use this module in app.py to switch from OpenAI file_search to custom Pinecone RAG
"""
from typing import Optional
from fastapi import Request
from .search import RAGSearch
from .config import config


class BotRAG:
    """
    Custom RAG integration for the Teams Bot.
    Replaces OpenAI file_search with Pinecone-based retrieval.
    """

    def __init__(self):
        self.search = RAGSearch()

    async def get_response(
        self,
        query: str,
        system_instructions: Optional[str] = None,
        use_reranking: bool = False
    ) -> str:
        """
        Get a RAG-enhanced response for a user query.

        Args:
            query: The user's question
            system_instructions: Custom system prompt
            use_reranking: Whether to use GPT-based reranking

        Returns:
            The generated response with source citations
        """
        # Search and generate response
        if use_reranking:
            # Two-stage retrieval with reranking
            chunks = await self.search.search_with_reranking(
                query=query,
                initial_top_k=20,
                final_top_k=config.rerank_top_n
            )
            # Manual response generation with reranked chunks
            result = await self._generate_from_chunks(query, chunks, system_instructions)
        else:
            # Standard search and generate
            result = await self.search.search_and_generate(
                query=query,
                system_instructions=system_instructions
            )

        # Format response with sources
        response = result["response"]

        # Add sources if available
        if result.get("sources"):
            sources_text = self.search.format_sources_for_display(
                result["sources"],
                max_sources=3
            )
            response += sources_text

        return response

    async def _generate_from_chunks(
        self,
        query: str,
        chunks: list,
        system_instructions: Optional[str] = None
    ) -> dict:
        """Generate response from pre-retrieved chunks"""
        return await self.search.search_and_generate(
            query=query,
            system_instructions=system_instructions
        )


# Global instance for bot
bot_rag = BotRAG()


async def get_custom_rag_response(
    query: str,
    system_instructions: Optional[str] = None
) -> str:
    """
    Drop-in replacement for get_assistant_response_streaming.

    Usage in app.py:
        from rag.bot_integration import get_custom_rag_response

        # Replace:
        # assistant_response = await get_assistant_response_streaming(request, thread_key, user_message)

        # With:
        assistant_response = await get_custom_rag_response(user_message, SYSTEM_INSTRUCTIONS)
    """
    return await bot_rag.get_response(query, system_instructions)


# Example system instructions for RÜKO
RUEKO_SYSTEM_INSTRUCTIONS = """Du bist ein Experten-Dokumentenassistent für RÜKO.

WICHTIGE REGELN:
1. Beantworte Fragen NUR basierend auf den abgerufenen Dokumenten
2. Zitiere IMMER die Quellen in deiner Antwort
3. Wenn die Information nicht in den Dokumenten ist, sage es klar
4. Passe die Antwortlänge an die Komplexität der Frage an
5. Verwende Markdown für strukturierte Antworten

ANTWORTFORMAT:
- Beginne mit einer direkten Antwort
- Erkläre Details mit Zitaten
- Strukturiere mit Aufzählungen bei mehreren Punkten
- Schließe mit den wichtigsten Takeaways ab

WENN KEINE INFORMATION GEFUNDEN:
Sage: "Diese Information konnte ich in den RÜKO-Dokumenten nicht finden. Bitte wenden Sie sich an [zuständige Abteilung] oder formulieren Sie Ihre Frage anders."
"""
