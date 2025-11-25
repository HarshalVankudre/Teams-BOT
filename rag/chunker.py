"""
Semantic Chunker using GPT-5.1 with low reasoning
Chunks documents by topic/theme, not by tokens or pages
"""
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from .config import config


class SemanticChunker:
    """
    Smart document chunker that uses GPT-5.1 to:
    - Split documents by semantic topics/themes
    - Extract rich metadata for each chunk
    - Handle different document types (text, JSON, tables)
    """

    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.model = config.chunking_model
        self.reasoning_effort = config.chunking_reasoning

    async def chunk_document(
        self,
        content: str,
        source_file: str,
        doc_type: str = "text",
        additional_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantically chunk a document into topic-based segments.

        Args:
            content: The document content
            source_file: Name/path of the source file
            doc_type: Type of document (text, pdf, json, table)
            additional_context: Extra context about the document

        Returns:
            List of chunk dictionaries with content and metadata
        """
        system_prompt = """Du bist ein Experte für Dokumentenanalyse und semantische Segmentierung.

Deine Aufgabe ist es, das Dokument in semantische Chunks aufzuteilen, wobei JEDER Chunk ein vollständiges Thema/Konzept darstellt.

REGELN:
1. Jeder Chunk sollte ein EINZELNES, VOLLSTÄNDIGES Thema behandeln
2. Chunks sollten in sich geschlossen und verständlich sein
3. Extrahiere reichhaltige Metadaten für jeden Chunk
4. Behalte wichtige Kontext-Beziehungen zwischen Chunks bei
5. Identifiziere Schlüsselwörter und Kategorien präzise

Gib JSON zurück mit folgendem Schema:
{
    "chunks": [
        {
            "title": "Aussagekräftiger Titel des Themas",
            "content": "Der vollständige Inhalt des Chunks",
            "summary": "2-3 Sätze Zusammenfassung",
            "keywords": ["schlüsselwort1", "schlüsselwort2", "schlüsselwort3"],
            "category": "policy|procedure|guide|faq|template|regulation|other",
            "importance": "high|medium|low",
            "related_topics": ["verwandtes_thema1", "verwandtes_thema2"],
            "entities": {
                "people": ["Name1"],
                "departments": ["Abteilung1"],
                "dates": ["Datum1"],
                "numbers": ["wichtige_zahl1"]
            },
            "section_hierarchy": "Hauptabschnitt > Unterabschnitt",
            "language": "de|en",
            "requires_action": true|false,
            "target_audience": "all|management|employees|specific_role"
        }
    ],
    "document_metadata": {
        "main_topic": "Hauptthema des Dokuments",
        "document_type": "policy|procedure|guide|etc",
        "total_chunks": 5,
        "languages_detected": ["de", "en"]
    }
}"""

        user_prompt = f"""Analysiere und segmentiere das folgende Dokument semantisch:

DOKUMENTTYP: {doc_type}
QUELLDATEI: {source_file}
{f'ZUSÄTZLICHER KONTEXT: {additional_context}' if additional_context else ''}

DOKUMENTINHALT:
{content}

Gib NUR valides JSON zurück, keine zusätzlichen Erklärungen."""

        try:
            # Build request params - only include reasoning if not "none"
            request_params = {
                "model": self.model,
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "text": {"format": {"type": "json_object"}},
                "max_output_tokens": 4000
            }
            if self.reasoning_effort and self.reasoning_effort != "none":
                request_params["reasoning"] = {"effort": self.reasoning_effort}

            response = await self.client.responses.create(**request_params)

            # Parse the JSON response
            result = json.loads(response.output_text)
            chunks = result.get("chunks", [])
            doc_metadata = result.get("document_metadata", {})

            # Enrich chunks with additional metadata
            enriched_chunks = []
            for i, chunk in enumerate(chunks):
                enriched_chunk = self._enrich_chunk(
                    chunk=chunk,
                    source_file=source_file,
                    doc_type=doc_type,
                    chunk_index=i,
                    total_chunks=len(chunks),
                    doc_metadata=doc_metadata
                )
                enriched_chunks.append(enriched_chunk)

            return enriched_chunks

        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            # Fallback: create chunk(s) - may return list if content is large
            fallback = self._create_fallback_chunk(content, source_file, doc_type)
            return fallback if isinstance(fallback, list) else [fallback]
        except Exception as e:
            print(f"Chunking error: {e}")
            fallback = self._create_fallback_chunk(content, source_file, doc_type)
            return fallback if isinstance(fallback, list) else [fallback]

    async def chunk_json_document(
        self,
        json_data: Dict[str, Any],
        source_file: str,
        max_concurrent: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Chunk a JSON document where each meaningful object becomes a vector.
        Uses parallel processing for speed.

        Args:
            json_data: The JSON data as a dictionary
            source_file: Name of the source file
            max_concurrent: Max parallel API calls (default 20)

        Returns:
            List of chunks, one per meaningful JSON object
        """
        import asyncio

        # Step 1: Collect all meaningful objects first (no API calls)
        objects_to_process = []

        def collect_objects(obj: Any, path: str = ""):
            """Recursively collect meaningful objects"""
            if isinstance(obj, dict):
                if self._is_meaningful_object(obj):
                    objects_to_process.append((obj, path))
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    collect_objects(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    collect_objects(item, f"{path}[{i}]")

        collect_objects(json_data)
        total = len(objects_to_process)
        print(f"Found {total} objects to process in parallel (max {max_concurrent} concurrent)")

        # Step 2: Process in parallel with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        processed = [0]  # Use list to allow mutation in closure

        async def process_with_semaphore(obj, path):
            async with semaphore:
                chunk = await self._create_json_chunk(obj, path, source_file)
                processed[0] += 1
                if processed[0] % 50 == 0 or processed[0] == total:
                    print(f"  Progress: {processed[0]}/{total} objects")
                return chunk

        # Run all in parallel
        tasks = [process_with_semaphore(obj, path) for obj, path in objects_to_process]
        chunks = await asyncio.gather(*tasks)

        return list(chunks)

    def _is_meaningful_object(self, obj: Dict) -> bool:
        """Determine if a JSON object is meaningful enough for its own vector"""
        # Object should have at least 2 keys
        if len(obj) < 2:
            return False

        keys = set(obj.keys())

        # Skip simple value/unit pairs
        if keys == {"value", "unit"} or keys <= {"value", "unit", "type"}:
            return False

        # STRICT: Only match objects that look like actual entities/records
        # Must have at least one primary identifier
        primary_ids = {"name", "id", "title", "serial_number", "nummer", "bezeichnung"}
        has_primary = bool(keys & primary_ids)

        # Must also have at least one secondary attribute
        secondary_attrs = {"inventory_number", "description", "properties", "kategorie",
                          "type", "status", "data", "content", "value", "details"}
        has_secondary = bool(keys & secondary_attrs)

        # Only match if has BOTH a primary ID and secondary data
        if has_primary and has_secondary:
            text_content = json.dumps(obj, ensure_ascii=False)
            return len(text_content) > 50

        return False

    def _extract_machine_identifiers(self, obj: Dict) -> Dict[str, Any]:
        """Extract searchable identifiers from a machine/equipment object"""
        identifiers = {
            "machine_name": "",
            "serial_number": "",
            "inventory_number": "",
            "machine_type": "",
            "searchable_ids": []  # Combined list for keyword search
        }

        # Direct field extraction
        if "name" in obj:
            identifiers["machine_name"] = str(obj["name"])
            identifiers["searchable_ids"].append(str(obj["name"]))

        if "serial_number" in obj:
            identifiers["serial_number"] = str(obj["serial_number"])
            identifiers["searchable_ids"].append(str(obj["serial_number"]))

        if "inventory_number" in obj:
            identifiers["inventory_number"] = str(obj["inventory_number"])
            identifiers["searchable_ids"].append(str(obj["inventory_number"]))

        # Try to extract machine type from name (e.g., "MF 2500 CS" -> type)
        if identifiers["machine_name"]:
            # Common patterns: letters + numbers, e.g., "MF 2500 CS", "300.9D"
            name = identifiers["machine_name"]
            identifiers["machine_type"] = name.split()[0] if " " in name else name

        return identifiers

    async def _create_json_chunk(
        self,
        obj: Dict,
        json_path: str,
        source_file: str
    ) -> Dict[str, Any]:
        """Create a chunk from a JSON object with extracted identifiers for hybrid search"""
        content = json.dumps(obj, ensure_ascii=False, indent=2)

        # Extract machine identifiers for metadata filtering
        identifiers = self._extract_machine_identifiers(obj)

        # Use GPT to extract additional metadata
        try:
            request_params = {
                "model": self.model,
                "input": f"""Analysiere dieses JSON-Objekt und extrahiere Metadaten:

{content}

Gib JSON zurück:
{{
    "title": "Beschreibender Titel",
    "summary": "Kurze Zusammenfassung",
    "keywords": ["keyword1", "keyword2"],
    "category": "data|config|record|other"
}}""",
                "text": {"format": {"type": "json_object"}},
                "max_output_tokens": 500
            }
            if self.reasoning_effort and self.reasoning_effort != "none":
                request_params["reasoning"] = {"effort": self.reasoning_effort}

            response = await self.client.responses.create(**request_params)
            metadata = json.loads(response.output_text)
        except Exception:
            metadata = {
                "title": json_path or "JSON Object",
                "summary": "JSON data object",
                "keywords": list(obj.keys())[:5],
                "category": "data"
            }

        chunk_id = self._generate_chunk_id(content, source_file, json_path)

        # Build comprehensive keywords from identifiers + AI-extracted
        all_keywords = list(set(
            identifiers["searchable_ids"] +
            metadata.get("keywords", [])
        ))

        # Build a better title using machine name
        title = identifiers["machine_name"] if identifiers["machine_name"] else metadata.get("title", json_path)
        if identifiers["serial_number"]:
            title = f"{title} ({identifiers['serial_number']})"

        return {
            "id": chunk_id,
            "content": content,
            "title": title,
            "summary": metadata.get("summary", ""),
            "keywords": all_keywords,
            "category": metadata.get("category", "data"),
            "importance": "medium",
            "source_file": source_file,
            "json_path": json_path,
            "object_keys": list(obj.keys()),
            "doc_type": "json",
            "indexed_at": datetime.utcnow().isoformat(),
            "chunk_type": "json_object",
            # NEW: Searchable identifier fields for hybrid search
            "machine_name": identifiers["machine_name"],
            "serial_number": identifiers["serial_number"],
            "inventory_number": identifiers["inventory_number"],
            "machine_type": identifiers["machine_type"]
        }

    def _enrich_chunk(
        self,
        chunk: Dict,
        source_file: str,
        doc_type: str,
        chunk_index: int,
        total_chunks: int,
        doc_metadata: Dict
    ) -> Dict[str, Any]:
        """Add additional metadata to a chunk"""
        content = chunk.get("content", "")
        chunk_id = self._generate_chunk_id(content, source_file, str(chunk_index))

        return {
            # Core identifiers
            "id": chunk_id,
            "content": content,

            # AI-extracted metadata
            "title": chunk.get("title", f"Chunk {chunk_index + 1}"),
            "summary": chunk.get("summary", ""),
            "keywords": chunk.get("keywords", []),
            "category": chunk.get("category", "other"),
            "importance": chunk.get("importance", "medium"),
            "related_topics": chunk.get("related_topics", []),
            "entities": chunk.get("entities", {}),
            "section_hierarchy": chunk.get("section_hierarchy", ""),
            "language": chunk.get("language", "de"),
            "requires_action": chunk.get("requires_action", False),
            "target_audience": chunk.get("target_audience", "all"),

            # Source tracking
            "source_file": source_file,
            "doc_type": doc_type,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,

            # Document-level metadata
            "document_main_topic": doc_metadata.get("main_topic", ""),
            "document_type": doc_metadata.get("document_type", ""),

            # Timestamps
            "indexed_at": datetime.utcnow().isoformat(),

            # Chunk type
            "chunk_type": "semantic"
        }

    def _create_fallback_chunk(
        self,
        content: str,
        source_file: str,
        doc_type: str
    ) -> Dict[str, Any]:
        """Create a basic chunk when semantic chunking fails"""
        # Check if content is too large (embedding model limit ~8192 tokens, ~4 chars/token = 30000 chars safe)
        MAX_CHUNK_CHARS = 25000

        if len(content) > MAX_CHUNK_CHARS:
            # Split into multiple chunks
            return self._split_large_content(content, source_file, doc_type, MAX_CHUNK_CHARS)

        chunk_id = self._generate_chunk_id(content, source_file, "0")

        return {
            "id": chunk_id,
            "content": content,
            "title": source_file,
            "summary": content[:200] + "..." if len(content) > 200 else content,
            "keywords": [],
            "category": "other",
            "importance": "medium",
            "related_topics": [],
            "entities": {},
            "source_file": source_file,
            "doc_type": doc_type,
            "chunk_index": 0,
            "total_chunks": 1,
            "indexed_at": datetime.utcnow().isoformat(),
            "chunk_type": "fallback"
        }

    def _split_large_content(
        self,
        content: str,
        source_file: str,
        doc_type: str,
        max_chars: int
    ) -> List[Dict[str, Any]]:
        """Split large content into multiple chunks by paragraphs or sentences"""
        chunks = []

        # Try to split by double newlines (paragraphs) first
        paragraphs = content.split('\n\n')

        current_chunk = ""
        chunk_index = 0

        for para in paragraphs:
            # If adding this paragraph would exceed limit, save current and start new
            if len(current_chunk) + len(para) + 2 > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    chunk_index += 1

                # If single paragraph is too large, split by sentences
                if len(para) > max_chars:
                    sentence_chunks = self._split_by_sentences(para, max_chars)
                    chunks.extend(sentence_chunks)
                    current_chunk = ""
                else:
                    current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Convert to chunk dicts
        total_chunks = len(chunks)
        result = []
        for i, chunk_content in enumerate(chunks):
            chunk_id = self._generate_chunk_id(chunk_content, source_file, str(i))
            result.append({
                "id": chunk_id,
                "content": chunk_content,
                "title": f"{source_file} (Teil {i+1}/{total_chunks})",
                "summary": chunk_content[:200] + "..." if len(chunk_content) > 200 else chunk_content,
                "keywords": [],
                "category": "other",
                "importance": "medium",
                "related_topics": [],
                "entities": {},
                "source_file": source_file,
                "doc_type": doc_type,
                "chunk_index": i,
                "total_chunks": total_chunks,
                "indexed_at": datetime.utcnow().isoformat(),
                "chunk_type": "fallback_split"
            })

        return result

    def _split_by_sentences(self, text: str, max_chars: int) -> List[str]:
        """Split text by sentences when paragraphs are too large"""
        import re
        # Split by sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) + 1 > max_chars:
                if current:
                    chunks.append(current.strip())
                # If single sentence is too long, force split
                if len(sentence) > max_chars:
                    for i in range(0, len(sentence), max_chars):
                        chunks.append(sentence[i:i+max_chars])
                    current = ""
                else:
                    current = sentence
            else:
                current += " " + sentence if current else sentence

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def _generate_chunk_id(self, content: str, source: str, identifier: str) -> str:
        """Generate a unique ID for a chunk"""
        hash_input = f"{source}:{identifier}:{content[:100]}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
