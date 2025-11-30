#!/usr/bin/env python3
"""
Machinery Data Indexing Script
Indexes machinery data from JSONL into Pinecone with comprehensive metadata
for direct lookups, aggregations, and recommendations.

Usage:
    python scripts/index_machinery.py --file docs/sema_ki_rag.jsonl
    python scripts/index_machinery.py --stats
    python scripts/index_machinery.py --search "Fertiger für 9m Straße"
    python scripts/index_machinery.py --lookup --seriennummer 12535521
    python scripts/index_machinery.py --count --filter "motor_hersteller=Cummins"
"""
import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import pinecone
from rag.config import config
from rag.embeddings import EmbeddingService

MACHINERY_NAMESPACE = "machinery-data"


class MachineryIndexer:
    """Indexes machinery data with full metadata for comprehensive querying."""

    def __init__(self):
        self.pc = pinecone.Pinecone(api_key=config.pinecone_api_key)
        self.index = self.pc.Index(host=config.pinecone_host)
        self.namespace = MACHINERY_NAMESPACE
        self.embedding_service = EmbeddingService()

    def build_embedding_text(self, record: Dict[str, Any]) -> str:
        """
        Build rich text for embedding that captures all searchable aspects.
        This enables semantic search for recommendations.
        """
        parts = []

        # Title and basic info
        titel = record.get('titel', '')
        bezeichnung = record.get('bezeichnung', '')
        hersteller = record.get('hersteller', '')
        geraetegruppe = record.get('geraetegruppe', '')
        kategorie = record.get('kategorie', '')
        verwendung = record.get('verwendung', '')

        parts.append(f"{titel} - {geraetegruppe}")
        parts.append(f"Hersteller: {hersteller} | Modell: {bezeichnung}")
        if kategorie:
            parts.append(f"Kategorie: {kategorie}")
        if verwendung:
            parts.append(f"Verwendung: {verwendung}")

        # Get eigenschaften (all 171 technical properties)
        eigenschaften = record.get('eigenschaften', {}) or {}

        # Key technical specs in natural language
        tech_parts = []
        if eigenschaften.get('gewicht_kg'):
            kg = eigenschaften['gewicht_kg']
            try:
                kg = float(kg)
                tonnen = kg / 1000
                tech_parts.append(f"Gewicht: {kg:.0f} kg ({tonnen:.1f} Tonnen)")
            except:
                tech_parts.append(f"Gewicht: {kg}")
        if eigenschaften.get('arbeitsbreite_mm'):
            mm = eigenschaften['arbeitsbreite_mm']
            try:
                mm = float(mm)
                m = mm / 1000
                tech_parts.append(f"Arbeitsbreite: {mm:.0f} mm ({m:.1f} Meter)")
            except:
                tech_parts.append(f"Arbeitsbreite: {mm}")
        if eigenschaften.get('breite_mm'):
            mm = eigenschaften['breite_mm']
            try:
                mm = float(mm)
                m = mm / 1000
                tech_parts.append(f"Breite: {mm:.0f} mm ({m:.2f} Meter)")
            except:
                tech_parts.append(f"Breite: {mm}")
        if eigenschaften.get('motor_leistung_kw'):
            kw = eigenschaften['motor_leistung_kw']
            try:
                kw = float(kw)
                ps = kw * 1.36
                tech_parts.append(f"Motorleistung: {kw:.0f} kW ({ps:.0f} PS)")
            except:
                tech_parts.append(f"Motorleistung: {kw}")
        if eigenschaften.get('motor_hersteller'):
            tech_parts.append(f"Motor: {eigenschaften['motor_hersteller']}")
        if eigenschaften.get('motor_typ'):
            tech_parts.append(f"Motortyp: {eigenschaften['motor_typ']}")
        if eigenschaften.get('abgasstufe_eu'):
            tech_parts.append(f"Abgasstufe EU: {eigenschaften['abgasstufe_eu']}")

        if tech_parts:
            parts.append("Technische Daten: " + ", ".join(tech_parts))

        # Equipment/features
        equipment_list = []
        equipment_keys = [
            'klimaanlage', 'dieselpartikelfilter', 'zentralschmierung',
            'hochfahrbare_kabine', 'allradantrieb', 'allradlenkung',
            'kabine', 'knicklenkung', 'oszillation', 'funkfernsteuerung'
        ]
        for key in equipment_keys:
            if eigenschaften.get(key):
                equipment_list.append(key.replace('_', ' ').title())
        if equipment_list:
            parts.append("Ausstattung: " + ", ".join(equipment_list))

        # Application areas (lists)
        einsatzgebiete = record.get('einsatzgebiete', [])
        if einsatzgebiete and isinstance(einsatzgebiete, list):
            parts.append("Einsatzgebiete: " + ", ".join(einsatzgebiete))

        geeignet_fuer = record.get('geeignet_fuer')
        if geeignet_fuer:
            parts.append(f"Geeignet für: {geeignet_fuer}")

        typische_aufgaben = record.get('typische_aufgaben', [])
        if typische_aufgaben and isinstance(typische_aufgaben, list):
            parts.append("Typische Aufgaben: " + ", ".join(typische_aufgaben))

        gelaendetypen = record.get('gelaendetypen', [])
        if gelaendetypen and isinstance(gelaendetypen, list):
            parts.append("Geländetypen: " + ", ".join(gelaendetypen))

        # Original content
        inhalt = record.get('inhalt', '')
        if inhalt:
            parts.append(f"Beschreibung: {inhalt}")

        return "\n".join(parts)

    def _str_or_missing(self, value, max_len: int = None) -> str:
        """Return value as string or 'nicht-vorhanden' if empty/null."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return "nicht-vorhanden"
        result = str(value).strip()
        if max_len:
            result = result[:max_len]
        return result

    def _num_or_zero(self, value) -> float:
        """Return numeric value or 0 if empty/null."""
        if value is None:
            return 0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0

    def _is_numeric_field(self, key: str, value: Any) -> bool:
        """Check if a field should be treated as numeric."""
        # Common numeric suffixes in German technical data
        numeric_suffixes = ['_mm', '_m', '_kg', '_t', '_kw', '_kva', '_bar', '_hz',
                           '_l_min', '_t_h', '_km_h', '_u_min', '_g_km', '_v', '_a',
                           '_knm', '%']
        key_lower = key.lower()
        for suffix in numeric_suffixes:
            if key_lower.endswith(suffix):
                return True
        # Also check if value is numeric
        if isinstance(value, (int, float)):
            return True
        return False

    def _is_boolean_field(self, value: Any) -> bool:
        """Check if a field is boolean."""
        return isinstance(value, bool)

    def build_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build comprehensive metadata for filtering and full retrieval.
        Captures all 171 eigenschaften properties plus top-level properties.
        Missing string values are replaced with 'nicht-vorhanden'.
        """
        # Helper for list fields
        def list_or_missing(lst):
            if not lst:
                return "nicht-vorhanden"
            if isinstance(lst, list):
                return ",".join(str(item) for item in lst)
            return str(lst)

        metadata = {}

        # === TOP-LEVEL PROPERTIES (19) ===
        metadata["id"] = self._str_or_missing(record.get('id'))
        metadata["seriennummer"] = self._str_or_missing(record.get('seriennummer'))
        metadata["inventarnummer"] = self._str_or_missing(record.get('inventarnummer'))
        metadata["primaerschluessel"] = record.get('primaerschluessel', 0)
        metadata["bezeichnung"] = self._str_or_missing(record.get('bezeichnung'), 200)
        metadata["hersteller"] = self._str_or_missing(record.get('hersteller'))
        metadata["hersteller_code"] = self._str_or_missing(record.get('hersteller_code'))
        metadata["geraetegruppe"] = self._str_or_missing(record.get('geraetegruppe'))
        metadata["geraetegruppe_code"] = self._str_or_missing(record.get('geraetegruppe_code'))
        metadata["kategorie"] = self._str_or_missing(record.get('kategorie'))
        metadata["verwendung"] = self._str_or_missing(record.get('verwendung'))
        metadata["verwendung_code"] = self._str_or_missing(record.get('verwendung_code'))
        metadata["titel"] = self._str_or_missing(record.get('titel'), 200)
        metadata["inhalt"] = self._str_or_missing(record.get('inhalt'), 1500)

        # List fields (comma-separated)
        metadata["einsatzgebiete"] = list_or_missing(record.get('einsatzgebiete'))
        metadata["gelaendetypen"] = list_or_missing(record.get('gelaendetypen'))
        metadata["typische_aufgaben"] = list_or_missing(record.get('typische_aufgaben'))
        metadata["geeignet_fuer"] = self._str_or_missing(record.get('geeignet_fuer'))

        # === ALL 171 EIGENSCHAFTEN PROPERTIES ===
        eigenschaften = record.get('eigenschaften', {}) or {}

        for key, value in eigenschaften.items():
            # Sanitize key for Pinecone (replace special chars)
            safe_key = key.replace('.', '_').replace('?', '').replace('/', '_')

            if self._is_boolean_field(value):
                # Boolean fields stay as boolean
                metadata[safe_key] = bool(value)
            elif self._is_numeric_field(key, value):
                # Numeric fields - use 0 for missing
                metadata[safe_key] = self._num_or_zero(value)
            else:
                # String fields - use 'nicht-vorhanden' for missing
                metadata[safe_key] = self._str_or_missing(value)

        # === FULL DATA FOR COMPLETE RETRIEVAL ===
        metadata["full_data_json"] = json.dumps(record, ensure_ascii=False)[:30000]

        return metadata

    async def index_file(self, file_path: str, batch_size: int = 100) -> Dict[str, Any]:
        """Index all machinery records from a JSONL file."""

        print(f"Reading {file_path}...")
        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        print(f"Found {len(records)} records")

        # Build embedding texts
        print("Building embedding texts...")
        texts = [self.build_embedding_text(r) for r in records]

        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} records...")
        embeddings = await self.embedding_service.embed_texts(texts)

        # Prepare vectors
        vectors = []
        for record, embedding in zip(records, embeddings):
            vector_id = str(record.get('primaerschluessel', record.get('id')))
            metadata = self.build_metadata(record)
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })

        # Upsert in batches
        print(f"Uploading to namespace '{self.namespace}'...")
        total_upserted = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)
            total_upserted += len(batch)
            print(f"  Uploaded batch {i // batch_size + 1}: {total_upserted}/{len(vectors)} vectors")

        print(f"\nSuccessfully indexed {total_upserted} machinery records!")
        return {"status": "success", "indexed_count": total_upserted}

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Semantic search with optional filters."""

        query_embedding = await self.embedding_service.embed_query(query)

        # Build Pinecone filter
        pinecone_filter = None
        if filters:
            pinecone_filter = {}
            for key, value in filters.items():
                if isinstance(value, dict):
                    pinecone_filter[key] = value
                elif isinstance(value, list):
                    pinecone_filter[key] = {"$in": value}
                else:
                    pinecone_filter[key] = {"$eq": value}

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
            filter=pinecone_filter
        )

        formatted = []
        for match in results.matches:
            formatted.append({
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata
            })
        return formatted

    async def lookup_by_field(
        self,
        field: str,
        value: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Lookup machines by a specific field value.
        Uses a dummy query with strict filter.
        """
        # Create a generic query embedding
        query_embedding = await self.embedding_service.embed_query(f"{field} {value}")

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
            filter={field: {"$eq": value}}
        )

        formatted = []
        for match in results.matches:
            formatted.append({
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata
            })
        return formatted

    async def count_by_filter(self, filters: Dict[str, Any]) -> int:
        """
        Count machines matching a filter.
        Note: Pinecone doesn't have native count, so we fetch and count.
        """
        # Create a generic query
        query_embedding = await self.embedding_service.embed_query("Maschine Gerät")

        pinecone_filter = {}
        for key, value in filters.items():
            if isinstance(value, dict):
                pinecone_filter[key] = value
            else:
                pinecone_filter[key] = {"$eq": value}

        # Fetch up to 10000 to count
        results = self.index.query(
            vector=query_embedding,
            top_k=10000,
            namespace=self.namespace,
            include_metadata=False,
            filter=pinecone_filter
        )

        return len(results.matches)

    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics for machinery namespace."""
        stats = self.index.describe_index_stats()

        ns_stats = stats.namespaces.get(self.namespace, {})
        return {
            "namespace": self.namespace,
            "vector_count": getattr(ns_stats, 'vector_count', 0),
            "total_vectors_all_namespaces": stats.total_vector_count,
            "dimension": stats.dimension
        }

    async def delete_all(self) -> Dict[str, Any]:
        """Delete all vectors in machinery namespace."""
        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
            return {"status": "deleted", "namespace": self.namespace}
        except Exception as e:
            return {"status": "error", "message": str(e)}


async def main():
    parser = argparse.ArgumentParser(description="Index machinery data into Pinecone")

    parser.add_argument("--file", "-f", type=str, help="JSONL file to index")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--search", "-s", type=str, help="Semantic search query")
    parser.add_argument("--lookup", action="store_true", help="Lookup by field")
    parser.add_argument("--seriennummer", type=str, help="Serial number for lookup")
    parser.add_argument("--inventarnummer", type=str, help="Inventory number for lookup")
    parser.add_argument("--count", action="store_true", help="Count matching records")
    parser.add_argument("--filter", type=str, help="Filter for count (key=value)")
    parser.add_argument("--delete-all", action="store_true", help="Delete all machinery data")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results")

    args = parser.parse_args()
    indexer = MachineryIndexer()

    if args.file:
        result = await indexer.index_file(args.file)
        print(f"\nResult: {result}")

    elif args.stats:
        stats = await indexer.get_stats()
        print("\nMachinery Namespace Statistics:")
        print(f"  Namespace: {stats['namespace']}")
        print(f"  Vectors: {stats['vector_count']}")
        print(f"  Dimension: {stats['dimension']}")

    elif args.search:
        print(f"\nSearching for: '{args.search}'")
        results = await indexer.search(args.search, top_k=args.top_k)
        print(f"Found {len(results)} results:\n")
        for i, r in enumerate(results):
            m = r['metadata']
            print(f"{i+1}. {m.get('titel', 'N/A')} (Score: {r['score']:.2%})")
            print(f"   Hersteller: {m.get('hersteller')} | Kategorie: {m.get('kategorie')}")
            print(f"   Seriennummer: {m.get('seriennummer')} | Verwendung: {m.get('verwendung')}")
            if m.get('motor_leistung_kw'):
                print(f"   Motor: {m.get('motor_leistung_kw')}kW | Gewicht: {m.get('gewicht_kg')}kg")
            print()

    elif args.lookup:
        if args.seriennummer:
            print(f"\nLooking up Seriennummer: {args.seriennummer}")
            results = await indexer.lookup_by_field("seriennummer", args.seriennummer)
        elif args.inventarnummer:
            print(f"\nLooking up Inventarnummer: {args.inventarnummer}")
            results = await indexer.lookup_by_field("inventarnummer", args.inventarnummer)
        else:
            print("Please specify --seriennummer or --inventarnummer")
            return

        if results:
            for r in results:
                full_data = json.loads(r['metadata'].get('full_data_json', '{}'))
                print(json.dumps(full_data, indent=2, ensure_ascii=False))
        else:
            print("No results found")

    elif args.count:
        if args.filter:
            filter_str = args.filter
            filters = {}

            # Parse filter with comparison operators
            if ">=" in filter_str:
                key, value = filter_str.split(">=", 1)
                filters = {key.strip(): {"$gte": float(value.strip())}}
            elif "<=" in filter_str:
                key, value = filter_str.split("<=", 1)
                filters = {key.strip(): {"$lte": float(value.strip())}}
            elif ">" in filter_str:
                key, value = filter_str.split(">", 1)
                filters = {key.strip(): {"$gt": float(value.strip())}}
            elif "<" in filter_str:
                key, value = filter_str.split("<", 1)
                filters = {key.strip(): {"$lt": float(value.strip())}}
            elif "=" in filter_str:
                key, value = filter_str.split("=", 1)
                key = key.strip()
                value = value.strip()
                # Try to parse as number
                try:
                    value = float(value)
                except:
                    pass
                filters = {key: value}
            else:
                print("Invalid filter format. Use: key=value, key>value, key<value")
                return

            count = await indexer.count_by_filter(filters)
            print(f"\nCount for {args.filter}: {count} machines")
        else:
            print("Please specify --filter key=value or key>value or key<value")

    elif args.delete_all:
        confirm = input("Are you sure you want to delete all machinery data? (yes/no): ")
        if confirm.lower() == "yes":
            result = await indexer.delete_all()
            print(f"Result: {result}")
        else:
            print("Cancelled")

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
