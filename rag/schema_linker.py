"""
Schema Linker for Semantic Column Resolution.

Provides semantic retrieval of relevant columns for text-to-SQL queries.
Replaces the full-schema approach with a reduced-schema approach.

Key components:
- ColumnID: Canonical table.column identifier
- ColumnMetadata: Rich metadata for each column
- ReducedSchema: Subset of schema relevant to a query
- SchemaLinker: Main class for column retrieval

Phase 2 adds optional vector-based retrieval using Pinecone.
"""
from __future__ import annotations

import asyncio
import logging
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass(frozen=True)
class ColumnID:
    """
    Canonical column identifier - always table.column format.

    Usage:
        col = ColumnID("equipment_matrix", "prop_e1730_gewicht_kg")
        print(col.qualified)  # "equipment_matrix.prop_e1730_gewicht_kg"
    """
    table: str
    column: str

    @property
    def qualified(self) -> str:
        """Return fully-qualified table.column string."""
        return f"{self.table}.{self.column}"

    @classmethod
    def from_string(cls, s: str, default_table: str = "equipment_matrix") -> "ColumnID":
        """
        Parse a column reference string.

        Args:
            s: Column string, either "column" or "table.column"
            default_table: Table to use if not specified

        Returns:
            ColumnID instance
        """
        if "." in s:
            parts = s.split(".", 1)
            return cls(table=parts[0], column=parts[1])
        return cls(table=default_table, column=s)

    def __str__(self) -> str:
        return self.qualified

    def __hash__(self) -> int:
        return hash(self.qualified)


@dataclass
class ColumnMetadata:
    """
    Rich metadata for a database column.

    Used for:
    - Building embeddings for semantic search
    - Generating reduced schema prompts
    - Providing examples and type information to LLM
    """
    column_id: ColumnID
    business_name: str
    description: str = ""
    data_type: str = "TEXT"  # TEXT, NUMERIC_TEXT, CATEGORICAL, BOOLEAN, BIGINT
    unit: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    null_ratio: float = 0.0
    cast_recipe: Optional[str] = None
    deprecated: bool = False
    alternative: Optional[str] = None  # Alternative column if deprecated
    code: Optional[str] = None  # E-code like E1730

    @property
    def is_usable(self) -> bool:
        """Return True if column should be used (not deprecated, has data)."""
        return not self.deprecated and self.null_ratio < 0.99

    def to_embedding_text(self) -> str:
        """Build text for embedding generation."""
        parts = [
            self.business_name,
            self.description,
        ]
        if self.synonyms:
            parts.append(f"Synonyms: {', '.join(self.synonyms)}")
        if self.examples:
            parts.append(f"Examples: {', '.join(self.examples[:5])}")
        if self.unit:
            parts.append(f"Unit: {self.unit}")
        return "\n".join(filter(None, parts))


@dataclass
class ValueHint:
    """Suggested value for a categorical column."""
    value: str
    confidence: float = 1.0
    match_type: str = "exact"  # exact, fuzzy, semantic


@dataclass
class JoinPath:
    """Allowed join between tables."""
    from_table: str
    from_col: str
    to_table: str
    to_col: str
    join_type: str = "INNER"


@dataclass
class ReducedSchema:
    """
    Reduced schema for SQL generation.

    Contains only the columns relevant to a specific query,
    plus core columns that are always needed.
    """
    # Allowed columns (as table.column strings)
    allowed_columns: Set[str] = field(default_factory=set)

    # Allowed tables (derived from columns)
    allowed_tables: Set[str] = field(default_factory=set)

    # Column metadata for prompt generation
    column_info: Dict[str, ColumnMetadata] = field(default_factory=dict)

    # Value hints per categorical column
    value_hints: Dict[str, List[ValueHint]] = field(default_factory=dict)

    # Join graph (if multi-table allowed)
    allowed_joins: Optional[List[JoinPath]] = None

    # Core columns (always included, not from retrieval)
    core_columns: Set[str] = field(default_factory=set)

    # Casting recipes for NUMERIC_TEXT columns
    cast_recipes: Dict[str, str] = field(default_factory=dict)

    def is_column_allowed(self, table: str, column: str) -> bool:
        """Check if a column is in the allowed set."""
        return f"{table}.{column}" in self.allowed_columns

    def get_cast_recipe(self, column_id: str) -> Optional[str]:
        """Get SQL cast expression for a column."""
        return self.cast_recipes.get(column_id)

    def to_prompt(self) -> str:
        """
        Generate prompt section for LLM.

        Returns a formatted string describing available columns,
        their types, examples, and value hints.
        """
        lines = [
            "AVAILABLE COLUMNS (use ONLY these, always qualify as table.column):",
            "-" * 50
        ]

        # Group by core vs property columns
        core_cols = []
        prop_cols = []

        for col_id in sorted(self.allowed_columns):
            meta = self.column_info.get(col_id)
            if col_id in self.core_columns:
                core_cols.append((col_id, meta))
            else:
                prop_cols.append((col_id, meta))

        # Core columns first
        if core_cols:
            lines.append("\nCORE COLUMNS:")
            for col_id, meta in core_cols:
                lines.append(self._format_column_entry(col_id, meta))

        # Property columns
        if prop_cols:
            lines.append("\nPROPERTY COLUMNS:")
            for col_id, meta in prop_cols:
                lines.append(self._format_column_entry(col_id, meta))

        # Value hints
        if self.value_hints:
            lines.append("")
            lines.append("VALUE HINTS (suggested values, verify with ILIKE if uncertain):")
            lines.append("-" * 50)
            for col_id, hints in self.value_hints.items():
                values = [h.value for h in hints[:5]]
                lines.append(f"  {col_id}: {', '.join(values)}")
                if len(hints) > 5:
                    lines.append(f"    ... and {len(hints) - 5} more values")

        # Cast recipes
        numeric_cols = [
            (col_id, meta) for col_id, meta in self.column_info.items()
            if meta and meta.data_type == "NUMERIC_TEXT" and col_id in self.allowed_columns
        ]
        if numeric_cols:
            lines.append("")
            lines.append("NUMERIC COLUMNS (stored as text, use cast recipe):")
            lines.append("-" * 50)
            for col_id, meta in numeric_cols[:5]:
                recipe = self.cast_recipes.get(col_id) or self.cast_recipes.get(meta.unit)
                if recipe:
                    lines.append(f"  {col_id}: {recipe.replace('{col}', col_id)}")

        return "\n".join(lines)

    def _format_column_entry(self, col_id: str, meta: Optional[ColumnMetadata]) -> str:
        """Format a single column entry for the prompt."""
        if meta:
            entry = f"  {col_id}: {meta.business_name}"
            if meta.unit:
                entry += f" [{meta.unit}]"
            if meta.data_type == "CATEGORICAL":
                entry += " (categorical)"
            elif meta.data_type == "BOOLEAN":
                entry += " (boolean - check for 'Ja')"
            elif meta.data_type == "NUMERIC_TEXT":
                entry += " (numeric in text)"
            if meta.examples:
                entry += f" e.g., {', '.join(meta.examples[:2])}"
        else:
            entry = f"  {col_id}"
        return entry


# =============================================================================
# Schema Linker
# =============================================================================

class SchemaLinker:
    """
    Main class for semantic schema linking.

    Loads column catalog and provides methods to retrieve
    relevant columns for a query.
    """

    # Default core columns that are always included
    DEFAULT_CORE_COLUMNS = frozenset([
        "equipment_matrix.id",
        "equipment_matrix.bezeichnung",
        "equipment_matrix.seriennummer",
        "equipment_matrix.inventarnummer",
        "equipment_matrix.hersteller_name",
        "equipment_matrix.geraetegruppe_name",
        "equipment_matrix.verwendung_code",
        "equipment_matrix.nuclos_state",
    ])

    # Default table (single-table policy)
    DEFAULT_TABLE = "equipment_matrix"

    def __init__(self, catalog_path: Optional[Path] = None):
        """
        Initialize the schema linker.

        Args:
            catalog_path: Path to column_catalog.yaml. If None, uses default.
        """
        self._columns: Dict[str, ColumnMetadata] = {}
        self._core_columns: Dict[str, ColumnMetadata] = {}
        self._cast_recipes: Dict[str, str] = {}
        self._synonyms_index: Dict[str, List[str]] = {}  # synonym -> [column_ids]
        self._initialized = False

        if catalog_path is None:
            catalog_path = Path(__file__).parent.parent / "sql_export" / "column_catalog.yaml"

        self._catalog_path = catalog_path

    def initialize(self) -> None:
        """Load the column catalog from YAML."""
        if self._initialized:
            return

        if not self._catalog_path.exists():
            logger.warning(f"[SchemaLinker] Catalog not found: {self._catalog_path}")
            self._initialized = True
            return

        try:
            with open(self._catalog_path, "r", encoding="utf-8") as f:
                catalog = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"[SchemaLinker] Failed to load catalog: {e}")
            self._initialized = True
            return

        table = catalog.get("table", self.DEFAULT_TABLE)

        # Load cast recipes
        self._cast_recipes = catalog.get("cast_recipes", {})

        # Load core columns
        for entry in catalog.get("core_columns", []):
            col_id = ColumnID(table, entry["column"])
            meta = self._parse_column_entry(entry, table)
            self._core_columns[col_id.qualified] = meta
            self._index_synonyms(col_id.qualified, meta.synonyms)

        # Load property columns
        for entry in catalog.get("columns", []):
            col_id = ColumnID(table, entry["column"])
            meta = self._parse_column_entry(entry, table)
            self._columns[col_id.qualified] = meta
            self._index_synonyms(col_id.qualified, meta.synonyms)

        logger.info(
            f"[SchemaLinker] Loaded {len(self._core_columns)} core columns, "
            f"{len(self._columns)} property columns"
        )
        self._initialized = True

    def _parse_column_entry(self, entry: Dict[str, Any], table: str) -> ColumnMetadata:
        """Parse a column entry from the catalog YAML."""
        col_id = ColumnID(table, entry["column"])

        # Get cast recipe for this column's unit
        unit = entry.get("unit")
        cast_recipe = None
        if unit and unit in self._cast_recipes:
            cast_recipe = self._cast_recipes[unit]

        return ColumnMetadata(
            column_id=col_id,
            business_name=entry.get("business_name", entry["column"]),
            description=entry.get("description", ""),
            data_type=entry.get("data_type", "TEXT"),
            unit=unit,
            synonyms=entry.get("synonyms", []),
            examples=entry.get("examples", []),
            null_ratio=entry.get("null_ratio", 0.0),
            cast_recipe=cast_recipe,
            deprecated=entry.get("deprecated", False),
            alternative=entry.get("alternative"),
            code=entry.get("code"),
        )

    def _index_synonyms(self, col_id: str, synonyms: List[str]) -> None:
        """Index synonyms for keyword-based retrieval."""
        for syn in synonyms:
            syn_lower = syn.lower()
            if syn_lower not in self._synonyms_index:
                self._synonyms_index[syn_lower] = []
            self._synonyms_index[syn_lower].append(col_id)

    def get_reduced_schema(
        self,
        query: str,
        top_k: int = 15,
        include_core: bool = True,
    ) -> ReducedSchema:
        """
        Get a reduced schema relevant to the query.

        This is a basic keyword-based implementation.
        Phase 2 will add vector-based semantic retrieval.

        Args:
            query: User's natural language query
            top_k: Maximum number of property columns to include
            include_core: Whether to always include core columns

        Returns:
            ReducedSchema with relevant columns
        """
        self.initialize()

        allowed_columns: Set[str] = set()
        column_info: Dict[str, ColumnMetadata] = {}
        cast_recipes: Dict[str, str] = {}

        # Always include core columns if requested
        core_column_ids: Set[str] = set()
        if include_core:
            for col_id, meta in self._core_columns.items():
                allowed_columns.add(col_id)
                column_info[col_id] = meta
                core_column_ids.add(col_id)

        # Keyword-based retrieval from query
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        # Score columns based on keyword matches
        scored_columns: List[tuple] = []
        for col_id, meta in self._columns.items():
            if meta.deprecated:
                continue  # Skip deprecated columns

            score = self._score_column(query_lower, query_terms, meta)

            # Apply null_ratio penalty
            penalty = 1.0 - min(meta.null_ratio, 0.98)
            adjusted_score = score * penalty

            if adjusted_score > 0:
                scored_columns.append((adjusted_score, col_id, meta))

        # Sort by score and take top_k
        scored_columns.sort(key=lambda x: x[0], reverse=True)

        for score, col_id, meta in scored_columns[:top_k]:
            allowed_columns.add(col_id)
            column_info[col_id] = meta
            if meta.cast_recipe:
                cast_recipes[col_id] = meta.cast_recipe
            elif meta.unit and meta.unit in self._cast_recipes:
                cast_recipes[col_id] = self._cast_recipes[meta.unit]

        # Derive allowed tables
        allowed_tables = {col.split(".")[0] for col in allowed_columns}

        return ReducedSchema(
            allowed_columns=allowed_columns,
            allowed_tables=allowed_tables,
            column_info=column_info,
            value_hints={},  # Will be populated by value resolution step
            core_columns=core_column_ids,
            cast_recipes=cast_recipes,
            allowed_joins=None,  # Single-table default
        )

    def _score_column(
        self,
        query_lower: str,
        query_terms: Set[str],
        meta: ColumnMetadata
    ) -> float:
        """
        Score a column's relevance to the query.

        Basic keyword matching - will be enhanced with embeddings in Phase 2.
        """
        score = 0.0

        # Check business name
        if meta.business_name.lower() in query_lower:
            score += 2.0

        # Check synonyms
        for syn in meta.synonyms:
            if syn.lower() in query_lower:
                score += 1.5
                break

        # Check individual terms
        for term in query_terms:
            if len(term) < 3:
                continue  # Skip short terms
            if term in meta.business_name.lower():
                score += 1.0
            for syn in meta.synonyms:
                if term in syn.lower():
                    score += 0.5
                    break

        # Check description
        if meta.description:
            desc_lower = meta.description.lower()
            for term in query_terms:
                if len(term) >= 4 and term in desc_lower:
                    score += 0.3

        return score

    def get_column_by_synonym(self, term: str) -> Optional[ColumnMetadata]:
        """Look up a column by synonym."""
        self.initialize()
        term_lower = term.lower()
        col_ids = self._synonyms_index.get(term_lower, [])
        if col_ids:
            col_id = col_ids[0]
            return self._columns.get(col_id) or self._core_columns.get(col_id)
        return None

    def get_all_columns(self, include_deprecated: bool = False) -> List[ColumnMetadata]:
        """Get all columns from the catalog."""
        self.initialize()
        all_cols = list(self._core_columns.values()) + list(self._columns.values())
        if not include_deprecated:
            all_cols = [c for c in all_cols if not c.deprecated]
        return all_cols

    def get_usable_columns(self) -> List[ColumnMetadata]:
        """Get columns that are usable (not deprecated, has data)."""
        self.initialize()
        return [c for c in self.get_all_columns() if c.is_usable]

    # =========================================================================
    # Phase 2: Vector-Based Retrieval (Optional Enhancement)
    # =========================================================================

    def enable_vector_search(
        self,
        embedding_service: "EmbeddingService",
        pinecone_index: Any,
        namespace: str = "column-metadata"
    ) -> None:
        """
        Enable vector-based column retrieval.

        Args:
            embedding_service: EmbeddingService for generating embeddings
            pinecone_index: Pinecone index instance
            namespace: Namespace for column vectors
        """
        self._embedding_service = embedding_service
        self._pinecone_index = pinecone_index
        self._vector_namespace = namespace
        self._vector_enabled = True
        logger.info("[SchemaLinker] Vector search enabled")

    async def index_columns_for_vector_search(self) -> int:
        """
        Index all columns into Pinecone for vector search.

        Returns:
            Number of columns indexed
        """
        if not getattr(self, "_vector_enabled", False):
            logger.warning("[SchemaLinker] Vector search not enabled")
            return 0

        self.initialize()

        # Build texts for embedding
        texts = []
        col_ids = []

        for col_id, meta in self._columns.items():
            if meta.deprecated:
                continue
            text = meta.to_embedding_text()
            texts.append(text)
            col_ids.append(col_id)

        if not texts:
            return 0

        # Generate embeddings
        embeddings = await self._embedding_service.embed_texts(texts)

        # Prepare vectors for upsert
        vectors = []
        for col_id, embedding, meta in zip(col_ids, embeddings, [self._columns[c] for c in col_ids]):
            vectors.append({
                "id": col_id,
                "values": embedding,
                "metadata": {
                    "table": meta.column_id.table,
                    "column": meta.column_id.column,
                    "qualified": col_id,
                    "business_name": meta.business_name,
                    "data_type": meta.data_type,
                    "unit": meta.unit or "",
                    "null_ratio": meta.null_ratio,
                }
            })

        # Upsert to Pinecone
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self._pinecone_index.upsert(
                vectors=batch,
                namespace=self._vector_namespace
            )

        logger.info(f"[SchemaLinker] Indexed {len(vectors)} columns for vector search")
        return len(vectors)

    async def get_reduced_schema_hybrid(
        self,
        query: str,
        top_k: int = 15,
        include_core: bool = True,
        vector_weight: float = 0.6,
    ) -> ReducedSchema:
        """
        Get reduced schema using hybrid keyword + vector retrieval.

        Args:
            query: User's natural language query
            top_k: Maximum number of property columns to include
            include_core: Whether to always include core columns
            vector_weight: Weight for vector scores (0-1), rest is keyword

        Returns:
            ReducedSchema with relevant columns
        """
        if not getattr(self, "_vector_enabled", False):
            # Fall back to keyword-only
            return self.get_reduced_schema(query, top_k, include_core)

        self.initialize()

        # Get keyword scores
        keyword_scores = self._get_keyword_scores(query)

        # Get vector scores
        vector_scores = await self._get_vector_scores(query, top_k * 2)

        # Combine scores with weights
        keyword_weight = 1.0 - vector_weight
        all_col_ids = set(keyword_scores.keys()) | set(vector_scores.keys())

        combined_scores: List[tuple] = []
        for col_id in all_col_ids:
            meta = self._columns.get(col_id)
            if not meta or meta.deprecated:
                continue

            kw_score = keyword_scores.get(col_id, 0.0)
            vec_score = vector_scores.get(col_id, 0.0)

            # Normalize keyword score (rough normalization)
            kw_normalized = min(kw_score / 5.0, 1.0)

            combined = (keyword_weight * kw_normalized) + (vector_weight * vec_score)

            # Apply null_ratio penalty
            penalty = 1.0 - min(meta.null_ratio, 0.98)
            adjusted = combined * penalty

            combined_scores.append((adjusted, col_id, meta))

        # Sort and build result
        combined_scores.sort(key=lambda x: x[0], reverse=True)

        allowed_columns: Set[str] = set()
        column_info: Dict[str, ColumnMetadata] = {}
        cast_recipes: Dict[str, str] = {}
        core_column_ids: Set[str] = set()

        # Add core columns if requested
        if include_core:
            for col_id, meta in self._core_columns.items():
                allowed_columns.add(col_id)
                column_info[col_id] = meta
                core_column_ids.add(col_id)

        # Add top-K retrieved columns
        for score, col_id, meta in combined_scores[:top_k]:
            allowed_columns.add(col_id)
            column_info[col_id] = meta
            if meta.cast_recipe:
                cast_recipes[col_id] = meta.cast_recipe
            elif meta.unit and meta.unit in self._cast_recipes:
                cast_recipes[col_id] = self._cast_recipes[meta.unit]

        allowed_tables = {col.split(".")[0] for col in allowed_columns}

        return ReducedSchema(
            allowed_columns=allowed_columns,
            allowed_tables=allowed_tables,
            column_info=column_info,
            value_hints={},
            core_columns=core_column_ids,
            cast_recipes=cast_recipes,
            allowed_joins=None,
        )

    def _get_keyword_scores(self, query: str) -> Dict[str, float]:
        """Get keyword-based scores for all columns."""
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        scores = {}
        for col_id, meta in self._columns.items():
            if meta.deprecated:
                continue
            score = self._score_column(query_lower, query_terms, meta)
            if score > 0:
                scores[col_id] = score

        return scores

    async def _get_vector_scores(self, query: str, top_k: int) -> Dict[str, float]:
        """Get vector-based scores from Pinecone."""
        try:
            query_embedding = await self._embedding_service.embed_query(query)

            results = self._pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=self._vector_namespace,
                include_metadata=True
            )

            scores = {}
            for match in results.matches:
                scores[match.id] = match.score

            return scores
        except Exception as e:
            logger.warning(f"[SchemaLinker] Vector search failed: {e}")
            return {}


# Global singleton instance
schema_linker = SchemaLinker()
