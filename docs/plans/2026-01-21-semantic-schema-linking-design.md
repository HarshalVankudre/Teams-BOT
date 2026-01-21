# Semantic Schema Linking for Text-to-SQL

**Date:** 2026-01-21
**Status:** Design Review - Revision 2
**Author:** AI Assistant
**Reviewer Feedback:** Incorporated

---

## Executive Summary

This document proposes a semantic schema linking system to replace the current approach of feeding the LLM the entire database schema. The new system will:

1. Use **semantic retrieval** to find relevant columns for each query
2. Generate SQL on a **reduced schema** (top-K columns instead of 180+)
3. Add a **validation + auto-repair loop** to fix invalid SQL
4. Build a **value index** for categorical columns
5. Implement **alias learning** from successful queries

Expected outcomes:
- Higher query accuracy (from ~70% to ~90%+)
- Reduced token usage (smaller prompts)
- Automatic handling of synonyms without hardcoding
- Self-improving system that learns from usage

---

## Revision 2 Changes (Based on Design Review)

| Issue | Fix |
|-------|-----|
| Column IDs not fully-qualified | All IDs now `table.column` format |
| Interface mismatches in validator | New `ReducedSchema` object passed everywhere |
| Join strategy undefined | Single-table default, explicit join graph required for multi-table |
| Value matching via raw literals | Extract predicate pairs from AST |
| Core columns `score=1.0` risks ranking | Separate core columns list, forced inclusion without affecting rank |
| Dead columns not down-ranked | `null_ratio` penalty in retrieval scoring |
| Alias uniqueness fails with NULLs | Split into two tables: column_aliases + value_aliases |
| No observability metrics | Added retrieval recall, validation failure breakdown |
| No golden set harness | Phase 0 creates test harness before implementation |

---

## 1. Current Architecture & Limitations

### Current Flow

```
User Query
    |
    v
[SQLGuard: Intent Detection]
    |
    v
[SingleAgent: Full Schema in Prompt (~180 columns)]
    |
    v
[LLM generates SQL]
    |
    v
[Basic validation: safety checks only]
    |
    v
[Execute SQL]
```

### Current Limitations

| Problem | Example | Impact |
|---------|---------|--------|
| **Full schema overload** | 180+ columns in prompt | LLM guesses, hallucinations |
| **No semantic matching** | User: "Kette" → LLM tries `prop_e2100_mobil_kette` (empty) | Wrong column selection |
| **No value matching** | User: "Bomag" vs DB: "BOMAG" | Filter misses |
| **No repair loop** | Invalid SQL → error | User must rephrase |
| **Manual synonyms** | Hardcoded column mappings | Doesn't scale |
| **Dead columns returned** | `prop_e2100_mobil_kette` (99.9% NULL) | Useless results |

---

## 2. Proposed Architecture

### New Flow (2-Step Pipeline)

```
User Query
    |
    v
[Step 0: Intent Detection] - unchanged
    |
    v
[Step 1: SEMANTIC COLUMN RETRIEVAL] ← NEW
    |   - Embed user query
    |   - Retrieve top-K columns (with null_ratio penalty)
    |   - Force-include core columns (separate list)
    |   - Output: ReducedSchema object
    |
    v
[Step 2: SQL GENERATION with Structured Output] ← MODIFIED
    |   - Prompt contains ONLY relevant columns (as table.column)
    |   - Request JSON output: {sql, columns_used, filters}
    |   - Single-table default (equipment_matrix)
    |
    v
[Step 3: VALIDATION + AUTO-REPAIR] ← NEW
    |   - Parse SQL (sqlglot)
    |   - Extract predicate pairs (column, op, value) from AST
    |   - Check columns exist in ReducedSchema.allowed_columns
    |   - Check values against value_index (warnings, not hard errors)
    |   - If invalid: repair prompt → retry (max 2 attempts)
    |
    v
[Step 4: Execute SQL]
    |
    v
[Step 5: ALIAS LEARNING] ← NEW (gated)
    |   - On success: log column/value mappings
    |   - Only use aliases with use_count >= 3 AND confidence >= 0.75
```

---

## 3. Core Data Structures

### 3.1 Canonical Column ID

**CRITICAL:** All column references use fully-qualified `table.column` format.

```python
@dataclass
class ColumnID:
    """Canonical column identifier - always table.column"""
    table: str
    column: str

    @property
    def qualified(self) -> str:
        return f"{self.table}.{self.column}"

    @classmethod
    def from_string(cls, s: str) -> "ColumnID":
        parts = s.split(".", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid column ID: {s}, expected table.column")
        return cls(table=parts[0], column=parts[1])

    def __hash__(self):
        return hash(self.qualified)

    def __eq__(self, other):
        return self.qualified == other.qualified
```

### 3.2 ReducedSchema Object

**Purpose:** Single object passed through retrieval → generation → validation.

```python
@dataclass
class ReducedSchema:
    """
    Reduced schema for SQL generation.
    Contains only relevant columns for current query.
    """
    # Allowed columns (as table.column strings)
    allowed_columns: Set[str]

    # Allowed tables (derived from columns)
    allowed_tables: Set[str]

    # Column metadata for prompt generation
    column_info: Dict[str, ColumnMetadata]  # table.column -> metadata

    # Value hints per categorical column
    value_hints: Dict[str, List[ValueHint]]  # table.column -> hints

    # Join graph (if multi-table allowed)
    allowed_joins: Optional[List[JoinPath]] = None

    # Core columns (always included, not from retrieval)
    core_columns: Set[str] = field(default_factory=set)

    # Casting recipes for NUMERIC_TEXT columns
    cast_recipes: Dict[str, str] = field(default_factory=dict)

    def is_column_allowed(self, table: str, column: str) -> bool:
        return f"{table}.{column}" in self.allowed_columns

    def get_cast_recipe(self, column_id: str) -> Optional[str]:
        """Get SQL cast expression for NUMERIC_TEXT columns."""
        return self.cast_recipes.get(column_id)

    def to_prompt(self) -> str:
        """Generate prompt section for LLM."""
        lines = [
            "AVAILABLE COLUMNS (use ONLY these, always qualify as table.column):",
            "-" * 50
        ]

        for col_id in sorted(self.allowed_columns):
            meta = self.column_info.get(col_id)
            if meta:
                entry = f"  {col_id}: {meta.business_name}"
                if meta.unit:
                    entry += f" [{meta.unit}]"
                if meta.examples:
                    entry += f" (e.g., {', '.join(meta.examples[:3])})"
                if meta.cast_recipe:
                    entry += f" [use: {meta.cast_recipe}]"
            else:
                entry = f"  {col_id}"
            lines.append(entry)

        if self.value_hints:
            lines.append("")
            lines.append("VALUE HINTS (suggested values, verify with ILIKE if uncertain):")
            lines.append("-" * 50)
            for col_id, hints in self.value_hints.items():
                values = [h.value for h in hints[:5]]
                lines.append(f"  {col_id}: {', '.join(values)}")

        if self.allowed_joins:
            lines.append("")
            lines.append("ALLOWED JOINS:")
            lines.append("-" * 50)
            for jp in self.allowed_joins:
                lines.append(f"  {jp.from_table}.{jp.from_col} = {jp.to_table}.{jp.to_col}")

        return "\n".join(lines)


@dataclass
class ColumnMetadata:
    """Metadata for a single column."""
    business_name: str
    description: str
    data_type: str  # TEXT, NUMERIC, NUMERIC_TEXT, CATEGORICAL, BOOLEAN
    unit: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    null_ratio: float = 0.0
    cast_recipe: Optional[str] = None  # SQL expression for type conversion


@dataclass
class ValueHint:
    """Suggested value for a categorical column."""
    value: str
    confidence: float
    match_type: str  # exact, fuzzy, semantic


@dataclass
class JoinPath:
    """Allowed join between tables."""
    from_table: str
    from_col: str
    to_table: str
    to_col: str
    join_type: str = "INNER"  # INNER, LEFT, etc.
```

---

## 4. Component Design

### 4.1 Semantic Column Catalog

**Storage:** YAML file (`sql_export/column_catalog.yaml`) + versioning hash.

```yaml
# column_catalog.yaml
version: "1.0.0"  # Increment on changes, triggers re-indexing

columns:
  - table: equipment_matrix
    column: prop_e1740_grabtiefe_mm
    business_name: Grabtiefe
    description: |
      Maximale Grabtiefe des Baggers in Millimetern.
      Relevant für Erdarbeiten, Aushub, Fundamentbau.
    synonyms:
      - grabtiefe
      - tiefe
      - digging depth
      - excavation depth
      - aushubtiefe
    data_type: NUMERIC_TEXT
    unit: mm
    examples: ["3410 mm", "5720 mm", "7000 mm"]
    cast_recipe: "CAST(NULLIF(regexp_replace({col}, '[^0-9]', '', 'g'), '') AS NUMERIC)"
    common_filters: [">=", "<=", "BETWEEN"]

  - table: equipment_matrix
    column: geraetegruppe_name
    business_name: Gerätegruppe / Equipment Type
    description: |
      Kategorie der Maschine. Enthält Kette/Mobil/Rad-Unterscheidung.
      WICHTIG: Für Kette vs Mobil immer diese Spalte verwenden!
    synonyms:
      - gerätegruppe
      - maschinentyp
      - equipment type
      - kategorie
      - kette
      - mobil
      - kettenbagger
      - mobilbagger
    data_type: CATEGORICAL
    examples: ["Kettenbagger", "Mobilbagger", "Kettenfertiger"]

  - table: equipment_matrix
    column: prop_e2100_mobil_kette
    business_name: Mobil - Kette (DEPRECATED)
    description: |
      WARNUNG: Diese Spalte ist zu 99.9% NULL und sollte NICHT verwendet werden.
      Für Kette/Mobil-Unterscheidung nutze geraetegruppe_name.
    data_type: BOOLEAN
    null_ratio: 0.999
    deprecated: true
    alternative: equipment_matrix.geraetegruppe_name
```

### 4.2 Column Vector Index with null_ratio Penalty

```python
class ColumnVectorIndex:
    """
    Vector index over column metadata for semantic retrieval.
    Uses Pinecone with dedicated namespace.
    """

    NAMESPACE = "column-metadata"

    # Core columns - always included, never ranked
    CORE_COLUMNS = frozenset([
        "equipment_matrix.id",
        "equipment_matrix.bezeichnung",
        "equipment_matrix.seriennummer",
        "equipment_matrix.hersteller_name",
        "equipment_matrix.geraetegruppe_name",
        "equipment_matrix.verwendung_code",
        "equipment_matrix.nuclos_state",
    ])

    def __init__(self, pinecone_client, embedding_model):
        self.pinecone = pinecone_client
        self.embedder = embedding_model
        self._catalog_version: Optional[str] = None
        self._column_null_ratios: Dict[str, float] = {}

    def index_columns(self, columns: List[SemanticColumnInfo], catalog_version: str):
        """
        Index all column metadata.
        Only re-indexes if catalog version changed.
        """
        if self._catalog_version == catalog_version:
            return  # Already indexed this version

        for col in columns:
            col_id = f"{col.table_name}.{col.column_name}"

            # Store null_ratio for scoring penalty
            self._column_null_ratios[col_id] = col.null_ratio

            # Skip deprecated columns
            if col.deprecated:
                continue

            text = self._build_embedding_text(col)
            embedding = self.embedder.embed(text)

            self.pinecone.upsert(
                namespace=self.NAMESPACE,
                vectors=[{
                    "id": col_id,
                    "values": embedding,
                    "metadata": {
                        "table": col.table_name,
                        "column": col.column_name,
                        "qualified": col_id,
                        "business_name": col.business_name,
                        "data_type": col.data_type,
                        "unit": col.unit,
                        "examples": col.example_values[:5],
                        "null_ratio": col.null_ratio,
                        "cast_recipe": col.cast_recipe,
                    }
                }]
            )

        self._catalog_version = catalog_version

    def retrieve_columns(
        self,
        query: str,
        top_k: int = 15,
    ) -> ReducedSchema:
        """
        Retrieve most relevant columns for a query.

        Returns ReducedSchema with:
        - Retrieved columns (ranked by semantic score * null_ratio penalty)
        - Core columns (always included, not ranked)
        """
        query_embedding = self.embedder.embed(query)

        # Retrieve more than needed, then re-rank
        results = self.pinecone.query(
            namespace=self.NAMESPACE,
            vector=query_embedding,
            top_k=top_k * 2,
            include_metadata=True
        )

        # Apply null_ratio penalty to scores
        scored_matches = []
        for r in results.matches:
            null_ratio = r.metadata.get("null_ratio", 0.0)
            # Penalty: final_score = semantic_score * (1 - clamp(null_ratio, 0, 0.98))
            penalty = 1.0 - min(null_ratio, 0.98)
            adjusted_score = r.score * penalty

            scored_matches.append((adjusted_score, r))

        # Sort by adjusted score
        scored_matches.sort(key=lambda x: x[0], reverse=True)

        # Build column sets
        allowed_columns: Set[str] = set()
        column_info: Dict[str, ColumnMetadata] = {}
        cast_recipes: Dict[str, str] = {}

        # Add top-K retrieved columns
        for score, r in scored_matches[:top_k]:
            col_id = r.metadata["qualified"]
            allowed_columns.add(col_id)

            column_info[col_id] = ColumnMetadata(
                business_name=r.metadata["business_name"],
                description="",  # Not stored in vector
                data_type=r.metadata["data_type"],
                unit=r.metadata.get("unit"),
                examples=r.metadata.get("examples", []),
                null_ratio=r.metadata.get("null_ratio", 0.0),
                cast_recipe=r.metadata.get("cast_recipe"),
            )

            if r.metadata.get("cast_recipe"):
                cast_recipes[col_id] = r.metadata["cast_recipe"]

        # Add core columns (always included, separate from ranking)
        core_columns = set(self.CORE_COLUMNS)
        allowed_columns.update(core_columns)

        # Derive allowed tables
        allowed_tables = {col.split(".")[0] for col in allowed_columns}

        return ReducedSchema(
            allowed_columns=allowed_columns,
            allowed_tables=allowed_tables,
            column_info=column_info,
            value_hints={},  # Populated by value resolution step
            core_columns=core_columns,
            cast_recipes=cast_recipes,
            allowed_joins=None,  # Single-table default
        )
```

### 4.3 Join Strategy: Single-Table Default

```python
class JoinPolicy:
    """
    Controls when multi-table queries are allowed.

    DEFAULT: Single-table only (equipment_matrix)
    EXPLICIT: Multi-table allowed only with explicit join graph
    """

    # Current policy: single-table
    ALLOWED_TABLES = frozenset(["equipment_matrix"])

    # If we need joins later, define explicit paths
    ALLOWED_JOINS: List[JoinPath] = [
        # Example (not currently used):
        # JoinPath("equipment_matrix", "hersteller_code", "manufacturers", "code"),
    ]

    @classmethod
    def validate_tables(cls, tables: Set[str]) -> List[str]:
        """Return errors if tables violate policy."""
        errors = []
        disallowed = tables - cls.ALLOWED_TABLES
        if disallowed:
            errors.append(
                f"Query references disallowed tables: {disallowed}. "
                f"Only these tables are allowed: {cls.ALLOWED_TABLES}"
            )
        return errors

    @classmethod
    def get_join_graph(cls) -> Optional[List[JoinPath]]:
        """Return allowed joins if multi-table is enabled."""
        if len(cls.ALLOWED_TABLES) > 1:
            return cls.ALLOWED_JOINS
        return None
```

### 4.4 SQL Validator with Predicate Extraction

```python
import sqlglot
from sqlglot import exp

class SQLValidator:
    """
    Validate and repair generated SQL using AST analysis.

    Key fix from review: Extract predicate pairs (column, op, value)
    from AST, not raw string literals.
    """

    def validate(
        self,
        sql: str,
        schema: ReducedSchema,
        value_index: "ValueIndex"
    ) -> "ValidationResult":
        """Validate SQL against reduced schema."""

        issues: List[str] = []
        warnings: List[str] = []
        suggestions: List[str] = []

        # 1. Parse SQL
        try:
            parsed = sqlglot.parse_one(sql, dialect="postgres")
        except Exception as e:
            return ValidationResult(
                valid=False,
                issues=[f"Syntax error: {e}"],
                suggestions=["Check SQL syntax"],
                schema=schema,
            )

        # 2. Extract referenced tables
        tables = self._extract_tables(parsed)
        table_errors = JoinPolicy.validate_tables(tables)
        issues.extend(table_errors)

        # 3. Extract referenced columns (as table.column)
        columns = self._extract_qualified_columns(parsed, default_table="equipment_matrix")

        # 4. Check columns exist in schema
        for col_id in columns:
            if col_id not in schema.allowed_columns:
                issues.append(f"Unknown column: {col_id}")
                closest = self._find_closest(col_id, schema.allowed_columns)
                if closest:
                    suggestions.append(f"Did you mean '{closest}' instead of '{col_id}'?")

        # 5. Extract predicate pairs (column, operator, value) from AST
        predicates = self._extract_predicates(parsed)

        for pred in predicates:
            col_id = pred.column_id
            value = pred.value

            # Check if column is categorical and value might be wrong
            if col_id in value_index.categorical_columns:
                matches = value_index.find_matching_values(col_id, value)
                if not matches:
                    # WARNING, not error - allow ILIKE fallback
                    warnings.append(
                        f"Value '{value}' not found in {col_id}. "
                        f"Consider using ILIKE or check spelling."
                    )
                    close_vals = value_index.get_close_values(col_id, value, limit=3)
                    if close_vals:
                        suggestions.append(f"Similar values: {', '.join(close_vals)}")
                elif matches[0].match_type != "exact":
                    # Found fuzzy match - suggest correction
                    suggestions.append(
                        f"'{value}' → '{matches[0].value}' ({matches[0].match_type} match)"
                    )

        # 6. Check GROUP BY completeness
        group_by_issues = self._check_group_by(parsed)
        issues.extend(group_by_issues)

        return ValidationResult(
            valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions,
            parsed=parsed,
            schema=schema,
            predicates=predicates,
        )

    def _extract_predicates(self, parsed: exp.Expression) -> List["Predicate"]:
        """
        Extract predicate pairs from SQL AST.

        Handles:
        - col = 'X'
        - col IN ('X', 'Y')
        - col ILIKE '%X%'
        - LOWER(col) = 'x'
        """
        predicates = []

        for node in parsed.walk():
            # Handle: col = 'value'
            if isinstance(node, exp.EQ):
                col, val = self._extract_eq_predicate(node)
                if col and val:
                    predicates.append(Predicate(column_id=col, op="=", value=val))

            # Handle: col IN ('a', 'b', 'c')
            elif isinstance(node, exp.In):
                col, vals = self._extract_in_predicate(node)
                if col and vals:
                    for v in vals:
                        predicates.append(Predicate(column_id=col, op="IN", value=v))

            # Handle: col ILIKE '%x%'
            elif isinstance(node, exp.ILike):
                col, pattern = self._extract_like_predicate(node)
                if col and pattern:
                    predicates.append(Predicate(column_id=col, op="ILIKE", value=pattern))

        return predicates

    def _extract_eq_predicate(self, node: exp.EQ) -> tuple:
        """Extract column and value from equality predicate."""
        left, right = node.left, node.right

        # Check both orders: col = 'val' and 'val' = col
        if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
            col_id = self._qualify_column(left)
            return col_id, str(right.this)
        elif isinstance(right, exp.Column) and isinstance(left, exp.Literal):
            col_id = self._qualify_column(right)
            return col_id, str(left.this)

        return None, None

    def _qualify_column(self, col: exp.Column, default_table: str = "equipment_matrix") -> str:
        """Convert Column AST node to qualified table.column string."""
        table = col.table or default_table
        return f"{table}.{col.name}"


@dataclass
class Predicate:
    """A filter predicate extracted from SQL."""
    column_id: str  # table.column
    op: str         # =, IN, ILIKE, >=, etc.
    value: str      # The literal value


@dataclass
class ValidationResult:
    """Result of SQL validation."""
    valid: bool
    issues: List[str]
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    parsed: Optional[exp.Expression] = None
    schema: Optional[ReducedSchema] = None
    predicates: List[Predicate] = field(default_factory=list)
```

### 4.5 Structured Output for SQL Generation

**Key improvement:** Request JSON output instead of raw SQL.

```python
SQL_GENERATION_PROMPT = """
Generate a SQL query to answer the user's question.

USER QUESTION: {question}

{reduced_schema_prompt}

RULES:
1. Use ONLY columns from the AVAILABLE COLUMNS list above
2. Always qualify columns as table.column (e.g., equipment_matrix.bezeichnung)
3. For NUMERIC_TEXT columns, use the cast recipe shown
4. For categorical filters, use the VALUE HINTS if provided
5. Single table queries only (no JOINs)

Respond with JSON in this exact format:
{{
    "sql": "SELECT ... FROM equipment_matrix WHERE ...",
    "columns_used": ["equipment_matrix.col1", "equipment_matrix.col2"],
    "filters": [
        {{"column": "equipment_matrix.col1", "op": "=", "value": "X"}},
        {{"column": "equipment_matrix.col2", "op": ">=", "value": "100"}}
    ],
    "reasoning": "Brief explanation of column choices"
}}
"""
```

### 4.6 Alias Learning Tables (Split for NULL Uniqueness)

```sql
-- Learned column aliases (when user term maps to column)
CREATE TABLE learned_column_aliases (
    id SERIAL PRIMARY KEY,
    alias_text VARCHAR(255) NOT NULL,
    target_table VARCHAR(255) NOT NULL,
    target_column VARCHAR(255) NOT NULL,
    confidence FLOAT DEFAULT 0.55,
    use_count INT DEFAULT 1,
    success_count INT DEFAULT 1,
    failure_count INT DEFAULT 0,
    source VARCHAR(50) DEFAULT 'auto',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(alias_text, target_table, target_column)
);

-- Learned value aliases (when user term maps to DB value)
CREATE TABLE learned_value_aliases (
    id SERIAL PRIMARY KEY,
    alias_text VARCHAR(255) NOT NULL,
    target_table VARCHAR(255) NOT NULL,
    target_column VARCHAR(255) NOT NULL,
    target_value VARCHAR(500) NOT NULL,
    confidence FLOAT DEFAULT 0.55,
    use_count INT DEFAULT 1,
    success_count INT DEFAULT 1,
    failure_count INT DEFAULT 0,
    source VARCHAR(50) DEFAULT 'auto',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(alias_text, target_table, target_column, target_value)
);

-- Indexes for fast lookup (exact match on normalized text)
CREATE INDEX idx_col_aliases_text ON learned_column_aliases(LOWER(alias_text));
CREATE INDEX idx_val_aliases_text ON learned_value_aliases(LOWER(alias_text));
```

**Alias Learning Gating:**

```python
class AliasLearner:
    """
    Learn column/value mappings from successful queries.

    GATING RULES (from review):
    - Create aliases at low confidence (0.55)
    - Only USE aliases when:
      - use_count >= 3 AND
      - confidence >= 0.75 AND
      - failure_count < success_count
    """

    MIN_USE_COUNT = 3
    MIN_CONFIDENCE = 0.75

    def get_usable_column_aliases(self, term: str) -> List[ColumnAlias]:
        """Get aliases that pass gating threshold."""
        return self.db.query("""
            SELECT * FROM learned_column_aliases
            WHERE LOWER(alias_text) = LOWER(%s)
              AND use_count >= %s
              AND confidence >= %s
              AND failure_count < success_count
            ORDER BY confidence DESC
            LIMIT 3
        """, [term, self.MIN_USE_COUNT, self.MIN_CONFIDENCE])

    def record_outcome(self, alias_id: int, success: bool):
        """Update alias based on query outcome."""
        if success:
            self.db.execute("""
                UPDATE learned_column_aliases
                SET success_count = success_count + 1,
                    use_count = use_count + 1,
                    confidence = LEAST(0.95, confidence + 0.05 * (1 - confidence)),
                    updated_at = NOW()
                WHERE id = %s
            """, [alias_id])
        else:
            self.db.execute("""
                UPDATE learned_column_aliases
                SET failure_count = failure_count + 1,
                    use_count = use_count + 1,
                    confidence = GREATEST(0.1, confidence - 0.1),
                    updated_at = NOW()
                WHERE id = %s
            """, [alias_id])
```

---

## 5. Observability & Metrics

### 5.1 Metrics to Track

| Metric | Description | Target |
|--------|-------------|--------|
| **retrieval_recall@K** | Does correct column appear in top K? | >95% at K=15 |
| **schema_reduction_size** | Columns in reduced schema | 10-20 |
| **validation_pass_rate** | % queries that pass validation first try | >80% |
| **repair_success_rate** | % failed validations fixed by repair | >80% |
| **value_match_confidence** | Average confidence of value matches | >0.8 |
| **empty_result_rate** | % queries returning 0 rows | <10% |
| **alias_usage_rate** | % queries using learned aliases | Track trend |

### 5.2 Logging Structure

```python
@dataclass
class QueryMetrics:
    """Metrics logged for each query."""
    query_id: str
    timestamp: datetime

    # Retrieval
    columns_retrieved: List[str]
    columns_used_in_sql: List[str]
    retrieval_recall: float  # 1.0 if all used columns were retrieved

    # Validation
    validation_attempts: int
    validation_issues: List[str]
    repair_applied: bool

    # Value matching
    value_matches: List[Dict]  # {column, user_value, db_value, confidence}

    # Outcome
    sql_executed: bool
    rows_returned: int
    execution_time_ms: int

    # Aliases
    aliases_used: List[str]
    aliases_created: List[str]
```

---

## 6. Implementation Plan

### Phase 0: Test Harness (Week 0.5)

**CRITICAL: Create before implementation**

1. Collect 100-300 real user queries (German + English)
2. Annotate with:
   - Expected columns
   - Expected filters (column, op, value)
   - Expected result count (approximate)
3. Create automated test runner
4. Establish baseline metrics with current system

### Phase 1: Foundation (Week 1)

1. Create `sql_export/column_catalog.yaml`
   - Auto-generate from existing `property_types.csv`
   - Add null_ratios from database
   - Manually enrich top 30 columns with descriptions/synonyms
2. Implement `ReducedSchema` and `ColumnID` data structures
3. Create `rag/schema_linker.py` with basic retrieval (no vector yet)

### Phase 2: Vector Index (Week 2)

1. Add Pinecone namespace `column-metadata`
2. Implement `ColumnVectorIndex` with null_ratio penalty
3. Add hybrid retrieval (vector + keyword)
4. Integrate into `SingleAgent`
5. **Test:** Measure retrieval_recall@15 on golden set

### Phase 3: Validation (Week 2-3)

1. Add `sqlglot` dependency
2. Implement `SQLValidator` with predicate extraction
3. Add structured output (JSON) for SQL generation
4. Implement repair loop (max 2 retries)
5. **Test:** Measure validation_pass_rate, repair_success_rate

### Phase 4: Value Index (Week 3)

1. Implement `ValueIndex` class
2. Index categorical columns with fuzzy matching
3. Integrate value hints into reduced schema prompt
4. **Test:** Measure value_match_confidence

### Phase 5: Alias Learning (Week 4)

1. Create `learned_column_aliases` and `learned_value_aliases` tables
2. Implement `AliasLearner` with gating
3. Add success/failure tracking
4. **Test:** Monitor alias_usage_rate over time

### Phase 6: Tuning (Ongoing)

1. Tune retrieval K and null_ratio penalty
2. Tune alias confidence thresholds
3. Add more columns to catalog as needed
4. Monitor and iterate

---

## 7. Resolved Open Questions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Embedding model | Start with `text-embedding-3-small` | Evaluate recall@15 on golden set; upgrade to `-large` if German recall is weak |
| Pinecone namespace | Separate `column-metadata` namespace | Simpler ops, easier rebuild |
| Alias threshold | use_count >= 3, confidence >= 0.75 | Prevents noisy self-learning |
| Repair model | Same as generation | Best dialect consistency; optimize later |

---

## 8. Example Scenarios (Updated)

### Scenario 1: "Kette oder Mobil?"

**Step 1: Retrieval**
- Query embedding finds: `equipment_matrix.geraetegruppe_name` (score: 0.92)
- `equipment_matrix.prop_e2100_mobil_kette` retrieved but penalized (null_ratio=0.999 → score: 0.92 * 0.02 = 0.018)
- Result: `geraetegruppe_name` ranked high, `prop_e2100_mobil_kette` ranked very low (excluded)

**Step 2: Generation**
- Reduced schema shows only `geraetegruppe_name` for mobility
- Structured output:
```json
{
  "sql": "SELECT equipment_matrix.geraetegruppe_name, COUNT(*) FROM equipment_matrix WHERE equipment_matrix.id IN (...) GROUP BY equipment_matrix.geraetegruppe_name",
  "columns_used": ["equipment_matrix.geraetegruppe_name"],
  "filters": []
}
```

**Step 3: Validation**
- All columns exist ✓
- No value issues ✓
- GROUP BY correct ✓

**Result:** Correct query, returns "2 Kettenbagger, 3 Mobilbagger"

### Scenario 2: "Bomag Walzen"

**Step 2: Generation**
- LLM generates: `WHERE equipment_matrix.hersteller_name = 'Bomag'`

**Step 3: Validation**
- Predicate extraction finds: `(equipment_matrix.hersteller_name, =, 'Bomag')`
- Value index lookup: "Bomag" → "BOMAG" (fuzzy match, 0.95)
- Warning: "Value 'Bomag' not exact match"
- Suggestion: "'Bomag' → 'BOMAG'"

**Repair:**
- LLM corrects to: `WHERE equipment_matrix.hersteller_name = 'BOMAG'`

**Result:** Correct query after 1 repair

---

## 9. Appendix: Casting Recipes for NUMERIC_TEXT

Common patterns for columns stored as TEXT but containing numbers:

```yaml
cast_recipes:
  # Millimeters: "3410 mm" → 3410
  mm: "CAST(NULLIF(regexp_replace({col}, '[^0-9]', '', 'g'), '') AS NUMERIC)"

  # Meters with comma: "3,20 m" → 3.20
  m_comma: "CAST(NULLIF(REPLACE(regexp_replace({col}, '[^0-9,]', '', 'g'), ',', '.'), '') AS NUMERIC)"

  # Kilograms: "15420 kg" → 15420
  kg: "CAST(NULLIF(regexp_replace({col}, '[^0-9]', '', 'g'), '') AS NUMERIC)"

  # Kilowatts: "75 kW" → 75
  kw: "CAST(NULLIF(regexp_replace({col}, '[^0-9]', '', 'g'), '') AS NUMERIC)"
```

These recipes are stored per-column in the catalog and included in the reduced schema prompt.
