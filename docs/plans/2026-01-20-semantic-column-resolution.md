# Semantic Column Resolution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace hardcoded column patterns with LLM-based semantic column resolution so users can query using any phrasing and the AI finds the correct database columns.

**Architecture:** Create a ColumnCatalog that loads all column metadata at startup and provides it to the LLM in the system prompt. The LLM uses semantic understanding to pick correct columns. Remove all hardcoded column patterns from sql_guard, schema, and property_resolver.

**Tech Stack:** Python, PostgreSQL, OpenAI API

---

### Task 1: Create ColumnCatalog Module

**Files:**
- Create: `rag/column_catalog.py`

**Step 1: Create the ColumnCatalog class**

```python
"""
Column Catalog for Semantic Column Resolution.

Loads all database columns at startup and provides them to the LLM
in a format that enables semantic understanding of user queries.
"""
import csv
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ColumnInfo:
    """Metadata for a database column."""
    column_name: str          # e.g., prop_e1740_grabtiefe_mm
    display_name: str         # e.g., Grabtiefe
    code: str                 # e.g., E1740
    unit: Optional[str]       # e.g., mm, m, kg, kW
    data_type: str            # TEXT, BOOLEAN, NUMERIC
    category: str             # dimension, boolean, identifier, category
    description: str          # Human-readable description
    parse_hint: Optional[str] # How to parse (e.g., "extract numeric from TEXT")


class ColumnCatalog:
    """
    Catalog of all database columns for semantic resolution.

    Loaded once at startup, cached for all queries.
    The LLM uses this catalog to understand user intent and pick correct columns.
    """

    # Core columns that are always available
    CORE_COLUMNS = {
        "id": "Unique identifier (BIGINT)",
        "inventarnummer": "Inventory number (TEXT)",
        "seriennummer": "Serial number (TEXT)",
        "bezeichnung": "Equipment name/model (TEXT)",
        "hersteller_name": "Manufacturer name (TEXT)",
        "hersteller_code": "Manufacturer code (TEXT)",
        "geraetegruppe_name": "Equipment category name (TEXT) - e.g., Kettenfertiger, Mobilbagger",
        "geraetegruppe_code": "Equipment category code (TEXT)",
        "verwendung_code": "Usage type: MIET (rental) or VK (sale)",
        "verwendung_name": "Usage type name: Vermietung or Verkauf",
        "nuclos_state": "Availability: Released (available) or Locked (not available)",
    }

    # Unit patterns for categorization
    UNIT_PATTERNS = {
        "mm": "millimeter",
        "m": "meter",
        "kg": "kilogram",
        "t": "ton",
        "kW": "kilowatt",
        "kVA": "kilovolt-ampere",
        "bar": "bar (pressure)",
        "l/min": "liters per minute",
        "m³": "cubic meter",
        "km/h": "kilometers per hour",
        "Hz": "hertz",
        "h": "hours",
        "U/min": "revolutions per minute",
    }

    def __init__(self):
        self._columns: Dict[str, ColumnInfo] = {}
        self._cached_prompt: Optional[str] = None
        self._initialized: bool = False

    def initialize(self, postgres_service: Optional[Any] = None) -> None:
        """
        Load column metadata from database and/or CSV.
        Called once at application startup.
        """
        if self._initialized:
            return

        # Load property types from CSV (always available)
        csv_path = Path(__file__).resolve().parent.parent / "sql_export" / "property_types.csv"
        self._load_from_csv(csv_path)

        # Try to get actual columns from database for validation
        if postgres_service and getattr(postgres_service, "available", False):
            self._validate_with_database(postgres_service)

        # Build the cached prompt section
        self._cached_prompt = self._build_prompt_section()
        self._initialized = True

        print(f"[ColumnCatalog] Initialized with {len(self._columns)} property columns")

    def _load_from_csv(self, csv_path: Path) -> None:
        """Load property metadata from CSV file."""
        if not csv_path.exists():
            print(f"[ColumnCatalog] CSV not found: {csv_path}")
            return

        try:
            with csv_path.open("r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    code = (row.get("code") or "").strip()
                    name = (row.get("name") or "").strip()
                    if code and name:
                        self._add_property(code, name)
        except Exception as e:
            print(f"[ColumnCatalog] CSV load error: {e}")

    def _add_property(self, code: str, name: str) -> None:
        """Add a property column to the catalog."""
        # Parse unit from name like "Grabtiefe [mm]" or "Gewicht [kg]"
        unit_match = re.search(r'\[([^\]]+)\]', name)
        unit = unit_match.group(1) if unit_match else None

        # Clean display name (remove unit suffix)
        display_name = re.sub(r'\s*\[[^\]]+\]', '', name).strip()

        # Generate column name: prop_e1740_grabtiefe_mm
        slug = self._slugify(name)
        column_name = f"prop_{code.lower()}_{slug}"

        # Determine category and data type
        category, data_type, parse_hint = self._categorize_column(name, unit)

        # Build description
        description = display_name
        if unit:
            description += f" [{unit}]"

        self._columns[column_name] = ColumnInfo(
            column_name=column_name,
            display_name=display_name,
            code=code,
            unit=unit,
            data_type=data_type,
            category=category,
            description=description,
            parse_hint=parse_hint,
        )

    def _slugify(self, text: str) -> str:
        """Convert display name to column slug."""
        # Normalize unicode
        import unicodedata
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        # Lowercase and replace non-alphanumeric with underscore
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)  # Remove unit brackets
        text = re.sub(r'[^a-z0-9]+', '_', text)
        text = text.strip('_')
        return text[:50] if text else "unknown"

    def _categorize_column(self, name: str, unit: Optional[str]) -> tuple:
        """Determine category, data type, and parse hint for a column."""
        name_lower = name.lower()

        # Boolean columns (no unit, typically yes/no features)
        boolean_indicators = [
            "allrad", "klimaanlage", "funkfernsteuerung", "kabine",
            "dieselpartikelfilter", "elektrostarter", "reversierbar",
            "drehbar", "oszillation", "schnellgang", "knicklenkung",
        ]
        if any(ind in name_lower for ind in boolean_indicators) and not unit:
            return "boolean", "TEXT", "Check for 'Ja' or IS NOT NULL"

        # Mobility type columns
        if "mobil -" in name_lower or name_lower in ["mobil - kette", "mobil - rad", "mobil - semi"]:
            return "mobility", "TEXT", "Check for 'Ja' or IS NOT NULL"

        # Dimension columns (have numeric units)
        if unit in ["mm", "m", "kg", "t", "kW", "kVA", "bar", "l/min", "m³", "km/h"]:
            return "dimension", "TEXT", f"Extract numeric: CAST(NULLIF(regexp_replace(col, '[^0-9]', '', 'g'), '') AS NUMERIC)"

        # Count/quantity columns
        if unit in ["Stück", "h", "U/min", "Hz"]:
            return "quantity", "TEXT", "Extract numeric value"

        # Type/category columns (no unit, descriptive)
        type_indicators = ["typ", "art", "hersteller", "klasse", "größe", "grosse"]
        if any(ind in name_lower for ind in type_indicators):
            return "type", "TEXT", None

        # Default to feature
        return "feature", "TEXT", None

    def _validate_with_database(self, postgres_service: Any) -> None:
        """Validate columns exist in database (optional enhancement)."""
        try:
            result = postgres_service.execute_query(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'equipment_matrix' AND column_name LIKE 'prop_%' "
                "LIMIT 1"
            )
            if result:
                print("[ColumnCatalog] Database validation: columns exist")
        except Exception as e:
            print(f"[ColumnCatalog] Database validation skipped: {e}")

    def _build_prompt_section(self) -> str:
        """Build the prompt section for the LLM."""
        lines = []
        lines.append("=" * 70)
        lines.append("PROPERTY COLUMNS CATALOG")
        lines.append("Use this catalog to find the correct column for user queries.")
        lines.append("The LLM should semantically match user terms to these columns.")
        lines.append("=" * 70)
        lines.append("")

        # Group by category
        categories = {}
        for col_name, info in sorted(self._columns.items()):
            if info.category not in categories:
                categories[info.category] = []
            categories[info.category].append(info)

        # Output each category
        category_order = ["dimension", "boolean", "mobility", "quantity", "type", "feature"]
        category_titles = {
            "dimension": "DIMENSIONS (numeric values - parse from TEXT)",
            "boolean": "BOOLEAN FEATURES (check for 'Ja' or IS NOT NULL)",
            "mobility": "MOBILITY TYPE (Kette/Rad/Semi)",
            "quantity": "QUANTITIES (counts, rates)",
            "type": "TYPE/CATEGORY (descriptive text)",
            "feature": "OTHER FEATURES",
        }

        for cat in category_order:
            if cat not in categories:
                continue
            cols = categories[cat]
            lines.append(f"\n{category_titles.get(cat, cat.upper())}:")
            lines.append("-" * 50)
            for info in cols:
                # Format: column_name -> DisplayName [unit] (code)
                entry = f"  {info.column_name}"
                entry += f" -> {info.display_name}"
                if info.unit:
                    entry += f" [{info.unit}]"
                lines.append(entry)
            lines.append("")

        # Add SQL pattern hints
        lines.append("SQL PATTERNS FOR TEXT NUMERIC COLUMNS:")
        lines.append("-" * 50)
        lines.append("  To filter numeric values stored as TEXT (e.g., '3410 mm - Millimeter'):")
        lines.append("  CAST(NULLIF(regexp_replace(column_name, '[^0-9]', '', 'g'), '') AS NUMERIC)")
        lines.append("")
        lines.append("  Example - Grabtiefe >= 3000mm:")
        lines.append("  WHERE CAST(NULLIF(regexp_replace(prop_e1740_grabtiefe_mm, '[^0-9]', '', 'g'), '') AS NUMERIC) >= 3000")
        lines.append("")
        lines.append("BOOLEAN COLUMNS:")
        lines.append("-" * 50)
        lines.append("  Check for 'Ja': WHERE column_name = 'Ja'")
        lines.append("  Check exists: WHERE column_name IS NOT NULL")
        lines.append("")

        return "\n".join(lines)

    def get_prompt_section(self) -> str:
        """Get the cached prompt section for LLM system prompt."""
        if not self._initialized:
            raise RuntimeError("ColumnCatalog not initialized. Call initialize() first.")
        return self._cached_prompt or ""

    def get_column_info(self, column_name: str) -> Optional[ColumnInfo]:
        """Get info for a specific column."""
        return self._columns.get(column_name)

    def get_all_columns(self) -> Dict[str, ColumnInfo]:
        """Get all property columns."""
        return dict(self._columns)

    def search_columns(self, term: str) -> List[ColumnInfo]:
        """Search columns by display name or description (for debugging)."""
        term_lower = term.lower()
        results = []
        for info in self._columns.values():
            if (term_lower in info.display_name.lower() or
                term_lower in info.column_name.lower() or
                term_lower in info.description.lower()):
                results.append(info)
        return results


# Global singleton instance
column_catalog = ColumnCatalog()
```

**Step 2: Verify module imports correctly**

Run: `python -c "from rag.column_catalog import column_catalog; print('Import OK')"`
Expected: `Import OK`

**Step 3: Commit**

```bash
git add rag/column_catalog.py
git commit -m "feat: add ColumnCatalog for semantic column resolution"
```

---

### Task 2: Simplify Schema Module

**Files:**
- Modify: `rag/schema.py`

**Step 1: Replace hardcoded schema with dynamic structure**

Replace the entire content of `rag/schema.py` with:

```python
"""
PostgreSQL Database Schema for the SEMA equipment database.

This module provides the base schema structure. Property columns are
loaded dynamically by ColumnCatalog - no hardcoded column patterns here.
"""
import os

_SCHEMA = os.getenv("POSTGRES_SCHEMA", "public")
_TABLE = os.getenv("POSTGRES_EQUIPMENT_TABLE", "equipment_matrix")
EQUIPMENT_TABLE_FQN = f"{_SCHEMA}.{_TABLE}"

# Core schema that doesn't change - property columns come from ColumnCatalog
DATABASE_SCHEMA = f"""
DATABASE: SEMA Equipment (public.equipment_matrix)

TABLE: {EQUIPMENT_TABLE_FQN}
Contains ~2400 equipment records with properties.

CORE COLUMNS (always use these):
- id (BIGINT) - Unique identifier
- inventarnummer (TEXT) - Inventory number
- seriennummer (TEXT) - Serial number
- bezeichnung (TEXT) - Equipment name/model
- hersteller_name (TEXT) - Manufacturer name (e.g., 'Bomag', 'Liebherr')
- hersteller_code (TEXT) - Manufacturer code
- geraetegruppe_name (TEXT) - Category name (e.g., 'Kettenfertiger', 'Mobilbagger', 'Radlader')
- geraetegruppe_code (TEXT) - Category code
- verwendung_code (TEXT) - Usage: 'MIET' (rental) or 'VK' (sale)
- verwendung_name (TEXT) - Usage name: 'Vermietung' or 'Verkauf'
- nuclos_state (TEXT) - Availability: 'Released' (available) or 'Locked' (unavailable)

EQUIPMENT CATEGORIES (geraetegruppe_name values):
Use exact names for filtering. Examples:
- Kettenfertiger, Radfertiger (pavers)
- Kettenbagger, Mobilbagger, Minibagger, Kompaktbagger (excavators)
- Radlader (wheel loaders)
- Walze, Tandemwalze (rollers)
- Kaltfraese (Kette), Kaltfraese (Rad) (cold milling machines)
- Telekran (Kette), Telekran (Rad) (cranes)

For unknown categories, query first:
SELECT DISTINCT geraetegruppe_name, COUNT(*)
FROM {EQUIPMENT_TABLE_FQN}
WHERE geraetegruppe_name ILIKE '%suchbegriff%'
GROUP BY geraetegruppe_name;

AVAILABILITY:
- Released = available/ready
- Locked = not available

USAGE:
- MIET = rental
- VK = sale/purchase

KOSTENSTELLE (cost center):
Stored in ibs_nuclet_geraete_kostenstelle as "CODE - Name"
Example: "200 - Mietpark", "100 - Handel"
Query with: WHERE ibs_nuclet_geraete_kostenstelle ILIKE '%200%'
"""

# Alias for backward compatibility
SQL_AGENT_SCHEMA = DATABASE_SCHEMA
```

**Step 2: Verify schema imports correctly**

Run: `python -c "from rag.schema import DATABASE_SCHEMA, EQUIPMENT_TABLE_FQN; print(f'Table: {EQUIPMENT_TABLE_FQN}'); print('OK')"`
Expected: Shows table name and `OK`

**Step 3: Commit**

```bash
git add rag/schema.py
git commit -m "refactor: simplify schema.py, remove hardcoded column patterns"
```

---

### Task 3: Simplify SQL Guard

**Files:**
- Modify: `rag/sql_guard.py`

**Step 1: Remove hardcoded column patterns, keep only safety checks**

The sql_guard.py has many hardcoded patterns like `_EQUIPMENT_PROP_RE`, `_AC_RE`, `_WEIGHT_RE`, etc.
We need to remove these and keep only:
- Safety checks (block DELETE, UPDATE, DROP, etc.)
- LIMIT enforcement
- Basic SQL validation

Find and remove these patterns (lines ~75-88):
```python
# DELETE THESE LINES:
_EQUIPMENT_PROP_RE = re.compile(...)
_RENTAL_RE = re.compile(...)
_SALES_RE = re.compile(...)
_AVAIL_RE = re.compile(...)
_NULL_RE = re.compile(...)
_AC_RE = re.compile(...)
_WEIGHT_RE = re.compile(...)
_HEAVIEST_RE = re.compile(...)
_POWER_RE = re.compile(...)
_WIDTH_RE = re.compile(...)
```

Keep the safety patterns and document-related patterns.

In the `SQLGuard` class, simplify `extract_intent()` to not rely on hardcoded column patterns.
The LLM will handle column selection - the guard only needs to:
1. Detect if SQL is needed vs documents
2. Block unsafe operations
3. Enforce limits

**Step 2: Verify sql_guard still works**

Run: `python -c "from rag.sql_guard import SQLGuard; print('Import OK')"`
Expected: `Import OK`

**Step 3: Commit**

```bash
git add rag/sql_guard.py
git commit -m "refactor: remove hardcoded column patterns from sql_guard"
```

---

### Task 4: Simplify SQL Verifier

**Files:**
- Modify: `rag/sql_verifier.py`

**Step 1: Remove column validation, keep only safety checks**

The sql_verifier should only:
1. Block unsafe operations (DELETE, UPDATE, DROP, etc.)
2. Provide suggestions (not errors) for missing LIMIT
3. NOT validate column names (LLM picks correct columns)

Simplify the `verify()` method to only check for unsafe patterns.
Remove or simplify `_extract_column_refs()` since we no longer validate columns.

**Step 2: Verify sql_verifier works**

Run: `python -c "from rag.sql_verifier import SQLVerifier; print('Import OK')"`
Expected: `Import OK`

**Step 3: Commit**

```bash
git add rag/sql_verifier.py
git commit -m "refactor: simplify sql_verifier to safety checks only"
```

---

### Task 5: Integrate ColumnCatalog into SingleAgent

**Files:**
- Modify: `rag/single_agent.py`

**Step 1: Import and initialize ColumnCatalog**

At the top of `single_agent.py`, add import:
```python
from .column_catalog import column_catalog
```

In the `SingleAgent.__init__()` method, initialize the catalog:
```python
# Initialize column catalog (loads once, cached)
column_catalog.initialize(self.postgres)
```

**Step 2: Add column catalog to system prompt**

In the `process()` method where the system prompt is built, add the column catalog section:
```python
# Add column catalog for semantic column resolution
column_catalog_section = column_catalog.get_prompt_section()
system_prompt = f"{system_prompt}\n\n{column_catalog_section}"
```

**Step 3: Remove PropertyResolver dependency**

Find and remove any usage of `PropertyResolver` or `property_resolver` in the file.
The ColumnCatalog replaces this functionality.

**Step 4: Verify SingleAgent works**

Run: `python -c "from rag.single_agent import SingleAgent; print('Import OK')"`
Expected: `Import OK`

**Step 5: Commit**

```bash
git add rag/single_agent.py
git commit -m "feat: integrate ColumnCatalog into SingleAgent system prompt"
```

---

### Task 6: Remove PropertyResolver

**Files:**
- Delete: `rag/property_resolver.py` (or keep as deprecated)
- Modify: Any files that import it

**Step 1: Find all usages of PropertyResolver**

Search for imports:
```bash
grep -r "property_resolver\|PropertyResolver" rag/
```

**Step 2: Remove or update imports**

For each file that imports PropertyResolver:
- If it's only used for column resolution, remove the import
- The ColumnCatalog now handles this

**Step 3: Delete or deprecate the file**

Option A: Delete `rag/property_resolver.py`
Option B: Keep but add deprecation warning at top

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: remove PropertyResolver, replaced by ColumnCatalog"
```

---

### Task 7: Test Semantic Resolution

**Files:**
- Create: `tests/test_column_catalog.py`

**Step 1: Write tests for ColumnCatalog**

```python
"""Tests for ColumnCatalog semantic column resolution."""
import pytest
from rag.column_catalog import ColumnCatalog, column_catalog


class TestColumnCatalog:
    def test_initialization(self):
        """Test catalog initializes and loads columns."""
        catalog = ColumnCatalog()
        catalog.initialize()
        assert len(catalog.get_all_columns()) > 100  # Should have 170+ columns

    def test_get_prompt_section(self):
        """Test prompt section is generated."""
        catalog = ColumnCatalog()
        catalog.initialize()
        prompt = catalog.get_prompt_section()
        assert "PROPERTY COLUMNS CATALOG" in prompt
        assert "prop_e1740_grabtiefe" in prompt.lower()
        assert "prop_e1330_breite" in prompt.lower()

    def test_search_columns_grabtiefe(self):
        """Test searching for Grabtiefe column."""
        catalog = ColumnCatalog()
        catalog.initialize()
        results = catalog.search_columns("grabtiefe")
        assert len(results) >= 1
        assert any("grabtiefe" in r.display_name.lower() for r in results)

    def test_search_columns_breite(self):
        """Test searching for Breite column."""
        catalog = ColumnCatalog()
        catalog.initialize()
        results = catalog.search_columns("breite")
        assert len(results) >= 1

    def test_search_columns_klimaanlage(self):
        """Test searching for Klimaanlage column."""
        catalog = ColumnCatalog()
        catalog.initialize()
        results = catalog.search_columns("klimaanlage")
        assert len(results) >= 1
        assert any(r.category == "boolean" for r in results)

    def test_column_categorization(self):
        """Test columns are categorized correctly."""
        catalog = ColumnCatalog()
        catalog.initialize()

        # Dimension column should have unit
        grabtiefe = catalog.search_columns("grabtiefe")
        if grabtiefe:
            assert grabtiefe[0].category == "dimension"
            assert grabtiefe[0].unit == "mm"

        # Boolean column
        klima = catalog.search_columns("klimaanlage")
        if klima:
            assert klima[0].category == "boolean"

    def test_global_singleton(self):
        """Test global singleton works."""
        column_catalog.initialize()
        assert column_catalog.get_prompt_section() != ""
```

**Step 2: Run tests**

Run: `pytest tests/test_column_catalog.py -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add tests/test_column_catalog.py
git commit -m "test: add tests for ColumnCatalog"
```

---

### Task 8: Integration Test

**Files:**
- None (manual testing)

**Step 1: Test end-to-end with sample queries**

Run interactive test:
```python
import asyncio
from rag.single_agent import create_single_agent
from rag.vector_store import PineconeStore

async def test():
    pinecone = PineconeStore()
    agent = create_single_agent(verbose=True, pinecone_service=pinecone)

    # Test queries that use different phrasings
    queries = [
        "Bagger mit 3m Grabtiefe",           # Should find prop_e1740_grabtiefe_mm
        "Maschinen unter 2m breit",          # Should find prop_e1330_breite_mm
        "schwere Maschinen über 20 Tonnen",  # Should find prop_e1730_gewicht_kg
        "Fertiger mit Klimaanlage",          # Should find prop_e1930_klimaanlage
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        result = await agent.process(q, [], thread_key="test")
        print(f"Tools: {result.tools_used}")
        print(f"SQL rows: {result.sql_results_count}")
        print(f"Response: {result.response[:200]}...")

asyncio.run(test())
```

**Step 2: Verify LLM picks correct columns**

Check the verbose output shows SQL using correct column names like:
- `prop_e1740_grabtiefe_mm` for "Grabtiefe"
- `prop_e1330_breite_mm` for "Breite"
- `prop_e1730_gewicht_kg` for "Gewicht"
- `prop_e1930_klimaanlage` for "Klimaanlage"

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete semantic column resolution implementation"
```

---

## Summary

After completing all tasks:

1. **ColumnCatalog** loads 170+ property columns at startup
2. **System prompt** includes full column catalog for LLM
3. **LLM** semantically matches user terms to correct columns
4. **No hardcoded patterns** - works with any phrasing
5. **Safety checks** remain in sql_guard and sql_verifier

The user can now say:
- "Tiefe von 3 Metern" → LLM finds Grabtiefe column
- "wie breit" → LLM finds Breite column
- "Gewicht" → LLM finds Gewicht column
- Any synonym or phrasing → LLM understands semantically
