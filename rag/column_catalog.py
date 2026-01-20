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

import logging

logger = logging.getLogger(__name__)


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

        logger.info(f"[ColumnCatalog] Initialized with {len(self._columns)} property columns")

    def _load_from_csv(self, csv_path: Path) -> None:
        """Load property metadata from CSV file."""
        if not csv_path.exists():
            logger.warning(f"[ColumnCatalog] CSV not found: {csv_path}")
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
            logger.error(f"[ColumnCatalog] CSV load error: {e}")

    def _add_property(self, code: str, name: str) -> None:
        """Add a property column to the catalog."""
        # Parse unit from name like "Grabtiefe [mm]" or "Gewicht [kg]"
        unit_match = re.search(r'\[([^\]]+)\]', name)
        unit = unit_match.group(1) if unit_match else None

        # Clean display name (remove unit suffix)
        display_name = re.sub(r'\s*\[[^\]]+\]', '', name).strip()

        # Generate column name: prop_e1740_grabtiefe_mm (include unit in column name)
        slug = self._slugify(name)
        if unit:
            unit_slug = unit.lower().replace("/", "_").replace("³", "3")
            column_name = f"prop_{code.lower()}_{slug}_{unit_slug}"
        else:
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
            "absauganlage", "asphaltmanager", "backenbrecher",
            "bio-hydrauliköl", "dachprofil", "dieselmotor",
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
        type_indicators = ["typ", "art", "hersteller", "klasse", "größe", "grosse", "achser", "stufe"]
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
                logger.info("[ColumnCatalog] Database validation: columns exist")
        except Exception as e:
            logger.debug(f"[ColumnCatalog] Database validation skipped: {e}")

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
        categories: Dict[str, List[ColumnInfo]] = {}
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
