"""
Column Catalog for Semantic Column Resolution.

Loads all database columns at startup and provides them to the LLM
in a format that enables semantic understanding of user queries.

Enhanced with column statistics to help the AI discover:
- Which columns are empty (always NULL) vs populated
- Distinct values for categorical columns
- Data quality indicators
"""
import csv
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set

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
    # Statistics (populated from database)
    null_ratio: float = 1.0   # Ratio of NULL values (1.0 = all NULL, 0.0 = no NULLs)
    distinct_values: Optional[List[str]] = None  # For categorical columns, top distinct values
    is_empty: bool = True     # True if column is 100% NULL (useless for queries)
    sample_values: Optional[List[str]] = None  # Sample non-NULL values


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

    # Categorical columns for which we should fetch distinct values
    CATEGORICAL_COLUMNS = [
        "geraetegruppe_name",
        "hersteller_name",
        "verwendung_code",
        "nuclos_state",
    ]

    # Columns that need statistics to determine if they're empty
    # These are commonly expected but often have no data
    MOBILITY_COLUMNS = [
        "prop_e2100_mobil_kette",
        "prop_e2110_mobil_rad",
        "prop_e2120_mobil_semi",  # Note: e2120, not e2115
    ]

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
        # Statistics from database
        self._categorical_values: Dict[str, List[str]] = {}  # column -> distinct values
        self._empty_columns: Set[str] = set()  # Columns that are 100% NULL
        self._column_stats_loaded: bool = False

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
            # Load column statistics (NULL ratios, distinct values for categoricals)
            self._load_column_statistics(postgres_service)

        # Build the cached prompt section
        self._cached_prompt = self._build_prompt_section()
        self._initialized = True

        stats_info = f", {len(self._empty_columns)} empty columns detected" if self._column_stats_loaded else ""
        logger.info(f"[ColumnCatalog] Initialized with {len(self._columns)} property columns{stats_info}")

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

    def _load_column_statistics(self, postgres_service: Any) -> None:
        """
        Load column statistics from database:
        1. Distinct values for categorical columns (geraetegruppe_name, etc.)
        2. NULL ratios for mobility columns to detect empty columns

        This enables the LLM to know which columns have data and what values exist.
        """
        table = getattr(postgres_service, "equipment_table", "equipment_matrix")

        # 1. Load distinct values for categorical columns
        for col in self.CATEGORICAL_COLUMNS:
            try:
                result = postgres_service.execute_query(
                    f"SELECT DISTINCT {col} FROM {table} "
                    f"WHERE {col} IS NOT NULL "
                    f"ORDER BY {col} LIMIT 50"
                )
                if result:
                    values = [row.get(col) for row in result if row.get(col)]
                    self._categorical_values[col] = values
                    logger.debug(f"[ColumnCatalog] {col}: {len(values)} distinct values")
            except Exception as e:
                logger.debug(f"[ColumnCatalog] Failed to load {col} values: {e}")

        # 2. Check mobility columns for NULL ratio (these are often completely empty or nearly empty)
        # We consider a column "effectively empty" if >99% of values are NULL
        EMPTY_THRESHOLD = 0.99  # 99% NULL = effectively unusable

        for col in self.MOBILITY_COLUMNS:
            try:
                result = postgres_service.execute_query(
                    f"SELECT COUNT(*) as total, "
                    f"COUNT({col}) as non_null "
                    f"FROM {table}"
                )
                if result:
                    total = result[0].get("total", 0)
                    non_null = result[0].get("non_null", 0)
                    null_ratio = 1.0 - (non_null / total) if total > 0 else 1.0
                    # Mark as empty if NULL ratio exceeds threshold
                    if null_ratio >= EMPTY_THRESHOLD:
                        self._empty_columns.add(col)
                        logger.info(f"[ColumnCatalog] {col} is effectively EMPTY ({null_ratio*100:.1f}% NULL)")
                        # Update the ColumnInfo if it exists
                        if col in self._columns:
                            self._columns[col].is_empty = True
                            self._columns[col].null_ratio = null_ratio
            except Exception as e:
                logger.debug(f"[ColumnCatalog] Failed to check {col} NULL ratio: {e}")

        # 3. Check a sample of prop columns for data quality
        try:
            # Get columns that exist in the database
            result = postgres_service.execute_query(
                "SELECT column_name FROM information_schema.columns "
                f"WHERE table_name = '{table}' AND column_name LIKE 'prop_%'"
            )
            if result:
                db_columns = [row.get("column_name") for row in result]
                # Check a sample of important prop columns
                sample_columns = [c for c in db_columns if c in self._columns][:20]
                for col in sample_columns:
                    try:
                        stats_result = postgres_service.execute_query(
                            f"SELECT COUNT(*) as total, COUNT({col}) as non_null "
                            f"FROM {table}"
                        )
                        if stats_result:
                            total = stats_result[0].get("total", 0)
                            non_null = stats_result[0].get("non_null", 0)
                            if total > 0:
                                null_ratio = 1.0 - (non_null / total)
                                if col in self._columns:
                                    self._columns[col].null_ratio = null_ratio
                                    self._columns[col].is_empty = (non_null == 0)
                                    if non_null == 0:
                                        self._empty_columns.add(col)
                    except Exception:
                        pass  # Silently skip columns that fail
        except Exception as e:
            logger.debug(f"[ColumnCatalog] Failed to load prop column stats: {e}")

        self._column_stats_loaded = True
        logger.info(f"[ColumnCatalog] Loaded statistics: {len(self._categorical_values)} categorical columns, "
                   f"{len(self._empty_columns)} empty columns")

    def _build_prompt_section(self) -> str:
        """Build the prompt section for the LLM."""
        lines = []
        lines.append("=" * 70)
        lines.append("PROPERTY COLUMNS CATALOG")
        lines.append("Use this catalog to find the correct column for user queries.")
        lines.append("The LLM should semantically match user terms to these columns.")
        lines.append("=" * 70)
        lines.append("")

        # CRITICAL: Show empty columns first so the LLM knows to avoid them
        if self._empty_columns:
            lines.append("⚠️  EMPTY COLUMNS (100% NULL - DO NOT USE!):")
            lines.append("-" * 50)
            for col in sorted(self._empty_columns):
                info = self._columns.get(col)
                if info:
                    lines.append(f"  {col} -> {info.display_name} [ALWAYS NULL - USELESS]")
                else:
                    lines.append(f"  {col} [ALWAYS NULL - USELESS]")
            lines.append("")
            lines.append("These columns exist but contain NO DATA. Never query them!")
            lines.append("Use geraetegruppe_name instead for Kette/Mobil/Rad distinction.")
            lines.append("")

        # Show categorical column values
        if self._categorical_values:
            lines.append("📊 CATEGORICAL COLUMN VALUES (use these for filtering):")
            lines.append("-" * 50)
            for col, values in self._categorical_values.items():
                if values:
                    # Show first 15 values
                    display_values = values[:15]
                    lines.append(f"  {col}:")
                    for v in display_values:
                        lines.append(f"    - '{v}'")
                    if len(values) > 15:
                        lines.append(f"    ... and {len(values) - 15} more")
            lines.append("")

        # Group by category
        categories: Dict[str, List[ColumnInfo]] = {}
        for col_name, info in sorted(self._columns.items()):
            # Skip empty columns in the main listing
            if col_name in self._empty_columns:
                continue
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
                # Show data quality indicator
                if info.null_ratio >= 0.9 and info.null_ratio < 1.0:
                    entry += " [sparse data]"
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

    def get_prompt_section(self, compact: bool = False) -> str:
        """Get the cached prompt section for LLM system prompt.

        Args:
            compact: If True, return a compact version with only key columns and patterns.
                    Reduces prompt size significantly for faster responses.
        """
        if not self._initialized:
            raise RuntimeError("ColumnCatalog not initialized. Call initialize() first.")

        if compact:
            return self._build_compact_prompt()
        return self._cached_prompt or ""

    def _build_compact_prompt(self) -> str:
        """Build a compact version of the prompt (much smaller)."""
        lines = []
        lines.append("PROPERTY COLUMNS (compact reference):")
        lines.append("Column format: prop_<code>_<name>_<unit>")
        lines.append("")

        # CRITICAL: Show empty columns first
        if self._empty_columns:
            lines.append("⚠️  EMPTY COLUMNS (NEVER USE - 100% NULL):")
            for col in sorted(self._empty_columns):
                lines.append(f"  {col} [ALWAYS NULL]")
            lines.append("Use geraetegruppe_name for Kette/Mobil/Rad!")
            lines.append("")

        # Show key categorical values
        if "geraetegruppe_name" in self._categorical_values:
            values = self._categorical_values["geraetegruppe_name"]
            lines.append("GERAETEGRUPPE_NAME VALUES (use for equipment type):")
            for v in values[:20]:
                lines.append(f"  - '{v}'")
            if len(values) > 20:
                lines.append(f"  ... and {len(values) - 20} more")
            lines.append("")

        # Group key columns by function (excluding empty ones)
        key_dimensions = [
            ("prop_e1150_arbeitsbreite_mm", "Arbeitsbreite [mm]"),
            ("prop_e1480_einbaubreite_max_m", "Einbaubreite max [m]"),
            ("prop_e1470_einbaubreite_grundbohle_m", "Einbaubreite Grundbohle [m]"),
            ("prop_e1740_grabtiefe_mm", "Grabtiefe [mm]"),
            ("prop_e2370_reichweite_m", "Reichweite [m]"),
            ("prop_e1730_gewicht_kg", "Gewicht [kg]"),
            ("prop_e2180_motor_leistung_kw", "Motor Leistung [kW]"),
            ("prop_e2490_tragfahigkeit_kg", "Tragfähigkeit [kg]"),
            ("prop_e1900_hubhohe_m", "Hubhöhe [m]"),
        ]

        key_booleans = [
            ("prop_e1110_allradantrieb", "Allradantrieb"),
            ("prop_e2040_klimaanlage", "Klimaanlage"),
        ]
        # Note: prop_e2100_mobil_kette and prop_e2110_mobil_rad are excluded (always NULL)

        lines.append("KEY DIMENSIONS (extract numeric with regexp_replace):")
        for col, desc in key_dimensions:
            if col not in self._empty_columns:
                lines.append(f"  {col} -> {desc}")

        lines.append("")
        lines.append("KEY BOOLEAN FEATURES (check 'Ja' or IS NOT NULL):")
        for col, desc in key_booleans:
            if col not in self._empty_columns:
                lines.append(f"  {col} -> {desc}")

        lines.append("")
        lines.append("SQL PATTERNS:")
        lines.append("  Numeric from TEXT: CAST(NULLIF(regexp_replace(col, '[^0-9]', '', 'g'), '') AS NUMERIC)")
        lines.append("  Boolean check: WHERE col = 'Ja' OR col IS NOT NULL")
        lines.append("")
        lines.append(f"Total property columns available: {len(self._columns)}")
        lines.append("For complete list, query: SELECT column_name FROM information_schema.columns WHERE column_name LIKE 'prop_%'")

        return "\n".join(lines)

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

    def get_categorical_values(self, column: str) -> List[str]:
        """Get distinct values for a categorical column."""
        return self._categorical_values.get(column, [])

    def is_column_empty(self, column: str) -> bool:
        """Check if a column is known to be empty (100% NULL)."""
        return column in self._empty_columns

    def get_empty_columns(self) -> Set[str]:
        """Get all columns that are known to be empty."""
        return self._empty_columns.copy()

    def get_column_recommendation(self, user_term: str) -> Optional[str]:
        """
        Get a column recommendation based on user's term.

        This helps when the user asks about something that maps to a column
        differently than expected (e.g., "Kette" -> geraetegruppe_name, not prop_e2100_mobil_kette).

        Returns recommendation string or None.
        """
        term_lower = user_term.lower()

        # Special case: Kette/Mobil/Rad → use geraetegruppe_name
        if any(t in term_lower for t in ["kette", "mobil", "rad", "semi"]):
            if "geraetegruppe_name" in self._categorical_values:
                values = self._categorical_values["geraetegruppe_name"]
                relevant = [v for v in values if any(
                    t in v.lower() for t in ["kette", "mobil", "rad"]
                )]
                if relevant:
                    return (
                        f"For '{user_term}' queries, use geraetegruppe_name column. "
                        f"Available values: {', '.join(relevant[:5])}"
                    )

        return None


# Global singleton instance
column_catalog = ColumnCatalog()
