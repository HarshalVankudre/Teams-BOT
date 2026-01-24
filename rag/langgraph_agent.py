"""LangGraph ReAct agent for RÜKO equipment queries - V2 with improved numeric handling."""

import re
import time
import logging
import asyncio
from typing import List, Optional, Union
from dataclasses import dataclass, field

from langchain_core.tools import tool

from rag.config import config

# Verbose logging flag
VERBOSE = config.agent_verbose if hasattr(config, "agent_verbose") else False

from rag.postgres import PostgresService
from rag.vector_store import PineconeStore

logger = logging.getLogger(__name__)


# =============================================================================
# Timing Utilities
# =============================================================================

@dataclass
class TimingRecord:
    """Record of a timed operation."""
    operation: str
    duration_ms: float
    details: Optional[str] = None


class TimingCollector:
    """Collects timing information for request analysis."""

    def __init__(self):
        self.records: List[TimingRecord] = []
        self.start_time: float = time.time()

    def record(self, operation: str, duration_ms: float, details: str = None):
        """Record a timing measurement."""
        self.records.append(TimingRecord(operation, duration_ms, details))
        # Always log timing info
        detail_str = f" ({details})" if details else ""
        print(f"⏱️  [{operation}] {duration_ms:.0f}ms{detail_str}")

    def get_summary(self) -> dict:
        """Get timing summary."""
        total_ms = (time.time() - self.start_time) * 1000
        by_category = {}
        for r in self.records:
            cat = r.operation.split(":")[0] if ":" in r.operation else r.operation
            by_category[cat] = by_category.get(cat, 0) + r.duration_ms

        return {
            "total_ms": round(total_ms, 1),
            "breakdown": {r.operation: round(r.duration_ms, 1) for r in self.records},
            "by_category": {k: round(v, 1) for k, v in by_category.items()}
        }

    def print_summary(self):
        """Print a formatted timing summary."""
        summary = self.get_summary()
        print("\n" + "="*60)
        print("⏱️  TIMING SUMMARY")
        print("="*60)
        print(f"Total: {summary['total_ms']:.0f}ms")
        print("-"*40)
        for op, ms in summary['breakdown'].items():
            pct = (ms / summary['total_ms'] * 100) if summary['total_ms'] > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"{op:30} {ms:6.0f}ms {bar} {pct:5.1f}%")
        print("="*60 + "\n")


# Thread-local timing collector
_current_timing: Optional[TimingCollector] = None


def get_timing_collector() -> Optional[TimingCollector]:
    """Get current timing collector."""
    global _current_timing
    return _current_timing


def set_timing_collector(collector: Optional[TimingCollector]):
    """Set current timing collector."""
    global _current_timing
    _current_timing = collector

# Table name - using optimized materialized view
EQUIPMENT_TABLE = "public.equipment_matrix_v2"

# Initialize services (singleton pattern)
_postgres: Optional[PostgresService] = None
_pinecone: Optional[PineconeStore] = None


def set_shared_postgres(postgres: PostgresService):
    """Set a shared PostgresService instance."""
    global _postgres
    _postgres = postgres


def _get_postgres() -> PostgresService:
    global _postgres
    if _postgres is None:
        _postgres = PostgresService()
    return _postgres


def set_shared_pinecone(pinecone: PineconeStore):
    """Set a shared PineconeStore instance."""
    global _pinecone
    _pinecone = pinecone


def _get_pinecone() -> PineconeStore:
    global _pinecone
    if _pinecone is None:
        _pinecone = PineconeStore()
    return _pinecone


# =============================================================================
# Tools
# =============================================================================

@tool
def execute_sql(sql: str, purpose: str) -> dict:
    """Execute a read-only SQL query against the equipment database.

    Use this tool to query public.equipment_matrix_v2 for equipment data.
    Only SELECT queries are allowed. Results limited to 50 rows.

    IMPORTANT: For text comparisons, ALWAYS use ILIKE with wildcards:
    ✅ WHERE hersteller_name ILIKE '%bomag%'
    ❌ WHERE hersteller_name = 'BOMAG'

    Args:
        sql: The SELECT query to execute.
        purpose: Brief description of what this query is for.

    Returns:
        Dict with row_count, results (max 50 rows), and result_ids for follow-ups.
    """
    tool_start = time.time()
    timing = get_timing_collector()

    postgres = _get_postgres()

    # DEBUG: Print the SQL being executed
    print(f"\n{'='*60}")
    print(f"🔍 EXECUTE_SQL CALLED")
    print(f"📝 Purpose: {purpose}")
    print(f"📄 SQL Query:\n{sql}")
    print(f"{'='*60}\n")
    logger.info(f"SQL Query: {sql}")

    # Validate and prepare SQL
    prepare_start = time.time()
    prepared, error = postgres.prepare_readonly_sql(sql)
    prepare_ms = (time.time() - prepare_start) * 1000

    if error:
        print(f"❌ SQL Preparation Error: {error}")
        logger.error(f"SQL preparation error: {error}")
        if timing:
            timing.record("tool:execute_sql:error", (time.time() - tool_start) * 1000, error)
        return {"error": error, "sql": sql}

    try:
        query_start = time.time()
        results = postgres.execute_query(prepared)
        query_ms = (time.time() - query_start) * 1000

        result_ids = [r.get("id") for r in results if r.get("id")]

        # DEBUG: Print results summary
        print(f"✅ Query returned {len(results)} rows")
        if results:
            print(f"📊 First result: {results[0]}")
        logger.info(f"SQL returned {len(results)} rows")

        tool_ms = (time.time() - tool_start) * 1000
        if timing:
            timing.record("tool:execute_sql", tool_ms, f"{len(results)} rows, db={query_ms:.0f}ms")

        return {
            "purpose": purpose,
            "sql": prepared,
            "row_count": len(results),
            "results": results[:50],
            "result_ids": result_ids[:100],
            "_timing_ms": {"total": tool_ms, "db_query": query_ms, "prepare": prepare_ms}
        }
    except Exception as e:
        print(f"❌ SQL Execution Error: {e}")
        logger.error(f"SQL execution error: {e}")
        if timing:
            timing.record("tool:execute_sql:error", (time.time() - tool_start) * 1000, str(e))
        return {"error": str(e), "sql": prepared}


@tool
def query_equipment(
    category: str = None,
    manufacturer: str = None,
    property_column: str = None,
    property_operator: str = None,
    property_value: Union[str, int, float, None] = None,
    property_column_2: str = None,
    property_operator_2: str = None,
    property_value_2: Union[str, int, float, None] = None,
    usage_type: str = None,
    limit: int = 10
) -> dict:
    """Query equipment with filters.

    Args:
        category: Equipment category (e.g., 'Fertiger', 'Bagger', 'Fräse', 'Walze')
        manufacturer: Manufacturer filter (e.g., 'Bomag', 'Liebherr', 'Voegele')
        property_column: Column for filter. Use *_num columns for numeric, prop_* for text/boolean
        property_operator: Comparison: '>=', '<=', '>', '<', '=' (use '=' for text)
        property_value: Value - number for *_num columns, text for prop_* (e.g., "Ja", "Nein")
        property_column_2: Second filter column (optional)
        property_operator_2: Second operator (optional)
        property_value_2: Second value (optional)
        usage_type: 'MIET' (rental) or 'VK' (sale)
        limit: Max results (default 10, max 20)

    Returns:
        Dict with matching equipment.
    """
    tool_start = time.time()
    timing = get_timing_collector()

    postgres = _get_postgres()

    print(f"\n{'='*60}")
    print(f"🔧 QUERY_EQUIPMENT CALLED")
    print(f"  category: {category}")
    print(f"  manufacturer: {manufacturer}")
    print(f"  property_column: {property_column}")
    print(f"  property_operator: {property_operator}")
    print(f"  property_value: {property_value}")
    if property_column_2:
        print(f"  property_column_2: {property_column_2}")
        print(f"  property_operator_2: {property_operator_2}")
        print(f"  property_value_2: {property_value_2}")
    print(f"  usage_type: {usage_type}")
    print(f"{'='*60}\n")

    # Build WHERE clauses
    where_clauses = []

    if category:
        where_clauses.append(f"geraetegruppe_name ILIKE '%{category}%'")

    if manufacturer:
        where_clauses.append(f"hersteller_name ILIKE '%{manufacturer}%'")

    if usage_type:
        where_clauses.append(f"verwendung_code = '{usage_type}'")

    # Helper to add property filter
    def add_filter(col, op, val):
        if not col or not op or val is None:
            return
        if op not in ('>=', '<=', '>', '<', '='):
            return

        # Check if numeric column (*_num) or text column (prop_*)
        if col.endswith('_num'):
            # Numeric - use value directly
            where_clauses.append(f"{col} {op} {val}")
            print(f"⚡ Numeric filter: {col} {op} {val}")
        else:
            # Text/Boolean - quote the value
            where_clauses.append(f"\"{col}\" = '{val}'")
            print(f"⚡ Text filter: {col} = '{val}'")

    add_filter(property_column, property_operator, property_value)
    add_filter(property_column_2, property_operator_2, property_value_2)

    # Build query
    limit = min(limit, 20)  # Cap at 20 to save tokens
    where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"

    sql = f"""
        SELECT id, bezeichnung, hersteller_name, geraetegruppe_name, verwendung_code
        FROM {EQUIPMENT_TABLE}
        WHERE {where_sql}
        LIMIT {limit}
    """

    print(f"📄 SQL:\n{sql}")

    try:
        query_start = time.time()
        results = postgres.execute_query(sql)
        query_ms = (time.time() - query_start) * 1000

        print(f"✅ {len(results)} rows")

        tool_ms = (time.time() - tool_start) * 1000
        if timing:
            timing.record("tool:query_equipment", tool_ms, f"{len(results)} rows, db={query_ms:.0f}ms")

        return {
            "row_count": len(results),
            "results": results,
            "sql": sql.strip(),
            "_timing_ms": {"total": tool_ms, "db_query": query_ms}
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        if timing:
            timing.record("tool:query_equipment:error", (time.time() - tool_start) * 1000, str(e))
        return {"error": str(e), "sql": sql}


@tool
def lookup_equipment(search_term: str, include_fields: str = "all") -> dict:
    """Look up specific equipment by name, model, or serial number.

    Use this when the user asks about a SPECIFIC machine by name.

    Args:
        search_term: Machine name, model, or serial number to search for
                     (e.g., "Super 800i", "R 926", "Liebherr A 914")
        include_fields: Which fields to return:
                       - "basic": id, bezeichnung, hersteller_name, seriennummer
                       - "all": all core fields + all properties via JSON

    Returns:
        Dict with matching equipment details including serial numbers and ALL properties.

    Example:
        lookup_equipment("Super 800i") → Returns all Super 800i machines with all properties
    """
    tool_start = time.time()
    timing = get_timing_collector()

    postgres = _get_postgres()

    print(f"\n{'='*60}")
    print(f"🔍 LOOKUP_EQUIPMENT: '{search_term}'")
    print(f"{'='*60}\n")

    # Determine which fields to select
    if include_fields == "basic":
        fields = "id, bezeichnung, hersteller_name, seriennummer, geraetegruppe_name"
    else:
        # Include core fields + properties_jsonb for ALL dynamic properties
        fields = """id, bezeichnung, hersteller_name, seriennummer, inventarnummer,
                    geraetegruppe_name, verwendung_code, nuclos_state,
                    properties_jsonb"""

    # Build search query - search in bezeichnung, seriennummer, and inventarnummer
    sql = f"""
        SELECT {fields}
        FROM {EQUIPMENT_TABLE}
        WHERE bezeichnung ILIKE '%{search_term}%'
           OR seriennummer ILIKE '%{search_term}%'
           OR inventarnummer ILIKE '%{search_term}%'
        LIMIT 5
    """

    print(f"📄 SQL:\n{sql}")

    try:
        query_start = time.time()
        results = postgres.execute_query(sql)
        query_ms = (time.time() - query_start) * 1000

        print(f"✅ Found {len(results)} machines matching '{search_term}'")
        if results:
            print(f"📊 First result keys: {list(results[0].keys())}")

        tool_ms = (time.time() - tool_start) * 1000
        if timing:
            timing.record("tool:lookup_equipment", tool_ms, f"{len(results)} found, db={query_ms:.0f}ms")

        return {
            "search_term": search_term,
            "found": len(results),
            "machines": results,
            "_timing_ms": {"total": tool_ms, "db_query": query_ms}
        }
    except Exception as e:
        print(f"❌ Lookup error: {e}")
        logger.error(f"Lookup equipment error: {e}")
        if timing:
            timing.record("tool:lookup_equipment:error", (time.time() - tool_start) * 1000, str(e))
        return {"error": str(e), "search_term": search_term}


@tool
def get_equipment_details(
    equipment_id: int = None,
    serial_number: str = None,
    property_filter: str = None
) -> dict:
    """Get detailed properties of a specific machine by ID or serial number.

    Use this after lookup_equipment to get specific property details like:
    - Bohlentyp, Verbreiterungen, Nivelliersystem (for Fertiger)
    - Grabtiefe, Löffelinhalt, Reichweite (for Bagger)
    - Any other technical specifications

    Args:
        equipment_id: The machine ID from previous lookup
        serial_number: Or the serial number to look up
        property_filter: Optional keyword to filter properties (e.g., 'bohle', 'verbreiterung', 'grab')

    Returns:
        Dict with all non-null properties of the machine.
    """
    tool_start = time.time()
    timing = get_timing_collector()

    postgres = _get_postgres()

    if not equipment_id and not serial_number:
        if timing:
            timing.record("tool:get_equipment_details:error", 0, "no id or serial")
        return {"error": "Provide either equipment_id or serial_number"}

    print(f"\n{'='*60}")
    print(f"🔎 GET_EQUIPMENT_DETAILS: id={equipment_id}, serial={serial_number}, filter={property_filter}")
    print(f"{'='*60}\n")

    # Build query
    if equipment_id:
        where = f"id = {equipment_id}"
    else:
        where = f"seriennummer = '{serial_number}'"

    sql = f"""
        SELECT id, bezeichnung, hersteller_name, seriennummer, geraetegruppe_name,
               verwendung_code, properties_jsonb
        FROM {EQUIPMENT_TABLE}
        WHERE {where}
        LIMIT 1
    """

    try:
        query_start = time.time()
        results = postgres.execute_query(sql)
        query_ms = (time.time() - query_start) * 1000

        if not results:
            if timing:
                timing.record("tool:get_equipment_details", (time.time() - tool_start) * 1000, "not found")
            return {"error": f"No equipment found", "equipment_id": equipment_id, "serial_number": serial_number}

        machine = results[0]
        properties = machine.get("properties_jsonb", {}) or {}

        # Filter out None/null values
        properties = {k: v for k, v in properties.items() if v is not None and v != ""}

        # Apply keyword filter if provided
        if property_filter:
            filter_lower = property_filter.lower()
            properties = {k: v for k, v in properties.items() if filter_lower in k.lower() or filter_lower in str(v).lower()}

        print(f"✅ Found {len(properties)} properties for {machine.get('bezeichnung')}")

        tool_ms = (time.time() - tool_start) * 1000
        if timing:
            timing.record("tool:get_equipment_details", tool_ms, f"{len(properties)} props, db={query_ms:.0f}ms")

        return {
            "id": machine.get("id"),
            "bezeichnung": machine.get("bezeichnung"),
            "hersteller": machine.get("hersteller_name"),
            "seriennummer": machine.get("seriennummer"),
            "geraetegruppe": machine.get("geraetegruppe_name"),
            "verwendung": machine.get("verwendung_code"),
            "property_count": len(properties),
            "properties": properties,
            "_timing_ms": {"total": tool_ms, "db_query": query_ms}
        }
    except Exception as e:
        print(f"❌ Details error: {e}")
        logger.error(f"Get equipment details error: {e}")
        if timing:
            timing.record("tool:get_equipment_details:error", (time.time() - tool_start) * 1000, str(e))
        return {"error": str(e)}


@tool
async def search_documents(query: str, top_k: int = 10) -> dict:
    """Search equipment manuals, documentation, and technical specifications.

    Use this for questions about operating instructions, maintenance,
    technical details not in the database, or general equipment information.

    Args:
        query: Search query in German (e.g., "Kettenfertiger Wartung")
        top_k: Number of results to return (default 10, max 20)

    Returns:
        Dict with matches containing title, content snippet, and source.
    """
    tool_start = time.time()
    timing = get_timing_collector()

    pinecone = _get_pinecone()
    top_k = min(top_k, 20)

    print(f"\n🔎 SEARCH_DOCUMENTS: '{query}' (top_k={top_k})")

    try:
        search_start = time.time()
        results = await pinecone.search(query, top_k=top_k)
        search_ms = (time.time() - search_start) * 1000

        matches = []
        for r in results:
            matches.append({
                "title": r.get("metadata", {}).get("title", "Untitled"),
                "content": r.get("metadata", {}).get("content", "")[:500],
                "source": r.get("metadata", {}).get("source_file", "unknown"),
                "score": r.get("score", 0)
            })

        print(f"✅ Found {len(matches)} document matches")

        tool_ms = (time.time() - tool_start) * 1000
        if timing:
            timing.record("tool:search_documents", tool_ms, f"{len(matches)} matches, search={search_ms:.0f}ms")

        return {
            "query": query,
            "match_count": len(matches),
            "matches": matches,
            "_timing_ms": {"total": tool_ms, "pinecone_search": search_ms}
        }
    except Exception as e:
        print(f"❌ Document search error: {e}")
        logger.error(f"Document search error: {e}")
        if timing:
            timing.record("tool:search_documents:error", (time.time() - tool_start) * 1000, str(e))
        return {"error": str(e), "query": query}


@tool
def find_columns(search_term: str) -> dict:
    """Find database columns by keyword search.

    Use this to discover which columns contain specific data before writing SQL queries.
    Searches column names for matching keywords.

    Args:
        search_term: Keyword to search for in column names (e.g., 'breite', 'tiefe', 'gewicht')

    Returns:
        Dict with matching column names and their data types.
    """
    tool_start = time.time()
    timing = get_timing_collector()

    postgres = _get_postgres()

    print(f"\n🔎 FIND_COLUMNS: '{search_term}'")

    try:
        query_start = time.time()
        # Query information schema for matching columns
        sql = f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'equipment_matrix_v2'
              AND column_name ILIKE '%{search_term}%'
            ORDER BY column_name
            LIMIT 30
        """
        results = postgres.execute_query(sql)
        query_ms = (time.time() - query_start) * 1000

        columns = []
        for r in results:
            col_name = r.get("column_name", "")
            # Identify numeric columns (those ending with _num)
            is_numeric = col_name.endswith("_num")
            columns.append({
                "name": col_name,
                "type": r.get("data_type", "unknown"),
                "is_numeric": is_numeric
            })

        print(f"✅ Found {len(columns)} matching columns")

        tool_ms = (time.time() - tool_start) * 1000
        if timing:
            timing.record("tool:find_columns", tool_ms, f"{len(columns)} columns, db={query_ms:.0f}ms")

        return {
            "search_term": search_term,
            "column_count": len(columns),
            "columns": columns,
            "_timing_ms": {"total": tool_ms, "db_query": query_ms}
        }
    except Exception as e:
        print(f"❌ Find columns error: {e}")
        logger.error(f"Find columns error: {e}")
        if timing:
            timing.record("tool:find_columns:error", (time.time() - tool_start) * 1000, str(e))
        return {"error": str(e), "search_term": search_term}


@tool
def explore_column(column_name: str) -> dict:
    """Show distinct values in a database column.

    Use this to understand what values exist in a column before writing queries.
    Helpful for categories, manufacturers, or status fields.

    Args:
        column_name: The column to explore (e.g., 'hersteller_name', 'geraetegruppe_name')

    Returns:
        Dict with distinct values (max 50).
    """
    tool_start = time.time()
    timing = get_timing_collector()

    postgres = _get_postgres()

    print(f"\n🔎 EXPLORE_COLUMN: '{column_name}'")

    # Validate column name format (PostgreSQL identifier)
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', column_name):
        print(f"❌ Invalid column name: {column_name}")
        if timing:
            timing.record("tool:explore_column:error", (time.time() - tool_start) * 1000, "invalid column")
        return {"error": f"Invalid column name: {column_name}"}

    sql = f"""
        SELECT DISTINCT "{column_name}"
        FROM {EQUIPMENT_TABLE}
        WHERE "{column_name}" IS NOT NULL
        ORDER BY "{column_name}"
        LIMIT 50
    """

    try:
        query_start = time.time()
        results = postgres.execute_query(sql)
        query_ms = (time.time() - query_start) * 1000

        values = [r.get(column_name) for r in results if r.get(column_name)]

        print(f"✅ Found {len(values)} distinct values")

        tool_ms = (time.time() - tool_start) * 1000
        if timing:
            timing.record("tool:explore_column", tool_ms, f"{len(values)} values, db={query_ms:.0f}ms")

        return {
            "column": column_name,
            "distinct_count": len(values),
            "values": values,
            "_timing_ms": {"total": tool_ms, "db_query": query_ms}
        }
    except Exception as e:
        print(f"❌ Column exploration error: {e}")
        logger.error(f"Column exploration error: {e}")
        if timing:
            timing.record("tool:explore_column:error", (time.time() - tool_start) * 1000, str(e))
        return {"error": str(e), "column": column_name}


# =============================================================================
# System Prompt - Simplified with tool-based approach
# =============================================================================

SYSTEM_PROMPT = """Du bist der RÜKO Baumaschinen-Assistent.

DATENBANK: public.equipment_matrix_v2 (~2400 Maschinen)

═══════════════════════════════════════════════════════════════
KERN-SPALTEN:
═══════════════════════════════════════════════════════════════
id, bezeichnung, seriennummer, inventarnummer, hersteller_name,
geraetegruppe_name, verwendung_code ('MIET'/'VK'), nuclos_state,
nuclos_process, kostenstelle, equipment_key

═══════════════════════════════════════════════════════════════
NUMERISCHE FILTER-SPALTEN (*_num) - für WHERE-Bedingungen:
═══════════════════════════════════════════════════════════════
arbeitsbreite_mm_num, breite_mm_num, einbaubreite_grundbohle_m_num,
einbaubreite_max_m_num (FERTIGER!), einbaubreite_verbreiterungen_m_num,
fraesbreite_mm_num (FRÄSEN!), fraestiefe_mm_num, gewicht_kg_num,
grabtiefe_mm_num (BAGGER!), hoehe_mm_num, hubhoehe_mm_num, laenge_mm_num,
loeffelstiel_mm_num, motor_leistung_kw_num, nutzlast_kg_num,
schnittbreite_mm_num, schnitttiefe_mm_num, tragkraft_max_kg_num,
tragkraft_spitze_kg_num

═══════════════════════════════════════════════════════════════
ALLE EIGENSCHAFTS-SPALTEN (prop_*):
═══════════════════════════════════════════════════════════════
prop_e1010_1_achser, prop_e1020_2_achser, prop_e1030_3_achser,
prop_e1040_4_achser, prop_e1050_abb_arbeitsbereichsbegrenzung,
prop_e1070_abgasstufe_eu, prop_e1080_abgasstufe_usa, prop_e1090_absauganlage,
prop_e1100_absauganlage_vcs, prop_e1110_allradantrieb, prop_e1120_allradlenkung,
prop_e1130_anzahl_zaehne, prop_e1150_arbeitsbreite_mm, prop_e1160_arbeitsdruck_bar,
prop_e1170_arbeitshoehe_m, prop_e1180_asphaltmanager, prop_e1200_aufgabe_mm,
prop_e1210_ausladung_m, prop_e1220_ausleger_m, prop_e1230_backenbrecher,
prop_e1240_ballast_t, prop_e1250_bandbreite_mm, prop_e1270_bio_hydraulikoel,
prop_e1280_bodenplatten_mm, prop_e1320_brechkraft_t, prop_e1330_breite_mm,
prop_e1340_dachprofilverstellung, prop_e1360_dauerleistung_kva,
prop_e1370_dieselmotor, prop_e1380_dieselpartikelfilter,
prop_e1390_distanzkontrolle_automatisch, prop_e1400_drehbar, prop_e1430_druck_bar,
prop_e1450_durchsatzmenge_t_h, prop_e1460_e_heizung,
prop_e1470_einbaubreite_grundbohle_m, prop_e1480_einbaubreite_max_m,
prop_e1490_einbaubreite_mit_verbreiterungen_m, prop_e1510_elektrostarter,
prop_e1530_fahrgeschwindigkeit_km_h, prop_e1540_fcs_flexible_cutter_system,
prop_e1570_foerderhoehe_m, prop_e1580_foerderkapazitaet_t_h,
prop_e1590_foerderlaenge_m, prop_e1610_fraesbreite_mm,
prop_e1620_fraesmeissel_anzahl, prop_e1630_fraestiefe_mm, prop_e1650_frequenz_hz,
prop_e1660_frontschild_mm, prop_e1670_funkfernsteuerung,
prop_e1680_gabelaufnahme_beschickerkuebel, prop_e1690_gas_heizung,
prop_e1700_gegengewicht_t, prop_e1720_geteilte_bandage, prop_e1730_gewicht_kg,
prop_e1740_grabtiefe_mm, prop_e1750_greiferdreheinrichtung,
prop_e1760_greiferhydraulik, prop_e1770_hakenhoehe_m, prop_e1780_hammerhydraulik,
prop_e1830_hochdruckreiniger, prop_e1840_hochfahrbare_kabine, prop_e1860_hoehe_mm,
prop_e1870_hubhoehe_mm, prop_e1880_inhalt_m3, prop_e1890_kabine,
prop_e1900_kantenschneidgeraet_stueck, prop_e1920_klappschild,
prop_e1930_klimaanlage, prop_e1940_knicklenkung, prop_e1960_koernung_mm,
prop_e1970_kreiselbrecher, prop_e1990_laenge_mm, prop_e2000_laufzeit_h,
prop_e2010_leistung_kva, prop_e2020_leistungsaufnahme_kw, prop_e2030_level_pro,
prop_e2040_loeffelstiel_mm, prop_e2080_mittelschar_mm, prop_e2100_mobil_kette,
prop_e2110_mobil_rad, prop_e2120_mobil_semi, prop_e2130_monoausleger,
prop_e2140_motor_benzin, prop_e2150_motor_diesel, prop_e2160_motor_elektro,
prop_e2170_motor_hersteller, prop_e2180_motor_leistung_kw, prop_e2190_motor_typ,
prop_e2200_muldenerhoehung, prop_e2210_muldenheizung, prop_e2220_muldenvolumen_m3,
prop_e2230_nennspannung_v, prop_e2240_nennstrom_a, prop_e2250_nutzlast_kg,
prop_e2260_oszillation, prop_e2270_pat_schild_mm, prop_e2280_plattformhoehe_mm,
prop_e2300_powertilt, prop_e2310_prallmuehle, prop_e2320_pratzenabstuetzung,
prop_e2340_rampen_hydraulisch, prop_e2350_rampen_mechanisch,
prop_e2370_reversierbar, prop_e2390_schaufelvolumen_m3, prop_e2400_scherenhydraulik,
prop_e2420_schildabstuetzung, prop_e2430_schnellgang, prop_e2440_schnellwechsler_typ,
prop_e2450_schnellwechsler_henle, prop_e2460_schnellwechsler_hydr,
prop_e2470_schnellwechsler_mech, prop_e2480_schnellwechsler_oilquick,
prop_e2490_schnittbreite_mm, prop_e2510_schnittlaenge_mm, prop_e2520_schnitttiefe_mm,
prop_e2530_schuetthoehe_mm, prop_e2540_schutzklasse_ip, prop_e2550_schwenkband,
prop_e2560_seitenknickausleger, prop_e2580_s_schild_mm, prop_e2590_starres_band,
prop_e2610_steigfaehigkeit_ohne_vibration, prop_e2615_steigfaehigkeit_mit_vibration,
prop_e2620_su_schild_mm, prop_e2640_teleskopausleger,
prop_e2650_temperaturmessung_asphalt, prop_e2670_tiltrotator,
prop_e2680_traegergeraet_typ, prop_e2700_tragkraft_an_der_spitze_kg,
prop_e2710_tragkraft_max_kg, prop_e2760_truck_assist, prop_e2770_turmsystem_typ,
prop_e2780_u_schild_mm, prop_e2790_verdichtungsleistung_kg,
prop_e2800_verdichtungsmesser, prop_e2810_verstellausleger,
prop_e2820_vor_und_ruecklauf, prop_e2830_vorlauf, prop_e2840_vorruestung_2d_steuerung,
prop_e2850_vorruestung_3d_steuerung, prop_e2860_walzendrehvorrichtung,
prop_e2900_wechselhaltersystem_typ, prop_e2910_wegmessesensoren_zylinder,
prop_e2920_wetterschutzdach, prop_e2930_winde_typ, prop_e2940_zahntyp,
prop_e2950_zentralschmierung, prop_e2970_bohle_typ,
prop_e2980_rotationsgeschwindigkeit_u_min, prop_e2990_durchflussmenge_l_min,
prop_e3000_einbaustaerke_mm, prop_e3010_zul_reisskraft_knm,
prop_e3020_empf_baggerklasse_t, prop_e3030_vm_38_schnittstelle,
prop_e3040_vorruestung_navitronic, prop_e3050_drehmulde, prop_e3060_vorruestung_voelkel,
prop_e3070_einbau_von_hgt_schotter, prop_e3080_fuehrerscheinklasse,
prop_e3090_stuetzlast_kg, prop_e3100_streben_stege, prop_e3110_farbe,
prop_e3120_getriebe_typ, prop_e3130_getriebe_art, prop_e3150_reifengroesse,
prop_e3160_co2_emissionen_g_km, prop_e3170_umweltplakette_de, prop_e3180_splittstreuer,
prop_e3190_anbauplattenverdichter, prop_e3200_batterie_typ, properties_jsonb

═══════════════════════════════════════════════════════════════
WERKZEUGE:
═══════════════════════════════════════════════════════════════
1. query_equipment - Hauptwerkzeug für Maschinensuche
   category, manufacturer, property_column (*_num!), property_operator, property_value
   
2. lookup_equipment - Spezifische Maschine nach Name/Seriennummer

3. get_equipment_details - Detaillierte Eigenschaften einer Maschine
   equipment_id oder serial_number, property_filter (optional)
   → Für Bohlentyp, Verbreiterungen, technische Details

4. execute_sql - Direkte SQL (IMMER ILIKE für Text!)

5. search_documents - Technische Dokumentation

6. explore_column - Werte einer Spalte anzeigen

═══════════════════════════════════════════════════════════════
WICHTIGE SPALTEN-ZUORDNUNG:
═══════════════════════════════════════════════════════════════
FERTIGER Einbaubreite → einbaubreite_max_m_num (IMMER!)
BAGGER Grabtiefe → grabtiefe_mm_num
FRÄSE Fräsbreite → fraesbreite_mm_num
Durchfahrtsbreite → breite_mm_num

═══════════════════════════════════════════════════════════════
BEISPIELE:
═══════════════════════════════════════════════════════════════
Fertiger 2m Einbaubreite:
→ query_equipment(category="Fertiger", property_column="einbaubreite_max_m_num", property_operator=">=", property_value=2.0)

Bagger 5m Grabtiefe:
→ query_equipment(category="Bagger", property_column="grabtiefe_mm_num", property_operator=">=", property_value=5000)

Seriennummer Super 800i:
→ lookup_equipment("Super 800i")

Bohlentyp einer Maschine:
→ get_equipment_details(serial_number="09901387", property_filter="bohle")

═══════════════════════════════════════════════════════════════
REGELN:
═══════════════════════════════════════════════════════════════
🔴 Numerische Filter: *_num Spalten verwenden!
🔴 Einheiten: _mm_num=Millimeter, _m_num=Meter, _kg_num=Kilogramm
🔴 DU HAST ZUGRIFF! Sage niemals "kein Zugriff"

Antworte auf Deutsch. Kurz und präzise.
"""


# =============================================================================
# Agent Class
# =============================================================================

@dataclass
class AgentResult:
    """Result from LangGraph agent processing."""
    response: str
    tools_used: List[str]
    execution_time_ms: int
    token_usage: Optional[dict] = None
    sources: Optional[List[str]] = None
    timing_breakdown: Optional[dict] = None


class LangGraphAgent:
    """LangGraph ReAct agent for equipment queries."""

    def __init__(self, redis_url: Optional[str] = None):
        """Initialize the LangGraph agent."""
        from langgraph.prebuilt import create_react_agent
        from langgraph.checkpoint.memory import MemorySaver

        # Initialize LLM - prefer Groq if configured, fallback to OpenAI
        if config.groq_api_key:
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(
                model=config.groq_model,
                temperature=0,
                api_key=config.groq_api_key
            )
            logger.info(f"LangGraph using Groq model: {config.groq_model}")
        else:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=config.openai_model,
                temperature=0,
                api_key=config.openai_api_key
            )
            logger.info(f"LangGraph using OpenAI model: {config.openai_model}")

        # Collect tools - query_equipment is the main one for structured queries
        self.tools = [
            query_equipment,
            lookup_equipment,
            get_equipment_details,
            execute_sql,
            search_documents,
            find_columns,
            explore_column
        ]

        # Enable MemorySaver - preserves context including tool calls between turns
        self.checkpointer = MemorySaver()
        logger.info("LangGraph checkpointer enabled (MemorySaver)")

        # Create ReAct agent with checkpointer (tool output is reduced to limit tokens)
        self.graph = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT,
            checkpointer=self.checkpointer
        )

        logger.info(f"LangGraph agent initialized with {len(self.tools)} tools")

    async def process(
        self,
        user_query: str,
        thread_key: str,
        conversation_history: Optional[List[dict]] = None  # Kept for API compat, ignored
    ) -> AgentResult:
        """Process a user query through the ReAct agent.

        With MemorySaver checkpointer, context (including tool calls) is preserved per thread_key.
        """
        start_time = time.time()

        # Initialize timing collector
        timing = TimingCollector()
        set_timing_collector(timing)

        if VERBOSE:
            logger.info(f"LangGraph processing query: {user_query[:100]}...")

        # Just send new message - checkpointer preserves full history per thread_key
        messages = [("user", user_query)]

        # Configure with thread_id for checkpointing
        run_config = {"configurable": {"thread_id": thread_key}}

        try:
            # Invoke the graph with rate limit retry
            llm_start = time.time()
            try:
                result = await self.graph.ainvoke(
                    {"messages": messages},
                    config=run_config
                )
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    logger.warning("Rate limited, waiting 5s before retry...")
                    timing.record("llm:rate_limit_wait", 5000, "429 retry")
                    await asyncio.sleep(5)
                    llm_start = time.time()
                    result = await self.graph.ainvoke(
                        {"messages": messages},
                        config=run_config
                    )
                else:
                    raise

            llm_total_ms = (time.time() - llm_start) * 1000

            # Extract response from last message
            final_message = result["messages"][-1]
            response = final_message.content if hasattr(final_message, "content") else str(final_message)

            # Extract tools used and count LLM calls
            tools_used = self._extract_tools_used(result["messages"])
            llm_call_count = self._count_llm_calls(result["messages"])

            # Record LLM timing (total minus tool time gives approximate LLM time)
            tool_time = sum(r.duration_ms for r in timing.records if r.operation.startswith("tool:"))
            llm_time = llm_total_ms - tool_time
            timing.record("llm:total", llm_time, f"{llm_call_count} calls")

            execution_time = int((time.time() - start_time) * 1000)

            # Print timing summary
            timing.print_summary()

            # Clear timing collector
            set_timing_collector(None)

            return AgentResult(
                response=response,
                tools_used=tools_used,
                execution_time_ms=execution_time,
                timing_breakdown=timing.get_summary()
            )

        except Exception as e:
            logger.error(f"LangGraph agent error: {e}", exc_info=True)
            set_timing_collector(None)
            raise

    def _extract_tools_used(self, messages: list) -> List[str]:
        """Extract unique tool names from message history."""
        tools = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    if name and name not in tools:
                        tools.append(name)
        return tools

    def _count_llm_calls(self, messages: list) -> int:
        """Count the number of LLM calls (AI messages) in the conversation."""
        count = 0
        for msg in messages:
            # AIMessage indicates an LLM response
            msg_type = type(msg).__name__
            if msg_type == "AIMessage" or (hasattr(msg, "type") and msg.type == "ai"):
                count += 1
        return count


# Singleton instance
_agent_instance: Optional[LangGraphAgent] = None


def get_langgraph_agent() -> LangGraphAgent:
    """Get or create the singleton LangGraph agent instance."""
    global _agent_instance
    if _agent_instance is None:
        redis_url = config.redis_url if hasattr(config, "redis_url") else None
        _agent_instance = LangGraphAgent(redis_url=redis_url)
    return _agent_instance