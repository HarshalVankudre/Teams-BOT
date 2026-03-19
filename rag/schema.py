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
DATABASE: SEMA Equipment ({EQUIPMENT_TABLE_FQN})

TABLE: {EQUIPMENT_TABLE_FQN}
Contains equipment inventory and machine properties.

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
- nuclos_state (TEXT) - Bestandsstatus im System, NICHT die Live-Verfuegbarkeit

EQUIPMENT CATEGORIES (geraetegruppe_name values):
Use exact names for filtering. Examples:
- Kettenfertiger, Radfertiger (pavers)
- Kettenbagger, Mobilbagger, Minibagger, Kompaktbagger (excavators)
- Radlader (wheel loaders)
- Walze, Tandemwalze (rollers)
- Kaltfraese (Kette), Kaltfraese (Rad) (cold milling machines)
- Telekran (Kette), Telekran (Rad) (cranes)

IMPORTANT - Category filtering:
- Use exact match: WHERE geraetegruppe_name = 'Kettenfertiger'
- If unsure, query distinct values first instead of guessing.

NUCLOS-STATUS:
- Released = im Bestand / im System freigegeben
- Locked = gesperrt / nicht freigegeben
- Wichtig: Aus `nuclos_state` keine echte Dispositions- oder Miet-Verfuegbarkeit ableiten

USAGE:
- MIET = rental
- VK = sale/purchase

KOSTENSTELLE:
- Stored in ibs_nuclet_geraete_kostenstelle as "CODE - Name"
- Example: "200 - Mietpark", "100 - Handel"

PROPERTY COLUMNS:
- Named like prop_e####_name_unit
- Many values are stored as TEXT with units
- For numeric filters use numeric helper columns where available, otherwise cast cleaned text
- Boolean properties are often 'Ja' or non-null

BASIC QUERY PATTERNS:
- Count total records
- Filter by manufacturer with hersteller_name ILIKE
- Search by seriennummer or inventarnummer
- Search by bezeichnung for model or name lookups
"""

# Alias for backward compatibility
SQL_AGENT_SCHEMA = DATABASE_SCHEMA
