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

IMPORTANT - Category filtering:
- Use exact match: WHERE geraetegruppe_name = 'Kettenfertiger'
- NOT: WHERE geraetegruppe_name ILIKE '%fertiger%' AND prop_e2100_mobil_kette IS NOT NULL

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

PROPERTY COLUMNS:
Property columns are named prop_e####_name_unit (e.g., prop_e1740_grabtiefe_mm).
Values are stored as TEXT with units like "3410 mm - Millimeter".
To filter numeric values:
  CAST(NULLIF(regexp_replace(column_name, '[^0-9]', '', 'g'), '') AS NUMERIC)

For boolean properties: WHERE column_name = 'Ja' or IS NOT NULL

See the PROPERTY COLUMNS CATALOG section for full list of available columns.

BASIC QUERY PATTERNS:
---------------------
Count total:
  SELECT COUNT(*) AS count FROM {EQUIPMENT_TABLE_FQN};

Rental machines:
  SELECT COUNT(*) FROM {EQUIPMENT_TABLE_FQN} WHERE verwendung_code = 'MIET';

Filter by manufacturer:
  SELECT id, bezeichnung, hersteller_name FROM {EQUIPMENT_TABLE_FQN}
  WHERE hersteller_name ILIKE '%bomag%' LIMIT 10;

Search by serial/inventory:
  SELECT id, bezeichnung FROM {EQUIPMENT_TABLE_FQN}
  WHERE seriennummer ILIKE '%search%' OR inventarnummer ILIKE '%search%' LIMIT 10;

Search by model/name (bezeichnung):
  SELECT id, bezeichnung, hersteller_name FROM {EQUIPMENT_TABLE_FQN}
  WHERE bezeichnung ILIKE '%search_term%' LIMIT 10;
"""

# Alias for backward compatibility
SQL_AGENT_SCHEMA = DATABASE_SCHEMA
