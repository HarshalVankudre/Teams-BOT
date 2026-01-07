"""
PostgreSQL Database Schema for the SEMA equipment database.

This is the SINGLE SOURCE OF TRUTH for the schema used by the SQL agent prompts.
The bot connects to the database configured via env vars (see `.env.example`).
"""

import os

_SCHEMA = os.getenv("POSTGRES_SCHEMA") or "<POSTGRES_SCHEMA>"
_TABLE = os.getenv("POSTGRES_EQUIPMENT_TABLE") or "<POSTGRES_EQUIPMENT_TABLE>"
_EQUIPMENT_TABLE_FQN = f"{_SCHEMA}.{_TABLE}"

# =============================================================================
# COMPLETE DATABASE SCHEMA (SEMA_CHATBOT)
# =============================================================================

DATABASE_SCHEMA = f"""
================================================================================
DATABASE: sema_chatbot (SEMA Equipment)
================================================================================
BASE TABLE
----------
public.equipment_wide
- Raw export with ibs_nuclet_geraete_* core columns and prop_e####_* properties (TEXT).
- Many prop values include units (e.g., "3,20 m - Meter"). Parse when needed.

COMPATIBILITY VIEW
------------------
{_EQUIPMENT_TABLE_FQN}
- View over public.equipment_wide that exposes normalized columns used by the agent.
- Use the view for numeric filters (parsed to DOUBLE) and booleans.

RAW CORE COLUMNS (public.equipment_wide)
----------------------------------------
- ibs_nuclet_geraete_primarykey (INTEGER) -> unique id
- ibs_nuclet_geraete_bezeichnung (TEXT)
- ibs_nuclet_geraete_seriennummer (TEXT)
- ibs_nuclet_geraete_inventarnummer (TEXT)
- ibs_nuclet_geraete_hersteller (TEXT)        -> "CODE - Name"
- ibs_nuclet_geraete_geraetegruppe (TEXT)     -> "CODE - Name"
- ibs_nuclet_geraete_verwendung (TEXT)        -> "MIET - Vermietung", "VK - Verkauf", ...
- ibs_nuclet_geraete_nuclosstate (TEXT)       -> Released / Locked / Verkauft / Created
- ibs_nuclet_geraete_nuclosprocess (TEXT)

NORMALIZED CORE COLUMNS (view)
------------------------------
- id (BIGINT)                     -> ibs_nuclet_geraete_primarykey
- bezeichnung (TEXT)              -> ibs_nuclet_geraete_bezeichnung
- seriennummer (TEXT)             -> ibs_nuclet_geraete_seriennummer
- inventarnummer (TEXT)           -> ibs_nuclet_geraete_inventarnummer
- hersteller_code (TEXT)          -> split_part(ibs_nuclet_geraete_hersteller, ' - ', 1)
- hersteller_name (TEXT)          -> split_part(ibs_nuclet_geraete_hersteller, ' - ', 2)
- geraetegruppe_code (TEXT)       -> split_part(ibs_nuclet_geraete_geraetegruppe, ' - ', 1)
- geraetegruppe_name (TEXT)       -> split_part(ibs_nuclet_geraete_geraetegruppe, ' - ', 2)
- verwendung_code (TEXT)          -> split_part(ibs_nuclet_geraete_verwendung, ' - ', 1)
- verwendung_name (TEXT)          -> split_part(ibs_nuclet_geraete_verwendung, ' - ', 2)
- nuclos_state (TEXT)             -> ibs_nuclet_geraete_nuclosstate
- nuclos_process (TEXT)           -> ibs_nuclet_geraete_nuclosprocess

GERAETEGRUPPEN / EQUIPMENT CATEGORIES (KRITISCH!)
-------------------------------------------------
geraetegruppe_name enthaelt EXAKTE Kategorienamen - KEINE Property-Kombinationen!
WICHTIG: "Kettenfertiger", "Radfertiger", "Mobilbagger" etc. sind eigenstaendige Kategorien!

FALSCH: WHERE geraetegruppe_name ILIKE '%fertiger%' AND prop_e2100_mobil_kette IS NOT NULL
RICHTIG: WHERE geraetegruppe_name = 'Kettenfertiger'

Beispiele fuer Kategorien mit "Kette/Rad/Mobil" im Namen:
- Kettenfertiger (98 Maschinen) -> WHERE geraetegruppe_name = 'Kettenfertiger'
- Radfertiger (9 Maschinen) -> WHERE geraetegruppe_name = 'Radfertiger'  
- Kaltfraese (Kette) (118) -> WHERE geraetegruppe_name = 'Kaltfraese (Kette)'
- Kaltfraese (Rad) (26) -> WHERE geraetegruppe_name = 'Kaltfraese (Rad)'
- Mobilbagger (53) -> WHERE geraetegruppe_name = 'Mobilbagger'
- Kettenbagger (21) -> WHERE geraetegruppe_name = 'Kettenbagger'
- Radlader (71) -> WHERE geraetegruppe_name = 'Radlader'
- Telekran (Rad) (16) -> WHERE geraetegruppe_name = 'Telekran (Rad)'
- Telekran (Kette) (14) -> WHERE geraetegruppe_name = 'Telekran (Kette)'

Bei unbekannten Kategorien ZUERST abfragen:
  SELECT DISTINCT geraetegruppe_name, COUNT(*) as cnt 
  FROM {_EQUIPMENT_TABLE_FQN} 
  WHERE geraetegruppe_name ILIKE '%suchbegriff%'
  GROUP BY geraetegruppe_name;

AVAILABILITY (practical)
------------------------
- Prefer `nuclos_state = 'Released'` (or ibs_nuclet_geraete_nuclosstate = 'Released') for "verfuegbar".
- Treat `nuclos_state = 'Locked'` as not available (only mention as fallback with caveat).

DERIVED PROPERTIES (view)
-------------------------
- prop_klimaanlage (BOOLEAN) from prop_e1930_klimaanlage ("Ja"/"Nein")
- prop_gewicht (DOUBLE) from prop_e1730_gewicht_kg (kg)
- prop_motor_leistung (DOUBLE) from prop_e2180_motor_leistung_kw (kW)
- prop_einbaubreite_max (DOUBLE) from prop_e1480_einbaubreite_max_m (m)
- prop_einbaubreite_grundbohle (DOUBLE) from prop_e1470_einbaubreite_grundbohle_m
  (m; range uses the upper bound when present)
- prop_einbaubreite_mit_verbreiterungen (DOUBLE) from prop_e1490_einbaubreite_mit_verbreiterungen_m (m)
- prop_arbeitsbreite (DOUBLE) from prop_e1150_arbeitsbreite_mm (mm)

RAW PROPERTIES (equipment_wide)
-------------------------------
- prop_e####_* are TEXT columns with units.
- Use property_name_map to map human labels to columns:
  SELECT property_name, column_name
  FROM public.property_name_map
  WHERE property_name ILIKE '%Klimaanlage%';

If you are unsure which columns exist (or their data types), query:
  SELECT column_name, data_type
  FROM information_schema.columns
  WHERE table_schema = '{_SCHEMA}' AND table_name = '{_TABLE}'
  ORDER BY ordinal_position;

SAFE QUERY PATTERNS (copy/paste friendly)
----------------------------------------
Total count:
  SELECT COUNT(*) AS count FROM {_EQUIPMENT_TABLE_FQN};

Mietmaschinen (MIET):
  SELECT COUNT(*) AS count
  FROM {_EQUIPMENT_TABLE_FQN}
  WHERE verwendung_code = 'MIET';

Raw alternative:
  SELECT COUNT(*) AS count
  FROM {_EQUIPMENT_TABLE_FQN}
  WHERE ibs_nuclet_geraete_verwendung ILIKE 'MIET -%';

5 Bomag-Maschinen:
  SELECT id, bezeichnung, seriennummer, inventarnummer, verwendung_code, nuclos_state
  FROM {_EQUIPMENT_TABLE_FQN}
  WHERE (hersteller_name ILIKE '%bomag%' OR hersteller_code = 'BOM')
  ORDER BY bezeichnung
  LIMIT 5;

Mietmaschinen von Bomag mit Klimaanlage:
  SELECT id, inventarnummer, seriennummer, bezeichnung
  FROM {_EQUIPMENT_TABLE_FQN}
  WHERE verwendung_code = 'MIET'
    AND (hersteller_name ILIKE '%bomag%' OR hersteller_code = 'BOM')
    AND prop_klimaanlage IS TRUE
  LIMIT 10;

Fertiger fuer 3.0m Einbaubreite (Beispiel):
  SELECT id, bezeichnung, hersteller_name, geraetegruppe_name, verwendung_code,
         nuclos_state, prop_einbaubreite_max, prop_einbaubreite_grundbohle
  FROM {_EQUIPMENT_TABLE_FQN}
  WHERE geraetegruppe_name ILIKE '%fertiger%'
    AND COALESCE(prop_einbaubreite_max, 0) >= 3.0
    AND verwendung_code = 'MIET'
    AND nuclos_state = 'Released'
  ORDER BY prop_einbaubreite_max DESC
  LIMIT 10;

Recommendation ranking (multi-criteria; example for 3.0m):
  WITH candidates AS (
    SELECT
      e.*,
      (COALESCE(e.prop_einbaubreite_max, 0) - 3.0) AS fit_delta,
      (SELECT COUNT(*) FROM jsonb_each(jsonb_strip_nulls(to_jsonb(e)))) AS data_completeness
    FROM {_EQUIPMENT_TABLE_FQN} e
    WHERE e.geraetegruppe_name ILIKE '%fertiger%'
      AND COALESCE(e.prop_einbaubreite_max, 0) >= 3.0
      AND e.verwendung_code = 'MIET'
  )
  SELECT
    id, bezeichnung, hersteller_name, geraetegruppe_name, verwendung_code, nuclos_state,
    prop_einbaubreite_max, prop_einbaubreite_grundbohle,
    prop_motor_leistung, prop_gewicht,
    fit_delta, data_completeness
  FROM candidates
  ORDER BY (nuclos_state = 'Released') DESC,
           fit_delta ASC NULLS LAST,
           data_completeness DESC,
           prop_motor_leistung DESC NULLS LAST
  LIMIT 10;

Search by serial/inventory:
  SELECT id, bezeichnung, hersteller_name, geraetegruppe_name, verwendung_code
  FROM {_EQUIPMENT_TABLE_FQN}
  WHERE seriennummer ILIKE '%10187%'
     OR inventarnummer ILIKE '%18653%'
  LIMIT 10;
"""

# The agent prompt imports this name.
SQL_AGENT_SCHEMA = DATABASE_SCHEMA
