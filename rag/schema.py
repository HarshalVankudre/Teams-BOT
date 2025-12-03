"""
PostgreSQL Database Schema for RUKO Equipment Database

This is the SINGLE SOURCE OF TRUTH for the database schema.
All agents and services should import from here.

Table: geraete (Baumaschinen/Construction Equipment)
Total Records: 2395

Schema rebuilt from raw SEMA export (2024-12-03):
- id: BIGINT PRIMARY KEY (SEMA primaryKey)
- Direct columns for basic equipment info
- eigenschaften: JSONB for all technical properties
- Common numeric/boolean columns extracted for fast queries
"""

# =============================================================================
# COMPLETE DATABASE SCHEMA
# =============================================================================

DATABASE_SCHEMA = """
================================================================================
DATENBANK-SCHEMA: Tabelle "geraete" (Baumaschinen-Inventar)
================================================================================

Schema basiert auf SEMA Raw-Export mit 2395 Geraeten.
Jedes Geraet hat eine eindeutige ID (SEMA primaryKey).

================================================================================
IDENTIFIKATION
================================================================================
- id: BIGINT PRIMARY KEY (SEMA primaryKey, z.B. 67251777)
- bezeichnung: TEXT NOT NULL - Modellname (z.B. "BW 174 AP-5 AM", "CAT 320")
- seriennummer: TEXT - Seriennummer (1286 eindeutige)
- inventarnummer: TEXT - Interne Inventarnummer

================================================================================
KLASSIFIKATION
================================================================================
- hersteller: TEXT - ACHTUNG: Meist einfache Namen, ABER manche haben 'CODE - Name' Format!
  Einfach: 'Caterpillar', 'Liebherr', 'Bomag', 'Volvo', 'Wirtgen'
  Mit Code: 'VÖG - Vögele' (283), '??? - Sonstige' (65)
  EMPFEHLUNG: Immer ILIKE '%suchbegriff%' verwenden!
- hersteller_code: TEXT - z.B. 'CAT', 'LIE', 'BOM', 'VÖG'

- geraetegruppe: TEXT - WICHTIGSTE SPALTE! z.B.:
  Bagger: 'Mobilbagger', 'Kettenbagger', 'Minibagger (0,0 to - 4,4 to)'
  Walzen: 'Tandemwalze', 'Walzenzug (Glattmantel)', 'Gummiradwalze'
  Fertiger: 'Radfertiger', 'Kettenfertiger', 'Beschicker'
  Fraesen: 'Kaltfraese (Kette)', 'Kaltfraese (Rad)', 'Anbaufraese'
- geraetegruppe_code: TEXT - z.B. '5.6030.000', '2.2010.000'

- kategorie: TEXT - Oberkategorie (abgeleitet aus geraetegruppe_code):
  'bagger', 'anbaugeraet', 'verdichter', 'beschicker', 'fertiger',
  'fraese', 'kran', 'lader', 'dumper', 'sonstige'

- verwendung: TEXT - Werte und ihre Codes:
  'Vermietung' (MIET) - 794 Geraete
  'Verkauf' (VK) - 1371 Geraete
  'Fuhrpark' (FP) - 8 Geraete
  'Externes Geraet' (EXG) - 221 Geraete
- verwendung_code: TEXT - 'MIET', 'VK', 'FP', 'EXG'

- abrechnungsgruppe: TEXT - Format: 'CODE - Beschreibung'
  z.B. '4.3030.010 - Bohlenverbreiterungen fuer Fertiger'
  WICHTIG: Immer ILIKE '%suchbegriff%' verwenden!

- kostenstelle: TEXT - Format: 'CODE - Name'
  '100 - Handel' (1462 Geraete)
  '200 - Mietpark' (854 Geraete)
  '90000 - Fuhrpark' (9 Geraete)
  WICHTIG: Fuer Code-Suche ILIKE '200%' verwenden, NICHT '200-' oder '200'!

================================================================================
EXTRAHIERTE NUMERISCHE SPALTEN (direkt abfragbar)
================================================================================
Haeufig abgefragte Werte sind als direkte Spalten verfuegbar:

- breite_mm: NUMERIC - Breite in mm
- hoehe_mm: NUMERIC - Hoehe in mm
- laenge_mm: NUMERIC - Laenge in mm
- gewicht_kg: NUMERIC - Betriebsgewicht in kg
- motor_leistung_kw: NUMERIC - Motorleistung in kW

Zugriff: WHERE gewicht_kg > 15000

================================================================================
EXTRAHIERTE BOOLEAN SPALTEN (direkt abfragbar)
================================================================================
- klimaanlage: BOOLEAN
- zentralschmierung: BOOLEAN

Zugriff: WHERE klimaanlage = true

================================================================================
JSONB SPALTE: eigenschaften
================================================================================
Alle technischen Eigenschaften aus SEMA als JSONB gespeichert.

Format: {"Eigenschaft Name": {"wert": "Wert", "einheit": "Einheit"}}

Beispiele:
  {"Klimaanlage": {"wert": "Ja", "einheit": null}}
  {"Arbeitsbreite [mm]": {"wert": "1700", "einheit": "mm"}}
  {"Motor - Leistung [kW]": {"wert": "85", "einheit": "kW"}}
  {"Oszillation": {"wert": "Ja", "einheit": null}}

JSONB-Zugriff:
  -- Pruefen ob Eigenschaft existiert (Boolean):
  eigenschaften ? 'Klimaanlage'

  -- Wert einer Eigenschaft:
  eigenschaften->'Klimaanlage'->>'wert' = 'Ja'

  -- Numerischen Wert extrahieren:
  (eigenschaften->'Arbeitsbreite [mm]'->>'wert')::numeric > 1500

Haeufige Eigenschaften in JSONB:
  Boolean (Wert: 'Ja'):
    - Klimaanlage, Zentralschmierung, Hammerhydraulik
    - Oszillation, Verdichtungsmesser, Allradantrieb
    - Dieselpartikelfilter, Kabine, Gas-Heizung
    - Tiltrotator, Powertilt, Schnellwechsler (hydr.)

  Numerisch:
    - Gewicht [kg], Breite [mm], Hoehe [mm], Laenge [mm]
    - Motor - Leistung [kW], Arbeitsbreite [mm]
    - Grabtiefe [mm], Loeffelstiel [mm]
    - Fraesbreite [mm], Fraestiefe [mm]
    - Ausladung [m], Ausleger [m], Hakenhoehe [m]

================================================================================
TIMESTAMPS
================================================================================
- created_at: TIMESTAMP - Erstellungsdatum

================================================================================
SQL-BEISPIELE
================================================================================

-- Alle Bagger zaehlen:
SELECT COUNT(*) FROM geraete WHERE geraetegruppe ILIKE '%bagger%'

-- Liebherr Maschinen auflisten:
SELECT bezeichnung, geraetegruppe, seriennummer
FROM geraete
WHERE hersteller = 'Liebherr'

-- Bagger ueber 15t mit Klimaanlage:
SELECT bezeichnung, hersteller, gewicht_kg
FROM geraete
WHERE geraetegruppe ILIKE '%bagger%'
  AND gewicht_kg > 15000
  AND klimaanlage = true

-- Walzen mit Oszillation (via JSONB):
SELECT bezeichnung, hersteller
FROM geraete
WHERE geraetegruppe ILIKE '%walze%'
  AND eigenschaften ? 'Oszillation'

-- Fertiger mit Gas-Heizung (via JSONB):
SELECT bezeichnung, hersteller
FROM geraete
WHERE geraetegruppe ILIKE '%fertiger%'
  AND eigenschaften->'Gas-Heizung'->>'wert' = 'Ja'

-- Mietmaschinen:
SELECT hersteller, bezeichnung, geraetegruppe
FROM geraete
WHERE verwendung = 'Vermietung'

-- Geraete nach Kostenstelle (ILIKE fuer Code-Suche!):
SELECT COUNT(*) FROM geraete WHERE kostenstelle ILIKE '200%'
-- Ergebnis: 854 (findet '200 - Mietpark')
-- FALSCH: WHERE kostenstelle = '200' (findet nichts!)

-- Voegele Maschinen (Hersteller mit Code-Format):
SELECT COUNT(*) FROM geraete WHERE hersteller ILIKE '%voegele%' OR hersteller ILIKE '%VÖG%'
-- Ergebnis: 283

================================================================================
WICHTIGE REGELN
================================================================================
1. Fuer Geraetetypen IMMER geraetegruppe verwenden, NICHT kategorie!
2. verwendung-Filter NUR wenn explizit angefragt!
3. KEIN LIMIT bei Auflistungs-/Zaehlabfragen!
4. Direkte Spalten wenn verfuegbar: gewicht_kg, klimaanlage
5. JSONB fuer alle anderen Eigenschaften
6. NULL-Werte pruefen bei Aggregationen: gewicht_kg IS NOT NULL

================================================================================
ILIKE-REGEL (SEHR WICHTIG!)
================================================================================
Viele Spalten haben 'CODE - Name' Format. Bei Suche nach Code/Teilstring:
- FALSCH: WHERE kostenstelle = '200' (findet nichts!)
- RICHTIG: WHERE kostenstelle ILIKE '200%' (findet '200 - Mietpark')

Spalten mit 'CODE - Name' Format:
- kostenstelle: '100 - Handel', '200 - Mietpark', '90000 - Fuhrpark'
- abrechnungsgruppe: '4.3030.010 - Bohlenverbreiterungen...'
- hersteller (teilweise): 'VÖG - Vögele', '??? - Sonstige'

Bei Unsicherheit IMMER ILIKE '%suchbegriff%' verwenden!
"""

# =============================================================================
# SCHEMA FOR SQL AGENT
# =============================================================================

SQL_AGENT_SCHEMA = DATABASE_SCHEMA

# =============================================================================
# SCHEMA FOR ORCHESTRATOR (simplified)
# =============================================================================

ORCHESTRATOR_SCHEMA = """
DATENBANK-SCHEMA: Tabelle "geraete" (2395 Baumaschinen)

DIREKTE SPALTEN:
- id: BIGINT Primary Key (SEMA primaryKey)
- bezeichnung: Modellname (z.B. "CAT 320", "BW 174 AP-5 AM")
- hersteller: z.B. 'Caterpillar', 'Liebherr', 'Bomag'
- geraetegruppe: WICHTIGSTE SPALTE! z.B. 'Mobilbagger', 'Tandemwalze'
- kategorie: Oberkategorie (oft NULL - geraetegruppe bevorzugen!)
- verwendung: 'Vermietung', 'Verkauf', 'Fuhrpark'
- seriennummer, inventarnummer

NUMERISCHE SPALTEN (direkt):
- gewicht_kg, motor_leistung_kw, breite_mm, hoehe_mm, laenge_mm

BOOLEAN SPALTEN (direkt):
- klimaanlage, zentralschmierung

JSONB (eigenschaften):
- Alle anderen Eigenschaften: Oszillation, Hammerhydraulik, etc.
- Format: {"Name": {"wert": "Wert", "einheit": "..."}}

BEISPIEL-ANFRAGEN:
- "Liebherr Maschinen" -> hersteller = 'Liebherr'
- "Alle Bagger" -> geraetegruppe ILIKE '%bagger%'
- "Mietmaschinen" -> verwendung = 'Vermietung'
- "Bagger ueber 15t" -> gewicht_kg > 15000
- "Mit Klimaanlage" -> klimaanlage = true
- "Walzen mit Oszillation" -> eigenschaften ? 'Oszillation'
"""

# =============================================================================
# VALUE LISTS (from actual database)
# =============================================================================

HERSTELLER_VALUES = [
    'ABG', 'Ahlmann', 'AL-KO', 'Ammann', 'Atlas', 'Atlas-Copco', 'AUGER TORQUE',
    'Baumgaertner', 'Bema', 'Bergmann', 'Boart Longyear', 'Bomag', 'Breining',
    'Brian James Trailers', 'Cardi', 'Caterpillar', 'Cedima', 'Compair', 'Containex',
    'CP', 'Demag', 'Ditch Witch', 'DMS', 'Dynapac', 'Egli', 'Endress', 'Engcon',
    'Epiroc', 'Ford', 'Format', 'Furukawa', 'GEKO', 'Genie', 'GRUEN GmbH', 'Gruenig',
    'Hamm', 'HBM NOBAS', 'Henle', 'Heylo', 'HIMOINSA', 'Hitachi', 'HKS', 'Hufgard',
    'Hulco', 'Hydraulikgreifer-Technologie GmbH', 'Hydrema', 'Hyster', 'Hyundai',
    'JCB', 'John Deere', 'Jungheinrich', 'Kaeser', 'Kinshofer', 'Kleemann', 'Kobelco',
    'Komatsu', 'Korte', 'Kramer Allrad', 'Kraenzle', 'Kroll', 'Krupp', 'KSG', 'Ksw',
    'Kubota', 'Kwanglim', 'Lehnhoff', 'Liebherr', 'MAEDA', 'MAN', 'Manitou', 'MBU',
    'McCloskey', 'Mercedes-Benz', 'Merlo', 'Moba', 'MTS', 'Mueller Mitteltal', 'Neuson',
    'New Holland', 'Niftylift', 'Nilfisk Alto', 'NOZAR', 'Oilquick', 'O&K', 'Opel',
    'Paus', 'Potain', 'Rammax', 'Renault', 'Reschke', 'Rototilt', 'RUEKO', 'Schaeff',
    'Sennebogen', 'Sitech', 'Skoda', 'SMP', 'Sobernheimer', 'Sonstige', 'Steelwrist',
    'Stehr', 'Stihl', 'Strassmayr', 'Streumaster', 'Takeuchi', 'Tesla', 'Theis',
    'Thwaites', 'Tracto-Technik', 'Trimble', 'TS Industrie', 'Tuchel', 'Voegele',
    'Volkswagen', 'Volvo', 'Wacker Neuson', 'Weber mt', 'Weber Stahl', 'Weiro',
    'Wirtgen', 'Yanmar', 'Zeppelin', 'ZFE GmbH'
]

GERAETEGRUPPE_VALUES = [
    # Bagger
    'Mobilbagger', 'Kettenbagger', 'Minibagger (0,0 to - 4,4 to)',
    'Kompaktbagger (4,5 to - 10,9 to)',
    # Walzen
    'Tandemwalze', 'Walzenzug (Glattmantel)', 'Walzenzug (Schaffuss)',
    'Gummiradwalze', 'Grabenwalze', 'Kombiwalze',
    # Fertiger
    'Radfertiger', 'Kettenfertiger', 'Beschicker',
    # Fraesen
    'Kaltfraese (Kette)', 'Kaltfraese (Rad)', 'Anbaufraese', 'Grabenfraese', 'Anbaugrabenfraese',
    # Krane
    'Autokran', 'Telekran (Kette)', 'Telekran (Rad)', 'Miniraupenkran',
    'Obendreher-Kran', 'Untendreher-Kran',
    # Lader
    'Radlader', 'Teleskoplader (starr)', 'Kettendumper', 'Raddumper', 'Laderaupe',
    # Loeffel & Anbaugeraete
    'Tiefloeffel', 'Tiefloeffel (hydraulisch schwenkbar)', 'Grabenraeumloeffel (starr)',
    'Grabenraeumloeffel (hydraulisch schwenkbar)', 'Siebloeffel', 'Brecherloeffel',
    'Spatensloeffel', 'Teleloeffel',
    # Greifer & Zangen
    'Abbruch- und Sortiergreifer', 'Abbruchzange/Pulverisierer', 'Pendelgreifer',
    # Sonstige
    'Hydraulikhammer', 'Tiltrotator (Sandwich)', 'Vibrationsplatte', 'Vibrationsstampfer',
    'Stromerzeuger', 'Kompressor', 'Fugenschneider', 'Kernbohrgeraet', 'Hochdruckreiniger',
    'Gabelstapler', 'Anhaengerarbeitsbuehne', 'Scherenarbeitsbuehne (elektrisch)',
    'Gelenk-Teleskoparbeitsbuehne', 'Siebanlage', 'Backenbrecher', 'Prallbrecher'
]

KATEGORIE_VALUES = [
    'bagger', 'anbaugeraet', 'verdichter', 'beschicker', 'fertiger',
    'fraese', 'kran', 'lader', 'dumper', 'sonstige'
]

VERWENDUNG_VALUES = [
    'Vermietung', 'Verkauf', 'Fuhrpark', 'Externes Geraet', 'keine'
]

KOSTENSTELLE_VALUES = [
    '100 - Handel',      # 1462 Geraete
    '200 - Mietpark',    # 854 Geraete
    '90000 - Fuhrpark',  # 9 Geraete
]

# =============================================================================
# CODE-NAME FORMAT COLUMNS (critical for ILIKE queries)
# =============================================================================

CODE_NAME_FORMAT_COLUMNS = {
    'kostenstelle': ['100 - Handel', '200 - Mietpark', '90000 - Fuhrpark'],
    'abrechnungsgruppe': ['4.3030.010 - Bohlenverbreiterungen...'],
    'hersteller': ['VÖG - Vögele', '??? - Sonstige'],  # Partial - most are simple names
}

# SQL Agent rules for CODE-Name format (imported by sql_agent.py)
SQL_ILIKE_RULES = """
SEHR WICHTIG - 'CODE - Name' FORMAT:
Diese Spalten haben Format 'CODE - Name' (z.B. '200 - Mietpark'):
- kostenstelle: '100 - Handel', '200 - Mietpark', '90000 - Fuhrpark'
- abrechnungsgruppe: '4.3030.010 - Beschreibung'
- hersteller (teilweise): 'VÖG - Vögele'

Bei Suche nach Code/Nummer IMMER ILIKE verwenden:
- FALSCH: WHERE kostenstelle = '200' (findet NICHTS!)
- RICHTIG: WHERE kostenstelle ILIKE '200%' (findet '200 - Mietpark')
- FALSCH: WHERE kostenstelle = '200-' (findet NICHTS!)
- RICHTIG: WHERE kostenstelle ILIKE '200 -%' (findet '200 - Mietpark')
"""

# German umlaut handling rules for SQL agent
SQL_UMLAUT_RULES = """
DEUTSCHE UMLAUTE:
Die Datenbank verwendet ASCII-Ersetzungen fuer Umlaute:
- ae statt ä (z.B. 'Kaltfraese', 'Geraet')
- oe statt ö (z.B. 'Voegele', 'Loeffel')
- ue statt ü (z.B. 'Muell', 'Gruenig')
- ss statt ß (z.B. 'Strasse')

BEISPIELE:
- Voegele Maschinen: WHERE hersteller ILIKE '%voegele%' (283 Treffer)
- Fraesen: WHERE geraetegruppe ILIKE '%fraese%' (156 Treffer)
- Loeffel: WHERE geraetegruppe ILIKE '%loeffel%' (345 Treffer)
- Geraete: WHERE geraetegruppe ILIKE '%geraet%'
"""

# Combined rules for SQL agent (both CODE-Name and umlauts)
SQL_SPECIAL_RULES = SQL_ILIKE_RULES + SQL_UMLAUT_RULES

# Common JSONB property names (from eigenschaften column)
JSONB_PROPERTIES = [
    # Boolean properties (value is 'Ja')
    'Klimaanlage', 'Zentralschmierung', 'Hammerhydraulik', 'Greiferhydraulik',
    'Oszillation', 'Verdichtungsmesser', 'Allradantrieb', 'Allradlenkung',
    'Knicklenkung', 'Dieselpartikelfilter', 'Kabine', 'hochfahrbare Kabine',
    'Gas-Heizung', 'E-Heizung', 'Tiltrotator', 'Powertilt',
    'Schnellwechsler (mech.)', 'Schnellwechsler (hydr.)', 'Funkfernsteuerung',
    'ABB - Arbeitsbereichsbegrenzung', 'Monoausleger', 'Verstellausleger',

    # Numeric properties
    'Gewicht [kg]', 'Breite [mm]', 'Hoehe [mm]', 'Laenge [mm]',
    'Motor - Leistung [kW]', 'Arbeitsbreite [mm]', 'Grabtiefe [mm]',
    'Loeffelstiel [mm]', 'Fraesbreite [mm]', 'Fraestiefe [mm]',
    'Ausladung [m]', 'Ausleger [m]', 'Hakenhoehe [m]', 'Tragkraft max. [kg]',
    'Fahrgeschwindigkeit [km/h]', 'Nutzlast [kg]', 'Ballast [t]',

    # Text properties
    'Abgasstufe EU', 'Abgasstufe USA', 'Motor - Hersteller', 'Motor [Typ]',
    'Getriebe [Art]', 'Getriebe [Typ]', 'Farbe', 'Fuehrerscheinklasse'
]

# Backwards compatibility
BOOLEAN_FIELDS = ['klimaanlage', 'zentralschmierung']
NUMERIC_FIELDS = ['gewicht_kg', 'motor_leistung_kw', 'breite_mm', 'hoehe_mm', 'laenge_mm']
