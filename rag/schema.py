"""
PostgreSQL Database Schema for RÜKO Equipment Database

This is the SINGLE SOURCE OF TRUTH for the database schema.
All agents and services should import from here.

Table: geraete (Baumaschinen/Construction Equipment)
Total Records: ~2400

CRITICAL: All numeric and boolean values are stored ONLY in eigenschaften_json (JSONB)!
The direct columns exist but are EMPTY (NULL). Always use JSONB access patterns.
"""

# =============================================================================
# COMPLETE DATABASE SCHEMA
# =============================================================================

DATABASE_SCHEMA = """
================================================================================
DATENBANK-SCHEMA: Tabelle "geraete" (Baumaschinen-Inventar)
================================================================================

DATENSTRUKTUR (nach Migration v1.2):
- TEXT-Spalten (hersteller, geraetegruppe, etc.): Direkte Werte
- NUMERISCHE Spalten: Jetzt direkt verfügbar (gewicht_kg, motor_leistung_kw, etc.)
- BOOLEAN Spalten: Jetzt direkt verfügbar (klimaanlage, hammerhydraulik, etc.)
- eigenschaften_json (JSONB): Backup/Fallback, enthält auch 'nicht-vorhanden' Marker

================================================================================
IDENTIFIKATION (VARCHAR/TEXT)
================================================================================
- id: VARCHAR PRIMARY KEY (z.B. "sema_12535521")
- primaerschluessel: BIGINT - Primärschlüssel aus SEMA
- seriennummer: VARCHAR - Seriennummer des Geräts
- inventarnummer: VARCHAR - Interne Inventarnummer
- bezeichnung: VARCHAR - Modellname (z.B. "CAT 320", "HC 130i")
- titel: VARCHAR - Titel/Kurzbeschreibung
- inhalt: TEXT - Volltext-Beschreibung

================================================================================
KLASSIFIKATION (VARCHAR)
================================================================================
- hersteller: VARCHAR - 124 verschiedene Hersteller!
  Häufige: 'Caterpillar', 'Liebherr', 'Bomag', 'Vögele', 'Hamm', 'Wirtgen',
           'Kubota', 'Volvo', 'Hitachi', 'Komatsu', 'Dynapac', 'Ammann',
           'Wacker Neuson', 'Yanmar', 'Takeuchi', 'Hyundai', 'JCB'
- hersteller_code: VARCHAR - Herstellercode (z.B. "HAM", "CAT")

- geraetegruppe: VARCHAR - WICHTIGSTE SPALTE! 122 verschiedene Gruppen!
  Bagger: 'Mobilbagger', 'Kettenbagger', 'Minibagger (0,0 to - 4,4 to)',
          'Kompaktbagger (4,5 to - 10,9 to)'
  Walzen: 'Tandemwalze', 'Walzenzug (Glattmantel)', 'Walzenzug (Schaffuß)',
          'Gummiradwalze', 'Grabenwalze', 'Kombiwalze'
  Fertiger: 'Radfertiger', 'Kettenfertiger', 'Beschicker'
  Fräsen: 'Kaltfräse (Kette)', 'Kaltfräse (Rad)', 'Anbaufräse', 'Grabenfräse'
  Krane: 'Autokran', 'Telekran (Kette)', 'Telekran (Rad)', 'Miniraupenkran'
  Lader: 'Radlader', 'Teleskoplader (starr)', 'Kettendumper', 'Raddumper'
  Löffel: 'Tieflöffel', 'Grabenräumlöffel (starr)', 'Sieblöffel', etc.
- geraetegruppe_code: VARCHAR - Gerätegruppen-Code

- kategorie: VARCHAR - 8 Oberkategorien (kann NULL sein!)
  Werte: 'bagger', 'lader', 'verdichter', 'fertiger', 'fraese', 'kran',
         'einbauunterstuetzung', 'transportfahrzeug'
  ACHTUNG: Oft NULL - geraetegruppe ist zuverlässiger!

- verwendung: VARCHAR - Verwendungszweck
  Werte: 'Vermietung', 'Verkauf', 'Fuhrpark', 'Externes Gerät', 'keine'
  WICHTIG: NUR filtern wenn explizit angefragt!
- verwendung_code: VARCHAR

================================================================================
ARRAY SPALTEN
================================================================================
- einsatzgebiete: TEXT[] - Array mit Einsatzgebieten
- gelaendetypen: TEXT[] - Array mit Geländetypen
- typische_aufgaben: TEXT[] - Array mit typischen Aufgaben
- geeignet_fuer: TEXT - Geeignet für (Freitext)

================================================================================
NUMERISCHE SPALTEN (DOUBLE PRECISION) - DIREKT VERFÜGBAR!
================================================================================

WICHTIG: Nach Migration sind numerische Werte als DIREKTE SPALTEN verfügbar!
Zugriff: gewicht_kg > 15000 (direkt, kein JSONB nötig)
NULL-Werte: Wo JSONB 'nicht-vorhanden' hatte, ist die Spalte NULL

Abmessungen & Gewicht:
- gewicht_kg: Betriebsgewicht in kg
- breite_mm: Breite in mm
- hoehe_mm: Höhe in mm
- laenge_mm: Länge in mm

Motor & Leistung:
- motor_leistung_kw: Motorleistung in kW
- fahrgeschwindigkeit_km_h: Max. Geschwindigkeit

Bagger-spezifisch:
- grabtiefe_mm: Grabtiefe
- loeffelstiel_mm: Löffelstiel-Länge
- anzahl_zaehne: Anzahl Zähne

Walzen & Verdichter:
- arbeitsbreite_mm: Arbeitsbreite
- verdichtungsleistung_kg: Verdichtungsleistung
- steigfaehigkeit_mit_vibration__pct: Steigfähigkeit mit Vibration %
- steigfaehigkeit_ohne_vibration__pct: Steigfähigkeit ohne Vibration %

Fertiger:
- einbaubreite_max__m: Max. Einbaubreite
- einbaubreite_mit_verbreiterungen_m: Einbaubreite mit Verbreiterungen
- durchsatzmenge_t_h: Durchsatzmenge t/h
- foerderkapazitaet_t_h: Förderkapazität t/h
- bandbreite_mm: Bandbreite

Fräsen:
- fraesbreite_mm: Fräsbreite
- fraestiefe_mm: Frästiefe
- schnittbreite_mm: Schnittbreite

Krane:
- ausladung_m: Ausladung
- ausleger_m: Auslegerlänge
- hakenhoehe_m: Hakenhöhe
- tragkraft_max__kg: Max. Tragkraft
- ballast_t: Ballast in Tonnen

Sonstige:
- nutzlast_kg: Nutzlast
- stuetzlast_kg: Stützlast
- inhalt_m3: Inhalt in m³
- muldenvolumen_m3: Muldenvolumen
- arbeitsdruck_bar: Arbeitsdruck
- druck_bar: Druck
- zul__reisskraft_knm: Zul. Reißkraft
- bodenplatten_mm: Bodenplatten
- kantenschneidgeraet_stueck: Kantenschneidgerät Anzahl

================================================================================
BOOLEAN SPALTEN - DIREKT VERFÜGBAR!
================================================================================

WICHTIG: Nach Migration sind Boolean-Werte als DIREKTE SPALTEN verfügbar!
Zugriff: klimaanlage = true (direkt, kein JSONB nötig)
NULL-Werte: Wo JSONB 'nicht-vorhanden' hatte, ist die Spalte NULL

Antrieb & Fahrwerk:
- allradantrieb, allradlenkung, knicklenkung
- dieselmotor, motor_diesel, motor_benzin
- dieselpartikelfilter, elektrostarter

Kabine & Komfort:
- kabine, klimaanlage, hochfahrbare_kabine, wetterschutzdach

Hydraulik & Anbaugeräte:
- hammerhydraulik, greiferhydraulik, scherenhydraulik
- greiferdreheinrichtung, schnellwechsler_mech_, schnellwechsler_hydr_
- tiltrotator, powertilt, zentralschmierung, bio_hydraulikoel

Ausleger & Abstützung:
- monoausleger, verstellausleger, seitenknickausleger, teleskopausleger
- pratzenabstuetzung, schildabstuetzung

Walzen & Verdichter:
- oszillation, verdichtungsmesser, geteilte_bandage, anbauplattenverdichter

Fertiger:
- gas_heizung, e_heizung, temperaturmessung_asphalt
- truck_assist, schwenkband, splittstreuer

Fräsen:
- absauganlage, reversierbar

Steuerung & Elektronik:
- vorruestung_2d_steuerung, vorruestung_3d_steuerung
- vorruestung_navitronic, vorruestung_voelkel
- vm_38_schnittstelle, distanzkontrolle_automatisch
- asphaltmanager, funkfernsteuerung, abb_arbeitsbereichsbegrenzung

Krane & Transport:
- gabelaufnahme_beschickerkuebel
- rampen_hydraulisch, rampen_mechanisch
- muldenerhoehung, muldenheizung, schnellgang

================================================================================
TEXT SPALTEN (zusätzliche Eigenschaften)
================================================================================
- abgasstufe_eu: 'Stufe III', 'Stufe IV', 'Stufe V', etc.
- abgasstufe_usa: US-Abgasstufe
- motor_hersteller: 'Deutz', 'Cummins', 'Kubota', 'Perkins'
- motor_typ: Motortyp
- motor_elektro: Elektromotor-Info
- getriebe_art: Getriebeart
- getriebe_typ: Getriebetyp
- reifengroesse: Reifengröße
- farbe: Farbe
- batterie_typ: Batterietyp
- fuehrerscheinklasse: Führerscheinklasse
- bohle_typ: Bohlentyp
- winde_typ: Windentyp
- turmsystem_typ: Turmsystem-Typ
- schnellwechsler_typ: Schnellwechsler-Typ
- schnellwechsler_oilquick: OilQuick-Info
- schnellwechsler_henle: Henle-Info
- wechselhaltersystem_typ: Wechselhaltersystem
- zahntyp: Zahntyp
- empf__baggerklasse_t: Empfohlene Baggerklasse
- fraesmeissel_anzahl: Fräsmeißel Anzahl
- level_pro: Level Pro Info
- absauganlage_vcs: VCS Absauganlage
- fcs_flexible_cutter_system: FCS System
- walzendrehvorrichtung: Walzendrehvorrichtung
- einbaustaerke_mm: Einbaustärke
- einbaubreite_grundbohle_m: Grundbohle Einbaubreite
- frequenz_hz: Frequenz
- durchflussmenge_l_min: Durchflussmenge
- gegengewicht_t: Gegengewicht
- tragkraft_an_der_spitze_kg: Tragkraft Spitze
- co2_emissionen_g_km: CO2 Emissionen
- umweltplakette_de: Umweltplakette
- dachprofilverstellung: Dachprofilverstellung
- wegmessesensoren_zylinder: Wegmesssensoren
- einbau_von_hgt_schotter: HGT Schotter Einbau

================================================================================
JSONB SPALTE: eigenschaften_json
================================================================================
Enthält weitere Key-Value Paare die nicht als direkte Spalten existieren.
Zugriff: eigenschaften_json->>'feldname'

================================================================================
TIMESTAMPS
================================================================================
- created_at: TIMESTAMP - Erstellungsdatum
- updated_at: TIMESTAMP - Letzte Aktualisierung

================================================================================
SQL-BEISPIELE
================================================================================

-- Alle Bagger zählen (geraetegruppe verwenden!):
SELECT COUNT(*) FROM geraete WHERE geraetegruppe ILIKE '%bagger%'

-- Alle Mietmaschinen (KEIN LIMIT bei "alle" Anfragen!):
SELECT hersteller, bezeichnung, geraetegruppe
FROM geraete
WHERE verwendung = 'Vermietung'

-- Bagger über 15t mit Klimaanlage (DIREKTE Spalten!):
SELECT hersteller, bezeichnung, gewicht_kg
FROM geraete
WHERE geraetegruppe ILIKE '%bagger%'
  AND gewicht_kg > 15000
  AND klimaanlage = true
ORDER BY gewicht_kg DESC

-- Durchschnittsgewicht nach Gerätegruppe:
SELECT geraetegruppe,
       COUNT(*) as anzahl,
       ROUND(AVG(gewicht_kg)) as avg_gewicht
FROM geraete
WHERE geraetegruppe ILIKE '%bagger%'
  AND gewicht_kg IS NOT NULL
GROUP BY geraetegruppe
ORDER BY anzahl DESC

-- Vergleich Kettenbagger vs Mobilbagger:
SELECT geraetegruppe,
       COUNT(*) as anzahl,
       ROUND(AVG(gewicht_kg)) as avg_gewicht,
       MIN(gewicht_kg) as min_gewicht,
       MAX(gewicht_kg) as max_gewicht
FROM geraete
WHERE geraetegruppe IN ('Kettenbagger', 'Mobilbagger')
  AND gewicht_kg IS NOT NULL
GROUP BY geraetegruppe

-- Walzen mit Oszillation:
SELECT hersteller, bezeichnung, arbeitsbreite_mm
FROM geraete
WHERE geraetegruppe ILIKE '%walze%'
  AND oszillation = true

-- Fertiger mit Gasheizung:
SELECT hersteller, bezeichnung, einbaubreite_mit_verbreiterungen_m
FROM geraete
WHERE geraetegruppe ILIKE '%fertiger%'
  AND gas_heizung = true

================================================================================
WICHTIGE REGELN
================================================================================

1. Für Gerätetypen IMMER geraetegruppe verwenden, NICHT kategorie!
2. verwendung-Filter NUR wenn explizit angefragt!
3. KEIN LIMIT bei Auflistungs-/Zählabfragen!
4. BOOLEAN-SPALTEN DIREKT: klimaanlage = true
5. NUMERISCHE SPALTEN DIREKT: gewicht_kg > 15000
6. NULL-Werte prüfen: gewicht_kg IS NOT NULL (für Aggregationen)
7. TEXT-Spalten (hersteller, geraetegruppe, verwendung) direkt abfragen
"""

# =============================================================================
# SCHEMA FOR SQL AGENT
# =============================================================================

SQL_AGENT_SCHEMA = DATABASE_SCHEMA

# =============================================================================
# SCHEMA FOR ORCHESTRATOR (simplified)
# =============================================================================

ORCHESTRATOR_SCHEMA = """
DATENBANK-SCHEMA: Tabelle "geraete" (~2400 Baumaschinen)

ALLE SPALTEN DIREKT VERFÜGBAR:
- id: VARCHAR Primary Key
- hersteller: 124 verschiedene (Caterpillar, Liebherr, Bomag, etc.)
- geraetegruppe: WICHTIGSTE SPALTE! 122 verschiedene Gruppen
  Bagger: 'Mobilbagger', 'Kettenbagger', 'Minibagger (0,0 to - 4,4 to)'
  Walzen: 'Tandemwalze', 'Walzenzug', 'Gummiradwalze'
  Fertiger: 'Radfertiger', 'Kettenfertiger'
  Fräsen: 'Kaltfräse (Kette)', 'Kaltfräse (Rad)'
- kategorie: 8 Oberkategorien (oft NULL!)
- bezeichnung: Modellname
- verwendung: 'Vermietung', 'Verkauf', 'Fuhrpark', etc.

NUMERISCHE SPALTEN (direkt):
- gewicht_kg, motor_leistung_kw, grabtiefe_mm, arbeitsbreite_mm, etc.

BOOLEAN SPALTEN (direkt):
- klimaanlage, hammerhydraulik, tiltrotator, oszillation, etc.

BEISPIEL-ANFRAGEN:
- "Liebherr Maschinen" → hersteller = 'Liebherr'
- "Alle Bagger" → geraetegruppe ILIKE '%bagger%'
- "Mietmaschinen" → verwendung = 'Vermietung'
- "Bagger über 15t" → gewicht_kg > 15000
- "Mit Klimaanlage" → klimaanlage = true
"""

# =============================================================================
# VALUE LISTS (from actual database)
# =============================================================================

HERSTELLER_VALUES = [
    'ABG', 'Ahlmann', 'AL-KO', 'Ammann', 'Atlas', 'Atlas-Copco', 'AUGER TORQUE',
    'Baumgärtner', 'Bema', 'Bergmann', 'Boart Longyear', 'Bomag', 'Breining',
    'Brian James Trailers', 'Cardi', 'Caterpillar', 'Cedima', 'Compair', 'Containex',
    'CP', 'Demag', 'Ditch Witch', 'DMS', 'Dynapac', 'Egli', 'Endress', 'Engcon',
    'Epiroc', 'Ford', 'Format', 'Furukawa', 'GEKO', 'Genie', 'GRÜN GmbH', 'Grünig',
    'Hamm', 'HBM NOBAS', 'Henle', 'Heylo', 'HIMOINSA', 'Hitachi', 'HKS', 'Hufgard',
    'Hulco', 'Hydraulikgreifer-Technologie GmbH', 'Hydrema', 'Hyster', 'Hyundai',
    'JCB', 'John Deere', 'Jungheinrich', 'Kaeser', 'Kinshofer', 'Kleemann', 'Kobelco',
    'Komatsu', 'Korte', 'Kramer Allrad', 'Kränzle', 'Kroll', 'Krupp', 'KSG', 'Ksw',
    'Kubota', 'Kwanglim', 'Lehnhoff', 'Liebherr', 'MAEDA', 'MAN', 'Manitou', 'MBU',
    'McCloskey', 'Mercedes-Benz', 'Merlo', 'Moba', 'MTS', 'Müller Mitteltal', 'Neuson',
    'New Holland', 'Niftylift', 'Nilfisk Alto', 'NOZAR', 'Oilquick', 'O&K', 'Opel',
    'Paus', 'Potain', 'Rammax', 'Renault', 'Reschke', 'Rototilt', 'RÜKO', 'Schaeff',
    'Sennebogen', 'Sitech', 'Skoda', 'SMP', 'Sobernheimer', 'Sonstige', 'Steelwrist',
    'Stehr', 'Stihl', 'Straßmayr', 'Streumaster', 'Takeuchi', 'Tesla', 'Theis',
    'Thwaites', 'Tracto-Technik', 'Trimble', 'TS Industrie', 'Tuchel', 'Vögele',
    'Volkswagen', 'Volvo', 'Wacker Neuson', 'Weber mt', 'Weber Stahl', 'Weiro',
    'Wirtgen', 'Yanmar', 'Zeppelin', 'ZFE GmbH'
]

GERAETEGRUPPE_VALUES = [
    # Bagger
    'Mobilbagger', 'Kettenbagger', 'Minibagger (0,0 to - 4,4 to)',
    'Kompaktbagger (4,5 to - 10,9 to)',
    # Walzen
    'Tandemwalze', 'Walzenzug (Glattmantel)', 'Walzenzug (Schaffuß)',
    'Gummiradwalze', 'Grabenwalze', 'Kombiwalze',
    # Fertiger
    'Radfertiger', 'Kettenfertiger', 'Beschicker',
    # Fräsen
    'Kaltfräse (Kette)', 'Kaltfräse (Rad)', 'Anbaufräse', 'Grabenfräse', 'Anbaugrabenfräse',
    # Krane
    'Autokran', 'Telekran (Kette)', 'Telekran (Rad)', 'Miniraupenkran',
    'Obendreher-Kran', 'Untendreher-Kran',
    # Lader
    'Radlader', 'Teleskoplader (starr)', 'Kettendumper', 'Raddumper', 'Laderaupe',
    # Löffel & Anbaugeräte
    'Tieflöffel', 'Tieflöffel (hydraulisch schwenkbar)', 'Grabenräumlöffel (starr)',
    'Grabenräumlöffel (hydraulisch schwenkbar)', 'Sieblöffel', 'Brecherlöffel',
    'Spatenslöffel', 'Telelöffel',
    # Greifer & Zangen
    'Abbruch- und Sortiergreifer', 'Abbruchzange/Pulverisierer', 'Pendelgreifer',
    # Sonstige
    'Hydraulikhammer', 'Tiltrotator (Sandwich)', 'Vibrationsplatte', 'Vibrationsstampfer',
    'Stromerzeuger', 'Kompressor', 'Fugenschneider', 'Kernbohrgerät', 'Hochdruckreiniger',
    'Gabelstapler', 'Anhängerarbeitsbühne', 'Scherenarbeitsbühne (elektrisch)',
    'Gelenk-Teleskoparbeitsbühne', 'Siebanlage', 'Backenbrecher', 'Prallbrecher'
]

KATEGORIE_VALUES = [
    'bagger', 'lader', 'verdichter', 'fertiger', 'fraese', 'kran',
    'einbauunterstuetzung', 'transportfahrzeug'
]

VERWENDUNG_VALUES = [
    'Vermietung', 'Verkauf', 'Fuhrpark', 'Externes Gerät', 'keine'
]

# Boolean fields in eigenschaften_json (query via: eigenschaften_json->>'field' = 'true')
# Values are TEXT strings: 'true', 'false', 'nicht-vorhanden'
BOOLEAN_FIELDS = [
    'allradantrieb', 'allradlenkung', 'knicklenkung', 'dieselmotor', 'motor_diesel',
    'motor_benzin', 'dieselpartikelfilter', 'elektrostarter', 'kabine', 'klimaanlage',
    'hochfahrbare_kabine', 'wetterschutzdach', 'hammerhydraulik', 'greiferhydraulik',
    'scherenhydraulik', 'greiferdreheinrichtung', 'schnellwechsler_mech_',
    'schnellwechsler_hydr_', 'tiltrotator', 'powertilt', 'zentralschmierung',
    'bio_hydraulikoel', 'monoausleger', 'verstellausleger', 'seitenknickausleger',
    'teleskopausleger', 'pratzenabstuetzung', 'schildabstuetzung', 'oszillation',
    'verdichtungsmesser', 'geteilte_bandage', 'anbauplattenverdichter', 'gas_heizung',
    'e_heizung', 'temperaturmessung_asphalt', 'truck_assist', 'schwenkband',
    'splittstreuer', 'absauganlage', 'reversierbar', 'vorruestung_2d_steuerung',
    'vorruestung_3d_steuerung', 'vorruestung_navitronic', 'vorruestung_voelkel',
    'vm_38_schnittstelle', 'distanzkontrolle_automatisch', 'asphaltmanager',
    'funkfernsteuerung', 'abb_arbeitsbereichsbegrenzung', 'gabelaufnahme_beschickerkuebel',
    'rampen_hydraulisch', 'rampen_mechanisch', 'muldenerhoehung', 'muldenheizung',
    'schnellgang', 'vor_und_ruecklauf', 'vorlauf'
]

# Numeric fields in eigenschaften_json (query via: (eigenschaften_json->>'field')::numeric)
# Values are TEXT strings: '250.0', 'nicht-vorhanden', etc.
NUMERIC_FIELDS = [
    'gewicht_kg', 'motor_leistung_kw', 'laenge_mm', 'grabtiefe_mm', 'loeffelstiel_mm',
    'anzahl_zaehne', 'arbeitsbreite_mm', 'steigfaehigkeit_mit_vibration__pct',
    'steigfaehigkeit_ohne_vibration__pct', 'einbaubreite_max__m',
    'einbaubreite_mit_verbreiterungen_m', 'durchsatzmenge_t_h', 'foerderkapazitaet_t_h',
    'bandbreite_mm', 'fraesbreite_mm', 'fraestiefe_mm', 'schnittbreite_mm',
    'ausladung_m', 'ausleger_m', 'hakenhoehe_m', 'tragkraft_max__kg', 'ballast_t',
    'nutzlast_kg', 'stuetzlast_kg', 'inhalt_m3', 'muldenvolumen_m3', 'arbeitsdruck_bar',
    'druck_bar', 'zul__reisskraft_knm', 'bodenplatten_mm', 'kantenschneidgeraet_stueck',
    'fahrgeschwindigkeit_km_h'
]

# Backwards compatibility aliases (deprecated - use BOOLEAN_FIELDS/NUMERIC_FIELDS)
BOOLEAN_COLUMNS = BOOLEAN_FIELDS
NUMERIC_COLUMNS = NUMERIC_FIELDS
