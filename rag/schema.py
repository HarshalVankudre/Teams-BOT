"""
PostgreSQL Database Schema for RÜKO Equipment Database

This is the SINGLE SOURCE OF TRUTH for the database schema.
All agents and services should import from here.

Table: geraete (Baumaschinen/Construction Equipment)
Total Records: ~2400

IMPORTANT: Many properties are now DIRECT COLUMNS, not just in eigenschaften_json!
"""

# =============================================================================
# COMPLETE DATABASE SCHEMA
# =============================================================================

DATABASE_SCHEMA = """
================================================================================
DATENBANK-SCHEMA: Tabelle "geraete" (Baumaschinen-Inventar)
================================================================================

WICHTIG: Die Tabelle hat sowohl direkte Spalten ALS AUCH eine JSONB-Spalte!
Viele numerische/boolean Werte existieren als DIREKTE SPALTEN.

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
NUMERISCHE SPALTEN (DOUBLE PRECISION) - DIREKTE SPALTEN!
================================================================================

Abmessungen & Gewicht:
- gewicht_kg: Betriebsgewicht in kg
- breite_mm: Breite in mm (TEXT!)
- hoehe_mm: Höhe in mm (TEXT!)
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
- verdichtungsleistung_kg: Verdichtungsleistung (TEXT!)
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
BOOLEAN SPALTEN - DIREKTE SPALTEN!
================================================================================

Antrieb & Fahrwerk:
- allradantrieb: Allradantrieb
- allradlenkung: Allradlenkung
- knicklenkung: Knicklenkung
- dieselmotor: Dieselmotor
- motor_diesel: Diesel (redundant)
- motor_benzin: Benzinmotor
- dieselpartikelfilter: Dieselpartikelfilter
- elektrostarter: Elektrostarter

Kabine & Komfort:
- kabine: Kabine vorhanden
- klimaanlage: Klimaanlage
- hochfahrbare_kabine: Hochfahrbare Kabine
- wetterschutzdach: Wetterschutzdach

Hydraulik & Anbaugeräte:
- hammerhydraulik: Hammerhydraulik
- greiferhydraulik: Greiferhydraulik
- scherenhydraulik: Scherenhydraulik
- greiferdreheinrichtung: Greiferdreheinrichtung
- schnellwechsler_mech_: Mechanischer Schnellwechsler
- schnellwechsler_hydr_: Hydraulischer Schnellwechsler
- tiltrotator: Tiltrotator
- powertilt: PowerTilt
- zentralschmierung: Zentralschmierung
- bio_hydraulikoel: Bio-Hydrauliköl

Ausleger & Abstützung:
- monoausleger: Monoausleger
- verstellausleger: Verstellausleger
- seitenknickausleger: Seitenknickausleger
- teleskopausleger: Teleskopausleger
- pratzenabstuetzung: Pratzenabstützung
- schildabstuetzung: Schildabstützung

Walzen & Verdichter:
- oszillation: Oszillation
- verdichtungsmesser: Verdichtungsmesser
- geteilte_bandage: Geteilte Bandage
- anbauplattenverdichter: Anbauplattenverdichter

Fertiger:
- gas_heizung: Gasheizung
- e_heizung: Elektrische Heizung
- temperaturmessung_asphalt: Temperaturmessung Asphalt
- truck_assist: Truck Assist
- schwenkband: Schwenkband
- splittstreuer: Splittstreuer

Fräsen:
- absauganlage: Absauganlage
- reversierbar: Reversierbar

Steuerung & Elektronik:
- vorruestung_2d_steuerung: 2D Steuerung vorbereitet
- vorruestung_3d_steuerung: 3D Steuerung vorbereitet
- vorruestung_navitronic: Navitronic vorbereitet
- vorruestung_voelkel: Völkel vorbereitet
- vm_38_schnittstelle: VM 38 Schnittstelle
- distanzkontrolle_automatisch: Automatische Distanzkontrolle
- asphaltmanager: Asphaltmanager
- funkfernsteuerung: Funkfernsteuerung
- abb_arbeitsbereichsbegrenzung: Arbeitsbereichsbegrenzung

Krane & Hebezeuge:
- gabelaufnahme_beschickerkuebel: Gabelaufnahme Beschickerküebel

Transport:
- 1_achser, 2_achser, 3_achser, 4_achser: Achsanzahl
- rampen_hydraulisch: Hydraulische Rampen
- rampen_mechanisch: Mechanische Rampen
- muldenerhoehung: Muldenerhöhung
- muldenheizung: Muldenheizung

Sonstiges:
- schnellgang: Schnellgang
- vor_und_ruecklauf: Vor- und Rücklauf
- vorlauf: Vorlauf

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

-- Bagger über 15t mit Klimaanlage (direkte Spalten!):
SELECT hersteller, bezeichnung, gewicht_kg
FROM geraete
WHERE geraetegruppe ILIKE '%bagger%'
AND gewicht_kg > 15000
AND klimaanlage = true
ORDER BY gewicht_kg DESC

-- Durchschnittsgewicht nach Gerätegruppe:
SELECT geraetegruppe, COUNT(*) as anzahl, ROUND(AVG(gewicht_kg)) as avg_gewicht
FROM geraete
WHERE geraetegruppe ILIKE '%bagger%' AND gewicht_kg IS NOT NULL
GROUP BY geraetegruppe
ORDER BY anzahl DESC

-- Walzen mit Oszillation:
SELECT hersteller, bezeichnung, arbeitsbreite_mm
FROM geraete
WHERE geraetegruppe ILIKE '%walze%' AND oszillation = true

-- Fertiger mit Gasheizung:
SELECT hersteller, bezeichnung, einbaubreite_max__m
FROM geraete
WHERE geraetegruppe ILIKE '%fertiger%' AND gas_heizung = true

================================================================================
WICHTIGE REGELN
================================================================================

1. Für Gerätetypen IMMER geraetegruppe verwenden, NICHT kategorie!
2. verwendung-Filter NUR wenn explizit angefragt!
3. KEIN LIMIT bei Auflistungs-/Zählabfragen!
4. Boolean-Spalten direkt abfragen: klimaanlage = true
5. Numerische Spalten direkt abfragen: gewicht_kg > 15000
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

HAUPTSPALTEN:
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

DIREKTE NUMERISCHE SPALTEN:
- gewicht_kg, motor_leistung_kw, grabtiefe_mm, arbeitsbreite_mm, etc.

DIREKTE BOOLEAN SPALTEN:
- klimaanlage, schnellwechsler_hydr_, hammerhydraulik, tiltrotator, gps, etc.

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

# Boolean columns that can be queried directly
BOOLEAN_COLUMNS = [
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

# Numeric columns that can be queried directly
NUMERIC_COLUMNS = [
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
