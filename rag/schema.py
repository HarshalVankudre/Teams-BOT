"""
PostgreSQL Database Schema for RUKO Equipment Database

This is the SINGLE SOURCE OF TRUTH for the database schema.
All agents and services should import from here.

Table: geraete (Baumaschinen/Construction Equipment)
Total Records: ~2395

Schema updated 2024-12-07:
- id: BIGINT PRIMARY KEY (SEMA primaryKey)
- Direct columns for basic equipment info
- 171 prop_ columns for all technical properties (migrated from JSONB)
- eigenschaften: JSONB column (backwards compatible, synced from prop_* columns)
- German umlauts converted to ASCII (ae, oe, ue, ss)

NOTE: Both prop_* columns AND eigenschaften JSONB are available.
Prefer prop_* columns for direct queries, eigenschaften for flexibility.
"""

# =============================================================================
# COMPLETE DATABASE SCHEMA
# =============================================================================

DATABASE_SCHEMA = """
================================================================================
DATENBANK-SCHEMA: Tabelle "geraete" (Baumaschinen-Inventar)
================================================================================

Schema basiert auf SEMA Raw-Export.
Jedes Geraet hat eine eindeutige ID (SEMA primaryKey).

WICHTIG: Alle deutschen Umlaute wurden zu ASCII konvertiert:
  ae statt ae, oe statt oe, ue statt ue, ss statt ss

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
  Mit Code: 'VOeG - Voegele' (283), '??? - Sonstige' (65)
  EMPFEHLUNG: Immer ILIKE '%suchbegriff%' verwenden!
- hersteller_code: TEXT - z.B. 'CAT', 'LIE', 'BOM', 'VOeG'

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
LEGACY NUMERISCHE/BOOLEAN SPALTEN
================================================================================
Diese Spalten existieren noch fuer Rueckwaertskompatibilitaet:
- breite_mm, hoehe_mm, laenge_mm, gewicht_kg, motor_leistung_kw: NUMERIC
- klimaanlage, zentralschmierung: BOOLEAN

BEVORZUGE die neuen prop_ Spalten (siehe unten)!

================================================================================
PROPERTY SPALTEN (prop_*) - BEVORZUGT FUER ABFRAGEN!
================================================================================
Alle 171 technischen Eigenschaften sind als direkte TEXT-Spalten verfuegbar.
Werte enthalten die Einheit wenn vorhanden (z.B. "1400 mm", "620 kg").

DIMENSIONEN:
- prop_breite: Breite mit Einheit (z.B. "1400 mm", "2550 mm")
- prop_hoehe: Hoehe mit Einheit (z.B. "3100 mm", "1400 mm")
- prop_laenge: Laenge mit Einheit (z.B. "9200 mm", "1770 mm")
- prop_gewicht: Gewicht mit Einheit (z.B. "20000 kg", "620 kg")
- prop_arbeitsbreite: Arbeitsbreite (z.B. "1680 mm")
- prop_arbeitshoehe: Arbeitshoehe (z.B. "16 m")
- prop_grabtiefe: Grabtiefe (z.B. "5000 mm")
- prop_hubhoehe: Hubhoehe in mm
- prop_plattformhoehe: Plattformhoehe in mm
- prop_schnittbreite, prop_schnittlaenge, prop_schnitttiefe
- prop_fraesbreite, prop_fraestiefe: Fraesabmessungen
- prop_einbaubreite_max, prop_einbaubreite_grundbohle
- prop_einbaustaerke: Einbaustaerke

MOTOR & ANTRIEB:
- prop_motor_leistung: Motorleistung (z.B. "129 kW", "55,4 kW")
- prop_motor_hersteller: z.B. "Deutz", "Cummins"
- prop_motor: Motortyp
- prop_motor_diesel, prop_motor_elektro, prop_motor_benzin: Antriebsart (Ja/Nein)
- prop_getriebe, prop_getriebe_1: Getriebetyp
- prop_fahrgeschwindigkeit: Fahrgeschwindigkeit (z.B. "12 km/h")

AUSSTATTUNG (Boolean-Werte: "Ja" oder "Nein"):
- prop_klimaanlage: Klimaanlage vorhanden
- prop_zentralschmierung: Zentralschmierung vorhanden
- prop_oszillation: Oszillation
- prop_allradantrieb, prop_allradlenkung, prop_knicklenkung
- prop_dieselpartikelfilter
- prop_kabine, prop_hochfahrbare_kabine
- prop_gas_heizung, prop_e_heizung
- prop_funkfernsteuerung
- prop_schnellwechsler_hydr, prop_schnellwechsler_mech
- prop_tiltrotator, prop_powertilt
- prop_hammerhydraulik, prop_greiferhydraulik
- prop_verdichtungsmesser
- prop_bio_hydraulikoel

BAGGER-SPEZIFISCH:
- prop_loeffelstiel: Loeffelstiellänge
- prop_monoausleger, prop_verstellausleger, prop_teleskopausleger
- prop_greiferdreheinrichtung
- prop_abb_arbeitsbereichsbegrenzung

WALZEN-SPEZIFISCH:
- prop_verdichtungsleistung: Verdichtungsleistung in kg
- prop_geteilte_bandage
- prop_steigfaehigkeit_mit_vibration, prop_steigfaehigkeit_ohne_vibration

FERTIGER-SPEZIFISCH:
- prop_bohle: Bohlentyp (z.B. "AB 340-3 TV")
- prop_einbaubreite_mit_verbreiterungen
- prop_asphaltmanager
- prop_temperaturmessung_asphalt

KRAN-SPEZIFISCH:
- prop_ausladung: Ausladung (z.B. "22,0 m")
- prop_ausleger: Auslegerlänge (z.B. "45 m")
- prop_hakenhoehe: Hakenhoehe
- prop_tragkraft_max, prop_tragkraft_an_der_spitze
- prop_ballast, prop_gegengewicht

BRECHER & SIEBANLAGEN:
- prop_backenbrecher, prop_kreiselbrecher, prop_prallmuehle
- prop_koernung: Koernung
- prop_durchsatzmenge: Durchsatzmenge

STROMERZEUGER:
- prop_leistung, prop_dauerleistung: Leistung in kVA
- prop_nennspannung, prop_nennstrom
- prop_frequenz

ABGAS & UMWELT:
- prop_abgasstufe_eu: z.B. "Stufe IV", "Tier 4i"
- prop_abgasstufe_usa: z.B. "Tier 3", "Tier 4f"
- prop_umweltplakette_de
- prop_co2_emissionen

ACHSEN & CHASSIS:
- prop_1_achser, prop_2_achser, prop_3_achser, prop_4_achser
- prop_bodenplatten: Bodenplattenbreite
- prop_reifengroesse

SONSTIGE:
- prop_batterie: Batterietyp
- prop_farbe: Farbe
- prop_fuehrerscheinklasse
- prop_laufzeit: Betriebsstunden
- prop_nutzlast, prop_stuetzlast
- prop_schaufelvolumen, prop_muldenvolumen
- prop_inhalt: Inhalt in m3
- prop_druck, prop_arbeitsdruck
- prop_durchflussmenge
- prop_rotationsgeschwindigkeit

ALLE 171 PROPERTY-SPALTEN (vollstaendige Liste):
prop_1_achser, prop_2_achser, prop_3_achser, prop_4_achser,
prop_abb_arbeitsbereichsbegrenzung, prop_abgasstufe_eu, prop_abgasstufe_usa,
prop_absauganlage, prop_absauganlage_vcs, prop_allradantrieb, prop_allradlenkung,
prop_anbauplattenverdichter, prop_anzahl_zaehne, prop_arbeitsbreite,
prop_arbeitsdruck, prop_arbeitshoehe, prop_asphaltmanager, prop_aufgabe,
prop_ausladung, prop_ausleger, prop_backenbrecher, prop_ballast, prop_bandbreite,
prop_batterie, prop_bio_hydraulikoel, prop_bodenplatten, prop_bohle,
prop_brechkraft, prop_breite, prop_co2_emissionen, prop_dachprofilverstellung,
prop_dauerleistung, prop_dieselmotor, prop_dieselpartikelfilter,
prop_distanzkontrolle_automatisch, prop_drehbar, prop_drehmulde, prop_druck,
prop_durchflussmenge, prop_durchsatzmenge, prop_e_heizung,
prop_einbau_von_hgt_schotter, prop_einbaubreite_grundbohle, prop_einbaubreite_max,
prop_einbaubreite_mit_verbreiterungen, prop_einbaustaerke, prop_elektrostarter,
prop_empf_baggerklasse, prop_fahrgeschwindigkeit, prop_farbe,
prop_fcs_flexible_cutter_system, prop_foerderhoehe, prop_foerderkapazitaet,
prop_foerderlaenge, prop_fraesbreite, prop_fraesmeissel_anzahl, prop_fraestiefe,
prop_frequenz, prop_frontschild, prop_fuehrerscheinklasse, prop_funkfernsteuerung,
prop_gabelaufnahme_beschickerkuebel, prop_gas_heizung, prop_gegengewicht,
prop_geteilte_bandage, prop_getriebe, prop_getriebe_1, prop_gewicht,
prop_grabtiefe, prop_greiferdreheinrichtung, prop_greiferhydraulik,
prop_hakenhoehe, prop_hammerhydraulik, prop_hochdruckreiniger,
prop_hochfahrbare_kabine, prop_hoehe, prop_hubhoehe, prop_inhalt, prop_kabine,
prop_kantenschneidgeraet, prop_klappschild, prop_klimaanlage, prop_knicklenkung,
prop_koernung, prop_kreiselbrecher, prop_laenge, prop_laufzeit, prop_leistung,
prop_leistungsaufnahme, prop_level_pro, prop_loeffelstiel, prop_mittelschar,
prop_mobil_kette, prop_mobil_rad, prop_mobil_semi, prop_monoausleger,
prop_motor, prop_motor_benzin, prop_motor_diesel, prop_motor_elektro,
prop_motor_hersteller, prop_motor_leistung, prop_muldenerhoehung,
prop_muldenheizung, prop_muldenvolumen, prop_nennspannung, prop_nennstrom,
prop_nutzlast, prop_oszillation, prop_pat_schild, prop_plattformhoehe,
prop_powertilt, prop_prallmuehle, prop_pratzenabstuetzung, prop_rampen_hydraulisch,
prop_rampen_mechanisch, prop_reifengroesse, prop_reversierbar,
prop_rotationsgeschwindigkeit, prop_s_schild, prop_schaufelvolumen,
prop_scherenhydraulik, prop_schildabstuetzung, prop_schnellgang,
prop_schnellwechsler, prop_schnellwechsler_henle, prop_schnellwechsler_hydr,
prop_schnellwechsler_mech, prop_schnellwechsler_oilquick, prop_schnittbreite,
prop_schnittlaenge, prop_schnitttiefe, prop_schuetthoehe, prop_schutzklasse,
prop_schwenkband, prop_seitenknickausleger, prop_splittstreuer, prop_starres_band,
prop_steigfaehigkeit_mit_vibration, prop_steigfaehigkeit_ohne_vibration,
prop_streben_stege, prop_stuetzlast, prop_su_schild, prop_teleskopausleger,
prop_temperaturmessung_asphalt, prop_tiltrotator, prop_traegergeraet,
prop_tragkraft_an_der_spitze, prop_tragkraft_max, prop_truck_assist,
prop_turmsystem, prop_u_schild, prop_umweltplakette_de, prop_verdichtungsleistung,
prop_verdichtungsmesser, prop_verstellausleger, prop_vm_38_schnittstelle,
prop_vor_und_ruecklauf, prop_vorlauf, prop_vorruestung_2d_steuerung,
prop_vorruestung_3d_steuerung, prop_vorruestung_navitronic, prop_vorruestung_voelkel,
prop_walzendrehvorrichtung, prop_wechselhaltersystem, prop_wegmessesensoren_zylinder,
prop_wetterschutzdach, prop_winde, prop_zahntyp, prop_zentralschmierung,
prop_zul_reisskraft

================================================================================
SQL-BEISPIELE MIT NEUEN PROP_ SPALTEN
================================================================================

-- Alle Bagger zaehlen:
SELECT COUNT(*) FROM geraete WHERE geraetegruppe ILIKE '%bagger%'

-- Liebherr Maschinen mit Abmessungen:
SELECT bezeichnung, geraetegruppe, prop_breite, prop_hoehe, prop_gewicht
FROM geraete
WHERE hersteller = 'Liebherr'

-- Bagger mit Klimaanlage (direkte Spaltenabfrage):
SELECT bezeichnung, hersteller, prop_gewicht, prop_klimaanlage
FROM geraete
WHERE geraetegruppe ILIKE '%bagger%'
  AND prop_klimaanlage = 'Ja'

-- Walzen mit Oszillation:
SELECT bezeichnung, hersteller, prop_oszillation, prop_verdichtungsleistung
FROM geraete
WHERE geraetegruppe ILIKE '%walze%'
  AND prop_oszillation = 'Ja'

-- Fertiger mit bestimmter Einbaubreite:
SELECT bezeichnung, hersteller, prop_einbaubreite_max, prop_bohle
FROM geraete
WHERE geraetegruppe ILIKE '%fertiger%'
  AND prop_einbaubreite_max IS NOT NULL

-- Geraete mit Motorleistung (Text-Wert mit Einheit):
-- HINWEIS: Werte sind TEXT mit Einheit, z.B. "129 kW"
SELECT bezeichnung, prop_motor_leistung, prop_motor_hersteller
FROM geraete
WHERE prop_motor_leistung IS NOT NULL
  AND prop_motor_leistung != 'Ja'

-- Mietmaschinen mit Klimaanlage:
SELECT hersteller, bezeichnung, prop_klimaanlage
FROM geraete
WHERE verwendung = 'Vermietung'
  AND prop_klimaanlage = 'Ja'

-- Voegele Maschinen (Hersteller mit Code-Format):
SELECT COUNT(*) FROM geraete WHERE hersteller ILIKE '%voegele%' OR hersteller ILIKE '%VOeG%'

================================================================================
WICHTIGE REGELN
================================================================================
1. Fuer Geraetetypen IMMER geraetegruppe verwenden, NICHT kategorie!
2. verwendung-Filter NUR wenn explizit angefragt!
3. KEIN LIMIT bei Auflistungs-/Zaehlabfragen!
4. BEVORZUGE prop_ Spalten statt JSONB oder alte Spalten!
5. prop_ Werte sind TEXT mit Einheit (z.B. "1400 mm", "Ja", "Nein")
6. Boolean-Eigenschaften: prop_klimaanlage = 'Ja' (nicht true!)
7. NULL-Werte pruefen: prop_gewicht IS NOT NULL

================================================================================
ILIKE-REGEL (SEHR WICHTIG!)
================================================================================
Viele Spalten haben 'CODE - Name' Format. Bei Suche nach Code/Teilstring:
- FALSCH: WHERE kostenstelle = '200' (findet nichts!)
- RICHTIG: WHERE kostenstelle ILIKE '200%' (findet '200 - Mietpark')

Spalten mit 'CODE - Name' Format:
- kostenstelle: '100 - Handel', '200 - Mietpark', '90000 - Fuhrpark'
- abrechnungsgruppe: '4.3030.010 - Bohlenverbreiterungen...'
- hersteller (teilweise): 'VOeG - Voegele', '??? - Sonstige'

Bei Unsicherheit IMMER ILIKE '%suchbegriff%' verwenden!

================================================================================
TIMESTAMPS
================================================================================
- created_at: TIMESTAMP - Erstellungsdatum
"""

# =============================================================================
# SCHEMA FOR SQL AGENT
# =============================================================================

SQL_AGENT_SCHEMA = DATABASE_SCHEMA

# =============================================================================
# SCHEMA FOR ORCHESTRATOR (simplified)
# =============================================================================

ORCHESTRATOR_SCHEMA = """
DATENBANK-SCHEMA: Tabelle "geraete" (Baumaschinen)

DIREKTE SPALTEN:
- id: BIGINT Primary Key (SEMA primaryKey)
- bezeichnung: Modellname (z.B. "CAT 320", "BW 174 AP-5 AM")
- hersteller: z.B. 'Caterpillar', 'Liebherr', 'Bomag'
- geraetegruppe: WICHTIGSTE SPALTE! z.B. 'Mobilbagger', 'Tandemwalze'
- kategorie: Oberkategorie (oft NULL - geraetegruppe bevorzugen!)
- verwendung: 'Vermietung', 'Verkauf', 'Fuhrpark'
- seriennummer, inventarnummer

PROPERTY SPALTEN (prop_*) - 171 Spalten fuer alle Eigenschaften:
- prop_breite, prop_hoehe, prop_laenge, prop_gewicht (mit Einheit, z.B. "1400 mm")
- prop_motor_leistung (z.B. "129 kW")
- prop_klimaanlage, prop_oszillation (Werte: "Ja" oder "Nein")
- prop_abgasstufe_eu, prop_motor_hersteller (Text)
- ... und 160+ weitere prop_ Spalten

WICHTIG:
- prop_ Spalten sind TEXT mit Einheit (z.B. "1400 mm", "Ja")
- Boolean-Abfragen: prop_klimaanlage = 'Ja' (nicht true!)
- Bevorzuge prop_ Spalten statt JSONB!

BEISPIEL-ANFRAGEN:
- "Liebherr Maschinen" -> hersteller = 'Liebherr'
- "Alle Bagger" -> geraetegruppe ILIKE '%bagger%'
- "Mietmaschinen" -> verwendung = 'Vermietung'
- "Mit Klimaanlage" -> prop_klimaanlage = 'Ja'
- "Walzen mit Oszillation" -> prop_oszillation = 'Ja'
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
    'hersteller': ['VOeG - Voegele', '??? - Sonstige'],  # Partial - most are simple names
}

# SQL Agent rules for CODE-Name format (imported by sql_agent.py)
SQL_ILIKE_RULES = """
SEHR WICHTIG - 'CODE - Name' FORMAT:
Diese Spalten haben Format 'CODE - Name' (z.B. '200 - Mietpark'):
- kostenstelle: '100 - Handel', '200 - Mietpark', '90000 - Fuhrpark'
- abrechnungsgruppe: '4.3030.010 - Beschreibung'
- hersteller (teilweise): 'VOeG - Voegele'

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
- ae statt ae (z.B. 'Kaltfraese', 'Geraet')
- oe statt oe (z.B. 'Voegele', 'Loeffel')
- ue statt ue (z.B. 'Muell', 'Gruenig')
- ss statt ss (z.B. 'Strasse')

BEISPIELE:
- Voegele Maschinen: WHERE hersteller ILIKE '%voegele%' (283 Treffer)
- Fraesen: WHERE geraetegruppe ILIKE '%fraese%' (156 Treffer)
- Loeffel: WHERE geraetegruppe ILIKE '%loeffel%' (345 Treffer)
- Geraete: WHERE geraetegruppe ILIKE '%geraet%'
"""

# Combined rules for SQL agent (both CODE-Name and umlauts)
SQL_SPECIAL_RULES = SQL_ILIKE_RULES + SQL_UMLAUT_RULES

# =============================================================================
# PROPERTY COLUMNS LIST (all 171 prop_ columns)
# =============================================================================

PROPERTY_COLUMNS = [
    'prop_1_achser', 'prop_2_achser', 'prop_3_achser', 'prop_4_achser',
    'prop_abb_arbeitsbereichsbegrenzung', 'prop_abgasstufe_eu', 'prop_abgasstufe_usa',
    'prop_absauganlage', 'prop_absauganlage_vcs', 'prop_allradantrieb', 'prop_allradlenkung',
    'prop_anbauplattenverdichter', 'prop_anzahl_zaehne', 'prop_arbeitsbreite',
    'prop_arbeitsdruck', 'prop_arbeitshoehe', 'prop_asphaltmanager', 'prop_aufgabe',
    'prop_ausladung', 'prop_ausleger', 'prop_backenbrecher', 'prop_ballast', 'prop_bandbreite',
    'prop_batterie', 'prop_bio_hydraulikoel', 'prop_bodenplatten', 'prop_bohle',
    'prop_brechkraft', 'prop_breite', 'prop_co2_emissionen', 'prop_dachprofilverstellung',
    'prop_dauerleistung', 'prop_dieselmotor', 'prop_dieselpartikelfilter',
    'prop_distanzkontrolle_automatisch', 'prop_drehbar', 'prop_drehmulde', 'prop_druck',
    'prop_durchflussmenge', 'prop_durchsatzmenge', 'prop_e_heizung',
    'prop_einbau_von_hgt_schotter', 'prop_einbaubreite_grundbohle', 'prop_einbaubreite_max',
    'prop_einbaubreite_mit_verbreiterungen', 'prop_einbaustaerke', 'prop_elektrostarter',
    'prop_empf_baggerklasse', 'prop_fahrgeschwindigkeit', 'prop_farbe',
    'prop_fcs_flexible_cutter_system', 'prop_foerderhoehe', 'prop_foerderkapazitaet',
    'prop_foerderlaenge', 'prop_fraesbreite', 'prop_fraesmeissel_anzahl', 'prop_fraestiefe',
    'prop_frequenz', 'prop_frontschild', 'prop_fuehrerscheinklasse', 'prop_funkfernsteuerung',
    'prop_gabelaufnahme_beschickerkuebel', 'prop_gas_heizung', 'prop_gegengewicht',
    'prop_geteilte_bandage', 'prop_getriebe', 'prop_getriebe_1', 'prop_gewicht',
    'prop_grabtiefe', 'prop_greiferdreheinrichtung', 'prop_greiferhydraulik',
    'prop_hakenhoehe', 'prop_hammerhydraulik', 'prop_hochdruckreiniger',
    'prop_hochfahrbare_kabine', 'prop_hoehe', 'prop_hubhoehe', 'prop_inhalt', 'prop_kabine',
    'prop_kantenschneidgeraet', 'prop_klappschild', 'prop_klimaanlage', 'prop_knicklenkung',
    'prop_koernung', 'prop_kreiselbrecher', 'prop_laenge', 'prop_laufzeit', 'prop_leistung',
    'prop_leistungsaufnahme', 'prop_level_pro', 'prop_loeffelstiel', 'prop_mittelschar',
    'prop_mobil_kette', 'prop_mobil_rad', 'prop_mobil_semi', 'prop_monoausleger',
    'prop_motor', 'prop_motor_benzin', 'prop_motor_diesel', 'prop_motor_elektro',
    'prop_motor_hersteller', 'prop_motor_leistung', 'prop_muldenerhoehung',
    'prop_muldenheizung', 'prop_muldenvolumen', 'prop_nennspannung', 'prop_nennstrom',
    'prop_nutzlast', 'prop_oszillation', 'prop_pat_schild', 'prop_plattformhoehe',
    'prop_powertilt', 'prop_prallmuehle', 'prop_pratzenabstuetzung', 'prop_rampen_hydraulisch',
    'prop_rampen_mechanisch', 'prop_reifengroesse', 'prop_reversierbar',
    'prop_rotationsgeschwindigkeit', 'prop_s_schild', 'prop_schaufelvolumen',
    'prop_scherenhydraulik', 'prop_schildabstuetzung', 'prop_schnellgang',
    'prop_schnellwechsler', 'prop_schnellwechsler_henle', 'prop_schnellwechsler_hydr',
    'prop_schnellwechsler_mech', 'prop_schnellwechsler_oilquick', 'prop_schnittbreite',
    'prop_schnittlaenge', 'prop_schnitttiefe', 'prop_schuetthoehe', 'prop_schutzklasse',
    'prop_schwenkband', 'prop_seitenknickausleger', 'prop_splittstreuer', 'prop_starres_band',
    'prop_steigfaehigkeit_mit_vibration', 'prop_steigfaehigkeit_ohne_vibration',
    'prop_streben_stege', 'prop_stuetzlast', 'prop_su_schild', 'prop_teleskopausleger',
    'prop_temperaturmessung_asphalt', 'prop_tiltrotator', 'prop_traegergeraet',
    'prop_tragkraft_an_der_spitze', 'prop_tragkraft_max', 'prop_truck_assist',
    'prop_turmsystem', 'prop_u_schild', 'prop_umweltplakette_de', 'prop_verdichtungsleistung',
    'prop_verdichtungsmesser', 'prop_verstellausleger', 'prop_vm_38_schnittstelle',
    'prop_vor_und_ruecklauf', 'prop_vorlauf', 'prop_vorruestung_2d_steuerung',
    'prop_vorruestung_3d_steuerung', 'prop_vorruestung_navitronic', 'prop_vorruestung_voelkel',
    'prop_walzendrehvorrichtung', 'prop_wechselhaltersystem', 'prop_wegmessesensoren_zylinder',
    'prop_wetterschutzdach', 'prop_winde', 'prop_zahntyp', 'prop_zentralschmierung',
    'prop_zul_reisskraft'
]

# =============================================================================
# DEPRECATED: Legacy references (eigenschaften column has been removed)
# =============================================================================
# The following are kept for reference only. Use prop_* columns instead!
# Map: Old JSONB key -> New prop_ column
LEGACY_PROPERTY_MAPPING = {
    'Klimaanlage': 'prop_klimaanlage',
    'Zentralschmierung': 'prop_zentralschmierung',
    'Hammerhydraulik': 'prop_hammerhydraulik',
    'Oszillation': 'prop_oszillation',
    'Gewicht [kg]': 'prop_gewicht',
    'Breite [mm]': 'prop_breite',
    'Hoehe [mm]': 'prop_hoehe',
    'Motor - Leistung [kW]': 'prop_motor_leistung',
    'Arbeitsbreite [mm]': 'prop_arbeitsbreite',
    'Abgasstufe EU': 'prop_abgasstufe_eu',
    'Motor - Hersteller': 'prop_motor_hersteller',
}

# Base columns (not from JSONB, still in use)
BOOLEAN_FIELDS = ['klimaanlage', 'zentralschmierung']
NUMERIC_FIELDS = ['gewicht_kg', 'motor_leistung_kw']
