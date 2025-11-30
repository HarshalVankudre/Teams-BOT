-- ============================================================
-- SEMA EQUIPMENT DATABASE - COMPLETE SCHEMA
-- PostgreSQL 14+
-- ============================================================

-- Drop existing tables
DROP TABLE IF EXISTS geraete CASCADE;

-- ============================================================
-- MAIN EQUIPMENT TABLE (2,395 records × 171 properties)
-- ============================================================

CREATE TABLE geraete (
    -- ========================================
    -- IDENTIFICATION
    -- ========================================
    id VARCHAR(100) PRIMARY KEY,
    primaerschluessel BIGINT,
    seriennummer VARCHAR(100),
    inventarnummer VARCHAR(100),
    
    -- ========================================
    -- CLASSIFICATION
    -- ========================================
    bezeichnung VARCHAR(255),
    hersteller VARCHAR(100),
    hersteller_code VARCHAR(20),
    geraetegruppe VARCHAR(100),
    geraetegruppe_code VARCHAR(20),
    verwendung VARCHAR(100),
    verwendung_code VARCHAR(20),
    kategorie VARCHAR(50),
    
    -- ========================================
    -- USE CASES & SEMANTIC CONTENT
    -- ========================================
    einsatzgebiete TEXT[],
    geeignet_fuer TEXT,
    gelaendetypen TEXT[],
    typische_aufgaben TEXT[],
    inhalt TEXT,  -- For semantic search context
    titel VARCHAR(255),
    
    -- ========================================
    -- NUMERIC PROPERTIES (32 fields)
    -- For range queries: >, <, BETWEEN
    -- ========================================
    anzahl_zaehne FLOAT,
    arbeitsbreite_mm FLOAT,
    arbeitsdruck_bar FLOAT,
    ausladung_m FLOAT,
    ausleger_m FLOAT,
    ballast_t FLOAT,
    bandbreite_mm FLOAT,
    bodenplatten_mm FLOAT,
    druck_bar FLOAT,
    durchsatzmenge_t_h FLOAT,
    einbaubreite_max__m FLOAT,
    einbaubreite_mit_verbreiterungen_m FLOAT,
    fahrgeschwindigkeit_km_h FLOAT,
    foerderkapazitaet_t_h FLOAT,
    fraesbreite_mm FLOAT,
    fraestiefe_mm FLOAT,
    gewicht_kg FLOAT,
    grabtiefe_mm FLOAT,
    hakenhoehe_m FLOAT,
    inhalt_m3 FLOAT,
    kantenschneidgeraet_stueck FLOAT,
    laenge_mm FLOAT,
    loeffelstiel_mm FLOAT,
    motor_leistung_kw FLOAT,
    muldenvolumen_m3 FLOAT,
    nutzlast_kg FLOAT,
    schnittbreite_mm FLOAT,
    steigfaehigkeit_mit_vibration__pct FLOAT,
    steigfaehigkeit_ohne_vibration__pct FLOAT,
    stuetzlast_kg FLOAT,
    tragkraft_max__kg FLOAT,
    zul__reisskraft_knm FLOAT,

    -- ========================================
    -- BOOLEAN PROPERTIES (61 fields)
    -- For feature queries: = true/false
    -- ========================================
    1_achser BOOLEAN,
    2_achser BOOLEAN,
    3_achser BOOLEAN,
    4_achser BOOLEAN,
    abb_arbeitsbereichsbegrenzung BOOLEAN,
    absauganlage BOOLEAN,
    allradantrieb BOOLEAN,
    allradlenkung BOOLEAN,
    anbauplattenverdichter BOOLEAN,
    asphaltmanager BOOLEAN,
    bio_hydraulikoel BOOLEAN,
    dieselmotor BOOLEAN,
    dieselpartikelfilter BOOLEAN,
    distanzkontrolle_automatisch BOOLEAN,
    e_heizung BOOLEAN,
    elektrostarter BOOLEAN,
    funkfernsteuerung BOOLEAN,
    gabelaufnahme_beschickerkuebel BOOLEAN,
    gas_heizung BOOLEAN,
    geteilte_bandage BOOLEAN,
    greiferdreheinrichtung BOOLEAN,
    greiferhydraulik BOOLEAN,
    hammerhydraulik BOOLEAN,
    hochfahrbare_kabine BOOLEAN,
    kabine BOOLEAN,
    klimaanlage BOOLEAN,
    knicklenkung BOOLEAN,
    monoausleger BOOLEAN,
    motor_benzin BOOLEAN,
    motor_diesel BOOLEAN,
    muldenerhoehung BOOLEAN,
    muldenheizung BOOLEAN,
    oszillation BOOLEAN,
    powertilt BOOLEAN,
    pratzenabstuetzung BOOLEAN,
    rampen_hydraulisch BOOLEAN,
    rampen_mechanisch BOOLEAN,
    reversierbar BOOLEAN,
    scherenhydraulik BOOLEAN,
    schildabstuetzung BOOLEAN,
    schnellgang BOOLEAN,
    schnellwechsler_hydr_ BOOLEAN,
    schnellwechsler_mech_ BOOLEAN,
    schwenkband BOOLEAN,
    seitenknickausleger BOOLEAN,
    splittstreuer BOOLEAN,
    teleskopausleger BOOLEAN,
    temperaturmessung_asphalt BOOLEAN,
    tiltrotator BOOLEAN,
    truck_assist BOOLEAN,
    verdichtungsmesser BOOLEAN,
    verstellausleger BOOLEAN,
    vm_38_schnittstelle BOOLEAN,
    vor_und_ruecklauf BOOLEAN,
    vorlauf BOOLEAN,
    vorruestung_2d_steuerung BOOLEAN,
    vorruestung_3d_steuerung BOOLEAN,
    vorruestung_navitronic BOOLEAN,
    vorruestung_voelkel BOOLEAN,
    wetterschutzdach BOOLEAN,
    zentralschmierung BOOLEAN,

    -- ========================================
    -- TEXT/CATEGORICAL PROPERTIES (78 fields)
    -- For exact match: = 'value'
    -- ========================================
    abgasstufe_eu TEXT,
    abgasstufe_usa TEXT,
    absauganlage_vcs TEXT,
    arbeitshoehe_m TEXT,
    aufgabe_mm TEXT,
    backenbrecher TEXT,
    batterie_typ TEXT,
    bohle_typ TEXT,
    brechkraft_t TEXT,
    breite_mm TEXT,
    co2_emissionen_g_km TEXT,
    dachprofilverstellung TEXT,
    dauerleistung_kva TEXT,
    drehbar TEXT,
    drehmulde TEXT,
    durchflussmenge_l_min TEXT,
    einbau_von_hgt_schotter TEXT,
    einbaubreite_grundbohle_m TEXT,
    einbaustaerke_mm TEXT,
    empf__baggerklasse_t TEXT,
    farbe TEXT,
    fcs_flexible_cutter_system TEXT,
    foerderhoehe_m TEXT,
    foerderlaenge_m TEXT,
    fraesmeissel_anzahl TEXT,
    frequenz_hz TEXT,
    frontschild_mm TEXT,
    fuehrerscheinklasse TEXT,
    gegengewicht_t TEXT,
    getriebe_art TEXT,
    getriebe_typ TEXT,
    hochdruckreiniger TEXT,
    hoehe_mm TEXT,
    hubhoehe_mm TEXT,
    klappschild TEXT,
    koernung_mm TEXT,
    kreiselbrecher TEXT,
    laufzeit_h TEXT,
    leistung_kva TEXT,
    leistungsaufnahme_kw TEXT,
    level_pro TEXT,
    mittelschar_mm TEXT,
    mobil_kette TEXT,
    mobil_rad TEXT,
    mobil_semi TEXT,
    motor_elektro TEXT,
    motor_hersteller TEXT,
    motor_typ TEXT,
    nennspannung_v TEXT,
    nennstrom_a TEXT,
    pat_schild_mm TEXT,
    plattformhoehe_mm TEXT,
    prallmuehle TEXT,
    reifengroesse TEXT,
    rotationsgeschwindigkeit_u_min TEXT,
    s_schild_mm TEXT,
    schaufelvolumen_m3 TEXT,
    schnellwechsler_henle TEXT,
    schnellwechsler_oilquick TEXT,
    schnellwechsler_typ TEXT,
    schnittlaenge_mm TEXT,
    schnitttiefe_mm TEXT,
    schuetthoehe_mm TEXT,
    schutzklasse_ip TEXT,
    starres_band TEXT,
    streben_stege TEXT,
    su_schild_mm TEXT,
    traegergeraet_typ TEXT,
    tragkraft_an_der_spitze_kg TEXT,
    turmsystem_typ TEXT,
    u_schild_mm TEXT,
    umweltplakette_de TEXT,
    verdichtungsleistung_kg TEXT,
    walzendrehvorrichtung TEXT,
    wechselhaltersystem_typ TEXT,
    wegmessesensoren_zylinder TEXT,
    winde_typ TEXT,
    zahntyp TEXT,

    -- ========================================
    -- FULL JSON (for flexibility)
    -- ========================================
    eigenschaften_json JSONB,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- INDEXES FOR FAST QUERIES
-- ============================================================

-- Classification (most common filters)
CREATE INDEX idx_kategorie ON geraete(kategorie);
CREATE INDEX idx_hersteller ON geraete(hersteller);
CREATE INDEX idx_geraetegruppe ON geraete(geraetegruppe);
CREATE INDEX idx_verwendung ON geraete(verwendung);

-- Numeric (for range queries)
CREATE INDEX idx_gewicht_kg ON geraete(gewicht_kg);
CREATE INDEX idx_motor_leistung_kw ON geraete(motor_leistung_kw);
CREATE INDEX idx_breite_mm ON geraete(breite_mm);
CREATE INDEX idx_hoehe_mm ON geraete(hoehe_mm);
CREATE INDEX idx_laenge_mm ON geraete(laenge_mm);
CREATE INDEX idx_grabtiefe_mm ON geraete(grabtiefe_mm);
CREATE INDEX idx_arbeitsbreite_mm ON geraete(arbeitsbreite_mm);

-- Boolean (for feature queries)
CREATE INDEX idx_klimaanlage ON geraete(klimaanlage);
CREATE INDEX idx_allradantrieb ON geraete(allradantrieb);
CREATE INDEX idx_allradlenkung ON geraete(allradlenkung);
CREATE INDEX idx_tiltrotator ON geraete(tiltrotator);
CREATE INDEX idx_zentralschmierung ON geraete(zentralschmierung);

-- Text (for categorical queries)
CREATE INDEX idx_motor_hersteller ON geraete(motor_hersteller);
CREATE INDEX idx_abgasstufe_eu ON geraete(abgasstufe_eu);
CREATE INDEX idx_schnellwechsler_typ ON geraete(schnellwechsler_typ);

-- Full text search on content
CREATE INDEX idx_inhalt_fts ON geraete USING GIN(to_tsvector('german', COALESCE(inhalt, '')));

-- Array search on einsatzgebiete
CREATE INDEX idx_einsatzgebiete ON geraete USING GIN(einsatzgebiete);

-- JSONB for flexible queries
CREATE INDEX idx_eigenschaften ON geraete USING GIN(eigenschaften_json);

-- ============================================================
-- USEFUL VIEWS
-- ============================================================

-- Bagger overview
CREATE VIEW v_bagger AS
SELECT 
    id, bezeichnung, hersteller, geraetegruppe,
    gewicht_kg, motor_leistung_kw, grabtiefe_mm,
    klimaanlage, tiltrotator, hammerhydraulik,
    motor_hersteller, abgasstufe_eu, schnellwechsler_typ
FROM geraete 
WHERE kategorie = 'bagger' OR geraetegruppe ILIKE '%bagger%';

-- Fertiger overview
CREATE VIEW v_fertiger AS
SELECT 
    id, bezeichnung, hersteller, geraetegruppe,
    gewicht_kg, motor_leistung_kw,
    einbaubreite_max__m, einbaubreite_grundbohle_m,
    bohle_typ, abgasstufe_eu
FROM geraete 
WHERE kategorie = 'fertiger' OR geraetegruppe ILIKE '%fertiger%';

-- Walzen overview
CREATE VIEW v_walzen AS
SELECT 
    id, bezeichnung, hersteller, geraetegruppe,
    gewicht_kg, motor_leistung_kw, arbeitsbreite_mm,
    oszillation, verdichtungsmesser, asphaltmanager,
    motor_hersteller, abgasstufe_eu
FROM geraete 
WHERE kategorie = 'verdichter' OR geraetegruppe ILIKE '%walze%';

-- Statistics view
CREATE VIEW v_statistics AS
SELECT 
    'total' as metric,
    COUNT(*)::TEXT as value
FROM geraete
UNION ALL
SELECT 
    'by_kategorie_' || COALESCE(kategorie, 'unknown'),
    COUNT(*)::TEXT
FROM geraete GROUP BY kategorie
UNION ALL
SELECT 
    'by_hersteller_' || COALESCE(hersteller, 'unknown'),
    COUNT(*)::TEXT
FROM geraete GROUP BY hersteller
HAVING COUNT(*) >= 10;

-- ============================================================
-- HELPER FUNCTION: Get counts by any column
-- ============================================================

CREATE OR REPLACE FUNCTION get_counts(column_name TEXT)
RETURNS TABLE(label TEXT, count BIGINT) AS $$
BEGIN
    RETURN QUERY EXECUTE format(
        'SELECT COALESCE(%I::TEXT, ''unbekannt'') as label, COUNT(*) as count 
         FROM geraete 
         GROUP BY %I 
         ORDER BY count DESC',
        column_name, column_name
    );
END;
$$ LANGUAGE plpgsql;

-- Usage: SELECT * FROM get_counts('kategorie');
-- Usage: SELECT * FROM get_counts('hersteller');
