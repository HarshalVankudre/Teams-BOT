
-- =====================================================
-- SEMA Equipment Database Schema
-- Generated for PostgreSQL
-- =====================================================

-- Drop tables if they exist (in reverse order of dependencies)
DROP TABLE IF EXISTS equipment_properties CASCADE;
DROP TABLE IF EXISTS equipment CASCADE;
DROP TABLE IF EXISTS property_types CASCADE;
DROP TABLE IF EXISTS units CASCADE;
DROP TABLE IF EXISTS manufacturers CASCADE;
DROP TABLE IF EXISTS equipment_groups CASCADE;
DROP TABLE IF EXISTS billing_groups CASCADE;
DROP TABLE IF EXISTS usage_types CASCADE;
DROP TABLE IF EXISTS cost_centers CASCADE;
DROP TABLE IF EXISTS statuses CASCADE;
DROP TABLE IF EXISTS processes CASCADE;

-- =====================================================
-- LOOKUP TABLES
-- =====================================================

CREATE TABLE manufacturers (
    manufacturer_id SERIAL PRIMARY KEY,
    code VARCHAR(20),
    name VARCHAR(255),
    full_name VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE equipment_groups (
    equipment_group_id SERIAL PRIMARY KEY,
    code VARCHAR(50),
    name VARCHAR(255),
    full_name VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE billing_groups (
    billing_group_id SERIAL PRIMARY KEY,
    code VARCHAR(50),
    name VARCHAR(255),
    full_name VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE usage_types (
    usage_type_id SERIAL PRIMARY KEY,
    code VARCHAR(20),
    name VARCHAR(100),
    full_name VARCHAR(100) UNIQUE NOT NULL
);

CREATE TABLE cost_centers (
    cost_center_id SERIAL PRIMARY KEY,
    code VARCHAR(20),
    name VARCHAR(100),
    full_name VARCHAR(100) UNIQUE NOT NULL
);

CREATE TABLE statuses (
    status_id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE processes (
    process_id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE property_types (
    property_type_id SERIAL PRIMARY KEY,
    code VARCHAR(20),
    name VARCHAR(255),
    full_name VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE units (
    unit_id SERIAL PRIMARY KEY,
    code VARCHAR(20),
    name VARCHAR(100),
    full_name VARCHAR(100) UNIQUE NOT NULL
);

-- =====================================================
-- MAIN TABLES
-- =====================================================

CREATE TABLE equipment (
    equipment_id BIGINT PRIMARY KEY,
    designation VARCHAR(255) NOT NULL,
    serial_number VARCHAR(100),
    inventory_number VARCHAR(100),
    manufacturer_id INTEGER REFERENCES manufacturers(manufacturer_id),
    equipment_group_id INTEGER REFERENCES equipment_groups(equipment_group_id),
    billing_group_id INTEGER REFERENCES billing_groups(billing_group_id),
    usage_type_id INTEGER REFERENCES usage_types(usage_type_id),
    cost_center_id INTEGER REFERENCES cost_centers(cost_center_id),
    status_id INTEGER REFERENCES statuses(status_id),
    process_id INTEGER REFERENCES processes(process_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE equipment_properties (
    property_id BIGINT PRIMARY KEY,
    equipment_serial_number VARCHAR(100),
    equipment_inventory_number VARCHAR(100),
    equipment_designation VARCHAR(255),
    property_type_id INTEGER REFERENCES property_types(property_type_id),
    value VARCHAR(500),
    unit_id INTEGER REFERENCES units(unit_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

CREATE INDEX idx_equipment_manufacturer ON equipment(manufacturer_id);
CREATE INDEX idx_equipment_group ON equipment(equipment_group_id);
CREATE INDEX idx_equipment_status ON equipment(status_id);
CREATE INDEX idx_equipment_usage ON equipment(usage_type_id);
CREATE INDEX idx_equipment_serial ON equipment(serial_number);
CREATE INDEX idx_equipment_inventory ON equipment(inventory_number);
CREATE INDEX idx_equipment_designation ON equipment(designation);

CREATE INDEX idx_properties_serial ON equipment_properties(equipment_serial_number);
CREATE INDEX idx_properties_inventory ON equipment_properties(equipment_inventory_number);
CREATE INDEX idx_properties_type ON equipment_properties(property_type_id);

-- =====================================================
-- USEFUL VIEWS
-- =====================================================

CREATE OR REPLACE VIEW v_equipment_full AS
SELECT 
    e.equipment_id,
    e.designation,
    e.serial_number,
    e.inventory_number,
    m.full_name AS manufacturer,
    eg.full_name AS equipment_group,
    bg.full_name AS billing_group,
    ut.full_name AS usage_type,
    cc.full_name AS cost_center,
    s.name AS status,
    p.name AS process
FROM equipment e
LEFT JOIN manufacturers m ON e.manufacturer_id = m.manufacturer_id
LEFT JOIN equipment_groups eg ON e.equipment_group_id = eg.equipment_group_id
LEFT JOIN billing_groups bg ON e.billing_group_id = bg.billing_group_id
LEFT JOIN usage_types ut ON e.usage_type_id = ut.usage_type_id
LEFT JOIN cost_centers cc ON e.cost_center_id = cc.cost_center_id
LEFT JOIN statuses s ON e.status_id = s.status_id
LEFT JOIN processes p ON e.process_id = p.process_id;

CREATE OR REPLACE VIEW v_equipment_properties_full AS
SELECT 
    ep.property_id,
    ep.equipment_designation,
    ep.equipment_serial_number,
    ep.equipment_inventory_number,
    pt.full_name AS property_type,
    ep.value,
    u.full_name AS unit
FROM equipment_properties ep
LEFT JOIN property_types pt ON ep.property_type_id = pt.property_type_id
LEFT JOIN units u ON ep.unit_id = u.unit_id;
