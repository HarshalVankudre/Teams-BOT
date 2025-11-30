#!/usr/bin/env python3
"""
SEMA Equipment Database - Data Import Script
Imports JSONL data into PostgreSQL

Usage:
    python import_to_postgres.py --jsonl sema_komplett.jsonl
    
Environment variables:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""

import json
import psycopg2
from psycopg2.extras import execute_batch, Json
import os
import sys
import re
from typing import Any, Optional

# ============================================================
# CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "sema"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

# ============================================================
# FIELD NAME SANITIZATION
# ============================================================

def sanitize_field_name(field: str) -> str:
    """Convert field name to valid PostgreSQL column name"""
    sql_field = field
    sql_field = sql_field.replace('.', '_')
    sql_field = sql_field.replace('%', '_pct')
    sql_field = sql_field.replace('?', '')
    sql_field = sql_field.replace('³', '3')
    sql_field = sql_field.replace('ä', 'ae')
    sql_field = sql_field.replace('ö', 'oe')
    sql_field = sql_field.replace('ü', 'ue')
    sql_field = sql_field.replace('ß', 'ss')
    # Truncate to PostgreSQL limit
    if len(sql_field) > 63:
        sql_field = sql_field[:63]
    return sql_field


# ============================================================
# VALUE CLEANING
# ============================================================

def clean_value(value: Any) -> Any:
    """Convert 'nicht-vorhanden' to NULL, handle other types"""
    
    if value == "nicht-vorhanden":
        return None
    
    if value is None:
        return None
    
    # Handle dict (range values) - convert to JSON string
    if isinstance(value, dict):
        return json.dumps(value)
    
    return value


# ============================================================
# IMPORT FUNCTION
# ============================================================

def import_data(jsonl_path: str, batch_size: int = 100):
    """Import JSONL data into PostgreSQL"""
    
    print(f"📂 Loading data from: {jsonl_path}")
    
    # Load all records
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    
    print(f"   Found {len(records)} records")
    
    # Get all unique property fields from first record
    if not records:
        print("❌ No records found!")
        return
    
    sample = records[0]
    eigenschaften_fields = list(sample.get('eigenschaften', {}).keys())
    
    # Build column list
    base_columns = [
        "id", "primaerschluessel", "seriennummer", "inventarnummer",
        "bezeichnung", "hersteller", "hersteller_code",
        "geraetegruppe", "geraetegruppe_code",
        "verwendung", "verwendung_code", "kategorie",
        "einsatzgebiete", "geeignet_fuer", "gelaendetypen",
        "typische_aufgaben", "inhalt", "titel", "eigenschaften_json"
    ]
    
    # Map original field names to sanitized SQL column names
    field_mapping = {f: sanitize_field_name(f) for f in eigenschaften_fields}
    
    all_columns = base_columns + list(field_mapping.values())
    
    # Build INSERT query
    columns_str = ", ".join(all_columns)
    placeholders = ", ".join(["%s"] * len(all_columns))
    
    insert_sql = f"""
        INSERT INTO geraete ({columns_str})
        VALUES ({placeholders})
        ON CONFLICT (id) DO UPDATE SET
            updated_at = CURRENT_TIMESTAMP,
            eigenschaften_json = EXCLUDED.eigenschaften_json
    """
    
    # Connect to database
    print(f"🔌 Connecting to PostgreSQL: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Prepare batches
    batch = []
    total_imported = 0
    
    for record in records:
        # Extract base fields
        row = [
            record.get('id'),
            record.get('primaerschluessel'),
            clean_value(record.get('seriennummer')),
            clean_value(record.get('inventarnummer')),
            clean_value(record.get('bezeichnung')),
            clean_value(record.get('hersteller')),
            clean_value(record.get('hersteller_code')),
            clean_value(record.get('geraetegruppe')),
            clean_value(record.get('geraetegruppe_code')),
            clean_value(record.get('verwendung')),
            clean_value(record.get('verwendung_code')),
            clean_value(record.get('kategorie')),
            record.get('einsatzgebiete', []),  # Array
            clean_value(record.get('geeignet_fuer')),
            record.get('gelaendetypen', []),  # Array
            record.get('typische_aufgaben', []),  # Array
            record.get('inhalt'),
            record.get('titel'),
            Json(record.get('eigenschaften', {})),  # JSONB
        ]
        
        # Add all property fields in the same order
        eigenschaften = record.get('eigenschaften', {})
        for orig_field in eigenschaften_fields:
            value = eigenschaften.get(orig_field)
            row.append(clean_value(value))
        
        batch.append(tuple(row))
        
        # Execute batch
        if len(batch) >= batch_size:
            execute_batch(cursor, insert_sql, batch)
            conn.commit()
            total_imported += len(batch)
            print(f"   Imported {total_imported} records...")
            batch = []
    
    # Import remaining
    if batch:
        execute_batch(cursor, insert_sql, batch)
        conn.commit()
        total_imported += len(batch)
    
    print(f"✅ Imported {total_imported} records")
    
    # Show statistics
    print("\n📊 Database Statistics:")
    
    cursor.execute("SELECT COUNT(*) FROM geraete")
    total = cursor.fetchone()[0]
    print(f"   Total records: {total}")
    
    cursor.execute("""
        SELECT kategorie, COUNT(*) 
        FROM geraete 
        WHERE kategorie IS NOT NULL 
        GROUP BY kategorie 
        ORDER BY COUNT(*) DESC
    """)
    print("\n   By category:")
    for row in cursor.fetchall():
        print(f"      {row[0]}: {row[1]}")
    
    cursor.execute("""
        SELECT hersteller, COUNT(*) 
        FROM geraete 
        WHERE hersteller IS NOT NULL 
        GROUP BY hersteller 
        ORDER BY COUNT(*) DESC 
        LIMIT 10
    """)
    print("\n   Top 10 manufacturers:")
    for row in cursor.fetchall():
        print(f"      {row[0]}: {row[1]}")
    
    cursor.close()
    conn.close()
    
    print("\n✅ Import complete!")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Import SEMA data into PostgreSQL")
    parser.add_argument("--jsonl", required=True, help="Path to JSONL file")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.jsonl):
        print(f"❌ File not found: {args.jsonl}")
        sys.exit(1)
    
    import_data(args.jsonl, args.batch_size)
