"""
Database Migration: Populate direct columns from eigenschaften_json

This script copies data from eigenschaften_json (JSONB) to the direct columns
that currently exist but are empty.

BEFORE RUNNING:
- Backup created: db_backup_pre_migration.json
- Git tag: v1.1-jsonb-schema-fix

TO REVERT:
- Run: python scripts/revert_migration.py
- Or manually: UPDATE geraete SET gewicht_kg = NULL, klimaanlage = NULL, ...
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.postgres import postgres_service
from rag.schema import NUMERIC_FIELDS, BOOLEAN_FIELDS


def migrate_numeric_columns():
    """Migrate numeric fields from JSONB to direct columns"""
    print("\n=== Migrating NUMERIC columns ===")

    for field in NUMERIC_FIELDS:
        # First, handle simple numeric values (most common)
        sql_simple = f"""
        UPDATE geraete
        SET {field} = (eigenschaften_json->>'{field}')::numeric
        WHERE eigenschaften_json->>'{field}' IS NOT NULL
          AND eigenschaften_json->>'{field}' != 'nicht-vorhanden'
          AND eigenschaften_json->>'{field}' != ''
          AND eigenschaften_json->>'{field}' NOT LIKE '{{%'
          AND {field} IS NULL
        """

        # Second, handle nested JSON objects (extract 'durchschnitt' average)
        sql_nested = f"""
        UPDATE geraete
        SET {field} = ((eigenschaften_json->'{field}'->>'durchschnitt')::numeric)
        WHERE eigenschaften_json->>'{field}' LIKE '{{%'
          AND eigenschaften_json->'{field}'->>'durchschnitt' IS NOT NULL
          AND {field} IS NULL
        """

        try:
            conn = postgres_service._get_connection()
            cursor = conn.cursor()

            # Execute simple values first
            cursor.execute(sql_simple)
            rows_simple = cursor.rowcount

            # Execute nested JSON values
            cursor.execute(sql_nested)
            rows_nested = cursor.rowcount

            conn.commit()
            cursor.close()
            conn.close()

            total = rows_simple + rows_nested
            if total > 0:
                msg = f"  {field}: {total} rows updated"
                if rows_nested > 0:
                    msg += f" ({rows_nested} from nested JSON)"
                print(msg)
            else:
                print(f"  {field}: no valid data to migrate")

        except Exception as e:
            print(f"  {field}: ERROR - {e}")


def migrate_boolean_columns():
    """Migrate boolean fields from JSONB to direct columns"""
    print("\n=== Migrating BOOLEAN columns ===")

    for field in BOOLEAN_FIELDS:
        # Build UPDATE query that:
        # 1. Sets TRUE where JSONB value is 'true'
        # 2. Sets FALSE where JSONB value is 'false'
        # 3. Leaves NULL where 'nicht-vorhanden' or missing
        sql = f"""
        UPDATE geraete
        SET {field} = CASE
            WHEN eigenschaften_json->>'{field}' = 'true' THEN TRUE
            WHEN eigenschaften_json->>'{field}' = 'false' THEN FALSE
            ELSE NULL
        END
        WHERE eigenschaften_json->>'{field}' IN ('true', 'false')
          AND {field} IS NULL
        """

        try:
            conn = postgres_service._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql)
            rows_updated = cursor.rowcount
            conn.commit()
            cursor.close()
            conn.close()

            if rows_updated > 0:
                print(f"  {field}: {rows_updated} rows updated")
            else:
                print(f"  {field}: no valid data to migrate")

        except Exception as e:
            print(f"  {field}: ERROR - {e}")


def verify_migration():
    """Verify the migration was successful"""
    print("\n=== Verification ===")

    # Check a few key columns
    sql = """
    SELECT
        COUNT(*) as total,
        COUNT(gewicht_kg) as gewicht_populated,
        COUNT(klimaanlage) as klimaanlage_populated,
        COUNT(motor_leistung_kw) as motor_populated
    FROM geraete
    """

    result = postgres_service.execute_query(sql)
    if result:
        r = result[0]
        print(f"  Total rows: {r['total']}")
        print(f"  gewicht_kg populated: {r['gewicht_populated']}")
        print(f"  klimaanlage populated: {r['klimaanlage_populated']}")
        print(f"  motor_leistung_kw populated: {r['motor_populated']}")

    # Sample comparison
    sql = """
    SELECT hersteller, bezeichnung,
           gewicht_kg as direct,
           eigenschaften_json->>'gewicht_kg' as jsonb
    FROM geraete
    WHERE gewicht_kg IS NOT NULL
    LIMIT 3
    """
    result = postgres_service.execute_query(sql)
    print("\n  Sample data (direct vs JSONB):")
    for r in result:
        print(f"    {r['hersteller']} {r['bezeichnung']}: direct={r['direct']}, jsonb={r['jsonb']}")


def main():
    print("=" * 60)
    print("DATABASE MIGRATION: JSONB -> Direct Columns")
    print("=" * 60)

    # Confirm
    response = input("\nThis will populate empty direct columns from JSONB data.\nContinue? (yes/no): ")
    if response.lower() != 'yes':
        print("Migration cancelled.")
        return

    # Run migrations
    migrate_numeric_columns()
    migrate_boolean_columns()

    # Verify
    verify_migration()

    print("\n" + "=" * 60)
    print("Migration complete!")
    print("To revert: python scripts/revert_migration.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
