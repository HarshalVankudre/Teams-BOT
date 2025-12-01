"""
Revert Database Migration: Set direct columns back to NULL

This script reverts the migration by setting all direct numeric/boolean
columns back to NULL, restoring the original state where only JSONB has data.

USE THIS IF:
- Migration caused issues
- You want to go back to JSONB-only access
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.postgres import postgres_service
from rag.schema import NUMERIC_FIELDS, BOOLEAN_FIELDS


def revert_numeric_columns():
    """Set all numeric direct columns to NULL"""
    print("\n=== Reverting NUMERIC columns to NULL ===")

    for field in NUMERIC_FIELDS:
        sql = f"UPDATE geraete SET {field} = NULL WHERE {field} IS NOT NULL"

        try:
            conn = postgres_service._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql)
            rows_updated = cursor.rowcount
            conn.commit()
            cursor.close()
            conn.close()

            if rows_updated > 0:
                print(f"  {field}: {rows_updated} rows reverted to NULL")
            else:
                print(f"  {field}: already NULL")

        except Exception as e:
            print(f"  {field}: ERROR - {e}")


def revert_boolean_columns():
    """Set all boolean direct columns to NULL"""
    print("\n=== Reverting BOOLEAN columns to NULL ===")

    for field in BOOLEAN_FIELDS:
        sql = f"UPDATE geraete SET {field} = NULL WHERE {field} IS NOT NULL"

        try:
            conn = postgres_service._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql)
            rows_updated = cursor.rowcount
            conn.commit()
            cursor.close()
            conn.close()

            if rows_updated > 0:
                print(f"  {field}: {rows_updated} rows reverted to NULL")
            else:
                print(f"  {field}: already NULL")

        except Exception as e:
            print(f"  {field}: ERROR - {e}")


def verify_revert():
    """Verify the revert was successful"""
    print("\n=== Verification ===")

    sql = """
    SELECT
        COUNT(*) as total,
        COUNT(gewicht_kg) as gewicht_populated,
        COUNT(klimaanlage) as klimaanlage_populated
    FROM geraete
    """

    result = postgres_service.execute_query(sql)
    if result:
        r = result[0]
        print(f"  Total rows: {r['total']}")
        print(f"  gewicht_kg populated: {r['gewicht_populated']} (should be 0)")
        print(f"  klimaanlage populated: {r['klimaanlage_populated']} (should be 0)")


def main():
    print("=" * 60)
    print("REVERT MIGRATION: Direct Columns -> NULL")
    print("=" * 60)

    response = input("\nThis will set ALL direct numeric/boolean columns to NULL.\nContinue? (yes/no): ")
    if response.lower() != 'yes':
        print("Revert cancelled.")
        return

    revert_numeric_columns()
    revert_boolean_columns()
    verify_revert()

    print("\n" + "=" * 60)
    print("Revert complete! Direct columns are now NULL.")
    print("Data is preserved in eigenschaften_json (JSONB).")
    print("=" * 60)


if __name__ == "__main__":
    main()
