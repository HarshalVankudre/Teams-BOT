"""Debug script to investigate test failures by querying database directly"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.postgres import postgres_service

def debug_queries():
    """Run debug queries to understand the data"""
    pg = postgres_service

    if not pg.available:
        print("PostgreSQL not available!")
        return

    print("=" * 60)
    print("DEBUGGING TEST FAILURES")
    print("=" * 60)

    # Q1: How many Bagger?
    print("\n### Q1: Wie viele Bagger haben wir im Bestand? ###")
    print("Expected: 153")

    # Test different queries
    queries = [
        ("kategorie = 'bagger'", "SELECT COUNT(*) FROM geraete WHERE kategorie = 'bagger'"),
        ("geraetegruppe ILIKE '%bagger%'", "SELECT COUNT(*) FROM geraete WHERE geraetegruppe ILIKE '%bagger%'"),
        ("kategorie = 'bagger' OR geraetegruppe ILIKE '%bagger%'",
         "SELECT COUNT(*) FROM geraete WHERE kategorie = 'bagger' OR geraetegruppe ILIKE '%bagger%'"),
    ]

    for desc, sql in queries:
        result = pg.execute_query(sql)
        count = result[0]['count'] if result else 0
        print(f"  {desc}: {count}")

    # Show geraetegruppe breakdown for bagger category
    sql = """SELECT geraetegruppe, COUNT(*) as cnt
             FROM geraete
             WHERE kategorie = 'bagger' OR geraetegruppe ILIKE '%bagger%'
             GROUP BY geraetegruppe
             ORDER BY cnt DESC"""
    result = pg.execute_query(sql)
    print("\n  Breakdown by geraetegruppe:")
    for r in result:
        print(f"    - {r['geraetegruppe']}: {r['cnt']}")

    # Q2: Liebherr vs Caterpillar
    print("\n### Q2: Wie viele Liebherr vs Caterpillar? ###")
    print("Expected: Liebherr=677, Caterpillar=144")

    sql = "SELECT hersteller, COUNT(*) as cnt FROM geraete WHERE hersteller IN ('Liebherr', 'Caterpillar') GROUP BY hersteller"
    result = pg.execute_query(sql)
    for r in result:
        print(f"  {r['hersteller']}: {r['cnt']}")

    # Q3: Heaviest machine and lightest bagger
    print("\n### Q3: Schwerste Maschine / Leichtester Bagger ###")
    print("Expected: Heaviest=42,000kg Sennebogen, Lightest bagger=935kg CAT 300.9D")

    # Heaviest overall
    sql = """SELECT hersteller, bezeichnung, eigenschaften_json->>'gewicht_kg' as gewicht
             FROM geraete
             WHERE eigenschaften_json->>'gewicht_kg' ~ '^[0-9.]+$'
             ORDER BY (eigenschaften_json->>'gewicht_kg')::numeric DESC
             LIMIT 5"""
    result = pg.execute_query(sql)
    print("\n  Top 5 heaviest machines:")
    for r in result:
        print(f"    - {r['hersteller']} {r['bezeichnung']}: {r['gewicht']} kg")

    # Lightest bagger
    sql = """SELECT hersteller, bezeichnung, eigenschaften_json->>'gewicht_kg' as gewicht
             FROM geraete
             WHERE (kategorie = 'bagger' OR geraetegruppe ILIKE '%bagger%')
               AND eigenschaften_json->>'gewicht_kg' ~ '^[0-9.]+$'
             ORDER BY (eigenschaften_json->>'gewicht_kg')::numeric ASC
             LIMIT 5"""
    result = pg.execute_query(sql)
    print("\n  Top 5 lightest baggers:")
    for r in result:
        print(f"    - {r['hersteller']} {r['bezeichnung']}: {r['gewicht']} kg")

    # Q4: Klimaanlage
    print("\n### Q4: Geräte mit Klimaanlage ###")
    print("Expected: 75")

    queries = [
        ("klimaanlage = 'true'", "SELECT COUNT(*) FROM geraete WHERE eigenschaften_json->>'klimaanlage' = 'true'"),
        ("klimaanlage = true (boolean)", "SELECT COUNT(*) FROM geraete WHERE (eigenschaften_json->>'klimaanlage')::boolean = true"),
    ]

    for desc, sql in queries:
        try:
            result = pg.execute_query(sql)
            count = result[0]['count'] if result else 0
            print(f"  {desc}: {count}")
        except Exception as e:
            print(f"  {desc}: ERROR - {e}")

    # Q6: Kettenbagger vs Mobilbagger average weight
    print("\n### Q6: Kettenbagger vs Mobilbagger (Durchschnittsgewicht) ###")
    print("Expected: Kettenbagger=21,862kg avg, Mobilbagger=15,066kg avg")

    sql = """SELECT geraetegruppe,
                    COUNT(*) as anzahl,
                    ROUND(AVG((eigenschaften_json->>'gewicht_kg')::numeric)) as avg_gewicht,
                    MAX((eigenschaften_json->>'gewicht_kg')::numeric) as max_gewicht
             FROM geraete
             WHERE geraetegruppe IN ('Kettenbagger', 'Mobilbagger')
               AND eigenschaften_json->>'gewicht_kg' ~ '^[0-9.]+$'
             GROUP BY geraetegruppe"""
    result = pg.execute_query(sql)
    for r in result:
        print(f"  {r['geraetegruppe']}: Count={r['anzahl']}, Avg={r['avg_gewicht']}kg, Max={r['max_gewicht']}kg")

    # Check unique kategorie values
    print("\n### Unique kategorie values ###")
    sql = "SELECT DISTINCT kategorie, COUNT(*) as cnt FROM geraete GROUP BY kategorie ORDER BY cnt DESC"
    result = pg.execute_query(sql)
    for r in result:
        print(f"  {r['kategorie']}: {r['cnt']}")

    # Check unique hersteller values
    print("\n### Top 10 Hersteller ###")
    sql = "SELECT hersteller, COUNT(*) as cnt FROM geraete GROUP BY hersteller ORDER BY cnt DESC LIMIT 10"
    result = pg.execute_query(sql)
    for r in result:
        print(f"  {r['hersteller']}: {r['cnt']}")


if __name__ == "__main__":
    debug_queries()
