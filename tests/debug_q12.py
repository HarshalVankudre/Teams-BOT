"""Debug Q12: What data exists for nature reserve requirements"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.postgres import postgres_service

def debug_q12():
    """Investigate data for Q12 requirements"""
    pg = postgres_service

    if not pg.available:
        print("PostgreSQL not available!")
        return

    print("=" * 60)
    print("Q12: Radweg durch Naturschutzgebiet")
    print("Requirements: Stufe V, leicht, emissionsarm, wenig Bodendruck")
    print("=" * 60)

    # Check abgasstufe values
    print("\n### Unique abgasstufe_eu values ###")
    sql = """SELECT eigenschaften_json->>'abgasstufe_eu' as abgasstufe, COUNT(*) as cnt
             FROM geraete
             WHERE eigenschaften_json->>'abgasstufe_eu' IS NOT NULL
             GROUP BY eigenschaften_json->>'abgasstufe_eu'
             ORDER BY cnt DESC"""
    result = pg.execute_query(sql)
    for r in result:
        print(f"  {r['abgasstufe']}: {r['cnt']}")

    # Light machines (< 5000kg)
    print("\n### Light machines (< 5000kg) ###")
    sql = """SELECT COUNT(*) FROM geraete
             WHERE eigenschaften_json->>'gewicht_kg' ~ '^[0-9.]+$'
               AND (eigenschaften_json->>'gewicht_kg')::numeric < 5000"""
    result = pg.execute_query(sql)
    print(f"  Total < 5000kg: {result[0]['count']}")

    # Very light machines (< 3000kg)
    sql = """SELECT COUNT(*) FROM geraete
             WHERE eigenschaften_json->>'gewicht_kg' ~ '^[0-9.]+$'
               AND (eigenschaften_json->>'gewicht_kg')::numeric < 3000"""
    result = pg.execute_query(sql)
    print(f"  Total < 3000kg: {result[0]['count']}")

    # Stufe V machines
    print("\n### Stufe V machines ###")
    queries = [
        ("= 'Stufe V'", "eigenschaften_json->>'abgasstufe_eu' = 'Stufe V'"),
        ("ILIKE '%stufe v%'", "eigenschaften_json->>'abgasstufe_eu' ILIKE '%stufe v%'"),
        ("ILIKE '%V%'", "eigenschaften_json->>'abgasstufe_eu' ILIKE '%V%'"),
    ]
    for desc, condition in queries:
        sql = f"SELECT COUNT(*) FROM geraete WHERE {condition}"
        result = pg.execute_query(sql)
        print(f"  abgasstufe_eu {desc}: {result[0]['count']}")

    # Stufe V AND light
    print("\n### Stufe V AND light machines ###")
    sql = """SELECT COUNT(*) FROM geraete
             WHERE eigenschaften_json->>'abgasstufe_eu' ILIKE '%V%'
               AND eigenschaften_json->>'gewicht_kg' ~ '^[0-9.]+$'
               AND (eigenschaften_json->>'gewicht_kg')::numeric < 5000"""
    result = pg.execute_query(sql)
    print(f"  Stufe V AND < 5000kg: {result[0]['count']}")

    sql = """SELECT COUNT(*) FROM geraete
             WHERE eigenschaften_json->>'abgasstufe_eu' ILIKE '%V%'
               AND eigenschaften_json->>'gewicht_kg' ~ '^[0-9.]+$'
               AND (eigenschaften_json->>'gewicht_kg')::numeric < 10000"""
    result = pg.execute_query(sql)
    print(f"  Stufe V AND < 10000kg: {result[0]['count']}")

    # Show sample Stufe V machines
    print("\n### Sample Stufe V machines ###")
    sql = """SELECT hersteller, bezeichnung, geraetegruppe,
                    eigenschaften_json->>'gewicht_kg' as gewicht,
                    eigenschaften_json->>'abgasstufe_eu' as abgasstufe
             FROM geraete
             WHERE eigenschaften_json->>'abgasstufe_eu' ILIKE '%V%'
             ORDER BY (CASE WHEN eigenschaften_json->>'gewicht_kg' ~ '^[0-9.]+$'
                           THEN (eigenschaften_json->>'gewicht_kg')::numeric
                           ELSE 999999 END) ASC
             LIMIT 10"""
    result = pg.execute_query(sql)
    for r in result:
        print(f"  - {r['hersteller']} {r['bezeichnung']} ({r['geraetegruppe']}): {r['gewicht']} kg, {r['abgasstufe']}")

    # Show lightest machines suitable for nature reserves
    print("\n### Lightest machines (any abgasstufe) ###")
    sql = """SELECT hersteller, bezeichnung, geraetegruppe,
                    eigenschaften_json->>'gewicht_kg' as gewicht,
                    eigenschaften_json->>'abgasstufe_eu' as abgasstufe
             FROM geraete
             WHERE eigenschaften_json->>'gewicht_kg' ~ '^[0-9.]+$'
             ORDER BY (eigenschaften_json->>'gewicht_kg')::numeric ASC
             LIMIT 10"""
    result = pg.execute_query(sql)
    for r in result:
        print(f"  - {r['hersteller']} {r['bezeichnung']} ({r['geraetegruppe']}): {r['gewicht']} kg, {r['abgasstufe']}")

if __name__ == "__main__":
    debug_q12()
