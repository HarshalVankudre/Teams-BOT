"""Debug Q12: Check Stufe V machines with reasonable weights"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.postgres import postgres_service

def debug_q12():
    pg = postgres_service

    if not pg.available:
        print("PostgreSQL not available!")
        return

    # Check Stufe V machines with reasonable weights (> 100 kg)
    sql = """SELECT hersteller, bezeichnung, geraetegruppe,
                    eigenschaften_json->>'gewicht_kg' as gewicht,
                    eigenschaften_json->>'abgasstufe_eu' as abgasstufe
             FROM geraete
             WHERE eigenschaften_json->>'abgasstufe_eu' = 'Stufe V'
               AND eigenschaften_json->>'gewicht_kg' ~ '^[0-9.]+$'
               AND (eigenschaften_json->>'gewicht_kg')::numeric > 100
               AND (eigenschaften_json->>'gewicht_kg')::numeric < 10000
             ORDER BY (eigenschaften_json->>'gewicht_kg')::numeric ASC
             LIMIT 15"""

    result = pg.execute_query(sql)
    print(f"Stufe V machines 100-10000kg: {len(result)} found")
    for r in result:
        print(f"  {r['hersteller']} {r['bezeichnung']} ({r['geraetegruppe']}): {r['gewicht']} kg")

    # Q12 original requirements: < 5000 kg
    sql = """SELECT hersteller, bezeichnung, geraetegruppe,
                    eigenschaften_json->>'gewicht_kg' as gewicht,
                    eigenschaften_json->>'abgasstufe_eu' as abgasstufe
             FROM geraete
             WHERE eigenschaften_json->>'abgasstufe_eu' = 'Stufe V'
               AND eigenschaften_json->>'gewicht_kg' ~ '^[0-9.]+$'
               AND (eigenschaften_json->>'gewicht_kg')::numeric < 5000
               AND (eigenschaften_json->>'gewicht_kg')::numeric > 100
             ORDER BY (eigenschaften_json->>'gewicht_kg')::numeric ASC
             LIMIT 15"""

    result = pg.execute_query(sql)
    print(f"\nStufe V machines 100-5000kg (Q12 requirement): {len(result)} found")
    for r in result:
        print(f"  {r['hersteller']} {r['bezeichnung']} ({r['geraetegruppe']}): {r['gewicht']} kg")

if __name__ == "__main__":
    debug_q12()
