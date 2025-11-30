"""Debug SQL queries"""
import psycopg2
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

conn = psycopg2.connect(
    host=os.getenv('POSTGRES_HOST'),
    port=os.getenv('POSTGRES_PORT'),
    database=os.getenv('POSTGRES_DB'),
    user=os.getenv('POSTGRES_USER'),
    password=os.getenv('POSTGRES_PASSWORD')
)
cursor = conn.cursor()

# Test 1: AVG query
print('TEST 1: AVG Gewicht Bagger')
cursor.execute("""
SELECT AVG((eigenschaften_json->>'gewicht_kg')::numeric) AS avg_gewicht
FROM geraete
WHERE (kategorie = 'bagger' OR geraetegruppe ILIKE '%bagger%')
  AND eigenschaften_json->>'gewicht_kg' != 'nicht-vorhanden'
  AND eigenschaften_json->>'gewicht_kg' ~ '^[0-9.]+$'
""")
result = cursor.fetchone()
print(f'  AVG: {result[0]:.2f} kg')

# Test 2: Schwerster Kettenbagger
print('\nTEST 2: Schwerster Kettenbagger (Top 5)')
cursor.execute("""
SELECT bezeichnung, hersteller, (eigenschaften_json->>'gewicht_kg')::numeric as gewicht
FROM geraete
WHERE geraetegruppe ILIKE '%Kettenbagger%'
  AND eigenschaften_json->>'gewicht_kg' != 'nicht-vorhanden'
  AND eigenschaften_json->>'gewicht_kg' ~ '^[0-9.]+$'
ORDER BY gewicht DESC
LIMIT 5
""")
for row in cursor.fetchall():
    print(f'  {row[1]} {row[0]}: {row[2]} kg')

cursor.close()
conn.close()
