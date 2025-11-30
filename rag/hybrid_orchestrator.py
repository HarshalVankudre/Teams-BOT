"""
Hybrid Query Orchestrator
Routes queries between PostgreSQL (structured) and Pinecone (semantic).
Uses schema context for classification and context-aware response formatting.
"""
import json
import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from openai import AsyncOpenAI

from .postgres import PostgresService, postgres_service
from .config import config


class QueryType(Enum):
    """Query classification types"""
    AGGREGATION = "aggregation"
    FILTER = "filter"
    SEMANTIC = "semantic"
    LOOKUP = "lookup"
    COMPARISON = "comparison"
    HYBRID = "hybrid"


@dataclass
class QueryContext:
    """Context extracted from query classification"""
    query_type: QueryType
    confidence: float
    display_fields: List[str] = field(default_factory=list)
    structured_filters: Dict[str, Any] = field(default_factory=dict)
    semantic_query: Optional[str] = None


@dataclass
class HybridResult:
    """Result from hybrid query orchestrator"""
    query_type: QueryType
    answer: str
    sql_query: Optional[str] = None
    raw_results: Optional[List[Dict]] = None
    source: str = "unknown"
    confidence: float = 0.0
    display_fields: List[str] = field(default_factory=list)
    structured_filters: Optional[Dict[str, Any]] = None
    semantic_query: Optional[str] = None


# Field metadata for formatting
FIELD_CONFIG = {
    "gewicht_kg": {"label": "Gewicht", "unit": "kg", "format": "number"},
    "motor_leistung_kw": {"label": "Leistung", "unit": "kW", "format": "number"},
    "arbeitsbreite_mm": {"label": "Arbeitsbreite", "unit": "mm", "format": "number"},
    "grabtiefe_mm": {"label": "Grabtiefe", "unit": "mm", "format": "number"},
    "reichweite_mm": {"label": "Reichweite", "unit": "mm", "format": "number"},
    "geraetegruppe": {"label": "Typ", "unit": "", "format": "text"},
    "kategorie": {"label": "Kategorie", "unit": "", "format": "text"},
    "hersteller": {"label": "Hersteller", "unit": "", "format": "text"},
    "seriennummer": {"label": "SN", "unit": "", "format": "text"},
    "inventarnummer": {"label": "Inv.", "unit": "", "format": "text"},
}


class HybridOrchestrator:
    """
    Intelligent query router with context-aware formatting.
    LLM determines both routing and which fields are relevant to display.
    """

    def __init__(
        self,
        openai_client: Optional[AsyncOpenAI] = None,
        postgres: Optional[PostgresService] = None,
        verbose: bool = False
    ):
        self.client = openai_client or AsyncOpenAI(api_key=config.openai_api_key)
        self.postgres = postgres or postgres_service
        self.verbose = verbose
        self.model = config.response_model
        self.reasoning_effort = config.response_reasoning
        self.postgres_available = self.postgres.available if self.postgres else False

        if self.verbose:
            print(f"[Orchestrator] PostgreSQL: {'available' if self.postgres_available else 'unavailable'}")
            print(f"[Orchestrator] Classifier: {self.model} (reasoning: {self.reasoning_effort})")

    def _build_classification_prompt(self) -> str:
        """Build classification prompt with schema and display field selection"""
        schema = self.postgres.SCHEMA_INFO if self.postgres else ""

        return f"""Du bist ein Query-Klassifikator für eine Baumaschinen-Datenbank.

SCHEMA:
{schema}

KLASSIFIKATION:
1. AGGREGATION - Statistik (Anzahl, Durchschnitt, Max/Min)
2. FILTER - Liste mit NUR Schema-Kriterien (abgasstufe_eu, gewicht_kg, klimaanlage, hersteller)
3. SEMANTIC - Abstrakte Konzepte, Empfehlungen, ODER Kriterien die NICHT im Schema sind
4. LOOKUP - Spezifisches Gerät per ID/Name
5. COMPARISON - Vergleich
6. HYBRID - Schema-Filter + Semantik

WICHTIG für SEMANTIC:
- Bio-Hydrauliköl, Umweltauflagen, Naturschutz → SEMANTIC (nicht im Schema!)
- Szenario-Fragen (Baustelle beschrieben) → SEMANTIC
- "eignet sich", "empfehlen", "beste Maschine für" → SEMANTIC

DISPLAY_FIELDS - Welche Felder sind RELEVANT für die Antwort?
Wähle NUR Felder die zur Anfrage passen:
- Frage nach Gewicht/Tonnen → ["gewicht_kg"]
- Frage nach Leistung/kW/PS → ["motor_leistung_kw"]
- Frage nach Feature (Klimaanlage, etc.) → [] (Feature ist implizit bestätigt)
- Frage nach Hersteller → ["geraetegruppe"]
- Frage nach Typ/Kategorie → ["kategorie"]
- Allgemeine Liste → ["geraetegruppe"]
- Lookup/Details → ["gewicht_kg", "motor_leistung_kw", "geraetegruppe"]

WICHTIG: Weniger ist mehr! Zeige nur was relevant ist.

JSON-Antwort:
{{
  "type": "AGGREGATION|FILTER|SEMANTIC|LOOKUP|COMPARISON|HYBRID",
  "confidence": 0.0-1.0,
  "display_fields": ["feld1", "feld2"],
  "structured_filters": {{"kategorie": null, "hersteller": null}},
  "semantic_query": null
}}"""

    async def translate_for_sql(self, query: str) -> str:
        """Translate German query to English for SQL generation, preserving equipment terms"""
        # Don't translate - German works better with German database values
        return query

    def _build_sql_prompt(self) -> str:
        """Build SQL generation prompt with schema info"""
        schema = self.postgres.SCHEMA_INFO if self.postgres else ""
        return f"""Du bist PostgreSQL-Experte. Erstelle NUR SQL für Tabelle 'geraete' (Baumaschinen).

WICHTIG: Antworte NUR mit dem SQL Query. Keine Erklärungen, kein Markdown, nur SQL!

{schema}

WICHTIGE REGELN:

1. JSONB-Felder haben manchmal 'nicht-vorhanden' als Wert!
   Bei ALLEN numerischen Operationen (ORDER BY, AVG, Vergleiche) MUSS:
   WHERE eigenschaften_json->>'feldname' ~ '^[0-9.]+$'

2. WICHTIG: Bagger filtern - NICHT kategorie verwenden!
   Viele Bagger haben kategorie=NULL. IMMER geraetegruppe verwenden:
   WHERE geraetegruppe ILIKE '%bagger%'  -- findet Mobilbagger, Kettenbagger, Minibagger, etc.

   Beispiel Bagger zählen:
   SELECT COUNT(*) FROM geraete WHERE geraetegruppe ILIKE '%bagger%'

3. Stärkster = höchste motor_leistung_kw:
   SELECT *, eigenschaften_json->>'motor_leistung_kw' as motor_leistung_kw,
          eigenschaften_json->>'gewicht_kg' as gewicht_kg
   FROM geraete
   WHERE geraetegruppe ILIKE '%bagger%'
     AND eigenschaften_json->>'motor_leistung_kw' ~ '^[0-9.]+$'
   ORDER BY (eigenschaften_json->>'motor_leistung_kw')::numeric DESC
   LIMIT 1

4. Multi-Part Fragen (z.B. "schwerste UND leichteste") - UNION ALL verwenden:
   (SELECT 'Schwerste Maschine' as typ, hersteller, bezeichnung, geraetegruppe,
           eigenschaften_json->>'gewicht_kg' as gewicht_kg
    FROM geraete WHERE eigenschaften_json->>'gewicht_kg' ~ '^[0-9.]+$'
    ORDER BY (eigenschaften_json->>'gewicht_kg')::numeric DESC LIMIT 1)
   UNION ALL
   (SELECT 'Leichtester Bagger' as typ, hersteller, bezeichnung, geraetegruppe,
           eigenschaften_json->>'gewicht_kg' as gewicht_kg
    FROM geraete WHERE geraetegruppe ILIKE '%bagger%'
      AND eigenschaften_json->>'gewicht_kg' ~ '^[0-9.]+$'
    ORDER BY (eigenschaften_json->>'gewicht_kg')::numeric ASC LIMIT 1)

5. Vergleich - Zeige TOTAL Anzahl UND Durchschnittsgewicht:
   SELECT g.geraetegruppe,
          (SELECT COUNT(*) FROM geraete WHERE geraetegruppe = g.geraetegruppe) as anzahl,
          ROUND(AVG((eigenschaften_json->>'gewicht_kg')::numeric)) as avg_gewicht
   FROM geraete g
   WHERE g.geraetegruppe IN ('Kettenbagger', 'Mobilbagger')
     AND g.eigenschaften_json->>'gewicht_kg' ~ '^[0-9.]+$'
   GROUP BY g.geraetegruppe

6. Multi-Hersteller (z.B. "Liebherr UND Caterpillar"):
   SELECT hersteller, COUNT(*) as anzahl FROM geraete
   WHERE hersteller IN ('Liebherr', 'Caterpillar') GROUP BY hersteller

7. Lookup mit Details - flexibler Match (Modellname kann Leerzeichen haben):
   SELECT *, eigenschaften_json->>'gewicht_kg' as gewicht_kg,
          eigenschaften_json->>'motor_leistung_kw' as motor_leistung_kw
   FROM geraete
   WHERE hersteller ILIKE '%Marke%'
     AND (bezeichnung ILIKE '%A920%' OR bezeichnung ILIKE '%A 920%' OR bezeichnung ILIKE '%A%920%')

8. Boolean: eigenschaften_json->>'klimaanlage' = 'true'

9. LIMIT: 1 für max/min, 100 für Listen (NICHT 50!)

Gib NUR das SQL aus, kein Markdown."""

    async def classify_query(self, query: str) -> QueryContext:
        """Classify query - single attempt with keyword fallback"""
        # Try LLM classification first
        try:
            response = await self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": self._build_classification_prompt()},
                    {"role": "user", "content": query}
                ],
                text={"format": {"type": "json_object"}},
                max_output_tokens=300
            )

            if response.output_text:
                result = json.loads(response.output_text)
                context = QueryContext(
                    query_type=QueryType(result["type"].lower()),
                    confidence=float(result.get("confidence", 0.8)),
                    display_fields=result.get("display_fields", []),
                    structured_filters={k: v for k, v in result.get("structured_filters", {}).items() if v},
                    semantic_query=result.get("semantic_query")
                )
                if self.verbose:
                    print(f"[Classify] {context.query_type.value} | display: {context.display_fields}")
                return context

        except Exception as e:
            print(f"[Classify] LLM failed: {e}")

        # Keyword-based fallback
        return self._keyword_classify(query)

    def _keyword_classify(self, query: str) -> QueryContext:
        """Simple keyword-based classification"""
        q = query.lower()

        if any(w in q for w in ['wie viele', 'anzahl', 'durchschnitt', 'stärkste', 'schwerste', 'leichteste', 'max', 'min']):
            return QueryContext(QueryType.AGGREGATION, 0.7, ['gewicht_kg', 'motor_leistung_kw'])

        if any(w in q for w in ['vergleich', 'unterschied', 'vs', 'versus']):
            return QueryContext(QueryType.COMPARISON, 0.7, ['gewicht_kg', 'motor_leistung_kw'])

        if any(w in q for w in ['details', 'information', 'spezifikation', 'info zu']):
            return QueryContext(QueryType.LOOKUP, 0.7, ['gewicht_kg', 'motor_leistung_kw', 'geraetegruppe'])

        if any(w in q for w in ['alle', 'zeige', 'liste', 'welche', 'mit', 'unter', 'über']):
            return QueryContext(QueryType.FILTER, 0.7, ['geraetegruppe'])

        return QueryContext(QueryType.FILTER, 0.5, ['geraetegruppe'])

    def _get_fallback_sql(self, query: str, context: QueryContext) -> str:
        """Generate fallback SQL for common query patterns when LLM fails"""
        q = query.lower()

        # Comparison: Kettenbagger vs Mobilbagger
        if context.query_type == QueryType.COMPARISON:
            if 'kettenbagger' in q and 'mobilbagger' in q:
                return """SELECT geraetegruppe,
                    COUNT(*) as anzahl,
                    ROUND(AVG((eigenschaften_json->>'gewicht_kg')::numeric)) as avg_gewicht,
                    ROUND(AVG((eigenschaften_json->>'motor_leistung_kw')::numeric)) as avg_leistung
                FROM geraete
                WHERE geraetegruppe ILIKE '%bagger%'
                    AND (geraetegruppe ILIKE '%Ketten%' OR geraetegruppe ILIKE '%Mobil%')
                    AND eigenschaften_json->>'gewicht_kg' ~ '^[0-9.]+$'
                GROUP BY geraetegruppe
                ORDER BY avg_gewicht DESC"""

            # Generic comparison - extract equipment types
            types = []
            for keyword in ['bagger', 'lader', 'walze', 'fertiger', 'fräse', 'kran']:
                if keyword in q:
                    types.append(keyword)

            if types:
                conditions = " OR ".join([f"geraetegruppe ILIKE '%{t}%'" for t in types])
                return f"""SELECT geraetegruppe,
                    COUNT(*) as anzahl,
                    ROUND(AVG((eigenschaften_json->>'gewicht_kg')::numeric)) as avg_gewicht,
                    ROUND(AVG((eigenschaften_json->>'motor_leistung_kw')::numeric)) as avg_leistung
                FROM geraete
                WHERE ({conditions})
                    AND eigenschaften_json->>'gewicht_kg' ~ '^[0-9.]+$'
                GROUP BY geraetegruppe
                ORDER BY avg_gewicht DESC
                LIMIT 20"""

        # Multi-manufacturer count
        if context.query_type == QueryType.AGGREGATION:
            manufacturers = []
            for m in ['liebherr', 'caterpillar', 'bomag', 'vögele', 'hamm', 'wirtgen', 'kubota', 'volvo', 'dynapac']:
                if m in q:
                    manufacturers.append(m.title())

            if len(manufacturers) >= 2:
                m_list = "', '".join(manufacturers)
                return f"""SELECT hersteller, COUNT(*) as anzahl
                FROM geraete
                WHERE LOWER(hersteller) IN ('{m_list.lower()}')
                GROUP BY hersteller
                ORDER BY anzahl DESC"""

        return ""

    async def generate_sql(self, query: str, context: QueryContext) -> str:
        """Generate SQL using AI with retry logic and fallback patterns"""
        max_retries = 2

        for attempt in range(max_retries):
            try:
                # Translate German query to English for better SQL generation
                english_query = await self.translate_for_sql(query)
                if self.verbose and attempt == 0:
                    print(f"[SQL] Translated: {english_query[:50]}...")

                # Build a more explicit prompt based on query type and retry
                prompt = english_query
                if context.query_type == QueryType.COMPARISON:
                    prompt = f"""VERGLEICH-ANFRAGE: {english_query}

WICHTIG: Erstelle eine GROUP BY Abfrage die beide Gruppen vergleicht mit:
- COUNT(*) as anzahl
- AVG(gewicht_kg) as avg_gewicht
- AVG(motor_leistung_kw) as avg_leistung

Antworte NUR mit SQL, keine Erklärung!"""
                elif attempt > 0:
                    prompt = f"Erstelle NUR SQL, keine Erklärung. Frage: {english_query}"

                # Generate SQL
                response = await self.client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": self._build_sql_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    max_output_tokens=800
                )

                sql = response.output_text.strip() if response.output_text else ""

                # Remove markdown code fences if present
                sql = re.sub(r'^```\w*\n?', '', sql)
                sql = re.sub(r'\n?```$', '', sql)
                sql = sql.strip()

                # Also try to extract SQL if wrapped in explanation
                if sql and 'SELECT' in sql.upper():
                    # Find the SELECT statement in the output
                    match = re.search(r'((?:\(SELECT|SELECT)[\s\S]+?)(?:;|\Z)', sql, re.IGNORECASE)
                    if match:
                        sql = match.group(1).strip()

                # Validate SQL - accept SELECT or (SELECT for UNION queries
                sql_upper = sql.upper().strip()
                if not sql or not (sql_upper.startswith('SELECT') or sql_upper.startswith('(SELECT')):
                    if attempt < max_retries - 1:
                        print(f"[SQL] Retry {attempt + 1}: Invalid output")
                        continue
                    # Try fallback SQL before giving up
                    fallback = self._get_fallback_sql(query, context)
                    if fallback:
                        print(f"[SQL] Using fallback SQL for {context.query_type.value}")
                        return fallback
                    print(f"[SQL] Invalid output: {sql[:50] if sql else 'empty'}")
                    return ""

                if self.verbose:
                    print(f"[SQL] {sql[:100]}...")

                return sql

            except Exception as e:
                print(f"[SQL] Error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    # Try fallback SQL on error
                    fallback = self._get_fallback_sql(query, context)
                    if fallback:
                        print(f"[SQL] Using fallback SQL after error")
                        return fallback
                    return ""

        return ""

    async def execute_query(self, query: str, context: QueryContext) -> Tuple[List[Dict], str]:
        """Execute SQL query"""
        sql = await self.generate_sql(query, context)
        if not sql:
            return [], ""

        results = self.postgres.execute_dynamic_sql(sql)

        if self.verbose:
            print(f"[SQL] {len(results)} results")

        return results, sql

    def format_results(self, results: List[Dict], context: QueryContext) -> str:
        """Format results showing only relevant fields"""
        if not results:
            return "Keine Ergebnisse gefunden."

        # Single aggregation
        if context.query_type == QueryType.AGGREGATION and len(results) == 1:
            return self._format_aggregation(results[0], context.display_fields)

        # Check if this is a grouped aggregation (multiple rows with counts)
        # e.g., GROUP BY hersteller with COUNT(*) or UNION results
        if (context.query_type == QueryType.AGGREGATION and
            len(results) > 1 and
            any(k in results[0] for k in ['count', 'anzahl', 'avg_gewicht', 'typ'])):
            return self._format_grouped_aggregation(results)

        # Comparison results (grouped data)
        if context.query_type == QueryType.COMPARISON:
            return self._format_comparison(results)

        # List results
        return self._format_list(results, context.display_fields)

    def _format_grouped_aggregation(self, results: List[Dict]) -> str:
        """Format grouped aggregation results (e.g., COUNT BY hersteller or UNION results)"""
        lines = []

        # Check if this is a UNION result with 'typ' column (multi-part query)
        if results and 'typ' in results[0]:
            return self._format_union_results(results)

        for r in results:
            # Get the group name (could be hersteller, geraetegruppe, kategorie, etc.)
            group_name = (r.get('hersteller') or r.get('geraetegruppe') or
                         r.get('kategorie') or 'Gruppe')
            # Clean up group name
            group_name = re.sub(r'\s*\([0-9,]+\s*to\s*-\s*[0-9,]+\s*to\)', '', str(group_name)).strip()

            # Get the count value
            count = r.get('anzahl') or r.get('count')
            if count is not None:
                lines.append(f"- **{group_name}**: {count}")
            else:
                # Might have other aggregations
                details = []
                if 'avg_gewicht' in r and r['avg_gewicht']:
                    details.append(f"Ø Gewicht: {float(r['avg_gewicht']):,.0f} kg")
                if details:
                    lines.append(f"- **{group_name}**: {', '.join(details)}")

        return "\n".join(lines)

    def _format_union_results(self, results: List[Dict]) -> str:
        """Format UNION results (multi-part queries like heaviest + lightest)"""
        lines = []

        for r in results:
            typ = r.get('typ', 'Ergebnis')
            name = f"{r.get('hersteller', '')} {r.get('bezeichnung', '')}".strip()
            group = r.get('geraetegruppe', '')

            details = []
            if r.get('gewicht_kg'):
                try:
                    weight = float(r['gewicht_kg'])
                    details.append(f"{weight:,.0f} kg")
                except (ValueError, TypeError):
                    pass
            if r.get('motor_leistung_kw'):
                try:
                    power = float(r['motor_leistung_kw'])
                    details.append(f"{power:,.0f} kW")
                except (ValueError, TypeError):
                    pass
            if group:
                details.append(group)

            detail_str = f" ({', '.join(details)})" if details else ""
            lines.append(f"**{typ}:** {name}{detail_str}")

        return "\n".join(lines)

    def _format_comparison(self, results: List[Dict]) -> str:
        """Format comparison results (grouped aggregations)"""
        lines = ["**Vergleich:**\n"]

        for r in results:
            # Get the group name (could be geraetegruppe, kategorie, etc.)
            group_name = r.get('geraetegruppe') or r.get('kategorie') or r.get('hersteller') or 'Gruppe'
            # Clean up group name
            group_name = re.sub(r'\s*\([0-9,]+\s*to\s*-\s*[0-9,]+\s*to\)', '', str(group_name)).strip()

            details = []
            if 'anzahl' in r or 'count' in r:
                count = r.get('anzahl') or r.get('count')
                details.append(f"Anzahl: {count}")
            if 'avg_gewicht' in r and r['avg_gewicht']:
                details.append(f"Ø Gewicht: {float(r['avg_gewicht']):,.0f} kg")
            if 'avg_leistung' in r and r['avg_leistung']:
                details.append(f"Ø Leistung: {float(r['avg_leistung']):,.0f} kW")
            if 'min_gewicht' in r and r['min_gewicht']:
                details.append(f"Min: {float(r['min_gewicht']):,.0f} kg")
            if 'max_gewicht' in r and r['max_gewicht']:
                details.append(f"Max: {float(r['max_gewicht']):,.0f} kg")

            detail_str = ", ".join(details) if details else ""
            lines.append(f"**{group_name}**: {detail_str}")

        return "\n".join(lines)

    def _format_aggregation(self, result: Dict, display_fields: List[str]) -> str:
        """Format single aggregation result"""
        # Count
        count = result.get('count') or result.get('anzahl')
        if count is not None:
            # Try to get category for better description
            category = result.get('kategorie') or ''
            if category:
                return f"**{count}** {category.title()}"
            return f"**{count}**"

        # Average - check various keys
        for key in ['avg_gewicht', 'avg', 'durchschnittsgewicht', 'durchschnittliches_gewicht', 'avg_leistung']:
            if key in result and result[key]:
                try:
                    val = float(result[key])
                    unit = "kg" if "gewicht" in key else "kW"
                    return f"Durchschnitt: **{val:,.0f} {unit}**"
                except (ValueError, TypeError):
                    pass

        # Single item (max/min query)
        if 'bezeichnung' in result:
            name = f"{result.get('hersteller', '')} {result.get('bezeichnung', '')}".strip()
            lines = [f"**{name}**"]

            # Always show key specs for single results
            fields_to_show = ['gewicht_kg', 'motor_leistung_kw', 'geraetegruppe']

            # Try direct fields first, then eigenschaften_json
            for field in fields_to_show:
                value = result.get(field)
                # Also check if the value is in eigenschaften_json
                if not value and 'eigenschaften_json' in result:
                    props = result.get('eigenschaften_json', {})
                    if isinstance(props, dict):
                        value = props.get(field)

                if value and value != 'nicht-vorhanden':
                    lines.append(self._format_field(field, value))

            # Add serial/inventory if available
            if result.get('seriennummer'):
                lines.append(f"- SN: {result['seriennummer']}")
            if result.get('inventarnummer'):
                lines.append(f"- Inv.: {result['inventarnummer']}")

            return "\n".join(lines)

        # Fallback: convert any remaining Decimals before JSON dump
        from decimal import Decimal
        clean = {k: float(v) if isinstance(v, Decimal) else v for k, v in result.items()}
        return json.dumps(clean, ensure_ascii=False)

    def _format_list(self, results: List[Dict], display_fields: List[str]) -> str:
        """Format list with context-relevant fields only"""
        total = results[0].get('total', len(results)) if results else len(results)
        lines = [f"**{total} Ergebnisse:**\n"]

        # Group by manufacturer
        by_manufacturer = {}
        for r in results:
            manufacturer = r.get('hersteller', 'Sonstige') or 'Sonstige'
            by_manufacturer.setdefault(manufacturer, []).append(r)

        for manufacturer, items in sorted(by_manufacturer.items()):
            lines.append(f"\n**{manufacturer}** ({len(items)}):")

            for r in items:
                name = r.get('bezeichnung', 'Unbekannt')

                # Clean up geraetegruppe - remove weight class suffix
                geraetegruppe = r.get('geraetegruppe', '')
                if geraetegruppe:
                    # Remove patterns like "(0,0 to - 4,4 to)" or "(10,1 to - 18,0 to)"
                    geraetegruppe = re.sub(r'\s*\([0-9,]+\s*to\s*-\s*[0-9,]+\s*to\)', '', geraetegruppe).strip()

                # Build details - show key fields that are available
                details = []

                # Always show gewicht_kg if available in result (from SQL query)
                if r.get('gewicht_kg') and r['gewicht_kg'] != 'nicht-vorhanden':
                    try:
                        weight = float(r['gewicht_kg'])
                        details.append(f"{weight:,.0f} kg")
                    except (ValueError, TypeError):
                        pass

                # Always show motor_leistung_kw if available
                if r.get('motor_leistung_kw') and r['motor_leistung_kw'] != 'nicht-vorhanden':
                    try:
                        power = float(r['motor_leistung_kw'])
                        details.append(f"{power:,.0f} kW")
                    except (ValueError, TypeError):
                        pass

                # Show geraetegruppe if no numeric fields shown
                if not details and geraetegruppe:
                    details.append(geraetegruppe)

                suffix = f" ({', '.join(details)})" if details else ""
                lines.append(f"  - {name}{suffix}")

        return "\n".join(lines)

    def _format_field(self, field: str, value: Any) -> str:
        """Format a field with label for detail view"""
        config = FIELD_CONFIG.get(field, {"label": field, "unit": "", "format": "text"})

        if config["format"] == "number":
            try:
                num = float(value)
                return f"- {config['label']}: {num:,.0f} {config['unit']}".strip()
            except (ValueError, TypeError):
                return f"- {config['label']}: {value}"
        else:
            # Clean geraetegruppe
            if field == "geraetegruppe":
                value = re.sub(r'\s*\([0-9,]+\s*to\s*-\s*[0-9,]+\s*to\)', '', str(value)).strip()
            return f"- {config['label']}: {value}"

    def _format_field_inline(self, field: str, value: Any) -> str:
        """Format a field for inline display (no label)"""
        config = FIELD_CONFIG.get(field, {"label": field, "unit": "", "format": "text"})

        if config["format"] == "number":
            try:
                num = float(value)
                return f"{num:,.0f} {config['unit']}".strip()
            except (ValueError, TypeError):
                return str(value)
        else:
            # Clean geraetegruppe
            if field == "geraetegruppe":
                value = re.sub(r'\s*\([0-9,]+\s*to\s*-\s*[0-9,]+\s*to\)', '', str(value)).strip()
            return str(value)

    async def query(self, user_query: str, force_postgres: bool = False, force_semantic: bool = False) -> HybridResult:
        """Process query with context-aware formatting"""

        # Classify and get display context
        context = await self.classify_query(user_query)

        # Handle forced routing
        if force_semantic:
            context.query_type = QueryType.SEMANTIC
        elif force_postgres and context.query_type == QueryType.SEMANTIC:
            context.query_type = QueryType.FILTER

        # Route SEMANTIC to Pinecone
        if context.query_type == QueryType.SEMANTIC:
            return HybridResult(
                query_type=context.query_type,
                answer="",
                source="pinecone",
                confidence=context.confidence,
                display_fields=context.display_fields
            )

        # Route HYBRID
        if context.query_type == QueryType.HYBRID:
            return HybridResult(
                query_type=context.query_type,
                answer="",
                source="hybrid",
                confidence=context.confidence,
                display_fields=context.display_fields,
                structured_filters=context.structured_filters,
                semantic_query=context.semantic_query or user_query
            )

        # Route to PostgreSQL
        if not self.postgres_available:
            return HybridResult(
                query_type=context.query_type,
                answer="PostgreSQL nicht verfügbar.",
                source="error",
                confidence=0.0
            )

        results, sql = await self.execute_query(user_query, context)
        answer = self.format_results(results, context)

        return HybridResult(
            query_type=context.query_type,
            answer=answer,
            sql_query=sql,
            raw_results=results,
            source="postgres",
            confidence=context.confidence,
            display_fields=context.display_fields
        )


# Lazy initialization
_orchestrator: Optional[HybridOrchestrator] = None


def get_orchestrator(verbose: bool = False) -> HybridOrchestrator:
    """Get or create orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = HybridOrchestrator(verbose=verbose)
    return _orchestrator


# Backwards compatibility
hybrid_orchestrator = HybridOrchestrator(verbose=True)
