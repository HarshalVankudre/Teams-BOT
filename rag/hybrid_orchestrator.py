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
2. FILTER - Liste mit Schema-Kriterien
3. SEMANTIC - Abstrakte Konzepte, Empfehlungen
4. LOOKUP - Spezifisches Gerät per ID/Name
5. COMPARISON - Vergleich
6. HYBRID - Schema-Filter + Semantik

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

    def _build_sql_prompt(self, query_type: QueryType, display_fields: List[str]) -> str:
        """Build SQL prompt with relevant fields"""
        schema = self.postgres.SCHEMA_INFO if self.postgres else ""

        # Build SELECT fields based on display_fields
        select_hint = ""
        if display_fields:
            select_hint = f"\nWICHTIG: SELECT muss enthalten: bezeichnung, hersteller, {', '.join(display_fields)}"

        return f"""PostgreSQL-Experte. Generiere SQL.

{schema}

REGELN:
1. ILIKE für Text (case-insensitive)
2. LIMIT 50 für Listen
3. Bagger: kategorie = 'bagger' OR geraetegruppe ILIKE '%bagger%'
4. Tonnen → kg * 1000
5. COUNT(*) OVER() as total für Listen
6. Geraetegruppe IMMER mit ILIKE '%name%'{select_hint}

JSONB NUMERISCH:
- Filter: eigenschaften_json->>'feld' != 'nicht-vorhanden'
- Prüfen: eigenschaften_json->>'feld' ~ '^[0-9.]+$'
- Cast: (eigenschaften_json->>'feld')::numeric

JSONB BOOLEAN:
- eigenschaften_json->>'feld' = 'true'

Query-Typ: {query_type.value}

NUR SQL, kein Markdown."""

    async def classify_query(self, query: str) -> QueryContext:
        """Classify query and determine relevant display fields"""
        try:
            response_params = {
                "model": self.model,
                "input": [
                    {"role": "system", "content": self._build_classification_prompt()},
                    {"role": "user", "content": query}
                ],
                "text": {"format": {"type": "json_object"}},
                "max_output_tokens": 500
            }

            if self.reasoning_effort and self.reasoning_effort.lower() != "none":
                response_params["reasoning"] = {"effort": "low"}

            response = await self.client.responses.create(**response_params)
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
            print(f"[Classify] Error: {e}")
            return QueryContext(
                query_type=QueryType.SEMANTIC,
                confidence=0.5,
                display_fields=[]
            )

    async def generate_sql(self, query: str, context: QueryContext) -> str:
        """Generate SQL with context-aware field selection"""
        try:
            response_params = {
                "model": self.model,
                "input": [
                    {"role": "system", "content": self._build_sql_prompt(context.query_type, context.display_fields)},
                    {"role": "user", "content": query}
                ],
                "max_output_tokens": 1000
            }

            if self.reasoning_effort and self.reasoning_effort.lower() != "none":
                response_params["reasoning"] = {"effort": "low"}

            response = await self.client.responses.create(**response_params)
            sql = response.output_text.strip()
            sql = re.sub(r'^```sql\n?', '', sql)
            sql = re.sub(r'\n?```$', '', sql)

            if self.verbose:
                print(f"[SQL] {sql[:100]}...")

            return sql.strip()

        except Exception as e:
            print(f"[SQL] Error: {e}")
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

        # List results
        return self._format_list(results, context.display_fields)

    def _format_aggregation(self, result: Dict, display_fields: List[str]) -> str:
        """Format single aggregation result"""
        # Count
        count = result.get('count') or result.get('anzahl')
        if count is not None:
            return f"**{count}** Ergebnisse"

        # Average
        for key in ['avg_gewicht', 'avg', 'durchschnittsgewicht', 'avg_leistung']:
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

            # Show only relevant display fields
            fields_to_show = display_fields if display_fields else ['gewicht_kg', 'motor_leistung_kw']
            for field in fields_to_show:
                value = result.get(field)
                if value:
                    lines.append(self._format_field(field, value))

            return "\n".join(lines)

        return json.dumps(result, ensure_ascii=False)

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

                # Build details based on display_fields
                details = []

                # If no specific display fields, show cleaned geraetegruppe
                if not display_fields:
                    if geraetegruppe:
                        details.append(geraetegruppe)
                else:
                    for field in display_fields:
                        value = r.get(field)
                        if value:
                            formatted = self._format_field_inline(field, value)
                            if formatted:
                                details.append(formatted)

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
