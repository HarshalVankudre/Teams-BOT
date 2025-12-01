"""
SQL Generator Agent (Non-Reasoning Model)
Generates and executes SQL queries against the PostgreSQL database.
Uses a fast, non-reasoning model for efficiency.
"""
import json
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

from .base import BaseAgent, AgentContext, AgentResponse, AgentType
from .registry import AgentMetadata, AgentCapability, register_agent
from ..config import config
from ..postgres import postgres_service, PostgresService


# Agent metadata for registration
SQL_AGENT_METADATA = AgentMetadata(
    agent_id="sql",
    name="SQL Generator Agent",
    description="Generates and executes SQL queries against the equipment database",
    detailed_description="""Generiert und führt SQL-Abfragen gegen die PostgreSQL-Datenbank aus.
Verwende diesen Agenten für:
- Zählung von Geräten (z.B. "Wie viele Bagger haben wir?")
- Filterung nach Eigenschaften (z.B. "Bagger über 15 Tonnen")
- Aggregationen (z.B. "Durchschnittsgewicht der Mobilbagger")
- Vergleiche zwischen Gerätegruppen
- Suche nach Seriennummer oder Inventarnummer""",
    capabilities=[AgentCapability.DATABASE_QUERY, AgentCapability.DATA_ANALYSIS],
    uses_reasoning=False,
    default_model="gpt-4o-mini",
    parameters={
        "task_description": {
            "type": "string",
            "description": "Beschreibung der SQL-Aufgabe, z.B. 'Zähle alle Bagger im Bestand'"
        },
        "filters": {
            "type": "object",
            "description": "Optionale Filter wie {'hersteller': 'Liebherr', 'min_gewicht': 10000}"
        }
    },
    example_queries=[
        "Wie viele Bagger haben wir?",
        "Liste alle Kettenbagger von Liebherr",
        "Welche Geräte haben GPS?",
        "Durchschnittsgewicht der Mobilbagger",
        "Vergleich Kettenbagger vs Mobilbagger"
    ],
    priority=10
)


@register_agent(SQL_AGENT_METADATA)
class SQLGeneratorAgent(BaseAgent):
    """
    Generates SQL queries based on task descriptions and executes them.
    Uses tool-calling to generate valid SQL for the equipment database.
    """

    # Schema information for the LLM
    SCHEMA_INFO = """
DATENBANK-SCHEMA: Tabelle "geraete" (Baumaschinen)

KLASSIFIKATION:
- geraetegruppe: WICHTIGSTE SPALTE für Gerätetypen!
  Bagger: 'Mobilbagger', 'Kettenbagger', 'Minibagger (0,0 to - 4,4 to)'
  Walzen: 'Tandemwalze', 'Walzenzug', 'Gummiradwalze'
  Fertiger: 'Radfertiger', 'Kettenfertiger'
  Fräsen: 'Kaltfräse', 'Großfräse'
- kategorie: Kann NULL sein! Nur als Fallback verwenden
- hersteller: 'Caterpillar', 'Liebherr', 'Bomag', 'Vögele', 'Hamm', 'Wirtgen', 'Kubota', 'Volvo', etc.
- verwendung: 'Vermietung', 'Eigenbedarf', 'Verkauf', 'Externes Gerät' - NUR filtern wenn explizit angefragt!

IDENTIFIKATION:
- id: VARCHAR Primary Key
- seriennummer: Seriennummer
- inventarnummer: Inventarnummer
- bezeichnung: Modellname/Bezeichnung
- titel: Titel

JSONB SPALTE "eigenschaften_json":
Numerische Werte (als String-Dezimalzahlen wie "16300.0"):
- gewicht_kg: Betriebsgewicht
- motor_leistung_kw: Motorleistung
- breite_mm, hoehe_mm, laenge_mm: Abmessungen
- grabtiefe_mm: Grabtiefe (Bagger)
- arbeitsbreite_mm: Arbeitsbreite (Walzen, Fertiger)
- reichweite_mm: Reichweite
- hubkraft_kg: Hubkraft

Boolean Werte (als String "true"/"false"/"nicht-vorhanden"):
- klimaanlage, hammerhydraulik, schnellwechsler
- zentralschmierung, greifer, allradantrieb
- tiltrotator, rueckfahrkamera, gps

Text Werte:
- motor_hersteller: 'Deutz', 'Cummins', 'Kubota'
- abgasstufe_eu: 'Stufe III', 'Stufe IV', 'Stufe V', 'Stage V / TIER4f'

ARRAY SPALTE:
- einsatzgebiete: TEXT[] Array mit Einsatzgebieten

SQL-SYNTAX FÜR JSONB:
- String: eigenschaften_json->>'feldname' = 'wert'
- Boolean: eigenschaften_json->>'klimaanlage' = 'true'
- Numerisch filtern:
  eigenschaften_json->>'gewicht_kg' ~ '^[0-9]+(\\.[0-9]+)?$'
  AND (eigenschaften_json->>'gewicht_kg')::numeric > 15000
- Aggregation mit Filter für "nicht-vorhanden":
  AVG(CASE WHEN eigenschaften_json->>'gewicht_kg' ~ '^[0-9]+(\\.[0-9]+)?$'
           AND eigenschaften_json->>'gewicht_kg' NOT IN ('nicht-vorhanden', '')
      THEN (eigenschaften_json->>'gewicht_kg')::numeric END)
"""

    SQL_EXAMPLES = """
SQL-BEISPIELE:

-- Anzahl Bagger (WICHTIG: geraetegruppe verwenden, NICHT kategorie!):
SELECT COUNT(*) as anzahl FROM geraete WHERE geraetegruppe ILIKE '%bagger%'

-- Alle Mobilbagger:
SELECT COUNT(*) as anzahl FROM geraete WHERE geraetegruppe = 'Mobilbagger'

-- Bagger über 15t mit Klimaanlage:
SELECT hersteller, bezeichnung, eigenschaften_json->>'gewicht_kg' as gewicht_kg
FROM geraete
WHERE geraetegruppe ILIKE '%bagger%'
AND eigenschaften_json->>'gewicht_kg' ~ '^[0-9]+(\\.[0-9]+)?$'
AND (eigenschaften_json->>'gewicht_kg')::numeric > 15000
AND eigenschaften_json->>'klimaanlage' = 'true'
ORDER BY (eigenschaften_json->>'gewicht_kg')::numeric DESC
LIMIT 20

-- Durchschnittsgewicht nach Gerätegruppe für Bagger:
SELECT geraetegruppe,
       COUNT(*) as anzahl,
       ROUND(AVG(CASE
           WHEN eigenschaften_json->>'gewicht_kg' ~ '^[0-9]+(\\.[0-9]+)?$'
                AND eigenschaften_json->>'gewicht_kg' NOT IN ('nicht-vorhanden', '')
           THEN (eigenschaften_json->>'gewicht_kg')::numeric
       END)) as durchschnitt_kg
FROM geraete
WHERE geraetegruppe ILIKE '%bagger%'
GROUP BY geraetegruppe
ORDER BY anzahl DESC

-- Alle Walzen:
SELECT COUNT(*) as anzahl FROM geraete WHERE geraetegruppe IN ('Tandemwalze', 'Walzenzug', 'Gummiradwalze')

-- Suche nach Seriennummer:
SELECT * FROM geraete WHERE seriennummer ILIKE '%ABC123%' LIMIT 10

-- Geräte mit bestimmter Abgasstufe:
SELECT hersteller, bezeichnung, geraetegruppe
FROM geraete
WHERE eigenschaften_json->>'abgasstufe_eu' ILIKE '%Stufe V%'
LIMIT 20

-- Vergleich zweier Gerätegruppen:
SELECT geraetegruppe,
       COUNT(*) as anzahl,
       ROUND(AVG(CASE WHEN eigenschaften_json->>'gewicht_kg' ~ '^[0-9]+(\\.[0-9]+)?$'
                      AND eigenschaften_json->>'gewicht_kg' NOT IN ('nicht-vorhanden', '')
                 THEN (eigenschaften_json->>'gewicht_kg')::numeric END)) as avg_gewicht_kg,
       ROUND(AVG(CASE WHEN eigenschaften_json->>'motor_leistung_kw' ~ '^[0-9]+(\\.[0-9]+)?$'
                      AND eigenschaften_json->>'motor_leistung_kw' NOT IN ('nicht-vorhanden', '')
                 THEN (eigenschaften_json->>'motor_leistung_kw')::numeric END)) as avg_leistung_kw
FROM geraete
WHERE geraetegruppe IN ('Kettenbagger', 'Mobilbagger')
GROUP BY geraetegruppe
"""

    SYSTEM_PROMPT = f"""Du bist ein SQL-Generator für eine PostgreSQL-Datenbank mit Baumaschinen.

{SCHEMA_INFO}

{SQL_EXAMPLES}

REGELN:
1. Generiere NUR SELECT-Abfragen
2. Verwende IMMER korrekte JSONB-Syntax
3. Bei numerischen Vergleichen: Prüfe IMMER auf gültiges Zahlenformat mit Regex
4. Füge IMMER LIMIT hinzu (max 100)
5. Bei Boolean-Eigenschaften: 'true', 'false', oder 'nicht-vorhanden'
6. Verwende ILIKE für case-insensitive Textsuche
7. Bei AVG/SUM: Filtere 'nicht-vorhanden' und leere Werte aus
8. VERWENDE geraetegruppe ILIKE '%bagger%' für Gerätetypen, NICHT kategorie!
9. FÜGE KEINE Filter für 'verwendung' hinzu, außer explizit angefragt!

KRITISCH:
- Für Gerätetypen (Bagger, Walzen, etc.): IMMER geraetegruppe verwenden!
- kategorie ist oft NULL und unvollständig - NUR als Fallback!
- Keine unnötigen Filter hinzufügen!

WICHTIG:
- Gib das SQL als einzeiligen String zurück
- Keine Kommentare im SQL
- Keine Erklärungen, nur das SQL"""

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "execute_sql",
                "description": "Execute a SQL query against the geraete table",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "The PostgreSQL SELECT query to execute"
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Brief explanation of what this query does"
                        }
                    },
                    "required": ["sql"]
                }
            }
        }
    ]

    def __init__(
        self,
        openai_client: Optional[AsyncOpenAI] = None,
        model: Optional[str] = None,
        postgres: Optional[PostgresService] = None,
        verbose: bool = False
    ):
        super().__init__(verbose=verbose)
        self._agent_type = AgentType.SQL_GENERATOR
        self.client = openai_client or AsyncOpenAI(api_key=config.openai_api_key)

        # Use a fast, non-reasoning model for SQL generation
        # Fall back to config model if not specified
        self.model = model or config.chunking_model or "gpt-4o-mini"
        self.postgres = postgres or postgres_service

    async def _execute(self, context: AgentContext) -> AgentResponse:
        """
        Generate and execute SQL based on the task description.
        """
        # Extract task from context (set by orchestrator)
        task_description = context.metadata.get("sql_task", context.user_query)
        filters = context.metadata.get("sql_filters", {})

        # Build the prompt
        prompt = f"Generiere eine SQL-Abfrage für folgende Anfrage:\n\n{task_description}"

        if filters:
            prompt += f"\n\nZusätzliche Filter: {json.dumps(filters, ensure_ascii=False)}"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        self.log(f"Generating SQL for: {task_description[:50]}...")

        # Call LLM to generate SQL
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.TOOLS,
            tool_choice={"type": "function", "function": {"name": "execute_sql"}}
        )

        message = response.choices[0].message

        if not message.tool_calls:
            return AgentResponse.error_response(
                error="No SQL generated",
                agent_type=self._agent_type
            )

        # Extract SQL from tool call
        tool_call = message.tool_calls[0]
        try:
            args = json.loads(tool_call.function.arguments)
            sql = args.get("sql", "")
            explanation = args.get("explanation", "")
        except json.JSONDecodeError:
            return AgentResponse.error_response(
                error="Invalid SQL response format",
                agent_type=self._agent_type
            )

        if not sql:
            return AgentResponse.error_response(
                error="Empty SQL query",
                agent_type=self._agent_type
            )

        self.log(f"SQL: {sql[:100]}...")

        # Execute the SQL
        try:
            results = self.postgres.execute_dynamic_sql(sql)
            self.log(f"Results: {len(results)} rows")

            # Store results in context for reviewer
            context.sql_results = results

            return AgentResponse.success_response(
                data={
                    "sql": sql,
                    "results": results,
                    "row_count": len(results),
                    "explanation": explanation
                },
                agent_type=self._agent_type,
                reasoning=explanation,
                tool_calls=[{"name": "execute_sql", "sql": sql}],
                sources=[{"type": "postgresql", "query": sql, "row_count": len(results)}]
            )

        except Exception as e:
            self.log(f"SQL execution error: {str(e)}")

            # Try to generate a fallback query
            fallback_result = await self._try_fallback_query(task_description, str(e))
            if fallback_result:
                context.sql_results = fallback_result["results"]
                return AgentResponse.success_response(
                    data=fallback_result,
                    agent_type=self._agent_type,
                    reasoning=f"Fallback query used after error: {str(e)}"
                )

            return AgentResponse.error_response(
                error=f"SQL execution failed: {str(e)}",
                agent_type=self._agent_type
            )

    async def _try_fallback_query(
        self,
        task_description: str,
        error: str
    ) -> Optional[Dict[str, Any]]:
        """Try to generate a simpler fallback query after an error"""
        self.log(f"Attempting fallback query due to: {error}")

        fallback_prompt = f"""Die vorherige SQL-Abfrage ist fehlgeschlagen mit: {error}

Generiere eine EINFACHERE Abfrage für: {task_description}

WICHTIG:
- Verwende eine vereinfachte Version
- Vermeide komplexe JSONB-Operationen wenn möglich
- Stelle sicher, dass alle Spalten existieren"""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": fallback_prompt}
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.TOOLS,
                tool_choice={"type": "function", "function": {"name": "execute_sql"}}
            )

            message = response.choices[0].message
            if message.tool_calls:
                args = json.loads(message.tool_calls[0].function.arguments)
                sql = args.get("sql", "")

                if sql:
                    results = self.postgres.execute_dynamic_sql(sql)
                    self.log(f"Fallback successful: {len(results)} rows")
                    return {
                        "sql": sql,
                        "results": results,
                        "row_count": len(results),
                        "explanation": "Fallback query",
                        "is_fallback": True
                    }
        except Exception as e:
            self.log(f"Fallback also failed: {str(e)}")

        return None
