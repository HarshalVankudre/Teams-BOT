"""
Reviewer Agent (Reasoning Model)
Reviews data from sub-agents and generates natural language responses.
Handles smart display logic, pagination, response formatting, and file exports.
"""
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from openai import AsyncOpenAI

from .base import BaseAgent, AgentContext, AgentResponse, AgentType
from .registry import AgentMetadata, AgentCapability, register_agent
from ..config import config
from ..file_export import file_export_service, ExportResult


# Agent metadata for registration
REVIEWER_AGENT_METADATA = AgentMetadata(
    agent_id="reviewer",
    name="Reviewer Agent",
    description="Reviews collected data and generates natural language responses",
    detailed_description="""Überprüft alle gesammelten Daten und generiert natürliche Antworten.
Dieser Agent wird NACH allen anderen Agenten aufgerufen um:
- Alle Ergebnisse zu analysieren und zusammenzufassen
- Natürlichsprachliche Antworten zu formatieren
- Smart Display Logic anzuwenden (Pagination, Listen)
- Weiterführende Optionen vorzuschlagen""",
    capabilities=[AgentCapability.RESPONSE_GENERATION],
    uses_reasoning=True,
    default_model="gpt-5",
    parameters={},
    example_queries=[],
    priority=0,
    direct_invocation=False  # Only called by agent system, not orchestrator
)


@register_agent(REVIEWER_AGENT_METADATA)
class ReviewerAgent(BaseAgent):
    """
    Final reviewer that generates natural language responses from agent data.
    Uses a reasoning model to think about how to best present the information.
    """

    SYSTEM_PROMPT = """Du bist der RÜKO AI-Assistent. Deine Aufgabe ist es, die gesammelten Daten
in eine KURZE, prägnante Antwort umzuwandeln.

WICHTIGSTE REGEL - KÜRZE:
- Halte Antworten KURZ und auf den Punkt
- Maximal 5-8 Zeilen für einfache Fragen
- Maximal 15-20 Zeilen für komplexe Fragen
- Bei vielen Daten: ZUSAMMENFASSEN, nicht alles auflisten!

DATENPRIORITÄT:
1. Interne Datenbank (PostgreSQL, Pinecone) = HAUPTQUELLE
2. Web-Suche = NUR ergänzend

FORMATIERUNGS-REGELN:

1. **Bei Listen (mehr als 3 Ergebnisse):**
   - Zeige NUR die TOP 3 relevantesten
   - Schreibe: "...und X weitere verfügbar"
   - KEINE langen Auflistungen!

2. **Bei Zählungen:**
   - Ein Satz reicht: "Wir haben X Geräte im Bestand."

3. **Bei Vergleichen:**
   - Kurze Zusammenfassung der Unterschiede
   - Keine ausführlichen Tabellen

4. **Bei Einzelergebnissen:**
   - Nur die wichtigsten 3-5 Eigenschaften zeigen

5. **Bei Prozessen/Anleitungen:**
   - Kurze Zusammenfassung in 3-5 Schritten
   - Details nur auf Nachfrage

6. **Bei Empfehlungen:**
   - Eine klare Empfehlung + kurze Begründung

WEITERFÜHRENDE OPTIONEN (kurz halten):
```
💡 Weiterführende Optionen:
• [Option 1]
• [Option 2]
```

SPRACHE:
- Deutsch, professionell, direkt
- Keine Floskeln oder Füllwörter
- Kurze Sätze bevorzugen

WICHTIG:
- Erfinde NIEMALS Daten
- FASSE ZUSAMMEN statt aufzulisten
- Weniger ist mehr!"""

    def __init__(
        self,
        openai_client: Optional[AsyncOpenAI] = None,
        model: Optional[str] = None,
        reasoning_effort: str = "medium",
        verbose: bool = False
    ):
        super().__init__(verbose=verbose)
        self._agent_type = AgentType.REVIEWER
        self.client = openai_client or AsyncOpenAI(api_key=config.openai_api_key)

        # Use configured model or fall back to config
        self.model = model or config.response_model
        self.reasoning_effort = reasoning_effort

    def _supports_reasoning(self) -> bool:
        """Check if the model supports reasoning parameter via responses API"""
        if not self.model:
            return False
        model_lower = self.model.lower()
        # Only o1 and o3 series use responses API with reasoning
        # GPT-5 uses standard chat completions with max_completion_tokens
        return model_lower.startswith('o1') or model_lower.startswith('o3')

    def _uses_max_completion_tokens(self) -> bool:
        """Check if model uses max_completion_tokens instead of max_tokens"""
        if not self.model:
            return False
        model_lower = self.model.lower()
        # GPT-5 and o-series use max_completion_tokens
        return (model_lower.startswith('gpt-5') or
                model_lower.startswith('o1') or
                model_lower.startswith('o3'))

    def _build_conversation_context(self, context: AgentContext) -> str:
        """Build conversation history string for context-aware responses"""
        if not context.conversation_history:
            return ""

        # Use last 6 messages (3 exchanges) for reviewer context
        recent_history = context.conversation_history[-6:]
        if not recent_history:
            return ""

        history_lines = []
        for entry in recent_history:
            role = "Benutzer" if entry.get("role") == "user" else "Assistent"
            content = entry.get("content", "")[:500]  # Allow longer content for reviewer
            history_lines.append(f"{role}: {content}")

        return "VORHERIGE KONVERSATION:\n" + "\n".join(history_lines) + "\n\n"

    def _detect_export_request(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Detect if the query is requesting a file export.

        Returns:
            Tuple of (is_export_request, format) where format is 'excel', 'pdf', or None
        """
        query_lower = query.lower()

        # Excel keywords
        excel_keywords = ['excel', 'xlsx', '.xlsx', 'spreadsheet', 'tabelle']
        # PDF keywords
        pdf_keywords = ['pdf', '.pdf']
        # General export keywords
        export_keywords = ['exportier', 'export', 'herunterladen', 'download', 'datei',
                          'als datei', 'speichern', 'csv']

        # Check for explicit format requests
        for kw in excel_keywords:
            if kw in query_lower:
                return True, 'excel'

        for kw in pdf_keywords:
            if kw in query_lower:
                return True, 'pdf'

        # Check for general export request (default to Excel)
        for kw in export_keywords:
            if kw in query_lower:
                return True, 'excel'  # Default to Excel

        return False, None

    def _get_export_data(self, context: AgentContext) -> List[Dict[str, Any]]:
        """Extract exportable data from context"""
        data = []

        # Get SQL results
        if context.sql_results:
            for item in context.sql_results:
                if '_results' in item:
                    data.extend(item['_results'])
                else:
                    data.append(item)

        return data

    async def _handle_export(self, context: AgentContext, export_format: str) -> AgentResponse:
        """Handle file export request"""
        self.log(f"Export request detected: {export_format}")

        # Get data to export
        data = self._get_export_data(context)

        if not data:
            return AgentResponse.success_response(
                data={"response": "Keine Daten zum Exportieren gefunden. Bitte zuerst eine Suchanfrage stellen."},
                agent_type=self._agent_type,
                reasoning="No data for export"
            )

        # Generate title from query context
        title = "RUEKO Geraete-Export"
        if context.reasoning:
            # Try to extract a meaningful title
            title = f"RUEKO Export - {len(data)} Datensaetze"

        # Generate the file
        if export_format == 'pdf':
            result = file_export_service.export_to_pdf(data, title=title)
            format_name = "PDF"
        else:
            result = file_export_service.export_to_excel(data, title=title)
            format_name = "Excel"

        if not result.success:
            return AgentResponse.success_response(
                data={"response": f"Export fehlgeschlagen: {result.error}"},
                agent_type=self._agent_type,
                reasoning="Export failed"
            )

        # Build response with file attachment
        response_text = f"Hier ist dein {format_name}-Export mit {result.record_count} Datensaetzen."

        self.log(f"Export successful: {result.file_name}, {len(result.file_data)} bytes")

        return AgentResponse.success_response(
            data={
                "response": response_text,
                "file_export": {
                    "file_name": result.file_name,
                    "mime_type": result.mime_type,
                    "base64_data": result.base64_data,
                    "record_count": result.record_count
                }
            },
            agent_type=self._agent_type,
            reasoning=f"Generated {format_name} export with {result.record_count} records"
        )

    async def _execute(self, context: AgentContext) -> AgentResponse:
        """
        Generate a natural language response from all collected data.
        Also handles file export requests.
        """
        # Check for export request
        is_export, export_format = self._detect_export_request(context.user_query)

        if is_export:
            return await self._handle_export(context, export_format)

        # Build context from all agent results
        data_context = self._build_data_context(context)

        # Build conversation history context
        conversation_context = self._build_conversation_context(context)

        # Build the review prompt
        prompt = f"""Analysiere die folgenden Daten und erstelle eine hilfreiche Antwort.

{conversation_context}AKTUELLE FRAGE:
{context.user_query}

ORCHESTRATOR-ANALYSE:
{context.reasoning or 'Keine Analyse verfügbar'}

{data_context}

AUFGABE:
1. Berücksichtige den Konversationsverlauf bei der Antwort
2. Analysiere alle gesammelten Daten
3. Erstelle eine natürliche, hilfreiche Antwort
4. Formatiere die Antwort übersichtlich
5. Füge weiterführende Optionen am Ende hinzu

WICHTIG: Wenn sich die Frage auf vorherige Antworten bezieht (z.B. "davon", "diese", "welche davon"),
beziehe dich auf die vorherige Konversation!

Antworte jetzt:"""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        # For GPT-5 and reasoning models, use the responses API
        if self._supports_reasoning():
            return await self._call_with_reasoning(messages)

        # Standard chat completion for legacy models
        request_params = {
            "model": self.model,
            "messages": messages,
        }

        # Use correct token parameter based on model
        # GPT-5 uses reasoning tokens internally, so we need more headroom
        # to ensure output tokens are available after reasoning
        if self._uses_max_completion_tokens():
            request_params["max_completion_tokens"] = 4000  # Extra room for reasoning + output
        else:
            request_params["max_tokens"] = 1500

        self.log(f"Calling model: {self.model}")

        response = await self.client.chat.completions.create(**request_params)
        response_text = response.choices[0].message.content

        self.log(f"Response length: {len(response_text) if response_text else 0}")

        if not response_text:
            self.log(f"WARNING: Empty response from model")
            self.log(f"Full response: {response}")

        return AgentResponse.success_response(
            data={"response": response_text or "Keine Antwort vom Modell erhalten."},
            agent_type=self._agent_type,
            reasoning="Standard completion"
        )

    async def _call_with_reasoning(self, messages: List[Dict]) -> AgentResponse:
        """Call the reasoning API for models that support it"""
        try:
            # Convert messages to input format for responses API
            input_messages = []
            for msg in messages:
                input_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

            response = await self.client.responses.create(
                model=self.model,
                input=input_messages,
                reasoning={"effort": self.reasoning_effort},
                max_output_tokens=1500  # Concise responses
            )

            return AgentResponse.success_response(
                data={"response": response.output_text},
                agent_type=self._agent_type,
                reasoning=f"Reasoning effort: {self.reasoning_effort}"
            )

        except Exception as e:
            self.log(f"Reasoning API error: {str(e)}, falling back to chat")
            # Fallback to standard completion with correct token param
            request_params = {
                "model": self.model,
                "messages": messages,
            }
            if self._uses_max_completion_tokens():
                request_params["max_completion_tokens"] = 1500
            else:
                request_params["max_tokens"] = 1500

            response = await self.client.chat.completions.create(**request_params)
            return AgentResponse.success_response(
                data={"response": response.choices[0].message.content},
                agent_type=self._agent_type,
                reasoning="Fallback to standard completion"
            )

    def _build_data_context(self, context: AgentContext) -> str:
        """Build context string from all collected data"""
        sections = []

        # Count how many data sources have results (for dynamic limiting)
        source_count = sum([
            1 if context.sql_results else 0,
            1 if context.pinecone_results else 0,
            1 if context.web_results else 0
        ])
        # More sources = more aggressive limiting per source
        self._multi_source = source_count > 1

        # SQL Results
        if context.sql_results:
            sql_section = self._format_sql_results(context.sql_results)
            sections.append(f"## DATENBANK-ERGEBNISSE (PostgreSQL):\n{sql_section}")

        # Pinecone Results
        if context.pinecone_results:
            pinecone_section = self._format_pinecone_results(context.pinecone_results)
            sections.append(f"## SEMANTISCHE SUCHE (Pinecone):\n{pinecone_section}")

        # Web Results
        if context.web_results:
            web_section = self._format_web_results(context.web_results)
            sections.append(f"## WEB-ERGEBNISSE (ergänzend):\n{web_section}")

        if not sections:
            return "KEINE DATEN GEFUNDEN - Bitte dem Benutzer mitteilen."

        return "\n\n".join(sections)

    def _format_sql_results(self, results: List[Dict[str, Any]]) -> str:
        """Format SQL results for the prompt"""
        if not results:
            return "Keine Ergebnisse"

        # Dynamic limit based on multi-source query
        max_items = 8 if getattr(self, '_multi_source', False) else 20

        # Check if results have query context (new format)
        if results and isinstance(results[0], dict) and "_query_context" in results[0]:
            # Multiple query results with context
            sections = []
            for query_result in results:
                query_context = query_result.get("_query_context", "Unbekannte Abfrage")
                result_count = query_result.get("_result_count", 0)
                actual_results = query_result.get("_results", [])

                section = f"### Abfrage: {query_context}\n"
                section += f"**{result_count} Ergebnis(se)**\n\n"

                # Format actual results with dynamic limit
                formatted = []
                for i, row in enumerate(actual_results[:max_items], 1):
                    cleaned = {k: v for k, v in row.items() if v is not None}
                    formatted.append(f"{i}. {json.dumps(cleaned, ensure_ascii=False, default=str)}")

                if len(actual_results) > max_items:
                    formatted.append(f"... und {len(actual_results) - max_items} weitere Ergebnisse")

                section += "\n".join(formatted)
                sections.append(section)

            return "\n\n".join(sections)

        # Old format - flat list of results
        summary = f"**{len(results)} Datensätze gefunden**\n\n"

        # Format results with dynamic limit
        formatted_results = []
        for i, row in enumerate(results[:max_items], 1):
            # Filter out None values and format
            cleaned = {k: v for k, v in row.items() if v is not None}
            formatted_results.append(f"{i}. {json.dumps(cleaned, ensure_ascii=False, default=str)}")

        if len(results) > max_items:
            formatted_results.append(f"... und {len(results) - max_items} weitere")

        return summary + "\n".join(formatted_results)

    def _format_pinecone_results(self, results: List[Dict[str, Any]]) -> str:
        """Format Pinecone results for the prompt"""
        if not results:
            return "Keine Ergebnisse"

        # Dynamic limit based on multi-source query
        max_items = 3 if getattr(self, '_multi_source', False) else 10
        content_limit = 150 if getattr(self, '_multi_source', False) else 300

        summary = f"**{len(results)} Ergebnisse gefunden**\n\n"

        formatted_results = []
        for i, result in enumerate(results[:max_items], 1):
            score = result.get("score", 0)
            title = result.get("title", "Unbekannt")
            content = result.get("content", "")[:content_limit]

            formatted_results.append(f"""
{i}. **{title}** (Relevanz: {score:.2%})
{content}
""")

        if len(results) > max_items:
            formatted_results.append(f"... und {len(results) - max_items} weitere")

        return summary + "\n".join(formatted_results)

    def _format_web_results(self, results: List[Dict[str, Any]]) -> str:
        """Format web results for the prompt"""
        if not results:
            return "Keine Web-Ergebnisse"

        # Dynamic limit based on multi-source query
        max_items = 2 if getattr(self, '_multi_source', False) else 5
        content_limit = 100 if getattr(self, '_multi_source', False) else 200

        summary = f"**{len(results)} Web-Ergebnisse** (nur ergänzend verwenden)\n\n"

        formatted_results = []
        for i, result in enumerate(results[:max_items], 1):
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", "")[:content_limit]

            formatted_results.append(f"""
{i}. **{title}**
URL: {url}
{content}
""")

        if len(results) > max_items:
            formatted_results.append(f"... und {len(results) - max_items} weitere")

        return summary + "\n".join(formatted_results)


class ResponseFormatter:
    """
    Static utility class for formatting responses in different styles.
    Can be used by the reviewer or directly by the agent system.
    """

    @staticmethod
    def format_equipment_list(
        items: List[Dict[str, Any]],
        max_display: int = 5,
        show_weight: bool = True,
        show_power: bool = True
    ) -> str:
        """Format a list of equipment for display"""
        if not items:
            return "Keine Geräte gefunden."

        lines = []
        displayed = items[:max_display]

        for i, item in enumerate(displayed, 1):
            parts = []

            # Basic info
            hersteller = item.get("hersteller", "")
            bezeichnung = item.get("bezeichnung", item.get("title", ""))
            if hersteller:
                parts.append(f"**{hersteller} {bezeichnung}**")
            else:
                parts.append(f"**{bezeichnung}**")

            # Technical specs
            specs = []
            if show_weight and item.get("gewicht_kg"):
                specs.append(f"{item['gewicht_kg']} kg")
            if show_power and item.get("motor_leistung_kw"):
                specs.append(f"{item['motor_leistung_kw']} kW")

            if specs:
                parts.append(f"({', '.join(specs)})")

            lines.append(f"{i}. " + " ".join(parts))

        # Add "more" notice
        remaining = len(items) - max_display
        if remaining > 0:
            lines.append(f"\n📋 **{remaining} weitere Ergebnisse verfügbar**")

        return "\n".join(lines)

    @staticmethod
    def format_comparison(
        groups: Dict[str, Dict[str, Any]],
        metrics: List[str] = None
    ) -> str:
        """Format a comparison between groups"""
        if not groups:
            return "Keine Vergleichsdaten verfügbar."

        metrics = metrics or ["anzahl", "avg_gewicht_kg", "avg_leistung_kw"]

        lines = ["**Vergleich:**\n"]

        for group_name, data in groups.items():
            lines.append(f"**{group_name}:**")
            for metric in metrics:
                if metric in data and data[metric] is not None:
                    value = data[metric]
                    if isinstance(value, float):
                        value = f"{value:,.0f}"
                    metric_label = {
                        "anzahl": "Anzahl",
                        "avg_gewicht_kg": "Ø Gewicht (kg)",
                        "avg_leistung_kw": "Ø Leistung (kW)"
                    }.get(metric, metric)
                    lines.append(f"  - {metric_label}: {value}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def format_single_equipment(item: Dict[str, Any]) -> str:
        """Format detailed view of a single equipment"""
        lines = []

        # Header
        hersteller = item.get("hersteller", "")
        bezeichnung = item.get("bezeichnung", item.get("title", "Unbekannt"))
        lines.append(f"## {hersteller} {bezeichnung}\n")

        # Classification
        if item.get("kategorie") or item.get("geraetegruppe"):
            lines.append("**Klassifikation:**")
            if item.get("kategorie"):
                lines.append(f"- Kategorie: {item['kategorie']}")
            if item.get("geraetegruppe"):
                lines.append(f"- Gerätegruppe: {item['geraetegruppe']}")
            lines.append("")

        # IDs
        if item.get("seriennummer") or item.get("inventarnummer"):
            lines.append("**Identifikation:**")
            if item.get("seriennummer"):
                lines.append(f"- Seriennummer: {item['seriennummer']}")
            if item.get("inventarnummer"):
                lines.append(f"- Inventarnummer: {item['inventarnummer']}")
            lines.append("")

        # Technical specs - check direct columns, prop_* columns, and eigenschaften JSONB
        has_specs = False
        spec_lines = []

        # Direct numeric columns (base columns)
        direct_specs = [
            ("gewicht_kg", "Gewicht", "kg"),
            ("motor_leistung_kw", "Motorleistung", "kW"),
        ]
        for key, label, unit in direct_specs:
            val = item.get(key)
            if val and val not in ["nicht-vorhanden", ""]:
                spec_lines.append(f"- {label}: {val} {unit}")
                has_specs = True

        # Property columns (prop_* format - values include units)
        prop_specs = [
            ("prop_arbeitsbreite", "Arbeitsbreite"),
            ("prop_transportbreite", "Transportbreite"),
            ("prop_transportlaenge", "Transportlaenge"),
            ("prop_einsatzgewicht", "Einsatzgewicht"),
            ("prop_dienstgewicht", "Dienstgewicht"),
            ("prop_nutzlast", "Nutzlast"),
            ("prop_loeffelinhalt", "Loeffelinhalt"),
            ("prop_reisstiefe", "Reisstiefe"),
            ("prop_ausladung", "Ausladung"),
            ("prop_fahrgeschwindigkeit", "Fahrgeschwindigkeit"),
            ("prop_hubkraft", "Hubkraft"),
            ("prop_motor_hersteller", "Motor-Hersteller"),
            ("prop_motor_typ", "Motor-Typ"),
            ("prop_abgasstufe_eu", "Abgasstufe EU"),
        ]
        for key, label in prop_specs:
            val = item.get(key)
            if val and val not in ["nicht-vorhanden", "", "Nein"]:
                spec_lines.append(f"- {label}: {val}")
                has_specs = True

        # Fallback: JSONB eigenschaften column (backwards compatibility)
        props = item.get("eigenschaften", {})
        if isinstance(props, str):
            try:
                props = json.loads(props)
            except:
                props = {}
        if props and isinstance(props, dict):
            for key, value in list(props.items())[:10]:  # Limit to first 10
                if isinstance(value, dict):
                    val = value.get("wert")
                else:
                    val = value
                if val and val not in ["nicht-vorhanden", "", "Nein"]:
                    spec_lines.append(f"- {key}: {val}")
                    has_specs = True

        if has_specs:
            lines.append("**Technische Daten:**")
            lines.extend(spec_lines)

        # Boolean features - check direct columns, prop_* columns, and eigenschaften
        features = []

        # Direct boolean columns
        if item.get("klimaanlage") is True:
            features.append("Klimaanlage")
        if item.get("zentralschmierung") is True:
            features.append("Zentralschmierung")

        # Property columns with Ja/Nein values
        prop_features = [
            ("prop_hammerhydraulik", "Hammerhydraulik"),
            ("prop_schnellwechsler_hydr", "Schnellwechsler"),
            ("prop_oszillation", "Oszillation"),
            ("prop_tiltrotator", "Tiltrotator"),
            ("prop_gps", "GPS"),
            ("prop_rueckfahrkamera", "Rueckfahrkamera"),
        ]
        for key, label in prop_features:
            val = item.get(key)
            if val and val == "Ja":
                if label not in features:
                    features.append(label)

        # Fallback: Check eigenschaften JSONB for boolean features
        if props and isinstance(props, dict):
            jsonb_features = {
                "Klimaanlage": "Klimaanlage",
                "Hammerhydraulik": "Hammerhydraulik",
                "Schnellwechsler Hydr": "Schnellwechsler",
                "Oszillation": "Oszillation",
                "Tiltrotator": "Tiltrotator",
            }
            for key, label in jsonb_features.items():
                prop = props.get(key, {})
                if isinstance(prop, dict) and prop.get("wert") == "Ja":
                    if label not in features:
                        features.append(label)

        if features:
            lines.append(f"\n**Ausstattung:** {', '.join(features)}")

        return "\n".join(lines)

    @staticmethod
    def add_follow_up_options(
        response: str,
        options: List[str]
    ) -> str:
        """Add follow-up options to the end of a response"""
        if not options:
            return response

        follow_up = "\n\n💡 **Weiterführende Optionen:**\n"
        for option in options:
            follow_up += f"• \"{option}\"\n"

        return response + follow_up
