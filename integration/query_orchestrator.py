#!/usr/bin/env python3
"""
SEMA Query Orchestrator
Smart LLM-based routing between PostgreSQL and Pinecone

Usage:
    from query_orchestrator import QueryOrchestrator
    
    orchestrator = QueryOrchestrator(
        db_config={...},
        pinecone_index=index,
        openai_client=client
    )
    
    result = orchestrator.query("Wie viele Bagger haben wir?")
    print(result.answer)
"""

import json
import os
import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable
import psycopg2
from psycopg2.extras import RealDictCursor


# ============================================================
# QUERY TYPES
# ============================================================

class QueryType(Enum):
    AGGREGATION = "aggregation"      # COUNT, SUM, AVG, MAX, MIN
    FILTER = "filter"                # List with criteria
    SEMANTIC = "semantic"            # Recommendations, similarity
    LOOKUP = "lookup"                # Specific item by name
    COMPARISON = "comparison"        # Compare categories/items


@dataclass
class QueryResult:
    """Result from query orchestrator"""
    query_type: QueryType
    answer: str
    sql_query: Optional[str] = None
    raw_results: Optional[List[Dict]] = None
    source: str = "unknown"  # "postgres" or "pinecone"


# ============================================================
# DATABASE SCHEMA INFO (for LLM context)
# ============================================================

SCHEMA_INFO = """
PostgreSQL Table: geraete (2,395 construction equipment records)

CLASSIFICATION COLUMNS:
- kategorie: 'bagger', 'lader', 'verdichter', 'fertiger', 'fraese', 'kran', 'einbauunterstuetzung', 'transportfahrzeug'
- geraetegruppe: 'Mobilbagger', 'Kettenbagger', 'Minibagger', 'Tandemwalze', 'Walzenzug', 'Radfertiger', 'Kettenfertiger', 'Kaltfräse', etc.
- hersteller: 'Caterpillar', 'Liebherr', 'Bomag', 'Vögele', 'Hamm', 'Dynapac', 'Wirtgen', etc.
- verwendung: 'Vermietung', 'Eigenbedarf', etc.

NUMERIC COLUMNS (for range queries):
- gewicht_kg: Weight in kg (e.g., 5000, 20000, 50000)
- motor_leistung_kw: Engine power in kW (e.g., 55, 129, 250)
- breite_mm, hoehe_mm, laenge_mm: Dimensions in mm
- grabtiefe_mm: Digging depth in mm (excavators)
- arbeitsbreite_mm: Working width in mm (rollers, pavers)
- einbaubreite_max__m: Max paving width in meters (pavers)

BOOLEAN COLUMNS (for feature queries):
- klimaanlage: Air conditioning (true/false)
- allradantrieb: All-wheel drive
- allradlenkung: All-wheel steering
- tiltrotator: Tiltrotator attachment
- hammerhydraulik: Hammer hydraulics
- zentralschmierung: Central lubrication
- dieselpartikelfilter: DPF filter

TEXT COLUMNS (for exact match):
- motor_hersteller: 'Deutz', 'Cummins', 'Kubota', 'Caterpillar', 'Liebherr'
- abgasstufe_eu: 'Stufe III', 'Stufe IV', 'Stufe V'
- schnellwechsler_typ: 'OilQuick', 'Lehnhoff', etc.

ARRAY COLUMN:
- einsatzgebiete: Use cases array ['aushub', 'strassenbau', 'asphaltverdichtung', ...]
"""


# ============================================================
# QUERY ORCHESTRATOR
# ============================================================

class QueryOrchestrator:
    """
    Smart query router using LLM to decide between PostgreSQL and Pinecone
    """
    
    def __init__(
        self,
        db_config: Dict[str, str],
        openai_client: Any,  # OpenAI client
        pinecone_index: Any = None,  # Pinecone index (optional)
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        verbose: bool = False
    ):
        self.db_config = db_config
        self.openai = openai_client
        self.pinecone = pinecone_index
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.verbose = verbose
    
    # ========================================
    # MAIN QUERY METHOD
    # ========================================
    
    def query(self, user_query: str) -> QueryResult:
        """
        Process user query and return result
        
        1. Classify query using LLM
        2. Route to appropriate handler
        3. Generate natural language response
        """
        
        # Step 1: Classify query
        query_type, reasoning = self._classify_query(user_query)
        
        if self.verbose:
            print(f"Query Type: {query_type.value}")
            print(f"Reasoning: {reasoning}")
        
        # Step 2: Route to handler
        if query_type == QueryType.SEMANTIC:
            return self._handle_semantic(user_query)
        else:
            return self._handle_sql(user_query, query_type)
    
    # ========================================
    # QUERY CLASSIFICATION (LLM)
    # ========================================
    
    def _classify_query(self, query: str) -> tuple[QueryType, str]:
        """Use LLM to classify query type"""
        
        response = self.openai.chat.completions.create(
            model=self.llm_model,
            messages=[{
                "role": "system",
                "content": """You are a query classifier for a construction equipment database.

Classify the user's query into ONE of these types:

1. AGGREGATION - Questions about counts, totals, statistics, max/min, averages
   Examples: "Wie viele Bagger?", "Schwerster Bagger?", "Durchschnittliches Gewicht?", "Anzahl Geräte mit Klimaanlage?"
   
2. FILTER - Requests to list/show items matching specific criteria
   Examples: "Zeige alle Bagger mit Klimaanlage", "Geräte über 20 Tonnen", "Caterpillar Maschinen"
   
3. SEMANTIC - Recommendations, best fit, natural language questions about suitability
   Examples: "Was ist gut für Straßenbau?", "Empfehlung für enge Baustellen?", "Welches Gerät für Asphaltverdichtung?"
   
4. LOOKUP - Looking up a specific item by name or identifier
   Examples: "Zeige mir den CAT M317", "Details zum Liebherr A920", "Technische Daten Vögele Super 1800"
   
5. COMPARISON - Comparing categories or specific items
   Examples: "Vergleiche Kettenbagger und Mobilbagger", "Unterschied zwischen Tandemwalze und Walzenzug"

Respond in JSON format:
{"type": "AGGREGATION|FILTER|SEMANTIC|LOOKUP|COMPARISON", "reasoning": "brief explanation"}"""
            }, {
                "role": "user",
                "content": query
            }],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        query_type = QueryType(result["type"].lower())
        reasoning = result.get("reasoning", "")
        
        return query_type, reasoning
    
    # ========================================
    # SQL HANDLER (PostgreSQL)
    # ========================================
    
    def _handle_sql(self, query: str, query_type: QueryType) -> QueryResult:
        """Handle query using PostgreSQL"""
        
        # Generate SQL using LLM
        sql = self._generate_sql(query, query_type)
        
        if self.verbose:
            print(f"Generated SQL: {sql}")
        
        # Execute SQL
        results = self._execute_sql(sql)
        
        # Generate natural language answer
        answer = self._generate_answer(query, query_type, results, sql)
        
        return QueryResult(
            query_type=query_type,
            answer=answer,
            sql_query=sql,
            raw_results=results,
            source="postgres"
        )
    
    def _generate_sql(self, query: str, query_type: QueryType) -> str:
        """Use LLM to generate SQL query"""
        
        response = self.openai.chat.completions.create(
            model="gpt-4o",  # Use stronger model for SQL generation
            messages=[{
                "role": "system",
                "content": f"""You are a PostgreSQL expert. Generate a SQL query for the given question.

{SCHEMA_INFO}

RULES:
1. Use ILIKE for case-insensitive text matching
2. Use IS NOT NULL before numeric comparisons
3. For counts, use COUNT(*) with GROUP BY when appropriate
4. LIMIT results to 20 for list queries
5. For "bagger" queries, use: kategorie = 'bagger' OR geraetegruppe ILIKE '%bagger%'
6. For weight in tons, multiply by 1000 (e.g., 20 Tonnen = 20000 kg)
7. Always ORDER BY meaningful columns

Query type: {query_type.value}

Return ONLY the SQL query, nothing else. No markdown, no explanation."""
            }, {
                "role": "user", 
                "content": query
            }],
            temperature=0
        )
        
        sql = response.choices[0].message.content.strip()
        
        # Clean up any markdown
        sql = re.sub(r'^```sql\n?', '', sql)
        sql = re.sub(r'\n?```$', '', sql)
        sql = sql.strip()
        
        return sql
    
    def _execute_sql(self, sql: str) -> List[Dict]:
        """Execute SQL and return results"""
        
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute(sql)
            results = [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            if self.verbose:
                print(f"SQL Error: {e}")
            results = []
        finally:
            cursor.close()
            conn.close()
        
        return results
    
    # ========================================
    # SEMANTIC HANDLER (Pinecone)
    # ========================================
    
    def _handle_semantic(self, query: str) -> QueryResult:
        """Handle semantic query using Pinecone"""
        
        if not self.pinecone:
            # Fallback to SQL if Pinecone not available
            return self._handle_sql(query, QueryType.FILTER)
        
        # Generate embedding
        embedding_response = self.openai.embeddings.create(
            model=self.embedding_model,
            input=query
        )
        embedding = embedding_response.data[0].embedding
        
        # Query Pinecone
        pinecone_results = self.pinecone.query(
            vector=embedding,
            top_k=5,
            include_metadata=True
        )
        
        results = [match['metadata'] for match in pinecone_results.get('matches', [])]
        
        # Generate answer
        answer = self._generate_semantic_answer(query, results)
        
        return QueryResult(
            query_type=QueryType.SEMANTIC,
            answer=answer,
            raw_results=results,
            source="pinecone"
        )
    
    def _generate_semantic_answer(self, query: str, results: List[Dict]) -> str:
        """Generate answer from semantic search results"""
        
        if not results:
            return "Keine passenden Geräte gefunden."
        
        context = json.dumps(results, ensure_ascii=False, indent=2)
        
        response = self.openai.chat.completions.create(
            model=self.llm_model,
            messages=[{
                "role": "system",
                "content": """Du bist ein Experte für Baumaschinen. Beantworte die Frage basierend auf den Suchergebnissen.
                
Regeln:
- Antworte auf Deutsch
- Sei konkret und nenne spezifische Maschinen
- Erkläre kurz, warum diese Maschinen geeignet sind
- Nutze Markdown für Formatierung"""
            }, {
                "role": "user",
                "content": f"Frage: {query}\n\nGefundene Geräte:\n{context}"
            }],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    # ========================================
    # ANSWER GENERATION
    # ========================================
    
    def _generate_answer(
        self, 
        query: str, 
        query_type: QueryType, 
        results: List[Dict],
        sql: str
    ) -> str:
        """Generate natural language answer from SQL results"""
        
        if not results:
            return "Keine Ergebnisse gefunden für diese Anfrage."
        
        # For simple aggregations, format directly
        if query_type == QueryType.AGGREGATION:
            return self._format_aggregation_results(query, results)
        
        # For other types, use LLM
        context = json.dumps(results[:10], ensure_ascii=False, indent=2)
        
        response = self.openai.chat.completions.create(
            model=self.llm_model,
            messages=[{
                "role": "system",
                "content": """Formatiere die Datenbankergebnisse als hilfreiche deutsche Antwort.

Regeln:
- Nutze Markdown (**, -, etc.)
- Bei Listen: Zeige die wichtigsten Eigenschaften
- Bei Einzelergebnissen: Zeige alle Details
- Sei präzise und übersichtlich"""
            }, {
                "role": "user",
                "content": f"Frage: {query}\n\nErgebnisse ({len(results)} Treffer):\n{context}"
            }],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def _format_aggregation_results(self, query: str, results: List[Dict]) -> str:
        """Format aggregation results without LLM"""
        
        query_lower = query.lower()
        
        # Single count result
        if len(results) == 1:
            row = results[0]
            
            # Simple count
            if 'count' in row or 'anzahl' in row:
                count = row.get('count') or row.get('anzahl')
                
                if 'bagger' in query_lower:
                    return f"Wir haben **{count} Bagger** im Bestand."
                elif 'walze' in query_lower:
                    return f"Wir haben **{count} Walzen** im Bestand."
                elif 'fertiger' in query_lower:
                    return f"Wir haben **{count} Fertiger** im Bestand."
                elif 'klimaanlage' in query_lower:
                    return f"**{count} Geräte** haben eine Klimaanlage."
                else:
                    return f"Anzahl: **{count}**"
            
            # Single item (e.g., heaviest)
            if 'bezeichnung' in row:
                name = f"{row.get('hersteller', '')} {row.get('bezeichnung', '')}".strip()
                weight = row.get('gewicht_kg')
                power = row.get('motor_leistung_kw')
                
                answer = f"**{name}**"
                if weight:
                    answer += f"\n- Gewicht: {weight:,.0f} kg"
                if power:
                    answer += f"\n- Leistung: {power} kW"
                return answer
        
        # Grouped results
        if len(results) > 1 and ('count' in results[0] or 'anzahl' in results[0]):
            total = sum(r.get('count') or r.get('anzahl', 0) for r in results)
            
            lines = [f"**Gesamt: {total}**\n"]
            
            for r in results:
                # Find the label column
                label = (r.get('geraetegruppe') or r.get('kategorie') or 
                        r.get('hersteller') or r.get('abgasstufe_eu') or 'Unbekannt')
                count = r.get('count') or r.get('anzahl', 0)
                lines.append(f"- {label}: {count}")
            
            return "\n".join(lines)
        
        # Default: show as list
        lines = [f"**{len(results)} Ergebnisse:**\n"]
        for r in results[:10]:
            name = f"{r.get('hersteller', '')} {r.get('bezeichnung', '')}".strip()
            lines.append(f"- {name}")
        
        if len(results) > 10:
            lines.append(f"\n... und {len(results) - 10} weitere")
        
        return "\n".join(lines)


# ============================================================
# STANDALONE USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    from openai import OpenAI
    
    # Initialize
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    db_config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "database": os.getenv("DB_NAME", "sema"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres"),
    }
    
    orchestrator = QueryOrchestrator(
        db_config=db_config,
        openai_client=client,
        verbose=True
    )
    
    # Test queries
    test_queries = [
        "Wie viele Bagger haben wir im Bestand?",
        "Zeige alle Mobilbagger mit Klimaanlage",
        "Was ist der schwerste Bagger?",
        "Welches Gerät ist am besten für Straßenbau geeignet?",
        "Vergleiche Kettenbagger und Mobilbagger",
    ]
    
    print("=" * 60)
    print("SEMA Query Orchestrator Test")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 40)
        
        result = orchestrator.query(query)
        
        print(f"Type: {result.query_type.value}")
        print(f"Source: {result.source}")
        if result.sql_query:
            print(f"SQL: {result.sql_query[:100]}...")
        print(f"\n{result.answer}")
        print()
