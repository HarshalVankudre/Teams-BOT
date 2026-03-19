#!/usr/bin/env python3
"""
Teams Bot RAG Test Suite (Simplified)
Testing framework for LangGraph agent validation.
"""
import os
import sys
import json
import time
import asyncio
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

# Fix Windows console encoding for Unicode
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()


class TestStatus(Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


@dataclass
class TestCase:
    """Individual test case definition"""
    id: str
    name: str
    query: str
    category: str = "general"
    pipeline: str = "langgraph"
    reset_context: bool = False
    expected_agent: Optional[str] = None
    expected_tools: List[str] = field(default_factory=list)
    expected_tools_any: List[str] = field(default_factory=list)
    expected_keywords: List[str] = field(default_factory=list)
    forbidden_keywords: List[str] = field(default_factory=list)
    min_results: int = 0
    timeout_ms: int = 120000


@dataclass
class TestResult:
    """Result of a single test execution"""
    test_id: str
    test_name: str
    status: TestStatus
    duration_ms: int
    query: str = ""
    response: str = ""
    error: str = ""
    agent: str = ""
    tools_used: List[str] = field(default_factory=list)
    sql_rows: int = 0
    assertions_passed: int = 0
    assertions_failed: int = 0

    @property
    def passed(self) -> bool:
        return self.status == TestStatus.PASSED


@dataclass
class TestSuite:
    """Collection of test results"""
    name: str
    started_at: str
    ended_at: str = ""
    results: List[TestResult] = field(default_factory=list)
    
    @property
    def total(self) -> int:
        return len(self.results)
    
    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)
    
    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)
    
    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.ERROR)
    
    @property
    def duration_ms(self) -> int:
        return sum(r.duration_ms for r in self.results)
    
    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0


class TestRunner:
    """Test runner for LangGraph RAG pipeline"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.suite = TestSuite(name="RAG Pipeline Tests", started_at=datetime.now().isoformat())
        self.conversation_history: List[Dict] = []
        self.thread_key = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._init_services()
    
    def _log(self, msg: str, color: str = ""):
        if self.verbose:
            print(f"{color}{msg}{Colors.RESET}" if color else msg)
    
    def _init_services(self):
        """Initialize all required services"""
        print(f"\n{Colors.BOLD}Initializing Test Environment{Colors.RESET}")
        print("-" * 50)
        
        self.config = self.postgres = self.pinecone = self.langgraph_agent = self.rag_search = None
        
        # Config
        try:
            from rag.config import config
            self.config = config
            model_display = f"Gemini advisory={config.advisory_model}, LangGraph={config.langgraph_model}"
            print(f"  Config       {Colors.GREEN}OK{Colors.RESET}  {model_display}")
        except Exception as e:
            print(f"  Config       {Colors.RED}FAIL{Colors.RESET}  {e}")
        
        # PostgreSQL
        try:
            from rag.postgres import PostgresService
            self.postgres = PostgresService()
            if self.postgres.available:
                # Get record count
                try:
                    result = self.postgres.execute_query("SELECT COUNT(*) as cnt FROM public.equipment_matrix_v2")
                    count = result[0]['cnt'] if result else 0
                    print(f"  PostgreSQL   {Colors.GREEN}OK{Colors.RESET}  {self.postgres.equipment_table}, {count} equipment records ({self.postgres.equipment_table})")
                except:
                    print(f"  PostgreSQL   {Colors.GREEN}OK{Colors.RESET}  {self.postgres.equipment_table}")
            else:
                print(f"  PostgreSQL   {Colors.YELLOW}WARN{Colors.RESET}  {self.postgres.availability_error}")
        except Exception as e:
            print(f"  PostgreSQL   {Colors.RED}FAIL{Colors.RESET}  {e}")
        
        # Pinecone
        try:
            from rag.vector_store import PineconeStore
            self.pinecone = PineconeStore()
            print(f"  Pinecone     {Colors.GREEN}OK{Colors.RESET}")
        except Exception as e:
            print(f"  Pinecone     {Colors.YELLOW}WARN{Colors.RESET}  {e}")
        
        # LangGraph Agent (only agent now)
        try:
            from rag.langgraph_agent import get_langgraph_agent, set_shared_postgres, set_shared_pinecone
            if self.postgres:
                set_shared_postgres(self.postgres)
            if self.pinecone:
                set_shared_pinecone(self.pinecone)
            self.langgraph_agent = get_langgraph_agent()
            print(f"  LangGraph    {Colors.GREEN}OK{Colors.RESET}  (Priority 1)")
        except Exception as e:
            print(f"  LangGraph    {Colors.RED}FAIL{Colors.RESET}  {e}")

        # Full RAG pipeline
        try:
            from rag.search import RAGSearch
            self.rag_search = RAGSearch()
            print(f"  RAG Search   {Colors.GREEN}OK{Colors.RESET}  (Full routing)")
        except Exception as e:
            print(f"  RAG Search   {Colors.YELLOW}WARN{Colors.RESET}  {e}")
        
        print("-" * 50)
    
    async def run_test(self, test: TestCase) -> TestResult:
        """Execute a single test case"""
        start = time.time()
        if test.reset_context:
            self.clear_context()
        result = TestResult(
            test_id=test.id,
            test_name=test.name,
            status=TestStatus.PASSED,
            duration_ms=0,
            query=test.query
        )
        
        try:
            response = await self._execute_query(test.query, pipeline=test.pipeline)
            result.duration_ms = int((time.time() - start) * 1000)
            result.response = response.get("response", "")
            result.agent = response.get("agent", "")
            result.tools_used = response.get("tools_used", [])
            result.sql_rows = response.get("sql_rows", 0)
            
            # Check assertions
            if test.expected_agent:
                if result.agent == test.expected_agent:
                    result.assertions_passed += 1
                else:
                    result.assertions_failed += 1
                    result.status = TestStatus.FAILED

            if test.expected_tools:
                for tool in test.expected_tools:
                    if tool in result.tools_used:
                        result.assertions_passed += 1
                    else:
                        result.assertions_failed += 1
                        result.status = TestStatus.FAILED

            if test.expected_tools_any:
                if any(tool in result.tools_used for tool in test.expected_tools_any):
                    result.assertions_passed += 1
                else:
                    result.assertions_failed += 1
                    result.status = TestStatus.FAILED

            if test.expected_keywords:
                response_lower = result.response.lower()
                for kw in test.expected_keywords:
                    if kw.lower() in response_lower:
                        result.assertions_passed += 1
                    else:
                        result.assertions_failed += 1
                        result.status = TestStatus.FAILED

            if test.forbidden_keywords:
                response_lower = result.response.lower()
                for kw in test.forbidden_keywords:
                    if kw.lower() in response_lower:
                        result.assertions_failed += 1
                        result.status = TestStatus.FAILED
                    else:
                        result.assertions_passed += 1
            
        except Exception as e:
            result.status = TestStatus.ERROR
            result.error = str(e)
            result.duration_ms = int((time.time() - start) * 1000)
        
        return result
    
    async def _execute_query(self, query: str, pipeline: str = "langgraph") -> Dict[str, Any]:
        """Execute query through LangGraph or the full RAG pipeline."""
        if pipeline == "full":
            if not self.rag_search:
                raise RuntimeError("RAGSearch not available")

            print(f"\n{'='*60}")
            print(f"🤖 FULL RAG PROCESSING: {query}")
            print(f"🧵 Thread: {self.thread_key}")
            print(f"{'='*60}\n")

            result = await self.rag_search.search_and_generate(
                query=query,
                thread_key=self.thread_key,
            )

            response_text = result.get("response", "")
            tools_used = result.get("agents_used", [])
            processing_time = result.get("execution_time_ms", 0)
            agent = result.get("agent", "")
            sql_rows = 0
        else:
            if not self.langgraph_agent:
                raise RuntimeError("LangGraph agent not available")

            print(f"\n{'='*60}")
            print(f"🤖 LANGGRAPH PROCESSING: {query}")
            print(f"🧵 Thread: {self.thread_key}")
            print(f"{'='*60}\n")
        
            result = await self.langgraph_agent.process(
                user_query=query,
                thread_key=self.thread_key,
                conversation_history=self.conversation_history
            )
        
            # Handle both dict and AgentResult object
            if hasattr(result, 'response'):
                # AgentResult object
                response_text = result.response or ""
                tools_used = result.tools_used or []
                processing_time = getattr(
                    result,
                    'processing_time_ms',
                    getattr(result, 'execution_time_ms', 0),
                )
                sql_rows = getattr(result, 'sql_rows', 0)
                agent = "langgraph"
            else:
                # Dictionary
                response_text = result.get("response", "")
                tools_used = result.get("tools_used", [])
                processing_time = result.get("processing_time_ms", 0)
                sql_rows = result.get("sql_rows", 0)
                agent = result.get("agent", "langgraph")
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        # Keep last 6 messages
        if len(self.conversation_history) > 6:
            self.conversation_history = self.conversation_history[-6:]
        
        print(f"\n{'='*60}")
        print(f"✅ RESPONSE READY ({processing_time}ms)")
        print(f"🤖 Agent: {agent}")
        print(f"🔧 Tools used: {tools_used}")
        print(f"📝 Response: {response_text[:200]}..." if len(response_text) > 200 else f"📝 Response: {response_text}")
        print(f"{'='*60}\n")
        
        return {
            "response": response_text,
            "agent": agent,
            "tools_used": tools_used,
            "sql_rows": sql_rows
        }
    
    async def test_query(self, query: str) -> TestResult:
        """Run a single ad-hoc query test"""
        test = TestCase(id="adhoc", name="Ad-hoc Query", query=query)
        result = await self.run_test(test)
        self.suite.results.append(result)
        
        status_color = Colors.GREEN if result.passed else Colors.RED
        print(f"  [{status_color}{result.status.value}{Colors.RESET}] {test.name} ({result.duration_ms}ms)")
        
        if result.response:
            print(f"\n{Colors.BOLD}Response:{Colors.RESET}")
            print(result.response)

        if result.agent:
            print(f"\n{Colors.BOLD}Agent:{Colors.RESET}")
            print(result.agent)

        if result.tools_used:
            print(f"\n{Colors.BOLD}Tools Used:{Colors.RESET}")
            for tool in result.tools_used:
                print(f"  🔧 {tool}")
        
        if result.error:
            print(f"\n{Colors.RED}Error:{Colors.RESET} {result.error}")
        
        return result
    
    async def run_suite(self, tests: List[TestCase]) -> TestSuite:
        """Run a full test suite"""
        print(f"\n{Colors.BOLD}Running Test Suite: {len(tests)} tests{Colors.RESET}")
        print("=" * 60)
        
        for test in tests:
            result = await self.run_test(test)
            self.suite.results.append(result)
            
            status_color = Colors.GREEN if result.passed else Colors.RED
            print(f"  [{status_color}{result.status.value}{Colors.RESET}] {test.name} ({result.duration_ms}ms)")
            
            if result.error:
                print(f"    {Colors.RED}Error: {result.error}{Colors.RESET}")
        
        self.suite.ended_at = datetime.now().isoformat()
        return self.suite
    
    def print_summary(self) -> int:
        """Print test summary and return exit code"""
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}Test Summary{Colors.RESET}")
        print(f"{'='*60}")
        
        print(f"  Total:    {self.suite.total}")
        print(f"  {Colors.GREEN}Passed:   {self.suite.passed}{Colors.RESET}")
        print(f"  {Colors.RED}Failed:   {self.suite.failed}{Colors.RESET}")
        print(f"  {Colors.YELLOW}Errors:   {self.suite.errors}{Colors.RESET}")
        print(f"  Duration: {self.suite.duration_ms}ms")
        print(f"  Pass Rate: {self.suite.pass_rate:.1f}%")
        
        return 0 if self.suite.failed == 0 and self.suite.errors == 0 else 1
    
    def export_results(self, filepath: str = "test_results.json"):
        """Export results to JSON"""
        data = {
            "suite": self.suite.name,
            "started_at": self.suite.started_at,
            "ended_at": self.suite.ended_at,
            "summary": {
                "total": self.suite.total,
                "passed": self.suite.passed,
                "failed": self.suite.failed,
                "errors": self.suite.errors,
                "pass_rate": self.suite.pass_rate
            },
            "results": [asdict(r) for r in self.suite.results]
        }
        
        # Convert enums to strings
        for r in data["results"]:
            r["status"] = r["status"].value if hasattr(r["status"], "value") else str(r["status"])
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Results exported to: {filepath}")
    
    def test_sql(self, sql: str) -> Dict:
        """Run direct SQL test"""
        print(f"\n{Colors.BOLD}SQL Test{Colors.RESET}")
        print("-" * 50)
        
        if not self.postgres or not self.postgres.available:
            print(f"  {Colors.RED}[FAIL]{Colors.RESET} PostgreSQL not available")
            return {"error": "PostgreSQL not available"}
        
        start = time.time()
        try:
            results = self.postgres.execute_query(sql)
            duration = int((time.time() - start) * 1000)
            print(f"  {Colors.GREEN}[PASS]{Colors.RESET} {len(results)} rows in {duration}ms")
            
            if results:
                print(f"\n{Colors.BOLD}Results (first 5):{Colors.RESET}")
                for i, row in enumerate(results[:5], 1):
                    print(f"  {i}. {row}")
            
            return {"success": True, "rows": len(results), "duration_ms": duration, "data": results[:10]}
        except Exception as e:
            print(f"  {Colors.RED}[FAIL]{Colors.RESET} {e}")
            return {"error": str(e)}
    
    async def test_search(self, query: str) -> Dict:
        """Run document search test"""
        print(f"\n{Colors.BOLD}Search Test{Colors.RESET}")
        print("-" * 50)
        
        if not self.pinecone:
            print(f"  {Colors.RED}[FAIL]{Colors.RESET} Pinecone not available")
            return {"error": "Pinecone not available"}
        
        start = time.time()
        try:
            results = await self.pinecone.search(query, top_k=5)
            duration = int((time.time() - start) * 1000)
            print(f"  {Colors.GREEN}[PASS]{Colors.RESET} {len(results)} results in {duration}ms")
            
            if results:
                print(f"\n{Colors.BOLD}Results:{Colors.RESET}")
                for i, r in enumerate(results, 1):
                    title = r.get("title", r.get("metadata", {}).get("title", "Unknown"))
                    score = r.get("score", 0)
                    print(f"  {i}. {title} ({score:.1%})")
            
            return {"success": True, "count": len(results), "duration_ms": duration}
        except Exception as e:
            print(f"  {Colors.RED}[FAIL]{Colors.RESET} {e}")
            return {"error": str(e)}
    
    def show_stats(self):
        """Show database statistics"""
        print(f"\n{Colors.BOLD}Database Statistics{Colors.RESET}")
        print("-" * 50)
        
        if not self.postgres or not self.postgres.available:
            print(f"  {Colors.RED}PostgreSQL not available{Colors.RESET}")
            return
        
        try:
            # Total count
            result = self.postgres.execute_query("SELECT COUNT(*) as cnt FROM public.equipment_matrix_v2")
            total = result[0]['cnt'] if result else 0
            print(f"  Total Equipment: {total}")
            
            # By category
            result = self.postgres.execute_query("""
                SELECT geraetegruppe_name as category, COUNT(*) as cnt 
                FROM public.equipment_matrix_v2 
                GROUP BY geraetegruppe_name 
                ORDER BY cnt DESC LIMIT 8
            """)
            print(f"\n  {Colors.BOLD}Categories:{Colors.RESET}")
            for r in result:
                print(f"    {r['category']}: {r['cnt']}")
            
            # By manufacturer
            result = self.postgres.execute_query("""
                SELECT hersteller_name as manufacturer, COUNT(*) as cnt 
                FROM public.equipment_matrix_v2 
                GROUP BY hersteller_name 
                ORDER BY cnt DESC LIMIT 5
            """)
            print(f"\n  {Colors.BOLD}Manufacturers:{Colors.RESET}")
            for r in result:
                print(f"    {r['manufacturer']}: {r['cnt']}")
        except Exception as e:
            print(f"  {Colors.RED}Error: {e}{Colors.RESET}")
    
    def show_schema(self):
        """Show schema information"""
        print(f"\n{Colors.BOLD}Schema Information{Colors.RESET}")
        print("-" * 50)
        
        if not self.postgres or not self.postgres.available:
            print(f"  {Colors.RED}PostgreSQL not available{Colors.RESET}")
            return
        
        try:
            result = self.postgres.execute_query("""
                SELECT attname 
                FROM pg_attribute 
                WHERE attrelid = 'public.equipment_matrix_v2'::regclass 
                AND attnum > 0 AND NOT attisdropped
            """)
            columns = [r['attname'] for r in result]
            props = [c for c in columns if c.startswith("prop_")]
            num_cols = [c for c in columns if c.endswith("_num")]
            std = [c for c in columns if not c.startswith("prop_") and not c.endswith("_num")]
            
            print(f"  Total: {len(columns)} columns")
            print(f"  Standard: {len(std)}")
            print(f"  Numeric (*_num): {len(num_cols)}")
            print(f"  Properties (prop_*): {len(props)}")
        except Exception as e:
            print(f"  {Colors.RED}Error: {e}{Colors.RESET}")
    
    def clear_context(self):
        """Clear conversation context"""
        self.conversation_history = []
        self.thread_key = f"test_{datetime.now().strftime('%H%M%S')}"
        print("Conversation context cleared")


def get_default_tests() -> List[TestCase]:
    """Load default test cases"""
    tests = [
        TestCase(
            id="sql_count", 
            name="SQL Count Query", 
            query="Wie viele Maschinen haben wir?", 
            expected_tools_any=["count_equipment", "execute_sql"],
            expected_keywords=["maschinen"]
        ),
        TestCase(
            id="sql_filter", 
            name="SQL Filter Query", 
            query="Wie viele Bomag Maschinen?",
            expected_tools_any=["count_equipment", "execute_sql"],
            expected_keywords=["bomag"]
        ),
        TestCase(
            id="fertiger_width", 
            name="Fertiger Width Query", 
            query="Fertiger mit Einbaubreite mindestens 2m",
            expected_tools=["query_equipment"],
            expected_keywords=["fertiger"]
        ),
        TestCase(
            id="manufacturer", 
            name="Manufacturer Query", 
            query="Welche Hersteller gibt es?",
            expected_tools_any=["explore_column", "query_equipment"]
        ),
        TestCase(
            id="rental", 
            name="Rental Filter", 
            query="Welche Mietmaschinen haben wir?",
            expected_tools=["query_equipment"],
            expected_keywords=["miet"]
        ),
    ]
    
    return tests


def get_transcript_regression_tests() -> List[TestCase]:
    """Regression cases derived from real conversation failures."""
    return [
        TestCase(
            id="planner_seed",
            name="Planner Seed",
            query="Empfehle mir bitte eine Maschine fuer den Asphalteinbau von 3,5m",
            pipeline="full",
            reset_context=True,
        ),
        TestCase(
            id="planner_followup_override",
            name="Planner Follow-up Override",
            query="Welche Hoehe hat die Maschine mit der Seriennummer: KBCEZN5BHSWA55197",
            pipeline="full",
            expected_agent="postgres_direct",
            expected_keywords=["2540"],
            forbidden_keywords=["projekt-dossier"],
        ),
        TestCase(
            id="machine_pronoun_followup",
            name="Machine Pronoun Follow-up",
            query="Welche Verwendung hat diese Maschine?",
            pipeline="full",
            expected_agent="postgres_direct",
            expected_keywords=["verwendung", "miet"],
            forbidden_keywords=["projekt-dossier"],
        ),
        TestCase(
            id="machine_info_db",
            name="Machine Info From DB",
            query="Bitte gib mir alle Infos zur Maschine 101870941182 die du in der Datenbank finden kannst",
            pipeline="full",
            reset_context=True,
            expected_agent="postgres_direct",
            expected_keywords=["bomag", "seriennummer", "inventarnummer", "| eigenschaft | wert |"],
            forbidden_keywords=["projekt-dossier"],
        ),
        TestCase(
            id="model_info_lookup",
            name="Model Info Lookup",
            query="Gib mir naehere Infos zum Super 1300-3i",
            pipeline="full",
            reset_context=True,
            expected_agent="langgraph",
            expected_keywords=["super 1300-3i", "seriennummer"],
            forbidden_keywords=["projekt-dossier"],
        ),
        TestCase(
            id="raw_property_table",
            name="Raw Property Table",
            query="Bitte ohne Interpretation einfach nur Eigenschaft und Wert aus der Datenbank fuer Maschine 101870941182",
            pipeline="full",
            reset_context=True,
            expected_agent="postgres_direct",
            expected_keywords=["| eigenschaft | wert |", "seriennummer"],
            forbidden_keywords=["projekt-dossier"],
        ),
        TestCase(
            id="width_recommendation_extensions",
            name="Width Recommendation Extensions",
            query="Empfehle mir bitte eine Maschine fuer den Asphalteinbau von 3,2m aus dem Mietpark",
            pipeline="full",
            reset_context=True,
            expected_agent="langgraph",
            expected_keywords=["super 800", "verbreiter"],
            forbidden_keywords=["projekt-dossier"],
        ),
        TestCase(
            id="hgt_inventory_lookup",
            name="HGT Inventory Lookup",
            query="Welcher Fertiger aus dem Mietpark kann HGT einbauen?",
            pipeline="full",
            reset_context=True,
            expected_agent="langgraph",
            expected_keywords=["hgt"],
            forbidden_keywords=["projekt-dossier"],
        ),
        TestCase(
            id="serial_height_lookup",
            name="Serial Height Lookup",
            query="Welche Hoehe hat die Maschine mit der Seriennummer: KBCEZN5BHSWA55197",
            pipeline="full",
            reset_context=True,
            expected_agent="postgres_direct",
            expected_keywords=["2540"],
            forbidden_keywords=["projekt-dossier"],
        ),
    ]


async def interactive_mode(runner: TestRunner):
    """Interactive testing mode"""
    print(f"\n{Colors.BOLD}Interactive Mode{Colors.RESET}")
    print("Commands: /help /sql /search /stats /schema /batch /regressions /clear /export /exit")
    print("Plain text = run as query test\n")
    
    while True:
        try:
            cmd = input(f"{Colors.CYAN}>{Colors.RESET} ").strip()
            if not cmd:
                continue
            
            if cmd.startswith("/"):
                parts = cmd.split(maxsplit=1)
                action, arg = parts[0].lower(), parts[1] if len(parts) > 1 else ""
                
                if action in ["/exit", "/quit", "/q"]:
                    break
                elif action == "/help":
                    print("\n/sql <query>  - Direct SQL test")
                    print("/search <q>   - Document search")
                    print("/stats        - Database stats")
                    print("/schema       - Schema info")
                    print("/batch        - Run all tests")
                    print("/regressions  - Run transcript-derived regression tests")
                    print("/clear        - Clear context")
                    print("/export       - Export results")
                    print("/exit         - Exit\n")
                elif action == "/sql":
                    runner.test_sql(arg) if arg else print("Usage: /sql <query>")
                elif action == "/search":
                    await runner.test_search(arg) if arg else print("Usage: /search <query>")
                elif action == "/stats":
                    runner.show_stats()
                elif action == "/schema":
                    runner.show_schema()
                elif action == "/batch":
                    await runner.run_suite(get_default_tests())
                elif action == "/regressions":
                    await runner.run_suite(get_transcript_regression_tests())
                elif action == "/clear":
                    runner.clear_context()
                elif action == "/export":
                    runner.export_results()
                else:
                    print(f"Unknown command: {action}")
            else:
                await runner.test_query(cmd)
                
        except KeyboardInterrupt:
            print()
            break
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.RESET}")
    
    return runner.print_summary()


async def main():
    parser = argparse.ArgumentParser(description="Teams Bot RAG Test Suite")
    parser.add_argument("query", nargs="?", help="Query to test")
    parser.add_argument("-b", "--batch", action="store_true", help="Run all tests")
    parser.add_argument("-r", "--regressions", action="store_true", help="Run transcript-derived regression tests")
    parser.add_argument("-s", "--sql", help="Direct SQL test")
    parser.add_argument("-d", "--search", help="Document search test")
    parser.add_argument("--stats", action="store_true", help="Show DB stats")
    parser.add_argument("--schema", action="store_true", help="Show schema")
    parser.add_argument("-o", "--output", help="Export results to file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    runner = TestRunner(verbose=args.verbose)
    exit_code = 0
    
    if args.batch:
        await runner.run_suite(get_default_tests())
        exit_code = runner.print_summary()
    elif args.regressions:
        await runner.run_suite(get_transcript_regression_tests())
        exit_code = runner.print_summary()
    elif args.sql:
        runner.test_sql(args.sql)
    elif args.search:
        await runner.test_search(args.search)
    elif args.stats:
        runner.show_stats()
    elif args.schema:
        runner.show_schema()
    elif args.query:
        await runner.test_query(args.query)
        exit_code = runner.print_summary()
    elif args.interactive or len(sys.argv) == 1:
        exit_code = await interactive_mode(runner)
    
    if args.output:
        runner.export_results(args.output)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
