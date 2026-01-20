#!/usr/bin/env python3
"""
Teams Bot RAG Test Suite
Industry-standard testing framework for RAG pipeline validation.
"""
import os
import sys
import json
import time
import asyncio
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
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
    expected_tools: List[str] = field(default_factory=list)
    expected_keywords: List[str] = field(default_factory=list)
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
    tools_used: List[str] = field(default_factory=list)
    sql_rows: int = 0
    assertions_passed: int = 0
    assertions_failed: int = 0
    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

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
    """Industry-standard test runner for RAG pipeline"""
    
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
        
        self.config = self.postgres = self.pinecone = self.agent = None
        
        try:
            from rag.config import config
            self.config = config
            print(f"  Config       {Colors.GREEN}OK{Colors.RESET}  Model: {config.response_model}")
        except Exception as e:
            print(f"  Config       {Colors.RED}FAIL{Colors.RESET}  {e}")
        
        try:
            from rag.postgres import PostgresService
            self.postgres = PostgresService()
            if self.postgres.available:
                print(f"  PostgreSQL   {Colors.GREEN}OK{Colors.RESET}  {self.postgres.equipment_table}")
            else:
                print(f"  PostgreSQL   {Colors.YELLOW}WARN{Colors.RESET}  {self.postgres.availability_error}")
        except Exception as e:
            print(f"  PostgreSQL   {Colors.RED}FAIL{Colors.RESET}  {e}")
        
        try:
            from rag.vector_store import PineconeStore
            self.pinecone = PineconeStore()
            print(f"  Pinecone     {Colors.GREEN}OK{Colors.RESET}")
        except Exception as e:
            print(f"  Pinecone     {Colors.YELLOW}WARN{Colors.RESET}  {e}")
        
        try:
            from rag.single_agent import create_single_agent
            self.agent = create_single_agent(verbose=False, pinecone_service=self.pinecone)
            print(f"  Agent        {Colors.GREEN}OK{Colors.RESET}")
        except Exception as e:
            print(f"  Agent        {Colors.RED}FAIL{Colors.RESET}  {e}")
        
        print("-" * 50)
    
    async def run_test(self, test: TestCase) -> TestResult:
        """Execute a single test case"""
        start = time.time()
        result = TestResult(
            test_id=test.id,
            test_name=test.name,
            status=TestStatus.ERROR,
            duration_ms=0,
            query=test.query
        )
        
        if not self.agent:
            result.error = "Agent not initialized"
            result.duration_ms = int((time.time() - start) * 1000)
            return result
        
        try:
            agent_result = await asyncio.wait_for(
                self.agent.process(
                    user_query=test.query,
                    conversation_history=self.conversation_history,
                    thread_key=self.thread_key
                ),
                timeout=test.timeout_ms / 1000
            )
            
            result.duration_ms = agent_result.execution_time_ms
            result.response = agent_result.response
            result.tools_used = agent_result.tools_used
            result.sql_rows = agent_result.sql_results_count
            # Token usage
            result.input_tokens = agent_result.input_tokens
            result.output_tokens = agent_result.output_tokens
            result.reasoning_tokens = agent_result.reasoning_tokens
            result.total_tokens = agent_result.total_tokens
            
            # Run assertions
            assertions_passed = 0
            assertions_failed = 0
            
            # Check if response exists
            if agent_result.response and len(agent_result.response) > 10:
                assertions_passed += 1
            else:
                assertions_failed += 1
            
            # Check expected tools
            if test.expected_tools:
                if all(t in agent_result.tools_used for t in test.expected_tools):
                    assertions_passed += 1
                else:
                    assertions_failed += 1
            
            # Check expected keywords in response
            if test.expected_keywords:
                response_lower = agent_result.response.lower()
                if any(kw.lower() in response_lower for kw in test.expected_keywords):
                    assertions_passed += 1
                else:
                    assertions_failed += 1
            
            # Check minimum results
            if test.min_results > 0:
                if agent_result.sql_results_count >= test.min_results:
                    assertions_passed += 1
                else:
                    assertions_failed += 1
            
            result.assertions_passed = assertions_passed
            result.assertions_failed = assertions_failed
            result.status = TestStatus.PASSED if assertions_failed == 0 and agent_result.success else TestStatus.FAILED
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": test.query})
            self.conversation_history.append({"role": "assistant", "content": agent_result.response})
            
        except asyncio.TimeoutError:
            result.error = f"Timeout after {test.timeout_ms}ms"
            result.status = TestStatus.ERROR
            result.duration_ms = test.timeout_ms
        except Exception as e:
            result.error = str(e)
            result.status = TestStatus.ERROR
            result.duration_ms = int((time.time() - start) * 1000)
        
        return result
    
    def _print_test_result(self, result: TestResult, index: int):
        """Print single test result in standard format"""
        status_color = {
            TestStatus.PASSED: Colors.GREEN,
            TestStatus.FAILED: Colors.RED,
            TestStatus.ERROR: Colors.RED,
            TestStatus.SKIPPED: Colors.YELLOW
        }.get(result.status, "")
        
        status_icon = {
            TestStatus.PASSED: "PASS",
            TestStatus.FAILED: "FAIL",
            TestStatus.ERROR: "ERR ",
            TestStatus.SKIPPED: "SKIP"
        }.get(result.status, "????")
        
        print(f"  {status_color}[{status_icon}]{Colors.RESET} {result.test_name} ({result.duration_ms}ms)")
        
        if result.status != TestStatus.PASSED and self.verbose:
            if result.error:
                print(f"         {Colors.DIM}Error: {result.error}{Colors.RESET}")
            if result.response:
                preview = result.response[:100].replace('\n', ' ')
                print(f"         {Colors.DIM}Response: {preview}...{Colors.RESET}")
    
    async def run_suite(self, tests: List[TestCase]) -> TestSuite:
        """Run all tests in the suite"""
        print(f"\n{Colors.BOLD}Running {len(tests)} tests{Colors.RESET}")
        print("=" * 50)
        
        for i, test in enumerate(tests, 1):
            result = await self.run_test(test)
            self.suite.results.append(result)
            self._print_test_result(result, i)
        
        self.suite.ended_at = datetime.now().isoformat()
        return self.suite
    
    def print_summary(self):
        """Print test summary in standard format"""
        s = self.suite
        print("\n" + "=" * 50)
        print(f"{Colors.BOLD}Test Results{Colors.RESET}")
        print("=" * 50)
        print(f"  Total:    {s.total}")
        print(f"  Passed:   {Colors.GREEN}{s.passed}{Colors.RESET}")
        print(f"  Failed:   {Colors.RED}{s.failed}{Colors.RESET}")
        print(f"  Errors:   {Colors.RED}{s.errors}{Colors.RESET}")
        print(f"  Duration: {s.duration_ms}ms")
        print(f"  Rate:     {s.pass_rate:.1f}%")
        print("=" * 50)
        
        if s.failed > 0 or s.errors > 0:
            print(f"\n{Colors.RED}FAILED{Colors.RESET}")
            return 1
        print(f"\n{Colors.GREEN}PASSED{Colors.RESET}")
        return 0
    
    def export_results(self, filepath: str = None):
        """Export results to JSON (JUnit-compatible structure)"""
        filepath = filepath or f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output = {
            "testsuites": {
                "name": self.suite.name,
                "tests": self.suite.total,
                "failures": self.suite.failed,
                "errors": self.suite.errors,
                "time": self.suite.duration_ms / 1000,
                "timestamp": self.suite.started_at
            },
            "testcases": [
                {
                    "name": r.test_name,
                    "classname": r.test_id,
                    "time": r.duration_ms / 1000,
                    "status": r.status.value,
                    "failure": r.error if r.status != TestStatus.PASSED else None,
                    "query": r.query,
                    "response": r.response[:500] if r.response else None
                }
                for r in self.suite.results
            ]
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults exported to {filepath}")
    
    async def test_query(self, query: str) -> TestResult:
        """Run ad-hoc query test"""
        test = TestCase(id="adhoc", name="Ad-hoc Query", query=query)
        result = await self.run_test(test)
        self.suite.results.append(result)
        self._print_test_result(result, 1)

        if result.response:
            print(f"\n{Colors.BOLD}Response:{Colors.RESET}")
            print(result.response)

        # Display token usage
        if result.total_tokens > 0:
            print(f"\n{Colors.BOLD}Token Usage:{Colors.RESET}")
            print(f"  Input:     {result.input_tokens:,}")
            print(f"  Output:    {result.output_tokens:,}")
            if result.reasoning_tokens > 0:
                print(f"  Reasoning: {result.reasoning_tokens:,}")
            print(f"  {Colors.CYAN}Total:     {result.total_tokens:,}{Colors.RESET}")

        return result
    
    def test_sql(self, sql: str) -> Dict:
        """Run direct SQL test"""
        print(f"\n{Colors.BOLD}SQL Test{Colors.RESET}")
        print("-" * 50)
        
        if not self.postgres or not self.postgres.available:
            print(f"  {Colors.RED}[FAIL]{Colors.RESET} PostgreSQL not available")
            return {"error": "PostgreSQL not available"}
        
        start = time.time()
        prepared, error = self.postgres.prepare_readonly_sql(sql)
        
        if error:
            print(f"  {Colors.RED}[FAIL]{Colors.RESET} {error}")
            return {"error": error}
        
        try:
            results = self.postgres.execute_query(prepared, raise_on_error=True)
            duration = int((time.time() - start) * 1000)
            print(f"  {Colors.GREEN}[PASS]{Colors.RESET} {len(results)} rows in {duration}ms")
            
            if results:
                print(f"\n{Colors.BOLD}Results:{Colors.RESET}")
                for i, row in enumerate(results[:5], 1):
                    print(f"  {i}. {json.dumps(row, default=str)[:100]}...")
                if len(results) > 5:
                    print(f"  ... and {len(results) - 5} more rows")
            
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
        
        stats = self.postgres.get_statistics()
        print(f"  Total Equipment: {stats['total_count']}")
        print(f"\n  {Colors.BOLD}Categories:{Colors.RESET}")
        for c in stats.get("by_category", [])[:8]:
            print(f"    {c.get('equipment_group')}: {c.get('count')}")
        print(f"\n  {Colors.BOLD}Manufacturers:{Colors.RESET}")
        for m in stats.get("by_manufacturer", [])[:5]:
            print(f"    {m.get('manufacturer')}: {m.get('count')}")
    
    def show_schema(self):
        """Show schema information"""
        print(f"\n{Colors.BOLD}Schema Information{Colors.RESET}")
        print("-" * 50)
        
        if not self.postgres or not self.postgres.available:
            print(f"  {Colors.RED}PostgreSQL not available{Colors.RESET}")
            return
        
        columns = self.postgres.get_column_info(refresh=True)
        props = [c for c in columns if c.startswith("prop_")]
        std = [c for c in columns if not c.startswith("prop_")]
        
        print(f"  Total: {len(columns)} columns")
        print(f"  Standard: {len(std)}")
        print(f"  Properties: {len(props)}")
    
    def clear_context(self):
        """Clear conversation context"""
        self.conversation_history = []
        self.thread_key = f"test_{datetime.now().strftime('%H%M%S')}"
        print("Conversation context cleared")


def get_default_tests() -> List[TestCase]:
    """Load default test cases"""
    tests = [
        TestCase(id="sql_count", name="SQL Count Query", query="Wie viele Maschinen haben wir?", 
                 expected_tools=["execute_sql"], expected_keywords=["2404", "maschinen"]),
        TestCase(id="sql_filter", name="SQL Filter Query", query="Wie viele Bomag Maschinen?",
                 expected_tools=["execute_sql"], expected_keywords=["bomag", "123"]),
        TestCase(id="sql_category", name="Category Query", query="Zeige alle Kettenfertiger",
                 expected_tools=["execute_sql"], expected_keywords=["kettenfertiger"]),
        TestCase(id="manufacturer", name="Manufacturer Query", query="Welche Hersteller gibt es?",
                 expected_tools=["execute_sql"]),
        TestCase(id="rental", name="Rental Filter", query="Welche Mietmaschinen haben wir?",
                 expected_tools=["execute_sql"], expected_keywords=["miet", "vermiet"]),
    ]
    
    # Load from qa_pairs.json if exists
    qa_path = Path(__file__).parent / "rag" / "qa_pairs.json"
    if qa_path.exists():
        try:
            data = json.load(open(qa_path, "r", encoding="utf-8"))
            for qa in data.get("qa_pairs", []):
                tests.append(TestCase(
                    id=f"qa_{qa.get('id', len(tests))}",
                    name=qa.get("kategorie", "QA Test"),
                    query=qa.get("frage", ""),
                    category=qa.get("kategorie", "general")
                ))
        except Exception:
            pass
    
    return tests


async def interactive_mode(runner: TestRunner):
    """Interactive testing mode"""
    print(f"\n{Colors.BOLD}Interactive Mode{Colors.RESET}")
    print("Commands: /help /sql /search /stats /schema /batch /clear /export /exit")
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
