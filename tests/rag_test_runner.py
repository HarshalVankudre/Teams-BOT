"""
RAG Test Runner with Semantic Evaluation
Uses GPT-5 with reasoning to evaluate responses semantically.
Not deployed to production - for testing only.
"""
import asyncio
import json
import yaml
import os
import sys
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment
load_dotenv(Path(__file__).parent.parent / ".env")

from rag.hybrid_orchestrator import HybridOrchestrator, QueryType
from rag.config import config
from rag.search import RAGSearch


@dataclass
class TestResult:
    """Result of a single test"""
    test_id: str
    category: str
    question: str
    expected_answer: str
    actual_answer: str
    source: str
    query_type: str
    passed: bool
    score: float  # 0.0 to 1.0
    evaluation: str  # Detailed evaluation from LLM
    criteria_results: Dict[str, bool] = field(default_factory=dict)
    execution_time_ms: int = 0
    error: Optional[str] = None


@dataclass
class TestSuiteReport:
    """Complete test suite report"""
    name: str
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    score: float  # Overall score 0-100
    results: List[TestResult]
    summary: str
    recommendations: List[str]

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "score": self.score,
            "pass_rate": f"{(self.passed/self.total_tests)*100:.1f}%" if self.total_tests > 0 else "0%",
            "results": [
                {
                    "test_id": r.test_id,
                    "category": r.category,
                    "question": r.question,
                    "passed": r.passed,
                    "score": r.score,
                    "source": r.source,
                    "query_type": r.query_type,
                    "evaluation": r.evaluation,
                    "criteria_results": r.criteria_results,
                    "execution_time_ms": r.execution_time_ms,
                    "error": r.error
                }
                for r in self.results
            ],
            "summary": self.summary,
            "recommendations": self.recommendations
        }


class SemanticEvaluator:
    """Uses GPT-5 with reasoning to evaluate responses semantically"""

    def __init__(self, model: str = "gpt-5", reasoning_effort: str = "medium"):
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.model = model
        self.reasoning_effort = reasoning_effort

    async def evaluate(
        self,
        question: str,
        expected_answer: str,
        actual_answer: str,
        criteria: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate actual answer against expected answer semantically.
        Returns score, pass/fail, and detailed evaluation.
        """

        criteria_text = "\n".join(f"- {c}" for c in criteria) if criteria else "- Antwort sollte korrekt und relevant sein"

        evaluation_prompt = f"""Du bist ein strenger aber fairer Evaluator für ein RAG-System.
Bewerte die TATSÄCHLICHE ANTWORT gegen die ERWARTETE ANTWORT.

FRAGE: {question}

ERWARTETE ANTWORT / KRITERIEN:
{expected_answer}

BEWERTUNGSKRITERIEN:
{criteria_text}

TATSÄCHLICHE ANTWORT:
{actual_answer}

BEWERTUNGSREGELN:
1. Semantische Korrektheit zählt - exakter Wortlaut ist nicht erforderlich
2. Faktische Richtigkeit ist wichtiger als Vollständigkeit
3. Wenn die Antwort "Keine Ergebnisse" sagt aber welche erwartet wurden → FAIL
4. Wenn die Antwort halluziniert (erfindet Daten) → FAIL
5. Zahlen müssen ungefähr stimmen (±10% Toleranz)

Antworte im JSON-Format:
{{
    "passed": true/false,
    "score": 0.0-1.0,
    "evaluation": "Detaillierte Begründung der Bewertung",
    "criteria_results": {{
        "kriterium1": true/false,
        "kriterium2": true/false
    }},
    "strengths": ["Was gut war"],
    "weaknesses": ["Was verbessert werden könnte"]
}}"""

        try:
            # Build request params
            request_params = {
                "model": self.model,
                "input": [
                    {"role": "system", "content": "Du bist ein präziser Test-Evaluator. Antworte NUR mit validem JSON."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                "text": {"format": {"type": "json_object"}},
                "max_output_tokens": 1500
            }

            # Only add reasoning for supported models
            if 'gpt-5' in self.model or 'o1' in self.model or 'o3' in self.model:
                request_params["reasoning"] = {"effort": self.reasoning_effort}

            response = await self.client.responses.create(**request_params)

            # Check for empty response
            output_text = response.output_text.strip() if response.output_text else ""
            if not output_text:
                print(f"[Evaluator] Warning: Empty response from model")
                return {
                    "passed": False,
                    "score": 0.5,  # Give partial score since we can't evaluate
                    "evaluation": "Evaluator returned empty response",
                    "criteria_results": {},
                    "strengths": [],
                    "weaknesses": ["Evaluation response was empty"]
                }

            # Try to parse JSON
            try:
                result = json.loads(output_text)
                return result
            except json.JSONDecodeError as json_err:
                # Try to extract JSON from response if it has extra text
                import re
                json_match = re.search(r'\{[\s\S]*\}', output_text)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                        return result
                    except:
                        pass
                print(f"[Evaluator] JSON parse error: {json_err}")
                return {
                    "passed": False,
                    "score": 0.3,
                    "evaluation": f"Could not parse evaluator response: {output_text[:200]}...",
                    "criteria_results": {},
                    "strengths": [],
                    "weaknesses": ["Response was not valid JSON"]
                }

        except Exception as e:
            print(f"[Evaluator] Error: {e}")
            return {
                "passed": False,
                "score": 0.0,
                "evaluation": f"Evaluation failed: {str(e)}",
                "criteria_results": {},
                "strengths": [],
                "weaknesses": ["Evaluation konnte nicht durchgeführt werden"]
            }


class RAGTestRunner:
    """Runs tests against the Hybrid RAG system"""

    def __init__(self, test_file: str = None):
        # Default to txt file in test-questions folder
        if test_file is None:
            # Try txt first, then md as fallback
            txt_path = Path(__file__).parent.parent / "test-questions" / "questions.txt"
            md_path = Path(__file__).parent.parent / "test-questions" / "TEST_QUESTIONS.md"
            self.test_file = txt_path if txt_path.exists() else md_path
        elif test_file.endswith('.yaml') or test_file.endswith('.yml'):
            self.test_file = Path(__file__).parent / test_file
        else:
            self.test_file = Path(__file__).parent.parent / "test-questions" / test_file

        self.orchestrator = HybridOrchestrator(verbose=False)
        self.rag_search = RAGSearch()  # For semantic/hybrid queries
        self.evaluator = SemanticEvaluator()
        self.results: List[TestResult] = []

    def load_tests(self) -> List[Dict]:
        """Load test cases from YAML, Markdown, or TXT file"""
        if self.test_file.suffix in ['.yaml', '.yml']:
            return self._load_yaml_tests()
        elif self.test_file.suffix == '.md':
            return self._load_markdown_tests()
        elif self.test_file.suffix == '.txt':
            return self._load_txt_tests()
        else:
            raise ValueError(f"Unsupported file format: {self.test_file.suffix}")

    def _load_yaml_tests(self) -> List[Dict]:
        """Load test cases from YAML file"""
        with open(self.test_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data.get('test_cases', [])

    def _load_markdown_tests(self) -> List[Dict]:
        """Parse test cases from markdown file"""
        import re

        with open(self.test_file, 'r', encoding='utf-8') as f:
            content = f.read()

        tests = []
        current_category = "unknown"

        # Category mapping from markdown headers
        category_map = {
            "AGGREGATION": "aggregation",
            "FILTER": "filter",
            "COMPARISON": "comparison",
            "COMPLEX": "complex_filter",
            "LOOKUP": "lookup",
            "EDGE": "edge_case",
        }

        # Split by question headers (### Q1:, ### Q2:, etc.)
        sections = re.split(r'###\s+Q(\d+):\s*', content)

        # First section is intro, skip it but check for category headers
        for i in range(1, len(sections), 2):
            if i + 1 >= len(sections):
                break

            q_num = sections[i]
            q_content = sections[i + 1]

            # Check for category in preceding text
            prev_text = sections[i - 1] if i > 0 else ""
            for key, cat in category_map.items():
                if key in prev_text.upper():
                    current_category = cat
                    break

            # Parse question and answer
            lines = q_content.strip().split('\n')
            question = lines[0].strip() if lines else ""

            # Find answer line
            expected_answer = ""
            answer_lines = []
            in_answer = False

            for line in lines[1:]:
                if line.startswith('**✅ Answer:**') or line.startswith('**✅Answer:**'):
                    in_answer = True
                    answer_part = line.replace('**✅ Answer:**', '').replace('**✅Answer:**', '').strip()
                    if answer_part:
                        answer_lines.append(answer_part)
                elif line.startswith('---'):
                    break
                elif in_answer and line.strip():
                    # Stop at next section header or empty line after content
                    if line.startswith('##') or line.startswith('###'):
                        break
                    answer_lines.append(line.strip())

            expected_answer = ' '.join(answer_lines).strip()
            if expected_answer.startswith('-'):
                expected_answer = expected_answer[1:].strip()

            # Build test case
            test = {
                'id': f"Q{q_num.zfill(2)}",
                'category': current_category,
                'question': question,
                'expected_answer': expected_answer[:500],  # Truncate long answers
                'criteria': [
                    "Antwort sollte semantisch korrekt sein",
                    "Zahlen sollten ungefaehr stimmen (±20% Toleranz)"
                ],
                'expected_source': 'postgres'  # Most questions expect PostgreSQL
            }

            tests.append(test)

        return tests

    def _load_txt_tests(self) -> List[Dict]:
        """
        Parse test cases from TXT file.

        Supported formats:
        1. Q/A pairs: "Q1: Question" followed by "A1: Answer"
        2. Simple: One question per line
        3. Pipe format: "Question | Expected Answer"

        Lines starting with # or = are comments/separators.
        Empty lines are ignored.
        """
        import re

        with open(self.test_file, 'r', encoding='utf-8') as f:
            content = f.read()

        tests = []
        current_category = "general"

        # Category mapping from section headers
        category_map = {
            'COUNT': 'count',
            'MANUFACTURER': 'manufacturer',
            'MAX': 'aggregation',
            'MIN': 'aggregation',
            'AVG': 'aggregation',
            'FILTER': 'filter',
            'MULTI-FILTER': 'complex_filter',
            'COMPLEX': 'complex_filter',
            'COMPARISON': 'comparison',
            'LOOKUP': 'lookup',
            'DETAIL': 'lookup',
            'SEMANTIC': 'semantic',
            'RECOMMENDATION': 'semantic',
            'EDGE': 'edge_case',
            'BONUS': 'quick',
            'TYPO': 'typo_test',
        }

        # Find section headers to track categories
        lines = content.split('\n')
        line_categories = {}
        current_line_category = "general"

        for i, line in enumerate(lines):
            # Check for SECTION headers
            if 'SECTION' in line.upper() or line.startswith('==='):
                for key, cat in category_map.items():
                    if key in line.upper():
                        current_line_category = cat
                        break
            line_categories[i] = current_line_category

        # Parse Q/A pairs using regex
        qa_pattern = re.compile(r'Q(\d+):\s*(.+?)(?:\n|$)(?:A\1:\s*(.+?))?(?=\nQ\d+:|\n\n|\n===|$)', re.DOTALL | re.IGNORECASE)

        for match in qa_pattern.finditer(content):
            q_num = match.group(1)
            question = match.group(2).strip()
            answer = match.group(3).strip() if match.group(3) else ""

            # Find category based on position in file
            start_pos = match.start()
            line_num = content[:start_pos].count('\n')
            category = line_categories.get(line_num, "general")

            # Clean up multi-line answers
            if answer:
                answer = ' '.join(answer.split())

            test = {
                'id': f"Q{q_num.zfill(2)}",
                'category': category,
                'question': question,
                'expected_answer': answer or "Semantisch korrekte Antwort erwartet",
                'criteria': [
                    "Antwort sollte semantisch korrekt sein",
                    "Zahlen sollten ungefaehr stimmen (±20% Toleranz)"
                ],
                'expected_source': 'semantic' if category == 'semantic' else 'postgres'
            }
            tests.append(test)

        # Also parse simple "Q: question" / "A: answer" pairs (no numbers)
        simple_qa = re.compile(r'^Q:\s*(.+?)$\nA:\s*(.+?)$', re.MULTILINE)
        simple_num = len(tests) + 1

        for match in simple_qa.finditer(content):
            question = match.group(1).strip()
            answer = match.group(2).strip()

            test = {
                'id': f"Q{str(simple_num).zfill(2)}",
                'category': 'quick',
                'question': question,
                'expected_answer': answer,
                'criteria': [
                    "Antwort sollte semantisch korrekt sein"
                ],
                'expected_source': 'postgres'
            }
            tests.append(test)
            simple_num += 1

        return tests

    async def run_single_test(self, test: Dict) -> TestResult:
        """Run a single test case"""
        test_id = test.get('id', 'UNKNOWN')
        question = test.get('question', '')
        expected = test.get('expected_answer', '')
        criteria = test.get('criteria', [])
        category = test.get('category', 'unknown')
        expected_source = test.get('expected_source', 'any')

        # Sanitize question for console output (avoid Unicode encoding errors on Windows)
        safe_question = question[:50].encode('ascii', 'replace').decode('ascii')
        print(f"  [{test_id}] {safe_question}...")

        start_time = datetime.now()
        error = None
        actual_answer = ""
        source = "unknown"
        query_type = "unknown"

        try:
            # Call the orchestrator
            result = await self.orchestrator.query(question)
            actual_answer = result.answer or "Keine Antwort generiert"
            source = result.source
            query_type = result.query_type.value

            # If routed to Pinecone/hybrid, perform actual semantic search
            if not result.answer and result.source in ["pinecone", "hybrid"]:
                # Build filters from orchestrator if available
                filters = None
                if result.structured_filters:
                    filters = {}
                    for key, value in result.structured_filters.items():
                        if key == "kategorie" and value:
                            filters["kategorie"] = {"$eq": value.lower()}
                        elif key == "hersteller" and value:
                            filters["hersteller"] = {"$eq": value}

                # Use RAGSearch for semantic search
                semantic_query = result.semantic_query or question
                rag_result = await self.rag_search.search_and_generate(
                    query=semantic_query,
                    filters=filters
                )
                actual_answer = rag_result.get("response", "Keine Antwort aus Pinecone")
                source = "pinecone"

        except Exception as e:
            error = str(e)
            actual_answer = f"ERROR: {error}"

        execution_time = int((datetime.now() - start_time).total_seconds() * 1000)

        # Evaluate the response
        eval_result = await self.evaluator.evaluate(
            question=question,
            expected_answer=expected,
            actual_answer=actual_answer,
            criteria=criteria
        )

        # Check source routing
        source_correct = True
        if expected_source != "any":
            source_correct = source == expected_source
            if not source_correct:
                eval_result["evaluation"] += f"\n[WARNING] Routing-Fehler: Erwartet '{expected_source}', bekam '{source}'"
                eval_result["score"] = max(0, eval_result.get("score", 0) - 0.2)

        passed = eval_result.get("passed", False) and source_correct and not error

        return TestResult(
            test_id=test_id,
            category=category,
            question=question,
            expected_answer=expected,
            actual_answer=actual_answer,
            source=source,
            query_type=query_type,
            passed=passed,
            score=eval_result.get("score", 0.0),
            evaluation=eval_result.get("evaluation", ""),
            criteria_results=eval_result.get("criteria_results", {}),
            execution_time_ms=execution_time,
            error=error
        )

    async def run_all_tests(self, categories: Optional[List[str]] = None) -> TestSuiteReport:
        """Run all tests and generate report"""
        tests = self.load_tests()

        # Filter by category if specified
        if categories:
            tests = [t for t in tests if t.get('category') in categories]

        print(f"\n{'='*60}")
        print(f"RAG TEST SUITE")
        print(f"{'='*60}")
        print(f"Total tests to run: {len(tests)}")
        print(f"{'='*60}\n")

        self.results = []

        for test in tests:
            result = await self.run_single_test(test)
            self.results.append(result)

            status = "[PASS]" if result.passed else "[FAIL]"
            print(f"    {status} (Score: {result.score:.2f}, {result.execution_time_ms}ms)")

        # Generate summary
        report = await self._generate_report(tests)

        return report

    async def _generate_report(self, tests: List[Dict]) -> TestSuiteReport:
        """Generate comprehensive test report"""
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        total_score = sum(r.score for r in self.results) / len(self.results) if self.results else 0

        # Generate summary using LLM
        summary_prompt = f"""Erstelle eine kurze Zusammenfassung der Testergebnisse:

Gesamt: {len(self.results)} Tests
Bestanden: {passed}
Fehlgeschlagen: {failed}
Durchschnittliche Punktzahl: {total_score:.2f}

Fehlgeschlagene Tests:
{self._format_failures()}

Erstelle:
1. Eine kurze Zusammenfassung (2-3 Sätze)
2. 3-5 konkrete Verbesserungsvorschläge

JSON-Format:
{{
    "summary": "Zusammenfassung...",
    "recommendations": ["Empfehlung 1", "Empfehlung 2", ...]
}}"""

        try:
            response = await self.evaluator.client.responses.create(
                model=self.evaluator.model,
                input=[{"role": "user", "content": summary_prompt}],
                reasoning={"effort": "low"},
                text={"format": {"type": "json_object"}},
                max_output_tokens=1000
            )
            summary_data = json.loads(response.output_text)
        except:
            summary_data = {
                "summary": f"{passed}/{len(self.results)} Tests bestanden ({total_score*100:.1f}%)",
                "recommendations": ["Keine automatischen Empfehlungen verfügbar"]
            }

        return TestSuiteReport(
            name="RÜKO RAG System Tests",
            timestamp=datetime.now().isoformat(),
            total_tests=len(self.results),
            passed=passed,
            failed=failed,
            score=total_score * 100,
            results=self.results,
            summary=summary_data.get("summary", ""),
            recommendations=summary_data.get("recommendations", [])
        )

    def _format_failures(self) -> str:
        """Format failed tests for summary"""
        failures = [r for r in self.results if not r.passed]
        if not failures:
            return "Keine"

        lines = []
        for f in failures[:5]:  # Limit to 5
            lines.append(f"- [{f.test_id}] {f.question[:40]}... (Score: {f.score:.2f})")

        if len(failures) > 5:
            lines.append(f"... und {len(failures) - 5} weitere")

        return "\n".join(lines)

    def _safe_print(self, text: str) -> str:
        """Sanitize text for Windows console (cp1252) encoding"""
        if not text:
            return ""
        return text.encode('ascii', 'replace').decode('ascii')

    def print_report(self, report: TestSuiteReport):
        """Print formatted report to console"""
        print(f"\n{'='*60}")
        print(f"TEST REPORT: {self._safe_print(report.name)}")
        print(f"{'='*60}")
        print(f"Timestamp: {report.timestamp}")
        print(f"Total Tests: {report.total_tests}")
        print(f"Passed: {report.passed}")
        print(f"Failed: {report.failed}")
        print(f"Score: {report.score:.1f}/100")
        print(f"{'='*60}")

        # Results by category
        categories = {}
        for r in report.results:
            cat = r.category
            if cat not in categories:
                categories[cat] = {"passed": 0, "failed": 0, "total_score": 0}
            categories[cat]["total_score"] += r.score
            if r.passed:
                categories[cat]["passed"] += 1
            else:
                categories[cat]["failed"] += 1

        print("\nRESULTS BY CATEGORY:")
        print("-" * 40)
        for cat, stats in sorted(categories.items()):
            total = stats["passed"] + stats["failed"]
            avg_score = stats["total_score"] / total if total > 0 else 0
            print(f"  {cat.upper()}: {stats['passed']}/{total} passed (avg: {avg_score:.2f})")

        # Failed tests detail
        failed = [r for r in report.results if not r.passed]
        if failed:
            print(f"\n{'='*60}")
            print("FAILED TESTS:")
            print("-" * 40)
            for r in failed:
                print(f"\n[{r.test_id}] {r.category.upper()}")
                print(f"  Question: {self._safe_print(r.question[:80])}")
                print(f"  Expected: {self._safe_print(r.expected_answer[:100])}...")
                print(f"  Actual: {self._safe_print(r.actual_answer[:100])}...")
                print(f"  Score: {r.score:.2f}")
                print(f"  Evaluation: {self._safe_print(r.evaluation[:200])}...")
                if r.error:
                    print(f"  Error: {self._safe_print(r.error)}")

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY:")
        print("-" * 40)
        print(self._safe_print(report.summary))

        print(f"\nRECOMMENDATIONS:")
        print("-" * 40)
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {self._safe_print(rec)}")

        print(f"\n{'='*60}")

    def save_report(self, report: TestSuiteReport, filename: str = None):
        """Save report to JSON file"""
        if not filename:
            filename = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = Path(__file__).parent / "reports" / filename
        filepath.parent.mkdir(exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        print(f"\nReport saved to: {filepath}")
        return filepath


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='RAG Test Runner')
    parser.add_argument('--categories', '-c', nargs='+', help='Test categories to run')
    parser.add_argument('--test-file', '-f', default=None,
                        help='Test cases file (default: test-questions/TEST_QUESTIONS.md). Supports .md and .yaml')
    parser.add_argument('--save', '-s', action='store_true', help='Save report to file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    runner = RAGTestRunner(test_file=args.test_file)
    print(f"Loading tests from: {runner.test_file}")
    report = await runner.run_all_tests(categories=args.categories)

    runner.print_report(report)

    if args.save:
        runner.save_report(report)

    # Exit with error code if tests failed
    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
