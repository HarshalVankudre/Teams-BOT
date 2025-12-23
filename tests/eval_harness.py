import argparse
import json
import re
import sys
from typing import Any, Dict, List, Tuple

try:
    import sqlglot
    SQLGLOT_AVAILABLE = True
except Exception:
    SQLGLOT_AVAILABLE = False


REFUSAL_RE = re.compile(
    r"(kann.*nicht|darf\s+nicht|kann.*keine|keine\s+(zugang|zugangsdaten|geheimnisse)|"
    r"cannot\s+share|cannot\s+provide)",
    re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def compile_patterns(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns or []]


def count_sentences(text: str) -> int:
    text = (text or "").strip()
    if not text:
        return 0
    return len(SENTENCE_SPLIT_RE.split(text))


def count_bullets(text: str) -> int:
    lines = [line.strip() for line in (text or "").splitlines()]
    return sum(1 for line in lines if line.startswith("-") or line.startswith("*"))


def evaluate_sql(expect_sql: Dict[str, Any], sql: str) -> List[str]:
    failures: List[str] = []
    normalized_sql = (sql or "").strip()

    if expect_sql.get("require_sql") and not normalized_sql:
        failures.append("expected SQL but got empty")
        return failures

    for pattern in compile_patterns(expect_sql.get("required_regex", [])):
        if not pattern.search(normalized_sql):
            failures.append(f"missing SQL pattern: {pattern.pattern}")

    for pattern in compile_patterns(expect_sql.get("forbidden_regex", [])):
        if pattern.search(normalized_sql):
            failures.append(f"forbidden SQL pattern matched: {pattern.pattern}")

    if expect_sql.get("limit") is not None:
        expected = int(expect_sql["limit"])
        match = re.search(r"\blimit\s+(\d+)\b", normalized_sql, flags=re.IGNORECASE)
        if not match or int(match.group(1)) != expected:
            failures.append(f"expected LIMIT {expected}")

    if SQLGLOT_AVAILABLE and normalized_sql:
        try:
            sqlglot.parse_one(normalized_sql, read="postgres")
        except Exception as exc:
            failures.append(f"sqlglot parse failed: {exc}")

    return failures


def evaluate_answer(expect_answer: Dict[str, Any], answer: str) -> List[str]:
    failures: List[str] = []
    normalized = (answer or "").strip()
    lower = normalized.lower()

    if expect_answer.get("require_refusal"):
        if not REFUSAL_RE.search(normalized):
            failures.append("expected refusal")

    if expect_answer.get("require_clarification"):
        if "?" not in normalized:
            failures.append("expected clarification question")

    if expect_answer.get("require_citations"):
        if "quelle" not in lower:
            failures.append("expected citations")

    for fragment in expect_answer.get("must_include", []) or []:
        if fragment.lower() not in lower:
            failures.append(f"missing answer fragment: {fragment}")

    for fragment in expect_answer.get("must_not_include", []) or []:
        if fragment.lower() in lower:
            failures.append(f"forbidden answer fragment: {fragment}")

    max_sentences = expect_answer.get("max_sentences")
    if max_sentences is not None and count_sentences(normalized) > int(max_sentences):
        failures.append(f"too many sentences (> {max_sentences})")

    max_bullets = expect_answer.get("max_bullets")
    if max_bullets is not None and count_bullets(normalized) > int(max_bullets):
        failures.append(f"too many bullets (> {max_bullets})")

    max_chars = expect_answer.get("max_chars")
    if max_chars is not None and len(normalized) > int(max_chars):
        failures.append(f"answer too long (> {max_chars} chars)")

    return failures


def evaluate_cases(cases: List[Dict[str, Any]], results: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]]]:
    by_id = {case["id"]: case for case in cases}
    result_map = {case["id"]: case for case in results.get("cases", [])}
    failures_report: List[Dict[str, Any]] = []
    failure_count = 0

    for case_id, case in by_id.items():
        result = result_map.get(case_id, {})
        sql = result.get("sql", "")
        answer = result.get("answer", "")

        case_failures: List[str] = []
        expect = case.get("expect", {})
        if "sql" in expect:
            case_failures.extend(evaluate_sql(expect.get("sql", {}), sql))
        if "answer" in expect:
            case_failures.extend(evaluate_answer(expect.get("answer", {}), answer))

        if case_failures:
            failure_count += len(case_failures)
            failures_report.append({
                "id": case_id,
                "failures": case_failures,
            })

    return failure_count, failures_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="tests/qa_cases.json")
    parser.add_argument("--results", default="tests/sample_results.json")
    args = parser.parse_args()

    cases_payload = load_json(args.cases)
    results_payload = load_json(args.results)

    failures, report = evaluate_cases(cases_payload.get("cases", []), results_payload)

    summary = {
        "total_cases": len(cases_payload.get("cases", [])),
        "failures": failures,
        "failed_cases": report,
    }
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
