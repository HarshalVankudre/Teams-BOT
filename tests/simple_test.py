"""
Intelligent RAG Test Runner
- Uses LLM to parse ANY file format (txt, yaml, json, md, etc.)
- Extracts questions and expected answers automatically
- Runs each question through the RAG system
- Evaluates answers semantically
- Outputs detailed score report
"""
import asyncio
import json
from typing import List, Dict, Any
from openai import AsyncOpenAI
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.search import RAGSearch
from rag.config import config


async def parse_test_file(client: AsyncOpenAI, content: str, filename: str) -> List[Dict[str, Any]]:
    """Use LLM to intelligently parse any test file format"""

    parse_prompt = f"""You are parsing a test file to extract question-answer pairs.
The file may be in ANY format: YAML, JSON, TXT, Markdown, or custom format.

File: {filename}
Content:
{content[:15000]}

Extract ALL test cases from this file. For each test case, identify:
1. The question (FRAGE, question, Q, etc.)
2. The expected answer (ANTWORT, expected_answer, A, etc.)
3. A short ID if present (like Q1, TEST-1, etc.)
4. Query type if mentioned (aggregation, filter, semantic, etc.)

Return a JSON object with:
{{
  "test_cases": [
    {{
      "id": "Q1",
      "question": "the question text",
      "expected_answer": "the expected answer (can be multi-line, include key facts)",
      "query_type": "aggregation/filter/semantic/scenario/complex (if mentioned)"
    }}
  ],
  "total_found": number_of_tests
}}

IMPORTANT:
- Extract the ACTUAL question text, not summaries
- For expected answers, include the KEY FACTS that should be in the answer
- If the answer has specific numbers (counts, weights, etc.), include them
- Handle German text (FRAGE/ANTWORT) correctly
"""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You extract test cases from files and return JSON."},
            {"role": "user", "content": parse_prompt}
        ],
        response_format={"type": "json_object"},
        max_tokens=8000
    )

    result = json.loads(response.choices[0].message.content)
    return result.get('test_cases', [])


async def run_tests(test_file: str) -> None:
    """Run all tests with intelligent parsing and semantic evaluation"""

    # Read file content
    print(f"Loading tests from: {test_file}")
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()

    filename = os.path.basename(test_file)
    file_ext = os.path.splitext(filename)[1].lower()

    # Initialize
    client = AsyncOpenAI(api_key=config.openai_api_key)
    rag = RAGSearch()

    # Phase 0: Intelligent file parsing
    print("=" * 60)
    print("PHASE 0: Intelligent File Parsing")
    print("=" * 60)
    print(f"File type: {file_ext or 'unknown'}")
    print("Using LLM to extract test cases...")

    tests = await parse_test_file(client, content, filename)
    print(f"Found {len(tests)} test cases\n")

    if not tests:
        print("ERROR: No test cases found in file!")
        return

    # Show extracted tests
    print("Extracted tests:")
    for i, t in enumerate(tests, 1):
        qtype = t.get('query_type', 'unknown')
        print(f"  {i}. [{t.get('id', f'Q{i}')}] ({qtype}) {t['question'][:50]}...")
    print()

    # Store results
    results = []

    # Phase 1: Run all questions through the RAG
    print("=" * 60)
    print("PHASE 1: Running Questions Through RAG")
    print("=" * 60)

    for i, test in enumerate(tests, 1):
        question = test['question']
        expected = test['expected_answer']
        test_id = test.get('id', f'Q{i}')
        query_type = test.get('query_type', 'unknown')

        print(f"\n[{i}/{len(tests)}] {test_id} ({query_type})")
        print(f"  Q: {question[:70]}...")

        try:
            # Run question through RAG
            response = await rag.search_and_generate(question)
            actual = response.get('response', 'No response')

            results.append({
                'id': test_id,
                'question': question,
                'expected': expected,
                'actual': actual,
                'query_type': query_type
            })

            # Show snippet of answer
            answer_preview = actual[:100].replace('\n', ' ')
            print(f"  A: {answer_preview}...")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'id': test_id,
                'question': question,
                'expected': expected,
                'actual': f"ERROR: {e}",
                'query_type': query_type
            })

    # Phase 2: Semantic evaluation of all results
    print("\n" + "=" * 60)
    print("PHASE 2: Semantic Evaluation")
    print("=" * 60)

    # Build evaluation prompt - batch in groups to avoid token limits
    batch_size = 5
    all_scores = []

    for batch_start in range(0, len(results), batch_size):
        batch = results[batch_start:batch_start + batch_size]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (len(results) + batch_size - 1) // batch_size

        print(f"\nEvaluating batch {batch_num}/{total_batches}...")

        eval_prompt = """You are evaluating a RAG system's answers for a machinery/equipment database.
Compare each actual answer to the expected answer SEMANTICALLY.

Scoring guidelines:
- 1.0: Fully correct - contains all key facts, numbers match (within ~10%)
- 0.8-0.9: Mostly correct - key info present, minor differences acceptable
- 0.6-0.7: Partially correct - some key facts present but missing important details
- 0.3-0.5: Weak - has some relevant info but misses main points
- 0.0-0.2: Wrong - incorrect info or "no results" when results expected

For equipment queries:
- Check if counts are close (e.g., "75" vs "78" is fine)
- Check if recommended equipment types match
- Check if key specifications are mentioned
- Scenario answers should address the constraints mentioned

Return JSON:
{
  "scores": [{"id": "ID", "score": 0.0-1.0, "reason": "brief explanation"}]
}

Test results to evaluate:
"""

        for r in batch:
            eval_prompt += f"""
---
ID: {r['id']}
Type: {r['query_type']}
Question: {r['question']}
Expected Key Facts: {r['expected'][:600]}
Actual Answer: {r['actual'][:800]}
"""

        try:
            eval_response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You evaluate RAG test results and return JSON scores."},
                    {"role": "user", "content": eval_prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=1500
            )

            eval_result = json.loads(eval_response.choices[0].message.content)
            batch_scores = eval_result.get('scores', [])
            all_scores.extend(batch_scores)

            # Print batch results
            for s in batch_scores:
                status = "PASS" if s['score'] >= 0.7 else "FAIL"
                reason = s.get('reason', '')[:50]
                print(f"  [{status}] {s['id']}: {s['score']:.2f} - {reason}...")

        except Exception as e:
            print(f"  Batch evaluation error: {e}")
            # Add placeholder scores for failed batch
            for r in batch:
                all_scores.append({
                    'id': r['id'],
                    'score': 0.5,
                    'reason': f'Evaluation error: {str(e)[:30]}'
                })

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    if all_scores:
        total_score = sum(s['score'] for s in all_scores) / len(all_scores)
        passed = sum(1 for s in all_scores if s['score'] >= 0.7)

        # Group by query type
        by_type = {}
        for r, s in zip(results, all_scores):
            qtype = r.get('query_type', 'unknown')
            if qtype not in by_type:
                by_type[qtype] = []
            by_type[qtype].append(s['score'])

        print("\nScores by query type:")
        for qtype, scores in sorted(by_type.items()):
            avg = sum(scores) / len(scores)
            print(f"  {qtype}: {avg*100:.1f}% ({len(scores)} tests)")

        print(f"\n{'='*40}")
        print(f"TOTAL SCORE: {total_score * 100:.1f}/100")
        print(f"PASSED: {passed}/{len(all_scores)} ({passed/len(all_scores)*100:.0f}%)")
        print(f"{'='*40}")

        # List failures
        failures = [s for s in all_scores if s['score'] < 0.7]
        if failures:
            print(f"\nFailed tests ({len(failures)}):")
            for f in failures:
                print(f"  - {f['id']}: {f['score']:.2f} - {f.get('reason', 'N/A')[:60]}")
    else:
        print("No scores generated!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Intelligent RAG Test Runner")
    parser.add_argument('-f', '--file', required=True, help='Test file (any format: txt, yaml, json, md)')
    args = parser.parse_args()

    # Handle relative path
    if not os.path.isabs(args.file):
        if not os.path.exists(args.file):
            test_dir = os.path.dirname(__file__)
            args.file = os.path.join(test_dir, os.path.basename(args.file))

    if not os.path.exists(args.file):
        print(f"ERROR: File not found: {args.file}")
        sys.exit(1)

    asyncio.run(run_tests(args.file))
