"""
Simple RAG Test Runner
1. Reads questions from test file
2. Runs each question through the app
3. Stores all answers
4. Does semantic comparison at the end
5. Outputs score
"""
import asyncio
import yaml
import json
from typing import List, Dict, Any
from openai import AsyncOpenAI
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.search import RAGSearch
from rag.config import config


async def run_tests(test_file: str) -> None:
    """Run all tests and evaluate semantically at the end"""

    # Load test cases
    print(f"Loading tests from: {test_file}")
    with open(test_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    tests = data.get('test_cases', [])
    print(f"Found {len(tests)} test cases\n")

    # Initialize RAG
    rag = RAGSearch()
    client = AsyncOpenAI(api_key=config.openai_api_key)

    # Store results
    results = []

    # Phase 1: Run all questions through the app
    print("=" * 60)
    print("PHASE 1: Running Questions")
    print("=" * 60)

    for i, test in enumerate(tests, 1):
        question = test['question']
        expected = test['expected_answer']
        test_id = test.get('id', f'TEST-{i}')

        print(f"\n[{i}/{len(tests)}] {test_id}: {question[:50]}...")

        try:
            # Run question through RAG
            response = await rag.search_and_generate(question)
            actual = response.get('response', 'No response')

            results.append({
                'id': test_id,
                'question': question,
                'expected': expected,
                'actual': actual,
                'criteria': test.get('criteria', [])
            })

            print(f"  -> Got answer ({len(actual)} chars)")

        except Exception as e:
            print(f"  -> ERROR: {e}")
            results.append({
                'id': test_id,
                'question': question,
                'expected': expected,
                'actual': f"ERROR: {e}",
                'criteria': test.get('criteria', [])
            })

    # Phase 2: Semantic evaluation of all results
    print("\n" + "=" * 60)
    print("PHASE 2: Semantic Evaluation")
    print("=" * 60)

    # Build evaluation prompt
    eval_prompt = """You are evaluating a RAG system's answers.
For each test, compare the actual answer to the expected answer SEMANTICALLY (not exact match).

Score each answer 0.0 to 1.0:
- 1.0: Fully correct, contains expected information
- 0.7-0.9: Mostly correct, minor issues
- 0.4-0.6: Partially correct
- 0.1-0.3: Mostly wrong but has some relevant info
- 0.0: Completely wrong or "no results" when results expected

Return a JSON object with:
{
  "scores": [{"id": "TEST-ID", "score": 0.0-1.0, "reason": "brief reason"}],
  "total_score": average_score,
  "summary": "overall summary"
}

Here are the test results to evaluate:
"""

    # Add results to prompt
    for r in results:
        eval_prompt += f"""
---
ID: {r['id']}
Question: {r['question']}
Expected: {r['expected']}
Actual: {r['actual'][:500]}...
Criteria: {', '.join(r['criteria']) if r['criteria'] else 'N/A'}
"""

    print("\nEvaluating all results...")

    try:
        eval_response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You evaluate test results and return JSON."},
                {"role": "user", "content": eval_prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=2000
        )

        eval_result = json.loads(eval_response.choices[0].message.content)

        # Print results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        scores = eval_result.get('scores', [])
        for s in scores:
            status = "PASS" if s['score'] >= 0.7 else "FAIL"
            print(f"[{status}] {s['id']}: {s['score']:.2f} - {s['reason'][:60]}...")

        total = eval_result.get('total_score', 0)
        passed = sum(1 for s in scores if s['score'] >= 0.7)

        print("\n" + "=" * 60)
        print(f"FINAL SCORE: {total * 100:.1f}/100")
        print(f"PASSED: {passed}/{len(scores)}")
        print("=" * 60)
        print(f"\nSummary: {eval_result.get('summary', 'N/A')}")

    except Exception as e:
        print(f"Evaluation error: {e}")
        # Fallback: simple scoring
        print("\nFallback scoring (checking for 'keine ergebnisse'):")
        passed = 0
        for r in results:
            has_result = 'keine ergebnisse' not in r['actual'].lower()
            if has_result:
                passed += 1
                print(f"  [OK] {r['id']}")
            else:
                print(f"  [FAIL] {r['id']}")
        print(f"\nSimple score: {passed}/{len(results)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default='test_cases.yaml', help='Test file')
    args = parser.parse_args()

    # Handle relative path - check if it's just a filename
    if not os.path.isabs(args.file):
        # If path doesn't exist as-is, try in tests directory
        if not os.path.exists(args.file):
            test_dir = os.path.dirname(__file__)
            # Only use basename if a path was given
            basename = os.path.basename(args.file)
            args.file = os.path.join(test_dir, basename)

    asyncio.run(run_tests(args.file))
