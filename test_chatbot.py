"""
Test script to verify chatbot answers against expected results.
Tests the hybrid RAG system with questions from TEST_QUESTIONS.md
"""
import os
import sys
import re
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Import the search module
from rag.search import RAGSearch

# Test questions with expected answers (key facts to verify)
TEST_CASES = [
    # AGGREGATION QUERIES
    {
        "id": "Q1",
        "question": "Wie viele Geräte haben wir insgesamt im Bestand?",
        "expected_keywords": ["2395", "2.395", "2,395"],
        "category": "AGGREGATION"
    },
    {
        "id": "Q3",
        "question": "Was ist die schwerste Maschine im Bestand?",
        "expected_keywords": ["Sennebogen", "643", "42000", "42.000", "42,000"],
        "category": "AGGREGATION"
    },
    {
        "id": "Q4",
        "question": "Was ist der leichteste Bagger?",
        "expected_keywords": ["Caterpillar", "300.9D", "935"],
        "category": "AGGREGATION"
    },
    {
        "id": "Q5",
        "question": "Welche Maschine hat die höchste Motorleistung?",
        "expected_keywords": ["Wirtgen", "W 150", "261"],
        "category": "AGGREGATION"
    },
    {
        "id": "Q8",
        "question": "Wie viele Geräte haben eine Klimaanlage?",
        "expected_keywords": ["75"],
        "category": "AGGREGATION"
    },
    {
        "id": "Q10",
        "question": "Wie viele Geräte haben einen Deutz Motor?",
        "expected_keywords": ["79"],
        "category": "AGGREGATION"
    },

    # FILTER QUERIES
    {
        "id": "Q11",
        "question": "Welche Mobilbagger haben eine Klimaanlage?",
        "expected_keywords": ["6", "Wacker", "EW65", "Caterpillar", "M 317", "Liebherr"],
        "category": "FILTER"
    },
    {
        "id": "Q12",
        "question": "Welche Bagger haben einen Tiltrotator?",
        "expected_keywords": ["3", "Liebherr", "A 918"],
        "category": "FILTER"
    },
    {
        "id": "Q16",
        "question": "Welche Walzen haben Oszillation?",
        "expected_keywords": ["2", "Bomag", "BW 154", "BW 174"],
        "category": "FILTER"
    },

    # COMPARISON QUERIES
    {
        "id": "Q19",
        "question": "Vergleiche Kettenbagger und Mobilbagger",
        "expected_keywords": ["Kettenbagger", "Mobilbagger", "21", "53"],
        "category": "COMPARISON"
    },
    {
        "id": "Q20",
        "question": "Vergleiche Tandemwalze und Walzenzug",
        "expected_keywords": ["Tandemwalze", "Walzenzug", "95", "35"],
        "category": "COMPARISON"
    },

    # COMPLEX MULTI-FILTER QUERIES
    {
        "id": "Q22",
        "question": "Welche Bagger über 15 Tonnen haben Klimaanlage UND Stufe V?",
        "expected_keywords": ["8", "Liebherr", "Caterpillar"],
        "category": "COMPLEX"
    },
    {
        "id": "Q23",
        "question": "Welche Liebherr Bagger haben Hammerhydraulik?",
        "expected_keywords": ["5", "R 914", "A 914", "R 926"],
        "category": "COMPLEX"
    },

    # LOOKUP QUERIES
    {
        "id": "Q26",
        "question": "Zeige mir den Caterpillar 320-07C",
        "expected_keywords": ["320-07C", "Kettenbagger", "22600", "22.600", "129"],
        "category": "LOOKUP"
    },

    # EDGE CASES
    {
        "id": "E1",
        "question": "Wie viele Tesla Bagger haben wir?",
        "expected_keywords": ["0", "keine"],
        "category": "EDGE"
    },
]

async def check_answer_with_llm(client, question: str, response: str, expected_keywords: list) -> tuple:
    """Use LLM to check if response is semantically correct"""
    expected_info = ", ".join(str(k) for k in expected_keywords)

    prompt = f"""Du bist ein großzügiger Testprüfer. Bewerte ob die Antwort INHALTLICH KORREKT ist.

FRAGE: {question}

ERWARTETE INFOS (nicht alle müssen exakt vorkommen): {expected_info}

ANTWORT: {response[:2000]}

SEI GROSSZÜGIG - Markiere als BESTANDEN wenn:
- Die Hauptinformation korrekt ist (z.B. richtiges Modell, richtige Anzahl)
- Zahlen im richtigen Bereich sind (±10% Toleranz)
- Formatunterschiede: 2395 = 2.395 = 2,395 = "2395"
- Ähnliche Namen: "Sennebogen 643 R Serie E" = "Sennebogen 643"
- "0", "keine", "None", "kein" sind alle gleichwertig
- Duplikate in Listen sind OK (dasselbe Gerät mehrfach)
- Wenn die wesentlichen Modelle/Hersteller genannt werden, ist es OK

FAIL nur wenn:
- Die Antwort komplett falsch ist
- Wichtige Zahlen stark abweichen (>50% Fehler)
- Die gefragte Information gar nicht vorhanden ist

Antworte NUR mit JSON:
{{"passed": true/false, "reason": "kurze Begründung"}}"""

    try:
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        result_text = completion.choices[0].message.content.strip()
        # Parse JSON
        import json
        # Clean up response
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        result = json.loads(result_text.strip())
        passed = result.get("passed", False)
        reason = result.get("reason", "")
        return passed, 1.0 if passed else 0.0, reason
    except Exception as e:
        print(f"LLM evaluation error: {e}")
        # Fallback to keyword check
        response_lower = response.lower()
        found = sum(1 for k in expected_keywords if str(k).lower() in response_lower)
        score = found / len(expected_keywords) if expected_keywords else 0
        return score >= 0.5, score, "Fallback keyword check"

async def run_tests():
    """Run all test cases and report results"""
    from openai import AsyncOpenAI

    print("=" * 70)
    print("SEMA CHATBOT TEST SUITE (LLM Evaluation)")
    print("=" * 70)
    print()

    # Initialize RAG Search and OpenAI client for evaluation
    rag = RAGSearch()
    eval_client = AsyncOpenAI()

    results = {
        "AGGREGATION": {"passed": 0, "total": 0},
        "FILTER": {"passed": 0, "total": 0},
        "COMPARISON": {"passed": 0, "total": 0},
        "COMPLEX": {"passed": 0, "total": 0},
        "LOOKUP": {"passed": 0, "total": 0},
        "EDGE": {"passed": 0, "total": 0},
    }

    detailed_results = []

    for test in TEST_CASES:
        print(f"\n[{test['id']}] {test['category']}")
        print(f"Q: {test['question']}")
        print("-" * 50)

        try:
            # Call search_and_generate function (async)
            result = await rag.search_and_generate(test['question'])
            response = result.get("response", "")

            # Check answer with LLM
            passed, score, reason = await check_answer_with_llm(
                eval_client, test['question'], response, test['expected_keywords']
            )

            # Update results
            results[test['category']]['total'] += 1
            if passed:
                results[test['category']]['passed'] += 1

            # Print result
            status = "PASS" if passed else "FAIL"
            print(f"Status: {status}")
            print(f"Reason: {reason}")
            print(f"Response preview: {response[:250]}...")

            detailed_results.append({
                "id": test['id'],
                "question": test['question'],
                "passed": passed,
                "reason": reason,
                "response": response[:500]
            })

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[test['category']]['total'] += 1
            detailed_results.append({
                "id": test['id'],
                "question": test['question'],
                "passed": False,
                "score": 0,
                "error": str(e)
            })

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    total_passed = 0
    total_tests = 0

    for category, counts in results.items():
        if counts['total'] > 0:
            pct = counts['passed'] / counts['total'] * 100
            print(f"{category:15} {counts['passed']}/{counts['total']} ({pct:.0f}%)")
            total_passed += counts['passed']
            total_tests += counts['total']

    print("-" * 30)
    overall_pct = total_passed / total_tests * 100 if total_tests > 0 else 0
    print(f"{'OVERALL':15} {total_passed}/{total_tests} ({overall_pct:.0f}%)")
    print("=" * 70)

    return detailed_results, results

if __name__ == "__main__":
    asyncio.run(run_tests())
