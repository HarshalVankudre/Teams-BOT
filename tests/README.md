# Tests and Evaluation

## Quickstart

- Unit checks: `python -m unittest tests/test_quality_guards.py`
- Evaluation harness: `python tests/eval_harness.py --cases tests/qa_cases.json --results tests/sample_results.json`

## Results Format

Create a JSON file with this shape:

```
{
  "run_id": "optional-id",
  "cases": [
    {
      "id": "case-id",
      "sql": "SELECT ...",
      "answer": "Model response text",
      "sources": []
    }
  ]
}
```

`tests/eval_harness.py` validates SQL patterns and answer quality rules defined in `tests/qa_cases.json`.
