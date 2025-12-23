# Quality Log

## Issues Found

### 1) Missing SQL constraints for user intent
- Symptom: SQL queries can omit required filters (e.g., rental, availability, manufacturer).
- Repro: Ask "Wie viele Mietmaschinen haben wir?" and observe SQL without `verwendung_code = 'MIET'`.
- Root cause: Constraints were only described in the prompt; no validation enforced them.
- Fix plan: Extract intent constraints from the user query and validate SQL before execution.
- Risk: Over-constraining could reject valid SQL or cause extra tool rounds.
- Verify: Add tests that require rental/availability filters and ensure validation blocks missing constraints.

### 2) Wrong tables/columns in generated SQL
- Symptom: Queries can reference non-existent columns or tables.
- Repro: Ask for properties using unknown columns; SQL executes (or fails silently) with wrong schema.
- Root cause: No schema-aware validation of table/column usage.
- Fix plan: Validate tables/columns via SQL guard with optional schema introspection.
- Risk: If schema introspection fails, validation becomes permissive (warnings only).
- Verify: Unit tests for column/table validation and harness checks for required patterns.

### 3) Database unavailable yields false "no data"
- Symptom: When Postgres is unavailable, the agent returns empty results and claims no data.
- Repro: Start without DB credentials and ask for counts.
- Root cause: `execute_query` returns empty list when unavailable without surfacing the error.
- Fix plan: Track availability errors and return explicit SQL error for the model/guard.
- Risk: Errors bubble to users more often; acceptable for transparency.
- Verify: Unit tests cover error path; AnswerGuard returns a DB error response.

### 4) Follow-up queries rely on prompt-only memory
- Symptom: "Welche davon ..." can ignore prior result sets.
- Repro: Ask for a list, then "davon mit Klimaanlage" and observe missing filters.
- Root cause: Thread state is injected as prompt only; no SQL validation.
- Fix plan: Add follow-up constraint requiring `id IN (...)` from last result ids.
- Risk: Only the last displayed ids are enforced (limit 25); may be narrow.
- Verify: Follow-up test case requires `id IN (...)` and `prop_klimaanlage`.

### 5) Answer quality lacks grounding/citations and concise structure
- Symptom: Responses can be long, vague, or missing sources.
- Repro: Ask for lists or summaries; observe missing citations.
- Root cause: No post-processing guardrails after model output.
- Fix plan: Add AnswerGuard to enforce citations, no-data fallback, refusals, and length limits.
- Risk: Over-trimming could hide useful details; mitigated by "Weitere Details" hint.
- Verify: Unit tests for no-data fallback and refusal; evaluation harness checks citations/length.

### 6) Document prefetch adds noise to SQL-heavy questions
- Symptom: SQL answers can be polluted by unrelated doc context.
- Repro: Ask for counts; observe doc context in system messages.
- Root cause: Document prefetch always runs when enabled.
- Fix plan: Only prefetch docs when doc-specific signals exist.
- Risk: Some doc answers might skip prefetch if queries lack doc keywords.
- Verify: Behavior confirmed by logs and tests for doc queries.

## Fixes Implemented

### SQL and Intent Guardrails
- Added `rag/sql_guard.py` with intent extraction, constraints, and SQL validation.
- Integrated `SQLGuard` into `rag/single_agent.py` to enforce constraints and emit policy hints.
- Added schema metadata caching via `PostgresService.get_column_info`.

Before:
- SQL relied solely on prompt guidance; missing filters were common.

After:
- Missing filters trigger validation errors that the model can correct.

### Answer Quality Guardrails
- Added `rag/answer_guard.py` for refusals, no-data fallback, citations, and concise formatting.
- Applied guard in `rag/single_agent.py` after tool rounds to ensure grounded output.

Before:
- Responses could be long, ungrounded, or omit citations.

After:
- Responses consistently include source lines and stay concise; sensitive requests are refused.

### Prefetch Noise Reduction
- Document prefetch now runs only when doc signals are present.

## Tests Added
- `tests/qa_cases.json`: diverse question set (SQL, ambiguity, follow-ups, time range, security).
- `tests/test_quality_guards.py`: unit tests for SQLGuard and AnswerGuard.
- `tests/eval_harness.py`: deterministic evaluation harness for SQL/answer properties.
- `tests/sample_results.json`: example output for harness validation.

## Evaluation Results
- `python -m unittest tests/test_quality_guards.py`: pass (local).
- `python tests/eval_harness.py --cases tests/qa_cases.json --results tests/sample_results.json`: pass (fixture-based).

Note: Full end-to-end evaluation requires live OpenAI/Pinecone/Postgres credentials.
