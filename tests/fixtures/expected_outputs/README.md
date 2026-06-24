# Expected Outputs — Regression Reference Fixtures

This directory holds reference outputs that the test suite compares against
to detect regressions. Each file documents the expected shape (not exact
content) of a particular agent output, since LLM-generated content varies
between runs.

Files in this directory:

- `expected_requirements_schema.json` — schema for `state["requirements"]`
- `expected_candidate_match_schema.json` — schema for one entry in `state["current_shortlist"]`
- `expected_screening_round_schema.json` — schema for one entry in `state["screening_rounds"]`
- `expected_comparison_result_schema.json` — schema for `state["comparison_result"]`
- `expected_match_report_sections.md` — required sections in a generated match report
- `expected_intent_routing.md` — expected intent classification for sample inputs

The test suite (`tests/test_agent_flows.py`) loads these schemas and
verifies that agent outputs conform to them. We test **shape**, not exact
content, because LLM outputs vary.
