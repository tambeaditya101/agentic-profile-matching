# Expected Match Report Sections

A generated match report (from `src.reports.match_report.generate_match_report`)
must contain the following markdown sections, in order:

1. `# Candidate Match Report: <Name>` — H1 header with candidate name
2. `*Candidate ID: <id>*` — italicized candidate ID line
3. `## Summary` — narrative explanation
4. `## Scores` — markdown table with must-have / nice-to-have / composite scores
5. `## Must-Have Criteria Breakdown` — per-skill evidence table (only if must_have is non-empty)
6. `## Nice-to-Have Criteria Breakdown` — per-skill evidence table (only if nice_to_have is non-empty)
7. `## Strengths` — bullet list (only if strengths non-empty)
8. `## Gaps` — bullet list (only if gaps non-empty)
9. `## Evidence from Resume` — numbered list of excerpts (only if excerpts non-empty)
10. `## Improvement Suggestions` — bullet list (only if suggestions non-empty)
11. `## Hire Recommendation: <LABEL>` — final verdict, where LABEL is one of:
    - `STRONG HIRE`
    - `NO HIRE`
    - `BORDERLINE`

The test suite verifies that each required section header is present in the
generated markdown. Optional sections (those gated on non-empty input) are
checked only when their input is non-empty.
