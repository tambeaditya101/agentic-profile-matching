# Demo Script — Agentic Profile Matching Agent

> **Duration**: 5–6 minutes
> **Audience**: Reviewer / instructor
> **Setup**: Two terminals side-by-side
> - Terminal 1: project root, ready to launch Streamlit
> - Terminal 2: project root, ready to launch CLI
>
> **Prerequisites**:
> - `pip install -e .` has been run
> - `python scripts/ingest_resumes.py` has been run (sample resumes indexed)
> - Either `GEMINI_API_KEY` is set in `.env` **or** Ollama is running locally
> - Sample JD available at `tests/fixtures/sample_jd.txt`

---

## 0:00 – 0:30 — Introduction & Architecture

**Narration**:
> "This is the Agentic Profile Matching agent — a LangGraph-based system that
> takes a job description, retrieves matching candidates from a resume corpus
> via RAG, and runs them through a 3-round screening pipeline with full
> explainability. Let me walk you through the architecture quickly."

**On-screen actions**:
1. Open `docs/state_machine_diagram.png` (or the architecture.md Section 5
   diagram in a Markdown preview).
2. Point to the linear pipeline: `ParseJD → ExtractRequirements →
   SearchResumes → RankCandidates → GenerateReport`.
3. Point to the `HumanFeedbackLoop` compound state with its five
   interactive branches: refine, compare, questions, explain, report.

**Key talking points**:
- 8-phase implementation plan, fully free-tier (Gemini Free / Ollama / ChromaDB local).
- State is a single `AgentState` TypedDict with 14 fields, threaded through every node.
- All scores are explainable: each candidate has a markdown match report.

---

## 0:30 – 1:30 — Launch UI & Run the Pipeline

**Narration**:
> "I'll launch the Streamlit interface now and feed it the sample JD for a
> Senior React Frontend Developer."

**On-screen actions**:
1. Terminal 1: `python matching_agent.py streamlit` (or `streamlit run ui/streamlit_app.py`).
2. Browser opens automatically. Point at the sidebar:
   - JD upload widget
   - Empty Requirements panel ("No requirements extracted yet")
   - Empty Screening Progress (all 3 rounds "pending")
3. Click "Browse…" → select `tests/fixtures/sample_jd.txt` (or paste the JD
   text into the text area).
4. Click `▶ Run Pipeline`.

**Expected agent response** (in the chat area):
```
Pipeline complete. Screening rounds: R1: 4, R2: 2, R3: 2.
Top 2 candidates:

| # | Candidate              | Composite | Must | Nice | Recommendation |
|---|------------------------|-----------|------|------|----------------|
| 1 | Alice Johnson          | 0.85      | 0.90 | 0.70 | STRONG HIRE    |
| 2 | Bob Smith              | 0.65      | 0.70 | 0.50 | BORDERLINE     |
```

**Sidebar should now show**:
- Requirements panel populated (React, TypeScript as must-have; AWS as nice-to-have; 5+ years; BS degree; frontend/web domain keywords).
- Screening progress: ✓ Round 1 (4 evaluated → 3 shortlisted), ✓ Round 2 (3 evaluated → 2 shortlisted), ✓ Round 3 (2 evaluated → 2 shortlisted).
- Generated reports count.

**Key talking points**:
- Three-round funnel: RAG retrieval + keyword filter → deep analysis with red flags → final hire/no-hire.
- The composite score is `0.7 × must_have + 0.3 × nice_to_have`.
- Reports were generated for every shortlisted candidate.

---

## 1:30 – 2:30 — Match Reports & Explainability

**Narration**:
> "Each candidate has a detailed match report. Let me show you Alice's."

**On-screen actions**:
1. Click the **📄 Reports** tab at the top of the main area.
2. Select "Alice Johnson" from the dropdown.
3. Scroll through the report — show the Summary, Scores table, Must-Have
   Criteria Breakdown, Strengths, Gaps, Evidence from Resume, Hire
   Recommendation.

**Expected report sections**:
- `## Summary` — narrative explanation of the score.
- `## Scores` — markdown table with must-have / nice-to-have / composite.
- `## Must-Have Criteria Breakdown` — per-skill evidence excerpts.
- `## Strengths` — bullet list of matched strengths.
- `## Gaps` — bullet list of missing/weak areas.
- `## Evidence from Resume` — direct excerpts from the resume.
- `## Improvement Suggestions` — actionable suggestions for borderline candidates.
- `## Hire Recommendation: STRONG HIRE` — final verdict.

**CLI alternative** (if no browser):
```bash
python matching_agent.py cli --jd tests/fixtures/sample_jd.txt
# then at the prompt:
> show report for Alice
```

**Key talking points**:
- Every claim in the report is backed by a resume excerpt — no hallucinated evidence.
- Improvement suggestions are generated for borderline candidates (e.g., "Gain AWS certification").
- Hire recommendations map directly to composite score thresholds.

---

## 2:30 – 3:30 — Interactive Refinement & Re-ranking

**Narration**:
> "Now I'll demonstrate the iterative refinement feature. Suppose the hiring
> manager decides AWS is actually a must-have, not a nice-to-have."

**On-screen actions**:
1. Switch back to the **💬 Chat** tab.
2. Type into the chat input:
   ```
   Drop the AWS requirement and add TypeScript as a must-have
   ```
   (or click the "Drop AWS" suggestion chip in the sidebar).
3. Press Enter.

**Expected agent response**:
```
### Ranking Updated

Removed AWS from nice-to-have.
TypeScript moved from nice-to-have to must-have (weight 1.00).
New requirements version: 2.

Ranking changes:
- Alice Johnson: 0.85 → 0.88 (↑1)
- Bob Smith: 0.65 → 0.52 (↓1)
- Carol Williams: 0.55 → 0.45 (eliminated, below threshold)
```

**Key talking points**:
- The agent diffs the old vs. new rankings and produces a delta summary.
- The `requirements_version` counter increments from 1 to 2.
- Candidates below the threshold (0.30) are eliminated.
- Reports are regenerated for the new shortlist.

**Follow-up to show**: Type "show me the refined shortlist" or click "Status"
in the CLI to see the updated ranking.

---

## 3:30 – 4:15 — Compare Candidates & Explain Rankings

**Narration**:
> "Let's compare the top 2 candidates side-by-side and ask the agent why
> Alice ranked higher than Bob."

**On-screen actions**:
1. Type into chat:
   ```
   Compare the top 2 candidates side by side
   ```
2. Agent returns a markdown table with criteria rows (Overall Score,
   Must-Have, Nice-to-Have, Recommendation) and one column per candidate.
3. Now type:
   ```
   Why did Alice rank higher than Bob?
   ```
4. Agent returns a narrative explanation referencing specific evidence:
   - "Alice has 5 years of React experience vs Bob's 3 years"
   - "Alice has explicit TypeScript experience; Bob has only JavaScript"
   - "Alice led a React migration; Bob built React components"

**Key talking points**:
- Intent classification routes the message to the correct node
  (`compare` → `compare_candidates`, `explain` → `explain_ranking`).
- Comparisons use the LLM with structured output for a consistent schema.
- Explanations cite specific resume excerpts, not just scores.

---

## 4:15 – 4:45 — Generate Interview Questions

**Narration**:
> "Finally, let's generate targeted interview questions for Alice based on
> the gaps in her profile."

**On-screen actions**:
1. Type into chat:
   ```
   Generate interview questions for Alice
   ```
   (or click the "Interview questions" suggestion chip).
2. Agent returns 5 questions, each tagged with:
   - Category (technical / behavioral / situational)
   - Difficulty (easy / medium / hard)
   - Targeted gap (e.g., "AWS experience", "leadership")
   - Follow-up questions

**Expected output**:
```
### Interview Questions for Alice Johnson

**1. [TECHNICAL / medium]** Walk me through your React migration at TechCorp.
   - _Targets gap: large-scale React architecture_
   - _Follow-ups:_
     - What performance challenges did you encounter?
     - How did you measure success?

**2. [TECHNICAL / hard]** How would you design a state management strategy
for a 200+ component React application?
   - _Targets gap: state management depth_

**3. [BEHAVIORAL / easy]** How do you approach mentoring junior developers?
...
```

**Key talking points**:
- Questions are targeted at gaps in the candidate's profile, not generic.
- Each question has follow-ups to enable deep-dive interviewing.
- The same prompt template works for any candidate — fully data-driven.

---

## 4:45 – 5:00 — Multi-Round Screening Walkthrough

**Narration**:
> "Let me show the screening rounds briefly. In the sidebar, you can see
> all 3 rounds completed with their statistics."

**On-screen actions**:
1. Point at the sidebar "Screening Progress" section.
2. Show:
   - Round 1 (Initial): broad RAG retrieval, 100 candidates → top 10 by keyword filter.
   - Round 2 (Deep): full resume review, red-flag detection → top 5–7.
   - Round 3 (Final): hire / no-hire / borderline recommendation with evidence.
3. Mention the red-flag detection: employment gaps >3 months, job-hopping
   (>3 jobs in 2 years), inconsistent dates.

**CLI alternative** (if running CLI):
```
> status
```
prints the full screening progress + requirements + shortlist table.

---

## 5:00 – 5:30 — Summary & Q&A

**Narration**:
> "To recap: the agent parses a JD into structured requirements, retrieves
> candidates via RAG, runs a 3-round screening pipeline, and supports
> interactive refinement, comparison, explanation, and interview question
> generation — all explainable, all on free-tier infrastructure.
>
> The submission entry point is `matching_agent.py`, which launches either
> the Streamlit UI or the CLI REPL. The state machine diagram is in
> `docs/state_machine_diagram.png`. The full test suite of N tests passes
> with X% coverage.
>
> Questions?"

**On-screen actions**:
1. Show the project README and directory structure.
2. Show `docs/state_machine_diagram.png` one more time.
3. Run `pytest tests/ -q --cov=src` in a terminal to show the test summary
   and coverage report.

---

## Quick Reference: Demo Commands

```bash
# Setup (one-time)
pip install -e .
python scripts/ingest_resumes.py

# Launch UI (primary)
python matching_agent.py streamlit
# or
streamlit run ui/streamlit_app.py

# Launch CLI (fallback)
python matching_agent.py cli --jd tests/fixtures/sample_jd.txt

# Run tests
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing

# Export reports after a session
# (CLI auto-exports on 'done'; Streamlit: click "Export Reports" in sidebar)
```

## Quick Reference: Chat Prompts to Type

| Phase | Prompt |
|---|---|
| Refinement | `Drop the AWS requirement and re-rank` |
| Refinement | `Add TypeScript as a must-have skill` |
| Comparison | `Compare the top 3 candidates side by side` |
| Explanation | `Why did Alice rank higher than Bob?` |
| Questions | `Generate interview questions for Alice` |
| Report | `Show the full match report for Alice` |
| Status | `status` (CLI only) |
| Exit | `done` |

---

## Troubleshooting During Demo

| Symptom | Likely Cause | Fix |
|---|---|---|
| "No LLM available" error | Missing `GEMINI_API_KEY` and Ollama not running | Set key in `.env` or run `ollama pull gemma2:9b` |
| Empty shortlist | ChromaDB not indexed | Run `python scripts/ingest_resumes.py` |
| Streamlit won't start | Port 8501 in use | `streamlit run ... --server.port 8502` |
| LLM scoring fallback warnings | LLM rate-limited or offline | Acceptable — keyword fallback kicks in automatically |
| Comparison fails | Only 1 candidate in shortlist | Refine requirements to broaden the pool, or use a different JD |
