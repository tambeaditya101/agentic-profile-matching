# Demo Video Guide — How to Record Your Assignment Demo

> **Duration**: 5–6 minutes
> **Audience**: Your instructor / reviewer
> **Tool**: Any screen recorder (QuickTime on Mac, OBS Studio, Loom, etc.)

This guide tells you **exactly what to do, what to say, and what to show** in your demo video. It covers both scenarios: with an LLM (Gemini/Ollama) and without (keyword fallback).

---

## Before You Record

### 1. Set Up Your Environment

```bash
# Make sure everything is installed and working
cd agentic-profile-matching
source venv/bin/activate
pip install -e .
python scripts/ingest_resumes.py

# Check your LLM status — this is what the badge will show
python matching_agent.py info
```

You should see one of these:
- 🟢 **Gemini 2.0 Flash** — ideal for the demo (best quality analysis)
- 🟡 **Ollama (gemma2:9b)** — good for the demo (LLM-powered, fully offline)
- 🔴 **Keyword Fallback** — acceptable for the demo, but mention it explicitly

### 2. Decide Which Scenario to Demo

| If you have... | Demo this | Quality |
|---|---|---|
| A free Gemini API key | Scenario A (LLM-powered) | Best — shows full AI capabilities |
| Ollama installed locally | Scenario A (LLM-powered) | Good — shows full AI capabilities offline |
| Neither | Scenario B (keyword fallback) | Acceptable — shows the pipeline works, with honest caveats |

**Recommendation**: Spend 2 minutes getting a free Gemini key from https://aistudio.google.com/apikey — it makes the demo dramatically more impressive. Add it to `.env`:
```
GEMINI_API_KEY=your_key_here
```
Then restart the app. The sidebar badge will turn 🟢 green.

### 3. Prepare Your Screen

- Close all other apps (browser tabs, IDE, Slack, etc.)
- Use a clean desktop wallpaper
- Increase font size in your terminal: `Cmd +` (Mac) or `Ctrl +` (Windows/Linux)
- Open the project in your IDE (for the architecture overview part)
- Have the Streamlit app open at http://localhost:8501

### 4. Test Run First

**Before recording**, do a complete dry-run of the demo to make sure everything works. This avoids awkward pauses or errors during recording.

---

## The Demo Script (5–6 minutes)

### 0:00 – 0:30 — Introduction

**What to show**: Your face (if comfortable) or a title card with the project name.

**What to say**:
> "Hi, this is my Agentic Profile Matching agent. It's a LangGraph-based system that takes a job description, retrieves matching candidates from a resume corpus using RAG, runs them through a 3-round screening pipeline, and provides full explainability of every ranking decision. The entire stack runs on free-tier infrastructure — Gemini Free API, local ChromaDB, and ONNX embeddings."

**On-screen**: Show the project README or the state machine diagram (`docs/state_machine_diagram.png`).

### 0:30 – 1:00 — Architecture Overview

**What to show**: Open `docs/state_machine_diagram.png` and point to the key components.

**What to say**:
> "Here's the state machine. The linear pipeline runs: parse JD → extract requirements → search resumes → rank candidates → generate reports. Then the human feedback loop handles interactive queries: refine requirements, compare candidates, explain rankings, generate interview questions. The 3-round screening funnel narrows from ~100 candidates to a final hire/no-hire recommendation."

**Key points to mention**:
- Built with LangGraph (StateGraph with TypedDict state)
- 3-round screening: initial → deep analysis → final
- Every decision is explainable (markdown reports with evidence)
- Free-tier only (no paid services)

### 1:00 – 1:15 — Show the LLM Status Badge

**What to show**: Launch the Streamlit app (`python matching_agent.py streamlit`), point to the LLM badge at the top of the sidebar.

**What to say** (adapt to your setup):
- If 🟢 green: "I'm using the Gemini 2.0 Flash free tier for LLM-powered analysis."
- If 🟡 yellow: "I'm using Ollama with the gemma2:9b model running locally — fully offline, no API key."
- If 🔴 red: "I'm running without an LLM, using the keyword-based fallback. This still demonstrates the full pipeline, but the analysis is less nuanced. With a Gemini API key, the same flow produces LLM-powered reasoning."

### 1:15 – 2:15 — Run the Pipeline

**What to show**:
1. In the Streamlit sidebar, click **"Browse files"** and select `tests/fixtures/sample_jd.txt`
2. Click the blue **▶ Run Pipeline** button
3. Watch the chat area populate with the ranked shortlist
4. Point to the sidebar: requirements panel, screening progress (3 rounds with checkmarks)

**What to say**:
> "I'll load the sample JD for a Senior React Frontend Developer. The agent parses it, extracts 10 must-have and 11 nice-to-have requirements, retrieves candidates from the vector store, and runs the 3-round screening pipeline."

**After it completes**:
> "The pipeline found 4 candidates, ran them through 3 rounds, and shortlisted Alice Johnson with a composite score of 0.86 and a STRONG HIRE recommendation. The sidebar shows the extracted requirements and the 3 completed screening rounds."

### 2:15 – 3:00 — Show a Match Report

**What to show**:
1. Click the **📄 Reports** tab
2. Select "Alice Johnson" from the dropdown
3. Scroll through the report sections

**What to say**:
> "Each candidate gets a detailed match report. Here's Alice's. It includes a summary, a scores table breaking down must-have vs. nice-to-have, a per-skill evidence table with excerpts from her resume, her strengths, her gaps, improvement suggestions, and the final hire recommendation. Every claim is backed by evidence from the resume — no hallucinated content."

**Point to**:
- The "Evidence from Resume" section (direct quotes)
- The "Must-Have Criteria Breakdown" table (per-skill evidence)
- The "Hire Recommendation" at the bottom

### 3:00 – 3:45 — Compare Candidates

**What to show**:
1. Switch back to the **💬 Chat** tab
2. Type: `Compare the top 3 candidates side by side`
3. Press Enter
4. Show the comparison table that appears

**What to say**:
> "I can ask the agent to compare candidates. The intent classifier routes my message to the compare_candidates node, which produces a head-to-head comparison table with scores and a narrative summary."

**If you have 3+ candidates**: the table will have 3 columns. If you only have 1-2 (common with the sample data), say:
> "Since the pipeline shortlisted fewer candidates, let me compare the top 2."

### 3:45 – 4:30 — Explain Rankings

**What to show**:
Type: `Why did Alice rank higher than the second candidate?`

**What to say**:
> "I can ask the agent to explain its ranking decisions. The explain_ranking node generates a narrative explanation citing specific evidence — years of experience, particular skills, leadership experience — rather than just restating the scores."

**Point to** the explanation response and read 1-2 key points from it.

### 4:30 – 5:00 — Refine Requirements

**What to show**:
Type: `Drop the AWS requirement and re-rank`

**What to say**:
> "I can iteratively refine the requirements mid-conversation. When I drop the AWS requirement, the agent re-ranks all candidates, increments the requirements version, and produces a delta summary explaining what changed — who moved up, who moved down, and why."

**Point to**:
- The "Ranking Updated" heading
- The delta summary
- The sidebar's requirements panel (AWS should be gone)

### 5:00 – 5:30 — Generate Interview Questions

**What to show**:
Type: `Generate interview questions for Alice`

**What to say**:
> "Finally, I can generate targeted interview questions for a candidate. Each question is tagged with a category — technical, behavioral, situational — and targets a specific gap in the candidate's profile. The agent also suggests follow-up questions for deeper probing."

**Point to** the questions and mention the category tags and follow-ups.

### 5:30 – 6:00 — Summary & Q&A

**What to show**: Switch back to the README or the state machine diagram.

**What to say**:
> "To recap: the agent parses a JD, retrieves candidates via RAG, runs a 3-round screening pipeline, and supports interactive refinement, comparison, explanation, and interview question generation — all explainable, all on free-tier infrastructure. The submission entry point is `matching_agent.py`, the state machine diagram is in `docs/`, and the full test suite of 388 tests passes. Questions?"

---

## Scenario B — Demo Without an LLM (Keyword Fallback)

If you're recording without a Gemini key or Ollama, use the same script above but make these adjustments:

### At 1:00 — Be Honest About the LLM Status

**What to say**:
> "I'm running the agent in keyword-fallback mode — no LLM is configured. This means requirement extraction uses a deterministic keyword-based parser instead of LLM-based structured output, and scoring uses keyword matching instead of LLM reasoning. The pipeline is fully functional, but the analysis is less nuanced. With a free Gemini API key, the same flow produces LLM-powered reasoning, comparison narratives, and explanations."

### Adjust Expectations

- The report's "Summary" will be a keyword count (e.g., "13/20 skills matched") instead of a narrative
- The comparison will use the fallback table (raw scores, no narrative)
- The explanation will be a simple score comparison
- Interview questions will be generic (gap-targeted, but no LLM-generated follow-ups)

### Show That It Still Works

The key point to demonstrate: **the pipeline works end-to-end without an LLM**. This is a deliberate design decision for graceful degradation — show it as a strength:

> "A key design decision was graceful degradation. If no LLM is available — say, in an air-gapped enterprise environment — the agent still produces ranked shortlists and match reports using keyword-based heuristics. The LLM enhances quality but is not a hard dependency."

---

## Tips for a Great Demo

### Do
- **Speak slowly and clearly** — 5 minutes is plenty of time
- **Point to things on screen** with your mouse cursor
- **Explain what's happening** at each step, don't just click silently
- **Mention the architecture** (LangGraph, RAG, 3-round screening, explainability)
- **Show the code structure** briefly (project tree in your IDE)
- **Mention the test suite**: "388 tests pass, covering all 7 architecture scenarios"
- **End with the submission artifacts**: `matching_agent.py`, `docs/state_machine_diagram.png`, `docs/demo_script.md`

### Don't
- Don't show installation steps (boring — just say "pip install -e ." in passing)
- Don't read the README word-for-word
- Don't apologize for the keyword fallback if you're using it — frame it as a feature
- Don't spend more than 30 seconds on any single screen
- Don't show error messages (if something breaks, edit it out)

### If Something Breaks
- Stay calm, say "let me try that again", and re-run the step
- If it keeps breaking, cut to a different part of the demo and come back to it
- Have a backup: pre-run the pipeline once before recording so the state is warm

---

## Recording Tools

| Tool | Platform | Cost | Notes |
|------|----------|------|-------|
| **QuickTime Player** | macOS | Free | File → New Screen Recording |
| **OBS Studio** | All | Free | Professional grade, more setup |
| **Loom** | Browser | Free (5 min) | Easy sharing, auto-generates link |
| **ScreenRec** | Windows | Free | Simple, no watermark |
| **Screencast-O-Matic** | All | Free (15 min) | Has editing tools |

### Recording Settings
- **Resolution**: 1920×1080 (1080p) — don't go higher, file gets too big
- **Frame rate**: 30 fps is fine
- **Audio**: Use a real microphone if possible (not the laptop built-in)
- **Format**: MP4 (most compatible)

---

## Submission Checklist

Before you submit, make sure you have:

- [ ] **Demo video** (5–6 minutes, MP4)
- [ ] **Project zip** (the `agentic-profile-matching.zip` file)
- [ ] **README.md** (included in the zip)
- [ ] **State machine diagram** (`docs/state_machine_diagram.png`)
- [ ] **Demo script** (`docs/demo_script.md`)
- [ ] **matching_agent.py** entry point (in the zip root)
- [ ] **All tests pass**: `pytest tests/ -q -m "not integration"` → 388 passed

---

## Quick Reference: Demo Commands

```bash
# Pre-demo setup
cd agentic-profile-matching
source venv/bin/activate
python scripts/ingest_resumes.py
python matching_agent.py info          # verify LLM status

# Launch the app
python matching_agent.py streamlit     # then open http://localhost:8501

# Chat prompts to type during the demo
Compare the top 3 candidates side by side
Why did Alice rank higher than the second candidate?
Drop the AWS requirement and re-rank
Generate interview questions for Alice
```

---

## What If the Reviewer Asks Questions?

Be prepared to answer:

**Q: Why LangGraph instead of plain LangChain?**
A: LangGraph gives us explicit state management with TypedDict, conditional routing, and a visual state machine. Plain LangChain chains are linear; LangGraph supports the interactive feedback loop with conditional edges.

**Q: Why ChromaDB instead of Pinecone?**
A: ChromaDB is local-first, requires no API key, and has no cost. Pinecone is a hosted service that would require a paid account for any real volume. The architecture spec (Section 12) explicitly requires free-tier only.

**Q: How does the 3-round screening work?**
A: Round 1 does broad RAG retrieval (top 100) plus a keyword filter to narrow to 10. Round 2 retrieves full resumes, verifies skill depth, checks for red flags (employment gaps, job-hopping), and narrows to 5-7. Round 3 generates the final hire/no-hire recommendation with evidence.

**Q: What's the composite score formula?**
A: `0.7 × must_have_score + 0.3 × nice_to_have_score`. Must-have criteria are the primary filter; nice-to-have breaks ties.

**Q: How are scores explainable?**
A: Every candidate has a markdown match report with per-skill evidence tables, strengths, gaps, resume excerpts, and improvement suggestions. The "Evidence from Resume" section quotes the exact resume text that justified each match.

**Q: What happens if the LLM is unavailable?**
A: The agent has a keyword-based fallback for requirement extraction and scoring. Quality is lower (keyword counts instead of narrative reasoning), but the pipeline produces ranked shortlists and reports. This is a deliberate design decision for graceful degradation in air-gapped environments.

**Q: How many tests are there?**
A: 388 tests pass (excluding 9 integration tests that require a live LLM). They cover all 7 architecture scenarios: happy path, refinement, comparison, explanation, questions, edge cases, and multi-round screening.

---

Good luck with your demo! 🎬
