# Agentic Profile Matching

An **AI-powered candidate-job matching agent** built with **LangGraph** that takes a job description (JD), retrieves matching candidates from a resume corpus via RAG, runs them through a 3-round screening pipeline, and provides full explainability of every ranking decision — all on **free-tier infrastructure** (Groq / Gemini / Ollama / ChromaDB local, with keyword fallback for no-LLM scenarios).

```
START → Parse JD → Extract Requirements → Search Resumes → Rank Candidates
      → Generate Report → Human Feedback Loop → END
                            ├── refine requirements  → re-rank
                            ├── compare candidates   → comparison table
                            ├── explain ranking      → narrative explanation
                            ├── generate questions   → interview questions
                            └── show report          → full match report
```

---

## Table of Contents

1. [Quick Start (5 minutes)](#1-quick-start-5-minutes)
2. [What You Need Installed](#2-what-you-need-installed)
3. [Running the App](#3-running-the-app)
4. [Feature Walkthrough — Testing Every Feature](#4-feature-walkthrough--testing-every-feature)
5. [The Sample JD and Resumes](#5-the-sample-jd-and-resumes)
6. [Adding Your Own Resumes](#6-adding-your-own-resumes)
7. [Troubleshooting](#7-troubleshooting)
8. [Architecture Overview](#8-architecture-overview)
9. [Running Tests](#9-running-tests)
10. [Project Structure](#10-project-structure)

---

## 1. Quick Start (5 minutes)

```bash
# 1. Unzip
unzip agentic-profile-matching.zip
cd agentic-profile-matching

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies (this is the answer to "do I need to install dependencies?")
pip install -e .

# 4. Index the sample resumes into the vector store
python scripts/ingest_resumes.py

# 5. (Optional) Add your free Gemini API key — OR skip this and use the
#    keyword-based fallback that works with NO LLM at all
cp .env.example .env
# Edit .env and add your free API keys (set BOTH for maximum reliability):
#   GEMINI_API_KEY=your_gemini_key    (get at https://aistudio.google.com/apikey)
#   GROQ_API_KEY=your_groq_key        (get at https://console.groq.com/keys — RECOMMENDED, higher free quota)

# 6. Launch the Streamlit UI
python matching_agent.py streamlit
# Or directly: streamlit run ui/streamlit_app.py
```

Open **http://localhost:8501** in your browser. You should see the chat interface with a sidebar on the left.

> **That's it.** Steps 1–4 are one-time setup. After that, just run `python matching_agent.py streamlit` to launch.

---

## 2. What You Need Installed

### Required (no way around these)

| Tool | Version | Why | Install |
|------|---------|-----|---------|
| **Python** | 3.11+ (3.12 or 3.13 recommended) | Runtime | https://python.org |
| **pip** | any recent | Package installer | Comes with Python |
| **The project dependencies** | — | langgraph, chromadb, streamlit, etc. | `pip install -e .` |

### Optional — LLM Providers (set at least one for AI-powered analysis)

The agent has a **4-tier fallback chain**: Groq → Gemini → Ollama → Keyword fallback. Set as many as you want — if one fails (e.g., quota exhausted), the next takes over automatically.

| Provider | Free Tier | Why | Get a Key |
|----------|-----------|-----|-----------|
| **Groq** (recommended) | 30 RPM, 14,400 req/day | Fastest inference (LPU-based), highest free quota, Llama 3.3 70B | https://console.groq.com/keys |
| **Gemini** | 15 RPM, 1,500 req/day | High quality, Google's flagship model | https://aistudio.google.com/apikey |
| **Ollama** (offline) | Unlimited (local) | Fully offline, no API key, runs `gemma2:9b` locally (~5 GB download) | https://ollama.com |
| **Keyword fallback** | Always available | No LLM, deterministic keyword-based extraction (lower quality) | (automatic) |

**Recommendation**: Set **both** Groq and Gemini keys in `.env`. Groq has the highest free quota and is the most reliable; Gemini is a backup. The agent automatically falls back between them at runtime.

### You do NOT need

- ❌ `torch` or `torchvision` — these are optional `transformers` deps that ChromaDB's ONNX embeddings don't actually use. If you see `ModuleNotFoundError: No module named 'torchvision'` in the Streamlit console, **ignore it** — it's a harmless warning from Streamlit's file watcher scanning the `transformers` library. The `.streamlit/config.toml` file in this project suppresses it; if you still see it, your Streamlit version may need `fileWatcherType = "none"` (already set).
- ❌ A GPU — all embeddings run on CPU via ONNX Runtime.
- ❌ A database server — ChromaDB runs embedded (SQLite + local files).
- ❌ Docker — everything runs natively in your venv.
- ❌ Any paid service — the entire stack is free-tier / open-source.

---

## 3. Running the App

You have **three ways** to run the agent. Pick whichever you prefer.

### Option A: Streamlit UI (recommended, primary interface)

```bash
python matching_agent.py streamlit
```

Then open http://localhost:8501. You'll see:

- **Left sidebar**: JD upload, requirements panel, screening progress, quick actions
- **Main area**: Chat tab + Reports tab
- **Chat input** at the bottom

### Option B: CLI (fallback, no browser needed)

```bash
# With a JD file
python matching_agent.py cli --jd tests/fixtures/sample_jd.txt

# With inline JD text
python matching_agent.py cli --jd-text "Senior React Developer with 5+ years experience"

# Interactive (it will prompt you to paste a JD)
python matching_agent.py cli
```

Inside the CLI, type any message and press Enter. Type `done` to exit. Type `help` to see special commands.

### Option C: As a Python library

```python
from matching_agent import create_agent

graph = create_agent()
result = graph.invoke({"raw_jd": "Senior React Developer with TypeScript and AWS", "messages": []})

for candidate in result["current_shortlist"]:
    print(f"{candidate['name']}: {candidate['score']:.2f} ({candidate['hire_recommendation']})")
```

---

## 4. Feature Walkthrough — Testing Every Feature

This is the section you asked for. **Follow these steps in order** to exercise every feature of the application.

### Prerequisites for this walkthrough

1. The app is running (`python matching_agent.py streamlit`)
2. You've run `python scripts/ingest_resumes.py` (4 sample resumes indexed)
3. Browser is open at http://localhost:8501

### Step 1: Load a Job Description

**What to do:**
1. In the left sidebar, find the **"1. Job Description"** section.
2. Either:
   - Click **"Browse files"** and select `tests/fixtures/sample_jd.txt`, OR
   - Paste the JD text into the text area below the uploader
3. Click the blue **▶ Run Pipeline** button.

**What you should see:**
- The chat area shows: `Pipeline complete. Screening rounds: R1: 4, R2: 1, R3: 1. Top 1 candidates:`
- A ranked table appears with candidate names, composite scores, must-have scores, nice-to-have scores, and hire recommendations.
- The sidebar's **Requirements** panel populates with must-have skills (React, TypeScript, CSS, etc.) and nice-to-have skills (Next.js, AWS, CI/CD, etc.).
- The sidebar's **Screening Progress** panel shows 3 completed rounds with candidate counts.

**If you see `⚠️ Error: No shortlisted candidates to generate reports for`:**
This means requirement extraction returned empty. Two causes:
1. You haven't run `python scripts/ingest_resumes.py` yet → run it.
2. You have an old version of the code → re-download the zip (the keyword-based fallback now handles this case).

### Step 2: View a Full Match Report

**What to do:**
1. Click the **📄 Reports** tab at the top of the main area.
2. Select a candidate from the dropdown (e.g., "Alice Johnson").

**What you should see:**
A detailed markdown report with these sections:
- `# Candidate Match Report: <Name>`
- `## Summary` — narrative explanation of the score
- `## Scores` — table with must-have / nice-to-have / composite scores
- `## Must-Have Criteria Breakdown` — per-skill evidence table
- `## Nice-to-Have Criteria Breakdown` — per-skill evidence table
- `## Strengths` — bullet list of matched strengths
- `## Gaps` — bullet list of missing/weak areas
- `## Evidence from Resume` — direct excerpts from the resume
- `## Improvement Suggestions` — actionable suggestions
- `## Hire Recommendation: STRONG HIRE / BORDERLINE / NO HIRE`

**CLI equivalent:**
```
> show report for Alice
```

### Step 3: Compare Candidates Side-by-Side

**What to do:**
1. Switch back to the **💬 Chat** tab.
2. Type into the chat input at the bottom:
   ```
   Compare the top 3 candidates side by side
   ```
3. Press Enter.

**What you should see:**
- A markdown table with one column per candidate and rows for Overall Score, Must-Have, Nice-to-Have, Recommendation.
- A narrative summary explaining who leads and why.

**Try also:**
```
Compare Alice and Bob
What's the difference between Alice and Bob?
```

### Step 4: Ask "Why Did X Rank Higher Than Y?"

**What to do:**
Type into the chat:
```
Why did Alice rank higher than Bob?
```

**What you should see:**
- A heading: `### Why Alice ranked higher than Bob`
- A narrative explanation referencing specific evidence:
  - "Alice has 5 years of React experience vs Bob's 3 years"
  - "Alice has explicit TypeScript experience; Bob does not"
  - "Alice led a React migration; Bob built React components"

**Try also:**
```
Explain the match score for candidate #1
What are the red flags for the top candidate?
```

### Step 5: Refine Requirements Mid-Conversation

**What to do:**
Type into the chat:
```
Drop the AWS requirement and re-rank
```

**What you should see:**
- `### Ranking Updated`
- A delta summary explaining what changed:
  - Which requirements were added/removed/modified
  - Which candidates moved up or down
  - Which candidates were eliminated or newly added
- The sidebar's Requirements panel updates to reflect the new requirements.
- The `requirements_version` counter increments (visible in the sidebar).

**Try also:**
```
Add TypeScript as a must-have skill
Make 5 years experience the minimum
Only show me candidates with a master's degree
```

### Step 6: Generate Interview Questions

**What to do:**
Type into the chat:
```
Generate interview questions for Alice
```

**What you should see:**
- `### Interview Questions for Alice Johnson`
- 5 questions, each with:
  - A category tag: `[TECHNICAL / medium]`, `[BEHAVIORAL / easy]`, etc.
  - The question text
  - The targeted gap (e.g., "Targets gap: AWS experience")
  - Suggested follow-up questions

**Try also:**
```
Create a technical assessment for the top candidate
What should I ask candidate #2 about their React experience?
```

### Step 7: Use the Quick-Action Suggestion Chips

**What to do:**
1. In the sidebar, find **"4. Quick Actions"**.
2. Click **"Show Suggestions"**.
3. Click any suggestion chip (e.g., "Compare top 3", "Why top 2?", "Drop AWS").

**What you should see:**
The chip's message is sent to the chat automatically, and the agent responds as if you typed it.

### Step 8: Export Reports to Disk

**What to do:**
1. In the sidebar, click **"Export Reports"**.

**What you should see:**
- A success message: `Wrote N report(s) to data/reports/`
- Markdown files appear in `data/reports/` (one per shortlisted candidate, named after the candidate).

**CLI equivalent:**
```
> export
```
Or just type `done` to exit — the CLI auto-exports on exit.

### Step 9: Reset the Session

**What to do:**
1. In the sidebar, click **"Reset"**.

**What you should see:**
- The chat history clears.
- The state resets to empty.
- You can load a new JD and start over.

### Step 10: Multi-Round Screening Walkthrough

**What to do:**
After running the pipeline (Step 1), look at the sidebar's **"3. Screening Progress"** section.

**What you should see:**
Three rounds, each with a checkmark (✓) and stats:

| Round | Type | What It Does |
|-------|------|--------------|
| 1 — Initial | `initial` | Broad RAG retrieval (top 100) + must-have keyword filter → top 10 |
| 2 — Deep | `deep_analysis` | Full resume review, skill-depth verification, red-flag detection → top 5–7 |
| 3 — Final | `final` | Hire / no-hire / borderline recommendation with supporting evidence |

Each round shows how many candidates were evaluated and how many advanced.

---

## 5. The Sample JD and Resumes

### Sample JD

Located at `tests/fixtures/sample_jd.txt`. It's a Senior React Frontend Developer role requiring:
- **Must-have**: React, TypeScript, state management (Redux/Zustand), responsive/accessible web apps, modern CSS, REST APIs and GraphQL, Bachelor's degree, 5+ years experience
- **Nice-to-have**: Next.js, AWS/GCP, CI/CD, performance optimization, testing frameworks, full-stack experience

### Sample Resumes

Located at `tests/fixtures/sample_resumes/` and copied to `data/resumes/` during setup:

| File | Candidate | Profile |
|------|-----------|---------|
| `Alice_Johnson_React_Developer.pdf` | Alice Johnson | Senior React Developer, 5 years, TypeScript, Tailwind — **strong match** |
| `Bob_Smith_Python_Backend.pdf` | Bob Smith | Python backend developer, some React — **partial match** |
| `Carol_Williams_Data_Scientist.pdf` | Carol Williams | Data scientist, minimal React — **weak match** |
| `David_Lee_DevOps_Engineer.pdf` | David Lee | DevOps engineer, AWS/Docker/K8s — **weak match for React role** |

When you run the pipeline against the sample JD, you should see Alice ranked #1 with a high score, and the others ranked lower or filtered out.

---

## 6. Adding Your Own Resumes

### To add real resumes

1. Drop PDF files into `data/resumes/`:
   ```bash
   cp ~/Downloads/resume_jane_doe.pdf data/resumes/
   cp ~/Downloads/resume_john_smith.pdf data/resumes/
   ```
2. Re-index:
   ```bash
   python scripts/ingest_resumes.py
   ```
   You should see output like:
   ```
   Ingesting 6 PDFs from data/resumes
   Persisting to data/chroma_db
   --- Ingestion Complete ---
     PDFs processed  : 6
     Total chunks    : 18
     Collection size : 18 chunks
   ```
3. Restart the Streamlit app (or just click Run Pipeline again).

### To generate more sample resumes

```bash
python scripts/generate_sample_resumes.py
```

This script generates additional synthetic resume PDFs for testing.

### To use a different JD

Either:
- Upload a different `.txt` or `.pdf` file via the sidebar uploader, OR
- Paste JD text into the text area, OR
- For the CLI: `python matching_agent.py cli --jd path/to/your/jd.txt`

---

## 7. Troubleshooting

### `ModuleNotFoundError: No module named 'ui'`

**Cause:** You're running an old version of the code, or Streamlit's sys.path isn't picking up the project root.

**Fix:**
1. Re-download the zip (the latest version has a sys.path bootstrap at the top of `ui/streamlit_app.py`).
2. Make sure you're launching from the project root: `cd agentic-profile-matching && python matching_agent.py streamlit`.

### `ModuleNotFoundError: No module named 'torchvision'` (lots of these in the console)

**Cause:** Streamlit's file watcher scans the `transformers` library (pulled in by ChromaDB), and many `transformers` submodules try to import `torchvision` (which isn't installed and isn't needed).

**Fix:** This is **harmless noise** — ignore it. The `.streamlit/config.toml` file in this project sets `fileWatcherType = "none"` to suppress it. If you still see it:
- Make sure `.streamlit/config.toml` exists in the project root.
- Or run Streamlit with: `streamlit run ui/streamlit_app.py --server.fileWatcherType none`

You do **NOT** need to install `torchvision`. It's a 2+ GB download and completely unused by this project.

### `⚠️ Error: No shortlisted candidates to generate reports for`

**Cause:** The screening pipeline filtered out all candidates. This happens when:
1. The resume corpus is empty → run `python scripts/ingest_resumes.py`.
2. The JD has no recognizable skills → use a more detailed JD.
3. (Fixed in latest version) The LLM was unavailable and there was no keyword fallback → re-download the zip.

**Fix:**
```bash
# Verify resumes are indexed
python -c "from src.rag.store import get_vector_store; from src.rag.retriever import ResumeRetriever; r = ResumeRetriever(get_vector_store('data/chroma_db')); print(f'{len(r.search(\"developer\", top_k=100))} candidates indexed')"
# If 0, re-index:
python scripts/ingest_resumes.py
```

### `RuntimeError: No LLM available` or `RESOURCE_EXHAUSTED` (429)

**Cause:** Gemini's free tier quota is exhausted (you'll see `429 RESOURCE_EXHAUSTED` in the error message). The free tier limits are: 15 requests/minute, 1,500 requests/day.

**Fix (pick one — Groq is recommended):**
1. **Add a Groq API key** (free, highest quota, fastest — RECOMMENDED):
   - Go to https://console.groq.com/keys
   - Create a key, copy it
   - Edit `.env`: `GROQ_API_KEY=your_key_here`
   - Optionally also set: `GROQ_MODEL=llama-3.3-70b-versatile`
   - Click "🔄 Re-check LLM" in the Streamlit sidebar
   - Groq's free tier: 30 RPM, 14,400 req/day — **10x Gemini's daily quota**
2. **Wait for Gemini's quota to reset** (typically resets in ~1 minute for RPM, or daily for the day cap):
   - The error message says "Please retry in Xs" — wait that long and click "🔄 Re-check LLM"
3. **Add a Gemini API key** (free, if you don't have one):
   - Go to https://aistudio.google.com/apikey
   - Edit `.env`: `GEMINI_API_KEY=your_key_here`
4. **Install Ollama** (free, fully offline, ~5 GB download):
   - Install from https://ollama.com
   - Run: `ollama pull gemma2:9b`
   - Restart the app
5. **Use without an LLM** (works fine for basic features):
   - The keyword-based fallback handles requirement extraction, scoring, comparison, and explanation.
   - Quality is lower but the app is fully functional.

**Pro tip**: Set **both** `GEMINI_API_KEY` and `GROQ_API_KEY` in `.env`. The agent tries Gemini first, and if Gemini's quota is exhausted, it automatically falls back to Groq. This gives you 15 + 30 = 45 RPM and 1,500 + 14,400 = 15,900 requests/day — enough for any demo or development session.

### `error: externally-managed-environment` (Linux / Homebrew Python)

**Cause:** Newer Python versions refuse to install packages globally.

**Fix:** Always use a virtual environment:
```bash
python3 -m venv venv
source venv/bin activate
pip install -e .
```

### Streamlit shows a blank page or "Please wait..."

**Cause:** The app is still starting up, or a previous run is still holding the port.

**Fix:**
1. Wait 10 seconds and refresh.
2. If still blank, kill any old Streamlit process: `pkill -f streamlit`
3. Try a different port: `streamlit run ui/streamlit_app.py --server.port 8502`

### Port 8501 is already in use

**Fix:** Use a different port:
```bash
streamlit run ui/streamlit_app.py --server.port 8502
```
Then open http://localhost:8502.

### The CLI crashes on startup

**Fix:** Check your Python version:
```bash
python --version  # must be 3.11+
```
And verify imports work:
```bash
python -c "from matching_agent import create_agent; print('OK')"
```

### Tests fail with `ModuleNotFoundError`

**Cause:** Tests must be run from the project root with the venv activated.

**Fix:**
```bash
cd agentic-profile-matching
source venv/bin/activate
pytest tests/ -q
```

---

## 8. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                         │
│  ┌───────────────────────┐  ┌────────────────────────────────────┐ │
│  │ Streamlit Chat UI     │  │ CLI REPL                           │ │
│  │ ui/streamlit_app.py   │  │ ui/cli_app.py                      │ │
│  └───────────┬───────────┘  └─────────────────┬──────────────────┘ │
│              │              ui/components.py (shared renderers)      │
└──────────────┼──────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LangGraph Agent (matching_agent.py)              │
│                                                                     │
│  START → parse_jd → extract_requirements → search_resumes          │
│        → rank_candidates → generate_report → human_feedback_loop   │
│        → (refine | compare | explain | questions | report) → END   │
│                                                                     │
│  src/agent/{state, models, nodes, edges, graph}.py                 │
└──────┬──────────────┬──────────────┬──────────────┬─────────────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────────────┐
│ RAG Layer  │ │ Tools      │ │ Scoring    │ │ Screening Pipeline │
│            │ │            │ │            │ │                    │
│ src/rag/   │ │ src/tools/ │ │ src/scor/  │ │ src/screening/     │
│ store.py   │ │ extract_   │ │ scorer.py  │ │ round1_initial.py  │
│ indexer.py │ │ rag_search │ │ ranker.py  │ │ round2_deep.py     │
│ retriever  │ │ compare    │ │ red_flags  │ │ round3_final.py    │
│            │ │ questions  │ │            │ │ pipeline.py        │
│ ChromaDB + │ │ file_tools │ │            │ │                    │
│ ONNX embed │ │            │ │            │ │                    │
└────────────┘ └────────────┘ └────────────┘ └────────────────────┘
       │              │
       ▼              ▼
┌────────────┐ ┌────────────────────┐
│ data/      │ │ LLM Provider       │
│ chroma_db/ │ │ src/llm/client.py  │
│ resumes/   │ │                    │
│ reports/   │ │ Groq (1°)          │
└────────────┘ │ Gemini (2°)        │
               │ Ollama (3°)        │
               │ Keyword fallback   │
               └────────────────────┘
```

**Key design decisions:**
- **LangGraph StateGraph** with `AgentState` TypedDict (14 fields) threaded through every node.
- **Composite score** = `0.7 × must_have + 0.3 × nice_to_have`.
- **3-round screening funnel**: broad RAG → deep analysis + red flags → final hire recommendation.
- **Free-tier only**: Gemini Free, Ollama, ChromaDB local, ONNX embeddings — zero cost.
- **Graceful degradation**: every LLM-dependent feature has a deterministic fallback.

See `docs/architecture.md` for the full spec and `docs/state_machine_diagram.png` for the visual diagram.

---

## 9. Running Tests

```bash
# Run all tests (excluding LLM-dependent integration tests)
pytest tests/ -q -m "not integration"

# Run all tests including integration tests (requires LLM)
pytest tests/ -v

# Run with coverage report
pytest tests/ -q -m "not integration" --cov=src --cov=ui --cov-report=term-missing

# Run a specific test file
pytest tests/test_agent_flows.py -v

# Run a specific scenario
pytest tests/test_agent_flows.py::TestScenario1HappyPath -v
```

**Expected output:**
```
388 passed, 9 deselected, 3 warnings in 22.18s
```

The 9 deselected tests are `@pytest.mark.integration` tests that require a live LLM (Gemini or Ollama). They automatically run once you configure an LLM.

---

## 10. Project Structure

```
agentic-profile-matching/
├── matching_agent.py          # ← Entry point: python matching_agent.py streamlit
├── pyproject.toml             # Dependencies (pip install -e .)
├── .env.example               # Copy to .env, add GEMINI_API_KEY
├── .streamlit/
│   └── config.toml            # Suppresses transformers/torchvision warnings
│
├── src/
│   ├── agent/                 # LangGraph state, nodes, edges, graph
│   │   ├── state.py           # AgentState TypedDict (14 fields)
│   │   ├── models.py          # Pydantic schemas for LLM structured output
│   │   ├── nodes.py           # 5 linear + 6 interactive node functions
│   │   ├── edges.py           # Conditional routing logic
│   │   └── graph.py           # StateGraph construction + compile()
│   ├── tools/                 # 5 agent tools (extract, rag_search, compare, questions, file)
│   ├── scoring/               # scorer, ranker, red_flags
│   ├── screening/             # 3-round pipeline (round1, round2, round3, pipeline)
│   ├── rag/                   # ChromaDB store, indexer, retriever
│   ├── prompts/               # LLM prompt templates (extraction, scoring, comparison, etc.)
│   ├── reports/               # Markdown match report generation
│   └── llm/                   # LLM client (Gemini → Ollama → fallback)
│
├── ui/
│   ├── streamlit_app.py       # Streamlit chat interface (primary)
│   ├── cli_app.py             # CLI REPL (fallback)
│   └── components.py           # Shared rendering helpers
│
├── tests/
│   ├── test_setup.py          # Phase 0: environment smoke tests
│   ├── test_rag.py            # Phase 1: RAG indexing/retrieval
│   ├── test_state.py          # Phase 2: state + Pydantic models
│   ├── test_tools.py          # Phase 3: tool unit tests
│   ├── test_nodes.py          # Phase 4: node unit tests
│   ├── test_graph_linear.py   # Phase 4: linear graph integration
│   ├── test_intent_classif.*  # Phase 5: intent classification
│   ├── test_conversation_*.py # Phase 5: interactive loop tests
│   ├── test_screening.py      # Phase 6: 3-round screening pipeline
│   ├── test_ui.py             # Phase 7: UI component tests
│   ├── test_agent_flows.py    # Phase 8: 7 end-to-end scenarios
│   └── fixtures/
│       ├── sample_jd.txt
│       ├── sample_resumes/    # 4 PDFs
│       └── expected_outputs/  # Schema reference files
│
├── docs/
│   ├── context.md             # Problem statement & requirements
│   ├── architecture.md        # Full architecture spec (16 sections)
│   ├── implementation-plan.md # 8-phase plan
│   ├── state_machine_diagram.png  # Visual graph diagram
│   ├── state_machine_diagram.svg  # Vector version
│   ├── state_machine_diagram.mmd  # Mermaid source
│   └── demo_script.md         # 5-6 min demo walkthrough script
│
├── scripts/
│   ├── ingest_resumes.py      # Index PDFs into ChromaDB
│   └── generate_sample_resumes.py
│
├── data/
│   ├── resumes/               # ← Drop your PDFs here
│   ├── chroma_db/             # Vector store (auto-created)
│   └── reports/               # Exported match reports (auto-created)
│
└── worklog.md                 # Multi-agent work log (8 phases)
```

---

## Quick Command Reference

```bash
# Setup (one-time)
pip install -e .
python scripts/ingest_resumes.py
cp .env.example .env  # then edit to add GEMINI_API_KEY (optional)

# Run
python matching_agent.py streamlit              # UI (primary)
python matching_agent.py cli --jd <path>        # CLI with JD file
python matching_agent.py cli --jd-text "<jd>"   # CLI with inline JD
python matching_agent.py info                   # Show env info

# Test
pytest tests/ -q -m "not integration"           # Fast tests (no LLM)
pytest tests/ -v                                # All tests (needs LLM)
pytest tests/ --cov=src --cov=ui                # With coverage

# Maintain
python scripts/ingest_resumes.py                # Re-index resumes
python scripts/generate_sample_resumes.py       # Generate test resumes
```

---

## Need Help?

- **Architecture**: read `docs/architecture.md`
- **Implementation plan**: read `docs/implementation-plan.md`
- **Demo walkthrough**: read `docs/demo_script.md`
- **Work log**: read `worklog.md` (8 phases documented)
- **State machine diagram**: open `docs/state_machine_diagram.png`

**Report bugs by** running `python matching_agent.py info` and sharing the output plus the full error traceback.
