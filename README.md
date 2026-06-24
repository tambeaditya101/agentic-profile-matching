# Agentic Profile Matching

A LangGraph-based agent that reads a job description, retrieves matching candidates from a resume corpus via RAG, runs them through a 3-round screening pipeline, and explains every ranking decision with evidence. Built to get hands-on with agentic coding patterns, RAG, and LangGraph's stateful multi-turn agents.

---

## Why I Built This

I wanted to move beyond "call an LLM and print the response" and actually build something where the LLM is one piece of a larger stateful system. The progression I had in mind:

1. **Agentic coding with LLMs** — not just chat, but an agent that decides what to do next based on user intent
2. **RAG in practice** — not the textbook "embed and query" demo, but RAG as the retrieval layer inside an agent that actually uses the results
3. **LangChain → LangGraph** — LangChain is great for chains, but once you need conditional routing, state accumulation across turns, and a human-in-the-loop, LangGraph's `StateGraph` is a cleaner fit

A candidate-job matching agent turned out to be a good testbed for all three: the JD parsing needs LLM reasoning, the candidate retrieval is a real RAG workload, and the interactive refinement ("drop the AWS requirement and re-rank") forces you to handle multi-turn state properly.

The whole thing runs on free-tier infrastructure — Groq, Gemini, Ollama, local ChromaDB. No paid APIs, no hosted vector databases. That constraint turned out to be the most interesting part of the build (see [Design Decisions](#design-decisions)).

---

## What It Does

- **Parses a JD** into structured must-have / nice-to-have requirements with weights and evidence
- **Retrieves candidates** from a local resume corpus using ChromaDB + ONNX embeddings (no API key needed for embeddings)
- **Runs a 3-round screening funnel**: broad RAG retrieval → deep analysis with red-flag detection → final hire/no-hire recommendation
- **Generates per-candidate match reports** with scores, strengths, gaps, resume evidence excerpts, and improvement suggestions
- **Supports multi-turn interaction**: compare candidates, explain rankings, refine requirements mid-conversation, generate interview questions
- **Explains itself**: every score traces back to specific resume text — no opaque rankings

---

## Quick Start

```bash
# 1. Clone and enter
git clone <your-repo-url>
cd agentic-profile-matching

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -e .

# 4. Index the sample resumes into the vector store
python scripts/ingest_resumes.py

# 5. Add your LLM API keys (optional but recommended)
cp .env.example .env
# Edit .env — add at least one of:
#   GROQ_API_KEY=...   (get at https://console.groq.com/keys — recommended)
#   GEMINI_API_KEY=... (get at https://aistudio.google.com/apikey)

# 6. Launch the Streamlit UI
python matching_agent.py streamlit
```

Open **http://localhost:8501**. That's it.

> The app also works with **zero API keys** — it falls back to a keyword-based extraction and scoring mode. Quality is lower, but the pipeline runs end-to-end. Useful for seeing the architecture without signing up for anything.

---

## What You Need

### Required

| Tool                     | Version                          | Why                             |
| ------------------------ | -------------------------------- | ------------------------------- |
| **Python**               | 3.11+ (3.12 or 3.13 recommended) | Runtime                         |
| **pip**                  | any recent                       | Package installer               |
| **Project dependencies** | —                                | `pip install -e .` handles this |

### Optional — LLM Providers

The agent has a 4-tier fallback chain. Set as many as you want — if one fails or hits a quota limit, the next takes over automatically.

| Provider               | Free Tier              | Why                                                                  | Get a Key                          |
| ---------------------- | ---------------------- | -------------------------------------------------------------------- | ---------------------------------- |
| **Groq** (recommended) | 30 RPM, 14,400 req/day | Fastest inference (LPU-based), highest free quota, Llama 3.3 70B     | https://console.groq.com/keys      |
| **Gemini**             | 15 RPM, 1,500 req/day  | High quality, Google's flagship model                                | https://aistudio.google.com/apikey |
| **Ollama** (offline)   | Unlimited (local)      | Fully offline, no API key, runs `gemma2:9b` locally (~5 GB download) | https://ollama.com                 |
| **Keyword fallback**   | Always available       | No LLM, deterministic keyword-based extraction (lower quality)       | (automatic)                        |

### You do NOT need

- ❌ `torch` or `torchvision` — ChromaDB's ONNX embeddings don't use PyTorch. If you see `ModuleNotFoundError: No module named 'torchvision'` in the Streamlit console, ignore it — it's Streamlit's file watcher scanning the `transformers` library. The `.streamlit/config.toml` file suppresses it.
- ❌ A GPU — all embeddings run on CPU via ONNX Runtime.
- ❌ A database server — ChromaDB runs embedded (SQLite + local files).
- ❌ Docker — everything runs natively in your venv.
- ❌ Any paid service.

---

## Running the App

Three ways to run it, depending on what you prefer:

### Streamlit UI (primary)

```bash
python matching_agent.py streamlit
```

Opens a chat interface at http://localhost:8501 with a sidebar showing extracted requirements, screening round progress, and a live LLM status badge.

### CLI (fallback)

```bash
# With a JD file
python matching_agent.py cli --jd tests/fixtures/sample_jd.txt

# With inline JD text
python matching_agent.py cli --jd-text "Senior React Developer with 5+ years experience"

# Interactive (prompts you to paste a JD)
python matching_agent.py cli
```

### As a library

```python
from matching_agent import create_agent

graph = create_agent()
result = graph.invoke({"raw_jd": "Senior React Developer with TypeScript and AWS", "messages": []})

for candidate in result["current_shortlist"]:
    print(f"{candidate['name']}: {candidate['score']:.2f} ({candidate['hire_recommendation']})")
```

---

## How It Works

The agent is a LangGraph `StateGraph` with 10 nodes. The state is a single `AgentState` TypedDict (14 fields) threaded through every node — no global mutable state, no hidden context.

```
START → parse_jd → extract_requirements → search_resumes
      → rank_candidates → generate_report → human_feedback_loop
                                                ├── refine requirements  → re-rank
                                                ├── compare candidates   → comparison table
                                                ├── explain ranking      → narrative explanation
                                                ├── generate questions   → interview questions
                                                └── show report          → full match report
                                                → END
```

### The 3-Round Screening Funnel

Instead of scoring all candidates in one pass, the screening happens in three rounds that progressively narrow and deepen:

1. **Round 1 — Initial** (`round1_initial.py`): Broad RAG retrieval (top 100), quick must-have keyword filter, lightweight scoring. Narrows to top 10.
2. **Round 2 — Deep** (`round2_deep.py`): For each of the top 10, retrieves the full resume, verifies skill depth (not just keyword mentions), extracts experience timeline, checks for red flags (employment gaps >3 months, job-hopping >3 jobs in 2 years, inconsistent dates). Re-ranks and narrows to 5–7.
3. **Round 3 — Final** (`round3_final.py`): Compiles all evidence from Rounds 1 and 2, generates a hire / no-hire / borderline recommendation with supporting evidence, and produces the full match report.

Each round records a `ScreeningRound` entry in the state so the UI can show the funnel progression.

### Composite Score

```
composite = 0.7 × must_have_score + 0.3 × nice_to_have_score
```

Must-have criteria are the primary filter; nice-to-have breaks ties. Both subscores are in `[0.0, 1.0]` and clamped after computation.

### Intent Classification

When you type something in the chat, the `human_feedback_loop` node classifies your message into one of: `refine`, `compare`, `questions`, `explain`, `report`, `done`, `new_search`. Classification uses an LLM with a keyword-based fallback (so it works even without an LLM). The classified intent routes to the appropriate interactive node via a conditional edge.

---

## Design Decisions

These are the parts of the build where I actually had to think through trade-offs rather than just follow a tutorial.

### The 4-tier LLM fallback chain

**The problem**: I started with Gemini as the only LLM. Within an afternoon of testing, I hit `429 RESOURCE_EXHAUSTED` — the free tier's 1,500 requests/day limit is tighter than it sounds when you're iterating on prompts.

**What I tried**: Added Ollama as a fallback. Worked, but the 5 GB model download is a barrier for anyone cloning the repo, and local inference is slow on a laptop without a GPU.

**What I ended up with**: A 4-tier chain — **Groq → Gemini → Ollama → keyword fallback** — where each provider is actually pinged with a tiny "Reply with: OK" prompt before being used. The first one that responds wins.

- **Groq** is primary because its free tier (14,400 req/day) is 10x Gemini's, and LPU-based inference is noticeably faster (~0.3s vs ~2-3s per call)
- **Gemini** is secondary — same quality, lower quota
- **Ollama** is the offline fallback for air-gapped environments
- **Keyword fallback** is the last resort: a deterministic regex-based skill extractor and keyword-matching scorer. Lower quality, but the pipeline never dead-ends

The status badge in the UI reflects the _actual_ working provider, not just the configured one. If Groq's quota exhausts mid-session, the badge turns yellow and runtime calls silently fall back to Gemini — you see it in the call history.

### Keyword fallback for requirement extraction

The first version of `extract_requirements` was LLM-only. When the LLM was unavailable, it returned empty requirements, which caused the screening pipeline to bail out with "No must-have requirements defined" — the whole app was dead.

I added a deterministic keyword-based extractor that uses a curated skill dictionary (~90 entries covering frontend, backend, databases, cloud/DevOps, data/ML, testing, mobile, soft skills, certifications, languages). It splits the JD into must-have / nice-to-have sections by detecting common headers ("Required Qualifications", "Preferred Qualifications"), then scans each section for known skills via regex.

It's conservative — I'd rather miss a skill than invent one. The LLM handles the long tail of unusual skills when it's available; the keyword fallback just keeps the pipeline moving when it's not.

### Runtime LLM call tracking

The original `get_llm()` just checked if the API key was non-empty and returned the client. That meant if the key was invalid or quota was exhausted, every tool call silently failed and fell back to keywords — the status badge showed green the whole time.

Now every LLM invoke goes through `record_llm_call(provider, success, error, tool, duration_ms)`. The status badge combines the startup ping result with runtime stats — if >50% of recent calls fail, it downgrades from green to yellow and shows the actual error. This caught a real bug during development where Gemini's quota was exhausted but I didn't realize it for an hour.

### Why LangGraph over plain LangChain

LangChain chains are linear — great for "parse this, then summarize, then format" pipelines. But once you need:

- Conditional routing based on user intent
- State accumulation across turns (the shortlist from turn 1 needs to be available in turn 3)
- A human-in-the-loop that can interrupt and redirect

...LangChain's `LCEL` chains get awkward fast. LangGraph's `StateGraph` with a `TypedDict` state and conditional edges is a much cleaner mental model. The state is explicit, the routing is explicit, and the graph compiles to a visualization you can actually reason about.

### ChromaDB's built-in ONNX embeddings

I initially planned to use `sentence-transformers` for embeddings, but that pulls in PyTorch (2.5 GB) and torch dependencies. ChromaDB ships with an ONNX-based embedding function (`all-MiniLM-L6-v2`) that runs on CPU via ONNX Runtime — same model, no PyTorch, no extra download. For a project where the embedding workload is ~100 resumes, this is the right trade-off.

---

## Trying It Out

Once the app is running, here's what to try:

### 1. Load a JD and run the pipeline

In the Streamlit sidebar, upload `tests/fixtures/sample_jd.txt` (or paste JD text), then click **Run Pipeline**. You should see a ranked shortlist appear in the chat, with the sidebar populating requirements and screening round progress.

### 2. View a match report

Click the **Reports** tab and select a candidate. The report includes a summary, scores table, per-skill evidence breakdown, strengths, gaps, resume excerpts, improvement suggestions, and a hire recommendation.

### 3. Compare candidates

In the chat, type:

```
Compare the top 3 candidates side by side
```

### 4. Ask for an explanation

```
Why did Alice rank higher than Bob?
```

The agent produces a narrative explanation citing specific evidence (years of experience, particular skills) rather than just restating scores.

### 5. Refine requirements mid-conversation

```
Drop the AWS requirement and re-rank
```

The agent re-ranks, increments the requirements version, and produces a delta summary explaining who moved up, who moved down, and why.

### 6. Generate interview questions

```
Generate interview questions for Alice
```

You get 5 questions, each tagged with a category (technical, behavioral, situational), difficulty, the targeted gap, and follow-up questions.

### 7. Check the LLM status badge

In the sidebar, the badge at the top shows which provider is actually working (🟢 Groq, 🟡 Ollama, 🔴 keyword fallback). Click "LLM details" to see runtime call history — each LLM invoke shows ✅/❌ with duration in ms.

---

## Sample Data

The project includes 4 sample resume PDFs in `tests/fixtures/sample_resumes/` (also copied to `data/resumes/` during setup):

| File                                | Candidate      | Profile                                                              |
| ----------------------------------- | -------------- | -------------------------------------------------------------------- |
| `Alice_Johnson_React_Developer.pdf` | Alice Johnson  | Senior React Developer, 5 years, TypeScript, Tailwind — strong match |
| `Bob_Smith_Python_Backend.pdf`      | Bob Smith      | Python backend developer, some React — partial match                 |
| `Carol_Williams_Data_Scientist.pdf` | Carol Williams | Data scientist, minimal React — weak match                           |
| `David_Lee_DevOps_Engineer.pdf`     | David Lee      | DevOps engineer, AWS/Docker/K8s — weak match for a React role        |

The sample JD (`tests/fixtures/sample_jd.txt`) is for a Senior React Frontend Developer. When you run the pipeline, Alice should rank #1 with a high score; the others rank lower or get filtered out.

---

## Adding Your Own Resumes

1. Drop PDF files into `data/resumes/`:
   ```bash
   cp ~/Downloads/resume_jane_doe.pdf data/resumes/
   ```
2. Re-index:
   ```bash
   python scripts/ingest_resumes.py
   ```
3. Restart the app (or just click Run Pipeline again).

To use a different JD, upload it via the sidebar uploader or paste the text into the text area.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'ui'`

You're running an old version, or Streamlit's `sys.path` isn't picking up the project root. Make sure you're launching from the project root:

```bash
cd agentic-profile-matching
python matching_agent.py streamlit
```

### `ModuleNotFoundError: No module named 'torchvision'` (lots of these)

Harmless noise from Streamlit's file watcher scanning the `transformers` library. The `.streamlit/config.toml` file sets `fileWatcherType = "none"` to suppress it. You do NOT need to install torchvision — it's a 2+ GB download and completely unused by this project.

### `429 RESOURCE_EXHAUSTED` (Gemini quota)

Gemini's free tier limit. Options:

1. Add a Groq key (recommended — 10x higher quota): https://console.groq.com/keys
2. Wait for the quota to reset (the error says "Please retry in Xs")
3. Install Ollama for offline inference: https://ollama.com
4. Use keyword fallback mode (works with no LLM)

The agent automatically falls back between providers, so setting both Groq and Gemini keys gives you 15,900 requests/day combined.

### `⚠️ Error: No shortlisted candidates to generate reports for`

The screening pipeline filtered out all candidates. Usually means:

1. The resume corpus is empty → run `python scripts/ingest_resumes.py`
2. The JD has no recognizable skills → use a more detailed JD
3. (Fixed in current version) The LLM was unavailable and there was no keyword fallback → re-download

### Port 8501 already in use

```bash
streamlit run ui/streamlit_app.py --server.port 8502
```

### Tests fail with `ModuleNotFoundError`

Run from the project root with the venv activated:

```bash
cd agentic-profile-matching
source venv/bin/activate
pytest tests/ -q
```

---

## Testing

```bash
# Run all tests (excluding LLM-dependent integration tests)
pytest tests/ -q -m "not integration"

# Run with coverage
pytest tests/ -q -m "not integration" --cov=src --cov=ui --cov-report=term-missing

# Run a specific test file
pytest tests/test_agent_flows.py -v
```

The test suite covers:

- **Environment and setup** — dependency imports, package structure
- **RAG** — indexing, retrieval, full resume reassembly
- **State and models** — TypedDict construction, Pydantic validation
- **Tools** — each tool in isolation (extract, rag_search, compare, questions, file)
- **Nodes** — individual graph node functions with mocked dependencies
- **Linear graph** — end-to-end pipeline invocation
- **Intent classification** — 39 parametrized inputs across 7 intent categories
- **Conversation flows** — multi-turn interactive node tests
- **Screening pipeline** — all 3 rounds, red-flag detection, orchestrator
- **UI components** — 41 tests covering all renderers and CLI entry points
- **End-to-end scenarios** — 7 full conversation flows (happy path, refinement, comparison, explanation, questions, edge cases, multi-round)

Run `pytest tests/ -q -m "not integration"` to verify — 388 tests should pass.

---

## Project Structure

```
agentic-profile-matching/
├── matching_agent.py              # Entry point: python matching_agent.py streamlit|cli
├── pyproject.toml                 # Dependencies (pip install -e .)
├── .env.example                   # Template for GROQ_API_KEY, GEMINI_API_KEY
├── .streamlit/
│   └── config.toml                # Suppresses transformers/torchvision watcher warnings
│
├── src/
│   ├── agent/                     # LangGraph state, nodes, edges, graph
│   │   ├── state.py               # AgentState TypedDict (14 fields)
│   │   ├── models.py              # Pydantic schemas for LLM structured output
│   │   ├── nodes.py               # 5 linear + 6 interactive node functions
│   │   ├── edges.py               # Conditional routing (intent → node)
│   │   └── graph.py               # StateGraph construction + compile()
│   ├── tools/                     # 5 agent tools (extract, rag_search, compare, questions, file)
│   ├── scoring/                   # scorer, ranker, red_flags
│   ├── screening/                 # 3-round pipeline (round1, round2, round3, pipeline)
│   ├── rag/                       # ChromaDB store, indexer, retriever
│   ├── prompts/                   # LLM prompt templates
│   ├── reports/                   # Markdown match report generation
│   └── llm/                       # 4-tier LLM client (Groq → Gemini → Ollama → keyword fallback)
│
├── ui/
│   ├── streamlit_app.py           # Streamlit chat interface (primary)
│   ├── cli_app.py                 # CLI REPL (fallback)
│   └── components.py              # Shared rendering helpers
│
├── tests/                         # 388 tests
│   └── fixtures/
│       ├── sample_jd.txt
│       ├── sample_resumes/        # 4 PDFs
│       └── expected_outputs/      # Schema reference files
│
├── docs/
│   ├── context.md                 # Problem statement & requirements
│   ├── architecture.md            # Full architecture spec (16 sections)
│   ├── implementation-plan.md     # 8-phase plan
│   └── state_machine_diagram.png  # Visual graph diagram
│
├── scripts/
│   ├── ingest_resumes.py          # Index PDFs into ChromaDB
│   └── generate_sample_resumes.py
│
└── data/
    ├── resumes/                   # Resume corpus (4 sample PDFs included)
    ├── chroma_db/                 # Persistent vector store (pre-built)
    └── reports/                   # Exported match reports (auto-created)
```

---

## Demo

<!-- Add your demo video link here once recorded. A few things worth showing:
     - The 3-round screening funnel narrowing candidates
     - A match report with evidence excerpts
     - Mid-conversation requirement refinement (the re-ranking + delta summary)
     - The LLM status badge reflecting the actual working provider.
-->

---

## Quick Command Reference

```bash
# Setup (one-time)
pip install -e .
python scripts/ingest_resumes.py
cp .env.example .env  # add GROQ_API_KEY and/or GEMINI_API_KEY

# Run
python matching_agent.py streamlit              # UI (primary)
python matching_agent.py cli --jd <path>        # CLI with JD file
python matching_agent.py cli --jd-text "<jd>"   # CLI with inline JD
python matching_agent.py info                   # Show LLM + RAG status

# Test
pytest tests/ -q -m "not integration"           # Fast tests (no LLM needed)
pytest tests/ -v                                # All tests (needs LLM for integration tests)
pytest tests/ --cov=src --cov=ui                # With coverage

# Maintain
python scripts/ingest_resumes.py                # Re-index resumes
```
