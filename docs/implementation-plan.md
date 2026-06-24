# Agentic Profile Matching — Phase-Wise Implementation Plan

> **Derived from**: `docs/architecture.md` (v1.1)
> **Version**: 1.1
> **Last Updated**: 2026-06-24
>
> **v1.1**: Replaced all paid-service dependencies with free-tier/open-source alternatives (Gemini Free Tier, Hugging Face local embeddings, Ollama fallback).
> **Total Phases**: 8
> **Estimated Duration**: 4–6 weeks (part-time) or 2–3 weeks (full-time)

---

## How to Read This Plan

Each phase follows this structure:

| Section | Meaning |
|---------|---------|
| **Goal** | What this phase achieves (1–2 sentences) |
| **Architecture Reference** | Which section(s) of `architecture.md` this phase implements |
| **Deliverables** | Files to create or modify with their expected content |
| **Tasks** | Ordered, atomic work items |
| **Acceptance Criteria** | How to verify the phase is complete |
| **Dependencies** | Which phases must finish before this one starts |
| **Checkpoint Command** | A runnable command to test the phase's output |

---

## Phase 0 — Project Scaffolding & Environment Setup

**Goal**: Set up the full directory structure, install all dependencies, and create configuration files so every subsequent phase has a clean foundation to build on.

**Architecture Reference**: Section 13 (Directory Structure), Section 12 (Technology Stack)

### Deliverables

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project metadata, all pinned dependencies |
| `requirements.txt` | Flat pip-installable list (generated from pyproject.toml) |
| `.env.example` | Template for API keys (GEMINI_API_KEY, etc.) |
| `.env` | Your actual API keys (gitignored) |
| `.gitignore` | Python + IDE + data ignores |
| `src/__init__.py` | Package root |
| `src/agent/__init__.py` | Agent sub-package |
| `src/tools/__init__.py` | Tools sub-package |
| `src/scoring/__init__.py` | Scoring sub-package |
| `src/screening/__init__.py` | Screening sub-package |
| `src/reports/__init__.py` | Reports sub-package |
| `src/rag/__init__.py` | RAG sub-package |
| `src/prompts/__init__.py` | Prompts sub-package |
| `ui/__init__.py` | UI sub-package |
| `tests/__init__.py` | Tests root |
| `tests/fixtures/` | Test data directory |
| `data/resumes/` | Empty directory for resume corpus |
| `data/chroma_db/` | Will hold the vector store |

### Tasks

1. Create the full directory tree from the architecture doc (Section 13)
2. Write `pyproject.toml` with these core dependencies:
   - `langgraph>=0.2.0`
   - `langchain-core>=0.3.0`
   - `langchain-google-genai>=1.0.0`
   - `langchain-community>=0.3.0`
   - `langchain-chroma>=0.1.0`
   - `chromadb>=0.5.0`
   - `pydantic>=2.0.0`
   - `streamlit>=1.35.0`
   - `click>=8.1.0`
   - `PyMuPDF>=1.24.0`
   - `pytest>=8.0.0`
   - `pytest-asyncio>=0.23.0`
   - `python-dotenv>=1.0.0`
3. Run `pip install -e .` (or `pip install -r requirements.txt`) and verify no errors
4. Copy `.env.example` to `.env`, fill in your free Gemini API key (or leave empty to use Ollama locally)
5. Write a smoke test: `tests/test_setup.py` that imports all packages and prints versions
6. Initialize git, make initial commit

### Acceptance Criteria

- [x] `python -c "import langgraph, langchain_core, chromadb, streamlit, pydantic"` succeeds
- [x] `pytest tests/test_setup.py` passes
- [x] Directory tree matches architecture doc Section 13 exactly
- [x] `.env` exists with a valid `GEMINI_API_KEY` (or Ollama is installed for offline mode)

### Dependencies

None — this is the starting phase.

### Checkpoint Command

```bash
python -c "
import langgraph; print(f'langgraph: {langgraph.__version__}')
import langchain_core; print(f'langchain_core: {langchain_core.__version__}')
import chromadb; print(f'chromadb: {chromadb.__version__}')
import pydantic; print(f'pydantic: {pydantic.__version__}')
print('Phase 0 complete.')
"
```

---

## Phase 1 — RAG Foundation (Resume Ingestion & Retrieval)

**Goal**: Build the vector store pipeline that ingests resume PDFs, generates embeddings, and supports semantic search — the data layer that the agent's `rag_search` tool depends on.

**Architecture Reference**: Section 6.2 (rag_search tool), Section 11 (Data Flow — Vector Store)

### Deliverables

| File | Purpose |
|------|---------|
| `src/rag/store.py` | ChromaDB client initialization, collection management |
| `src/rag/indexer.py` | Resume PDF ingestion, text extraction, chunking, embedding |
| `src/rag/retriever.py` | Query execution, result formatting, metadata filtering |
| `scripts/ingest_resumes.py` | Standalone script to build/rebuild the index |
| `tests/test_rag.py` | Unit tests for indexing and retrieval |
| `tests/fixtures/sample_resumes/` | 3–5 sample resume PDFs for testing |

### Tasks

1. **`store.py`** — Create a `get_vector_store(persist_dir)` function:
   - Initialize a persistent ChromaDB client at `data/chroma_db/`
   - Create (or load) a collection named `"resumes"`
   - Use ChromaDB's built-in ONNX embedding function (all-MiniLM-L6-v2) — ships with ChromaDB, no extra install, no PyTorch needed
   - Expose `collection` for other modules to use directly

2. **`indexer.py`** — Create a `ResumeIndexer` class:
   - `ingest_directory(dir_path)`: Scan `data/resumes/` for PDF files
   - `_extract_text(pdf_path)`: Use PyMuPDF (`fitz.open`) to extract all text from a resume PDF
   - `_chunk_text(text, chunk_size=1000, overlap=200)`: Split resume text into overlapping chunks with metadata (candidate name from filename, page number)
   - `_generate_embedding(text)`: Call the embedding function
   - `upsert(collection, chunks)`: Batch insert chunks with metadata (`candidate_id`, `name`, `chunk_index`, `full_text`)
   - Store the full resume text as a separate document or metadata field so it can be retrieved later for deep analysis

3. **`retriever.py`** — Create a `ResumeRetriever` class:
   - `search(query, top_k=20, filter=None)`: Execute similarity search on the collection
   - `get_full_resume(candidate_id)`: Retrieve all chunks for a candidate and reassemble the full resume text
   - Return standardized results: `list[{"candidate_id", "name", "score", "excerpt"}]`

4. **`ingest_resumes.py`** — Script that:
   - Loads `.env` for configuration
   - Instantiates `ResumeIndexer` and `get_vector_store`
   - Calls `ingest_directory("data/resumes/")`
   - Prints statistics: number of PDFs processed, total chunks, collection size

5. **Test data** — Create or obtain 3–5 sample resume PDFs in `tests/fixtures/sample_resumes/` with varied profiles (React developer, Python backend, data scientist, etc.)

6. **`tests/test_rag.py`**:
   - Test text extraction from a known PDF
   - Test chunking produces expected number of chunks
   - Test search returns results for a relevant query
   - Test `get_full_resume` reassembles correctly

### Acceptance Criteria

- [x] `python scripts/ingest_resumes.py` successfully indexes all PDFs in `data/resumes/`
- [x] A semantic query like `"React developer with 3 years experience"` returns matching resumes
- [x] `get_full_resume(candidate_id)` returns the complete resume text
- [x] All tests in `tests/test_rag.py` pass (26/26 Phase 1 + 47 prior = 73 total, 0 regressions)

### Dependencies

Phase 0 must be complete (packages installed, directories created).

### Checkpoint Command

```bash
python -c "
from src.rag.store import get_vector_store
from src.rag.retriever import ResumeRetriever
store = get_vector_store('data/chroma_db')
retriever = ResumeRetriever(store)
results = retriever.search('React developer', top_k=3)
print(f'Found {len(results)} results')
for r in results:
    print(f'  - {r[\"name\"]} (score: {r[\"score\"]:.3f})')
"
```

---

## Phase 2 — Agent State & Core Type Definitions

**Goal**: Define every TypedDict, Pydantic model, and type that the agent state and tools use — the shared contract that all other modules depend on.

**Architecture Reference**: Section 3 (Agent State Design) — all TypedDicts and their field responsibilities

### Deliverables

| File | Purpose |
|------|---------|
| `src/agent/state.py` | `AgentState`, `CandidateMatch`, `Requirements`, `ScreeningRound` TypedDicts |
| `src/agent/models.py` | Pydantic models for tool input/output validation |
| `tests/test_state.py` | Verify state construction, default values, type compliance |

### Tasks

1. **`state.py`** — Implement all TypedDicts from architecture Section 3:
   - `CandidateMatch`: All 11 fields (candidate_id, name, score, must_have_score, nice_to_have_score, reasoning, strengths, gaps, resume_excerpts, interview_questions, hire_recommendation, improvement_suggestions)
   - `Requirements`: raw_jd, must_have, nice_to_have, experience_min_years, education_level, domain_keywords
   - `ScreeningRound`: round_number, round_type, candidates_evaluated, shortlisted_ids, eliminated_ids, notes
   - `AgentState`: All 14 fields from architecture, including `messages` with `Annotated[list[BaseMessage], add_messages]` reducer

2. **`models.py`** — Pydantic models for structured LLM outputs:
   - `SkillRequirement(skill: str, type: str, weight: float, evidence: str)`
   - `ExtractedRequirements(must_have: list[SkillRequirement], nice_to_have: list[SkillRequirement], experience_min_years: int | None, education_level: str | None, domain_keywords: list[str])`
   - `CandidateScore(must_have_score: float, nice_to_have_score: float, reasoning: str, strengths: list[str], gaps: list[str], excerpts: list[str])`
   - `InterviewQuestion(question: str, category: str, targets_gap: str, difficulty: str, follow_ups: list[str])`
   - `ComparisonRow(criterion: str, values: list[str])`
   - `ComparisonResult(candidates: list[dict], comparison_table: dict, summary: str)`

3. **`tests/test_state.py`**:
   - Create a valid `AgentState` with all fields populated, verify no type errors
   - Create a minimal `AgentState` with only required fields, verify defaults
   - Test `add_messages` reducer accumulates messages correctly
   - Test Pydantic model serialization/deserialization round-trip

### Acceptance Criteria

- [x] `python -c "from src.agent.state import AgentState; print('OK')"` succeeds
- [x] `python -c "from src.agent.models import ExtractedRequirements; print('OK')"` succeeds
- [x] `AgentState` can be constructed with all fields, no runtime errors
- [x] Pydantic models validate and reject invalid data (e.g., score > 1.0)
- [x] All tests pass (37 Phase 2 + 10 Phase 0 = 47 total, 0 regressions)

### Dependencies

Phase 0 only.

### Checkpoint Command

```bash
pytest tests/test_state.py -v
```

---

## Phase 3 — Prompt Engineering & Tool Implementation

**Goal**: Implement all 5 agent tools (`extract_requirements`, `rag_search`, `compare_candidates`, `generate_interview_questions`, file tools) with their prompt templates, and verify each one works independently before wiring them into the graph.

**Architecture Reference**: Section 6 (Tool Registry) — all 5 tools with full signatures and implementation notes

### Deliverables

| File | Purpose |
|------|---------|
| `src/prompts/extraction.py` | Prompt template for JD → structured requirements |
| `src/prompts/scoring.py` | Prompt template for candidate scoring |
| `src/prompts/comparison.py` | Prompt template for head-to-head comparison |
| `src/prompts/questions.py` | Prompt template for interview question generation |
| `src/prompts/explanation.py` | Prompt template for ranking explanation |
| `src/tools/extract_requirements.py` | `@tool extract_requirements(jd)` implementation |
| `src/tools/rag_search.py` | `@tool rag_search(query, top_k, filter)` implementation |
| `src/tools/compare_candidates.py` | `@tool compare_candidates(candidate_ids)` implementation |
| `src/tools/generate_questions.py` | `@tool generate_interview_questions(candidate_id)` implementation |
| `src/tools/file_tools.py` | File system tools (list, read, write, search, metadata) |
| `tests/test_tools.py` | Unit tests for each tool in isolation |

### Tasks

1. **Prompt templates** (`src/prompts/`):
   - **`extraction.py`**: System prompt that instructs the LLM to classify each JD requirement into must-have vs. nice-to-have, tag skill types, assign weights, and extract domain keywords. Use `with_structured_output(ExtractedRequirements)` for guaranteed schema.
   - **`scoring.py`**: System prompt that takes a candidate's resume text and the structured requirements, then outputs a `CandidateScore` with per-criterion scoring. Include instructions to cite evidence excerpts from the resume.
   - **`comparison.py`**: System prompt that takes multiple candidate profiles and generates a structured comparison table with a narrative summary.
   - **`questions.py`**: System prompt that takes a candidate profile and JD, then generates targeted interview questions focused on gaps and borderline areas.
   - **`explanation.py`**: System prompt that takes two candidates' scores and produces a point-by-point explanation of why one ranked higher.

2. **Tool implementations** (`src/tools/`):
   - **`extract_requirements.py`**:
     - Initialize `ChatGoogleGenerativeAI(model="gemini-2.0-flash")` inside the tool (or `ChatOllama(model="gemma2:9b")` for offline fallback)
     - Use the extraction prompt with `with_structured_output(ExtractedRequirements)`
     - Return the structured dict
   - **`rag_search.py`**:
     - Instantiate `ResumeRetriever` from Phase 1
     - Forward query and top_k to `retriever.search()`
     - Format results as the tool's output schema
   - **`compare_candidates.py`**:
     - Accept candidate IDs, fetch their `CandidateMatch` data from state (passed via RunnableConfig or injected)
     - Use the comparison prompt with structured output
     - Return `ComparisonResult` as dict
   - **`generate_questions.py`**:
     - Accept candidate_id, fetch full resume + requirements
     - Use the questions prompt with structured output
     - Return list of `InterviewQuestion` dicts
   - **`file_tools.py`**:
     - `list_files(directory)`, `read_file(path)`, `write_file(path, content)`, `search_files(query)`, `get_file_metadata(path)`
     - Simple wrappers around `os` / `pathlib` — no LLM needed

3. **`tests/test_tools.py`**:
   - Test `extract_requirements` with a real JD text, verify must-have/nice-to-have separation
   - Test `rag_search` with a query against the indexed corpus from Phase 1
   - Test file tools with temporary files
   - Mock LLM calls for `compare_candidates` and `generate_questions` to avoid cost in CI

### Acceptance Criteria

- [x] Each tool can be called as a standalone function and returns the correct schema
- [ ] `extract_requirements(sample_jd)` returns valid `ExtractedRequirements` with at least 2 must-have and 1 nice-to-have *(requires LLM — verified when Gemini quota resets)*
- [x] `rag_search("React developer")` returns non-empty results
- [x] All prompt templates produce well-formatted system messages
- [x] All tests pass (102/102 unit tests; 4 integration tests skipped pending LLM availability)

### Dependencies

Phase 1 (for `rag_search`), Phase 2 (for state types and Pydantic models)

### Checkpoint Command

```bash
python -c "
from src.tools.extract_requirements import extract_requirements
from src.tools.rag_search import rag_search
# Quick smoke test
jd = open('tests/fixtures/sample_jd.txt').read()
result = extract_requirements.invoke({'jd': jd})
print(f'Must-have: {len(result[\"must_have\"])} items')
print(f'Nice-to-have: {len(result[\"nice_to_have\"])} items')
"
```

---

## Phase 4 — LangGraph Agent (Linear Pipeline)

**Goal**: Build the LangGraph `StateGraph` with the linear pipeline nodes (`parse_jd` → `extract_requirements` → `search_resumes` → `rank_candidates` → `generate_report`) and verify end-to-end execution produces a ranked shortlist with reports.

**Architecture Reference**: Section 4 (Graph Workflow — Nodes 4.1–4.6), Section 5 (State Machine Diagram — linear portion)

### Deliverables

| File | Purpose |
|------|---------|
| `src/agent/nodes.py` | All node functions (parse_jd, extract_requirements, search_resumes, rank_candidates, generate_report) |
| `src/agent/edges.py` | Edge definitions and routing logic |
| `src/agent/graph.py` | `StateGraph` construction, node registration, edge wiring, `compile()` |
| `src/scoring/scorer.py` | Candidate scoring logic (per-candidate LLM call) |
| `src/scoring/ranker.py` | Sorting, filtering, composite score calculation |
| `src/reports/match_report.py` | Per-candidate markdown report generation |
| `tests/test_nodes.py` | Unit tests for individual nodes |
| `tests/test_graph_linear.py` | Integration test: invoke graph with a JD, verify shortlist output |

### Tasks

1. **`nodes.py`** — Implement each node function:
   - `parse_jd(state)`: Validate `raw_jd` is non-empty, store in state
   - `extract_requirements_node(state)`: Call `extract_requirements` tool, store result in `state["requirements"]`
   - `search_resumes_node(state)`: Build query from requirements, call `rag_search(top_k=100)`, store IDs in `state["all_candidate_ids"]`
   - `rank_candidates_node(state)`: For each candidate ID, retrieve full resume, score against requirements, sort by composite score (0.7 × must_have + 0.3 × nice_to_have), populate `state["current_shortlist"]`
   - `generate_report_node(state)`: For each shortlisted candidate, call report generator, store in `state["generated_reports"]`, set `state["awaiting_human_feedback"] = True`

2. **`scorer.py`** — Implement the scoring pipeline:
   - `score_candidate(resume_text, requirements, llm)`: Call the scoring prompt, get back `CandidateScore`
   - `compute_composite_score(must_have_score, nice_to_have_score)`: Return `0.7 * must_have + 0.3 * nice_to_have`
   - Validate scores are in `[0.0, 1.0]`, clamp if needed

3. **`ranker.py`** — Implement ranking logic:
   - `rank_candidates(candidate_matches)`: Sort by composite score descending
   - `filter_by_threshold(candidates, threshold=0.3)`: Remove candidates below minimum score
   - `shortlist(candidates, n=10)`: Return top N candidates

4. **`match_report.py`** — Implement report generation:
   - `generate_match_report(candidate, requirements)`: Produce the markdown report from architecture Section 10.1
   - Include: summary, score table, must-have breakdown, nice-to-have breakdown, strengths, gaps, red flags, improvement suggestions, hire recommendation

5. **`graph.py`** — Build the linear graph:
   ```python
   from langgraph.graph import StateGraph, START, END
   graph = StateGraph(AgentState)
   graph.add_node("parse_jd", parse_jd)
   graph.add_node("extract_requirements", extract_requirements_node)
   graph.add_node("search_resumes", search_resumes_node)
   graph.add_node("rank_candidates", rank_candidates_node)
   graph.add_node("generate_report", generate_report_node)

   graph.add_edge(START, "parse_jd")
   graph.add_edge("parse_jd", "extract_requirements")
   graph.add_edge("extract_requirements", "search_resumes")
   graph.add_edge("search_resumes", "rank_candidates")
   graph.add_edge("rank_candidates", "generate_report")
   graph.add_edge("generate_report", END)

   compiled = graph.compile()
   ```

6. **`tests/test_graph_linear.py`** — End-to-end test:
   - Provide a sample JD
   - Invoke the compiled graph
   - Assert `state["requirements"]` is populated
   - Assert `state["all_candidate_ids"]` is non-empty
   - Assert `state["current_shortlist"]` is sorted by score
   - Assert `state["generated_reports"]` has one report per shortlisted candidate

### Acceptance Criteria

- [ ] `graph.invoke({"raw_jd": sample_jd, "messages": []})` runs to completion
- [ ] Output state has populated `requirements`, `all_candidate_ids`, `current_shortlist`, `generated_reports`
- [ ] Shortlist is sorted by composite score in descending order
- [ ] Each report contains: scores, strengths, gaps, evidence excerpts
- [ ] All node unit tests pass

### Dependencies

Phases 1, 2, 3 must all be complete.

### Checkpoint Command

```bash
python scripts/run_agent.py --jd tests/fixtures/sample_jd.txt --mode linear
# Or directly:
python -c "
from src.agent.graph import create_graph
graph = create_graph()
result = graph.invoke({'raw_jd': open('tests/fixtures/sample_jd.txt').read(), 'messages': []})
for c in result['current_shortlist'][:5]:
    print(f'{c[\"name\"]}: {c[\"score\"]:.2f}')
print(f'Total reports: {len(result[\"generated_reports\"])}')
"
```

---

## Phase 5 — Interactive Loop (Human-in-the-Loop)

**Goal**: Add the `human_feedback_loop` node with intent classification, routing, and all interactive sub-nodes (`refine_requirements`, `compare_candidates`, `explain_ranking`, `generate_interview_questions`, `route_natural_language`) — turning the linear pipeline into a conversational agent.

**Architecture Reference**: Section 4 (Nodes 4.7–4.12), Section 7 (Interactive Features), Section 8 (Iterative Refinement)

### Deliverables

| File | Purpose |
|------|---------|
| `src/agent/nodes.py` (updated) | Add all interactive nodes |
| `src/agent/edges.py` (updated) | Add conditional routing logic from `human_feedback_loop` |
| `src/agent/graph.py` (updated) | Wire the feedback loop with conditional edges |
| `tests/test_intent_classification.py` | Test intent routing for various user inputs |
| `tests/test_conversation_flows.py` | Test multi-turn conversations |

### Tasks

1. **Intent classification** — Implement `classify_intent(human_feedback: str) -> str`:
   - Use LLM with a prompt that classifies the message into one of: `refine`, `compare`, `questions`, `explain`, `report`, `done`, `new_search`
   - Use structured output (`Literal` type) for reliable classification
   - Include a keyword-based fallback for common patterns (e.g., "compare" → `compare`, "why" → `explain`)

2. **Interactive nodes** — Add to `nodes.py`:
   - `human_feedback_loop(state)`: Read `human_feedback`, call `classify_intent`, set `state["next_action"]`
   - `refine_requirements_node(state)`: Parse user's modification, update requirements, increment version, re-rank, explain deltas
   - `compare_candidates_node(state)`: Extract candidate references from message, call `compare_candidates` tool, store result
   - `explain_ranking_node(state)`: Identify candidates being compared, generate explanation from their scores and reasoning
   - `generate_questions_node(state)`: Identify candidate, call `generate_interview_questions` tool, format response
   - `route_natural_language(state)`: Alias for `human_feedback_loop` — classifies and routes

3. **Conditional edges** — Update `edges.py`:
   ```python
   def route_feedback(state: AgentState) -> str:
       action = state.get("next_action", "done")
       routing = {
           "refine": "refine_requirements",
           "compare": "compare_candidates",
           "questions": "generate_interview_questions",
           "explain": "explain_ranking",
           "report": "generate_report",
           "done": END,
           "new_search": "parse_jd",
       }
       return routing.get(action, "human_feedback_loop")
   ```

4. **Graph update** — Add all interactive nodes and conditional edges to the graph. The graph now has a loop: `generate_report → human_feedback_loop → (conditional) → ... → generate_report → human_feedback_loop → ...`

5. **Iterative refinement** — Implement the delta tracking:
   - Before modifying requirements, snapshot the old shortlist rankings
   - After re-ranking, compare old vs. new rankings
   - Generate a natural-language delta summary (who moved up, who moved down, why)

6. **Tests**:
   - `test_intent_classification.py`: Feed 20+ user messages, verify correct intent classification
   - `test_conversation_flows.py`: Simulate multi-turn conversations using `graph.stream()` with manual state injection

### Acceptance Criteria

- [ ] The graph supports multi-turn interaction via `graph.stream()`
- [ ] "Compare the top 3" correctly routes to `compare_candidates` and returns a comparison
- [ ] "Why did X rank higher than Y?" returns a meaningful explanation
- [ ] "Drop AWS, add TypeScript" correctly updates requirements and re-ranks
- [ ] "done" exits the graph cleanly
- [ ] Intent classification accuracy > 90% on the 20+ test inputs

### Dependencies

Phase 4 must be complete (linear pipeline working).

### Checkpoint Command

```bash
# Run a simulated conversation
python -c "
from src.agent.graph import create_graph
graph = create_graph()
config = {'configurable': {'thread_id': 'test-1'}}

# Turn 1: Initial pipeline
result = graph.invoke({'raw_jd': open('tests/fixtures/sample_jd.txt').read(), 'messages': []})
print(f'Initial shortlist: {len(result[\"current_shortlist\"])} candidates')

# Turn 2: Compare
result = graph.invoke({
    **{k: v for k, v in result.items() if k != 'messages'},
    'human_feedback': 'Compare the top 3 candidates',
    'awaiting_human_feedback': True,
    'messages': result['messages'],
})
print(f'Comparison done: {result.get(\"comparison_result\") is not None}')

# Turn 3: Explain
result = graph.invoke({
    **{k: v for k, v in result.items() if k != 'messages'},
    'human_feedback': 'Why did the top candidate rank first?',
    'awaiting_human_feedback': True,
    'messages': result['messages'],
})
print('Explanation generated.')
"
```

---

## Phase 6 — Multi-Round Screening Pipeline

**Goal**: Implement the three-round funnel (initial broad screen → deep analysis → final hire recommendation) that progressively narrows and deepens the candidate evaluation, replacing the single-pass scoring in Phase 4.

**Architecture Reference**: Section 9 (Multi-Round Screening Pipeline) — all three rounds with code sketches, Section 15 (Error Handling — red-flag detection)

### Deliverables

| File | Purpose |
|------|---------|
| `src/screening/round1_initial.py` | Initial broad screen: RAG retrieval + must-have keyword filter |
| `src/screening/round2_deep.py` | Deep analysis: full resume review, skill verification, red-flag detection |
| `src/screening/round3_final.py` | Final recommendation: hire/no-hire with evidence |
| `src/screening/pipeline.py` | Orchestrator that runs all 3 rounds in sequence |
| `src/scoring/red_flags.py` | Red-flag detection logic (employment gaps, job-hopping, inconsistencies) |
| `tests/test_screening.py` | Integration tests for each round and the full pipeline |

### Tasks

1. **`round1_initial.py`** — Implement `initial_screen(state) -> AgentState`:
   - Use `rag_search(query, top_k=100)` to retrieve broad pool
   - For each candidate, perform a quick must-have keyword check against resume excerpts
   - Score candidates using the lightweight scoring from Phase 4
   - Filter out candidates missing >30% of must-have criteria
   - Shortlist top 10
   - Record `ScreeningRound(round_number=1, ...)` in state

2. **`round2_deep.py`** — Implement `deep_analysis(state) -> AgentState`:
   - For each of the top 10, retrieve the full resume text
   - **Skill depth verification**: For each must-have skill, check if the resume provides evidence of actual usage (not just keyword mentions). Use LLM to verify.
   - **Experience timeline analysis**: Extract role durations, calculate total relevant experience, check career progression
   - **Red-flag detection**: Call `red_flags.py` to check for:
     - Unexplained employment gaps (>3 months)
     - Job-hopping pattern (>3 jobs in 2 years)
     - Inconsistent dates or titles
   - Re-score with the deeper evidence
   - Re-rank and shortlist top 5–7
   - Record `ScreeningRound(round_number=2, ...)`

3. **`round3_final.py`** — Implement `final_recommendation(state) -> AgentState`:
   - For each remaining candidate, compile all evidence from Rounds 1 and 2
   - Generate a hire/no-hire/borderline recommendation using LLM
   - Generate improvement suggestions for borderline candidates
   - Generate the full match report (from Phase 4's `match_report.py`)
   - Record `ScreeningRound(round_number=3, ...)`

4. **`red_flags.py`** — Implement detection functions:
   - `detect_employment_gaps(timeline)`: Find gaps >3 months, flag unexplained ones
   - `detect_job_hopping(timeline)`: Flag if >3 roles in 24 months
   - `detect_inconsistencies(resume_text)`: Check for date overlaps, conflicting titles

5. **`pipeline.py`** — Implement `run_screening_pipeline(state) -> AgentState`:
   - Sequentially call: `initial_screen` → `deep_analysis` → `final_recommendation`
   - After each round, update `state["current_round"]`
   - Handle edge cases: fewer than 10 candidates, all eliminated in a round

6. **Update `rank_candidates_node`** in `nodes.py`:
   - Replace single-pass scoring with `run_screening_pipeline`
   - This connects the multi-round pipeline into the graph

7. **Tests**:
   - Test Round 1 filters correctly (must-have check)
   - Test Round 2 deep scoring changes rankings from Round 1
   - Test Round 3 generates valid hire/no-hire recommendations
   - Test full pipeline with 3 sample resumes

### Acceptance Criteria

- [ ] `run_screening_pipeline(state)` processes all 3 rounds sequentially
- [ ] Round 1 narrows from ~100 to 10 candidates
- [ ] Round 2 re-ranks and narrows to 5–7 candidates
- [ ] Round 3 produces hire/no-hire/borderline for every candidate
- [ ] Red-flag detection correctly identifies gaps and job-hopping
- [ ] Each round records a `ScreeningRound` in `state["screening_rounds"]`
- [ ] All tests pass

### Dependencies

Phases 1, 2, 3, 4 must be complete. Phase 5 can run in parallel but the graph should use the linear pipeline for now.

### Checkpoint Command

```bash
python -c "
from src.agent.graph import create_graph
graph = create_graph()
result = graph.invoke({'raw_jd': open('tests/fixtures/sample_jd.txt').read(), 'messages': []})
print(f'Rounds completed: {len(result[\"screening_rounds\"])}')
for r in result['screening_rounds']:
    print(f'  Round {r[\"round_number\"]} ({r[\"round_type\"]}): '
          f'{r[\"candidates_evaluated\"]} evaluated, '
          f'{len(r[\"shortlisted_ids\"])} shortlisted')
for c in result['current_shortlist']:
    print(f'  {c[\"name\"]}: {c[\"hire_recommendation\"]} (score: {c[\"score\"]:.2f})')
"
```

---

## Phase 7 — User Interface (Streamlit + CLI)

**Goal**: Build the Streamlit chat interface and CLI fallback that users interact with, connecting the compiled LangGraph agent to a real conversation loop.

**Architecture Reference**: Section 14 (Interface Design) — Streamlit wireframe and CLI mock

### Deliverables

| File | Purpose |
|------|---------|
| `ui/streamlit_app.py` | Full Streamlit chat interface with sidebar |
| `ui/cli_app.py` | Click/Typer-based CLI interface |
| `ui/components.py` | Shared UI helpers (report rendering, comparison tables) |

### Tasks

1. **`streamlit_app.py`** — Build the Streamlit app:
   - **Sidebar**:
     - File uploader for JD (`.txt` or `.pdf`)
     - Display extracted requirements (must-have / nice-to-have checkboxes)
     - Screening round progress indicator (Round 1/2/3)
     - Quick action buttons: "Run Next Round", "Export Reports", "Reset Session"
   - **Main area**: `st.chat_message` based chat interface
   - **Conversation loop**:
     - User types message → append to state → invoke graph step → display response
     - Use `graph.stream()` for streaming responses
   - **Report display**: Render markdown reports using `st.markdown()` with expandable sections
   - **Comparison display**: Render comparison tables using `st.dataframe()`
   - **State management**: Use `st.session_state` to persist the agent state across reruns

2. **`cli_app.py`** — Build the CLI:
   - Use `click` for command structure
   - Command: `python -m src.cli_app start --jd <path>`
   - Interactive `input()` loop:
     - Print agent responses to stdout
     - Accept user input, feed to graph
     - Color-coded output (candidate names in bold, scores highlighted)
   - Exit on "done" or Ctrl+C
   - Save final reports to `data/reports/`

3. **`components.py`** — Shared helpers:
   - `render_match_report(report_md)`: Render a match report in both Streamlit and CLI formats
   - `render_comparison_table(comparison_result)`: Render comparison in both formats
   - `render_ranking_delta(old_shortlist, new_shortlist)`: Show ranking changes after refinement

### Acceptance Criteria

- [ ] `streamlit run ui/streamlit_app.py` launches without errors
- [ ] User can paste a JD and see extracted requirements in the sidebar
- [ ] Chat interface accepts free-form queries and returns agent responses
- [ ] "Compare the top 3" renders a comparison table
- [ ] Reports are displayed as formatted markdown
- [ ] CLI interface works: `python -m src.cli_app start --jd tests/fixtures/sample_jd.txt`
- [ ] Both interfaces support the full 7 test conversation patterns from architecture Section 7.2

### Dependencies

Phases 4, 5, 6 must be complete (the agent graph must be fully functional).

### Checkpoint Command

```bash
streamlit run ui/streamlit_app.py --server.headless true
# In another terminal:
python -m src.cli_app start --jd tests/fixtures/sample_jd.txt
```

---

## Phase 8 — Testing, Polish & Demo Preparation

**Goal**: Write all remaining tests (5+ end-to-end conversation flows), fix bugs, export the state machine diagram as a visual image, and prepare the demo video script.

**Architecture Reference**: Section 16 (Testing Strategy) — 7 test scenarios, Section 5 (State Machine Diagram), Submission Guidelines

### Deliverables

| File | Purpose |
|------|---------|
| `tests/test_agent_flows.py` | All 7 end-to-end conversation flow tests |
| `tests/fixtures/sample_jd.txt` | Standard test JD |
| `tests/fixtures/expected_outputs/` | Expected outputs for regression testing |
| `docs/state_machine_diagram.png` | Exported PNG of the Mermaid diagram |
| `docs/demo_script.md` | Step-by-step demo video script (5–6 minutes) |
| `matching_agent.py` | Final agent entry point (as required by submission) |

### Tasks

1. **End-to-end tests** (`tests/test_agent_flows.py`) — Implement all 7 scenarios from architecture Section 16.1:
   - **Scenario 1 (Happy path)**: Full JD → ranked shortlist → reports. Verify all state fields populated.
   - **Scenario 2 (Refinement)**: Run pipeline, then modify requirements, verify re-ranking and delta explanation.
   - **Scenario 3 (Comparison)**: Run pipeline, then compare 3 candidates, verify structured comparison output.
   - **Scenario 4 (Explanation)**: Run pipeline, ask "Why did X rank higher than Y?", verify explanation mentions both names.
   - **Scenario 5 (Interview questions)**: Generate questions for a candidate, verify 5 questions with categories.
   - **Scenario 6 (Edge case — no results)**: Use an impossible JD, verify graceful "no candidates" message.
   - **Scenario 7 (Multi-round)**: Run full 3-round pipeline, verify all rounds complete, hire recommendations generated.

2. **State machine diagram export**:
   - Take the Mermaid code from architecture Section 5
   - Use `mmdc` (Mermaid CLI) or an online renderer to produce a PNG
   - Save as `docs/state_machine_diagram.png`
   - Include in the project root and demo

3. **`matching_agent.py`** — Create the required submission entry point:
   - Imports and re-exports the compiled graph from `src/agent/graph.py`
   - Provides a simple `run()` function that starts the Streamlit app
   - Meets the assignment's requirement: "LangGraph-based agent implementation in `matching_agent.py`"

4. **Demo script** (`docs/demo_script.md`):
   - **0:00–0:30** — Introduction: show the project structure and explain the architecture
   - **0:30–1:30** — Run the agent with a JD: show JD parsing, requirement extraction, initial search
   - **1:30–2:30** — Show the ranked results: explain scores, show match reports
   - **2:30–3:30** — Interactive refinement: modify requirements, show re-ranking and delta explanation
   - **3:30–4:15** — Compare candidates: side-by-side comparison, explanation queries
   - **4:15–4:45** — Generate interview questions: show targeted questions for a candidate
   - **4:45–5:00** — Multi-round screening: walk through all 3 rounds briefly
   - **5:00–5:30** — Summary and Q&A
   - Include exact queries to type and expected agent responses

5. **Bug fixes and polish**:
   - Run full test suite, fix any failures
   - Add proper error messages for edge cases
   - Verify token usage is reasonable (no unnecessary re-processing)
   - Clean up print statements, add proper logging

6. **Coverage check**:
   - Run `pytest --cov=src --cov-report=term-missing`
   - Target: tools 90%+, scoring 95%+, screening 80%+

### Acceptance Criteria

- [ ] All 7 test scenarios pass end-to-end
- [ ] Overall test coverage meets targets (see architecture Section 16.3)
- [ ] State machine diagram is exported as a clear, readable PNG
- [ ] Demo script is detailed enough for a 5–6 minute walkthrough
- [ ] `matching_agent.py` exists and can be imported/invoked
- [ ] `streamlit run matching_agent.py` launches the app
- [ ] No open TODO comments or placeholder code

### Dependencies

All previous phases (0–7) must be complete.

### Checkpoint Command

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Phase Dependency Graph

```
Phase 0 (Setup)
    │
    ├──▶ Phase 1 (RAG)
    │       │
    ├──▶ Phase 2 (State & Types)
    │       │
    └──┬───┘
       │
       ▼
   Phase 3 (Tools) ◄── needs RAG + State
       │
       ▼
   Phase 4 (Linear Graph) ◄── needs Tools
       │
       ├──▶ Phase 5 (Interactive Loop) ◄── extends Graph
       │
       ├──▶ Phase 6 (Multi-Round Screening) ◄── extends Scoring
       │       │
       └───┬───┘
           │
           ▼
       Phase 7 (UI) ◄── needs full Agent
           │
           ▼
       Phase 8 (Testing & Demo) ◄── needs everything
```

**Note**: Phases 5 and 6 can be developed **in parallel** since they modify different parts of the graph (5 adds the interactive loop; 6 replaces the scoring pipeline). However, both must be complete before Phase 7.

---

## Effort Estimation

| Phase | Effort (hours) | Key Risk |
|-------|---------------|----------|
| 0 — Setup | 1–2 | Low — boilerplate |
| 1 — RAG | 4–6 | Medium — PDF parsing edge cases, embedding costs |
| 2 — State & Types | 2–3 | Low — pure data definitions |
| 3 — Tools & Prompts | 6–8 | High — prompt tuning requires iteration |
| 4 — Linear Graph | 4–6 | Medium — wiring nodes correctly |
| 5 — Interactive Loop | 6–8 | High — intent classification accuracy, state management |
| 6 — Multi-Round | 6–8 | High — red-flag heuristics, LLM consistency across rounds |
| 7 — UI | 4–6 | Medium — Streamlit session state quirks |
| 8 — Testing & Demo | 4–6 | Medium — test data setup, demo recording |
| **Total** | **37–53 hours** | |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LLM output inconsistency (same input, different scores) | High | High | Use structured output (`with_structured_output`), temperature=0, add few-shot examples in prompts |
| High token cost during development | Medium | Medium | Use Gemini Free Tier (zero cost); cache LLM calls in tests; mock in unit tests. With Ollama fallback, cost is always zero. |
| ChromaDB persistence issues | Low | Medium | Use `PersistentClient` with absolute path, test recovery after restart |
| Intent classification misroutes user queries | Medium | Medium | Keyword-based fallback, test with 20+ diverse inputs, add "I didn't understand" fallback response |
| Resume PDFs with unusual formatting | High | Low | Use PyMuPDF (robust), add fallback OCR if needed, log unparseable files |
| Multi-round scoring inconsistency (candidate drops between rounds) | Medium | Medium | Keep all candidates in state across rounds (just re-score), allow manual promotion |

---

## Quick Reference: What to Build in Each Phase

| Phase | Build This | Verify With |
|-------|-----------|-------------|
| 0 | Directory tree, `pyproject.toml`, `.env` | `import langgraph` works |
| 1 | `src/rag/` (store, indexer, retriever) | Search returns resumes |
| 2 | `src/agent/state.py`, `src/agent/models.py` | State construction, type validation |
| 3 | `src/prompts/`, `src/tools/` | Each tool returns correct schema |
| 4 | `src/agent/nodes.py`, `src/agent/graph.py` (linear) | JD in → ranked shortlist out |
| 5 | Interactive nodes, conditional edges, intent routing | Multi-turn chat works |
| 6 | `src/screening/` (3 rounds), `src/scoring/red_flags.py` | 3 rounds complete, hire/no-hire generated |
| 7 | `ui/streamlit_app.py`, `ui/cli_app.py` | Full chat UI works end-to-end |
| 8 | Tests (7 scenarios), diagram PNG, demo script | All tests pass, demo runnable |