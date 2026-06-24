# Agentic Profile Matching — Problem Statement & Requirements

## Project Overview

Build an **AI-powered candidate-job matching agent** using **LangGraph** that intelligently matches job descriptions (JDs) to candidate resumes from a document corpus. The agent must support multi-round screening, natural language interaction, and full explainability of its ranking decisions.

This project is structured as a **context-engineering** effort: the problem statement lives here (`context.md`), and all downstream design artifacts (architecture, implementation plans, etc.) are derived from it.

---

## Assignment Requirements

### Part A: Agent Architecture (40%)

#### Agent State Design
- Track **conversation history** (user queries and agent responses across turns)
- Maintain **job requirements understanding** (parsed and enriched over time)
- Store **candidate shortlist and reasoning** (ranked candidates with per-candidate match explanations)

#### Agent Workflow (Graph Structure)

```
START → Parse JD → Extract Requirements → Search Resumes → Rank Candidates → Generate Report → Human Feedback Loop → END
```

Each node is a discrete, typed function with well-defined input/output contracts.

#### Tools Available to Agent

**Carried forward from prior milestones:**
- All file system tools (from Milestone 1)
- RAG search tool (from Milestone 2)

**New tools for this milestone:**
| Tool | Signature | Purpose |
|------|-----------|---------|
| `extract_requirements` | `(jd: str) → Requirements` | Parse JD into must-have vs. nice-to-have criteria |
| `compare_candidates` | `(candidate_ids: list[str]) → Comparison` | Head-to-head comparison of multiple candidates |
| `generate_interview_questions` | `(candidate_id: str) → Questions` | Create targeted screening questions for a candidate |

---

### Part B: Interactive Features (30%)

#### Conversational Interface
The agent must accept free-form natural language queries, for example:
- "Find me candidates with React and 3+ years experience"
- "Compare the top 3 matches side by side"
- "Why did John rank higher than Jane?"

The agent should interpret intent, route to the appropriate tool(s), and return a human-readable response.

#### Iterative Refinement
- Users can **adjust requirements mid-conversation** (e.g., "Actually, drop the cloud requirement and add TypeScript")
- The agent **re-ranks candidates** based on the updated criteria
- The agent **explains what changed** in the rankings and why

---

### Part C: Advanced Capabilities (30%)

#### Multi-Round Screening Pipeline
1. **Initial Screen** — From ~100 resumes, shortlist the top 10
2. **Second Round** — Deep analysis of the top 10 (skill-depth verification, experience validation, red-flag detection)
3. **Final Round** — Generate a hire / no-hire recommendation per candidate with supporting evidence

#### Explainability
- Generate **detailed match reports** per candidate
- **Highlight strengths and gaps** relative to the JD
- Provide **improvement suggestions** for borderline candidates (e.g., "Candidate would be a stronger fit with a cloud certification")

---

## Submission Guidelines

| Deliverable | Description |
|-------------|-------------|
| LangGraph agent implementation | `matching_agent.py` — the full graph-based agent |
| State machine diagram | Visual representation of the agent graph |
| Chat interface | CLI or Streamlit/Gradio frontend |
| Test scenarios | 5+ documented conversation flows |
| Demo video | 5–6 minutes showing agent reasoning in action |

---

## Constraints & Assumptions

- **Framework**: LangGraph (Python)
- **Retrieval**: RAG over a corpus of resumes (from Milestone 2)
- **Interface**: CLI, Streamlit, or Gradio
- **Scope**: Single JD matching against a resume corpus per session
- **Language**: Python 3.11+

---

## Cost Constraints (Strict)

The entire project must be **buildable, runnable, and demonstrable using free-tier or open-source resources only**. This is a non-negotiable project constraint.

- **No paid services, subscriptions, or paid APIs.** Every dependency must be free or open-source.
- **LLM**: Gemini API (Free Tier) is the primary provider. Ollama (local models) is the fallback for fully offline operation.
- **Embeddings**: ChromaDB's built-in ONNX embedding function (local) — zero cost, no API key, no rate limits. Ships with ChromaDB, no extra install.
- **Vector Database**: ChromaDB (local, embedded) — no hosted services.
- **Infrastructure**: All local — SQLite/ChromaDB, local file storage, in-memory caching. No managed cloud services.
- **Decision policy**: Before introducing any new dependency, verify it is free. If multiple options exist, choose the one that best balances zero cost, developer experience, performance, and maintainability. If a feature cannot be implemented without payment, document the limitation explicitly rather than silently introducing a paid dependency.

### Reviewer Cloneability

The project must be **cloneable by any reviewer** who should be able to run it locally after adding **only freely obtainable API keys** (if required). No paid accounts, premium services, or proprietary infrastructure should be necessary. A reviewer with `git clone`, `pip install`, and a free Gemini API key (or Ollama installed locally) must be able to run every feature end-to-end.

---

## Success Criteria

1. The agent correctly parses a JD into structured requirements
2. The agent retrieves and ranks candidates using RAG + scoring logic
3. Users can interact conversationally and iteratively refine searches
4. Multi-round screening produces progressively deeper analysis
5. Every ranking decision is explainable (match reports with evidence)
6. The graph structure is clearly visualized and documented
7. At least 5 test conversation flows pass end-to-end