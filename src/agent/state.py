"""
Agentic Profile Matching — Agent State & Core Type Definitions.

Defines all TypedDict state schemas used by the LangGraph agent.
These TypedDicts serve as the single source of truth passed between
every node in the graph.

Architecture Reference: architecture.md Section 3 (Agent State Design)
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class CandidateMatch(TypedDict, total=False):
    """Represents a single candidate's match result.

    All fields are optional at the TypedDict level because the agent
    populates them progressively: basic identity first, scores later,
    and reasoning/excerpts last.
    """

    candidate_id: str
    name: str
    score: float  # 0.0 – 1.0
    must_have_score: float  # subset score for must-have criteria
    nice_to_have_score: float  # subset score for nice-to-have criteria
    reasoning: str  # LLM-generated explanation
    strengths: list[str]  # matched strengths
    gaps: list[str]  # missing or weak areas
    resume_excerpts: list[str]  # evidence snippets from resume
    interview_questions: list[str]  # generated screening questions
    hire_recommendation: str  # "hire" / "no_hire" / "borderline"
    improvement_suggestions: list[str]


class Requirements(TypedDict, total=False):
    """Structured representation of job requirements extracted from a JD.

    Each skill dict follows the schema:
        {"skill": str, "type": str, "weight": float}

    where type is one of "tech", "soft", "domain", "certification", etc.
    """

    raw_jd: str
    must_have: list[dict]  # [{"skill": "React", "type": "tech", "weight": 1.0}, ...]
    nice_to_have: list[dict]  # [{"skill": "AWS", "type": "tech", "weight": 0.5}, ...]
    experience_min_years: int | None
    education_level: str | None
    domain_keywords: list[str]


class ScreeningRound(TypedDict, total=False):
    """Tracks results for one round of the multi-round screening pipeline.

    Round types: "initial" | "deep_analysis" | "final"
    """

    round_number: int
    round_type: str  # "initial" | "deep_analysis" | "final"
    candidates_evaluated: int
    shortlisted_ids: list[str]
    eliminated_ids: list[str]
    notes: str


class AgentState(TypedDict, total=False):
    """Full state of the matching agent.

    This is the top-level state object passed between every node in the
    LangGraph StateGraph. All fields are optional (total=False) so that
    the state can be built up incrementally across nodes.

    The ``messages`` field uses the ``add_messages`` reducer to
    accumulate LangChain message objects across turns.
    """

    # --- Conversation ---
    messages: Annotated[list[BaseMessage], add_messages]
    conversation_history: list[dict]  # [{role, content, timestamp}]

    # --- Job Understanding ---
    raw_jd: str
    requirements: Requirements
    requirements_version: int  # increments on each refinement

    # --- Candidate Pipeline ---
    all_candidate_ids: list[str]  # all IDs retrieved from RAG
    current_shortlist: list[CandidateMatch]
    screening_rounds: list[ScreeningRound]
    current_round: int

    # --- Comparison ---
    comparison_result: dict  # head-to-head comparison output

    # --- Reports ---
    generated_reports: dict[str, str]  # candidate_id -> report markdown

    # --- Control Flow ---
    awaiting_human_feedback: bool
    human_feedback: str
    next_action: str  # routing hint for the graph
    error: str