"""
Agentic Profile Matching — Edge Definitions.

Defines routing logic and conditional edges for the LangGraph.
Phase 4: Linear edges (no conditionals on the main path).
Phase 5: Adds route_feedback for the interactive loop.

Architecture Reference: architecture.md Section 4 (Graph Workflow)
"""

from __future__ import annotations

from src.agent.state import AgentState


_FEEDBACK_ROUTING: dict[str, str] = {
    "refine": "refine_requirements",
    "compare": "compare_candidates",
    "questions": "generate_interview_questions",
    "explain": "explain_ranking",
    "report": "generate_report",
    "done": "__end__",
    "new_search": "parse_jd",
}


def route_after_parse_jd(state: AgentState) -> str:
    """Route after parse_jd. Route to END if JD is too short."""
    error = state.get("error", "")
    if error and "too short" in error.lower():
        return "__end__"
    return "extract_requirements"


def route_after_requirements(state: AgentState) -> str:
    """Route after extract_requirements. Always continue."""
    return "search_resumes"


def route_after_search(state: AgentState) -> str:
    """Route after search_resumes. Always continue."""
    return "rank_candidates"


def route_after_ranking(state: AgentState) -> str:
    """Route after rank_candidates. Always continue."""
    return "generate_report"


def route_feedback(state: AgentState) -> str:
    """Route based on classified user intent (Phase 5).

    Reads state["next_action"] and maps to the appropriate node.
    Falls back to human_feedback_loop if unrecognized.
    """
    action = state.get("next_action", "")
    return _FEEDBACK_ROUTING.get(action, "human_feedback_loop")


def route_after_interactive(state: AgentState) -> str:
    """Route after any interactive node back to human_feedback_loop."""
    return "human_feedback_loop"
