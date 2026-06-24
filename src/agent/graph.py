"""
Agentic Profile Matching — LangGraph StateGraph Construction.

Builds the full graph with linear pipeline + registered interactive nodes.

Phase 4 linear flow:
  START -> parse_jd -> extract_requirements -> search_resumes
         -> rank_candidates -> generate_report -> END

Phase 5 interactive nodes are registered on the graph and can be
invoked directly. Multi-turn interaction is managed by the caller
(UI layer in Phase 7) which re-invokes the graph with updated
state including human_feedback. The route_feedback edge function
maps intents to interactive nodes.

Architecture Reference: architecture.md Section 4, Section 5
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agent.edges import (
    route_after_parse_jd,
    route_after_ranking,
    route_after_requirements,
    route_after_search,
    route_feedback,
)
from src.agent.nodes import (
    compare_candidates_node,
    explain_ranking_node,
    extract_requirements_node,
    generate_questions_node,
    generate_report_node,
    human_feedback_loop,
    parse_jd,
    rank_candidates_node,
    refine_requirements_node,
    search_resumes_node,
)
from src.agent.state import AgentState


def create_graph() -> StateGraph:
    """Build and return the compiled graph.

    Linear pipeline (Phase 4):
      START -> parse_jd -> extract_requirements -> search_resumes
             -> rank_candidates -> generate_report -> END

    Interactive nodes (Phase 5) are registered and reachable via
    human_feedback_loop -> route_feedback -> interactive_node -> END.
    The caller manages multi-turn by re-invoking with updated state.

    Returns:
        Compiled LangGraph StateGraph ready for .invoke().
    """
    graph = StateGraph(AgentState)

    # --- Add all nodes (linear + interactive) ---
    graph.add_node("parse_jd", parse_jd)
    graph.add_node("extract_requirements", extract_requirements_node)
    graph.add_node("search_resumes", search_resumes_node)
    graph.add_node("rank_candidates", rank_candidates_node)
    graph.add_node("generate_report", generate_report_node)

    # Phase 5 interactive nodes
    graph.add_node("human_feedback_loop", human_feedback_loop)
    graph.add_node("refine_requirements", refine_requirements_node)
    graph.add_node("compare_candidates", compare_candidates_node)
    graph.add_node("explain_ranking", explain_ranking_node)
    graph.add_node("generate_interview_questions", generate_questions_node)

    # --- Linear pipeline edges (Phase 4) ---
    graph.add_edge(START, "parse_jd")
    graph.add_conditional_edges("parse_jd", route_after_parse_jd)
    graph.add_conditional_edges("extract_requirements", route_after_requirements)
    graph.add_conditional_edges("search_resumes", route_after_search)
    graph.add_conditional_edges("rank_candidates", route_after_ranking)
    graph.add_edge("generate_report", END)

    # --- Interactive feedback path (Phase 5) ---
    # human_feedback_loop classifies intent and routes to the
    # appropriate interactive node, each of which ends at END.
    graph.add_conditional_edges("human_feedback_loop", route_feedback)

    return graph.compile()


def invoke_linear_pipeline(raw_jd: str) -> dict:
    """Convenience function: invoke the linear pipeline with a JD.

    Args:
        raw_jd: The job description text.

    Returns:
        The final state dict after the graph completes.
    """
    graph = create_graph()
    return graph.invoke({"raw_jd": raw_jd, "messages": []})
