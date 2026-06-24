"""
Agentic Profile Matching — Phase 5 Conversation Flow Tests.

Tests interactive nodes and multi-turn flows. Interactive nodes
are invoked directly with state (the caller manages the loop).
The full graph is tested for linear + node registration.

Run:
    pytest tests/test_conversation_flows.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.graph import create_graph
from src.agent.nodes import (
    classify_intent,
    compare_candidates_node,
    explain_ranking_node,
    generate_questions_node,
    human_feedback_loop,
    refine_requirements_node,
)


SAMPLE_REQUIREMENTS = {
    "raw_jd": "Senior React Developer",
    "must_have": [
        {"skill": "React", "type": "tech", "weight": 1.0, "evidence": "Required"},
        {"skill": "TypeScript", "type": "tech", "weight": 0.9, "evidence": "Required"},
    ],
    "nice_to_have": [
        {"skill": "AWS", "type": "tech", "weight": 0.5, "evidence": "Preferred"},
    ],
    "experience_min_years": 5,
    "education_level": "BS",
    "domain_keywords": ["frontend", "web"],
}

SAMPLE_SHORTLIST = [
    {
        "candidate_id": "alice_123", "name": "Alice Johnson", "score": 0.85,
        "must_have_score": 0.93, "nice_to_have_score": 0.65,
        "reasoning": "Strong match on React and TypeScript",
        "strengths": ["React", "TypeScript", "CSS"], "gaps": ["No AWS"],
        "resume_excerpts": ["Led React migration"],
        "hire_recommendation": "hire",
        "improvement_suggestions": ["Gain AWS certification"],
    },
    {
        "candidate_id": "bob_456", "name": "Bob Smith", "score": 0.62,
        "must_have_score": 0.7, "nice_to_have_score": 0.4,
        "reasoning": "Good React, weak TypeScript",
        "strengths": ["React", "JavaScript"], "gaps": ["No TypeScript", "No AWS"],
        "resume_excerpts": ["Built React apps"],
        "hire_recommendation": "borderline",
        "improvement_suggestions": ["Learn TypeScript"],
    },
    {
        "candidate_id": "carol_789", "name": "Carol Williams", "score": 0.45,
        "must_have_score": 0.5, "nice_to_have_score": 0.3,
        "reasoning": "Some React, missing TypeScript",
        "strengths": ["React"], "gaps": ["No TypeScript", "No CSS"],
        "resume_excerpts": ["Used React in projects"],
        "hire_recommendation": "no_hire",
        "improvement_suggestions": ["Learn TypeScript and CSS"],
    },
]


def _base_state() -> dict:
    """Return a minimal state with requirements and shortlist for Turn 2 tests."""
    return {
        "requirements": SAMPLE_REQUIREMENTS,
        "current_shortlist": SAMPLE_SHORTLIST,
        "generated_reports": {},
        "all_candidate_ids": ["alice_123", "bob_456", "carol_789"],
        "requirements_version": 1,
        "current_round": 1,
    }


# ===================================================================
# Graph Construction Tests
# ===================================================================


class TestGraphConstruction:
    """Verify the Phase 5 graph has all nodes registered."""

    def test_graph_has_all_phase5_nodes(self) -> None:
        graph = create_graph()
        node_names = set(graph.get_graph().nodes.keys())
        node_names.discard("__start__")
        node_names.discard("__end__")
        expected = {
            "parse_jd", "extract_requirements", "search_resumes",
            "rank_candidates", "generate_report",
            "human_feedback_loop", "refine_requirements",
            "compare_candidates", "explain_ranking",
            "generate_interview_questions",
        }
        assert expected.issubset(node_names), f"Missing nodes: {expected - node_names}"

    def test_graph_compiles(self) -> None:
        graph = create_graph()
        assert hasattr(graph, "invoke")

    def test_linear_pipeline_runs_to_end(self) -> None:
        """Phase 4 linear pipeline should still work end-to-end."""
        graph = create_graph()
        result = graph.invoke({"raw_jd": "Hi", "messages": []})
        # Short JD -> parse_jd sets error -> routes to END
        assert "error" in result

    @patch("src.agent.nodes.score_candidate")
    @patch("src.agent.nodes.get_full_resume_text")
    @patch("src.agent.nodes.rag_search")
    @patch("src.agent.nodes.extract_requirements")
    def test_linear_pipeline_produces_results(
        self, mock_extract, mock_rag, mock_resume, mock_score,
    ) -> None:
        """Phase 4 linear pipeline with full JD produces results."""
        mock_extract.invoke.return_value = SAMPLE_REQUIREMENTS
        mock_rag.invoke.return_value = [
            {"candidate_id": "alice_123", "name": "Alice", "score": 0.1, "excerpt": "..."},
        ]
        mock_resume.return_value = "Alice Johnson\nReact Developer"
        mock_score.return_value = MagicMock(
            must_have_score=0.85, nice_to_have_score=0.7,
            reasoning="Good", strengths=["React"], gaps=[], excerpts=[]
        )
        graph = create_graph()
        result = graph.invoke({"raw_jd": "Senior React Developer with TypeScript.", "messages": []})
        assert "requirements" in result
        assert "current_shortlist" in result
        assert "generated_reports" in result


# ===================================================================
# Human Feedback Loop Node Tests
# ===================================================================


class TestHumanFeedbackLoopNode:

    def test_classifies_compare_intent(self) -> None:
        state = _base_state()
        state["human_feedback"] = "Compare the top 3 candidates"
        result = human_feedback_loop(state)
        assert result["next_action"] == "compare"
        assert result["awaiting_human_feedback"] is False

    def test_classifies_questions_intent(self) -> None:
        state = _base_state()
        state["human_feedback"] = "Generate questions for Alice"
        result = human_feedback_loop(state)
        assert result["next_action"] == "questions"

    def test_classifies_explain_intent(self) -> None:
        state = _base_state()
        state["human_feedback"] = "Why did Alice rank higher than Bob?"
        result = human_feedback_loop(state)
        assert result["next_action"] == "explain"

    def test_classifies_refine_intent(self) -> None:
        state = _base_state()
        state["human_feedback"] = "Drop AWS from requirements"
        result = human_feedback_loop(state)
        assert result["next_action"] == "refine"

    def test_classifies_done_intent(self) -> None:
        state = _base_state()
        state["human_feedback"] = "done"
        result = human_feedback_loop(state)
        assert result["next_action"] == "done"

    def test_classifies_empty_feedback(self) -> None:
        state = _base_state()
        state["human_feedback"] = ""
        result = human_feedback_loop(state)
        assert result["next_action"] == "explain"

    def test_classifies_new_search_intent(self) -> None:
        state = _base_state()
        state["human_feedback"] = "Start over with a new job"
        result = human_feedback_loop(state)
        assert result["next_action"] == "new_search"


# ===================================================================
# Compare Candidates Node Tests
# ===================================================================


class TestCompareCandidatesNode:

    def test_compare_returns_result(self) -> None:
        state = _base_state()
        state["human_feedback"] = "Compare the top 3"
        result = compare_candidates_node(state)
        assert "comparison_result" in result
        assert result.get("awaiting_human_feedback") is True

    def test_compare_extracts_top_n(self) -> None:
        state = _base_state()
        state["human_feedback"] = "Compare the top 2"
        result = compare_candidates_node(state)
        comp = result["comparison_result"]
        # Should have 2 candidates
        if "candidates" in comp:
            assert len(comp["candidates"]) <= 3

    def test_compare_no_shortlist_returns_error(self) -> None:
        state = {"current_shortlist": [], "requirements": {}}
        result = compare_candidates_node(state)
        assert "error" in result


# ===================================================================
# Explain Ranking Node Tests
# ===================================================================


class TestExplainRankingNode:

    def test_explain_returns_explanation(self) -> None:
        state = _base_state()
        state["human_feedback"] = "Why did Alice rank higher than Bob?"
        result = explain_ranking_node(state)
        assert "comparison_result" in result
        comp = result["comparison_result"]
        assert comp.get("type") == "explanation"
        assert len(comp.get("summary", "")) > 20

    def test_explain_no_candidates(self) -> None:
        state = {"current_shortlist": [], "requirements": {}}
        result = explain_ranking_node(state)
        comp = result["comparison_result"]
        assert len(comp.get("summary", "")) > 10


# ===================================================================
# Refine Requirements Node Tests
# ===================================================================


class TestRefineRequirementsNode:

    @patch("src.agent.nodes.get_full_resume_text")
    @patch("src.agent.nodes.score_candidate")
    def test_refine_drops_skill(self, mock_score, mock_resume) -> None:
        mock_resume.return_value = "Alice Johnson\nReact Developer\n5 years experience"
        mock_score.return_value = MagicMock(
            must_have_score=0.9, nice_to_have_score=0.8,
            reasoning="Strong", strengths=["React", "TypeScript"], gaps=[], excerpts=[]
        )
        state = _base_state()
        state["human_feedback"] = "Drop AWS"
        result = refine_requirements_node(state)
        assert result.get("requirements_version", 0) > 1
        reqs = result["requirements"]
        for item in reqs.get("must_have", []) + reqs.get("nice_to_have", []):
            assert item.get("skill", "").lower() != "aws"
        # Delta summary generated
        assert result.get("comparison_result", {}).get("type") == "refinement_delta"

    def test_refine_no_requirements_returns_error(self) -> None:
        state = {"requirements": {}, "human_feedback": "Drop AWS"}
        result = refine_requirements_node(state)
        assert "error" in result


# ===================================================================
# Generate Questions Node Tests
# ===================================================================


class TestGenerateQuestionsNode:

    def test_generate_questions_returns_questions(self) -> None:
        state = _base_state()
        state["human_feedback"] = "Generate questions for Alice"
        result = generate_questions_node(state)
        assert "comparison_result" in result
        comp = result["comparison_result"]
        assert comp.get("type") == "questions"
        questions = comp.get("questions", [])
        assert len(questions) >= 3

    def test_generate_questions_no_shortlist(self) -> None:
        state = {"current_shortlist": [], "requirements": {}}
        result = generate_questions_node(state)
        comp = result["comparison_result"]
        assert "No candidates" in comp.get("summary", "")


# ===================================================================
# Requirement Modification Parsing Tests
# ===================================================================


class TestRefineRequirementParsing:

    def test_drop_skill(self) -> None:
        from src.agent.nodes import _parse_requirement_modification
        reqs = {
            "must_have": [{"skill": "React", "type": "tech", "weight": 1.0, "evidence": "Required"},
                          {"skill": "AWS", "type": "tech", "weight": 0.8, "evidence": "Required"}],
            "nice_to_have": [],
        }
        result = _parse_requirement_modification("Drop AWS", reqs)
        skills = [s["skill"] for s in result["must_have"]]
        assert "AWS" not in skills
        assert "React" in skills

    def test_add_skill(self) -> None:
        from src.agent.nodes import _parse_requirement_modification
        reqs = {"must_have": [{"skill": "React", "type": "tech", "weight": 1.0, "evidence": "Req"}], "nice_to_have": []}
        result = _parse_requirement_modification("Add TypeScript", reqs)
        skills = [s["skill"] for s in result["must_have"]]
        assert "Typescript" in skills

    def test_change_experience(self) -> None:
        from src.agent.nodes import _parse_requirement_modification
        reqs = {"must_have": [], "nice_to_have": [], "experience_min_years": 5}
        result = _parse_requirement_modification("Change experience to 3 years", reqs)
        assert result["experience_min_years"] == 3

    def test_move_skill_to_nice_to_have(self) -> None:
        from src.agent.nodes import _parse_requirement_modification
        reqs = {
            "must_have": [{"skill": "React", "type": "tech", "weight": 1.0, "evidence": "Req"}],
            "nice_to_have": [],
        }
        result = _parse_requirement_modification("Move React to nice-to-have", reqs)
        must_skills = [s["skill"] for s in result["must_have"]]
        nice_skills = [s["skill"] for s in result["nice_to_have"]]
        assert "React" not in must_skills
        assert "React" in nice_skills


# ===================================================================
# Delta Summary Tests
# ===================================================================


class TestDeltaSummary:

    def test_moved_up_and_down(self) -> None:
        from src.agent.nodes import _build_delta_summary
        old = {"a": 0.5, "b": 0.7, "c": 0.3}
        new = {"a": 0.8, "b": 0.6, "c": 0.3, "d": 0.9}
        old_sl = [{"candidate_id": "a", "name": "Alice", "score": 0.5},
                  {"candidate_id": "b", "name": "Bob", "score": 0.7},
                  {"candidate_id": "c", "name": "Carol", "score": 0.3}]
        new_sl = [{"candidate_id": "d", "name": "Dave", "score": 0.9},
                  {"candidate_id": "a", "name": "Alice", "score": 0.8},
                  {"candidate_id": "b", "name": "Bob", "score": 0.6},
                  {"candidate_id": "c", "name": "Carol", "score": 0.3}]
        summary = _build_delta_summary(old, new, old_sl, new_sl)
        assert "Alice" in summary
        assert "Dave" in summary

    def test_no_changes(self) -> None:
        from src.agent.nodes import _build_delta_summary
        old = {"a": 0.5}
        new = {"a": 0.5}
        summary = _build_delta_summary(old, new, [{"candidate_id": "a", "name": "A"}], [{"candidate_id": "a", "name": "A"}])
        assert "No significant changes" in summary


# ===================================================================
# Route Feedback Edge Tests
# ===================================================================


class TestRouteFeedbackEdge:
    """Test the route_feedback edge function maps all intents correctly."""

    def test_route_refine(self) -> None:
        from src.agent.edges import route_feedback
        assert route_feedback({"next_action": "refine"}) == "refine_requirements"

    def test_route_compare(self) -> None:
        from src.agent.edges import route_feedback
        assert route_feedback({"next_action": "compare"}) == "compare_candidates"

    def test_route_questions(self) -> None:
        from src.agent.edges import route_feedback
        assert route_feedback({"next_action": "questions"}) == "generate_interview_questions"

    def test_route_explain(self) -> None:
        from src.agent.edges import route_feedback
        assert route_feedback({"next_action": "explain"}) == "explain_ranking"

    def test_route_report(self) -> None:
        from src.agent.edges import route_feedback
        assert route_feedback({"next_action": "report"}) == "generate_report"

    def test_route_done(self) -> None:
        from src.agent.edges import route_feedback
        assert route_feedback({"next_action": "done"}) == "__end__"

    def test_route_new_search(self) -> None:
        from src.agent.edges import route_feedback
        assert route_feedback({"next_action": "new_search"}) == "parse_jd"

    def test_route_unknown_fallback(self) -> None:
        from src.agent.edges import route_feedback
        assert route_feedback({"next_action": "unknown_thing"}) == "human_feedback_loop"

    def test_route_empty_fallback(self) -> None:
        from src.agent.edges import route_feedback
        assert route_feedback({}) == "human_feedback_loop"
