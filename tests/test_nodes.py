"""
Agentic Profile Matching — Phase 4 Node Unit Tests.

Tests each node function in isolation with mocked dependencies.
No LLM or RAG required — all external calls are mocked.

Run:
    pytest tests/test_nodes.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.nodes import (
    _build_search_query,
    extract_requirements_node,
    generate_report_node,
    parse_jd,
    rank_candidates_node,
    search_resumes_node,
)
from src.agent.state import AgentState
from src.scoring.ranker import (
    compute_hire_recommendation,
    filter_by_threshold,
    generate_improvement_suggestions,
    rank_candidates,
    shortlist,
)
from src.scoring.scorer import (
    _clamp,
    _keyword_fallback_score,
    compute_composite_score,
    score_candidate,
)
from src.reports.match_report import generate_match_report, _score_status, _recommendation_label

SAMPLE_JD_PATH = Path(__file__).parent / "fixtures" / "sample_jd.txt"

# Reusable test fixtures
SAMPLE_REQUIREMENTS = {
    "raw_jd": "Senior React Developer",
    "must_have": [
        {"skill": "React", "type": "tech", "weight": 1.0, "evidence": "Expert-level React"},
        {"skill": "TypeScript", "type": "tech", "weight": 0.9, "evidence": "Proficiency in TypeScript"},
        {"skill": "CSS", "type": "tech", "weight": 0.8, "evidence": "Modern CSS"},
    ],
    "nice_to_have": [
        {"skill": "Next.js", "type": "tech", "weight": 0.5, "evidence": "Experience with Next.js"},
        {"skill": "AWS", "type": "tech", "weight": 0.4, "evidence": "Familiarity with AWS"},
    ],
    "experience_min_years": 5,
    "education_level": "BS",
    "domain_keywords": ["frontend", "web"],
}

SAMPLE_RESUME = """Alice Johnson
Senior Frontend Developer

Experience:
- Senior React Developer at TechCorp (2021-2024): Led React migration for 200+ component SPA.
- Built responsive UIs with TypeScript and Tailwind CSS.
- 5 years of frontend development experience.
- Worked with REST APIs and GraphQL.

Education:
BS in Computer Science, State University

Skills: React, TypeScript, JavaScript, CSS, HTML, Tailwind CSS, REST APIs, GraphQL
"""

SAMPLE_CANDIDATE_MATCH = {
    "candidate_id": "alice_test_123",
    "name": "Alice Johnson",
    "score": 0.85,
    "must_have_score": 0.93,
    "nice_to_have_score": 0.65,
    "reasoning": "Strong match on React and TypeScript with 5 years experience.",
    "strengths": ["React", "TypeScript", "CSS", "5 years experience"],
    "gaps": ["No AWS experience"],
    "resume_excerpts": ["Led React migration for 200+ component SPA"],
    "hire_recommendation": "hire",
    "improvement_suggestions": ["Gain AWS certification"],
}


# ===================================================================
# 1. Scorer Tests
# ===================================================================


class TestScorer:
    """Test scoring functions (no LLM needed — tests fallback path)."""

    def test_compute_composite_score_formula(self) -> None:
        assert compute_composite_score(1.0, 1.0) == 1.0
        assert compute_composite_score(0.0, 0.0) == 0.0
        assert compute_composite_score(0.5, 0.5) == 0.5
        # 0.7 * 0.8 + 0.3 * 0.6 = 0.56 + 0.18 = 0.74
        assert compute_composite_score(0.8, 0.6) == pytest.approx(0.74)

    def test_clamp_within_range(self) -> None:
        assert _clamp(0.5) == 0.5
        assert _clamp(0.0) == 0.0
        assert _clamp(1.0) == 1.0

    def test_clamp_out_of_range(self) -> None:
        assert _clamp(1.5) == 1.0
        assert _clamp(-0.3) == 0.0

    def test_keyword_fallback_returns_candidate_score(self) -> None:
        result = _keyword_fallback_score(SAMPLE_RESUME, SAMPLE_REQUIREMENTS)
        assert result.must_have_score > 0
        assert result.nice_to_have_score >= 0
        assert len(result.reasoning) > 10
        assert isinstance(result.strengths, list)
        assert isinstance(result.gaps, list)

    def test_keyword_fallback_finds_react(self) -> None:
        result = _keyword_fallback_score(SAMPLE_RESUME, SAMPLE_REQUIREMENTS)
        skill_names = [s.lower() for s in result.strengths]
        assert any("react" in s for s in skill_names)

    def test_keyword_fallback_finds_typecript(self) -> None:
        result = _keyword_fallback_score(SAMPLE_RESUME, SAMPLE_REQUIREMENTS)
        skill_names = [s.lower() for s in result.strengths]
        assert any("typescript" in s for s in skill_names)

    def test_keyword_fallback_empty_resume(self) -> None:
        result = score_candidate("", SAMPLE_REQUIREMENTS)
        assert result.must_have_score == 0.0
        assert result.nice_to_have_score == 0.0

    def test_keyword_fallback_no_requirements(self) -> None:
        result = _keyword_fallback_score(SAMPLE_RESUME, {"must_have": [], "nice_to_have": []})
        assert result.must_have_score == 0.0
        assert result.nice_to_have_score == 0.0

    def test_keyword_fallback_excerpts_capped(self) -> None:
        # Resume with many skills
        resume = "React React React TypeScript TypeScript CSS CSS AWS AWS Next.js Next.js Docker Docker"
        reqs = {
            "must_have": [
                {"skill": "React", "type": "tech", "weight": 1.0, "evidence": ""},
                {"skill": "TypeScript", "type": "tech", "weight": 1.0, "evidence": ""},
                {"skill": "CSS", "type": "tech", "weight": 1.0, "evidence": ""},
                {"skill": "AWS", "type": "tech", "weight": 1.0, "evidence": ""},
                {"skill": "Next.js", "type": "tech", "weight": 1.0, "evidence": ""},
            ],
            "nice_to_have": [
                {"skill": "Docker", "type": "tech", "weight": 0.5, "evidence": ""},
            ],
        }
        result = _keyword_fallback_score(resume, reqs)
        assert len(result.excerpts) <= 4


# ===================================================================
# 2. Ranker Tests
# ===================================================================


class TestRanker:
    def test_rank_sorts_descending(self) -> None:
        candidates = [
            {"candidate_id": "b", "score": 0.5},
            {"candidate_id": "a", "score": 0.9},
            {"candidate_id": "c", "score": 0.7},
        ]
        ranked = rank_candidates(candidates)
        scores = [c["score"] for c in ranked]
        assert scores == [0.9, 0.7, 0.5]

    def test_rank_handles_missing_score(self) -> None:
        candidates = [
            {"candidate_id": "a"},  # no score key
            {"candidate_id": "b", "score": 0.5},
        ]
        ranked = rank_candidates(candidates)
        assert ranked[0]["candidate_id"] == "b"

    def test_filter_by_threshold(self) -> None:
        candidates = [
            {"candidate_id": "a", "score": 0.8},
            {"candidate_id": "b", "score": 0.3},
            {"candidate_id": "c", "score": 0.1},
        ]
        filtered = filter_by_threshold(candidates, threshold=0.3)
        ids = [c["candidate_id"] for c in filtered]
        assert "a" in ids
        assert "b" in ids
        assert "c" not in ids

    def test_shortlist_limits_n(self) -> None:
        candidates = [{"candidate_id": str(i), "score": float(i)} for i in range(20)]
        # shortlist() takes top N from the head — sort first (descending)
        from src.scoring.ranker import rank_candidates
        sorted_candidates = rank_candidates(candidates)
        result = shortlist(sorted_candidates, n=5)
        assert len(result) == 5
        assert result[0]["candidate_id"] == "19"  # highest score first

    def test_shortlist_empty(self) -> None:
        assert shortlist([], n=10) == []

    def test_compute_hire_recommendation(self) -> None:
        assert compute_hire_recommendation(0.9) == "hire"
        assert compute_hire_recommendation(0.8) == "hire"
        assert compute_hire_recommendation(0.7) == "borderline"
        assert compute_hire_recommendation(0.5) == "borderline"
        assert compute_hire_recommendation(0.4) == "no_hire"
        assert compute_hire_recommendation(0.1) == "no_hire"

    def test_generate_improvement_suggestions(self) -> None:
        gaps = ["No TypeScript experience found", "No cloud experience (nice-to-have)"]
        reqs = {
            "must_have": [{"skill": "TypeScript", "type": "tech", "weight": 1.0}],
            "nice_to_have": [{"skill": "AWS", "type": "tech", "weight": 0.5}],
        }
        suggestions = generate_improvement_suggestions(gaps, reqs)
        assert len(suggestions) >= 1
        assert any("TypeScript" in s for s in suggestions)

    def test_generate_improvement_suggestions_empty_gaps(self) -> None:
        suggestions = generate_improvement_suggestions([], SAMPLE_REQUIREMENTS)
        assert len(suggestions) >= 1  # Default suggestion


# ===================================================================
# 3. Report Generation Tests
# ===================================================================


class TestMatchReport:
    def test_report_contains_header(self) -> None:
        report = generate_match_report(SAMPLE_CANDIDATE_MATCH, SAMPLE_REQUIREMENTS)
        assert "# Candidate Match Report: Alice Johnson" in report

    def test_report_contains_scores_section(self) -> None:
        report = generate_match_report(SAMPLE_CANDIDATE_MATCH, SAMPLE_REQUIREMENTS)
        assert "## Scores" in report
        assert "0.85" in report

    def test_report_contains_strengths(self) -> None:
        report = generate_match_report(SAMPLE_CANDIDATE_MATCH, SAMPLE_REQUIREMENTS)
        assert "## Strengths" in report
        assert "React" in report

    def test_report_contains_gaps(self) -> None:
        report = generate_match_report(SAMPLE_CANDIDATE_MATCH, SAMPLE_REQUIREMENTS)
        assert "## Gaps" in report

    def test_report_contains_recommendation(self) -> None:
        report = generate_match_report(SAMPLE_CANDIDATE_MATCH, SAMPLE_REQUIREMENTS)
        assert "STRONG HIRE" in report

    def test_report_contains_criteria_breakdown(self) -> None:
        report = generate_match_report(SAMPLE_CANDIDATE_MATCH, SAMPLE_REQUIREMENTS)
        assert "## Must-Have Criteria Breakdown" in report
        assert "## Nice-to-Have Criteria Breakdown" in report

    def test_report_with_minimal_candidate(self) -> None:
        minimal = {
            "candidate_id": "x",
            "name": "Test",
            "score": 0.5,
            "must_have_score": 0.5,
            "nice_to_have_score": 0.5,
            "reasoning": "Average match",
            "strengths": [],
            "gaps": [],
            "resume_excerpts": [],
            "hire_recommendation": "borderline",
            "improvement_suggestions": [],
        }
        report = generate_match_report(minimal, SAMPLE_REQUIREMENTS)
        assert "Test" in report
        assert "BORDERLINE" in report

    def test_score_status_labels(self) -> None:
        assert _score_status(0.9) == "STRONG MATCH"
        assert _score_status(0.7) == "GOOD MATCH"
        assert _score_status(0.5) == "PARTIAL"
        assert _score_status(0.3) == "WEAK"
        assert _score_status(0.1) == "NO MATCH"

    def test_recommendation_label(self) -> None:
        assert _recommendation_label("hire") == "STRONG HIRE"
        assert _recommendation_label("borderline") == "BORDERLINE"
        assert _recommendation_label("no_hire") == "NO HIRE"


# ===================================================================
# 4. Node Function Tests (mocked dependencies)
# ===================================================================


class TestParseJdNode:
    def test_valid_jd(self) -> None:
        state: AgentState = {"raw_jd": "Senior React Developer with 5+ years experience building web apps."}
        result = parse_jd(state)
        assert "error" not in result
        assert "Senior React" in result["raw_jd"]

    def test_empty_jd(self) -> None:
        state: AgentState = {"raw_jd": ""}
        result = parse_jd(state)
        assert "error" in result
        assert "too short" in result["error"].lower()

    def test_short_jd(self) -> None:
        state: AgentState = {"raw_jd": "Hi"}
        result = parse_jd(state)
        assert "error" in result

    def test_whitespace_only_jd(self) -> None:
        state: AgentState = {"raw_jd": "   \\n   \\n   "}
        result = parse_jd(state)
        assert "error" in result

    def test_strips_whitespace(self) -> None:
        state: AgentState = {"raw_jd": "  Senior React Developer  "}
        result = parse_jd(state)
        assert result["raw_jd"] == "Senior React Developer"


class TestExtractRequirementsNode:
    @patch("src.agent.nodes.extract_requirements")
    def test_calls_tool_and_stores_result(self, mock_extract: MagicMock) -> None:
        mock_extract.invoke.return_value = {
            "must_have": [{"skill": "React", "type": "tech", "weight": 1.0, "evidence": "..."}],
            "nice_to_have": [{"skill": "AWS", "type": "tech", "weight": 0.5, "evidence": "..."}],
            "experience_min_years": 5,
            "education_level": "BS",
            "domain_keywords": ["frontend"],
        }
        state: AgentState = {"raw_jd": "Senior React Developer needed"}
        result = extract_requirements_node(state)
        assert "requirements" in result
        assert len(result["requirements"]["must_have"]) == 1
        assert result["requirements_version"] == 1
        mock_extract.invoke.assert_called_once()

    @patch("src.agent.nodes.extract_requirements")
    def test_handles_extraction_failure(self, mock_extract: MagicMock) -> None:
        mock_extract.invoke.side_effect = RuntimeError("LLM unavailable")
        state: AgentState = {"raw_jd": "Some JD text"}
        result = extract_requirements_node(state)
        assert "error" in result
        assert result["requirements"]["must_have"] == []

    def test_no_raw_jd(self) -> None:
        state: AgentState = {}
        result = extract_requirements_node(state)
        assert "error" in result


class TestSearchResumesNode:
    @patch("src.agent.nodes.rag_search")
    def test_stores_candidate_ids(self, mock_rag: MagicMock) -> None:
        mock_rag.invoke.return_value = [
            {"candidate_id": "alice_123", "name": "Alice", "score": 0.1, "excerpt": "..."},
            {"candidate_id": "bob_456", "name": "Bob", "score": 0.2, "excerpt": "..."},
        ]
        state: AgentState = {
            "requirements": SAMPLE_REQUIREMENTS,
        }
        result = search_resumes_node(state)
        assert result["all_candidate_ids"] == ["alice_123", "bob_456"]

    @patch("src.agent.nodes.rag_search")
    def test_empty_results(self, mock_rag: MagicMock) -> None:
        mock_rag.invoke.return_value = []
        state: AgentState = {"requirements": SAMPLE_REQUIREMENTS}
        result = search_resumes_node(state)
        assert result["all_candidate_ids"] == []

    def test_no_requirements(self) -> None:
        state: AgentState = {}
        result = search_resumes_node(state)
        assert "error" in result
        assert result["all_candidate_ids"] == []


class TestRankCandidatesNode:
    @patch("src.screening.pipeline.run_screening_pipeline")
    def test_scores_and_ranks(self, mock_pipeline: MagicMock) -> None:
        """rank_candidates_node delegates to the 3-round screening pipeline.

        Phase 6 changed rank_candidates_node from single-pass scoring to
        delegating to run_screening_pipeline. This test verifies the
        delegation: when the pipeline returns a shortlist, the node
        forwards it unchanged.
        """
        # Pipeline returns a known shortlist with all expected fields
        mock_pipeline.return_value = {
            "current_shortlist": [
                {
                    "candidate_id": "alice_123",
                    "name": "Alice Johnson",
                    "score": 0.84,
                    "must_have_score": 0.9,
                    "nice_to_have_score": 0.6,
                    "reasoning": "Strong match",
                    "strengths": ["React", "TypeScript"],
                    "gaps": ["No AWS"],
                    "hire_recommendation": "hire",
                    "improvement_suggestions": [],
                },
                {
                    "candidate_id": "bob_456",
                    "name": "Bob Smith",
                    "score": 0.66,
                    "must_have_score": 0.7,
                    "nice_to_have_score": 0.5,
                    "reasoning": "Partial match",
                    "strengths": ["React"],
                    "gaps": ["No TypeScript"],
                    "hire_recommendation": "borderline",
                    "improvement_suggestions": ["Learn TypeScript"],
                },
            ],
            "screening_rounds": [
                {"round_number": 1, "round_type": "initial", "candidates_evaluated": 100, "shortlisted_ids": ["alice_123", "bob_456"], "eliminated_ids": [], "notes": "R1"},
                {"round_number": 2, "round_type": "deep_analysis", "candidates_evaluated": 10, "shortlisted_ids": ["alice_123", "bob_456"], "eliminated_ids": [], "notes": "R2"},
                {"round_number": 3, "round_type": "final", "candidates_evaluated": 2, "shortlisted_ids": ["alice_123", "bob_456"], "eliminated_ids": [], "notes": "R3"},
            ],
            "generated_reports": {"alice_123": "# Alice report", "bob_456": "# Bob report"},
            "current_round": 3,
        }
        state: AgentState = {
            "all_candidate_ids": ["alice_123", "bob_456"],
            "requirements": SAMPLE_REQUIREMENTS,
        }
        result = rank_candidates_node(state)

        # Pipeline was called with the full state
        mock_pipeline.assert_called_once()

        # Node forwarded the pipeline's result
        assert len(result["current_shortlist"]) == 2
        assert result["current_round"] == 3
        assert len(result["screening_rounds"]) == 3
        # Verify composite score computed
        for c in result["current_shortlist"]:
            assert "score" in c
            assert "hire_recommendation" in c

    def test_empty_candidate_ids(self) -> None:
        state: AgentState = {"all_candidate_ids": [], "requirements": SAMPLE_REQUIREMENTS}
        result = rank_candidates_node(state)
        assert result["current_shortlist"] == []

    @patch("src.screening.pipeline.run_screening_pipeline")
    def test_skips_missing_resumes_via_pipeline(self, mock_pipeline: MagicMock) -> None:
        """When the pipeline produces no shortlist (e.g., all resumes missing),
        rank_candidates_node falls back to single-pass scoring, which also
        produces an empty shortlist if no resume can be retrieved.
        """
        # Pipeline returns empty shortlist (e.g., all candidates filtered out)
        mock_pipeline.return_value = {
            "current_shortlist": [],
            "screening_rounds": [
                {"round_number": 1, "round_type": "initial",
                 "candidates_evaluated": 0, "shortlisted_ids": [],
                 "eliminated_ids": [], "notes": "R1: no candidates passed"},
            ],
            "current_round": 1,
            "error": "No candidates survived Round 1 screening.",
        }
        state: AgentState = {
            "all_candidate_ids": ["ghost_123"],
            "requirements": SAMPLE_REQUIREMENTS,
        }
        result = rank_candidates_node(state)
        # Empty shortlist → fallback path → still empty (no resume for ghost)
        assert len(result["current_shortlist"]) == 0


class TestGenerateReportNode:
    def test_generates_reports_for_shortlist(self) -> None:
        state: AgentState = {
            "current_shortlist": [SAMPLE_CANDIDATE_MATCH],
            "requirements": SAMPLE_REQUIREMENTS,
        }
        result = generate_report_node(state)
        assert "alice_test_123" in result["generated_reports"]
        assert result["awaiting_human_feedback"] is True

    def test_empty_shortlist(self) -> None:
        state: AgentState = {"current_shortlist": [], "requirements": {}}
        result = generate_report_node(state)
        assert result["generated_reports"] == {}
        assert result["awaiting_human_feedback"] is True

    def test_report_content_valid_markdown(self) -> None:
        state: AgentState = {
            "current_shortlist": [SAMPLE_CANDIDATE_MATCH],
            "requirements": SAMPLE_REQUIREMENTS,
        }
        result = generate_report_node(state)
        report = result["generated_reports"]["alice_test_123"]
        assert "# Candidate Match Report" in report
        assert "## Scores" in report


# ===================================================================
# 5. Query Builder Tests
# ===================================================================


class TestQueryBuilder:
    def test_basic_query(self) -> None:
        reqs = {
            "must_have": [{"skill": "React"}, {"skill": "TypeScript"}],
            "nice_to_have": [{"skill": "AWS"}],
            "domain_keywords": ["frontend"],
        }
        query = _build_search_query(reqs)
        assert "React" in query
        assert "TypeScript" in query
        assert "AWS" in query
        assert "frontend" in query

    def test_empty_requirements_fallback(self) -> None:
        query = _build_search_query({})
        assert "developer" in query

    def test_with_experience_and_education(self) -> None:
        reqs = {
            "must_have": [],
            "nice_to_have": [],
            "experience_min_years": 5,
            "education_level": "BS",
        }
        query = _build_search_query(reqs)
        assert "5+ years experience" in query
        assert "BS degree" in query


# ===================================================================
# 6. Edge Routing Tests
# ===================================================================


class TestEdgeRouting:
    def test_route_after_parse_jd_normal(self) -> None:
        from src.agent.edges import route_after_parse_jd
        result = route_after_parse_jd({"raw_jd": "Valid JD"})
        assert result == "extract_requirements"

    def test_route_after_parse_jd_error(self) -> None:
        from src.agent.edges import route_after_parse_jd
        result = route_after_parse_jd({"error": "JD is too short"})
        assert result == "__end__"

    def test_route_after_requirements(self) -> None:
        from src.agent.edges import route_after_requirements
        assert route_after_requirements({}) == "search_resumes"

    def test_route_after_search(self) -> None:
        from src.agent.edges import route_after_search
        assert route_after_search({}) == "rank_candidates"

    def test_route_after_ranking(self) -> None:
        from src.agent.edges import route_after_ranking
        assert route_after_ranking({}) == "generate_report"


# ===================================================================
# 7. Graph Construction Tests
# ===================================================================


class TestGraphConstruction:
    def test_create_graph_returns_compiled(self) -> None:
        from src.agent.graph import create_graph
        graph = create_graph()
        assert graph is not None
        # Compiled graph should have a callable invoke
        assert hasattr(graph, 'invoke')

    def test_graph_has_all_nodes(self) -> None:
        from src.agent.graph import create_graph
        graph = create_graph()
        # The compiled graph's nodes attribute
        node_names = set(graph.get_graph().nodes.keys())
        # Filter out START/END
        node_names.discard("__start__")
        node_names.discard("__end__")
        expected = {"parse_jd", "extract_requirements", "search_resumes", "rank_candidates", "generate_report"}
        assert expected.issubset(node_names), f"Missing nodes: {expected - node_names}"
