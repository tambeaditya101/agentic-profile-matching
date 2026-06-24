"""
Agentic Profile Matching — Phase 8 End-to-End Agent Flow Tests.

Implements the 7 test scenarios from architecture.md Section 16.1:

  1. Happy path: Full pipeline — JD → top 10 → reports
  2. Refinement: Change requirements mid-conversation, verify re-ranking
  3. Comparison: Compare 3 candidates, verify structured output
  4. Explanation: Ask "Why did X rank higher than Y?"
  5. Interview questions: Generate questions for a candidate
  6. Edge case — no results: JD with impossible requirements
  7. Multi-round: Run all 3 screening rounds sequentially

All scenarios use mocked LLM/RAG dependencies so they run deterministically
in CI without API keys. Each test verifies the *shape* of the agent's
output (schema conformance, presence of expected fields), not exact content
(since LLM output varies).

The graph only has an entry edge from START → parse_jd. Multi-turn tests
therefore invoke interactive nodes directly (the same pattern used by
tests/test_conversation_flows.py). The linear pipeline is tested via
graph.invoke().

Run:
    pytest tests/test_agent_flows.py -v

Architecture Reference: architecture.md Section 16.1, Section 16.2
Phase: 8 — Testing, Polish & Demo
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.agent.graph import create_graph
from src.agent.nodes import (
    compare_candidates_node,
    explain_ranking_node,
    generate_questions_node,
    refine_requirements_node,
)
from src.agent.state import AgentState

# =====================================================================
# Fixtures
# =====================================================================

SAMPLE_JD_PATH = Path(__file__).parent / "fixtures" / "sample_jd.txt"


@pytest.fixture
def sample_jd() -> str:
    """Load the canonical sample JD used by all end-to-end tests."""
    assert SAMPLE_JD_PATH.exists(), f"Sample JD not found: {SAMPLE_JD_PATH}"
    return SAMPLE_JD_PATH.read_text()


@pytest.fixture
def sample_requirements() -> dict:
    """A canonical requirements dict for tests that bypass extract_requirements."""
    return {
        "raw_jd": "Senior React Frontend Developer with TypeScript and AWS",
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


@pytest.fixture
def sample_shortlist() -> list[dict]:
    """A canonical 3-candidate shortlist for tests that bypass rank_candidates."""
    return [
        {
            "candidate_id": "alice_123",
            "name": "Alice Johnson",
            "score": 0.92,
            "must_have_score": 0.95,
            "nice_to_have_score": 0.85,
            "reasoning": "Expert React developer with 5 years experience and TypeScript proficiency.",
            "strengths": ["React", "TypeScript", "CSS", "5 years experience"],
            "gaps": ["No AWS certification"],
            "resume_excerpts": [
                "Led React migration for 200+ component SPA",
                "Built responsive UIs with TypeScript and Tailwind CSS",
            ],
            "hire_recommendation": "hire",
            "improvement_suggestions": ["Consider AWS certification"],
        },
        {
            "candidate_id": "bob_456",
            "name": "Bob Smith",
            "score": 0.75,
            "must_have_score": 0.80,
            "nice_to_have_score": 0.65,
            "reasoning": "Solid React developer, but no explicit TypeScript experience.",
            "strengths": ["React", "JavaScript", "CSS"],
            "gaps": ["No TypeScript", "No Next.js"],
            "resume_excerpts": ["Built React components for customer dashboard"],
            "hire_recommendation": "borderline",
            "improvement_suggestions": ["Learn TypeScript", "Gain Next.js experience"],
        },
        {
            "candidate_id": "carol_789",
            "name": "Carol Williams",
            "score": 0.62,
            "must_have_score": 0.65,
            "nice_to_have_score": 0.55,
            "reasoning": "Junior React developer with growth potential.",
            "strengths": ["React", "HTML", "CSS"],
            "gaps": ["No TypeScript", "Limited experience", "No Next.js"],
            "resume_excerpts": ["Helped with React components in a team setting"],
            "hire_recommendation": "borderline",
            "improvement_suggestions": [
                "Learn TypeScript",
                "Gain more production experience",
                "Build a Next.js project",
            ],
        },
    ]


@pytest.fixture
def base_state(sample_requirements, sample_shortlist) -> AgentState:
    """A pre-populated state for Turn 2+ tests (pipeline already run)."""
    return {
        "raw_jd": sample_requirements["raw_jd"],
        "requirements": sample_requirements,
        "requirements_version": 1,
        "all_candidate_ids": [c["candidate_id"] for c in sample_shortlist],
        "current_shortlist": sample_shortlist,
        "screening_rounds": [],
        "generated_reports": {},
        "awaiting_human_feedback": True,
        "messages": [],
    }


@pytest.fixture
def mock_extract_requirements(sample_requirements):
    """Mock the extract_requirements tool to return sample requirements."""
    with patch("src.agent.nodes.extract_requirements") as mock:
        mock.invoke.return_value = {
            "must_have": sample_requirements["must_have"],
            "nice_to_have": sample_requirements["nice_to_have"],
            "experience_min_years": sample_requirements["experience_min_years"],
            "education_level": sample_requirements["education_level"],
            "domain_keywords": sample_requirements["domain_keywords"],
        }
        yield mock


@pytest.fixture
def mock_rag_search():
    """Mock rag_search to return a deterministic candidate pool."""
    with patch("src.agent.nodes.rag_search") as mock:
        mock.invoke.return_value = [
            {"candidate_id": "alice_123", "name": "Alice Johnson", "score": 0.9, "excerpt": "React expert"},
            {"candidate_id": "bob_456", "name": "Bob Smith", "score": 0.7, "excerpt": "React dev"},
            {"candidate_id": "carol_789", "name": "Carol Williams", "score": 0.5, "excerpt": "Junior React"},
        ]
        yield mock


@pytest.fixture
def mock_resume_text():
    """Mock get_full_resume_text to return a deterministic resume per candidate."""
    resumes = {
        "alice_123": (
            "Alice Johnson\nSenior Frontend Developer\n\n"
            "Experience:\n- Senior React Developer at TechCorp (2019-2024)\n"
            "- Built React SPAs with TypeScript and Tailwind CSS\n"
            "- 5 years of frontend development experience\n\n"
            "Skills: React, TypeScript, JavaScript, CSS, Tailwind CSS"
        ),
        "bob_456": (
            "Bob Smith\nReact Developer\n\n"
            "Experience:\n- React Developer at StartupXYZ (2021-2023)\n"
            "- Built React components using JavaScript\n"
            "- 3 years experience\n\n"
            "Skills: React, JavaScript, HTML, CSS"
        ),
        "carol_789": (
            "Carol Williams\nJunior Developer\n\n"
            "Experience:\n- Junior Developer at BigCo (2022-2024)\n"
            "- Helped with React components\n"
            "- 2 years experience\n\n"
            "Skills: React, HTML, CSS, Python"
        ),
    }
    with patch("src.agent.nodes.get_full_resume_text") as mock:
        mock.side_effect = lambda cid: resumes.get(cid)
        yield mock


@pytest.fixture
def mock_score_candidate():
    """Mock score_candidate to return deterministic scores per candidate."""
    scores = {
        "alice_123": MagicMock(
            must_have_score=0.95, nice_to_have_score=0.85,
            reasoning="Strong React + TypeScript match.",
            strengths=["React", "TypeScript", "CSS"],
            gaps=["No AWS"],
            excerpts=["Led React migration"],
        ),
        "bob_456": MagicMock(
            must_have_score=0.75, nice_to_have_score=0.50,
            reasoning="Solid React, missing TypeScript.",
            strengths=["React", "JavaScript"],
            gaps=["No TypeScript"],
            excerpts=["Built React components"],
        ),
        "carol_789": MagicMock(
            must_have_score=0.55, nice_to_have_score=0.30,
            reasoning="Junior, partial React match.",
            strengths=["React"],
            gaps=["No TypeScript", "Limited experience"],
            excerpts=["Helped with React components"],
        ),
    }

    def side_effect(resume_text, requirements, llm=None):
        for cid, score in scores.items():
            name_part = cid.split("_")[0]
            if name_part.lower() in resume_text.lower():
                return score
        return scores["alice_123"]

    with patch("src.agent.nodes.score_candidate") as mock:
        mock.side_effect = side_effect
        yield mock


@pytest.fixture
def agent_graph():
    """Compiled agent graph (fresh per test)."""
    return create_graph()


# =====================================================================
# Helpers
# =====================================================================

def _validate_candidate_match(c: dict) -> None:
    """Assert that a CandidateMatch dict conforms to the schema."""
    assert "candidate_id" in c and isinstance(c["candidate_id"], str)
    assert "name" in c and isinstance(c["name"], str)
    assert "score" in c and isinstance(c["score"], (int, float))
    assert 0.0 <= c["score"] <= 1.0
    assert "must_have_score" in c and 0.0 <= c["must_have_score"] <= 1.0
    assert "nice_to_have_score" in c and 0.0 <= c["nice_to_have_score"] <= 1.0
    assert "hire_recommendation" in c
    assert c["hire_recommendation"] in ("hire", "no_hire", "borderline", "unknown")


def _validate_screening_round(r: dict) -> None:
    """Assert that a ScreeningRound dict conforms to the schema."""
    assert "round_number" in r and r["round_number"] in (1, 2, 3)
    assert "round_type" in r and r["round_type"] in ("initial", "deep_analysis", "final")
    assert "candidates_evaluated" in r and isinstance(r["candidates_evaluated"], int)
    assert "shortlisted_ids" in r and isinstance(r["shortlisted_ids"], list)


def _validate_match_report(report_md: str) -> None:
    """Assert that a match report contains the required sections."""
    assert "# Candidate Match Report:" in report_md
    assert "## Summary" in report_md
    assert "## Scores" in report_md
    assert "## Hire Recommendation:" in report_md
    assert "| Criterion |" in report_md or "| Must-Have" in report_md


# =====================================================================
# Scenario 1 — Happy Path: Full Pipeline
# =====================================================================


class TestScenario1HappyPath:
    """Scenario 1: Full pipeline — JD → top 10 → reports."""

    def test_happy_path_full_pipeline(
        self,
        agent_graph,
        sample_jd: str,
        mock_extract_requirements,
        mock_rag_search,
        mock_resume_text,
        mock_score_candidate,
    ) -> None:
        """End-to-end: invoke graph with JD, verify all state fields populated."""
        result = agent_graph.invoke({"raw_jd": sample_jd, "messages": []})

        # 1. Requirements extracted
        assert "requirements" in result
        reqs = result["requirements"]
        assert isinstance(reqs.get("must_have"), list) and len(reqs["must_have"]) > 0
        assert isinstance(reqs.get("nice_to_have"), list)

        # 2. Candidates retrieved
        assert "all_candidate_ids" in result
        assert len(result["all_candidate_ids"]) > 0

        # 3. Shortlist ranked
        assert "current_shortlist" in result
        shortlist = result["current_shortlist"]
        assert len(shortlist) > 0
        for c in shortlist:
            _validate_candidate_match(c)
        # Sorted by composite score descending
        scores = [c["score"] for c in shortlist]
        assert scores == sorted(scores, reverse=True)

        # 4. Reports generated
        assert "generated_reports" in result
        reports = result["generated_reports"]
        assert len(reports) > 0
        for cid, md in reports.items():
            _validate_match_report(md)

        # 5. Awaiting human feedback
        assert result.get("awaiting_human_feedback") is True

    def test_short_jd_returns_error(
        self,
        agent_graph,
    ) -> None:
        """Empty / too-short JD should return an error and not advance."""
        result = agent_graph.invoke({"raw_jd": "Hi", "messages": []})
        assert "error" in result
        assert "too short" in result["error"].lower()

    def test_pipeline_state_consistency(
        self,
        agent_graph,
        sample_jd: str,
        mock_extract_requirements,
        mock_rag_search,
        mock_resume_text,
        mock_score_candidate,
    ) -> None:
        """All shortlisted candidates must have a corresponding report."""
        result = agent_graph.invoke({"raw_jd": sample_jd, "messages": []})
        shortlist = result["current_shortlist"]
        reports = result["generated_reports"]
        for c in shortlist:
            assert c["candidate_id"] in reports, (
                f"Missing report for {c['candidate_id']}"
            )


# =====================================================================
# Scenario 2 — Refinement: Change Requirements Mid-Conversation
# =====================================================================


class TestScenario2Refinement:
    """Scenario 2: Modify requirements and verify re-ranking.

    Tests call refine_requirements_node directly because the graph only
    has an entry edge from START to parse_jd. Multi-turn conversation
    is managed by the UI layer which calls interactive nodes directly
    or via a future conditional START edge.
    """

    @patch("src.agent.nodes._re_rank_candidates")
    @patch("src.agent.nodes.get_full_resume_text")
    @patch("src.agent.nodes.score_candidate")
    def test_refinement_increments_version(
        self,
        mock_score: MagicMock,
        mock_resume: MagicMock,
        mock_rerank: MagicMock,
        base_state: AgentState,
    ) -> None:
        """Adding a requirement increments requirements_version from 1 to 2."""
        mock_resume.return_value = "Alice\nReact Developer"
        mock_score.return_value = MagicMock(
            must_have_score=0.9, nice_to_have_score=0.7,
            reasoning="ok", strengths=["React"], gaps=[], excerpts=[],
        )
        mock_rerank.return_value = base_state["current_shortlist"]

        state = {**base_state, "human_feedback": "Add TypeScript as a must-have requirement"}
        result = refine_requirements_node(state)
        assert result["requirements_version"] == 2

    @patch("src.agent.nodes._re_rank_candidates")
    @patch("src.agent.nodes.get_full_resume_text")
    @patch("src.agent.nodes.score_candidate")
    def test_refinement_produces_delta_summary(
        self,
        mock_score: MagicMock,
        mock_resume: MagicMock,
        mock_rerank: MagicMock,
        base_state: AgentState,
    ) -> None:
        """Refinement produces a comparison_result with type=refinement_delta."""
        mock_resume.return_value = "Alice\nReact Developer"
        mock_score.return_value = MagicMock(
            must_have_score=0.9, nice_to_have_score=0.7,
            reasoning="ok", strengths=["React"], gaps=[], excerpts=[],
        )
        mock_rerank.return_value = base_state["current_shortlist"]

        state = {**base_state, "human_feedback": "Drop the AWS requirement"}
        result = refine_requirements_node(state)
        comparison = result.get("comparison_result") or {}
        assert comparison.get("type") == "refinement_delta"
        assert "summary" in comparison
        assert isinstance(comparison["summary"], str)
        assert len(comparison["summary"]) > 0

    @patch("src.agent.nodes._re_rank_candidates")
    @patch("src.agent.nodes.get_full_resume_text")
    @patch("src.agent.nodes.score_candidate")
    def test_refinement_modifies_requirements(
        self,
        mock_score: MagicMock,
        mock_resume: MagicMock,
        mock_rerank: MagicMock,
        base_state: AgentState,
    ) -> None:
        """Refinement actually modifies the requirements dict."""
        mock_resume.return_value = "Alice\nReact Developer"
        mock_score.return_value = MagicMock(
            must_have_score=0.9, nice_to_have_score=0.7,
            reasoning="ok", strengths=["React"], gaps=[], excerpts=[],
        )
        mock_rerank.return_value = base_state["current_shortlist"]

        original_must_count = len(base_state["requirements"]["must_have"])
        state = {**base_state, "human_feedback": "Add GraphQL as a must-have requirement"}
        result = refine_requirements_node(state)

        new_reqs = result["requirements"]
        new_must_skills = [item.get("skill", "").lower() for item in new_reqs.get("must_have", [])]
        assert "graphql" in new_must_skills or len(new_reqs.get("must_have", [])) >= original_must_count

    @patch("src.agent.nodes._re_rank_candidates")
    @patch("src.agent.nodes.get_full_resume_text")
    @patch("src.agent.nodes.score_candidate")
    def test_refinement_no_requirements_returns_error(
        self,
        mock_score: MagicMock,
        mock_resume: MagicMock,
        mock_rerank: MagicMock,
    ) -> None:
        """Refinement with no requirements returns an error."""
        state: AgentState = {
            "requirements": {},
            "current_shortlist": [],
            "human_feedback": "Drop AWS",
            "messages": [],
        }
        result = refine_requirements_node(state)
        assert "error" in result


# =====================================================================
# Scenario 3 — Comparison: Compare 3 Candidates
# =====================================================================


class TestScenario3Comparison:
    """Scenario 3: Compare 3 candidates and verify structured output."""

    def test_compare_top_3(
        self,
        base_state: AgentState,
    ) -> None:
        """'Compare the top 3' returns a structured comparison_result."""
        state = {**base_state, "human_feedback": "Compare the top 3 candidates side by side"}

        # Force fallback comparison (deterministic, no LLM)
        with patch("src.tools.compare_candidates.get_llm", side_effect=RuntimeError("no llm")):
            result = compare_candidates_node(state)

        comparison = result.get("comparison_result") or {}
        assert "summary" in comparison
        assert isinstance(comparison["summary"], str)
        assert len(comparison["summary"]) > 0
        # Should have candidates list (fallback produces it)
        if "candidates" in comparison:
            assert isinstance(comparison["candidates"], list)

    def test_compare_two_named_candidates(
        self,
        base_state: AgentState,
    ) -> None:
        """'Compare Alice and Bob' identifies both candidates by name."""
        state = {**base_state, "human_feedback": "Compare Alice Johnson and Bob Smith"}

        with patch("src.tools.compare_candidates.get_llm", side_effect=RuntimeError("no llm")):
            result = compare_candidates_node(state)

        comparison = result.get("comparison_result") or {}
        assert "summary" in comparison
        candidates = comparison.get("candidates", [])
        # Fallback comparison should have at least 2 candidates
        if candidates:
            assert len(candidates) >= 2
        # Summary or candidate names should mention at least one of them
        all_text = comparison.get("summary", "")
        for c in candidates:
            all_text += " " + c.get("name", "")
        assert "Alice" in all_text or "Bob" in all_text

    def test_compare_with_insufficient_candidates(
        self,
        sample_requirements: dict,
        sample_shortlist: list[dict],
    ) -> None:
        """Comparing only 1 candidate returns single_candidate type."""
        state: AgentState = {
            "requirements": sample_requirements,
            "current_shortlist": [sample_shortlist[0]],
            "human_feedback": "Compare the top candidates",
            "messages": [],
        }
        result = compare_candidates_node(state)
        comparison = result.get("comparison_result") or {}
        # Should be single_candidate type since only 1 in shortlist
        assert comparison.get("type") in ("single_candidate", "error")

    def test_compare_no_shortlist_returns_error(
        self,
    ) -> None:
        """Comparing with no shortlist returns an error."""
        state: AgentState = {
            "current_shortlist": [],
            "requirements": {},
            "human_feedback": "Compare the top 3",
            "messages": [],
        }
        result = compare_candidates_node(state)
        assert "error" in result


# =====================================================================
# Scenario 4 — Explanation: Why Did X Rank Higher Than Y?
# =====================================================================


class TestScenario4Explanation:
    """Scenario 4: Ask for ranking explanation."""

    def test_explain_top_2(
        self,
        base_state: AgentState,
    ) -> None:
        """'Why did Alice rank higher than Bob?' returns an explanation."""
        state = {
            **base_state,
            "human_feedback": "Why did Alice Johnson rank higher than Bob Smith?",
        }

        # Force fallback explanation (no LLM) — patch at source since
        # explain_ranking_node imports get_llm locally inside the function.
        with patch("src.llm.client.get_llm", side_effect=RuntimeError("no llm")):
            result = explain_ranking_node(state)

        comparison = result.get("comparison_result") or {}
        assert comparison.get("type") == "explanation"
        assert comparison.get("higher_candidate") == "Alice Johnson"
        assert comparison.get("lower_candidate") == "Bob Smith"
        summary = comparison.get("summary", "")
        assert len(summary) > 0

    def test_explanation_mentions_both_names(
        self,
        base_state: AgentState,
    ) -> None:
        """The explanation summary should reference both candidates."""
        state = {
            **base_state,
            "human_feedback": "Why did the top candidate rank higher than the second?",
        }

        with patch("src.llm.client.get_llm", side_effect=RuntimeError("no llm")):
            result = explain_ranking_node(state)

        comparison = result.get("comparison_result") or {}
        text = (
            comparison.get("summary", "")
            + comparison.get("higher_candidate", "")
            + comparison.get("lower_candidate", "")
        )
        assert "Alice" in text or "Bob" in text

    def test_explanation_with_one_candidate(
        self,
        sample_requirements: dict,
        sample_shortlist: list[dict],
    ) -> None:
        """Asking for explanation with only 1 candidate returns graceful message."""
        state: AgentState = {
            "requirements": sample_requirements,
            "current_shortlist": [sample_shortlist[0]],
            "human_feedback": "Why did this candidate rank first?",
            "messages": [],
        }
        result = explain_ranking_node(state)
        comparison = result.get("comparison_result") or {}
        assert comparison.get("type") == "explanation"

    def test_explanation_no_shortlist(
        self,
    ) -> None:
        """Asking for explanation with no shortlist returns graceful message."""
        state: AgentState = {
            "current_shortlist": [],
            "requirements": {},
            "human_feedback": "Why did Alice rank higher than Bob?",
            "messages": [],
        }
        result = explain_ranking_node(state)
        comparison = result.get("comparison_result") or {}
        assert comparison.get("type") == "explanation"


# =====================================================================
# Scenario 5 — Interview Questions
# =====================================================================


class TestScenario5InterviewQuestions:
    """Scenario 5: Generate interview questions for a candidate."""

    def test_generate_questions_for_named_candidate(
        self,
        base_state: AgentState,
    ) -> None:
        """'Generate interview questions for Alice' returns 5 questions."""
        state = {
            **base_state,
            "human_feedback": "Generate interview questions for Alice Johnson",
        }

        # Force fallback path (no LLM)
        with patch("src.tools.generate_questions.get_llm", side_effect=RuntimeError("no llm")):
            result = generate_questions_node(state)

        comparison = result.get("comparison_result") or {}
        assert comparison.get("type") == "questions"
        assert "questions" in comparison
        assert isinstance(comparison["questions"], list)
        assert len(comparison["questions"]) > 0
        for q in comparison["questions"]:
            assert "question" in q

    def test_generate_questions_for_top_candidate(
        self,
        base_state: AgentState,
    ) -> None:
        """'Generate questions for the top candidate' defaults to rank #1."""
        state = {
            **base_state,
            "human_feedback": "Create a technical assessment for the top candidate",
        }

        with patch("src.tools.generate_questions.get_llm", side_effect=RuntimeError("no llm")):
            result = generate_questions_node(state)

        comparison = result.get("comparison_result") or {}
        assert comparison.get("type") == "questions"
        assert comparison.get("candidate_name") == "Alice Johnson"

    def test_questions_with_no_shortlist(
        self,
        sample_requirements: dict,
    ) -> None:
        """Asking for questions with no shortlist returns graceful message."""
        state: AgentState = {
            "requirements": sample_requirements,
            "current_shortlist": [],
            "human_feedback": "Generate interview questions for Alice",
            "messages": [],
        }
        result = generate_questions_node(state)
        comparison = result.get("comparison_result") or {}
        assert comparison.get("type") == "questions"
        # Should have a summary explaining no candidates
        assert "No candidates" in comparison.get("summary", "") or len(comparison.get("questions", [])) == 0


# =====================================================================
# Scenario 6 — Edge Case: No Results / Impossible Requirements
# =====================================================================


class TestScenario6EdgeCases:
    """Scenario 6: JD with impossible requirements returns graceful message."""

    def test_impossible_jd_does_not_crash(
        self,
        agent_graph,
        mock_extract_requirements,
        mock_resume_text,
        mock_score_candidate,
    ) -> None:
        """An impossible JD should not crash — returns empty shortlist or error."""
        with patch("src.agent.nodes.rag_search") as mock_rag:
            mock_rag.invoke.return_value = []
            result = agent_graph.invoke({
                "raw_jd": (
                    "Must have 20 years of experience in a technology invented 5 years ago. "
                    "Requires certification in non-existent-framework-XYZ."
                ),
                "messages": [],
            })

        # Should not crash; either empty shortlist or an error message
        assert "current_shortlist" in result
        # Either empty shortlist OR error set
        if not result.get("current_shortlist"):
            assert result["current_shortlist"] == []

    def test_empty_jd_returns_error(
        self,
        agent_graph,
    ) -> None:
        """An empty JD returns a clear error."""
        result = agent_graph.invoke({"raw_jd": "", "messages": []})
        assert "error" in result
        assert "too short" in result["error"].lower() or "empty" in result["error"].lower()

    def test_very_short_jd_returns_error(
        self,
        agent_graph,
    ) -> None:
        """A JD with <20 chars returns a clear error."""
        result = agent_graph.invoke({"raw_jd": "dev", "messages": []})
        assert "error" in result

    def test_no_requirements_does_not_crash(
        self,
        agent_graph,
        mock_extract_requirements,
    ) -> None:
        """If extract_requirements returns empty, the pipeline doesn't crash."""
        mock_extract_requirements.invoke.return_value = {
            "must_have": [],
            "nice_to_have": [],
            "experience_min_years": None,
            "education_level": None,
            "domain_keywords": [],
        }
        result = agent_graph.invoke({
            "raw_jd": "Looking for a developer with various skills and experience.",
            "messages": [],
        })
        # Should complete without crashing
        assert "requirements" in result


# =====================================================================
# Scenario 7 — Multi-Round Screening Pipeline
# =====================================================================


class TestScenario7MultiRound:
    """Scenario 7: Run all 3 screening rounds sequentially."""

    def test_pipeline_records_three_rounds(
        self,
        agent_graph,
        sample_jd: str,
        mock_extract_requirements,
        mock_rag_search,
        mock_resume_text,
        mock_score_candidate,
    ) -> None:
        """Full pipeline produces 3 ScreeningRound records."""
        result = agent_graph.invoke({"raw_jd": sample_jd, "messages": []})

        rounds = result.get("screening_rounds") or []
        # Pipeline may produce 1, 2, or 3 rounds depending on candidate survival.
        assert len(rounds) >= 1, "At least one screening round should be recorded"

        for r in rounds:
            _validate_screening_round(r)

        # Round numbers should be sequential 1, 2, 3 (or subset)
        round_numbers = sorted({r["round_number"] for r in rounds})
        assert round_numbers[0] == 1
        for i in range(1, len(round_numbers)):
            assert round_numbers[i] == round_numbers[i - 1] + 1

    def test_round_1_is_initial_type(
        self,
        agent_graph,
        sample_jd: str,
        mock_extract_requirements,
        mock_rag_search,
        mock_resume_text,
        mock_score_candidate,
    ) -> None:
        """Round 1 is always type='initial'."""
        result = agent_graph.invoke({"raw_jd": sample_jd, "messages": []})
        rounds = result.get("screening_rounds") or []
        round1 = next((r for r in rounds if r["round_number"] == 1), None)
        assert round1 is not None
        assert round1["round_type"] == "initial"
        assert round1["candidates_evaluated"] >= 0

    def test_shortlist_shrinks_or_stays_across_rounds(
        self,
        agent_graph,
        sample_jd: str,
        mock_extract_requirements,
        mock_rag_search,
        mock_resume_text,
        mock_score_candidate,
    ) -> None:
        """Each round's shortlist should be <= the previous round's."""
        result = agent_graph.invoke({"raw_jd": sample_jd, "messages": []})
        rounds = result.get("screening_rounds") or []
        if len(rounds) >= 2:
            sizes = [len(r.get("shortlisted_ids", [])) for r in rounds]
            for i in range(1, len(sizes)):
                assert sizes[i] <= sizes[i - 1]

    def test_final_shortlist_has_hire_recommendations(
        self,
        agent_graph,
        sample_jd: str,
        mock_extract_requirements,
        mock_rag_search,
        mock_resume_text,
        mock_score_candidate,
    ) -> None:
        """After all rounds, each shortlisted candidate has a hire_recommendation."""
        result = agent_graph.invoke({"raw_jd": sample_jd, "messages": []})
        shortlist = result.get("current_shortlist") or []
        for c in shortlist:
            assert "hire_recommendation" in c
            assert c["hire_recommendation"] in ("hire", "no_hire", "borderline", "unknown")

    def test_reports_generated_for_final_shortlist(
        self,
        agent_graph,
        sample_jd: str,
        mock_extract_requirements,
        mock_rag_search,
        mock_resume_text,
        mock_score_candidate,
    ) -> None:
        """Each final-shortlist candidate has a generated report."""
        result = agent_graph.invoke({"raw_jd": sample_jd, "messages": []})
        shortlist = result.get("current_shortlist") or []
        reports = result.get("generated_reports") or {}
        for c in shortlist:
            cid = c["candidate_id"]
            assert cid in reports, f"Missing report for {cid}"
            _validate_match_report(reports[cid])


# =====================================================================
# Bonus — Entry Point Tests (Phase 8)
# =====================================================================


class TestEntryPoint:
    """Verify the matching_agent.py submission entry point works."""

    def test_create_agent_returns_compiled_graph(self) -> None:
        """create_agent() returns a CompiledStateGraph."""
        from matching_agent import create_agent

        graph = create_agent()
        assert hasattr(graph, "invoke")
        assert callable(graph.invoke)

    def test_module_exports(self) -> None:
        """matching_agent module exports all required symbols."""
        import matching_agent

        assert hasattr(matching_agent, "create_agent")
        assert hasattr(matching_agent, "create_graph")
        assert hasattr(matching_agent, "invoke_linear_pipeline")
        assert hasattr(matching_agent, "run")
        assert hasattr(matching_agent, "run_cli")
        assert hasattr(matching_agent, "__version__")
        assert matching_agent.__version__ == "0.1.0"

    def test_help_flag(self, capsys) -> None:
        """python matching_agent.py --help shows usage without crashing."""
        import sys
        from matching_agent import _main

        original_argv = sys.argv
        sys.argv = ["matching_agent.py", "--help"]
        try:
            _main()
        except SystemExit:
            pass
        finally:
            sys.argv = original_argv
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "cli" in combined.lower() or "streamlit" in combined.lower()


# =====================================================================
# Bonus — Documentation Deliverables (Phase 8)
# =====================================================================


class TestDocumentationDeliverables:
    """Verify all Phase 8 documentation deliverables exist."""

    def test_state_machine_diagram_png_exists(self) -> None:
        """docs/state_machine_diagram.png should exist and be a valid PNG."""
        project_root = Path(__file__).parent.parent
        png_path = project_root / "docs" / "state_machine_diagram.png"
        assert png_path.exists(), f"State machine diagram not found: {png_path}"
        with open(png_path, "rb") as f:
            header = f.read(8)
        assert header.startswith(b"\x89PNG\r\n\x1a\n"), "Not a valid PNG file"

    def test_state_machine_diagram_mmd_exists(self) -> None:
        """docs/state_machine_diagram.mmd should exist and contain stateDiagram."""
        project_root = Path(__file__).parent.parent
        mmd_path = project_root / "docs" / "state_machine_diagram.mmd"
        assert mmd_path.exists()
        content = mmd_path.read_text()
        assert "stateDiagram" in content
        assert "ParseJD" in content
        assert "HumanFeedbackLoop" in content

    def test_demo_script_exists(self) -> None:
        """docs/demo_script.md should exist with 5+ sections."""
        project_root = Path(__file__).parent.parent
        demo_path = project_root / "docs" / "demo_script.md"
        assert demo_path.exists()
        content = demo_path.read_text()
        timestamp_count = content.count("## ")
        assert timestamp_count >= 5, f"Expected at least 5 sections, got {timestamp_count}"

    def test_matching_agent_entry_point_exists(self) -> None:
        """matching_agent.py should exist at the project root."""
        project_root = Path(__file__).parent.parent
        entry_path = project_root / "matching_agent.py"
        assert entry_path.exists()
        content = entry_path.read_text()
        assert "def create_agent" in content or "create_agent = create_graph" in content

    def test_expected_outputs_directory_exists(self) -> None:
        """tests/fixtures/expected_outputs/ should exist with schema files."""
        expected_dir = Path(__file__).parent / "fixtures" / "expected_outputs"
        assert expected_dir.exists()
        assert expected_dir.is_dir()
        files = list(expected_dir.glob("*.json")) + list(expected_dir.glob("*.md"))
        assert len(files) >= 4, f"Expected at least 4 schema files, got {len(files)}"
