"""
Agentic Profile Matching — Phase 2 Tests.

Tests for Agent State (TypedDicts) and Pydantic Models covering:
  1. Full-field state construction
  2. Minimal/default state construction
  3. add_messages reducer accumulation
  4. Pydantic model serialization round-trips
  5. Pydantic validation rejection (invalid data)
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.models import (
    CandidateScore,
    ComparisonResult,
    ComparisonRow,
    ExtractedRequirements,
    InterviewQuestion,
    SkillRequirement,
)
from src.agent.state import (
    AgentState,
    CandidateMatch,
    Requirements,
    ScreeningRound,
)


# ===================================================================
# 1. TypedDict Construction Tests
# ===================================================================


class TestCandidateMatch:
    """Tests for CandidateMatch TypedDict."""

    def test_full_construction(self) -> None:
        match: CandidateMatch = {
            "candidate_id": "c001",
            "name": "Alice Johnson",
            "score": 0.87,
            "must_have_score": 0.92,
            "nice_to_have_score": 0.75,
            "reasoning": "Strong React background with 5 years experience.",
            "strengths": ["React", "TypeScript", "Team leadership"],
            "gaps": ["No cloud experience"],
            "resume_excerpts": ["Led a team of 5 developers on a React project"],
            "interview_questions": ["Describe your experience with React hooks."],
            "hire_recommendation": "hire",
            "improvement_suggestions": ["Gain AWS certification"],
        }
        assert match["candidate_id"] == "c001"
        assert match["score"] == 0.87
        assert len(match["strengths"]) == 3
        assert match["hire_recommendation"] in ("hire", "no_hire", "borderline")

    def test_partial_construction(self) -> None:
        """TypedDict with total=False allows partial construction."""
        match: CandidateMatch = {
            "candidate_id": "c002",
            "name": "Bob Smith",
        }
        assert match["candidate_id"] == "c002"
        assert "score" not in match  # not required


class TestRequirements:
    """Tests for Requirements TypedDict."""

    def test_full_construction(self) -> None:
        req: Requirements = {
            "raw_jd": "Senior React Developer with 5+ years experience...",
            "must_have": [
                {"skill": "React", "type": "tech", "weight": 1.0},
                {"skill": "TypeScript", "type": "tech", "weight": 0.9},
            ],
            "nice_to_have": [
                {"skill": "AWS", "type": "tech", "weight": 0.5},
            ],
            "experience_min_years": 5,
            "education_level": "BS",
            "domain_keywords": ["frontend", "web development", "SPA"],
        }
        assert len(req["must_have"]) == 2
        assert req["experience_min_years"] == 5
        assert "frontend" in req["domain_keywords"]

    def test_minimal_construction(self) -> None:
        req: Requirements = {"raw_jd": "A job description"}
        assert req["raw_jd"] == "A job description"
        assert "must_have" not in req


class TestScreeningRound:
    """Tests for ScreeningRound TypedDict."""

    def test_full_construction(self) -> None:
        round_data: ScreeningRound = {
            "round_number": 1,
            "round_type": "initial",
            "candidates_evaluated": 100,
            "shortlisted_ids": ["c001", "c002", "c003"],
            "eliminated_ids": ["c004", "c005"],
            "notes": "Narrowed from 100 to 3 based on must-have criteria.",
        }
        assert round_data["round_number"] == 1
        assert round_data["round_type"] == "initial"
        assert len(round_data["eliminated_ids"]) == 2


class TestAgentState:
    """Tests for the top-level AgentState TypedDict."""

    def test_full_construction(self) -> None:
        """Build a fully populated AgentState with all 14 fields."""
        from langgraph.graph.message import add_messages

        match: CandidateMatch = {
            "candidate_id": "c001",
            "name": "Alice",
            "score": 0.9,
            "must_have_score": 0.95,
            "nice_to_have_score": 0.8,
            "reasoning": "Great fit",
            "strengths": ["React"],
            "gaps": [],
            "resume_excerpts": ["5 years React"],
            "interview_questions": [],
            "hire_recommendation": "hire",
            "improvement_suggestions": [],
        }
        req: Requirements = {
            "raw_jd": "JD text",
            "must_have": [{"skill": "React", "type": "tech", "weight": 1.0}],
            "nice_to_have": [],
            "experience_min_years": 3,
            "education_level": "BS",
            "domain_keywords": ["frontend"],
        }
        round_data: ScreeningRound = {
            "round_number": 1,
            "round_type": "initial",
            "candidates_evaluated": 50,
            "shortlisted_ids": ["c001"],
            "eliminated_ids": [],
            "notes": "Round 1 complete",
        }

        state: AgentState = {
            "messages": [HumanMessage(content="Match this JD")],
            "conversation_history": [
                {"role": "user", "content": "Match this JD", "timestamp": "2026-06-24T10:00:00Z"}
            ],
            "raw_jd": "Senior React Developer",
            "requirements": req,
            "requirements_version": 1,
            "all_candidate_ids": ["c001", "c002"],
            "current_shortlist": [match],
            "screening_rounds": [round_data],
            "current_round": 1,
            "comparison_result": None,
            "generated_reports": {"c001": "# Alice Report\n..."},
            "awaiting_human_feedback": True,
            "human_feedback": None,
            "next_action": "generate_report",
            "error": None,
        }

        assert state["raw_jd"] == "Senior React Developer"
        assert len(state["current_shortlist"]) == 1
        assert state["requirements_version"] == 1
        assert state["awaiting_human_feedback"] is True
        assert state["comparison_result"] is None
        assert isinstance(state["messages"][0], HumanMessage)

    def test_minimal_construction(self) -> None:
        """AgentState with total=False — can be constructed with zero fields."""
        state: AgentState = {}
        assert len(state) == 0

    def test_partial_construction(self) -> None:
        """AgentState can be built incrementally."""
        state: AgentState = {
            "raw_jd": "A job description",
            "current_round": 0,
        }
        assert state["raw_jd"] == "A job description"
        assert state["current_round"] == 0


# ===================================================================
# 2. add_messages Reducer Tests
# ===================================================================


class TestAddMessagesReducer:
    """Verify the LangGraph add_messages reducer accumulates correctly."""

    def test_accumulates_messages(self) -> None:
        from langgraph.graph.message import add_messages

        existing: list = [HumanMessage(content="Hello")]
        new: list = [AIMessage(content="Hi there")]
        result = add_messages(existing, new)
        assert len(result) == 2
        assert result[0].content == "Hello"
        assert result[1].content == "Hi there"

    def test_idempotent_single_message(self) -> None:
        from langgraph.graph.message import add_messages

        existing: list = []
        new: list = [SystemMessage(content="You are a recruiter.")]
        result = add_messages(existing, new)
        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)

    def test_preserves_types(self) -> None:
        from langgraph.graph.message import add_messages

        existing: list = [
            HumanMessage(content="Q1"),
            AIMessage(content="A1"),
        ]
        new: list = [
            HumanMessage(content="Q2"),
            AIMessage(content="A2"),
        ]
        result = add_messages(existing, new)
        assert len(result) == 4
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], AIMessage)
        assert isinstance(result[2], HumanMessage)
        assert isinstance(result[3], AIMessage)


# ===================================================================
# 3. Pydantic Model Tests
# ===================================================================


class TestSkillRequirement:
    def test_valid_construction(self) -> None:
        sr = SkillRequirement(
            skill="React",
            type="tech",
            weight=0.9,
            evidence="Requires 3+ years of React experience",
        )
        assert sr.skill == "React"
        assert sr.weight == 0.9
        dumped = sr.model_dump()
        assert dumped["type"] == "tech"

    def test_rejects_empty_skill(self) -> None:
        with pytest.raises(Exception):
            SkillRequirement(
                skill="",
                type="tech",
                weight=0.5,
                evidence="Some evidence",
            )

    def test_rejects_invalid_type(self) -> None:
        with pytest.raises(Exception):
            SkillRequirement(
                skill="React",
                type="invalid_type",  # type: ignore[arg-type]
                weight=0.5,
                evidence="Some evidence",
            )

    def test_rejects_weight_out_of_range(self) -> None:
        with pytest.raises(Exception):
            SkillRequirement(
                skill="React",
                type="tech",
                weight=1.5,  # exceeds 1.0
                evidence="Some evidence",
            )

    def test_serialization_round_trip(self) -> None:
        sr = SkillRequirement(
            skill="Python",
            type="tech",
            weight=1.0,
            evidence="Must know Python",
        )
        json_str = sr.model_dump_json()
        restored = SkillRequirement.model_validate_json(json_str)
        assert restored.skill == "Python"
        assert restored.weight == 1.0


class TestExtractedRequirements:
    def test_full_construction(self) -> None:
        reqs = ExtractedRequirements(
            must_have=[
                SkillRequirement(
                    skill="React",
                    type="tech",
                    weight=1.0,
                    evidence="React is required",
                )
            ],
            nice_to_have=[
                SkillRequirement(
                    skill="AWS",
                    type="tech",
                    weight=0.5,
                    evidence="AWS preferred",
                )
            ],
            experience_min_years=5,
            education_level="BS",
            domain_keywords=["frontend", "web"],
        )
        assert len(reqs.must_have) == 1
        assert len(reqs.nice_to_have) == 1
        assert reqs.experience_min_years == 5
        assert reqs.education_level == "BS"

    def test_defaults_to_empty(self) -> None:
        reqs = ExtractedRequirements()
        assert reqs.must_have == []
        assert reqs.nice_to_have == []
        assert reqs.experience_min_years is None
        assert reqs.domain_keywords == []

    def test_serialization_round_trip(self) -> None:
        reqs = ExtractedRequirements(
            must_have=[
                SkillRequirement(
                    skill="Python",
                    type="tech",
                    weight=0.8,
                    evidence="Python required",
                )
            ],
            experience_min_years=3,
        )
        json_str = reqs.model_dump_json()
        restored = ExtractedRequirements.model_validate_json(json_str)
        assert len(restored.must_have) == 1
        assert restored.must_have[0].skill == "Python"
        assert restored.experience_min_years == 3


class TestCandidateScore:
    def test_valid_construction(self) -> None:
        score = CandidateScore(
            must_have_score=0.9,
            nice_to_have_score=0.7,
            reasoning="Strong match on all must-have criteria.",
            strengths=["React", "TypeScript"],
            gaps=["No DevOps experience"],
            excerpts=["5 years of React development"],
        )
        assert score.must_have_score == 0.9
        assert score.nice_to_have_score == 0.7
        assert len(score.strengths) == 2

    def test_composite_score(self) -> None:
        score = CandidateScore(
            must_have_score=0.8,
            nice_to_have_score=0.6,
            reasoning="Good fit",
            strengths=[],
            gaps=[],
            excerpts=[],
        )
        # 0.7 * 0.8 + 0.3 * 0.6 = 0.56 + 0.18 = 0.74
        assert abs(score.composite_score - 0.74) < 1e-9

    def test_rejects_score_above_one(self) -> None:
        with pytest.raises(Exception):
            CandidateScore(
                must_have_score=1.5,
                nice_to_have_score=0.5,
                reasoning="Invalid",
                strengths=[],
                gaps=[],
                excerpts=[],
            )

    def test_rejects_negative_score(self) -> None:
        with pytest.raises(Exception):
            CandidateScore(
                must_have_score=-0.1,
                nice_to_have_score=0.5,
                reasoning="Invalid",
                strengths=[],
                gaps=[],
                excerpts=[],
            )

    def test_rejects_empty_reasoning(self) -> None:
        with pytest.raises(Exception):
            CandidateScore(
                must_have_score=0.5,
                nice_to_have_score=0.5,
                reasoning="",
                strengths=[],
                gaps=[],
                excerpts=[],
            )

    def test_serialization_round_trip(self) -> None:
        score = CandidateScore(
            must_have_score=0.85,
            nice_to_have_score=0.65,
            reasoning="Strong candidate",
            strengths=["React", "Node.js"],
            gaps=["No cloud"],
            excerpts=["Led React team"],
        )
        json_str = score.model_dump_json()
        restored = CandidateScore.model_validate_json(json_str)
        assert restored.must_have_score == 0.85
        assert len(restored.strengths) == 2
        # composite_score is a property, not serialized, but should still work
        assert abs(restored.composite_score - (0.7 * 0.85 + 0.3 * 0.65)) < 1e-9


class TestInterviewQuestion:
    def test_valid_construction(self) -> None:
        q = InterviewQuestion(
            question="Describe your experience with React hooks.",
            category="technical",
            targets_gap="React hooks depth unknown",
            difficulty="medium",
            follow_ups=["Can you give an example of a custom hook?"],
        )
        assert q.category == "technical"
        assert q.difficulty == "medium"
        assert len(q.follow_ups) == 1

    def test_defaults_empty_follow_ups(self) -> None:
        q = InterviewQuestion(
            question="Tell me about a time you led a team.",
            category="behavioral",
            targets_gap="Leadership experience unclear",
            difficulty="easy",
        )
        assert q.follow_ups == []

    def test_rejects_invalid_category(self) -> None:
        with pytest.raises(Exception):
            InterviewQuestion(
                question="Some question",
                category="nonexistent",  # type: ignore[arg-type]
                targets_gap="gap",
                difficulty="medium",
            )

    def test_rejects_invalid_difficulty(self) -> None:
        with pytest.raises(Exception):
            InterviewQuestion(
                question="Some question",
                category="technical",
                targets_gap="gap",
                difficulty="expert",  # type: ignore[arg-type]
            )

    def test_serialization_round_trip(self) -> None:
        q = InterviewQuestion(
            question="Explain microservices.",
            category="technical",
            targets_gap="Architecture knowledge",
            difficulty="hard",
            follow_ups=["How do you handle inter-service communication?"],
        )
        json_str = q.model_dump_json()
        restored = InterviewQuestion.model_validate_json(json_str)
        assert restored.question == "Explain microservices."
        assert restored.difficulty == "hard"
        assert len(restored.follow_ups) == 1


class TestComparisonRow:
    def test_valid_construction(self) -> None:
        row = ComparisonRow(
            criterion="React experience",
            values=["5 years", "2 years"],
        )
        assert row.criterion == "React experience"
        assert len(row.values) == 2

    def test_rejects_single_value(self) -> None:
        with pytest.raises(Exception):
            ComparisonRow(
                criterion="React experience",
                values=["5 years"],  # need at least 2
            )


class TestComparisonResult:
    def test_valid_construction(self) -> None:
        result = ComparisonResult(
            candidates=[
                {"id": "c001", "name": "Alice"},
                {"id": "c002", "name": "Bob"},
            ],
            comparison_table={
                "React experience": ["5 years", "2 years"],
                "Education": ["MS CS", "BS CS"],
            },
            summary="Alice has stronger technical depth; Bob has room to grow.",
        )
        assert len(result.candidates) == 2
        assert len(result.comparison_table) == 2

    def test_rejects_inconsistent_table_lengths(self) -> None:
        with pytest.raises(Exception):
            ComparisonResult(
                candidates=[
                    {"id": "c001", "name": "Alice"},
                    {"id": "c002", "name": "Bob"},
                ],
                comparison_table={
                    "React": ["5 years", "2 years"],
                    "Education": ["MS CS"],  # only 1 value, should be 2
                },
                summary="Comparison",
            )

    def test_empty_table_is_valid(self) -> None:
        result = ComparisonResult(
            candidates=[
                {"id": "c001", "name": "Alice"},
                {"id": "c002", "name": "Bob"},
            ],
            comparison_table={},
            summary="No criteria compared.",
        )
        assert result.comparison_table == {}

    def test_serialization_round_trip(self) -> None:
        result = ComparisonResult(
            candidates=[
                {"id": "c001", "name": "Alice", "score": 0.9},
                {"id": "c002", "name": "Bob", "score": 0.7},
            ],
            comparison_table={
                "React": ["5 years", "2 years"],
                "Leadership": ["Led team of 10", "No leadership role"],
            },
            summary="Alice is the stronger candidate overall.",
        )
        json_str = result.model_dump_json()
        restored = ComparisonResult.model_validate_json(json_str)
        assert len(restored.candidates) == 2
        assert restored.comparison_table["React"] == ["5 years", "2 years"]
        assert "stronger" in restored.summary

    def test_rejects_single_candidate(self) -> None:
        with pytest.raises(Exception):
            ComparisonResult(
                candidates=[{"id": "c001", "name": "Alice"}],
                comparison_table={"React": ["5 years"]},
                summary="Single candidate comparison",
            )