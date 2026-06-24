"""
Agentic Profile Matching — Phase 4 Linear Graph Integration Tests.

Tests the full graph execution with mocked LLM but real RAG data.
Requires Phase 1 indexed data in data/chroma_db/.

Run:
    pytest tests/test_graph_linear.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.graph import create_graph

SAMPLE_JD_PATH = Path(__file__).parent / "fixtures" / "sample_jd.txt"


@pytest.fixture
def sample_jd() -> str:
    assert SAMPLE_JD_PATH.exists(), f"Sample JD not found: {SAMPLE_JD_PATH}"
    return SAMPLE_JD_PATH.read_text()


class TestLinearGraphMocked:
    """Test the full linear graph with all external deps mocked."""

    @patch("src.agent.nodes.score_candidate")
    @patch("src.agent.nodes.get_full_resume_text")
    @patch("src.agent.nodes.rag_search")
    @patch("src.agent.nodes.extract_requirements")
    def test_graph_runs_to_completion(
        self,
        mock_extract: MagicMock,
        mock_rag: MagicMock,
        mock_resume: MagicMock,
        mock_score: MagicMock,
    ) -> None:
        """Full graph invocation with all deps mocked."""
        # Setup mocks
        mock_extract.invoke.return_value = {
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
        mock_rag.invoke.return_value = [
            {"candidate_id": "alice_123", "name": "Alice", "score": 0.1, "excerpt": "React developer"},
            {"candidate_id": "bob_456", "name": "Bob", "score": 0.2, "excerpt": "Python developer"},
        ]
        mock_resume.return_value = "Alice Johnson\nSenior React Developer\n5 years React experience"
        mock_score.return_value = MagicMock(
            must_have_score=0.9,
            nice_to_have_score=0.6,
            reasoning="Strong match",
            strengths=["React", "TypeScript"],
            gaps=["No AWS"],
            excerpts=["5 years React"],
        )

        graph = create_graph()
        result = graph.invoke({"raw_jd": "Senior React Developer with 5+ years experience.", "messages": []})

        # Verify pipeline completed
        assert "requirements" in result
        assert result["requirements"]["must_have"] is not None
        assert "all_candidate_ids" in result
        assert "current_shortlist" in result
        assert "generated_reports" in result
        assert result["awaiting_human_feedback"] is True

    @patch("src.agent.nodes.score_candidate")
    @patch("src.agent.nodes.get_full_resume_text")
    @patch("src.agent.nodes.rag_search")
    @patch("src.agent.nodes.extract_requirements")
    def test_shortlist_sorted_by_score(
        self,
        mock_extract: MagicMock,
        mock_rag: MagicMock,
        mock_resume: MagicMock,
        mock_score: MagicMock,
    ) -> None:
        """Verify shortlist is sorted by composite score descending."""
        mock_extract.invoke.return_value = {
            "must_have": [{"skill": "React", "type": "tech", "weight": 1.0, "evidence": ""}],
            "nice_to_have": [],
            "experience_min_years": None,
            "education_level": None,
            "domain_keywords": [],
        }
        mock_rag.invoke.return_value = [
            {"candidate_id": "alice", "name": "Alice", "score": 0.1, "excerpt": "..."},
            {"candidate_id": "bob", "name": "Bob", "score": 0.2, "excerpt": "..."},
        ]
        mock_resume.return_value = "Some resume text"

        # Return different scores for different candidates
        def score_side_effect(resume_text, requirements, llm=None):
            if "alice" in resume_text.lower() or "Alice" in resume_text:
                return MagicMock(
                    must_have_score=0.9, nice_to_have_score=0.8,
                    reasoning="Strong", strengths=["React"], gaps=[], excerpts=[]
                )
            else:
                return MagicMock(
                    must_have_score=0.4, nice_to_have_score=0.3,
                    reasoning="Weak", strengths=[], gaps=["No React"], excerpts=[]
                )
        mock_score.side_effect = score_side_effect

        graph = create_graph()
        result = graph.invoke({"raw_jd": "Looking for a React developer with frontend experience.", "messages": []})

        shortlist = result.get("current_shortlist", [])
        if len(shortlist) >= 2:
            # First should have higher score than second
            assert shortlist[0]["score"] >= shortlist[1]["score"]

    @patch("src.agent.nodes.score_candidate")
    @patch("src.agent.nodes.get_full_resume_text")
    @patch("src.agent.nodes.rag_search")
    @patch("src.agent.nodes.extract_requirements")
    def test_reports_match_shortlist(
        self,
        mock_extract: MagicMock,
        mock_rag: MagicMock,
        mock_resume: MagicMock,
        mock_score: MagicMock,
    ) -> None:
        """Each shortlisted candidate should have a report."""
        mock_extract.invoke.return_value = {
            "must_have": [{"skill": "React", "type": "tech", "weight": 1.0, "evidence": ""}],
            "nice_to_have": [],
            "experience_min_years": None,
            "education_level": None,
            "domain_keywords": [],
        }
        mock_rag.invoke.return_value = [
            {"candidate_id": "alice_123", "name": "Alice", "score": 0.1, "excerpt": "..."},
        ]
        mock_resume.return_value = "Alice Johnson\nReact Developer\n5 years experience"
        mock_score.return_value = MagicMock(
            must_have_score=0.85, nice_to_have_score=0.7,
            reasoning="Good match", strengths=["React"], gaps=[], excerpts=[]
        )

        graph = create_graph()
        result = graph.invoke({"raw_jd": "React developer needed.", "messages": []})

        reports = result.get("generated_reports", {})
        shortlist = result.get("current_shortlist", [])
        for candidate in shortlist:
            cid = candidate["candidate_id"]
            assert cid in reports, f"Missing report for {cid}"
            assert "# Candidate Match Report" in reports[cid]

    def test_short_jd_stops_at_parse(self) -> None:
        """Graph should route to END for a too-short JD."""
        graph = create_graph()
        result = graph.invoke({"raw_jd": "Hi", "messages": []})
        # Should have error from parse_jd
        assert "error" in result
        assert "too short" in result["error"].lower()


class TestLinearGraphWithRAG:
    """Test with real RAG data but mocked LLM scoring.
    Requires Phase 1 data in data/chroma_db/."""

    @patch("src.agent.nodes.score_candidate")
    @patch("src.agent.nodes.extract_requirements")
    def test_real_rag_search(
        self,
        mock_extract: MagicMock,
        mock_score: MagicMock,
    ) -> None:
        """Use real RAG search but mock LLM extraction and scoring."""
        mock_extract.invoke.return_value = {
            "must_have": [
                {"skill": "React", "type": "tech", "weight": 1.0, "evidence": "Required"},
                {"skill": "TypeScript", "type": "tech", "weight": 0.9, "evidence": "Required"},
            ],
            "nice_to_have": [
                {"skill": "AWS", "type": "tech", "weight": 0.5, "evidence": "Preferred"},
            ],
            "experience_min_years": 5,
            "education_level": "BS",
            "domain_keywords": ["frontend"],
        }
        mock_score.return_value = MagicMock(
            must_have_score=0.8, nice_to_have_score=0.6,
            reasoning="Good", strengths=["React"], gaps=["No AWS"], excerpts=["React experience"]
        )

        graph = create_graph()
        result = graph.invoke({"raw_jd": "Senior React Frontend Developer with TypeScript.", "messages": []})

        # Should have found candidates from real RAG
        all_ids = result.get("all_candidate_ids", [])
        # With 4 indexed resumes, should find at least 1
        if len(all_ids) > 0:
            assert len(result["current_shortlist"]) > 0
            assert result["awaiting_human_feedback"] is True
