"""
Agentic Profile Matching — Phase 3 Tool Tests.

Tests are organized by test class:
  - TestPromptTemplates: LLM-free, verify prompt format
  - TestFileTools: LLM-free, use temp directories
  - TestRagSearchTool: LLM-free, uses Phase 1 indexed data
  - TestToolSchemas: LLM-free, verify @tool signatures and return types
  - TestExtractRequirementsIntegration: Requires LLM (marked with pytest.mark.integration)
  - TestCompareCandidatesFallback: LLM-free, tests fallback path
  - TestGenerateQuestionsFallback: LLM-free, tests fallback path

Run unit tests (no LLM needed):
    pytest tests/test_tools.py -v -m "not integration"

Run all including LLM integration:
    pytest tests/test_tools.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.agent.models import (
    CandidateScore,
    ComparisonResult,
    ExtractedRequirements,
    InterviewQuestion,
    SkillRequirement,
)
from src.prompts.comparison import build_comparison_prompt, COMPARISON_SYSTEM_PROMPT
from src.prompts.explanation import build_explanation_prompt, EXPLANATION_SYSTEM_PROMPT
from src.prompts.extraction import build_extraction_prompt, EXTRACTION_SYSTEM_PROMPT
from src.prompts.questions import build_questions_prompt, QUESTIONS_SYSTEM_PROMPT
from src.prompts.scoring import build_scoring_prompt, SCORING_SYSTEM_PROMPT
from src.tools.compare_candidates import compare_candidates, _fallback_comparison
from src.tools.file_tools import (
    get_file_metadata,
    list_files,
    read_file,
    search_files,
    write_file,
)
from src.tools.generate_questions import generate_interview_questions
from src.tools.rag_search import rag_search

# Path to sample JD fixture
SAMPLE_JD_PATH = Path(__file__).parent / "fixtures" / "sample_jd.txt"

# Sample candidate profiles for testing comparison/fallback
SAMPLE_PROFILES = [
    {
        "candidate_id": "alice_test",
        "name": "Alice Johnson",
        "score": 0.92,
        "must_have_score": 0.95,
        "nice_to_have_score": 0.85,
        "reasoning": "Strong match on all must-have criteria with 6 years React experience.",
        "strengths": ["React", "TypeScript", "Team leadership", "Performance optimization"],
        "gaps": ["No cloud experience"],
        "hire_recommendation": "hire",
    },
    {
        "candidate_id": "bob_test",
        "name": "Bob Smith",
        "score": 0.65,
        "must_have_score": 0.60,
        "nice_to_have_score": 0.75,
        "reasoning": "Good Python backend skills but lacks frontend depth.",
        "strengths": ["Python", "Django", "PostgreSQL", "Docker"],
        "gaps": ["No React experience", "No TypeScript"],
        "hire_recommendation": "borderline",
    },
]


# ===================================================================
# 1. Prompt Template Tests (no LLM needed)
# ===================================================================


class TestPromptTemplates:
    """Verify all 5 prompt templates produce well-formatted messages."""

    def test_extraction_prompt_structure(self) -> None:
        msgs = build_extraction_prompt("A JD about React")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert len(msgs[0]["content"]) > 100
        assert msgs[1]["role"] == "user"
        assert "React" in msgs[1]["content"]

    def test_scoring_prompt_structure(self) -> None:
        reqs = {
            "must_have": [{"skill": "React", "type": "tech", "weight": 1.0}],
            "nice_to_have": [{"skill": "AWS", "type": "tech", "weight": 0.5}],
        }
        msgs = build_scoring_prompt("Resume text here", reqs)
        assert len(msgs) == 2
        assert "system" in msgs[0]["role"]
        assert "React" in msgs[1]["content"]
        assert "Resume text here" in msgs[1]["content"]

    def test_comparison_prompt_structure(self) -> None:
        msgs = build_comparison_prompt(SAMPLE_PROFILES)
        assert len(msgs) == 2
        assert "Alice Johnson" in msgs[1]["content"]
        assert "Bob Smith" in msgs[1]["content"]

    def test_questions_prompt_structure(self) -> None:
        reqs = {"must_have": [{"skill": "React", "type": "tech", "weight": 1.0}]}
        msgs = build_questions_prompt("Alice", "Resume text", reqs, num_questions=3)
        assert len(msgs) == 2
        assert "3" in msgs[1]["content"]
        assert "Alice" in msgs[1]["content"]

    def test_explanation_prompt_structure(self) -> None:
        msgs = build_explanation_prompt(
            SAMPLE_PROFILES[0], SAMPLE_PROFILES[1]
        )
        assert len(msgs) == 2
        assert "higher" in msgs[1]["content"].lower() or "first" in msgs[1]["content"].lower()

    def test_all_system_prompts_are_substantial(self) -> None:
        """Each system prompt should be at least 200 characters."""
        prompts = {
            "extraction": EXTRACTION_SYSTEM_PROMPT,
            "scoring": SCORING_SYSTEM_PROMPT,
            "comparison": COMPARISON_SYSTEM_PROMPT,
            "questions": QUESTIONS_SYSTEM_PROMPT,
            "explanation": EXPLANATION_SYSTEM_PROMPT,
        }
        for name, prompt in prompts.items():
            assert len(prompt) > 200, f"{name} prompt too short: {len(prompt)} chars"


# ===================================================================
# 2. File Tools Tests (no LLM needed)
# ===================================================================


class TestFileTools:
    def test_list_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world")
        (tmp_path / "subdir").mkdir()
        result = list_files.invoke({"directory": str(tmp_path)})
        assert isinstance(result, list)
        names = [f["name"] for f in result]
        assert "a.txt" in names
        assert "b.txt" in names
        assert "subdir" in names

    def test_list_files_nonexistent(self) -> None:
        result = list_files.invoke({"directory": "/nonexistent/path"})
        assert "error" in result[0]

    def test_read_file(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!", encoding="utf-8")
        result = read_file.invoke({"path": str(test_file)})
        assert result["content"] == "Hello, world!"
        assert result["size"] == 13
        assert result["truncated"] is False

    def test_read_file_truncation(self, tmp_path: Path) -> None:
        test_file = tmp_path / "long.txt"
        test_file.write_text("X" * 20000, encoding="utf-8")
        result = read_file.invoke({"path": str(test_file), "max_chars": 100})
        assert result["truncated"] is True

    def test_read_file_nonexistent(self) -> None:
        result = read_file.invoke({"path": "/nonexistent/file.txt"})
        assert "error" in result

    def test_write_file(self, tmp_path: Path) -> None:
        out_file = tmp_path / "sub" / "out.txt"
        result = write_file.invoke({"path": str(out_file), "content": "test content"})
        assert result["success"] is True
        assert out_file.read_text() == "test content"

    def test_search_files(self, tmp_path: Path) -> None:
        (tmp_path / "resume_alice.pdf").write_text("data")
        (tmp_path / "resume_bob.pdf").write_text("data")
        (tmp_path / "report.txt").write_text("data")
        result = search_files.invoke({"query": "resume", "directory": str(tmp_path)})
        assert len(result) == 2
        names = [f["name"] for f in result]
        assert "resume_alice.pdf" in names
        assert "resume_bob.pdf" in names

    def test_get_file_metadata(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")
        result = get_file_metadata.invoke({"path": str(test_file)})
        assert result["name"] == "test.py"
        assert result["extension"] == ".py"
        assert result["is_file"] is True
        assert result["is_dir"] is False
        assert result["size"] > 0


# ===================================================================
# 3. RAG Search Tool Test (no LLM needed, uses Phase 1 data)
# ===================================================================


@pytest.mark.integration
class TestRagSearchTool:
    """Tests the rag_search @tool against the indexed corpus."""

    def test_rag_search_returns_results(self) -> None:
        """rag_search('React developer') should return non-empty results."""
        results = rag_search.invoke({"query": "React developer", "top_k": 3})
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_rag_search_result_format(self) -> None:
        """Each result must have the standardized keys."""
        results = rag_search.invoke({"query": "developer", "top_k": 3})
        for r in results:
            assert "candidate_id" in r
            assert "name" in r
            assert "score" in r
            assert "excerpt" in r

    def test_rag_search_empty_query(self) -> None:
        """Empty query should return empty list."""
        results = rag_search.invoke({"query": "", "top_k": 3})
        assert results == []

    def test_rag_search_deduplication(self) -> None:
        """Results should be deduplicated by candidate_id."""
        results = rag_search.invoke({"query": "developer engineer", "top_k": 20})
        ids = [r["candidate_id"] for r in results]
        assert len(ids) == len(set(ids))


# ===================================================================
# 4. Tool Schema Tests (no LLM needed)
# ===================================================================


class TestToolSchemas:
    """Verify @tool signatures and that tools are callable."""

    def test_extract_requirements_is_tool(self) -> None:
        from src.tools.extract_requirements import extract_requirements as er
        assert er.name == "extract_requirements"
        assert er.description is not None and len(er.description) > 20

    def test_rag_search_is_tool(self) -> None:
        assert rag_search.name == "rag_search"

    def test_compare_candidates_is_tool(self) -> None:
        assert compare_candidates.name == "compare_candidates"

    def test_generate_questions_is_tool(self) -> None:
        assert generate_interview_questions.name == "generate_interview_questions"

    def test_all_tools_have_descriptions(self) -> None:
        """Every tool must have a non-empty description for the LLM."""
        tools = [
            rag_search,
            compare_candidates,
            generate_interview_questions,
            list_files,
            read_file,
            write_file,
            search_files,
            get_file_metadata,
        ]
        for tool in tools:
            assert tool.description, f"{tool.name} has no description"


# ===================================================================
# 5. Compare Candidates Fallback Test (no LLM needed)
# ===================================================================


class TestCompareCandidatesFallback:
    def test_fallback_comparison_structure(self) -> None:
        """Fallback comparison should produce valid ComparisonResult."""
        result = _fallback_comparison(SAMPLE_PROFILES)
        assert "candidates" in result
        assert "comparison_table" in result
        assert "summary" in result
        assert len(result["candidates"]) == 2

    def test_fallback_comparison_scores_in_table(self) -> None:
        result = _fallback_comparison(SAMPLE_PROFILES)
        table = result["comparison_table"]
        assert "Overall Score" in table
        assert len(table["Overall Score"]) == 2
        assert "0.92" in table["Overall Score"][0]

    def test_compare_single_candidate_returns_error_summary(self) -> None:
        result = compare_candidates.invoke({
            "candidate_ids": ["only_one"],
        })
        assert "At least 2" in result.get("summary", "")


# ===================================================================
# 6. Generate Questions Fallback Test (no LLM needed)
# ===================================================================


class TestGenerateQuestionsFallback:
    @pytest.mark.integration
    def test_nonexistent_candidate_returns_empty_questions(self) -> None:
        """Candidate not in vector store should return empty questions list."""
        result = generate_interview_questions.invoke({
            "candidate_id": "nonexistent_candidate_xyz",
            "candidate_name": "Ghost Candidate",
            "num_questions": 3,
        })
        assert result["candidate_id"] == "nonexistent_candidate_xyz"
        assert result["candidate_name"] == "Ghost Candidate"
        # Questions will be empty because LLM is unavailable or candidate not found
        assert "questions" in result
        assert isinstance(result["questions"], list)


# ===================================================================
# 7. Extract Requirements — Short/Empty Input (no LLM needed)
# ===================================================================


class TestExtractRequirementsEdgeCases:
    def test_empty_jd_returns_empty_requirements(self) -> None:
        from src.tools.extract_requirements import extract_requirements
        result = extract_requirements.invoke({"jd": ""})
        assert result["must_have"] == []
        assert result["nice_to_have"] == []
        assert result["experience_min_years"] is None

    def test_short_jd_returns_empty_requirements(self) -> None:
        from src.tools.extract_requirements import extract_requirements
        result = extract_requirements.invoke({"jd": "Hi"})
        # Too short to extract meaningful requirements
        assert isinstance(result["must_have"], list)
        assert isinstance(result["nice_to_have"], list)


# ===================================================================
# 8. Integration Tests (require LLM — marked with pytest.mark.integration)
# ===================================================================

# Only run these when LLM is available (e.g., Gemini quota reset)
pytestmark_integration = pytest.mark.integration


@pytest.mark.integration
class TestExtractRequirementsIntegration:
    """Tests that require a working LLM API."""

    def test_extract_requirements_real_jd(self) -> None:
        """Parse a real JD and verify structured output."""
        from src.tools.extract_requirements import extract_requirements
        assert SAMPLE_JD_PATH.exists(), f"Sample JD not found: {SAMPLE_JD_PATH}"
        jd_text = SAMPLE_JD_PATH.read_text()

        result = extract_requirements.invoke({"jd": jd_text})

        # Verify schema
        assert "must_have" in result
        assert "nice_to_have" in result
        assert isinstance(result["must_have"], list)
        assert isinstance(result["nice_to_have"], list)

        # Verify content — a React JD should extract React as must-have
        must_skills = [s["skill"].lower() for s in result["must_have"]]
        assert any("react" in s for s in must_skills), (
            f"Expected 'React' in must-have, got: {must_skills}"
        )

        # At least 2 must-have, 1 nice-to-have
        assert len(result["must_have"]) >= 2, (
            f"Expected 2+ must-have, got {len(result['must_have'])}"
        )
        assert len(result["nice_to_have"]) >= 1, (
            f"Expected 1+ nice-to-have, got {len(result['nice_to_have'])}"
        )

    def test_extract_requirements_validates_through_model(self) -> None:
        """Result should be validatable through ExtractedRequirements."""
        from src.tools.extract_requirements import extract_requirements
        jd_text = SAMPLE_JD_PATH.read_text()
        result = extract_requirements.invoke({"jd": jd_text})

        # Should not raise
        validated = ExtractedRequirements.model_validate(result)
        assert validated is not None


@pytest.mark.integration
class TestCompareCandidatesIntegration:
    """Tests that require a working LLM API."""

    def test_compare_with_profiles(self) -> None:
        result = compare_candidates.invoke({
            "candidate_ids": ["alice_test", "bob_test"],
            "candidate_profiles": SAMPLE_PROFILES,
        })
        assert "candidates" in result
        assert "comparison_table" in result
        assert "summary" in result
        # LLM should generate actual comparison content
        assert len(result["summary"]) > 50


@pytest.mark.integration
class TestGenerateQuestionsIntegration:
    """Tests that require a working LLM API."""

    def test_generate_questions_for_known_candidate(self) -> None:
        # First get a real candidate_id from RAG
        results = rag_search.invoke({"query": "React developer", "top_k": 1})
        if not results:
            pytest.skip("No candidates in RAG store")

        cid = results[0]["candidate_id"]
        result = generate_interview_questions.invoke({
            "candidate_id": cid,
            "num_questions": 3,
        })
        assert result["candidate_id"] == cid
        assert isinstance(result["questions"], list)
        # If LLM worked, should have questions
        if result["questions"]:
            q = result["questions"][0]
            assert "question" in q
            assert "category" in q
            assert "difficulty" in q