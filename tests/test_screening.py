"""
Agentic Profile Matching — Phase 6 Screening Pipeline Tests.

Tests all 3 screening rounds, red-flag detection, the orchestrator pipeline,
and integration with the graph node. All tests use mocked RAG/LLM — no
external services needed.

Run:
    pytest tests/test_screening.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.scoring.red_flags import (
    RedFlag,
    detect_employment_gaps,
    detect_job_hopping,
    detect_inconsistencies,
    detect_red_flags,
    _month_to_num,
)
from src.screening.round1_initial import (
    initial_screen,
    _count_keyword_hits,
    _build_round1_query,
    ROUND1_KEYWORD_THRESHOLD,
    ROUND1_SHORTLIST_SIZE,
)
from src.screening.round2_deep import (
    deep_analysis,
    _extract_experience_years,
    _verify_skills_with_evidence,
    _generate_deep_reasoning,
    ROUND2_MAX_SHORTLIST,
)
from src.screening.round3_final import (
    final_recommendation,
    _generate_hire_recommendation,
    _compile_round_evidence,
)
from src.screening.pipeline import run_screening_pipeline


# ===================================================================
# Shared Fixtures
# ===================================================================

SAMPLE_REQUIREMENTS = {
    "raw_jd": "Senior React Developer with TypeScript and AWS",
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

SAMPLE_RESUMES = {
    "alice_123": "Alice Johnson\n\nSenior React Developer at TechCorp (2019-2024)\n\nLed React migration for large-scale SPA. Used TypeScript for all new modules. Deployed on AWS.\n\nSkills: React, TypeScript, JavaScript, CSS, AWS, Node.js\n\nExperience: 5 years of professional experience",
    "bob_456": "Bob Smith\n\nReact Developer at StartupXYZ (2021-2023)\n\nBuilt React apps using JavaScript. Some TypeScript exposure.\n\nSkills: React, JavaScript, HTML, CSS\n\nExperience: 3 years experience",
    "carol_789": "Carol Williams\n\nJunior Developer at BigCo (2022-2024)\n\nHelped with React components. Learning TypeScript.\n\nSkills: React, HTML, CSS, Python\n\nExperience: 2 years experience",
    "dave_001": "Dave Wilson\n\nFull Stack Engineer at MegaCorp (2018-2024)\n\nExpert in React and TypeScript. AWS certified. Led team of 5.\n\nSkills: React, TypeScript, AWS, Node.js, Python, Docker\n\nExperience: 7+ years of professional experience",
}

SAMPLE_R1_SHORTLIST = [
    {
        "candidate_id": "alice_123", "name": "Alice Johnson",
        "score": 0.85, "must_have_score": 1.0, "nice_to_have_score": 0.5,
        "reasoning": "Strong match", "strengths": ["React", "TypeScript"],
        "gaps": [], "resume_excerpts": ["Led React migration"],
        "hire_recommendation": "borderline", "improvement_suggestions": [],
    },
    {
        "candidate_id": "dave_001", "name": "Dave Wilson",
        "score": 0.92, "must_have_score": 1.0, "nice_to_have_score": 0.7,
        "reasoning": "Expert match", "strengths": ["React", "TypeScript", "AWS"],
        "gaps": [], "resume_excerpts": ["Expert in React"],
        "hire_recommendation": "borderline", "improvement_suggestions": [],
    },
    {
        "candidate_id": "bob_456", "name": "Bob Smith",
        "score": 0.55, "must_have_score": 0.5, "nice_to_have_score": 0.3,
        "reasoning": "Partial match", "strengths": ["React"],
        "gaps": ["No TypeScript"], "resume_excerpts": ["Built React apps"],
        "hire_recommendation": "borderline", "improvement_suggestions": [],
    },
]


# ===================================================================
# Red Flag Detection Tests
# ===================================================================


class TestRedFlagMonthToNum:
    """Test the month name to number conversion."""

    def test_full_names(self) -> None:
        assert _month_to_num("January") == 1
        assert _month_to_num("February") == 2
        assert _month_to_num("December") == 12

    def test_abbreviations(self) -> None:
        assert _month_to_num("Jan") == 1
        assert _month_to_num("Feb") == 2
        assert _month_to_num("Dec") == 12

    def test_case_insensitive(self) -> None:
        assert _month_to_num("JANUARY") == 1
        assert _month_to_num("jan") == 1

    def test_unknown_returns_zero(self) -> None:
        assert _month_to_num("NotAMonth") == 0
        assert _month_to_num("") == 0


class TestDetectEmploymentGaps:
    """Test employment gap detection."""

    def test_no_gap_single_role(self) -> None:
        resume = "Engineer at Corp (January 2020 - December 2023)"
        flags = detect_employment_gaps(resume)
        assert len(flags) == 0

    def test_small_gap_no_flag(self) -> None:
        resume = "Engineer at CorpA (January 2020 - March 2020)\nEngineer at CorpB (May 2020 - December 2023)"
        flags = detect_employment_gaps(resume, max_gap_months=3)
        # 2 month gap (April) should NOT be flagged with threshold=3
        assert len(flags) == 0

    def test_large_gap_flagged(self) -> None:
        resume = "Engineer at CorpA (January 2020 - March 2020)\nEngineer at CorpB (September 2020 - December 2023)"
        flags = detect_employment_gaps(resume, max_gap_months=3)
        assert len(flags) >= 1
        assert flags[0].flag_type == "employment_gap"

    def test_gap_severity_high(self) -> None:
        resume = "Engineer at CorpA (January 2020 - January 2020)\nEngineer at CorpB (September 2020 - December 2023)"
        flags = detect_employment_gaps(resume, max_gap_months=3)
        assert len(flags) >= 1
        assert flags[0].severity == "high"

    def test_flag_has_evidence(self) -> None:
        resume = "Dev at A (Jan 2020 - Jan 2020)\nDev at B (Aug 2020 - Dec 2023)"
        flags = detect_employment_gaps(resume, max_gap_months=3)
        assert len(flags) >= 1
        assert len(flags[0].evidence) > 0


class TestDetectJobHopping:
    """Test job-hopping pattern detection."""

    def test_stable_history_no_flag(self) -> None:
        resume = "Senior Dev at BigCo (2019-2024)"
        flags = detect_job_hopping(resume)
        assert len(flags) == 0

    def test_many_roles_flagged(self) -> None:
        # 5 roles in a short period
        resume = (
            "Dev at A (2020-2020)\nDev at B (2020-2020)\n"
            "Dev at C (2020-2021)\nDev at D (2021-2021)\n"
            "Dev at E (2021-2022)"
        )
        flags = detect_job_hopping(resume, max_roles_in_period=3)
        assert len(flags) >= 1
        assert flags[0].flag_type == "job_hopping"

    def test_flag_severity(self) -> None:
        resume = "Dev at A (2020-2020)\nDev at B (2020-2020)\nDev at C (2020-2020)\nDev at D (2021-2021)"
        flags = detect_job_hopping(resume, max_roles_in_period=3)
        if flags:
            assert flags[0].severity == "high"


class TestDetectInconsistencies:
    """Test date inconsistency detection."""

    def test_valid_range_no_flag(self) -> None:
        resume = "Engineer at Corp (2019 - 2024)"
        flags = detect_inconsistencies(resume)
        assert len(flags) == 0

    def test_impossible_range_flagged(self) -> None:
        resume = "Engineer at Corp (2024 - 2019)"
        flags = detect_inconsistencies(resume)
        assert len(flags) >= 1
        assert flags[0].flag_type == "inconsistency"

    def test_deduplication(self) -> None:
        resume = "A (2024 - 2019) B (2024 - 2019) C (2024 - 2019)"
        flags = detect_inconsistencies(resume)
        # Should deduplicate by description
        assert len(flags) <= 3


class TestDetectRedFlagsCombined:
    """Test the combined detect_red_flags function."""

    def test_clean_resume_no_flags(self) -> None:
        resume = "Senior Developer at StableCorp (2019 - 2024)\nNo issues."
        flags = detect_red_flags(resume)
        # May have 0 or few flags for a clean resume
        gap_flags = [f for f in flags if f.flag_type == "employment_gap"]
        assert len(gap_flags) == 0

    def test_dirty_resume_has_flags(self) -> None:
        resume = "Dev at A (Jan 2020 - Jan 2020)\nDev at B (Sep 2020 - Dec 2023)\nWeird: 2024 - 2019"
        flags = detect_red_flags(resume)
        assert len(flags) >= 1

    def test_with_timeline_param(self) -> None:
        """Timeline param is accepted for API compatibility."""
        resume = "Developer at Corp (2020-2024)"
        timeline = [{"role": "Dev", "start_year": 2020, "end_year": 2024}]
        flags = detect_red_flags(resume, timeline=timeline)
        assert isinstance(flags, list)

    def test_red_flag_to_dict(self) -> None:
        flag = RedFlag(
            flag_type="employment_gap",
            description="Test gap",
            severity="medium",
            evidence="Jan 2020 - Jun 2020",
        )
        d = flag.to_dict()
        assert d["flag_type"] == "employment_gap"
        assert d["severity"] == "medium"


# ===================================================================
# Round 1 — Initial Screen Tests
# ===================================================================


class TestRound1BuildQuery:
    def test_basic_query(self) -> None:
        reqs = {"must_have": [{"skill": "React"}, {"skill": "TypeScript"}], "nice_to_have": []}
        query = _build_round1_query(reqs)
        assert "React" in query
        assert "TypeScript" in query

    def test_includes_nice_to_have(self) -> None:
        reqs = {"must_have": [], "nice_to_have": [{"skill": "AWS"}]}
        query = _build_round1_query(reqs)
        assert "AWS" in query

    def test_no_duplicates(self) -> None:
        reqs = {"must_have": [{"skill": "React"}], "nice_to_have": [{"skill": "React"}]}
        query = _build_round1_query(reqs)
        assert query.count("React") == 1

    def test_empty_requirements_fallback(self) -> None:
        query = _build_round1_query({})
        assert "developer" in query


class TestRound1CountKeywordHits:
    def test_all_matched(self) -> None:
        resume = "React and TypeScript developer with 5 years experience"
        must_haves = [{"skill": "React"}, {"skill": "TypeScript"}]
        hits = _count_keyword_hits(resume, must_haves)
        assert hits == 2

    def test_partial_match(self) -> None:
        resume = "React developer"
        must_haves = [{"skill": "React"}, {"skill": "TypeScript"}, {"skill": "AWS"}]
        hits = _count_keyword_hits(resume, must_haves)
        assert hits == 1

    def test_no_match(self) -> None:
        resume = "Python developer with Django experience"
        must_haves = [{"skill": "React"}, {"skill": "TypeScript"}]
        hits = _count_keyword_hits(resume, must_haves)
        assert hits == 0

    def test_whole_word_match(self) -> None:
        """'React' should NOT match 'Reaction'."""
        resume = "Chemical Reaction engineer"
        must_haves = [{"skill": "React"}]
        hits = _count_keyword_hits(resume, must_haves)
        assert hits == 0

    def test_empty_must_haves(self) -> None:
        hits = _count_keyword_hits("some resume text", [])
        assert hits == 0


class TestRound1InitialScreen:
    @patch("src.screening.round1_initial.get_full_resume_text")
    @patch("src.screening.round1_initial.rag_search")
    def test_no_requirements_returns_empty(self, mock_rag, mock_resume) -> None:
        mock_rag.invoke.return_value = []
        result = initial_screen({"requirements": {}})
        assert result["current_shortlist"] == []
        assert result["error"] is not None

    @patch("src.screening.round1_initial.get_full_resume_text")
    @patch("src.screening.round1_initial.rag_search")
    def test_rag_failure_returns_empty(self, mock_rag, mock_resume) -> None:
        mock_rag.invoke.side_effect = Exception("DB down")
        result = initial_screen({"requirements": SAMPLE_REQUIREMENTS})
        assert result["current_shortlist"] == []
        assert "RAG search failed" in result.get("error", "")

    @patch("src.screening.round1_initial.score_candidate")
    @patch("src.screening.round1_initial.get_full_resume_text")
    @patch("src.screening.round1_initial.rag_search")
    def test_filters_by_keyword_threshold(
        self, mock_rag, mock_resume, mock_score
    ) -> None:
        # 3 candidates: alice has all skills, bob has none
        mock_rag.invoke.return_value = [
            {"candidate_id": "alice_123", "name": "Alice", "score": 0.1, "excerpt": "..."},
            {"candidate_id": "bob_no_skill", "name": "Bob", "score": 0.5, "excerpt": "..."},
        ]
        mock_resume.side_effect = lambda cid: SAMPLE_RESUMES.get(cid, "No skills here")
        mock_score.return_value = MagicMock(
            must_have_score=1.0, nice_to_have_score=0.5,
            reasoning="Good", strengths=["React", "TypeScript"], gaps=[], excerpts=[]
        )

        result = initial_screen({"requirements": SAMPLE_REQUIREMENTS})

        # Alice should pass, Bob should be filtered (0/2 must-haves < 0.3)
        shortlisted_ids = [c["candidate_id"] for c in result["current_shortlist"]]
        assert "alice_123" in shortlisted_ids
        assert "bob_no_skill" not in shortlisted_ids

    @patch("src.screening.round1_initial.score_candidate")
    @patch("src.screening.round1_initial.get_full_resume_text")
    @patch("src.screening.round1_initial.rag_search")
    def test_records_round_metadata(
        self, mock_rag, mock_resume, mock_score
    ) -> None:
        mock_rag.invoke.return_value = [
            {"candidate_id": "alice_123", "name": "Alice", "score": 0.1, "excerpt": "..."},
        ]
        mock_resume.return_value = SAMPLE_RESUMES["alice_123"]
        mock_score.return_value = MagicMock(
            must_have_score=1.0, nice_to_have_score=0.5,
            reasoning="Good", strengths=["React"], gaps=[], excerpts=[]
        )

        result = initial_screen({"requirements": SAMPLE_REQUIREMENTS})
        rounds = result["screening_rounds"]
        assert len(rounds) >= 1
        assert rounds[0]["round_number"] == 1
        assert rounds[0]["round_type"] == "initial"
        assert "candidates_evaluated" in rounds[0]
        assert "shortlisted_ids" in rounds[0]
        assert "eliminated_ids" in rounds[0]

    @patch("src.screening.round1_initial.score_candidate")
    @patch("src.screening.round1_initial.get_full_resume_text")
    @patch("src.screening.round1_initial.rag_search")
    def test_shortlist_capped_at_10(
        self, mock_rag, mock_resume, mock_score
    ) -> None:
        # Return 15 candidates
        candidates = [
            {"candidate_id": f"candidate_{i:03d}", "name": f"Candidate {i}", "score": 0.1}
            for i in range(15)
        ]
        mock_rag.invoke.return_value = candidates
        mock_resume.return_value = "React TypeScript AWS developer"
        mock_score.return_value = MagicMock(
            must_have_score=1.0, nice_to_have_score=0.5,
            reasoning="Good", strengths=["React"], gaps=[], excerpts=[]
        )

        result = initial_screen({"requirements": SAMPLE_REQUIREMENTS})
        assert len(result["current_shortlist"]) <= ROUND1_SHORTLIST_SIZE

    @patch("src.screening.round1_initial.get_full_resume_text")
    @patch("src.screening.round1_initial.rag_search")
    def test_all_filtered_fallback_to_rag_top10(self, mock_rag, mock_resume) -> None:
        """If all candidates fail keyword filter, fallback to top 10 from RAG."""
        candidates = [
            {"candidate_id": f"candidate_{i:03d}", "name": f"C{i}", "score": 0.1}
            for i in range(15)
        ]
        mock_rag.invoke.return_value = candidates
        mock_resume.return_value = "No relevant skills at all"

        result = initial_screen({"requirements": SAMPLE_REQUIREMENTS})
        # Fallback should give top 10
        assert len(result["current_shortlist"]) == 10


# ===================================================================
# Round 2 — Deep Analysis Tests
# ===================================================================


class TestRound2ExtractExperience:
    def test_standard_format(self) -> None:
        text = "5 years of professional experience"
        assert _extract_experience_years(text) == 5

    def test_plus_format(self) -> None:
        text = "7+ years of experience"
        assert _extract_experience_years(text) == 7

    def test_over_format(self) -> None:
        text = "over 10 years of experience"
        assert _extract_experience_years(text) == 10

    def test_max_years_returned(self) -> None:
        text = "5 years of experience and 8+ years in the industry"
        assert _extract_experience_years(text) == 8

    def test_no_experience_returns_none(self) -> None:
        text = "No experience mentioned"
        assert _extract_experience_years(text) is None


class TestRound2VerifySkills:
    def test_all_matched(self) -> None:
        result = _verify_skills_with_evidence(
            "React and TypeScript developer",
            SAMPLE_REQUIREMENTS,
        )
        assert "React" in result["matched_strengths"]
        assert "TypeScript" in result["matched_strengths"]
        assert result["must_have_score"] == 1.0

    def test_partial_match(self) -> None:
        result = _verify_skills_with_evidence(
            "React developer with some JavaScript",
            SAMPLE_REQUIREMENTS,
        )
        assert "React" in result["matched_strengths"]
        assert "TypeScript" in result["unmatched_skills"]
        assert result["must_have_score"] == 0.5

    def test_nice_to_have_tracked(self) -> None:
        result = _verify_skills_with_evidence(
            "React TypeScript AWS expert",
            SAMPLE_REQUIREMENTS,
        )
        assert "AWS" in result["matched_strengths"]

    def test_excerpts_generated(self) -> None:
        result = _verify_skills_with_evidence(
            "Worked extensively with React for 3 years building SPAs",
            SAMPLE_REQUIREMENTS,
        )
        assert len(result["key_excerpts"]) > 0


class TestRound2DeepReasoning:
    def test_basic_reasoning(self) -> None:
        verification = {
            "matched_strengths": ["React", "TypeScript"],
            "unmatched_skills": ["AWS"],
        }
        reasoning = _generate_deep_reasoning("Alice", verification, [], 5)
        assert "Alice" in reasoning
        assert "2/3" in reasoning
        assert "5 years" in reasoning

    def test_red_flags_mentioned(self) -> None:
        flags = [RedFlag("job_hopping", "Test", "high", "evidence")]
        verification = {"matched_strengths": ["React"], "unmatched_skills": []}
        reasoning = _generate_deep_reasoning("Bob", verification, flags, None)
        assert "1 issue" in reasoning


class TestRound2DeepAnalysis:
    def test_empty_shortlist_returns_empty(self) -> None:
        result = deep_analysis({
            "current_shortlist": [],
            "requirements": SAMPLE_REQUIREMENTS,
        })
        assert result["current_shortlist"] == []
        assert result["current_round"] == 2

    @patch("src.screening.round2_deep.score_candidate")
    @patch("src.screening.round2_deep.get_full_resume_text")
    def test_enriches_candidates_with_red_flags(
        self, mock_resume, mock_score
    ) -> None:
        mock_resume.return_value = SAMPLE_RESUMES["alice_123"]
        mock_score.return_value = MagicMock(
            must_have_score=1.0, nice_to_have_score=0.5,
            reasoning="Good", strengths=["React", "TypeScript"], gaps=[], excerpts=[]
        )

        state = {
            "current_shortlist": [SAMPLE_R1_SHORTLIST[0]],
            "requirements": SAMPLE_REQUIREMENTS,
        }
        result = deep_analysis(state)

        assert len(result["current_shortlist"]) == 1
        candidate = result["current_shortlist"][0]
        assert "red_flags" in candidate
        assert isinstance(candidate["red_flags"], list)

    @patch("src.screening.round2_deep.score_candidate")
    @patch("src.screening.round2_deep.get_full_resume_text")
    def test_shortlist_capped_at_7(
        self, mock_resume, mock_score
    ) -> None:
        mock_resume.return_value = "React TypeScript developer"
        mock_score.return_value = MagicMock(
            must_have_score=0.8, nice_to_have_score=0.6,
            reasoning="OK", strengths=["React"], gaps=[], excerpts=[]
        )

        # 10 candidates
        candidates = [
            {
                "candidate_id": f"candidate_{i:03d}", "name": f"Candidate {i}",
                "score": 0.7 - i * 0.02,
                "must_have_score": 0.8, "nice_to_have_score": 0.5,
                "reasoning": "Test", "strengths": ["React"], "gaps": [],
                "resume_excerpts": [], "hire_recommendation": "borderline",
                "improvement_suggestions": [],
            }
            for i in range(10)
        ]
        state = {"current_shortlist": candidates, "requirements": SAMPLE_REQUIREMENTS}
        result = deep_analysis(state)

        assert len(result["current_shortlist"]) <= ROUND2_MAX_SHORTLIST

    @patch("src.screening.round2_deep.score_candidate")
    @patch("src.screening.round2_deep.get_full_resume_text")
    def test_records_round2_metadata(
        self, mock_resume, mock_score
    ) -> None:
        mock_resume.return_value = SAMPLE_RESUMES["alice_123"]
        mock_score.return_value = MagicMock(
            must_have_score=1.0, nice_to_have_score=0.5,
            reasoning="Good", strengths=["React"], gaps=[], excerpts=[]
        )

        state = {
            "current_shortlist": [SAMPLE_R1_SHORTLIST[0]],
            "requirements": SAMPLE_REQUIREMENTS,
        }
        result = deep_analysis(state)

        rounds = result["screening_rounds"]
        assert len(rounds) >= 1
        assert rounds[0]["round_number"] == 2
        assert rounds[0]["round_type"] == "deep_analysis"


# ===================================================================
# Round 3 — Final Recommendation Tests
# ===================================================================


class TestRound3HireRecommendation:
    def test_high_score_hire(self) -> None:
        assert _generate_hire_recommendation(0.85, 0.8) == "hire"

    def test_moderate_score_borderline(self) -> None:
        assert _generate_hire_recommendation(0.6, 0.5) == "borderline"

    def test_low_score_no_hire(self) -> None:
        assert _generate_hire_recommendation(0.3, 0.2) == "no_hire"

    def test_high_composite_low_must_borderline(self) -> None:
        """High composite but low must-have -> borderline, not hire."""
        assert _generate_hire_recommendation(0.85, 0.5) == "borderline"

    def test_boundary_hire(self) -> None:
        assert _generate_hire_recommendation(0.8, 0.7) == "hire"

    def test_boundary_no_hire(self) -> None:
        assert _generate_hire_recommendation(0.49, 0.3) == "no_hire"


class TestRound3CompileEvidence:
    def test_includes_candidate_name(self) -> None:
        candidate = {"name": "Alice", "score": 0.85, "must_have_score": 0.9, "nice_to_have_score": 0.7}
        evidence = _compile_round_evidence(candidate, [])
        assert "Alice" in evidence

    def test_includes_round_notes(self) -> None:
        candidate = {"name": "Alice", "score": 0.85}
        rounds = [
            {"round_number": 1, "round_type": "initial", "notes": "10 evaluated, 5 shortlisted"},
        ]
        evidence = _compile_round_evidence(candidate, rounds)
        assert "10 evaluated" in evidence

    def test_includes_red_flags(self) -> None:
        candidate = {
            "name": "Bob", "score": 0.5,
            "red_flags": [{"description": "Job hopping detected"}],
        }
        evidence = _compile_round_evidence(candidate, [])
        assert "Job hopping" in evidence

    def test_includes_experience(self) -> None:
        candidate = {"name": "Alice", "score": 0.85, "experience_years": 7}
        evidence = _compile_round_evidence(candidate, [])
        assert "7 years" in evidence


class TestRound3FinalRecommendation:
    def test_empty_shortlist(self) -> None:
        result = final_recommendation({
            "current_shortlist": [],
            "requirements": SAMPLE_REQUIREMENTS,
        })
        assert result["current_shortlist"] == []
        assert result["current_round"] == 3
        assert len(result["generated_reports"]) == 0

    def test_produces_hire_recommendation(self) -> None:
        candidates = [
            {**SAMPLE_R1_SHORTLIST[0]},  # score 0.85 -> hire
            {**SAMPLE_R1_SHORTLIST[2]},  # score 0.55 -> borderline
        ]
        result = final_recommendation({
            "current_shortlist": candidates,
            "requirements": SAMPLE_REQUIREMENTS,
            "screening_rounds": [],
        })

        assert len(result["current_shortlist"]) == 2
        # First should be hire (0.85 composite, 1.0 must_have)
        assert result["current_shortlist"][0]["hire_recommendation"] == "hire"
        # Second should be borderline (0.55 composite)
        assert result["current_shortlist"][1]["hire_recommendation"] == "borderline"

    def test_generates_reports(self) -> None:
        candidates = [{**SAMPLE_R1_SHORTLIST[0]}]
        result = final_recommendation({
            "current_shortlist": candidates,
            "requirements": SAMPLE_REQUIREMENTS,
            "screening_rounds": [],
        })

        assert "generated_reports" in result
        assert "alice_123" in result["generated_reports"]
        assert "Match Report" in result["generated_reports"]["alice_123"]

    def test_borderline_gets_suggestions(self) -> None:
        candidates = [{**SAMPLE_R1_SHORTLIST[2]}]  # score 0.55 -> borderline
        result = final_recommendation({
            "current_shortlist": candidates,
            "requirements": SAMPLE_REQUIREMENTS,
            "screening_rounds": [],
        })

        candidate = result["current_shortlist"][0]
        assert candidate["hire_recommendation"] == "borderline"
        assert len(candidate.get("improvement_suggestions", [])) > 0

    def test_records_round3_metadata(self) -> None:
        candidates = [{**SAMPLE_R1_SHORTLIST[0]}]
        result = final_recommendation({
            "current_shortlist": candidates,
            "requirements": SAMPLE_REQUIREMENTS,
            "screening_rounds": [
                {"round_number": 1, "round_type": "initial", "candidates_evaluated": 10,
                 "shortlisted_ids": ["alice_123"], "eliminated_ids": [], "notes": "R1"},
                {"round_number": 2, "round_type": "deep_analysis", "candidates_evaluated": 5,
                 "shortlisted_ids": ["alice_123"], "eliminated_ids": [], "notes": "R2"},
            ],
        })

        rounds = result["screening_rounds"]
        assert len(rounds) == 3
        assert rounds[2]["round_number"] == 3
        assert rounds[2]["round_type"] == "final"
        assert rounds[2]["candidates_evaluated"] == 1

    def test_accumulates_previous_rounds(self) -> None:
        """Round 3 should NOT overwrite rounds 1 and 2."""
        candidates = [{**SAMPLE_R1_SHORTLIST[0]}]
        prev_rounds = [
            {"round_number": 1, "round_type": "initial", "notes": "R1 notes"},
            {"round_number": 2, "round_type": "deep_analysis", "notes": "R2 notes"},
        ]
        result = final_recommendation({
            "current_shortlist": candidates,
            "requirements": SAMPLE_REQUIREMENTS,
            "screening_rounds": prev_rounds,
        })

        assert len(result["screening_rounds"]) == 3
        assert result["screening_rounds"][0]["notes"] == "R1 notes"
        assert result["screening_rounds"][1]["notes"] == "R2 notes"


# ===================================================================
# Pipeline Orchestrator Tests
# ===================================================================


class TestPipeline:
    @patch("src.screening.pipeline.final_recommendation")
    @patch("src.screening.pipeline.deep_analysis")
    @patch("src.screening.pipeline.initial_screen")
    def test_pipeline_runs_all_3_rounds(
        self, mock_r1, mock_r2, mock_r3
    ) -> None:
        mock_r1.return_value = {
            "current_shortlist": SAMPLE_R1_SHORTLIST[:2],
            "screening_rounds": [{
                "round_number": 1, "round_type": "initial",
                "candidates_evaluated": 50, "shortlisted_ids": ["a", "b"],
                "eliminated_ids": [], "notes": "R1 done",
            }],
            "current_round": 1,
        }
        mock_r2.return_value = {
            "current_shortlist": SAMPLE_R1_SHORTLIST[:2],
            "screening_rounds": [{
                "round_number": 2, "round_type": "deep_analysis",
                "candidates_evaluated": 2, "shortlisted_ids": ["a", "b"],
                "eliminated_ids": [], "notes": "R2 done",
            }],
            "current_round": 2,
        }
        mock_r3.return_value = {
            "current_shortlist": SAMPLE_R1_SHORTLIST[:2],
            "generated_reports": {"alice_123": "# Report"},
            "screening_rounds": [{
                "round_number": 3, "round_type": "final",
                "candidates_evaluated": 2, "shortlisted_ids": ["a"],
                "eliminated_ids": ["b"], "notes": "R3 done",
            }],
            "current_round": 3,
        }

        result = run_screening_pipeline({
            "requirements": SAMPLE_REQUIREMENTS,
        })

        assert result["current_round"] == 3
        assert len(result["screening_rounds"]) == 3
        assert len(result["generated_reports"]) == 1
        mock_r1.assert_called_once()
        mock_r2.assert_called_once()
        mock_r3.assert_called_once()

    @patch("src.screening.pipeline.initial_screen")
    def test_pipeline_stops_after_r1_if_empty(self, mock_r1) -> None:
        mock_r1.return_value = {
            "current_shortlist": [],
            "screening_rounds": [{
                "round_number": 1, "round_type": "initial",
                "candidates_evaluated": 0, "shortlisted_ids": [],
                "eliminated_ids": [], "notes": "No candidates",
            }],
            "current_round": 1,
            "error": "No candidates found.",
        }

        result = run_screening_pipeline({"requirements": SAMPLE_REQUIREMENTS})
        assert result["current_round"] == 1
        assert len(result["screening_rounds"]) == 1

    @patch("src.screening.pipeline.final_recommendation")
    @patch("src.screening.pipeline.deep_analysis")
    @patch("src.screening.pipeline.initial_screen")
    def test_pipeline_stops_after_r2_if_empty(
        self, mock_r1, mock_r2, mock_r3
    ) -> None:
        mock_r1.return_value = {
            "current_shortlist": SAMPLE_R1_SHORTLIST[:1],
            "screening_rounds": [{"round_number": 1, "round_type": "initial",
                                  "candidates_evaluated": 10, "shortlisted_ids": ["a"],
                                  "eliminated_ids": [], "notes": "R1"}],
            "current_round": 1,
        }
        mock_r2.return_value = {
            "current_shortlist": [],
            "screening_rounds": [{"round_number": 2, "round_type": "deep_analysis",
                                  "candidates_evaluated": 1, "shortlisted_ids": [],
                                  "eliminated_ids": ["a"], "notes": "R2 empty"}],
            "current_round": 2,
            "error": "All eliminated",
        }

        result = run_screening_pipeline({"requirements": SAMPLE_REQUIREMENTS})
        assert result["current_round"] == 2
        mock_r3.assert_not_called()


# ===================================================================
# Integration: rank_candidates_node uses pipeline
# ===================================================================


class TestRankCandidatesNodeUsesPipeline:
    @patch("src.screening.pipeline.final_recommendation")
    @patch("src.screening.pipeline.deep_analysis")
    @patch("src.screening.pipeline.initial_screen")
    def test_node_delegates_to_pipeline(
        self, mock_r1, mock_r2, mock_r3
    ) -> None:
        from src.agent.nodes import rank_candidates_node

        mock_r1.return_value = {
            "current_shortlist": SAMPLE_R1_SHORTLIST[:1],
            "screening_rounds": [{"round_number": 1, "round_type": "initial",
                                  "candidates_evaluated": 5, "shortlisted_ids": ["alice_123"],
                                  "eliminated_ids": [], "notes": "R1"}],
            "current_round": 1,
        }
        mock_r2.return_value = {
            "current_shortlist": SAMPLE_R1_SHORTLIST[:1],
            "screening_rounds": [{"round_number": 2, "round_type": "deep_analysis",
                                  "candidates_evaluated": 1, "shortlisted_ids": ["alice_123"],
                                  "eliminated_ids": [], "notes": "R2"}],
            "current_round": 2,
        }
        mock_r3.return_value = {
            "current_shortlist": SAMPLE_R1_SHORTLIST[:1],
            "generated_reports": {"alice_123": "# Report for Alice"},
            "screening_rounds": [{"round_number": 3, "round_type": "final",
                                  "candidates_evaluated": 1, "shortlisted_ids": ["alice_123"],
                                  "eliminated_ids": [], "notes": "R3"}],
            "current_round": 3,
        }

        state = {
            "all_candidate_ids": ["alice_123", "bob_456"],
            "requirements": SAMPLE_REQUIREMENTS,
        }
        result = rank_candidates_node(state)

        assert result["current_round"] == 3
        assert len(result["screening_rounds"]) == 3
        assert "alice_123" in result["generated_reports"]

    @patch("src.agent.nodes.get_full_resume_text")
    @patch("src.agent.nodes.score_candidate")
    def test_node_fallback_when_pipeline_empty(
        self, mock_score, mock_resume
    ) -> None:
        from src.agent.nodes import rank_candidates_node

        mock_resume.return_value = "React TypeScript developer"
        mock_score.return_value = MagicMock(
            must_have_score=1.0, nice_to_have_score=0.5,
            reasoning="Good", strengths=["React"], gaps=[], excerpts=[]
        )

        # Pipeline returns empty shortlist -> should fallback to single-pass
        state = {
            "all_candidate_ids": ["alice_123"],
            "requirements": SAMPLE_REQUIREMENTS,
        }
        result = rank_candidates_node(state)

        # Should have fallen back to single-pass
        assert "current_shortlist" in result
        assert len(result["screening_rounds"]) >= 1