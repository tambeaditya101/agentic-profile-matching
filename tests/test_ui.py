"""
Agentic Profile Matching — Phase 7 UI Smoke Tests.

Tests the shared UI components (no Streamlit runtime required) and
verifies the CLI entrypoint can be invoked via Click's test runner.

Run:
    pytest tests/test_ui.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from ui import cli_app
from ui.components import (
    SUGGESTED_PROMPTS,
    build_agent_response,
    colorize,
    export_reports_to_dir,
    format_requirements_panel,
    format_screening_progress,
    format_shortlist_table,
    get_suggested_prompts,
    render_comparison_table,
    render_explanation,
    render_match_report,
    render_questions,
    render_ranking_delta,
    recommendation_label,
    score_status_label,
)


# =====================================================================
# Fixtures
# =====================================================================

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
        "candidate_id": "alice_123",
        "name": "Alice Johnson",
        "score": 0.85,
        "must_have_score": 0.9,
        "nice_to_have_score": 0.7,
        "hire_recommendation": "hire",
    },
    {
        "candidate_id": "bob_456",
        "name": "Bob Smith",
        "score": 0.65,
        "must_have_score": 0.7,
        "nice_to_have_score": 0.5,
        "hire_recommendation": "borderline",
    },
]

SAMPLE_ROUNDS = [
    {
        "round_number": 1,
        "round_type": "initial",
        "candidates_evaluated": 100,
        "shortlisted_ids": ["a", "b", "c"],
    },
    {
        "round_number": 2,
        "round_type": "deep_analysis",
        "candidates_evaluated": 10,
        "shortlisted_ids": ["a", "b"],
    },
]


# =====================================================================
# Status helpers
# =====================================================================


class TestStatusHelpers:
    def test_score_status_label_thresholds(self) -> None:
        assert score_status_label(0.85) == "STRONG MATCH"
        assert score_status_label(0.6) == "GOOD MATCH"
        assert score_status_label(0.4) == "PARTIAL"
        assert score_status_label(0.2) == "WEAK"
        assert score_status_label(0.0) == "NO MATCH"

    def test_recommendation_label_known(self) -> None:
        assert recommendation_label("hire") == "STRONG HIRE"
        assert recommendation_label("no_hire") == "NO HIRE"
        assert recommendation_label("borderline") == "BORDERLINE"
        assert recommendation_label("unknown") == "PENDING"

    def test_recommendation_label_unknown_falls_back(self) -> None:
        assert recommendation_label("custom_status") == "CUSTOM_STATUS"
        assert recommendation_label("") == "PENDING"


# =====================================================================
# Color helper
# =====================================================================


class TestColorize:
    def test_known_color_wraps_with_ansi(self) -> None:
        result = colorize("hello", "red")
        assert result.startswith("\033[31m")
        assert result.endswith("\033[0m")
        assert "hello" in result

    def test_unknown_color_is_passthrough(self) -> None:
        assert colorize("hello", "nonexistent") == "hello"


# =====================================================================
# Requirements panel
# =====================================================================


class TestRequirementsPanel:
    def test_empty_requirements_shows_hint(self) -> None:
        out = format_requirements_panel(None)
        assert "No requirements" in out

    def test_full_requirements_lists_must_and_nice(self) -> None:
        out = format_requirements_panel(SAMPLE_REQUIREMENTS)
        assert "React" in out
        assert "TypeScript" in out
        assert "AWS" in out
        assert "5+ years" in out
        assert "frontend" in out

    def test_no_must_have_shows_none(self) -> None:
        out = format_requirements_panel({"must_have": [], "nice_to_have": []})
        assert "_none_" in out


# =====================================================================
# Screening progress
# =====================================================================


class TestScreeningProgress:
    def test_no_rounds_shows_all_pending(self) -> None:
        out = format_screening_progress(None)
        assert "○" in out
        assert "pending" in out
        assert "Round 1" in out
        assert "Round 2" in out
        assert "Round 3" in out

    def test_partial_rounds(self) -> None:
        out = format_screening_progress(SAMPLE_ROUNDS)
        # Round 1 and 2 done
        assert "✓" in out
        # Round 3 still pending
        assert "○" in out
        assert "pending" in out
        assert "100 evaluated" in out
        assert "10 evaluated" in out


# =====================================================================
# Shortlist table
# =====================================================================


class TestShortlistTable:
    def test_empty_shortlist(self) -> None:
        out = format_shortlist_table(None)
        assert "No candidates" in out

    def test_shortlist_renders_table(self) -> None:
        out = format_shortlist_table(SAMPLE_SHORTLIST)
        assert "Alice Johnson" in out
        assert "Bob Smith" in out
        assert "STRONG HIRE" in out
        assert "BORDERLINE" in out

    def test_shortlist_top_n_truncates(self) -> None:
        out = format_shortlist_table(SAMPLE_SHORTLIST, top_n=1)
        assert "Alice Johnson" in out
        assert "Bob Smith" not in out


# =====================================================================
# Comparison table
# =====================================================================


class TestComparisonTable:
    def test_empty_comparison(self) -> None:
        out = render_comparison_table(None)
        assert "No comparison" in out

    def test_full_comparison_with_list_table(self) -> None:
        result = {
            "type": "comparison",
            "candidates": [{"name": "Alice"}, {"name": "Bob"}],
            "comparison_table": {
                "Overall Score": ["0.85", "0.65"],
                "Must-Have": ["0.9", "0.7"],
            },
            "summary": "Alice leads.",
        }
        out = render_comparison_table(result)
        assert "Comparing Alice, Bob" in out
        assert "Alice" in out
        assert "Bob" in out
        assert "0.85" in out
        assert "Alice leads." in out

    def test_single_candidate_error(self) -> None:
        result = {"type": "single_candidate", "summary": "Only one candidate."}
        out = render_comparison_table(result)
        assert "Only one candidate." in out

    def test_error_type(self) -> None:
        result = {"type": "error", "summary": "Could not identify."}
        out = render_comparison_table(result)
        assert "Could not identify." in out


# =====================================================================
# Questions / Explanation / Delta
# =====================================================================


class TestQuestionsRenderer:
    def test_empty_questions(self) -> None:
        out = render_questions(None)
        assert "No questions" in out

    def test_questions_with_dicts(self) -> None:
        result = {
            "type": "questions",
            "candidate_name": "Alice",
            "questions": [
                {
                    "question": "Walk me through your React migration.",
                    "category": "technical",
                    "difficulty": "medium",
                    "targets_gap": "React leadership",
                    "follow_ups": ["What challenges?"],
                },
                {
                    "question": "How do you mentor juniors?",
                    "category": "behavioral",
                    "difficulty": "easy",
                },
            ],
        }
        out = render_questions(result)
        assert "Interview Questions for Alice" in out
        assert "Walk me through" in out
        assert "[TECHNICAL / medium]" in out
        assert "Follow-ups:" in out
        assert "What challenges?" in out

    def test_questions_fallback_to_summary(self) -> None:
        result = {
            "type": "questions",
            "candidate_name": "Alice",
            "questions": [],
            "summary": "Could not generate.",
        }
        out = render_questions(result)
        assert "Could not generate." in out


class TestExplanationRenderer:
    def test_full_explanation(self) -> None:
        result = {
            "type": "explanation",
            "higher_candidate": "Alice",
            "lower_candidate": "Bob",
            "summary": "Alice has more experience.",
        }
        out = render_explanation(result)
        assert "Why Alice ranked higher than Bob" in out
        assert "Alice has more experience." in out

    def test_empty_explanation(self) -> None:
        out = render_explanation(None)
        assert "No explanation" in out


class TestRankingDelta:
    def test_delta_renders_summary(self) -> None:
        result = {
            "type": "refinement_delta",
            "summary": "Removed AWS. Bob moved up 2 spots.",
        }
        out = render_ranking_delta(result)
        assert "Ranking Updated" in out
        assert "Removed AWS" in out

    def test_empty_delta(self) -> None:
        out = render_ranking_delta(None)
        assert "No refinement" in out


# =====================================================================
# Match report
# =====================================================================


class TestMatchReport:
    def test_render_passthrough(self) -> None:
        md = "# Report\n\nSome content."
        assert render_match_report(md) == md

    def test_render_empty(self) -> None:
        assert "No report" in render_match_report(None)
        assert "No report" in render_match_report("")


# =====================================================================
# build_agent_response (state → markdown)
# =====================================================================


class TestBuildAgentResponse:
    def test_error_state(self) -> None:
        out = build_agent_response({"error": "Boom"})
        assert "Error" in out
        assert "Boom" in out

    def test_comparison_state(self) -> None:
        state = {
            "comparison_result": {
                "type": "explanation",
                "higher_candidate": "A",
                "lower_candidate": "B",
                "summary": "A is better.",
            }
        }
        out = build_agent_response(state)
        assert "Why A ranked higher than B" in out

    def test_shortlist_state(self) -> None:
        state = {
            "current_shortlist": SAMPLE_SHORTLIST,
            "awaiting_human_feedback": True,
            "screening_rounds": SAMPLE_ROUNDS,
            "generated_reports": {"alice_123": "# Alice report"},
        }
        out = build_agent_response(state)
        assert "Pipeline complete" in out
        assert "Alice Johnson" in out

    def test_empty_state(self) -> None:
        out = build_agent_response({})
        assert "No response" in out


# =====================================================================
# Suggested prompts
# =====================================================================


class TestSuggestedPrompts:
    def test_get_suggested_prompts(self) -> None:
        prompts = get_suggested_prompts()
        assert len(prompts) >= 5
        assert all("label" in p and "message" in p for p in prompts)

    def test_constant_matches_function(self) -> None:
        # The exported list and the function should agree
        assert SUGGESTED_PROMPTS == get_suggested_prompts()


# =====================================================================
# Export reports
# =====================================================================


class TestExportReports:
    def test_no_reports_returns_empty(self, tmp_path) -> None:
        result = export_reports_to_dir({}, str(tmp_path))
        assert result == []

    def test_writes_files(self, tmp_path) -> None:
        state = {
            "generated_reports": {
                "alice_123": "# Alice report\n\nbody",
                "bob_456": "# Bob report",
            },
            "current_shortlist": [
                {"candidate_id": "alice_123", "name": "Alice Johnson"},
                {"candidate_id": "bob_456", "name": "Bob Smith"},
            ],
        }
        written = export_reports_to_dir(state, str(tmp_path))
        assert len(written) == 2
        for p in written:
            assert p.endswith(".md")
        # All paths exist
        from pathlib import Path

        for p in written:
            assert Path(p).exists()

    def test_creates_output_dir(self, tmp_path) -> None:
        out = tmp_path / "nested" / "reports"
        state = {
            "generated_reports": {"x": "# X"},
            "current_shortlist": [],
        }
        written = export_reports_to_dir(state, str(out))
        assert len(written) == 1
        assert out.exists()


# =====================================================================
# CLI smoke tests (via Click's CliRunner)
# =====================================================================


class TestCliEntrypoint:
    def test_help_works(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_app.cli, ["--help"])
        assert result.exit_code == 0
        assert "Agentic Profile Matching" in result.output
        assert "start" in result.output
        assert "info" in result.output

    def test_version_flag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_app.cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_start_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_app.cli, ["start", "--help"])
        assert result.exit_code == 0
        assert "--jd" in result.output
        assert "--jd-text" in result.output

    @patch("ui.cli_app.create_graph", create=True)
    def test_start_with_jd_text_runs_pipeline(self, mock_create_graph) -> None:
        """Verify --jd-text invokes the linear pipeline (graph mocked)."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "requirements": SAMPLE_REQUIREMENTS,
            "current_shortlist": SAMPLE_SHORTLIST,
            "screening_rounds": SAMPLE_ROUNDS,
            "generated_reports": {"alice_123": "# Alice"},
            "awaiting_human_feedback": True,
        }
        # Patch the lazy import inside run_repl by injecting into sys.modules
        import sys

        fake_module = MagicMock()
        fake_module.create_graph.return_value = mock_graph
        original = sys.modules.get("src.agent.graph")
        sys.modules["src.agent.graph"] = fake_module
        try:
            runner = CliRunner()
            # Provide 'done' as stdin to exit the REPL immediately after the pipeline runs
            result = runner.invoke(
                cli_app.cli,
                ["start", "--jd-text", "Senior React Developer with 5+ years experience."],
                input="done\n",
            )
            assert result.exit_code == 0, f"CLI failed: {result.output}"
            assert "Pipeline complete" in result.output or "Alice Johnson" in result.output
        finally:
            if original is not None:
                sys.modules["src.agent.graph"] = original
            else:
                sys.modules.pop("src.agent.graph", None)

    def test_info_command(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_app.cli, ["info"])
        # info command should run without crashing (RAG may or may not be populated)
        assert result.exit_code == 0
        assert "Environment Info" in result.output


# =====================================================================
# Streamlit app module import
# =====================================================================


class TestStreamlitModuleImports:
    """Verify streamlit_app module can be imported (no syntax/import errors)."""

    def test_import_streamlit_app(self) -> None:
        import ui.streamlit_app as sa

        assert hasattr(sa, "main")
        assert hasattr(sa, "init_session_state")
        assert hasattr(sa, "render_sidebar")
        assert hasattr(sa, "render_chat_area")
        assert hasattr(sa, "render_reports_tab")
        assert hasattr(sa, "run_linear_pipeline")
        assert hasattr(sa, "run_feedback_turn")
