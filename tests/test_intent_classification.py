"""
Agentic Profile Matching — Phase 5 Intent Classification Tests.

Tests the classify_intent function and keyword fallback with 20+ inputs.
No LLM required — all tests use the keyword-based fallback path.

Run:
    pytest tests/test_intent_classification.py -v
"""

from __future__ import annotations

import pytest

from src.agent.nodes import classify_intent, _keyword_intent_classify


# ===================================================================
# Keyword Fallback Tests (no LLM needed)
# ===================================================================


class TestKeywordIntentClassify:
    """Test the fast keyword-based intent classifier."""

    # --- compare ---
    @pytest.mark.parametrize("msg", [
        "Compare the top 3 candidates",
        "compare Alice and Bob",
        "How do the top 2 stack up?",
        "Side by side comparison of Alice vs Bob",
        "Compare candidate 1 and candidate 2",
        "Let's compare the top candidates",
    ])
    def test_compare_intent(self, msg: str) -> None:
        assert _keyword_intent_classify(msg) == "compare", f"Failed for: {msg}"

    # --- questions ---
    @pytest.mark.parametrize("msg", [
        "Generate interview questions for Alice",
        "What should I ask candidate 1?",
        "Interview questions for the top candidate",
        "Questions for Bob",
        "I need questions for the second candidate",
    ])
    def test_questions_intent(self, msg: str) -> None:
        assert _keyword_intent_classify(msg) == "questions", f"Failed for: {msg}"

    # --- explain ---
    @pytest.mark.parametrize("msg", [
        "Why did Alice rank higher than Bob?",
        "Explain the ranking for the top candidate",
        "Why is Bob ranked lower?",
        "What makes the top candidate the best?",
        "Explain why Alice is first",
    ])
    def test_explain_intent(self, msg: str) -> None:
        assert _keyword_intent_classify(msg) == "explain", f"Failed for: {msg}"

    # --- refine ---
    @pytest.mark.parametrize("msg", [
        "Drop AWS from requirements",
        "Add TypeScript to must-have",
        "Remove the PhD requirement",
        "Change experience to 3 years",
        "Modify the React requirement",
        "Update education to Masters",
        "Make React less important",
        "Increase experience requirement to 7 years",
        "Lower the experience requirement",
    ])
    def test_refine_intent(self, msg: str) -> None:
        assert _keyword_intent_classify(msg) == "refine", f"Failed for: {msg}"

    # --- report ---
    @pytest.mark.parametrize("msg", [
        "Show me the report for Alice",
        "Generate reports for all candidates",
        "Full report for candidate 1",
        "I want to see the match report",
    ])
    def test_report_intent(self, msg: str) -> None:
        assert _keyword_intent_classify(msg) == "report", f"Failed for: {msg}"

    # --- new_search ---
    @pytest.mark.parametrize("msg", [
        "New search with a different JD",
        "Start over with a new job description",
        "Reset and search again",
        "I want to try a different job",
    ])
    def test_new_search_intent(self, msg: str) -> None:
        assert _keyword_intent_classify(msg) == "new_search", f"Failed for: {msg}"

    # --- done ---
    @pytest.mark.parametrize("msg", [
        "done",
        "that's all thanks",
        "thank you, I'm finished",
        "exit",
        "goodbye",
        "bye",
    ])
    def test_done_intent(self, msg: str) -> None:
        assert _keyword_intent_classify(msg) == "done", f"Failed for: {msg}"

    # --- no keyword match ---
    def test_no_keyword_match_returns_none(self) -> None:
        assert _keyword_intent_classify("Hello, how are you today?") is None

    def test_empty_returns_none(self) -> None:
        assert _keyword_intent_classify("") is None


class TestClassifyIntentIntegration:
    """Test the full classify_intent function (keyword path, no LLM).

    Since LLM is unavailable in test env, these test the fallback path.
    classify_intent falls back to keyword first, then LLM. If keyword
    matches, it returns immediately without calling LLM.
    """

    @pytest.mark.parametrize("msg,expected", [
        ("Compare the top 3", "compare"),
        ("Generate questions for Alice", "questions"),
        ("Why did Alice rank higher?", "explain"),
        ("Drop AWS", "refine"),
        ("Show me the report", "report"),
        ("done", "done"),
        ("Start over", "new_search"),
    ])
    def test_keyword_path_direct(self, msg: str, expected: str) -> None:
        """When keyword matches, classify_intent returns without LLM."""
        result = classify_intent(msg)
        assert result == expected, f"Expected '{expected}' for: {msg}"

    def test_empty_feedback_defaults_to_explain(self) -> None:
        result = classify_intent("")
        assert result == "explain"

    def test_none_feedback_defaults_to_explain(self) -> None:
        result = classify_intent(None)  # type: ignore[arg-type]
        assert result == "explain"

    # --- Priority: earlier patterns take precedence ---
    def test_compare_takes_priority(self) -> None:
        # "Compare" appears in compare keywords, should not match explain
        result = classify_intent("Compare why Alice is higher")
        assert result == "compare"

    def test_questions_takes_priority(self) -> None:
        # "question" appears before "why"
        result = classify_intent("I have a question about why Alice ranked higher")
        # "why" is in explain keywords but "question" is in questions keywords
        # and questions comes before explain in KEYWORD_INTENT_MAP
        assert result == "questions"
