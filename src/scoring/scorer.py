"""
Agentic Profile Matching — Candidate Scoring.

Scores a single candidate's resume against structured job requirements.
Uses LLM with structured output (CandidateScore) as primary path,
with a keyword-based fallback when LLM is unavailable.

Architecture Reference: architecture.md Section 4.5 (rank_candidates node)
"""

from __future__ import annotations

import logging
import re
import time

from langchain_core.messages import SystemMessage, HumanMessage

from src.agent.models import CandidateScore
from src.llm.client import get_llm, get_llm_provider_name, record_llm_call
from src.prompts.scoring import SCORING_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

MUST_HAVE_WEIGHT = 0.7
NICE_TO_HAVE_WEIGHT = 0.3


def compute_composite_score(
    must_have_score: float,
    nice_to_have_score: float,
) -> float:
    """Weighted composite score: 0.7 * must_have + 0.3 * nice_to_have.

    Args:
        must_have_score:   Score on must-have criteria (0.0-1.0).
        nice_to_have_score: Score on nice-to-have criteria (0.0-1.0).

    Returns:
        Composite score in [0.0, 1.0].
    """
    return round(MUST_HAVE_WEIGHT * must_have_score + NICE_TO_HAVE_WEIGHT * nice_to_have_score, 4)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a score to [low, high]."""
    return max(low, min(high, value))


def score_candidate(
    resume_text: str,
    requirements: dict,
    llm=None,
) -> CandidateScore:
    """Score a candidate's resume against structured requirements.

    Primary path: LLM with structured output (CandidateScore).
    Fallback: keyword-based scoring when LLM is unavailable.

    Args:
        resume_text: Full text of the candidate's resume.
        requirements: Dict with 'must_have' and 'nice_to_have' skill lists,
                      each item having 'skill', 'type', 'weight', 'evidence'.
        llm: Optional LLM instance. If None, uses get_llm().

    Returns:
        CandidateScore with must_have_score, nice_to_have_score,
        reasoning, strengths, gaps, excerpts.
    """
    if not resume_text or not resume_text.strip():
        return CandidateScore(
            must_have_score=0.0,
            nice_to_have_score=0.0,
            reasoning="No resume text available for scoring.",
            strengths=[],
            gaps=["No resume provided"],
            excerpts=[],
        )

    # Try LLM-based scoring first
    try:
        if llm is None:
            llm = get_llm()
        provider = get_llm_provider_name()

        structured_llm = llm.with_structured_output(CandidateScore)

        # Build user message
        req_text = _format_requirements_for_scoring(requirements)
        user_msg = (
            f"## Job Requirements\n{req_text}\n\n"
            f"## Candidate Resume\n{resume_text}\n\n"
            "Score this candidate against the requirements."
        )

        start_time = time.time()
        result: CandidateScore = structured_llm.invoke([
            SystemMessage(content=SCORING_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        duration_ms = (time.time() - start_time) * 1000
        record_llm_call(provider, True, None, tool="score_candidate", duration_ms=duration_ms)

        # Validate and clamp scores
        result.must_have_score = _clamp(result.must_have_score)
        result.nice_to_have_score = _clamp(result.nice_to_have_score)
        return result

    except Exception as e:
        record_llm_call(provider if 'provider' in dir() else "unknown", False, str(e), tool="score_candidate")
        logger.warning("LLM scoring failed, using keyword fallback: %s", e)
        return _keyword_fallback_score(resume_text, requirements)


def _keyword_fallback_score(
    resume_text: str,
    requirements: dict,
) -> CandidateScore:
    """Score a candidate using keyword matching (no LLM needed).

    Checks how many required/nice-to-have skills appear in the resume text.
    Provides a reasonable approximation when the LLM is unavailable.
    """
    resume_lower = resume_text.lower()
    must_have_items = requirements.get("must_have", [])
    nice_to_have_items = requirements.get("nice_to_have", [])

    strengths: list[str] = []
    gaps: list[str] = []
    excerpts: list[str] = []

    # Score must-have skills
    must_hits = 0
    for item in must_have_items:
        skill = item.get("skill", "").lower()
        if not skill:
            continue
        # Check for the skill (whole-word aware)
        pattern = re.compile(r"\b" + re.escape(skill) + r"\b", re.IGNORECASE)
        match = pattern.search(resume_text)
        if match:
            must_hits += 1
            strengths.append(item["skill"])
            # Extract surrounding context as excerpt
            start = max(0, match.start() - 30)
            end = min(len(resume_text), match.end() + 80)
            excerpt = resume_text[start:end].replace("\n", " ").strip()
            if excerpt:
                excerpts.append(f"...{excerpt}...")
        else:
            gaps.append(f"No {item['skill']} experience found")

    must_total = len(must_have_items) if must_have_items else 1
    must_have_score = _clamp(must_hits / must_total)

    # Score nice-to-have skills
    nice_hits = 0
    for item in nice_to_have_items:
        skill = item.get("skill", "").lower()
        if not skill:
            continue
        pattern = re.compile(r"\b" + re.escape(skill) + r"\b", re.IGNORECASE)
        if pattern.search(resume_text):
            nice_hits += 1
            strengths.append(item["skill"])
        else:
            gaps.append(f"No {item['skill']} experience (nice-to-have)")

    nice_total = len(nice_to_have_items) if nice_to_have_items else 1
    nice_to_have_score = _clamp(nice_hits / nice_total)

    # Build reasoning
    total_skills = len(must_have_items) + len(nice_to_have_items)
    total_hits = must_hits + nice_hits
    reasoning = (
        f"Keyword-based scoring: {total_hits}/{total_skills} skills found in resume. "
        f"Must-have: {must_hits}/{len(must_have_items)} matched. "
        f"Nice-to-have: {nice_hits}/{len(nice_to_have_items)} matched."
    )

    return CandidateScore(
        must_have_score=must_have_score,
        nice_to_have_score=nice_to_have_score,
        reasoning=reasoning,
        strengths=strengths,
        gaps=gaps,
        excerpts=excerpts[:4],  # Cap at 4 excerpts
    )


def _format_requirements_for_scoring(reqs: dict) -> str:
    """Format requirements for the scoring prompt."""
    lines = []
    lines.append("### Must-Have")
    for item in reqs.get("must_have", []):
        lines.append(
            f"- {item.get('skill', 'Unknown')} "
            f"(type: {item.get('type', '?')}, weight: {item.get('weight', '?')})"
        )
    lines.append("")
    lines.append("### Nice-to-Have")
    for item in reqs.get("nice_to_have", []):
        lines.append(
            f"- {item.get('skill', 'Unknown')} "
            f"(type: {item.get('type', '?')}, weight: {item.get('weight', '?')})"
        )
    if reqs.get("experience_min_years"):
        lines.append(f"\nMinimum experience: {reqs['experience_min_years']} years")
    if reqs.get("education_level"):
        lines.append(f"Education: {reqs['education_level']}")
    return "\n".join(lines)
