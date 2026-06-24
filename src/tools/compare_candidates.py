"""
Tool: compare_candidates — Head-to-head comparison of candidates.

Uses LLM to generate a structured comparison table with narrative summary.

Architecture Reference: architecture.md Section 6.3
"""

from __future__ import annotations

import logging
import time

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

from src.agent.models import ComparisonResult
from src.llm.client import get_llm, get_llm_provider_name, record_llm_call
from src.prompts.comparison import COMPARISON_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@tool
def compare_candidates(
    candidate_ids: list[str],
    candidate_profiles: list[dict] | None = None,
    job_requirements: dict | None = None,
) -> dict:
    """Perform a head-to-head comparison of multiple candidates.

    Args:
        candidate_ids: List of candidate IDs to compare (must have at least 2).
        candidate_profiles: Optional list of CandidateMatch dicts for each candidate.
                           If provided, must match candidate_ids order and length.
        job_requirements: Optional job requirements dict for additional context.

    Returns:
        A dict matching the ComparisonResult schema:
        {
            "candidates": [{"id": "...", "name": "...", ...}, ...],
            "comparison_table": {"Criterion": ["Value A", "Value B"], ...},
            "summary": "Narrative comparison..."
        }
    """
    if not candidate_ids or len(candidate_ids) < 2:
        return {
            "candidates": [],
            "comparison_table": {},
            "summary": "At least 2 candidates are required for comparison.",
            "error": "insufficient_candidates",
        }

    # If no profiles provided, create minimal stubs from IDs
    if not candidate_profiles:
        candidate_profiles = [
            {
                "candidate_id": cid,
                "name": cid.split("_")[0] if "_" in cid else cid,
                "score": 0.0,
                "must_have_score": 0.0,
                "nice_to_have_score": 0.0,
                "reasoning": "No profile data available",
                "strengths": [],
                "gaps": [],
                "hire_recommendation": "unknown",
            }
            for cid in candidate_ids
        ]

    llm = get_llm()
    provider = get_llm_provider_name()

    try:
        structured_llm = llm.with_structured_output(ComparisonResult)

        # Build the user message with profiles
        profiles_text = _format_profiles(candidate_profiles)
        req_text = ""
        if job_requirements:
            req_text = f"\n## Job Requirements\n{_format_reqs(job_requirements)}\n"

        start_time = time.time()
        result: ComparisonResult = structured_llm.invoke(
            [
                SystemMessage(content=COMPARISON_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"{req_text}"
                        f"## Candidate Profiles\n{profiles_text}\n\n"
                        "Compare these candidates head-to-head."
                    )
                ),
            ]
        )
        duration_ms = (time.time() - start_time) * 1000
        record_llm_call(provider, True, None, tool="compare_candidates", duration_ms=duration_ms)
        return result.model_dump()

    except Exception as e:
        record_llm_call(provider, False, str(e), tool="compare_candidates")
        logger.error("Comparison failed: %s", e)
        # Return a basic comparison from the profile data
        return _fallback_comparison(candidate_profiles)


def _format_profiles(profiles: list[dict]) -> str:
    lines = []
    for p in profiles:
        name = p.get("name", "Unknown")
        cid = p.get("candidate_id", "?")
        lines.append(f"### {name} (ID: {cid})")
        lines.append(f"- Score: {p.get('score', 'N/A')}")
        lines.append(f"- Must-have: {p.get('must_have_score', 'N/A')}")
        lines.append(f"- Nice-to-have: {p.get('nice_to_have_score', 'N/A')}")
        if p.get("strengths"):
            lines.append(f"- Strengths: {', '.join(p['strengths'])}")
        if p.get("gaps"):
            lines.append(f"- Gaps: {', '.join(p['gaps'])}")
        if p.get("reasoning"):
            lines.append(f"- Reasoning: {p['reasoning']}")
        lines.append("")
    return "\n".join(lines)


def _format_reqs(reqs: dict) -> str:
    lines = []
    for item in reqs.get("must_have", []):
        lines.append(f"- [Required] {item.get('skill', '?')}")
    for item in reqs.get("nice_to_have", []):
        lines.append(f"- [Preferred] {item.get('skill', '?')}")
    return "\n".join(lines) if lines else "Not provided"


def _fallback_comparison(profiles: list[dict]) -> dict:
    """Generate a basic comparison without LLM."""
    candidates = []
    table: dict[str, list[str]] = {"Overall Score": [], "Must-Have": [], "Nice-to-Have": [], "Recommendation": []}
    for p in profiles:
        candidates.append({
            "id": p.get("candidate_id", ""),
            "name": p.get("name", ""),
            "scores": {
                "overall": p.get("score", 0),
                "must_have": p.get("must_have_score", 0),
                "nice_to_have": p.get("nice_to_have_score", 0),
            },
            "highlights": p.get("strengths", []),
            "red_flags": p.get("gaps", []),
        })
        table["Overall Score"].append(str(p.get("score", 0)))
        table["Must-Have"].append(str(p.get("must_have_score", 0)))
        table["Nice-to-Have"].append(str(p.get("nice_to_have_score", 0)))
        table["Recommendation"].append(p.get("hire_recommendation", "unknown"))

    names = [p.get("name", "?") for p in profiles]
    return ComparisonResult(
        candidates=candidates,
        comparison_table=table,
        summary=f"Basic comparison of {', '.join(names)}. LLM comparison unavailable — showing raw scores.",
    ).model_dump()