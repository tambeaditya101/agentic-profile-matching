"""
Agentic Profile Matching — Round 3: Final Recommendation.

Final evaluation of top 5-7 candidates with:
  - Comprehensive evidence compilation from all rounds
  - Hire/no-hire/borderline recommendation via rule-based fallback
  - Improvement suggestions for borderline candidates
  - Full match report generation

Architecture Reference: architecture.md Section 9 (Round 3 — Final Recommendation)
"""

from __future__ import annotations

import logging
from typing import Any

from src.agent.state import AgentState
from src.reports.match_report import generate_match_report
from src.scoring.ranker import compute_hire_recommendation, generate_improvement_suggestions

logger = logging.getLogger(__name__)


def final_recommendation(state: AgentState) -> dict[str, Any]:
    """Round 3: Final recommendation with hire/no-hire decisions.

    For each candidate in the shortlist:
      1. Compile evidence from all previous rounds
      2. Generate hire/no-hire/borderline recommendation
      3. Generate improvement suggestions for borderline candidates
      4. Generate full match report

    Architecture Reference: architecture.md Section 9 (Round 3 — Final Recommendation)
    """
    shortlist = list(state.get("current_shortlist", []))
    requirements = state.get("requirements", {})
    screening_rounds = list(state.get("screening_rounds", []))

    if not shortlist:
        round_record = {
            "round_number": 3,
            "round_type": "final",
            "candidates_evaluated": 0,
            "shortlisted_ids": [],
            "eliminated_ids": [],
            "notes": "Round 3 failed: no candidates in shortlist.",
        }
        return {
            "current_shortlist": [],
            "generated_reports": {},
            "screening_rounds": screening_rounds + [round_record],
            "current_round": 3,
        }

    reports: dict[str, str] = {}
    hire_count = 0
    borderline_count = 0
    no_hire_ids: list[str] = []

    for candidate in shortlist:
        cid = candidate.get("candidate_id", "")
        if not cid:
            continue

        # 1. Compile evidence from previous rounds
        evidence = _compile_round_evidence(candidate, screening_rounds)

        # 2. Generate hire recommendation using rule-based logic
        composite_score = candidate.get("score", 0.0)
        must_have_score = candidate.get("must_have_score", 0.0)
        recommendation = _generate_hire_recommendation(composite_score, must_have_score)

        # 3. Generate improvement suggestions for borderline candidates
        suggestions: list[str] = []
        if recommendation == "borderline":
            suggestions = generate_improvement_suggestions(
                candidate.get("gaps", []), requirements
            )
            borderline_count += 1
        elif recommendation == "hire":
            hire_count += 1
        else:
            no_hire_ids.append(cid)

        # Update candidate with final recommendation and suggestions
        candidate["hire_recommendation"] = recommendation
        candidate["improvement_suggestions"] = suggestions

        # 4. Generate full match report
        try:
            report = generate_match_report(candidate, requirements)
            reports[cid] = report
        except Exception as e:
            logger.error("Round 3: Report failed for %s: %s", cid, e)
            reports[cid] = f"# Report Generation Error\n\nFailed to generate report: {e}"

    # Build round record
    round_record = {
        "round_number": 3,
        "round_type": "final",
        "candidates_evaluated": len(shortlist),
        "shortlisted_ids": [
            c["candidate_id"] for c in shortlist
            if c.get("hire_recommendation") == "hire"
        ],
        "eliminated_ids": no_hire_ids,
        "notes": (
            f"Final recommendations: {hire_count} hire, "
            f"{borderline_count} borderline, "
            f"{len(no_hire_ids)} no-hire. "
            f"Reports generated: {len(reports)} candidates."
        ),
    }

    return {
        "current_shortlist": shortlist,
        "generated_reports": reports,
        "screening_rounds": screening_rounds + [round_record],
        "current_round": 3,
    }


def _generate_hire_recommendation(composite_score: float, must_have_score: float) -> str:
    """Generate hire recommendation using rule-based logic.

    Decision logic (conservative):
      - composite >= 0.8 and must_have >= 0.7: "hire"
      - composite >= 0.5: "borderline"
      - otherwise: "no_hire"

    Args:
        composite_score: Overall composite score (0.0-1.0).
        must_have_score: Must-have criteria score (0.0-1.0).

    Returns:
        One of "hire", "borderline", "no_hire".
    """
    if composite_score >= 0.8 and must_have_score >= 0.7:
        return "hire"
    elif composite_score >= 0.5:
        return "borderline"
    else:
        return "no_hire"


def _compile_round_evidence(
    candidate: dict,
    screening_rounds: list[dict],
) -> str:
    """Compile all evidence from previous rounds into a text block.

    Args:
        candidate: The candidate dict.
        screening_rounds: List of previous round records.

    Returns:
        A multi-line text string summarizing evidence from all rounds.
    """
    parts: list[str] = []
    name = candidate.get("name", "Unknown")

    parts.append(f"Candidate: {name}")
    parts.append(f"Overall Score: {candidate.get('score', 0.0):.2f}")
    parts.append(f"Must-Have Score: {candidate.get('must_have_score', 0.0):.2f}")
    parts.append(f"Nice-to-Have Score: {candidate.get('nice_to_have_score', 0.0):.2f}")

    # Strengths and gaps
    strengths = candidate.get("strengths", [])
    if strengths:
        parts.append(f"Strengths: {', '.join(strengths)}")

    gaps = candidate.get("gaps", [])
    if gaps:
        parts.append(f"Gaps: {', '.join(gaps)}")

    # Red flags from Round 2
    red_flags = candidate.get("red_flags", [])
    if red_flags:
        flag_descriptions = [f.get("description", "") for f in red_flags if isinstance(f, dict)]
        if flag_descriptions:
            parts.append(f"Red Flags: {'; '.join(flag_descriptions[:3])}")

    # Experience from Round 2
    exp_years = candidate.get("experience_years")
    if exp_years:
        parts.append(f"Experience: {exp_years} years")

    # Round 1 and 2 notes
    for r in screening_rounds:
        r_num = r.get("round_number", "?")
        r_type = r.get("round_type", "?")
        r_notes = r.get("notes", "")
        if r_notes:
            parts.append(f"Round {r_num} ({r_type}): {r_notes}")

    return "\n".join(parts)