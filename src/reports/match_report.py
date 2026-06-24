"""
Agentic Profile Matching — Match Report Generation.

Generates per-candidate markdown match reports following the structure
from architecture.md Section 10.1.

Architecture Reference: architecture.md Section 10 (Explainability & Match Reports)
"""

from __future__ import annotations


def _score_status(score: float) -> str:
    """Return a human-readable status label for a score."""
    if score >= 0.8:
        return "STRONG MATCH"
    elif score >= 0.6:
        return "GOOD MATCH"
    elif score >= 0.4:
        return "PARTIAL"
    elif score >= 0.2:
        return "WEAK"
    else:
        return "NO MATCH"


def _recommendation_label(rec: str) -> str:
    """Format hire recommendation for display."""
    labels = {
        "hire": "STRONG HIRE",
        "no_hire": "NO HIRE",
        "borderline": "BORDERLINE",
    }
    return labels.get(rec, rec.upper())


def generate_match_report(
    candidate: dict,
    requirements: dict,
) -> str:
    """Generate a markdown match report for a single candidate.

    Follows the structure from architecture.md Section 10.1:
    - Summary
    - Score table (must-have, nice-to-have, composite)
    - Must-have criteria breakdown
    - Nice-to-have criteria breakdown
    - Strengths
    - Gaps
    - Improvement suggestions
    - Hire recommendation

    Args:
        candidate:    CandidateMatch dict with all scoring fields.
        requirements: Requirements dict with must_have/nice_to_have lists.

    Returns:
        Markdown string for the match report.
    """
    name = candidate.get("name", "Unknown Candidate")
    cid = candidate.get("candidate_id", "?")
    must_score = candidate.get("must_have_score", 0.0)
    nice_score = candidate.get("nice_to_have_score", 0.0)
    composite = candidate.get("score", 0.0)
    reasoning = candidate.get("reasoning", "")
    strengths = candidate.get("strengths", [])
    gaps = candidate.get("gaps", [])
    excerpts = candidate.get("resume_excerpts", [])
    recommendation = candidate.get("hire_recommendation", "borderline")
    suggestions = candidate.get("improvement_suggestions", [])

    lines: list[str] = []

    # Header
    lines.append(f"# Candidate Match Report: {name}")
    lines.append(f"*Candidate ID: {cid}*\n")

    # Summary
    lines.append("## Summary")
    lines.append(reasoning if reasoning else f"{name} has a composite match score of {composite:.2f}.")
    lines.append("")

    # Score table
    must_status = _score_status(must_score)
    nice_status = _score_status(nice_score)
    comp_status = _score_status(composite)

    lines.append("## Scores")
    lines.append("| Criterion | Weight | Score | Status |")
    lines.append("|-----------|--------|-------|--------|")
    lines.append(f"| Must-Have Overall | 70% | {must_score:.2f} | {must_status} |")
    lines.append(f"| Nice-to-Have Overall | 30% | {nice_score:.2f} | {nice_status} |")
    lines.append(f"| **Composite** | **100%** | **{composite:.2f}** | **{comp_status}** |")
    lines.append("")

    # Must-have breakdown
    must_items = requirements.get("must_have", [])
    if must_items:
        lines.append("## Must-Have Criteria Breakdown")
        lines.append("| Skill | Required | Found | Evidence |")
        lines.append("|-------|----------|-------|----------|")
        for item in must_items:
            skill = item.get("skill", "?")
            evidence = item.get("evidence", "")
            # Check if this skill is in the candidate's strengths
            found = "Yes" if any(s.lower() == skill.lower() for s in strengths) else "No"
            # Find relevant excerpt
            relevant_excerpt = ""
            for ex in excerpts:
                if skill.lower() in ex.lower():
                    relevant_excerpt = ex[:100]
                    break
            lines.append(f"| {skill} | Yes | {found} | {relevant_excerpt} |")
        lines.append("")

    # Nice-to-have breakdown
    nice_items = requirements.get("nice_to_have", [])
    if nice_items:
        lines.append("## Nice-to-Have Criteria Breakdown")
        lines.append("| Skill | Required | Found | Evidence |")
        lines.append("|-------|----------|-------|----------|")
        for item in nice_items:
            skill = item.get("skill", "?")
            evidence = item.get("evidence", "")
            found = "Yes" if any(s.lower() == skill.lower() for s in strengths) else "Partial"
            relevant_excerpt = ""
            for ex in excerpts:
                if skill.lower() in ex.lower():
                    relevant_excerpt = ex[:100]
                    break
            lines.append(f"| {skill} | Preferred | {found} | {relevant_excerpt} |")
        lines.append("")

    # Strengths
    if strengths:
        lines.append("## Strengths")
        for s in strengths:
            lines.append(f"- {s}")
        lines.append("")

    # Gaps
    if gaps:
        lines.append("## Gaps")
        for g in gaps:
            lines.append(f"- {g}")
        lines.append("")

    # Evidence excerpts
    if excerpts:
        lines.append("## Evidence from Resume")
        for i, ex in enumerate(excerpts, 1):
            lines.append(f"{i}. \"{ex}\"")
        lines.append("")

    # Improvement suggestions
    if suggestions:
        lines.append("## Improvement Suggestions")
        for s in suggestions:
            lines.append(f"- {s}")
        lines.append("")

    # Hire recommendation
    lines.append(f"## Hire Recommendation: {_recommendation_label(recommendation)}")
    if reasoning:
        lines.append(reasoning)
    lines.append("")

    return "\n".join(lines)
