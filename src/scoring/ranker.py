"""
Agentic Profile Matching — Candidate Ranking.

Provides sorting, filtering by threshold, and shortlisting logic.
All operations are pure functions over lists of CandidateMatch dicts.

Architecture Reference: architecture.md Section 4.5 (rank_candidates node)
"""

from __future__ import annotations


DEFAULT_THRESHOLD = 0.3
DEFAULT_SHORTLIST_SIZE = 10


def rank_candidates(
    candidate_matches: list[dict],
) -> list[dict]:
    """Sort candidates by composite score in descending order.

    Args:
        candidate_matches: List of CandidateMatch dicts, each with a 'score' key.

    Returns:
        New list sorted by score descending (highest first).
        Candidates without a score are placed at the end.
    """
    return sorted(
        candidate_matches,
        key=lambda c: c.get("score", 0.0),
        reverse=True,
    )


def filter_by_threshold(
    candidates: list[dict],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict]:
    """Remove candidates below the minimum composite score threshold.

    Args:
        candidates: List of CandidateMatch dicts.
        threshold:  Minimum composite score to keep (default 0.3).

    Returns:
        Filtered list with candidates scoring >= threshold.
    """
    return [c for c in candidates if c.get("score", 0.0) >= threshold]


def shortlist(
    candidates: list[dict],
    n: int = DEFAULT_SHORTLIST_SIZE,
) -> list[dict]:
    """Return the top N candidates from the (already sorted) list.

    Args:
        candidates: List of CandidateMatch dicts (should be pre-sorted).
        n:          Maximum number to return (default 10).

    Returns:
        At most N candidates from the head of the list.
    """
    return candidates[:n]


def compute_hire_recommendation(score: float) -> str:
    """Derive a hire recommendation from the composite score.

    Thresholds:
        - >= 0.8: "hire"
        - >= 0.5: "borderline"
        - <  0.5: "no_hire"

    Args:
        score: Composite score (0.0-1.0).

    Returns:
        One of "hire", "borderline", "no_hire".
    """
    if score >= 0.8:
        return "hire"
    elif score >= 0.5:
        return "borderline"
    else:
        return "no_hire"


def generate_improvement_suggestions(gaps: list[str], requirements: dict) -> list[str]:
    """Generate improvement suggestions based on gaps and requirements.

    Args:
        gaps:        List of gap descriptions from scoring.
        requirements: The job requirements dict.

    Returns:
        List of actionable improvement suggestions.
    """
    suggestions = []

    for gap in gaps:
        # Extract skill name from gap text like "No TypeScript experience found"
        for skill_item in requirements.get("must_have", []) + requirements.get("nice_to_have", []):
            skill = skill_item.get("skill", "")
            if skill.lower() in gap.lower():
                suggestions.append(
                    f"Gain experience or certification in {skill} to strengthen candidacy"
                )
                break

    # Deduplicate
    seen: set[str] = set()
    unique: list[str] = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return unique if unique else ["Consider strengthening skills in the must-have requirements area"]
