"""
Agentic Profile Matching — Round 1: Initial Broad Screen.

Broad retrieval from RAG followed by a lightweight must-have
keyword filter. Candidates missing >30% of must-have skills
are eliminated. Top 10 advance to Round 2.

Architecture Reference: architecture.md Section 9 (Round 1 — Initial Screen)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.agent.state import AgentState
from src.scoring.scorer import score_candidate
from src.scoring.ranker import filter_by_threshold, shortlist
from src.tools.rag_search import rag_search, get_full_resume_text

logger = logging.getLogger(__name__)

ROUND1_KEYWORD_THRESHOLD = 0.3  # Filter if <30% must-have skills found
ROUND1_SHORTLIST_SIZE = 10


def initial_screen(state: AgentState) -> dict[str, Any]:
    """Round 1: Initial broad screen using RAG + keyword filter.

    1. RAG search with top_k=100 to get a broad pool.
    2. Quick must-have keyword check against resume excerpts.
    3. Score candidates using Phase 4 scoring (LLM or keyword fallback).
    4. Filter out candidates missing >30% must-have skills.
    5. Shortlist top 10.
    6. Record ScreeningRound(round_number=1).

    Architecture Reference: architecture.md Section 9 (Round 1 — Initial Screen)
    """
    requirements = state.get("requirements", {})
    must_haves = requirements.get("must_have", [])

    if not must_haves:
        return _empty_round_result(state, "No must-have requirements defined.")

    # Step 1: RAG broad retrieval
    query = _build_round1_query(requirements)
    rag_results: list[dict] = []
    try:
        rag_results = rag_search.invoke({"query": query, "top_k": 100})
    except Exception as e:
        logger.error("Round 1 RAG search failed: %s", e)
        return _empty_round_result(state, f"RAG search failed: {e}")

    candidate_ids = list(dict.fromkeys(
        r["candidate_id"] for r in rag_results if r.get("candidate_id")
    ))
    logger.info("Round 1: RAG returned %d candidate IDs", len(candidate_ids))

    if not candidate_ids:
        return _empty_round_result(state, "No candidates found via RAG search.")

    # Step 2: Quick must-have keyword filter
    passed: list[str] = []
    for cid in candidate_ids:
        try:
            resume_text = get_full_resume_text(cid)
            if not resume_text:
                logger.warning("Round 1: No resume for %s, skipping", cid)
                continue

            hit_count = _count_keyword_hits(resume_text, must_haves)
            total = len(must_haves) if must_haves else 1
            hit_ratio = hit_count / total

            if hit_ratio < ROUND1_KEYWORD_THRESHOLD:
                logger.info(
                    "Round 1: %s filtered (must-have hit ratio %.0f < %.0f)",
                    cid, hit_ratio, ROUND1_KEYWORD_THRESHOLD,
                )
                continue

            passed.append(cid)
        except Exception as e:
            logger.warning("Round 1: Error checking %s: %s", cid, e)
            continue

    if not passed:
        logger.info(
            "Round 1: All candidates failed must-have filter, "
            "taking top %d from RAG results", ROUND1_SHORTLIST_SIZE
        )
        passed = candidate_ids[:ROUND1_SHORTLIST_SIZE]

    # Step 3: Score each passing candidate
    matches: list[dict[str, Any]] = []
    errors: list[str] = []

    for cid in passed:
        try:
            resume_text = get_full_resume_text(cid)
            if not resume_text:
                continue

            # Get candidate name from resume first line
            name = cid.replace("_", " ").split(" ")[0].title()
            first_line = resume_text.strip().split("\n")[0]
            if first_line and len(first_line) < 80:
                name = first_line.strip()

            score_result = score_candidate(resume_text, requirements)
            composite = 0.7 * score_result.must_have_score + 0.3 * score_result.nice_to_have_score

            matches.append({
                "candidate_id": cid,
                "name": name,
                "score": composite,
                "must_have_score": score_result.must_have_score,
                "nice_to_have_score": score_result.nice_to_have_score,
                "reasoning": score_result.reasoning,
                "strengths": score_result.strengths,
                "gaps": score_result.gaps,
                "resume_excerpts": score_result.excerpts,
                "hire_recommendation": "borderline",  # Round 1 doesn't finalize
                "improvement_suggestions": [],
            })
        except Exception as e:
            logger.error("Round 1: Error scoring %s: %s", cid, e)
            errors.append(f"{cid}: {e}")

    # Step 4: Rank, filter, shortlist
    matches.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    filtered = filter_by_threshold(matches, threshold=0.0)
    final_list = shortlist(filtered, n=ROUND1_SHORTLIST_SIZE)

    eliminated_ids = [
        c["candidate_id"]
        for c in matches
        if c["candidate_id"] not in {f["candidate_id"] for f in final_list}
    ]

    round_record = {
        "round_number": 1,
        "round_type": "initial",
        "candidates_evaluated": len(candidate_ids),
        "shortlisted_ids": [c["candidate_id"] for c in final_list],
        "eliminated_ids": eliminated_ids,
        "notes": (
            f"Round 1 broad screen: retrieved {len(candidate_ids)}, "
            f"passed keyword filter: {len(passed)}, "
            f"scored {len(matches)}, shortlisted {len(final_list)}"
        ),
    }

    result: dict[str, Any] = {
        "current_shortlist": final_list,
        "screening_rounds": [round_record],
        "current_round": 1,
    }
    if errors:
        result["error"] = f"Round 1 scoring errors: {'; '.join(errors)}"

    return result


def _build_round1_query(requirements: dict) -> str:
    """Build a search query focused on must-have skills."""
    parts: list[str] = []
    for item in requirements.get("must_have", []):
        skill = item.get("skill", "")
        if skill:
            parts.append(skill)
    for item in requirements.get("nice_to_have", []):
        skill = item.get("skill", "")
        if skill and skill not in parts:
            parts.append(skill)
    if requirements.get("experience_min_years"):
        parts.append(f"{requirements['experience_min_years']}+ years experience")
    if requirements.get("education_level"):
        parts.append(f"{requirements['education_level']} degree")
    return " ".join(parts) if parts else "developer engineer"


def _count_keyword_hits(
    resume_text: str, must_haves: list[dict],
) -> int:
    """Count how many must-have skills appear (case-insensitive whole-word match)."""
    count = 0
    for item in must_haves:
        skill = item.get("skill", "").lower()
        if skill:
            pattern = re.compile(r"\b" + re.escape(skill) + r"\b", re.IGNORECASE)
            if pattern.search(resume_text):
                count += 1
    return count


def _empty_round_result(state: AgentState, reason: str) -> dict[str, Any]:
    """Return an empty round result with an error."""
    return {
        "current_shortlist": [],
        "screening_rounds": [{
            "round_number": 1,
            "round_type": "initial",
            "candidates_evaluated": 0,
            "shortlisted_ids": [],
            "eliminated_ids": [],
            "notes": f"Round 1 failed: {reason}",
        }],
        "current_round": 1,
        "error": reason,
    }