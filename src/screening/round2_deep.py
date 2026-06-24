"""
Agentic Profile Matching — Round 2: Deep Analysis.

Deep analysis of top 10 candidates with:
  - Skill depth verification (keyword + LLM fallback)
  - Experience timeline analysis
  - Red-flag detection
  - Re-scoring with deeper evidence
  - Shortlist top 5-7

Architecture Reference: architecture.md Section 9 (Round 2 — Deep Analysis)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.agent.state import AgentState
from src.scoring.red_flags import detect_red_flags
from src.scoring.scorer import score_candidate
from src.scoring.ranker import shortlist
from src.tools.rag_search import get_full_resume_text

logger = logging.getLogger(__name__)

ROUND2_MAX_SHORTLIST = 7


def deep_analysis(state: AgentState) -> dict[str, Any]:
    """Round 2: Deep analysis of top candidates.

    For each candidate:
      1. Retrieve full resume text
      2. Skill depth verification
      3. Experience extraction
      4. Red-flag detection
      5. Re-score with composite formula
      6. Re-rank and shortlist top 5-7

    Architecture Reference: architecture.md Section 9 (Round 2 — Deep Analysis)
    """
    shortlist_in = list(state.get("current_shortlist", []))
    requirements = state.get("requirements", {})

    if not shortlist_in:
        return _empty_round_result("No candidates in shortlist for deep analysis.")

    # Track previous ranking for delta
    prev_ranking = {c["candidate_id"]: c.get("score", 0.0) for c in shortlist_in}

    matches: list[dict[str, Any]] = []
    errors: list[str] = []

    for candidate in shortlist_in:
        cid = candidate.get("candidate_id", "")
        try:
            # 1. Retrieve full resume
            full_resume = get_full_resume_text(cid)
            if not full_resume:
                logger.warning("Round 2: No resume for %s, keeping with R1 score", cid)
                matches.append(candidate)
                continue

            # Get candidate name
            name = candidate.get("name", cid.replace("_", " ").split(" ")[0].title())
            first_line = full_resume.strip().split("\n")[0]
            if first_line and len(first_line) < 80:
                name = first_line.strip()

            # 2. Skill depth verification
            skill_verification = _verify_skills_with_evidence(full_resume, requirements)

            # 3. Experience extraction
            experience_years = _extract_experience_years(full_resume)

            # 4. Red-flag detection
            red_flags = detect_red_flags(full_resume)

            # 5. Re-score using Phase 4 scorer for consistency
            score_result = score_candidate(full_resume, requirements)
            composite = 0.7 * score_result.must_have_score + 0.3 * score_result.nice_to_have_score

            # 6. Build enriched candidate match
            enriched = {
                "candidate_id": cid,
                "name": name,
                "score": composite,
                "must_have_score": score_result.must_have_score,
                "nice_to_have_score": score_result.nice_to_have_score,
                "reasoning": _generate_deep_reasoning(
                    name, skill_verification, red_flags, experience_years
                ),
                "strengths": score_result.strengths,
                "gaps": score_result.gaps,
                "resume_excerpts": score_result.excerpts,
                "hire_recommendation": "borderline",  # Finalized in round 3
                "improvement_suggestions": [],
                "red_flags": [f.to_dict() for f in red_flags],
                "experience_years": experience_years,
            }
            matches.append(enriched)

        except Exception as e:
            logger.error("Round 2: Error analyzing %s: %s", cid, e)
            errors.append(f"{cid}: {e}")
            # Keep the candidate with their Round 1 score
            matches.append(candidate)

    # Re-rank by composite score
    matches.sort(key=lambda c: c.get("score", 0.0), reverse=True)

    # Shortlist top 5-7
    final_list = shortlist(matches, n=ROUND2_MAX_SHORTLIST)

    eliminated_ids = [
        c["candidate_id"]
        for c in matches
        if c["candidate_id"] not in {f["candidate_id"] for f in final_list}
    ]

    # Count red flags for notes
    red_flag_count = sum(
        1 for c in matches if c.get("red_flags") and len(c["red_flags"]) > 0
    )

    # Build delta summary
    moved_up = []
    moved_down = []
    for c in matches:
        c_cid = c.get("candidate_id", "")
        if c_cid in prev_ranking:
            old_score = prev_ranking[c_cid]
            new_score = c.get("score", 0.0)
            if abs(new_score - old_score) > 0.01:
                if new_score > old_score:
                    moved_up.append(c_cid)
                else:
                    moved_down.append(c_cid)

    round_record = {
        "round_number": 2,
        "round_type": "deep_analysis",
        "candidates_evaluated": len(shortlist_in),
        "shortlisted_ids": [c["candidate_id"] for c in final_list],
        "eliminated_ids": eliminated_ids,
        "notes": (
            f"Deep analysis: evaluated {len(shortlist_in)}, "
            f"shortlisted {len(final_list)}. "
            f"Red flags found: {red_flag_count}. "
            f"Moved up: {len(moved_up)}, moved down: {len(moved_down)}."
        ),
    }

    comparison_result = None
    if moved_up or moved_down:
        delta_parts: list[str] = []
        if moved_up:
            delta_parts.append(f"Moved up: {', '.join(moved_up[:3])}")
        if moved_down:
            delta_parts.append(f"Moved down: {', '.join(moved_down[:3])}")
        comparison_result = {
            "type": "round2_delta",
            "summary": "Round 2 deep analysis. " + "; ".join(delta_parts),
        }

    result: dict[str, Any] = {
        "current_shortlist": final_list,
        "screening_rounds": [round_record],
        "current_round": 2,
    }
    if comparison_result:
        result["comparison_result"] = comparison_result
    if errors:
        result["error"] = f"Round 2 analysis errors: {'; '.join(errors)}"

    return result


def _extract_experience_years(resume_text: str) -> int | None:
    """Extract total years of relevant experience from resume text.

    Looks for patterns like "5 years of experience", "8+ years",
    "over X years". Returns the max year count found, or None.
    """
    patterns = [
        r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?experience',
        r'over\s+(\d+)\s+years?',
        r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:relevant\s+)?experience',
        r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:in\s+)?(?:the\s+)?industry',
        r'(\d+)\+?\s*years?\s+(?:of\s+)?industry\s+experience',
    ]
    max_years = 0
    for pattern in patterns:
        for match in re.finditer(pattern, resume_text, re.IGNORECASE):
            years = int(match.group(1))
            if years > max_years:
                max_years = years
    return max_years if max_years > 0 else None


def _verify_skills_with_evidence(
    resume_text: str,
    requirements: dict,
) -> dict[str, Any]:
    """Verify candidate's claimed skills with resume evidence.

    Uses keyword matching (fast, no LLM needed).

    Returns dict with matched_strengths, unmatched_skills, key_excerpts,
    must_have_score, nice_to_have_score.
    """
    must_haves = requirements.get("must_have", [])
    nice_haves = requirements.get("nice_to_have", [])

    matched_strengths: list[str] = []
    unmatched_skills: list[str] = []
    key_excerpts: list[str] = []

    must_hits = 0
    nice_hits = 0

    for item in must_haves:
        skill_name = item.get("skill", "")
        if not skill_name:
            continue
        pattern = re.compile(r"\b" + re.escape(skill_name) + r"\b", re.IGNORECASE)
        match_obj = pattern.search(resume_text)
        if match_obj:
            must_hits += 1
            matched_strengths.append(skill_name)
            start = max(0, match_obj.start() - 40)
            end = min(len(resume_text), match_obj.end() + 40)
            excerpt = resume_text[start:end].replace("\n", " ").strip()
            key_excerpts.append(excerpt[:120])
        else:
            unmatched_skills.append(skill_name)

    for item in nice_haves:
        skill_name = item.get("skill", "")
        if not skill_name:
            continue
        pattern = re.compile(r"\b" + re.escape(skill_name) + r"\b", re.IGNORECASE)
        if pattern.search(resume_text):
            nice_hits += 1
            matched_strengths.append(skill_name)

    must_total = len(must_haves) if must_haves else 1
    nice_total = len(nice_haves) if nice_haves else 1

    return {
        "must_have_score": must_hits / must_total,
        "nice_to_have_score": nice_hits / nice_total,
        "matched_strengths": matched_strengths,
        "unmatched_skills": unmatched_skills,
        "key_excerpts": key_excerpts,
    }


def _generate_deep_reasoning(
    name: str,
    skill_verification: dict,
    red_flags: list,
    experience_years: int | None,
) -> str:
    """Generate a detailed reasoning string for a candidate's deep analysis."""
    parts: list[str] = []

    matched = skill_verification.get("matched_strengths", [])
    unmatched = skill_verification.get("unmatched_skills", [])
    total_skills = len(matched) + len(unmatched)

    parts.append(
        f"{name}: "
        f"{len(matched)}/{total_skills} skills directly matched against resume. "
        f"{len(unmatched)} skills had no direct resume evidence."
    )

    if experience_years:
        parts.append(f"Reports {experience_years} years of relevant experience.")

    if red_flags:
        parts.append(f"Red Flags: {len(red_flags)} issue(s) detected.")

    if not unmatched:
        parts.append("All claimed skills were verified with resume evidence.")

    return " ".join(parts)


def _empty_round_result(reason: str) -> dict[str, Any]:
    """Return an empty round result with an error."""
    return {
        "current_shortlist": [],
        "screening_rounds": [{
            "round_number": 2,
            "round_type": "deep_analysis",
            "candidates_evaluated": 0,
            "shortlisted_ids": [],
            "eliminated_ids": [],
            "notes": f"Round 2 failed: {reason}",
        }],
        "current_round": 2,
        "error": reason,
    }