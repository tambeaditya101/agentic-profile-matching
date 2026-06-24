"""
Agentic Profile Matching — Screening Pipeline Orchestrator.

Orchestrates the 3-round screening funnel:
  Round 1 (initial_screen): Broad RAG retrieval + keyword filter -> top 10
  Round 2 (deep_analysis): Full resume review + skill verification + red flags -> top 5-7
  Round 3 (final_recommendation): Evidence compilation + hire recommendation -> reports

Architecture Reference: architecture.md Section 9 (Multi-Round Screening Pipeline)
"""

from __future__ import annotations

import logging
from typing import Any

from src.agent.state import AgentState
from src.screening.round1_initial import initial_screen
from src.screening.round2_deep import deep_analysis
from src.screening.round3_final import final_recommendation

logger = logging.getLogger(__name__)


def run_screening_pipeline(state: AgentState) -> dict[str, Any]:
    """Run the full 3-round screening pipeline sequentially.

    Each round receives the accumulated state (including screening_rounds
    from previous rounds) and returns an updated partial state dict.
    The caller (rank_candidates_node) merges these into the full state.

    Architecture Reference: architecture.md Section 9

    Args:
        state: Current agent state with requirements populated.

    Returns:
        Updated state partial dict with:
          - current_shortlist: final ranked list
          - screening_rounds: list of all 3 round records
          - generated_reports: per-candidate markdown reports
          - current_round: 3
    """
    all_rounds: list[dict] = list(state.get("screening_rounds", []))
    current_shortlist = list(state.get("current_shortlist", []))

    # --- Round 1: Initial broad screen ---
    logger.info("Starting 3-round screening pipeline...")
    r1_state = {
        **state,
        "screening_rounds": all_rounds,
        "current_shortlist": current_shortlist,
    }
    r1_result = initial_screen(r1_state)

    # Accumulate round 1 results
    all_rounds.extend(r1_result.get("screening_rounds", []))
    current_shortlist = r1_result.get("current_shortlist", [])
    r1_evaluated = 0
    for r in r1_result.get("screening_rounds", []):
        if r.get("round_number") == 1:
            r1_evaluated = r.get("candidates_evaluated", 0)

    logger.info(
        "Round 1 complete: %d evaluated, %d shortlisted",
        r1_evaluated, len(current_shortlist),
    )

    if not current_shortlist:
        logger.warning("Round 1 produced no candidates, skipping Rounds 2 and 3.")
        return {
            "current_shortlist": [],
            "screening_rounds": all_rounds,
            "current_round": 1,
            "error": "No candidates survived Round 1 screening.",
        }

    # --- Round 2: Deep analysis ---
    logger.info("Running Round 2: deep analysis...")
    r2_state = {
        **state,
        "screening_rounds": all_rounds,
        "current_shortlist": current_shortlist,
    }
    r2_result = deep_analysis(r2_state)

    # Accumulate round 2 results
    all_rounds.extend(r2_result.get("screening_rounds", []))
    current_shortlist = r2_result.get("current_shortlist", current_shortlist)
    r2_evaluated = 0
    for r in r2_result.get("screening_rounds", []):
        if r.get("round_number") == 2:
            r2_evaluated = r.get("candidates_evaluated", 0)

    logger.info(
        "Round 2 complete: %d evaluated, %d shortlisted",
        r2_evaluated, len(current_shortlist),
    )

    if not current_shortlist:
        logger.warning("Round 2 produced no candidates, skipping Round 3.")
        return {
            "current_shortlist": [],
            "screening_rounds": all_rounds,
            "current_round": 2,
            "generated_reports": {},
            "error": "No candidates survived Round 2 deep analysis.",
        }

    # --- Round 3: Final recommendation ---
    logger.info("Running Round 3: final recommendation...")
    r3_state = {
        **state,
        "screening_rounds": all_rounds,
        "current_shortlist": current_shortlist,
    }
    r3_result = final_recommendation(r3_state)

    # Accumulate round 3 results
    all_rounds.extend(r3_result.get("screening_rounds", []))
    current_shortlist = r3_result.get("current_shortlist", current_shortlist)
    generated_reports = r3_result.get("generated_reports", {})

    r3_evaluated = 0
    for r in r3_result.get("screening_rounds", []):
        if r.get("round_number") == 3:
            r3_evaluated = r.get("candidates_evaluated", 0)

    logger.info(
        "Round 3 complete: %d evaluated, %d reports generated.",
        r3_evaluated, len(generated_reports),
    )

    logger.info(
        "Screening pipeline complete. Rounds: R1=%d eval, R2=%d eval, R3=%d eval. "
        "Final shortlist: %d candidates.",
        r1_evaluated, r2_evaluated, r3_evaluated, len(current_shortlist),
    )

    return {
        "current_shortlist": current_shortlist,
        "screening_rounds": all_rounds,
        "generated_reports": generated_reports,
        "current_round": 3,
    }