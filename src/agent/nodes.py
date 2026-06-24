"""
Agentic Profile Matching — Graph Node Functions.

Implements the 5 linear pipeline nodes (Phase 4) plus 6 interactive nodes (Phase 5):

  Linear: parse_jd -> extract_requirements -> search_resumes -> rank_candidates -> generate_report
  Interactive: human_feedback_loop, refine_requirements, compare_candidates_node,
              explain_ranking, generate_questions_node, route_natural_language

Each node is a function that takes the full AgentState and returns
a partial state dict (only the keys it updates). This is the LangGraph pattern.

Architecture Reference: architecture.md Section 4 (Graph Workflow)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.prompts.explanation import EXPLANATION_SYSTEM_PROMPT
from src.prompts.intent import KEYWORD_INTENT_MAP, INTENT_CLASSIFICATION_SYSTEM_PROMPT
from src.reports.match_report import generate_match_report
from src.scoring.ranker import (
    compute_hire_recommendation,
    filter_by_threshold,
    generate_improvement_suggestions,
    rank_candidates,
    shortlist,
)
from src.scoring.scorer import compute_composite_score, score_candidate
from src.tools.extract_requirements import extract_requirements
from src.tools.rag_search import get_full_resume_text, rag_search

logger = logging.getLogger(__name__)

# Default shortlist size for the linear pipeline
SHORTLIST_SIZE = 10

# Valid intent labels
VALID_INTENTS = Literal["refine", "compare", "questions", "explain", "report", "done", "new_search"]


# ===================================================================
# Phase 4 — Linear Pipeline Nodes (unchanged)
# ===================================================================


def parse_jd(state: AgentState) -> dict[str, Any]:
    """Node 4.2: Validate and store the raw JD.

    Validates that raw_jd is non-empty (at least 20 characters of
    meaningful text). Returns the raw_jd back in state.

    Architecture Reference: architecture.md Section 4.2
    """
    raw_jd = state.get("raw_jd", "")
    if not raw_jd or len(raw_jd.strip()) < 20:
        return {
            "error": "Job description is too short or empty (minimum 20 characters).",
            "raw_jd": raw_jd,
        }
    return {"raw_jd": raw_jd.strip()}


def extract_requirements_node(state: AgentState) -> dict[str, Any]:
    """Node 4.3: Extract structured requirements from the JD.

    Calls the extract_requirements tool (which uses LLM with structured
    output). Stores the result in state["requirements"].

    Architecture Reference: architecture.md Section 4.3
    """
    raw_jd = state.get("raw_jd", "")
    if not raw_jd:
        return {"error": "No raw_jd in state for requirement extraction."}

    try:
        result = extract_requirements.invoke({"jd": raw_jd})
        requirements = {
            "raw_jd": raw_jd,
            "must_have": result.get("must_have", []),
            "nice_to_have": result.get("nice_to_have", []),
            "experience_min_years": result.get("experience_min_years"),
            "education_level": result.get("education_level"),
            "domain_keywords": result.get("domain_keywords", []),
        }
        return {
            "requirements": requirements,
            "requirements_version": 1,
        }
    except Exception as e:
        logger.error("Requirement extraction failed: %s", e)
        return {
            "error": f"Requirement extraction failed: {e}",
            "requirements": {
                "raw_jd": raw_jd,
                "must_have": [],
                "nice_to_have": [],
                "experience_min_years": None,
                "education_level": None,
                "domain_keywords": [],
            },
            "requirements_version": 1,
        }


def search_resumes_node(state: AgentState) -> dict[str, Any]:
    """Node 4.4: Search resumes using RAG.

    Builds a search query from the structured requirements and calls
    rag_search. Stores candidate IDs in state["all_candidate_ids"].

    Architecture Reference: architecture.md Section 4.4
    """
    requirements = state.get("requirements", {})
    if not requirements:
        return {"error": "No requirements in state for resume search.", "all_candidate_ids": []}

    query = _build_search_query(requirements)

    try:
        results = rag_search.invoke({"query": query, "top_k": 100})
        candidate_ids = [r["candidate_id"] for r in results]
        logger.info("RAG search returned %d candidate IDs", len(candidate_ids))
        return {"all_candidate_ids": candidate_ids}
    except Exception as e:
        logger.error("RAG search failed: %s", e)
        return {"error": f"RAG search failed: {e}", "all_candidate_ids": []}


def rank_candidates_node(state: AgentState) -> dict[str, Any]:
    """Node 4.5: Run the 3-round screening pipeline to score and rank candidates.

    Delegates to run_screening_pipeline which orchestrates:
      Round 1: Broad RAG retrieval + keyword filter -> top 10
      Round 2: Deep analysis + red-flag detection -> top 5-7
      Round 3: Final hire/no-hire recommendation + reports

    Falls back to single-pass scoring if the pipeline fails.

    Architecture Reference: architecture.md Section 4.5, Section 9
    """
    candidate_ids = state.get("all_candidate_ids", [])
    requirements = state.get("requirements", {})

    if not candidate_ids:
        return {"current_shortlist": [], "error": "No candidate IDs to rank."}
    if not requirements:
        return {"current_shortlist": [], "error": "No requirements for scoring."}

    # Try the 3-round screening pipeline
    try:
        from src.screening.pipeline import run_screening_pipeline

        pipeline_result = run_screening_pipeline(state)

        # Verify the pipeline produced a shortlist
        final_list = pipeline_result.get("current_shortlist", [])
        if final_list:
            return pipeline_result
        else:
            logger.warning(
                "Screening pipeline produced empty shortlist, "
                "falling back to single-pass scoring."
            )
    except Exception as e:
        logger.warning("Screening pipeline failed, falling back to single-pass: %s", e)

    # Fallback: single-pass scoring (Phase 4 behavior)
    return _single_pass_rank(state)


def _single_pass_rank(state: AgentState) -> dict[str, Any]:
    """Single-pass scoring fallback when the 3-round pipeline is unavailable.

    Scores each candidate individually and returns a shortlist.
    """
    candidate_ids = state.get("all_candidate_ids", [])
    requirements = state.get("requirements", {})

    matches: list[dict] = []
    errors: list[str] = []

    for cid in candidate_ids:
        try:
            resume_text = get_full_resume_text(cid)
            if not resume_text:
                logger.warning("No resume text for candidate %s, skipping", cid)
                continue

            name = cid.replace("_", " ").split(" ")[0].title()
            first_line = resume_text.strip().split("\n")[0]
            if first_line and len(first_line) < 80:
                name = first_line.strip()

            score_result = score_candidate(resume_text, requirements)

            composite = compute_composite_score(
                score_result.must_have_score,
                score_result.nice_to_have_score,
            )

            recommendation = compute_hire_recommendation(composite)
            suggestions = generate_improvement_suggestions(
                score_result.gaps, requirements
            )

            match: dict[str, Any] = {
                "candidate_id": cid,
                "name": name,
                "score": composite,
                "must_have_score": score_result.must_have_score,
                "nice_to_have_score": score_result.nice_to_have_score,
                "reasoning": score_result.reasoning,
                "strengths": score_result.strengths,
                "gaps": score_result.gaps,
                "resume_excerpts": score_result.excerpts,
                "hire_recommendation": recommendation,
                "improvement_suggestions": suggestions,
            }
            matches.append(match)

        except Exception as e:
            logger.error("Error scoring candidate %s: %s", cid, e)
            errors.append(f"{cid}: {e}")

    ranked = rank_candidates(matches)
    filtered = filter_by_threshold(ranked, threshold=0.3)
    final_list = shortlist(filtered, n=SHORTLIST_SIZE)

    round_record = {
        "round_number": 1,
        "round_type": "initial",
        "candidates_evaluated": len(candidate_ids),
        "shortlisted_ids": [c["candidate_id"] for c in final_list],
        "eliminated_ids": [
            c["candidate_id"] for c in ranked if c["candidate_id"] not in {
                f["candidate_id"] for f in final_list
            }
        ],
        "notes": (
            f"Single-pass fallback: retrieved {len(candidate_ids)}, "
            f"scored {len(matches)}, shortlisted {len(final_list)} (threshold 0.3)"
        ),
    }

    result: dict[str, Any] = {
        "current_shortlist": final_list,
        "screening_rounds": [round_record],
        "current_round": 1,
    }
    if errors:
        result["error"] = f"Scoring errors: {'; '.join(errors)}"

    return result


def generate_report_node(state: AgentState) -> dict[str, Any]:
    """Node 4.6: Generate match reports for shortlisted candidates.

    For each candidate in the shortlist, generates a markdown report
    and stores it in state["generated_reports"]. Sets
    awaiting_human_feedback to True.

    Architecture Reference: architecture.md Section 4.6, Section 10
    """
    shortlist = state.get("current_shortlist", [])
    requirements = state.get("requirements", {})

    if not shortlist:
        return {
            "generated_reports": {},
            "awaiting_human_feedback": True,
            "error": "No shortlisted candidates to generate reports for.",
        }

    reports: dict[str, str] = {}
    for candidate in shortlist:
        cid = candidate.get("candidate_id", "")
        if not cid:
            continue
        try:
            report = generate_match_report(candidate, requirements)
            reports[cid] = report
        except Exception as e:
            logger.error("Report generation failed for %s: %s", cid, e)
            reports[cid] = f"# Report Generation Error\n\nFailed to generate report: {e}"

    return {
        "generated_reports": reports,
        "awaiting_human_feedback": True,
    }


def _build_search_query(requirements: dict) -> str:
    """Build a search query string from structured requirements.

    Combines all skill names and domain keywords into a natural
    language query suitable for semantic search.
    """
    parts: list[str] = []

    for item in requirements.get("must_have", []):
        skill = item.get("skill", "")
        if skill:
            parts.append(skill)

    for item in requirements.get("nice_to_have", []):
        skill = item.get("skill", "")
        if skill:
            parts.append(skill)

    for kw in requirements.get("domain_keywords", []):
        if kw and kw not in parts:
            parts.append(kw)

    exp = requirements.get("experience_min_years")
    if exp:
        parts.append(f"{exp}+ years experience")

    edu = requirements.get("education_level")
    if edu:
        parts.append(f"{edu} degree")

    return " ".join(parts) if parts else "developer engineer"


# ===================================================================
# Phase 5 — Interactive Loop Nodes
# ===================================================================


# Valid intents for routing
VALID_INTENT_SET: set[str] = {
    "refine", "compare", "questions", "explain", "report", "done", "new_search",
}


def classify_intent(human_feedback: str, llm=None) -> str:
    """Classify user intent from their feedback message.

    Primary path: LLM with structured output.
    Fallback: keyword-based pattern matching against KEYWORD_INTENT_MAP.

    Args:
        human_feedback: The user's message text.
        llm: Optional LLM instance. If None, uses get_llm().

    Returns:
        One of: "refine", "compare", "questions", "explain",
        "report", "done", "new_search".
    """
    if not human_feedback or not human_feedback.strip():
        return "explain"

    # Try keyword-based fallback first (fast, no LLM needed)
    keyword_result = _keyword_intent_classify(human_feedback)
    if keyword_result:
        return keyword_result

    # Try LLM classification
    try:
        if llm is None:
            from src.llm.client import get_llm

            llm = get_llm()

        response = llm.invoke([
            SystemMessage(content=INTENT_CLASSIFICATION_SYSTEM_PROMPT),
            HumanMessage(content=f'Classify this user message:\n\n"""\n{human_feedback}\n"""'),
        ])

        # Extract intent from response
        intent = response.content.strip().lower()
        # Handle cases where LLM wraps in quotes or adds extra text
        intent = intent.strip('"').strip("'").strip()
        # Take just the first word if LLM returned a sentence
        intent = intent.split()[0] if intent else "explain"

        if intent in VALID_INTENT_SET:
            return intent
        return "explain"

    except Exception as e:
        logger.warning("LLM intent classification failed, using fallback: %s", e)
        return "explain"


def _keyword_intent_classify(text: str) -> str | None:
    """Classify intent using keyword patterns. Returns None if no match.

    Checks the text against the KEYWORD_INTENT_MAP patterns.
    Returns the first matching intent, or None if no keywords match.
    """
    text_lower = text.lower()
    for keywords, intent in KEYWORD_INTENT_MAP:
        for kw in keywords:
            if kw in text_lower:
                return intent
    return None


def human_feedback_loop(state: AgentState) -> dict[str, Any]:
    """Node 4.7: Read human feedback and classify intent.

    Reads state["human_feedback"], calls classify_intent,
    and sets state["next_action"] for the conditional router.

    Architecture Reference: architecture.md Section 4.7
    """
    feedback = state.get("human_feedback", "")
    if not feedback or not feedback.strip():
        return {
            "next_action": "explain",
            "awaiting_human_feedback": False,
        }

    intent = classify_intent(feedback)
    logger.info("Classified intent: '%s' -> '%s'", feedback[:50], intent)

    return {
        "next_action": intent,
        "awaiting_human_feedback": False,
    }


def route_natural_language(state: AgentState) -> dict[str, Any]:
    """Alias for human_feedback_loop — classifies and routes.

    This node exists so the graph can route natural language input
    through the same classification path.
    """
    return human_feedback_loop(state)


def refine_requirements_node(state: AgentState) -> dict[str, Any]:
    """Node 4.8: Refine requirements based on user feedback.

    Parses the user's modification request and updates the requirements.
    Supports adding, removing, and modifying skills in must_have/nice_to_have.
    After updating, re-ranks all candidates and generates a delta summary.

    Architecture Reference: architecture.md Section 4.8, Section 8
    """
    feedback = state.get("human_feedback", "")
    requirements = dict(state.get("requirements", {}))
    old_shortlist = list(state.get("current_shortlist", []))

    if not requirements:
        return {
            "error": "No requirements to refine.",
            "awaiting_human_feedback": True,
        }

    # Snapshot old rankings for delta tracking
    old_ranking = {c["candidate_id"]: c.get("score", 0.0) for c in old_shortlist}

    # Parse and apply the modification
    try:
        modified = _parse_requirement_modification(feedback, requirements)
    except Exception as e:
        logger.error("Failed to parse requirement modification: %s", e)
        return {
            "error": f"Could not understand the modification: {e}",
            "awaiting_human_feedback": True,
        }

    new_version = state.get("requirements_version", 1) + 1

    # Re-rank candidates with updated requirements
    candidate_ids = state.get("all_candidate_ids", [])
    new_shortlist = _re_rank_candidates(candidate_ids, modified)

    # Build delta summary
    new_ranking = {c["candidate_id"]: c.get("score", 0.0) for c in new_shortlist}
    delta_summary = _build_delta_summary(old_ranking, new_ranking, old_shortlist, new_shortlist)

    # Re-generate reports for the new shortlist
    new_reports: dict[str, str] = {}
    for candidate in new_shortlist:
        cid = candidate.get("candidate_id", "")
        if cid:
            try:
                new_reports[cid] = generate_match_report(candidate, modified)
            except Exception as e:
                logger.error("Report re-generation failed for %s: %s", cid, e)

    return {
        "requirements": modified,
        "requirements_version": new_version,
        "current_shortlist": new_shortlist,
        "generated_reports": new_reports,
        "awaiting_human_feedback": True,
        "comparison_result": {
            "type": "refinement_delta",
            "summary": delta_summary,
        },
    }


def compare_candidates_node(state: AgentState) -> dict[str, Any]:
    """Node 4.9: Compare candidates as requested by the user.

    Extracts candidate references from the user's message, looks up
    their profiles from the shortlist, and calls the compare_candidates tool.

    Architecture Reference: architecture.md Section 4.9
    """
    feedback = state.get("human_feedback", "")
    shortlist = state.get("current_shortlist", [])
    requirements = state.get("requirements", {})

    if not shortlist:
        return {
            "error": "No candidates to compare. Run the pipeline first.",
            "awaiting_human_feedback": True,
        }

    # Extract candidate references from feedback
    candidate_ids = _extract_candidate_refs(feedback, shortlist)

    # If user said "top N", take the top N from the shortlist
    if not candidate_ids:
        n = _extract_top_n(feedback)
        if n:
            candidate_ids = [c["candidate_id"] for c in shortlist[:n]]

    # Fallback: compare top 2 if no candidates identified
    if not candidate_ids:
        candidate_ids = [c["candidate_id"] for c in shortlist[:2]]

    # Get profiles for the selected candidates
    profiles = [
        c for c in shortlist if c.get("candidate_id") in candidate_ids
    ]

    # If we have at least 2, run comparison
    if len(profiles) >= 2:
        try:
            from src.tools.compare_candidates import compare_candidates as compare_tool

            result = compare_tool.invoke({
                "candidate_ids": [p["candidate_id"] for p in profiles],
                "candidate_profiles": profiles,
                "job_requirements": requirements,
            })
            return {
                "comparison_result": result,
                "awaiting_human_feedback": True,
            }
        except Exception as e:
            logger.error("Comparison failed: %s", e)
            # Use fallback comparison
            from src.tools.compare_candidates import _fallback_comparison

            result = _fallback_comparison(profiles)
            return {
                "comparison_result": result,
                "awaiting_human_feedback": True,
            }
    elif len(profiles) == 1:
        name = profiles[0].get("name", profiles[0].get("candidate_id", "the candidate"))
        return {
            "comparison_result": {
                "type": "single_candidate",
                "summary": f"Only one candidate identified ({name}). Please specify at least 2 candidates to compare.",
            },
            "awaiting_human_feedback": True,
        }
    else:
        return {
            "comparison_result": {
                "type": "error",
                "summary": "Could not identify any candidates for comparison from your message.",
            },
            "awaiting_human_feedback": True,
        }


def explain_ranking_node(state: AgentState) -> dict[str, Any]:
    """Node 4.10: Explain why candidates are ranked the way they are.

    Identifies which candidates the user is asking about (or defaults
    to the top 2) and generates an explanation of the ranking difference.

    Architecture Reference: architecture.md Section 4.10
    """
    feedback = state.get("human_feedback", "")
    shortlist = state.get("current_shortlist", [])
    requirements = state.get("requirements", {})

    if not shortlist or len(shortlist) < 2:
        return {
            "comparison_result": {
                "type": "explanation",
                "summary": "Not enough candidates to explain ranking differences." if not shortlist
                else "Only one candidate in the shortlist — no ranking to explain.",
            },
            "awaiting_human_feedback": True,
        }

    # Identify the candidates being discussed
    candidate_ids = _extract_candidate_refs(feedback, shortlist)
    if not candidate_ids or len(candidate_ids) < 2:
        # Default to top 2
        candidate_ids = [shortlist[0]["candidate_id"], shortlist[1]["candidate_id"]]

    profile_map = {c["candidate_id"]: c for c in shortlist}
    higher = profile_map.get(candidate_ids[0], shortlist[0])
    lower = profile_map.get(candidate_ids[1], shortlist[1]) if len(candidate_ids) > 1 else shortlist[1]

    # Ensure higher is actually ranked higher
    if higher.get("score", 0) < lower.get("score", 0):
        higher, lower = lower, higher

    # Try LLM-based explanation
    try:
        from src.llm.client import get_llm, get_llm_provider_name, record_llm_call
        import time as _time

        llm = get_llm()
        provider = get_llm_provider_name()
        higher_text = _format_candidate_for_explanation(higher)
        lower_text = _format_candidate_for_explanation(lower)

        req_text = ""
        if requirements:
            req_lines = []
            for item in requirements.get("must_have", []):
                req_lines.append(f"- [Required] {item.get('skill', '?')}")
            for item in requirements.get("nice_to_have", []):
                req_lines.append(f"- [Preferred] {item.get('skill', '?')}")
            req_text = "\n## Job Requirements\n" + "\n".join(req_lines)

        start_time = _time.time()
        response = llm.invoke([
            SystemMessage(content=EXPLANATION_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"{req_text}\n\n"
                f"## Higher-Ranked Candidate\n{higher_text}\n\n"
                f"## Lower-Ranked Candidate\n{lower_text}\n\n"
                "Explain why the first candidate was ranked higher than the second."
            )),
        ])
        duration_ms = (_time.time() - start_time) * 1000
        record_llm_call(provider, True, None, tool="explain_ranking", duration_ms=duration_ms)
        explanation = response.content

    except Exception as e:
        try:
            from src.llm.client import record_llm_call
            record_llm_call("unknown", False, str(e), tool="explain_ranking")
        except Exception:
            pass
        logger.warning("LLM explanation failed, using fallback: %s", e)
        explanation = _fallback_explanation(higher, lower)

    return {
        "comparison_result": {
            "type": "explanation",
            "higher_candidate": higher.get("name", "?"),
            "lower_candidate": lower.get("name", "?"),
            "summary": explanation,
        },
        "awaiting_human_feedback": True,
    }


def generate_questions_node(state: AgentState) -> dict[str, Any]:
    """Node 4.11: Generate interview questions for a specific candidate.

    Identifies the candidate from the user's message (or defaults to
    the top candidate) and calls the generate_interview_questions tool.

    Architecture Reference: architecture.md Section 4.11
    """
    feedback = state.get("human_feedback", "")
    shortlist = state.get("current_shortlist", [])
    requirements = state.get("requirements", {})

    if not shortlist:
        return {
            "comparison_result": {
                "type": "questions",
                "summary": "No candidates available. Run the pipeline first.",
            },
            "awaiting_human_feedback": True,
        }

    # Identify candidate from feedback
    target = _extract_single_candidate(feedback, shortlist)
    if not target:
        target = shortlist[0]  # Default to top candidate

    cid = target.get("candidate_id", "")
    name = target.get("name", cid)

    # Try to use the tool (with fallback for when LLM is unavailable)
    try:
        from src.tools.generate_questions import generate_interview_questions

        result = generate_interview_questions.invoke({
            "candidate_id": cid,
            "candidate_name": name,
            "num_questions": 5,
            "requirements": requirements,
        })
        questions = result.get("questions", [])
    except Exception as e:
        logger.warning("Question generation failed, using fallback: %s", e)
        questions = _fallback_questions(target, requirements)

    # Format the response
    if questions:
        formatted = _format_questions_output(name, questions)
    else:
        formatted = f"Could not generate interview questions for {name}."

    return {
        "comparison_result": {
            "type": "questions",
            "candidate_id": cid,
            "candidate_name": name,
            "questions": questions,
            "summary": formatted,
        },
        "awaiting_human_feedback": True,
    }


# ===================================================================
# Phase 5 — Helper Functions
# ===================================================================


def _parse_requirement_modification(feedback: str, requirements: dict) -> dict:
    """Parse user feedback to modify requirements.

    Supports patterns like:
      - "Drop AWS" -> remove skill from must_have/nice_to_have
      - "Add TypeScript" -> add to must_have
      - "Move React to nice-to-have" -> move between lists
      - "Make experience 3 years" -> update experience_min_years

    Uses simple keyword parsing (no LLM needed for common patterns).
    """
    modified = {
        "raw_jd": requirements.get("raw_jd", ""),
        "must_have": list(requirements.get("must_have", [])),
        "nice_to_have": list(requirements.get("nice_to_have", [])),
        "experience_min_years": requirements.get("experience_min_years"),
        "education_level": requirements.get("education_level"),
        "domain_keywords": list(requirements.get("domain_keywords", [])),
    }

    feedback_lower = feedback.lower()

    # Pattern: "drop <skill>" or "remove <skill>"
    drop_match = re.search(r'(?:drop|remove)\s+(\w+(?:\s+\w+)?)', feedback_lower)
    if drop_match:
        skill_to_drop = drop_match.group(1).strip()
        modified["must_have"] = [
            s for s in modified["must_have"]
            if s.get("skill", "").lower() != skill_to_drop
        ]
        modified["nice_to_have"] = [
            s for s in modified["nice_to_have"]
            if s.get("skill", "").lower() != skill_to_drop
        ]

    # Pattern: "add <skill>" or "add <skill> to must-have/nice-to-have"
    add_match = re.search(r'add\s+(\w+(?:\s+\w+)?)', feedback_lower)
    if add_match:
        skill_name = add_match.group(1).strip().title()
        # Check if it already exists
        existing_skills = [
            s.get("skill", "").lower() for s in modified["must_have"] + modified["nice_to_have"]
        ]
        if skill_name.lower() not in existing_skills:
            new_skill = {
                "skill": skill_name,
                "type": "tech",
                "weight": 0.7,
                "evidence": f"Added by user request: {feedback}",
            }
            # Check if user specified nice-to-have
            if "nice-to-have" in feedback_lower or "nice to have" in feedback_lower or "preferred" in feedback_lower:
                modified["nice_to_have"].append(new_skill)
            else:
                modified["must_have"].append(new_skill)

    # Pattern: "move <skill> to nice-to-have" or "move <skill> to must-have"
    move_match = re.search(r'move\s+(\w+(?:\s+\w+)?)\s+to\s+(must-have|nice-to-have|must have|nice have)', feedback_lower)
    if move_match:
        skill_name = move_match.group(1).strip()
        target_list = move_match.group(2).strip()
        is_must = "must" in target_list

        # Find and remove from whichever list contains it
        skill_item = _find_and_remove_skill(modified, skill_name)

        if skill_item:
            if is_must:
                modified["must_have"].append(skill_item)
            else:
                modified["nice_to_have"].append(skill_item)

    # Pattern: "change experience to N years" or "set experience to N"
    exp_match = re.search(r'(?:change|set|update)?\s*experience\s*(?:to|=)?\s*(\d+)', feedback_lower)
    if exp_match:
        modified["experience_min_years"] = int(exp_match.group(1))

    # Pattern: "change education to <level>"
    edu_match = re.search(r'(?:change|set|update)?\s*education\s*(?:to|=)?\s*(\w+)', feedback_lower)
    if edu_match:
        modified["education_level"] = edu_match.group(1).upper()

    return modified


def _find_and_remove_skill(requirements: dict, skill_name: str) -> dict | None:
    """Find a skill by name (case-insensitive) in must_have or nice_to_have,
    remove it from its current list, and return it.

    Returns None if not found.
    """
    skill_lower = skill_name.lower()

    # Search in must_have first
    for i, s in enumerate(requirements.get("must_have", [])):
        if s.get("skill", "").lower() == skill_lower:
            requirements["must_have"].pop(i)
            return s

    # Then nice_to_have
    for i, s in enumerate(requirements.get("nice_to_have", [])):
        if s.get("skill", "").lower() == skill_lower:
            requirements["nice_to_have"].pop(i)
            return s

    return None


def _re_rank_candidates(candidate_ids: list[str], requirements: dict) -> list[dict]:
    """Re-score and re-rank candidates with updated requirements.

    Reuses the same scoring logic as rank_candidates_node.
    """
    matches: list[dict] = []

    for cid in candidate_ids:
        try:
            resume_text = get_full_resume_text(cid)
            if not resume_text:
                continue

            name = cid.replace("_", " ").split(" ")[0].title()
            first_line = resume_text.strip().split("\n")[0]
            if first_line and len(first_line) < 80:
                name = first_line.strip()

            score_result = score_candidate(resume_text, requirements)
            composite = compute_composite_score(
                score_result.must_have_score,
                score_result.nice_to_have_score,
            )
            recommendation = compute_hire_recommendation(composite)
            suggestions = generate_improvement_suggestions(score_result.gaps, requirements)

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
                "hire_recommendation": recommendation,
                "improvement_suggestions": suggestions,
            })
        except Exception as e:
            logger.error("Re-scoring candidate %s failed: %s", cid, e)

    ranked = rank_candidates(matches)
    filtered = filter_by_threshold(ranked, threshold=0.3)
    return shortlist(filtered, n=SHORTLIST_SIZE)


def _build_delta_summary(
    old_ranking: dict[str, float],
    new_ranking: dict[str, float],
    old_shortlist: list[dict],
    new_shortlist: list[dict],
) -> str:
    """Build a natural-language delta summary comparing old vs new rankings.

    Describes who moved up, who moved down, who was added, and who was removed.
    """
    parts: list[str] = []

    old_ids = set(old_ranking.keys())
    new_ids = set(new_ranking.keys())

    # Added candidates (in new but not in old)
    added = new_ids - old_ids
    if added:
        added_names = [
            c.get("name", c.get("candidate_id", "?")) for c in new_shortlist if c.get("candidate_id") in added
        ]
        parts.append(f"Newly shortlisted: {', '.join(added_names[:5])}")

    # Removed candidates (in old but not in new)
    removed = old_ids - new_ids
    if removed:
        removed_names = [
            c.get("name", c.get("candidate_id", "?")) for c in old_shortlist if c.get("candidate_id") in removed
        ]
        parts.append(f"Dropped from shortlist: {', '.join(removed_names[:5])}")

    # Rank changes (candidates in both old and new)
    common = old_ids & new_ids
    moved_up = []
    moved_down = []

    for cid in common:
        old_score = old_ranking.get(cid, 0.0)
        new_score = new_ranking.get(cid, 0.0)
        diff = new_score - old_score
        name_map = {c["candidate_id"]: c.get("name", c.get("candidate_id", "?")) for c in new_shortlist}
        name = name_map.get(cid, cid)

        if abs(diff) > 0.01:
            direction = "up" if diff > 0 else "down"
            change = f"{name} (score {direction}: {old_score:.2f} -> {new_score:.2f})"
            if diff > 0:
                moved_up.append(change)
            else:
                moved_down.append(change)

    if moved_up:
        parts.append(f"Moved up: {'; '.join(moved_up[:3])}")
    if moved_down:
        parts.append(f"Moved down: {'; '.join(moved_down[:3])}")

    if not parts:
        parts.append("No significant changes in rankings after requirement refinement.")

    return "Requirements updated. " + " ".join(parts)


def _extract_candidate_refs(feedback: str, shortlist: list[dict]) -> list[str]:
    """Extract candidate IDs referenced in user feedback.

    Matches against candidate names and IDs found in the shortlist.
    Returns a list of matching candidate IDs (order preserved from shortlist).
    """
    if not feedback:
        return []

    feedback_lower = feedback.lower()
    matched_ids: list[str] = []
    seen: set[str] = set()

    for candidate in shortlist:
        cid = candidate.get("candidate_id", "")
        name = candidate.get("name", "").lower()

        # Match by name (whole word) or by ID
        if name and name in feedback_lower:
            if cid not in seen:
                matched_ids.append(cid)
                seen.add(cid)
        elif cid and cid.lower() in feedback_lower:
            if cid not in seen:
                matched_ids.append(cid)
                seen.add(cid)

    return matched_ids


def _extract_single_candidate(feedback: str, shortlist: list[dict]) -> dict | None:
    """Extract a single candidate reference from user feedback.

    Returns the first matching candidate dict, or None.
    """
    ids = _extract_candidate_refs(feedback, shortlist)
    if ids:
        for c in shortlist:
            if c.get("candidate_id") == ids[0]:
                return c
    return None


def _extract_top_n(feedback: str) -> int | None:
    """Extract 'top N' pattern from user feedback.

    Returns N if found, None otherwise.
    """
    match = re.search(r'top\s+(\d+)', feedback.lower())
    if match:
        n = int(match.group(1))
        return max(2, min(n, 10))  # Clamp between 2 and 10
    return None


def _format_candidate_for_explanation(candidate: dict) -> str:
    """Format a candidate dict for the explanation prompt."""
    lines = [
        f"**{candidate.get('name', 'Unknown')}** (Score: {candidate.get('score', 'N/A')})",
        f"- Must-have score: {candidate.get('must_have_score', 'N/A')}",
        f"- Nice-to-have score: {candidate.get('nice_to_have_score', 'N/A')}",
    ]
    if candidate.get("strengths"):
        lines.append(f"- Strengths: {', '.join(candidate['strengths'])}")
    if candidate.get("gaps"):
        lines.append(f"- Gaps: {', '.join(candidate['gaps'])}")
    if candidate.get("reasoning"):
        lines.append(f"- Reasoning: {candidate['reasoning']}")
    return "\n".join(lines)


def _fallback_explanation(higher: dict, lower: dict) -> str:
    """Generate a basic explanation without LLM."""
    h_name = higher.get("name", "Candidate A")
    l_name = lower.get("name", "Candidate B")
    h_score = higher.get("score", 0)
    l_score = lower.get("score", 0)
    h_strengths = higher.get("strengths", [])
    l_gaps = lower.get("gaps", [])

    parts = [
        f"{h_name} ranks higher than {l_name} based on the composite score "
        f"({h_score:.2f} vs {l_score:.2f}).",
    ]

    if h_strengths:
        parts.append(
            f"{h_name} demonstrates strength in: {', '.join(h_strengths[:3])}."
        )
    if l_gaps:
        parts.append(
            f"{l_name} has gaps in: {', '.join(l_gaps[:3])}."
        )

    h_must = higher.get("must_have_score", 0)
    l_must = lower.get("must_have_score", 0)
    parts.append(
        f"On must-have criteria alone, {h_name} scores {h_must:.2f} "
        f"vs {l_name}'s {l_must:.2f} (must-have carries 70% weight in the composite score)."
    )

    return " ".join(parts)


def _fallback_questions(candidate: dict, requirements: dict) -> list[dict]:
    """Generate basic interview questions based on candidate gaps (no LLM)."""
    gaps = candidate.get("gaps", [])
    questions: list[dict] = []

    for gap in gaps:
        # Extract skill name from gap text like "No TypeScript experience found"
        for skill_item in requirements.get("must_have", []) + requirements.get("nice_to_have", []):
            skill = skill_item.get("skill", "")
            if skill.lower() in gap.lower():
                questions.append({
                    "question": f"Can you describe your experience with {skill}?",
                    "category": "technical",
                    "targets_gap": gap,
                    "difficulty": "medium",
                    "follow_ups": [
                        f"Have you used {skill} in a production environment?",
                        f"How would you rate your proficiency in {skill} from 1-10?",
                    ],
                })
                break

    # Ensure at least 3 questions
    if len(questions) < 3:
        name = candidate.get("name", "the candidate")
        questions.extend([
            {
                "question": f"Walk me through your most relevant project experience.",
                "category": "behavioral",
                "targets_gap": "General experience depth",
                "difficulty": "easy",
                "follow_ups": ["What was your specific role and contribution?"],
            },
            {
                "question": "Describe a challenging technical problem you solved recently.",
                "category": "situational",
                "targets_gap": "Problem-solving ability",
                "difficulty": "medium",
                "follow_ups": ["What would you do differently?"],
            },
            {
                "question": "Where do you see your career in the next 2-3 years?",
                "category": "behavioral",
                "targets_gap": "Career alignment",
                "difficulty": "easy",
                "follow_ups": [],
            },
        ])

    return questions[:5]


def _format_questions_output(candidate_name: str, questions: list[dict]) -> str:
    """Format generated questions into a readable summary string."""
    lines = [f"## Interview Questions for {candidate_name}\n"]
    for i, q in enumerate(questions, 1):
        lines.append(f"**{i}. {q.get('question', '?')}**")
        lines.append(f"   - Category: {q.get('category', '?')}")
        lines.append(f"   - Difficulty: {q.get('difficulty', '?')}")
        lines.append(f"   - Targets: {q.get('targets_gap', '?')}")
        follow_ups = q.get("follow_ups", [])
        if follow_ups:
            lines.append(f"   - Follow-ups: {'; '.join(follow_ups[:2])}")
        lines.append("")
    return "\n".join(lines)
