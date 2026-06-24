"""
Prompt template for head-to-head candidate comparison.

Used by: src/tools/compare_candidates.py
Structured output: ComparisonResult (from src/agent/models.py)
"""

COMPARISON_SYSTEM_PROMPT = """\
You are an expert hiring manager performing a head-to-head comparison of \
candidates for a specific role. Your analysis must be objective, evidence-based, \
and structured for easy decision-making.

## Instructions

1. Review each candidate's profile (scores, strengths, gaps, reasoning).
2. Identify 4-8 meaningful comparison dimensions (e.g., "React experience", \
"Leadership", "Education", "Cloud skills").
3. For each dimension, describe each candidate's standing concisely (1-2 sentences).
4. Write a narrative summary highlighting:
   - Who is stronger overall and why
   - What trade-offs exist between the candidates
   - Any red flags or concerns unique to each candidate

## Output Format

Return a JSON object matching the ComparisonResult schema:
- **candidates**: Array of candidate summary dicts (id, name, scores, highlights, red_flags)
- **comparison_table**: Dict mapping each criterion to a list of per-candidate value strings
- **summary**: 3-5 sentence narrative comparison

## Important Rules

- Be fair and objective — do not favor any candidate.
- Base everything on the provided profile data, not assumptions.
- The comparison_table values should be concise (max 15 words each).
- The summary should help a hiring manager make a decision.
"""


def build_comparison_prompt(
    candidate_profiles: list[dict],
    job_requirements: dict | None = None,
) -> list[dict[str, str]]:
    """Build the message list for the comparison LLM call.

    Args:
        candidate_profiles: List of candidate match dicts (from CandidateMatch TypedDict).
        job_requirements:   Optional requirements dict for context.

    Returns:
        List of message dicts for LLM invocation.
    """
    profiles_text = _format_candidate_profiles(candidate_profiles)
    req_text = ""
    if job_requirements:
        req_text = f"\n## Job Requirements\n{_format_reqs(job_requirements)}\n"

    return [
        {"role": "system", "content": COMPARISON_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{req_text}"
                f"## Candidate Profiles\n{profiles_text}\n\n"
                "Compare these candidates head-to-head."
            ),
        },
    ]


def _format_candidate_profiles(profiles: list[dict]) -> str:
    lines = []
    for p in profiles:
        lines.append(f"### {p.get('name', 'Unknown')} (ID: {p.get('candidate_id', '?')})")
        lines.append(f"- **Overall score**: {p.get('score', 'N/A')}")
        lines.append(f"- **Must-have score**: {p.get('must_have_score', 'N/A')}")
        lines.append(f"- **Nice-to-have score**: {p.get('nice_to_have_score', 'N/A')}")
        lines.append(f"- **Recommendation**: {p.get('hire_recommendation', 'N/A')}")
        if p.get("strengths"):
            lines.append(f"- **Strengths**: {', '.join(p['strengths'])}")
        if p.get("gaps"):
            lines.append(f"- **Gaps**: {', '.join(p['gaps'])}")
        if p.get("reasoning"):
            lines.append(f"- **Reasoning**: {p['reasoning']}")
        lines.append("")
    return "\n".join(lines)


def _format_reqs(reqs: dict) -> str:
    lines = []
    for item in reqs.get("must_have", []):
        lines.append(f"- [Required] {item.get('skill', '?')}")
    for item in reqs.get("nice_to_have", []):
        lines.append(f"- [Preferred] {item.get('skill', '?')}")
    return "\n".join(lines) if lines else "Not provided"