"""
Prompt template for explaining ranking decisions between candidates.

Used by: src/agent/nodes.py (explain_ranking node, Phase 4+)
"""

EXPLANATION_SYSTEM_PROMPT = """\
You are an expert hiring manager explaining why candidates were ranked in a \
particular order. Your explanation must be clear, evidence-based, and help \
the hiring manager understand the reasoning behind each ranking decision.

## Instructions

1. Review the two candidates' scores, strengths, gaps, and reasoning.
2. Identify the specific factors that caused one candidate to rank higher.
3. Provide a point-by-point explanation covering:
   - Must-have criteria differences (the primary ranking driver)
   - Nice-to-have criteria differences (secondary)
   - Qualitative factors (leadership, communication, etc.)
   - Any concerns or caveats the hiring manager should know about

4. Be honest about uncertainty — if two candidates are very close, say so.

## Output

Return a clear, professional explanation (3-6 paragraphs) that a hiring \
manager can use in their decision-making process.
"""


def build_explanation_prompt(
    higher_candidate: dict,
    lower_candidate: dict,
    requirements: dict | None = None,
) -> list[dict[str, str]]:
    """Build the message list for the ranking explanation LLM call.

    Args:
        higher_candidate: The candidate ranked higher.
        lower_candidate:  The candidate ranked lower.
        requirements:     Optional job requirements for context.

    Returns:
        List of message dicts for LLM invocation.
    """
    higher_text = _format_candidate(higher_candidate)
    lower_text = _format_candidate(lower_candidate)

    req_text = ""
    if requirements:
        req_text = "\n## Job Requirements\n"
        for item in requirements.get("must_have", []):
            req_text += f"- [Required] {item.get('skill', '?')}\n"
        for item in requirements.get("nice_to_have", []):
            req_text += f"- [Preferred] {item.get('skill', '?')}\n"

    return [
        {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{req_text}\n"
                f"## Higher-Ranked Candidate\n{higher_text}\n\n"
                f"## Lower-Ranked Candidate\n{lower_text}\n\n"
                "Explain why the first candidate was ranked higher than the second."
            ),
        },
    ]


def _format_candidate(c: dict) -> str:
    lines = [
        f"**{c.get('name', 'Unknown')}** (Score: {c.get('score', 'N/A')})",
        f"- Must-have score: {c.get('must_have_score', 'N/A')}",
        f"- Nice-to-have score: {c.get('nice_to_have_score', 'N/A')}",
    ]
    if c.get("strengths"):
        lines.append(f"- Strengths: {', '.join(c['strengths'])}")
    if c.get("gaps"):
        lines.append(f"- Gaps: {', '.join(c['gaps'])}")
    if c.get("reasoning"):
        lines.append(f"- Reasoning: {c['reasoning']}")
    return "\n".join(lines)