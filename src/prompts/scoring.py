"""
Prompt template for scoring a candidate against structured requirements.

Used by: src/scoring/scorer.py (Phase 4) and potentially tools
Structured output: CandidateScore (from src/agent/models.py)
"""

SCORING_SYSTEM_PROMPT = """\
You are an expert technical recruiter evaluating a candidate's resume against \
a job description's requirements. You must provide a fair, evidence-based \
assessment.

## Instructions

1. Review the structured requirements (must-have and nice-to-have skills).
2. Read the candidate's full resume text carefully.
3. For each requirement, determine if the candidate meets it based on \
explicit evidence in the resume.

4. Assign scores:
   - **must_have_score** (0.0–1.0): What fraction of must-have requirements \
are clearly met? A 1.0 means every must-have skill is present with evidence. \
A 0.0 means none are met. Be strict — if a skill is mentioned but without \
depth or evidence, score it lower.
   - **nice_to_have_score** (0.0–1.0): What fraction of nice-to-have requirements \
are met? More lenient scoring is appropriate here.

5. Provide:
   - **reasoning**: A 2-4 sentence explanation of the overall match quality.
   - **strengths**: List 2-5 specific strengths with evidence (e.g. "5 years React experience").
   - **gaps**: List any missing or weak areas (e.g. "No cloud experience mentioned").
   - **excerpts**: 2-4 direct quotes from the resume that support the scores.

## Important Rules

- Base scores ONLY on what is explicitly in the resume. Do not infer or assume.
- If the resume mentions a skill but gives no evidence of depth, count it as \
a partial match (0.5 weight).
- Excerpts must be verbatim from the resume text.
"""


def build_scoring_prompt(
    resume_text: str,
    requirements: dict,
) -> list[dict[str, str]]:
    """Build the message list for the scoring LLM call.

    Args:
        resume_text:   Full resume text of the candidate.
        requirements:  Dict with 'must_have' and 'nice_to_have' skill lists.

    Returns:
        List of message dicts for LLM invocation.
    """
    req_text = _format_requirements(requirements)

    return [
        {"role": "system", "content": SCORING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"## Job Requirements\n{req_text}\n\n"
                f"## Candidate Resume\n{resume_text}\n\n"
                "Score this candidate against the requirements."
            ),
        },
    ]


def _format_requirements(reqs: dict) -> str:
    """Pretty-print requirements for the prompt."""
    lines = []
    lines.append("### Must-Have")
    for item in reqs.get("must_have", []):
        lines.append(f"- {item.get('skill', 'Unknown')} (type: {item.get('type', '?')}, weight: {item.get('weight', '?')})")
    lines.append("")
    lines.append("### Nice-to-Have")
    for item in reqs.get("nice_to_have", []):
        lines.append(f"- {item.get('skill', 'Unknown')} (type: {item.get('type', '?')}, weight: {item.get('weight', '?')})")
    if reqs.get("experience_min_years"):
        lines.append(f"\nMinimum experience: {reqs['experience_min_years']} years")
    if reqs.get("education_level"):
        lines.append(f"Education: {reqs['education_level']}")
    return "\n".join(lines)