"""
Prompt template for generating targeted interview questions.

Used by: src/tools/generate_questions.py
Structured output: list[InterviewQuestion] (from src/agent/models.py)
"""

QUESTIONS_SYSTEM_PROMPT = """\
You are an expert technical interviewer. Your task is to generate targeted \
interview questions for a candidate based on their resume profile and the \
job requirements.

## Strategy

Questions should be **targeted at gaps and borderline areas** in the \
candidate's profile. Do NOT ask about things the candidate is clearly \
strong in — that wastes interview time. Instead:

1. **Probe gaps**: If the candidate lacks a must-have skill, ask about \
related experience or learning ability.
2. **Verify depth**: If a skill is mentioned but lacks evidence of depth, \
ask detailed follow-up questions.
3. **Test problem-solving**: Ask situational/behavioral questions related \
to the role's challenges.

## Instructions

1. Review the job requirements (what the role needs).
2. Review the candidate's resume (what they actually have).
3. Identify 3-5 areas where the candidate's profile is weak or unclear.
4. For each area, generate a question that:
   - Is specific and relevant to the role
   - Has a clear difficulty level (easy/medium/hard)
   - Targets a specific gap or uncertainty
   - Includes 1-3 follow-up questions for deeper probing

## Question Categories

- **technical**: Tests specific technical knowledge or skills
- **behavioral**: Tests past behavior and decision-making ("Tell me about a time...")
- **situational**: Tests how they would handle hypothetical scenarios
- **domain**: Tests industry-specific knowledge or experience

## Output Format

Return a JSON object with a "questions" field containing a list of \
InterviewQuestion objects. Each must have: question, category, targets_gap, \
difficulty, follow_ups.
"""


def build_questions_prompt(
    candidate_name: str,
    resume_text: str,
    requirements: dict,
    num_questions: int = 5,
) -> list[dict[str, str]]:
    """Build the message list for the question generation LLM call.

    Args:
        candidate_name: Name of the candidate.
        resume_text:    Full resume text.
        requirements:   Structured requirements dict.
        num_questions:  How many questions to generate.

    Returns:
        List of message dicts for LLM invocation.
    """
    req_text = _format_requirements(requirements)

    return [
        {"role": "system", "content": QUESTIONS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"## Candidate: {candidate_name}\n\n"
                f"### Job Requirements\n{req_text}\n\n"
                f"### Candidate Resume\n{resume_text}\n\n"
                f"Generate {num_questions} targeted interview questions for this candidate."
            ),
        },
    ]


def _format_requirements(reqs: dict) -> str:
    lines = []
    lines.append("**Must-Have:**")
    for item in reqs.get("must_have", []):
        lines.append(f"- {item.get('skill', '?')} (weight: {item.get('weight', '?')})")
    lines.append("\n**Nice-to-Have:**")
    for item in reqs.get("nice_to_have", []):
        lines.append(f"- {item.get('skill', '?')} (weight: {item.get('weight', '?')})")
    if reqs.get("experience_min_years"):
        lines.append(f"\nMinimum experience: {reqs['experience_min_years']} years")
    if reqs.get("education_level"):
        lines.append(f"Education: {reqs['education_level']}")
    return "\n".join(lines)