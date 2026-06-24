"""
Tool: generate_interview_questions — Generate targeted interview questions.

Uses LLM to produce questions focused on candidate gaps and borderline areas.

Architecture Reference: architecture.md Section 6.4
"""

from __future__ import annotations

import logging
import time

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.agent.models import InterviewQuestion
from src.llm.client import get_llm, get_llm_provider_name, record_llm_call
from src.prompts.questions import QUESTIONS_SYSTEM_PROMPT
from src.tools.rag_search import get_full_resume_text

logger = logging.getLogger(__name__)


class QuestionListOutput(BaseModel):
    """Wrapper for LLM to return a list of interview questions."""

    questions: list[InterviewQuestion] = Field(
        default_factory=list, description="Generated interview questions"
    )


@tool
def generate_interview_questions(
    candidate_id: str,
    candidate_name: str | None = None,
    num_questions: int = 5,
    requirements: dict | None = None,
) -> dict:
    """Generate targeted interview questions for a specific candidate based on their profile and the JD.

    Args:
        candidate_id: ID of the candidate (as used in the vector store).
        candidate_name: Optional display name. If not provided, derived from candidate_id.
        num_questions: Number of questions to generate (default 5, max 10).
        requirements: Optional job requirements dict for context.

    Returns:
        A dict with keys:
            - "candidate_id" (str)
            - "candidate_name" (str)
            - "questions" (list of question dicts)
    """
    num_questions = max(1, min(num_questions, 10))

    # Derive name from ID if not provided
    if not candidate_name:
        candidate_name = candidate_id.split("_")[0] if "_" in candidate_id else candidate_id

    # Get full resume from RAG
    resume_text = get_full_resume_text(candidate_id)
    if not resume_text:
        logger.warning("No resume found for candidate %s, using minimal info", candidate_id)
        resume_text = f"No resume text available for candidate {candidate_id}."

    # Default requirements if not provided
    if not requirements:
        requirements = {"must_have": [], "nice_to_have": []}

    llm = get_llm()
    provider = get_llm_provider_name()

    try:
        structured_llm = llm.with_structured_output(QuestionListOutput)

        req_text = _format_requirements(requirements)

        start_time = time.time()
        result: QuestionListOutput = structured_llm.invoke(
            [
                SystemMessage(content=QUESTIONS_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"## Candidate: {candidate_name}\n\n"
                        f"### Job Requirements\n{req_text}\n\n"
                        f"### Candidate Resume\n{resume_text}\n\n"
                        f"Generate {num_questions} targeted interview questions."
                    )
                ),
            ]
        )
        duration_ms = (time.time() - start_time) * 1000
        record_llm_call(provider, True, None, tool="generate_questions", duration_ms=duration_ms)
        questions = [q.model_dump() for q in result.questions]

    except Exception as e:
        record_llm_call(provider, False, str(e), tool="generate_questions")
        logger.error("Question generation failed: %s", e)
        questions = []

    return {
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "questions": questions,
    }


def _format_requirements(reqs: dict) -> str:
    lines = ["**Must-Have:**"]
    for item in reqs.get("must_have", []):
        lines.append(f"- {item.get('skill', '?')} (weight: {item.get('weight', '?')})")
    lines.append("\n**Nice-to-Have:**")
    for item in reqs.get("nice_to_have", []):
        lines.append(f"- {item.get('skill', '?')} (weight: {item.get('weight', '?')})")
    if reqs.get("experience_min_years"):
        lines.append(f"\nMinimum experience: {reqs['experience_min_years']} years")
    return "\n".join(lines)