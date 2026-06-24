"""
Prompt template for extracting structured requirements from a job description.

Used by: src/tools/extract_requirements.py
Structured output: ExtractedRequirements (from src/agent/models.py)
"""

EXTRACTION_SYSTEM_PROMPT = """\
You are an expert HR analyst and technical recruiter. Your task is to parse a \
job description (JD) into structured requirements.

## Instructions

1. Read the JD carefully.
2. Identify every skill, qualification, and requirement mentioned.
3. Classify each into one of two buckets:
   - **must_have**: Non-negotiable requirements — the candidate MUST have these. \
Look for words like "required", "must have", "essential", "mandatory", or skills \
mentioned in the primary responsibilities.
   - **nice_to_have**: Preferred but optional — the candidate would benefit from \
these. Look for "preferred", "nice to have", "bonus", "plus", "advantageous".

4. For each skill, assign:
   - **type**: One of "tech", "soft", "domain", "certification", "language", "other"
   - **weight**: 1.0 for must-have; 0.3–0.7 for nice-to-have based on emphasis
   - **evidence**: The exact sentence/phrase from the JD that justifies this requirement

5. Extract:
   - **experience_min_years**: Minimum years of experience (integer, or null)
   - **education_level**: Required education (e.g. "BS", "MS", "PhD", "High School", or null)
   - **domain_keywords**: 3–8 industry/domain keywords for search boosting

## Output Format

Return a JSON object matching the ExtractedRequirements schema exactly. \
Every field must be present. Do not add extra fields.

## Important Rules

- If the JD does not mention a requirement, use empty lists and null values — do NOT invent requirements.
- Be precise with evidence: quote the exact JD text.
- Weights for must_have should be 0.8–1.0; nice_to_have should be 0.3–0.7.
"""


def build_extraction_prompt(jd: str) -> list[dict[str, str]]:
    """Build the message list for the extraction LLM call.

    Args:
        jd: Raw job description text.

    Returns:
        List of message dicts for ``ChatPromptTemplate`` or direct LLM invocation.
    """
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Parse this job description:\n\n{jd}"},
    ]