"""
Tool: extract_requirements — Parse a JD into structured requirements.

Uses LLM with structured output (ExtractedRequirements schema) to classify
each requirement into must-have vs. nice-to-have with weights and evidence.

If no LLM is available (no GEMINI_API_KEY and Ollama not running), falls
back to a deterministic keyword-based extractor so the rest of the
pipeline (search, screening, reports) still works end-to-end.

Architecture Reference: architecture.md Section 6.1
"""

from __future__ import annotations

import json
import logging
import re
import time

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

from src.agent.models import ExtractedRequirements, SkillRequirement
from src.llm.client import get_llm, get_llm_provider_name, record_llm_call
from src.prompts.extraction import EXTRACTION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@tool
def extract_requirements(jd: str) -> dict:
    """Parse a job description into structured must-have and nice-to-have requirements.

    Args:
        jd: Raw job description text (at least 50 characters).

    Returns:
        A dict matching the ExtractedRequirements schema:
        {
            "must_have": [{"skill": "React", "type": "tech", "weight": 1.0, "evidence": "..."}, ...],
            "nice_to_have": [{"skill": "AWS", "type": "tech", "weight": 0.5, "evidence": "..."}, ...],
            "experience_min_years": 3,
            "education_level": "BS",
            "domain_keywords": ["frontend", "web"]
        }
    """
    if not jd or len(jd.strip()) < 20:
        return ExtractedRequirements().model_dump()

    # Try LLM-based extraction first
    try:
        llm = get_llm()
        provider = get_llm_provider_name()
    except RuntimeError as e:
        logger.warning("No LLM available, using keyword-based fallback: %s", e)
        record_llm_call("none", False, str(e), tool="extract_requirements")
        return _keyword_fallback_extract(jd).model_dump()

    # Use with_structured_output for guaranteed schema compliance
    start_time = time.time()
    try:
        structured_llm = llm.with_structured_output(ExtractedRequirements)
        result: ExtractedRequirements = structured_llm.invoke(
            [
                SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(content=f"Parse this job description:\n\n{jd}"),
            ]
        )
        duration_ms = (time.time() - start_time) * 1000
        record_llm_call(provider, True, None, tool="extract_requirements", duration_ms=duration_ms)

        # If LLM returned empty must_have, supplement with keyword fallback
        if not result.must_have:
            logger.info("LLM returned empty must_have, supplementing with keyword fallback")
            kb = _keyword_fallback_extract(jd)
            if kb.must_have:
                result.must_have = kb.must_have
            if not result.nice_to_have and kb.nice_to_have:
                result.nice_to_have = kb.nice_to_have
            if result.experience_min_years is None:
                result.experience_min_years = kb.experience_min_years
            if result.education_level is None:
                result.education_level = kb.education_level
            if not result.domain_keywords:
                result.domain_keywords = kb.domain_keywords
        return result.model_dump()
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        record_llm_call(provider, False, str(e), tool="extract_requirements", duration_ms=duration_ms)
        logger.warning("Structured output failed, falling back to text parsing: %s", e)
        # Fallback: try regular invoke and parse JSON manually
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=EXTRACTION_SYSTEM_PROMPT + "\n\nRespond with ONLY valid JSON, no markdown."),
                    HumanMessage(content=f"Parse this job description:\n\n{jd}"),
                ]
            )
            text = response.content if hasattr(response, "content") else str(response)
            # Strip markdown code fences if present
            text = text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            parsed = json.loads(text)
            # Validate through the model
            validated = ExtractedRequirements.model_validate(parsed)
            record_llm_call(provider, True, None, tool="extract_requirements_text", duration_ms=duration_ms)
            return validated.model_dump()
        except Exception as e2:
            record_llm_call(provider, False, str(e2), tool="extract_requirements_text", duration_ms=duration_ms)
            logger.error("LLM parsing failed, using keyword fallback: %s", e2)
            return _keyword_fallback_extract(jd).model_dump()


# =====================================================================
# Keyword-based fallback (no LLM required)
# =====================================================================

# Curated skill dictionary — each entry maps an alias regex to a canonical
# skill name + type. We deliberately keep this short and high-precision
# rather than exhaustive; the LLM extractor handles the long tail.
_SKILL_DICTIONARY: list[tuple[str, str, str]] = [
    # (regex, canonical_name, type)
    # Frontend
    (r"\bReact(?:\.js|\.JS)?\b", "React", "tech"),
    (r"\bReactJS\b", "React", "tech"),
    (r"\bReact Native\b", "React Native", "tech"),
    (r"\bAngular(?:JS)?\b", "Angular", "tech"),
    (r"\bVue(?:\.js|\.JS)?\b", "Vue", "tech"),
    (r"\bNext\.?js\b", "Next.js", "tech"),
    (r"\bTypeScript\b", "TypeScript", "tech"),
    (r"\bJavaScript\b", "JavaScript", "tech"),
    (r"\bJS\b(?!\w)", "JavaScript", "tech"),
    (r"\bHTML5?\b", "HTML", "tech"),
    (r"\bCSS3?\b", "CSS", "tech"),
    (r"\bTailwind(?:\s*CSS)?\b", "Tailwind CSS", "tech"),
    (r"\bStyled Components\b", "Styled Components", "tech"),
    (r"\bSASS\b|(?<!\w)SCSS\b", "SASS", "tech"),
    (r"\bRedux\b", "Redux", "tech"),
    (r"\bZustand\b", "Zustand", "tech"),
    (r"\bGraphQL\b", "GraphQL", "tech"),
    (r"\bREST(?:ful)?\s*API\b", "REST APIs", "tech"),
    (r"\bREST\b", "REST APIs", "tech"),
    # Backend
    (r"\bPython\b", "Python", "tech"),
    (r"\bDjango\b", "Django", "tech"),
    (r"\bFlask\b", "Flask", "tech"),
    (r"\bFastAPI\b", "FastAPI", "tech"),
    (r"\bNode\.?js\b", "Node.js", "tech"),
    (r"\bExpress(?:\.js)?\b", "Express", "tech"),
    (r"\bJava\b(?!\s*Script)", "Java", "tech"),
    (r"\bSpring(?:\s*Boot)?\b", "Spring Boot", "tech"),
    (r"\bGo(?:lang)?\b", "Go", "tech"),
    (r"\bRust\b", "Rust", "tech"),
    (r"\bC\+\+\b", "C++", "tech"),
    (r"\bC#\b", "C#", "tech"),
    (r"\b\.NET\b", ".NET", "tech"),
    (r"\bRuby\b", "Ruby", "tech"),
    (r"\bRails\b", "Rails", "tech"),
    (r"\bPHP\b", "PHP", "tech"),
    (r"\bLaravel\b", "Laravel", "tech"),
    # Databases
    (r"\bPostgreSQL\b|\bPostgres\b", "PostgreSQL", "tech"),
    (r"\bMySQL\b", "MySQL", "tech"),
    (r"\bMongoDB\b", "MongoDB", "tech"),
    (r"\bRedis\b", "Redis", "tech"),
    (r"\bSQL\b", "SQL", "tech"),
    (r"\bNoSQL\b", "NoSQL", "tech"),
    (r"\bDynamoDB\b", "DynamoDB", "tech"),
    # Cloud / DevOps
    (r"\bAWS\b", "AWS", "tech"),
    (r"\bGCP\b|Google Cloud", "GCP", "tech"),
    (r"\bAzure\b", "Azure", "tech"),
    (r"\bDocker\b", "Docker", "tech"),
    (r"\bKubernetes\b|\bK8s\b", "Kubernetes", "tech"),
    (r"\bTerraform\b", "Terraform", "tech"),
    (r"\bCI/CD\b", "CI/CD", "tech"),
    (r"\bJenkins\b", "Jenkins", "tech"),
    (r"\bGitHub Actions\b", "GitHub Actions", "tech"),
    (r"\bGitLab CI\b", "GitLab CI", "tech"),
    (r"\bAnsible\b", "Ansible", "tech"),
    (r"\bLinux\b", "Linux", "tech"),
    (r"\bBash\b", "Bash", "tech"),
    # Data / ML
    (r"\bMachine Learning\b|\bML\b", "Machine Learning", "tech"),
    (r"\bDeep Learning\b", "Deep Learning", "tech"),
    (r"\bNLP\b|Natural Language Processing", "NLP", "tech"),
    (r"\bComputer Vision\b", "Computer Vision", "tech"),
    (r"\bTensorFlow\b", "TensorFlow", "tech"),
    (r"\bPyTorch\b", "PyTorch", "tech"),
    (r"\bscikit-learn\b|\bsklearn\b", "scikit-learn", "tech"),
    (r"\bPandas\b", "Pandas", "tech"),
    (r"\bNumPy\b", "NumPy", "tech"),
    (r"\bSpark\b|Apache Spark", "Spark", "tech"),
    (r"\bAirflow\b", "Airflow", "tech"),
    (r"\bdbt\b", "dbt", "tech"),
    (r"\bTableau\b", "Tableau", "tech"),
    (r"\bPower BI\b", "Power BI", "tech"),
    # Testing
    (r"\bJest\b", "Jest", "tech"),
    (r"\bMocha\b", "Mocha", "tech"),
    (r"\bPytest\b", "Pytest", "tech"),
    (r"\bCypress\b", "Cypress", "tech"),
    (r"\bSelenium\b", "Selenium", "tech"),
    (r"\bReact Testing Library\b", "React Testing Library", "tech"),
    # Mobile
    (r"\biOS\b", "iOS", "tech"),
    (r"\bAndroid\b", "Android", "tech"),
    (r"\bSwift\b", "Swift", "tech"),
    (r"\bKotlin\b", "Kotlin", "tech"),
    (r"\bFlutter\b", "Flutter", "tech"),
    # Soft skills
    (r"\bleadership\b", "Leadership", "soft"),
    (r"\bmentoring\b|\bmentor\b", "Mentoring", "soft"),
    (r"\bcommunication skills\b", "Communication", "soft"),
    (r"\bcollaboration\b", "Collaboration", "soft"),
    (r"\bproblem solving\b", "Problem Solving", "soft"),
    (r"\bagile\b", "Agile", "soft"),
    (r"\bscrum\b", "Scrum", "soft"),
    (r"\bproject management\b", "Project Management", "soft"),
    # Certifications
    (r"\bAWS Certified\b", "AWS Certified", "certification"),
    (r"\bAzure Certified\b", "Azure Certified", "certification"),
    (r"\bPMP\b", "PMP", "certification"),
    (r"\bCKA\b|Certified Kubernetes Administrator", "CKA", "certification"),
    # Languages (spoken)
    (r"\bEnglish\b", "English", "language"),
    (r"\bSpanish\b", "Spanish", "language"),
    (r"\bMandarin\b", "Mandarin", "language"),
    (r"\bFrench\b", "French", "language"),
    (r"\bGerman\b", "German", "language"),
]

# Phrases that signal a "must-have" requirement
_MUST_HAVE_SIGNALS = [
    "required", "must have", "must-have", "essential", "mandatory",
    "minimum requirements", "minimum qualification", "you have",
    "you need", "minimum of", "at least", "expert in", "expert-level",
    "proficiency in", "proficient in", "strong experience", "years of experience",
    "required qualification", "requirements",
]

# Phrases that signal a "nice-to-have" requirement
_NICE_TO_HAVE_SIGNALS = [
    "preferred", "nice to have", "nice-to-have", "bonus", "a plus",
    "advantageous", "optional", "preferred qualification", "preferred but",
    "would be a", "is a plus", "are a plus", "ideal candidate", "ideally",
]

# Section headers commonly used in JDs
_MUST_HAVE_HEADERS = [
    "required qualifications", "required skills", "requirements",
    "minimum requirements", "must have", "must-have", "essential qualifications",
    "what you'll need", "what you need", "basic qualifications",
]
_NICE_TO_HAVE_HEADERS = [
    "preferred qualifications", "preferred skills", "nice to have",
    "nice-to-have", "bonus qualifications", "bonus points", "preferred but not required",
    "preferred experience", "ideal candidate", "ideal qualifications",
]


def _keyword_fallback_extract(jd: str) -> ExtractedRequirements:
    """Deterministic keyword-based JD parser (no LLM required).

    Strategy:
      1. Split the JD into "must-have" and "nice-to-have" sections by
         detecting common section headers.
      2. Within each section, scan for known skills via the curated
         _SKILL_DICTIONARY (regex-based, case-insensitive).
      3. Extract minimum years of experience via regex.
      4. Extract education level via regex.
      5. Extract domain keywords from the JD title (first non-empty line).

    This is intentionally conservative — we'd rather miss a skill than
    invent one. The LLM extractor handles the long tail of unusual skills.
    """
    jd_lower = jd.lower()

    # 1. Split into must-have / nice-to-have sections
    must_section, nice_section = _split_jd_sections(jd)

    # 2. Extract skills from each section
    must_skills = _scan_for_skills(must_section or jd)
    nice_skills = _scan_for_skills(nice_section) if nice_section else []

    # If we found nothing in must_section, fall back to scanning the
    # entire JD with must_have classification (since the JD doesn't have
    # explicit section headers).
    if not must_skills:
        must_skills = _scan_for_skills(jd)
        # Filter the nice list to avoid duplicates with must
        must_names = {s.skill.lower() for s in must_skills}
        nice_skills = [s for s in nice_skills if s.skill.lower() not in must_names]

    # De-duplicate (keep first occurrence)
    must_skills = _dedupe_skills(must_skills)
    nice_skills = _dedupe_skills(nice_skills)
    # Remove nice-to-have items that are also in must-have
    must_names = {s.skill.lower() for s in must_skills}
    nice_skills = [s for s in nice_skills if s.skill.lower() not in must_names]

    # 3. Experience years
    experience = _extract_experience_years(jd)

    # 4. Education level
    education = _extract_education(jd)

    # 5. Domain keywords from title + first paragraph
    domain_keywords = _extract_domain_keywords(jd)

    return ExtractedRequirements(
        must_have=must_skills,
        nice_to_have=nice_skills,
        experience_min_years=experience,
        education_level=education,
        domain_keywords=domain_keywords,
    )


def _split_jd_sections(jd: str) -> tuple[str, str]:
    """Split JD into (must_have_section, nice_to_have_section) by headers.

    Returns empty strings for sections that aren't found.
    """
    lines = jd.split("\n")
    must_lines: list[str] = []
    nice_lines: list[str] = []

    current_section: str | None = None  # "must" | "nice" | None
    for line in lines:
        stripped = line.strip().lower().rstrip(":")
        # Detect section transitions
        if any(stripped == h or stripped.startswith(h) for h in _NICE_TO_HAVE_HEADERS):
            current_section = "nice"
            continue
        if any(stripped == h or stripped.startswith(h) for h in _MUST_HAVE_HEADERS):
            current_section = "must"
            continue
        # Detect inline "preferred" markers as nice-to-have
        if any(sig in stripped for sig in _NICE_TO_HAVE_SIGNALS) and current_section is None:
            nice_lines.append(line)
            continue

        if current_section == "must":
            must_lines.append(line)
        elif current_section == "nice":
            nice_lines.append(line)
        else:
            # No section context yet — defer to global must/nice scan later
            must_lines.append(line)  # default to must_have bucket

    return "\n".join(must_lines), "\n".join(nice_lines)


def _scan_for_skills(text: str) -> list[SkillRequirement]:
    """Scan text for known skills and return as SkillRequirement objects."""
    if not text:
        return []
    found: list[SkillRequirement] = []
    seen_names: set[str] = set()

    for pattern, name, skill_type in _SKILL_DICTIONARY:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and name.lower() not in seen_names:
            # Extract evidence: a window of ~80 chars around the match
            start = max(0, match.start() - 40)
            end = min(len(text), match.end() + 40)
            evidence = text[start:end].strip().replace("\n", " ")
            # Truncate to a clean sentence boundary if possible
            if len(evidence) > 120:
                evidence = evidence[:120].rsplit(" ", 1)[0] + "…"

            # Determine weight: must_have items default to 1.0,
            # nice_to_have items default to 0.5
            weight = 1.0 if skill_type != "soft" else 0.8
            found.append(SkillRequirement(
                skill=name,
                type=skill_type,  # type: ignore[arg-type]
                weight=weight,
                evidence=evidence or f"Found in JD: {name}",
            ))
            seen_names.add(name.lower())

    return found


def _dedupe_skills(skills: list[SkillRequirement]) -> list[SkillRequirement]:
    """Remove duplicates by skill name (case-insensitive), keeping first."""
    seen: set[str] = set()
    out: list[SkillRequirement] = []
    for s in skills:
        key = s.skill.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _extract_experience_years(jd: str) -> int | None:
    """Extract minimum years of experience from the JD."""
    # Patterns like "5+ years", "5 years", "3-5 years", "minimum 5 years"
    patterns = [
        r"(?:minimum\s+)?(\d+)\s*\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?experience",
        r"(\d+)\s*\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?experience",
        r"experience[:\s]+(?:minimum\s+)?(\d+)\s*\+?\s*years?",
        r"(\d+)\s*-\s*\d+\s*years?",  # "3-5 years" → take 3
    ]
    for pat in patterns:
        match = re.search(pat, jd, re.IGNORECASE)
        if match:
            try:
                years = int(match.group(1))
                if 0 <= years <= 30:  # sanity check
                    return years
            except (ValueError, IndexError):
                continue
    return None


def _extract_education(jd: str) -> str | None:
    """Extract required education level."""
    jd_lower = jd.lower()
    if "phd" in jd_lower or "doctorate" in jd_lower:
        return "PhD"
    if "master" in jd_lower or "ms degree" in jd_lower or "m.s." in jd_lower:
        return "MS"
    if "bachelor" in jd_lower or "bs degree" in jd_lower or "b.s." in jd_lower or "undergraduate degree" in jd_lower:
        return "BS"
    if "high school" in jd_lower:
        return "High School"
    return None


def _extract_domain_keywords(jd: str) -> list[str]:
    """Extract domain keywords from the JD title (first non-empty line)."""
    lines = [l.strip() for l in jd.split("\n") if l.strip()]
    if not lines:
        return []
    title = lines[0]
    # Take significant words from the title (length > 3, alphabetic)
    words = re.findall(r"\b[A-Za-z]{4,}\b", title)
    # Filter out generic words
    generic = {"senior", "junior", "lead", "principal", "staff", "engineer", "developer",
                "manager", "specialist", "analyst", "designer", "architect"}
    keywords = [w.lower() for w in words if w.lower() not in generic]
    # Dedupe, limit to 5
    seen: set[str] = set()
    out: list[str] = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            out.append(k)
        if len(out) >= 5:
            break
    return out