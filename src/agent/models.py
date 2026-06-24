"""
Agentic Profile Matching — Pydantic Models for LLM Structured Outputs.

These models are used for:
  1. Validating LLM tool-call arguments (input/output)
  2. Structured output parsing via ``with_structured_output()``
  3. Serialization/deserialization round-trips across the pipeline

Architecture Reference: architecture.md Section 3, Section 6 (Tool Registry)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Requirement Extraction Models
# ---------------------------------------------------------------------------


class SkillRequirement(BaseModel):
    """A single skill extracted from a job description.

    Attributes:
        skill:      The human-readable skill name (e.g. "React", "Leadership").
        type:       Skill category — one of "tech", "soft", "domain",
                    "certification", "language", "other".
        weight:     Importance weight in [0.0, 1.0]. 1.0 = critical.
        evidence:   The exact JD excerpt that justified this requirement.
    """

    skill: str = Field(..., min_length=1, description="Skill name")
    type: Literal["tech", "soft", "domain", "certification", "language", "other"] = Field(
        ..., description="Skill category"
    )
    weight: float = Field(..., ge=0.0, le=1.0, description="Importance weight 0.0-1.0")
    evidence: str = Field(..., min_length=1, description="JD excerpt justifying this requirement")


class ExtractedRequirements(BaseModel):
    """Structured output from the ``extract_requirements`` tool.

    The LLM parses a raw JD into must-have and nice-to-have skill lists
    along with scalar constraints (experience, education) and domain keywords.
    """

    must_have: list[SkillRequirement] = Field(
        default_factory=list, description="Non-negotiable requirements"
    )
    nice_to_have: list[SkillRequirement] = Field(
        default_factory=list, description="Preferred but optional requirements"
    )
    experience_min_years: int | None = Field(
        default=None, ge=0, description="Minimum years of experience required"
    )
    education_level: str | None = Field(
        default=None, description="Required education level (e.g. BS, MS, PhD)"
    )
    domain_keywords: list[str] = Field(
        default_factory=list, description="Industry/domain keywords for RAG boosting"
    )


# ---------------------------------------------------------------------------
# Candidate Scoring Models
# ---------------------------------------------------------------------------


class CandidateScore(BaseModel):
    """Structured output from per-candidate scoring.

    Each candidate is evaluated against the extracted requirements,
    producing sub-scores, reasoning, and evidence.
    """

    must_have_score: float = Field(..., ge=0.0, le=1.0, description="Score on must-have criteria")
    nice_to_have_score: float = Field(
        ..., ge=0.0, le=1.0, description="Score on nice-to-have criteria"
    )
    reasoning: str = Field(..., min_length=1, description="LLM-generated match explanation")
    strengths: list[str] = Field(default_factory=list, description="Candidate's matched strengths")
    gaps: list[str] = Field(default_factory=list, description="Missing or weak areas")
    excerpts: list[str] = Field(
        default_factory=list, description="Evidence snippets from the resume"
    )

    @property
    def composite_score(self) -> float:
        """Weighted composite: 0.7 * must_have + 0.3 * nice_to_have."""
        return 0.7 * self.must_have_score + 0.3 * self.nice_to_have_score


# ---------------------------------------------------------------------------
# Interview Question Generation
# ---------------------------------------------------------------------------


class InterviewQuestion(BaseModel):
    """A single interview question targeting a candidate's gap.

    Attributes:
        question:      The question text.
        category:      One of "technical", "behavioral", "situational", "domain".
        targets_gap:   Which gap or weakness this question probes.
        difficulty:    One of "easy", "medium", "hard".
        follow_ups:    Suggested follow-up questions for deeper probing.
    """

    question: str = Field(..., min_length=1, description="The interview question")
    category: Literal["technical", "behavioral", "situational", "domain"] = Field(
        ..., description="Question category"
    )
    targets_gap: str = Field(
        ..., min_length=1, description="Which gap this question probes"
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ..., description="Difficulty level"
    )
    follow_ups: list[str] = Field(
        default_factory=list, description="Suggested follow-up questions"
    )


# ---------------------------------------------------------------------------
# Candidate Comparison
# ---------------------------------------------------------------------------


class ComparisonRow(BaseModel):
    """One row in a head-to-head comparison table.

    Attributes:
        criterion:  The comparison dimension (e.g. "React experience").
        values:     One value per candidate, in order.
    """

    criterion: str = Field(..., min_length=1, description="Comparison dimension")
    values: list[str] = Field(
        ..., min_length=2, description="One value per candidate being compared"
    )


class ComparisonResult(BaseModel):
    """Structured output from the ``compare_candidates`` tool.

    Produces a comparison table plus a narrative summary.
    """

    candidates: list[dict] = Field(
        ..., min_length=2, description="Candidate summaries being compared"
    )
    comparison_table: dict[str, list[str]] = Field(
        ..., description="Mapping from criterion to list of per-candidate values"
    )
    summary: str = Field(..., min_length=1, description="Narrative comparison summary")

    @field_validator("comparison_table")
    @classmethod
    def table_values_consistent_length(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        """All value lists must have the same length (one per candidate)."""
        if not v:
            return v
        lengths = {len(vals) for vals in v.values()}
        if len(lengths) > 1:
            raise ValueError(
                f"All comparison_table value lists must have the same length, "
                f"got lengths: {lengths}"
            )
        return v