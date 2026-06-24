"""
Agentic Profile Matching — Red Flag Detection.

Detects employment-related red flags in candidate resumes:
  - Unexplained employment gaps (>3 months)
  - Job-hopping pattern (>3 roles in 24 months)
  - Inconsistent dates or conflicting titles

Architecture Reference: architecture.md Section 9 (Round 2 — Red-flag detection)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

MAX_GAP_MONTHS = 3
MAX_ROLES_IN_PERIOD = 3
MAX_PERIOD_MONTHS = 24


@dataclass
class RedFlag:
    """A single detected red flag."""

    flag_type: str  # "employment_gap" | "job_hopping" | "inconsistency"
    description: str
    severity: str  # "low" | "medium" | "high"
    evidence: str  # the text snippet that triggered this flag

    def to_dict(self) -> dict:
        return asdict(self)


def detect_employment_gaps(
    resume_text: str,
    max_gap_months: int = MAX_GAP_MONTHS,
) -> list[RedFlag]:
    """Detect unexplained employment gaps in a resume.

    Looks for date ranges like "Jan 2021 – Mar 2021" and checks
    for gaps between consecutive roles > max_gap_months.

    Args:
        resume_text: Full resume text.
        max_gap_months: Maximum gap (in months) before flagging.

    Returns:
        List of RedFlag objects for detected gaps.
    """
    flags: list[RedFlag] = []

    # Extract date ranges from resume text
    # Pattern matches: "Jan 2021 - Mar 2021", "January 2021 to March 2021",
    # "(Jan 2021 - Mar 2021)" — handles optional parentheses around dates
    _MONTH_RE = (
        r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|'
        r'Nov(?:ember)?|Dec(?:ember)?)'
    )
    date_pattern = re.compile(
        r'\(?\s*(' + _MONTH_RE + r')\s+(\d{4})'
        r'\s*[-–to]+\s*'
        r'(' + _MONTH_RE + r')\s+(\d{4})'
        r'\s*\)?',
        re.IGNORECASE,
    )

    ranges: list[tuple[int, int, str]] = []  # (start_month_num, end_month_num, evidence)
    for match in date_pattern.finditer(resume_text):
        try:
            start_month = _month_to_num(match.group(1))
            start_year = int(match.group(2))
            end_month = _month_to_num(match.group(3))
            end_year = int(match.group(4))

            start_total = start_year * 12 + start_month
            end_total = end_year * 12 + end_month
            evidence = match.group(0)

            ranges.append((start_total, end_total, evidence))
        except (ValueError, IndexError):
            continue

    # Sort by start date
    ranges.sort(key=lambda r: r[0])

    # Check gaps between consecutive roles
    for i in range(1, len(ranges)):
        prev_end = ranges[i - 1][1]
        curr_start = ranges[i][0]
        gap_months = curr_start - prev_end

        if gap_months > max_gap_months:
            severity = "high" if gap_months > 6 else "medium"
            flags.append(RedFlag(
                flag_type="employment_gap",
                description=(
                    f"Employment gap of {gap_months} months detected "
                    f"between roles"
                ),
                severity=severity,
                evidence=f"{ranges[i-1][2]} -> {ranges[i][2]}",
            ))

    return flags


def detect_job_hopping(
    resume_text: str,
    max_roles_in_period: int = MAX_ROLES_IN_PERIOD,
    max_period_months: int = MAX_PERIOD_MONTHS,
) -> list[RedFlag]:
    """Detect job-hopping pattern (too many roles in a short period).

    Flags if there are more than max_roles_in_period distinct roles
    within a sliding window of max_period_months months.

    Args:
        resume_text: Full resume text.
        max_roles_in_period: Maximum roles before flagging.
        max_period_months: Time window in months.

    Returns:
        List of RedFlag objects for detected job-hopping.
    """
    flags: list[RedFlag] = []

    # Extract all year mentions near role-related words
    # Pattern: "Role at Company (2020-2021)" or "2020 - 2021"
    role_year_pattern = re.compile(
        r'(?:Senior\s+|Junior\s+|Lead\s+)?'
        r'(\w+(?:\s+\w+)?)\s+at\s+([\w\s]+?)\s*[\(]?\s*(\d{4})\s*[-–]\s*(\d{4})',
        re.IGNORECASE,
    )

    roles: list[tuple[str, str, int, int]] = []  # (role, company, start_year, end_year)
    for match in role_year_pattern.finditer(resume_text):
        try:
            role = match.group(1).strip()
            company = match.group(2).strip()
            start_year = int(match.group(3))
            end_year = int(match.group(4))
            if role and company and start_year and end_year:
                roles.append((role, company, start_year, end_year))
        except (ValueError, IndexError):
            continue

    if len(roles) <= max_roles_in_period:
        return flags

    # Check sliding windows
    for i in range(len(roles)):
        window_roles = set()
        window_end_year = roles[i][3]  # end year of role i

        for j in range(len(roles)):
            role_start = roles[j][2]
            if abs(role_start - window_end_year) <= 2:
                window_roles.add(f"{roles[j][0]} at {roles[j][1]}")

        if len(window_roles) > max_roles_in_period:
            role_descriptions = sorted(window_roles)[:5]
            flags.append(RedFlag(
                flag_type="job_hopping",
                description=(
                    f"Job-hopping pattern: {len(window_roles)} distinct roles "
                    f"in approximately {max_period_months} months"
                ),
                severity="high",
                evidence=", ".join(role_descriptions),
            ))
            break  # Only flag once

    return flags


def detect_inconsistencies(
    resume_text: str,
) -> list[RedFlag]:
    """Detect inconsistencies in resume dates or claims.

    Checks for:
    - Impossible date ranges (end year before start year)

    Args:
        resume_text: Full resume text.

    Returns:
        List of RedFlag objects for detected inconsistencies.
    """
    flags: list[RedFlag] = []

    # Extract date ranges
    date_range_pattern = re.compile(
        r'(\d{4})\s*[-–]\s*(\d{4})',
    )

    for match in date_range_pattern.finditer(resume_text):
        try:
            start_year = int(match.group(1))
            end_year = int(match.group(2))

            if end_year < start_year:
                flags.append(RedFlag(
                    flag_type="inconsistency",
                    description=(
                        f"Impossible date range: {start_year} to {end_year} "
                        f"(end year is before start year)"
                    ),
                    severity="medium",
                    evidence=match.group(0),
                ))
        except (ValueError, IndexError):
            continue

    # Deduplicate by description
    seen: set[str] = set()
    unique_flags: list[RedFlag] = []
    for f in flags:
        if f.description not in seen:
            seen.add(f.description)
            unique_flags.append(f)

    return unique_flags


def detect_red_flags(
    resume_text: str,
    timeline: list[dict] | None = None,
) -> list[RedFlag]:
    """Run all red-flag detection functions and return combined results.

    Args:
        resume_text: Full resume text.
        timeline: Optional pre-extracted timeline (not used in simplified
                  version, kept for API compatibility).

    Returns:
        List of all detected RedFlag objects.
    """
    all_flags: list[RedFlag] = []

    all_flags.extend(detect_employment_gaps(resume_text))
    all_flags.extend(detect_job_hopping(resume_text))
    all_flags.extend(detect_inconsistencies(resume_text))

    return all_flags


def _month_to_num(month_str: str) -> int:
    """Convert month name or abbreviation to number (1-12).

    Args:
        month_str: Month name like "January", "Jan", "01".

    Returns:
        Month number 1-12, or 0 if unrecognised.
    """
    month_map = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12,
    }
    return month_map.get(month_str.strip().lower(), 0)