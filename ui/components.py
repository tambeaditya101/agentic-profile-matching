"""
Agentic Profile Matching — Shared UI Components.

Format-agnostic rendering helpers used by both the Streamlit chat
interface and the CLI REPL. Each function returns a Markdown string
(which Streamlit renders via ``st.markdown`` and the CLI prints as
plain text after optional ANSI colorization).

Components cover:
  - Job requirements panel
  - Screening round progress
  - Shortlist table
  - Per-candidate match report
  - Side-by-side comparison table
  - Refinement ranking delta
  - Interview question list
  - Ranking explanation
  - Natural-language agent response builder

Architecture Reference: architecture.md Section 14.3 (Shared Components)
Phase: 7 — User Interface
"""

from __future__ import annotations

from typing import Any


# =====================================================================
# Color helpers (used by CLI; Streamlit ignores them)
# =====================================================================

# ANSI escape codes — the CLI app wraps strings with these; the Streamlit
# app receives plain markdown and ignores them.
_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}


def colorize(text: str, color: str) -> str:
    """Wrap ``text`` in ANSI color codes. No-op if color unknown."""
    code = _ANSI.get(color, "")
    if not code:
        return text
    return f"{code}{text}{_ANSI['reset']}"


# =====================================================================
# Status helpers
# =====================================================================

def score_status_label(score: float) -> str:
    """Return a short human-readable status label for a 0..1 score."""
    if score >= 0.8:
        return "STRONG MATCH"
    if score >= 0.6:
        return "GOOD MATCH"
    if score >= 0.4:
        return "PARTIAL"
    if score >= 0.2:
        return "WEAK"
    return "NO MATCH"


def recommendation_label(rec: str) -> str:
    """Normalize hire recommendation labels for display."""
    mapping = {
        "hire": "STRONG HIRE",
        "no_hire": "NO HIRE",
        "borderline": "BORDERLINE",
        "unknown": "PENDING",
    }
    return mapping.get(rec, (rec or "PENDING").upper())


# =====================================================================
# Job requirements panel
# =====================================================================

def format_requirements_panel(requirements: dict | None) -> str:
    """Render the requirements panel (sidebar).

    Args:
        requirements: Requirements dict from AgentState, may be None/empty.

    Returns:
        Markdown string showing must-have / nice-to-have / experience /
        education / domain keywords.
    """
    if not requirements:
        return "_No requirements extracted yet. Upload a JD to begin._"

    must = requirements.get("must_have", []) or []
    nice = requirements.get("nice_to_have", []) or []
    exp = requirements.get("experience_min_years")
    edu = requirements.get("education_level")
    keywords = requirements.get("domain_keywords", []) or []

    lines: list[str] = []
    lines.append("**Must-Have**")
    if must:
        for item in must:
            skill = item.get("skill", "?")
            weight = item.get("weight", 1.0)
            lines.append(f"- ✅ {skill} _(weight {weight:.2f})_")
    else:
        lines.append("- _none_")

    lines.append("")
    lines.append("**Nice-to-Have**")
    if nice:
        for item in nice:
            skill = item.get("skill", "?")
            weight = item.get("weight", 0.5)
            lines.append(f"- ⭕ {skill} _(weight {weight:.2f})_")
    else:
        lines.append("- _none_")

    if exp is not None:
        lines.append("")
        lines.append(f"**Experience:** {exp}+ years")
    if edu:
        lines.append(f"**Education:** {edu}")
    if keywords:
        lines.append(f"**Domain:** {', '.join(keywords)}")

    return "\n".join(lines)


# =====================================================================
# Screening progress
# =====================================================================

def format_screening_progress(rounds: list[dict] | None) -> str:
    """Render the screening-round progress indicator.

    Args:
        rounds: list of ScreeningRound dicts from AgentState.

    Returns:
        Markdown string showing 3 round rows with completion state.
    """
    rounds = rounds or []
    by_num = {r.get("round_number"): r for r in rounds if r.get("round_number")}

    rows: list[str] = ["**Screening Pipeline**"]
    labels = [
        (1, "Initial", "Broad RAG retrieval + must-have keyword filter → top 10"),
        (2, "Deep", "Full resume review, skill verification, red flags → top 5–7"),
        (3, "Final", "Hire/no-hire recommendation with supporting evidence"),
    ]
    for num, name, desc in labels:
        r = by_num.get(num)
        if r is None:
            mark = "○"
            stat = "pending"
        else:
            mark = "✓"
            eval_count = r.get("candidates_evaluated", 0)
            shortlisted = len(r.get("shortlisted_ids", []) or [])
            stat = f"{eval_count} evaluated → {shortlisted} shortlisted"
        rows.append(f"{mark} **Round {num} — {name}**: {stat}")
        rows.append(f"  - _{desc}_")

    return "\n".join(rows)


# =====================================================================
# Shortlist table
# =====================================================================

def format_shortlist_table(shortlist: list[dict] | None, top_n: int | None = None) -> str:
    """Render the ranked shortlist as a markdown table.

    Args:
        shortlist: list of CandidateMatch dicts.
        top_n: if set, only show the top N candidates.

    Returns:
        Markdown table with rank, name, composite score, must-have score,
        nice-to-have score, recommendation.
    """
    if not shortlist:
        return "_No candidates in the shortlist yet._"

    items = shortlist[:top_n] if top_n else shortlist
    lines = [
        "| # | Candidate | Composite | Must | Nice | Recommendation |",
        "|---|-----------|-----------|------|------|----------------|",
    ]
    for i, c in enumerate(items, 1):
        name = c.get("name", c.get("candidate_id", "?"))
        composite = c.get("score", 0.0)
        must = c.get("must_have_score", 0.0)
        nice = c.get("nice_to_have_score", 0.0)
        rec = recommendation_label(c.get("hire_recommendation", "unknown"))
        lines.append(
            f"| {i} | {name} | {composite:.2f} | {must:.2f} | {nice:.2f} | {rec} |"
        )
    return "\n".join(lines)


# =====================================================================
# Per-candidate report
# =====================================================================

def render_match_report(report_md: str | None) -> str:
    """Render a pre-generated per-candidate match report.

    The report is already in markdown form (produced by
    ``src.reports.match_report.generate_match_report``). We pass it
    through unchanged so both Streamlit and CLI render identically.

    Args:
        report_md: Markdown string of the report.

    Returns:
        The same markdown string, or a placeholder if missing.
    """
    if not report_md:
        return "_No report available for this candidate._"
    return report_md


# =====================================================================
# Comparison table
# =====================================================================

def render_comparison_table(comparison_result: dict | None) -> str:
    """Render a head-to-head candidate comparison.

    Handles both LLM-generated structured comparisons and the fallback
    comparison produced by ``_fallback_comparison``.

    Args:
        comparison_result: dict from AgentState.comparison_result.

    Returns:
        Markdown with comparison table + narrative summary.
    """
    if not comparison_result:
        return "_No comparison available._"

    result_type = comparison_result.get("type", "comparison")
    summary = comparison_result.get("summary", "")

    # Refinement delta has its own renderer
    if result_type == "refinement_delta":
        return render_ranking_delta(comparison_result)

    # Single-candidate or error message — just show the summary
    if result_type in ("single_candidate", "error"):
        return f"_{summary}_"

    # Full comparison
    candidates = comparison_result.get("candidates", []) or []
    table = comparison_result.get("comparison_table", {}) or {}

    lines: list[str] = []

    if candidates:
        names = [c.get("name", c.get("id", "?")) for c in candidates]
        lines.append(f"### Comparing {', '.join(names)}")
        lines.append("")

    if table:
        # Build a markdown table whose first column is the criterion
        # and subsequent columns are one per candidate.
        # The fallback format stores lists like:
        #   {"Overall Score": ["0.85", "0.70"], ...}
        # The LLM ComparisonResult stores:
        #   {"criterion_a": {"value": "...", "candidate_id": "..."}, ...}
        # We try the list shape first; fall back to dict-of-dicts.
        first_key = next(iter(table))
        sample_val = table[first_key]
        if isinstance(sample_val, list):
            n_cols = len(sample_val)
            header = "| Criterion | " + " | ".join(
                [names[i] if i < len(names) else f"C{i+1}" for i in range(n_cols)]
            ) + " |"
            sep = "|---" * (n_cols + 1) + "|"
            lines.append(header)
            lines.append(sep)
            for crit, vals in table.items():
                if isinstance(vals, list):
                    row_vals = " | ".join(str(v) for v in vals)
                    lines.append(f"| {crit} | {row_vals} |")
        else:
            # dict-of-dicts shape (LLM ComparisonRow schema)
            lines.append("| Criterion | Value | Candidate |")
            lines.append("|---|---|---|")
            for crit, entry in table.items():
                if isinstance(entry, dict):
                    val = entry.get("value", "")
                    cid = entry.get("candidate_id", "?")
                    lines.append(f"| {crit} | {val} | {cid} |")
                else:
                    lines.append(f"| {crit} | {entry} | - |")
        lines.append("")

    if summary:
        lines.append("**Summary**")
        lines.append(summary)

    return "\n".join(lines) if lines else f"_{summary or 'No comparison data.'}_"


# =====================================================================
# Refinement delta
# =====================================================================

def render_ranking_delta(comparison_result: dict | None) -> str:
    """Render the ranking delta produced after a requirement refinement.

    Args:
        comparison_result: dict from AgentState.comparison_result with
            ``type == "refinement_delta"`` and a ``summary`` field.

    Returns:
        Markdown string with the delta summary.
    """
    if not comparison_result:
        return "_No refinement delta available._"

    if comparison_result.get("type") != "refinement_delta":
        # If given a generic comparison_result, just emit the summary
        return comparison_result.get("summary", "_No delta available._")

    summary = comparison_result.get("summary", "_No changes detected._")
    lines = [
        "### Ranking Updated",
        "",
        summary,
    ]
    return "\n".join(lines)


# =====================================================================
# Interview questions
# =====================================================================

def render_questions(comparison_result: dict | None) -> str:
    """Render interview questions for a candidate.

    Args:
        comparison_result: dict with ``type == "questions"`` containing
            ``candidate_name`` and either ``questions`` (list of dicts) or
            ``summary`` (pre-formatted text).

    Returns:
        Markdown bullet list of questions with category and difficulty.
    """
    if not comparison_result:
        return "_No questions available._"

    name = comparison_result.get("candidate_name", "the candidate")
    questions = comparison_result.get("questions") or []

    lines = [f"### Interview Questions for {name}", ""]

    if not questions:
        summary = comparison_result.get("summary")
        if summary:
            lines.append(summary)
        else:
            lines.append("_Could not generate questions._")
        return "\n".join(lines)

    for i, q in enumerate(questions, 1):
        if isinstance(q, dict):
            question = q.get("question", "?")
            category = q.get("category", "general")
            difficulty = q.get("difficulty", "")
            targets = q.get("targets_gap", "")
            follow_ups = q.get("follow_ups", []) or []

            tag = f"[{category.upper()}"
            if difficulty:
                tag += f" / {difficulty}"
            tag += "]"
            lines.append(f"**{i}. {tag}** {question}")
            if targets:
                lines.append(f"   - _Targets gap: {targets}_")
            if follow_ups:
                lines.append("   - _Follow-ups:_")
                for fu in follow_ups:
                    lines.append(f"     - {fu}")
            lines.append("")
        else:
            lines.append(f"{i}. {q}")
            lines.append("")

    return "\n".join(lines)


# =====================================================================
# Ranking explanation
# =====================================================================

def render_explanation(comparison_result: dict | None) -> str:
    """Render a ranking explanation.

    Args:
        comparison_result: dict with ``type == "explanation"`` containing
            ``higher_candidate``, ``lower_candidate``, and ``summary``.

    Returns:
        Markdown explanation string.
    """
    if not comparison_result:
        return "_No explanation available._"

    higher = comparison_result.get("higher_candidate", "")
    lower = comparison_result.get("lower_candidate", "")
    summary = comparison_result.get("summary", "")

    lines: list[str] = []
    if higher and lower:
        lines.append(f"### Why {higher} ranked higher than {lower}")
        lines.append("")
    if summary:
        lines.append(summary)
    else:
        lines.append("_No explanation available._")

    return "\n".join(lines)


# =====================================================================
# Natural-language agent response (built from state)
# =====================================================================

def build_agent_response(state: dict) -> str:
    """Build a natural-language agent response from the latest state.

    Called by both UIs after each graph invocation. Picks the most
    relevant information from the state and renders it as markdown.

    Decision logic:
      1. If state.error is set → return error message
      2. If comparison_result.type is set → render that
      3. If shortlist is non-empty and just produced → render shortlist
      4. If awaiting_human_feedback and shortlist exists → show prompt
      5. Otherwise → generic status

    Args:
        state: AgentState dict after the latest graph invocation.

    Returns:
        Markdown string for display in the chat area.
    """
    error = state.get("error")
    if error:
        return f"⚠️ **Error:** {error}"

    comparison = state.get("comparison_result")
    if comparison and isinstance(comparison, dict):
        ctype = comparison.get("type", "")
        if ctype == "refinement_delta":
            return render_ranking_delta(comparison)
        if ctype == "explanation":
            return render_explanation(comparison)
        if ctype == "questions":
            return render_questions(comparison)
        if ctype in ("single_candidate", "error"):
            return f"_{comparison.get('summary', '')}_"
        # default: full comparison
        return render_comparison_table(comparison)

    shortlist = state.get("current_shortlist") or []
    if shortlist:
        rounds = state.get("screening_rounds") or []
        reports = state.get("generated_reports") or {}
        lines: list[str] = []

        # If just completed initial pipeline, announce it
        if state.get("awaiting_human_feedback") and rounds:
            r_counts = ", ".join(
                f"R{r.get('round_number')}: {len(r.get('shortlisted_ids', []) or [])}"
                for r in rounds
            )
            lines.append(
                f"Pipeline complete. Screening rounds: {r_counts}. "
                f"**Top {len(shortlist)} candidates:**"
            )
            lines.append("")
        elif state.get("awaiting_human_feedback"):
            lines.append(f"Found **{len(shortlist)}** candidates:")
            lines.append("")

        lines.append(format_shortlist_table(shortlist))

        if reports:
            lines.append("")
            lines.append(
                f"_Full match reports generated for {len(reports)} candidate(s). "
                "Use 'show report for <name>' or open the sidebar to view._"
            )
        return "\n".join(lines)

    return "_No response yet. Please upload a JD to start the pipeline._"


# =====================================================================
# Quick-action suggestion chips
# =====================================================================

SUGGESTED_PROMPTS: list[dict[str, str]] = [
    {
        "label": "Compare top 3",
        "message": "Compare the top 3 candidates side by side",
    },
    {
        "label": "Why top 2?",
        "message": "Why did the top candidate rank higher than the second?",
    },
    {
        "label": "Interview questions",
        "message": "Generate interview questions for the top candidate",
    },
    {
        "label": "Drop AWS",
        "message": "Drop the AWS requirement and re-rank",
    },
    {
        "label": "Add TypeScript",
        "message": "Add TypeScript as a must-have skill",
    },
    {
        "label": "Show report",
        "message": "Show the full match report for the top candidate",
    },
    {
        "label": "Done",
        "message": "done",
    },
]


def get_suggested_prompts() -> list[dict[str, str]]:
    """Return the list of quick-action prompt suggestions."""
    return list(SUGGESTED_PROMPTS)


# =====================================================================
# Reports export
# =====================================================================

def export_reports_to_dir(state: dict, output_dir: str) -> list[str]:
    """Write all generated reports to ``output_dir`` as markdown files.

    Args:
        state: AgentState with ``generated_reports`` populated.
        output_dir: Directory to write files into (created if missing).

    Returns:
        List of file paths written.
    """
    import os
    from pathlib import Path

    reports = state.get("generated_reports") or {}
    if not reports:
        return []

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    shortlist = state.get("current_shortlist") or []
    name_by_id = {c.get("candidate_id", ""): c.get("name", c.get("candidate_id", "?")) for c in shortlist}

    for cid, md in reports.items():
        name = name_by_id.get(cid, cid)
        # sanitize filename
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        if not safe:
            safe = cid
        path = out / f"{safe}.md"
        path.write_text(md, encoding="utf-8")
        written.append(str(path))

    return written
