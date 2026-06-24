"""
Agentic Profile Matching — CLI Interface (Fallback).

A Click + Typer-free interactive REPL that connects the compiled LangGraph
agent to a terminal chat loop. Designed to run anywhere Python runs —
no browser, no Streamlit runtime required.

Usage:
    python -m ui.cli_app
    python -m ui.cli_app --jd tests/fixtures/sample_jd.txt
    python -m ui.cli_app start --jd tests/fixtures/sample_jd.txt

Architecture Reference: architecture.md Section 14.2 (CLI Interface)
Phase: 7 — User Interface
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any

# Ensure the project root (parent of ui/) is on sys.path before any
# `from ui.*` or `from src.*` imports. This makes the CLI work whether
# it's invoked via `python -m ui.cli_app`, `python ui/cli_app.py`, or
# `python matching_agent.py cli`.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import click

# Import lazily to keep `--help` fast when deps are partially installed
from ui.components import (
    build_agent_response,
    colorize,
    export_reports_to_dir,
    format_requirements_panel,
    format_screening_progress,
    format_shortlist_table,
    get_suggested_prompts,
)


# =====================================================================
# Banner / formatting
# =====================================================================

BANNER = r"""
   ___                  _____           _      __  __  ___           _
  / _ \ _ __   ___ _ __| ____|_  ____ _| |_   |  \/  |/ _ \ ___ _ __ | |_
 / /_)/ '_ \ / _ \ '_ \  _| \ \/ / _` | __|  | |\/| | | | / __| '_ \| __|
/ ___/| |_) |  __/ | | | |___ >  < (_| | |_   | |  | | |_| \__ \ |_) | |_
\/    | .__/ \___|_| |_|_____/_/\_\__,_|\__|  |_|  |_|\___/|___/ .__/ \__|
      |_|
"""


def print_banner() -> None:
    """Print the agent banner + LLM status + quick-start hint."""
    click.echo(colorize(BANNER, "cyan"))

    # Show LLM status prominently
    try:
        from src.llm.client import get_llm_status
        status = get_llm_status()
        color = status.get("color", "gray")
        label = status.get("label", "Unknown")
        detail = status.get("detail", "")

        # Map status color to ANSI color
        ansi_color = {"green": "green", "yellow": "yellow", "red": "red"}.get(color, "cyan")
        symbol = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(color, "⚪")
        click.echo(colorize(f"  {symbol} LLM: {label}", ansi_color))
        click.echo(colorize(f"     {detail}", "dim"))
    except Exception as e:
        click.echo(colorize(f"  ⚪ LLM status check failed: {e}", "red"))
    click.echo("")

    click.echo(colorize("Type a message and press Enter. Type 'done' or press Ctrl+C to exit.", "dim"))
    click.echo(colorize("Quick commands: 'compare top 3', 'why top 2?', 'questions for <name>', 'reset', 'help'", "dim"))
    click.echo("")


def print_divider() -> None:
    """Print a thin horizontal divider."""
    click.echo(colorize("-" * 60, "dim"))


def prompt_user() -> str:
    """Render the You: prompt and read a line of input."""
    return click.prompt(colorize("You", "green"), prompt_suffix="> ", default="", show_default=False)


def echo_agent(text: str) -> None:
    """Print an agent response with a small prefix and optional ANSI color."""
    click.echo("")
    click.echo(colorize("Agent", "magenta") + ">")
    # Indent multi-line responses for readability
    for line in text.splitlines():
        click.echo(f"  {line}")
    click.echo("")


# =====================================================================
# Agent invocation helpers
# =====================================================================

def _load_jd(jd_path: str | None, jd_text: str | None) -> str:
    """Resolve the JD from --jd path, --jd-text, or interactive prompt."""
    if jd_text:
        return jd_text
    if jd_path:
        p = Path(jd_path)
        if not p.exists():
            raise click.ClickException(f"JD file not found: {jd_path}")
        return p.read_text(encoding="utf-8")
    # Interactive: ask user to paste
    click.echo(colorize("Paste your job description. End with a line containing only 'EOF':", "dim"))
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == "EOF":
            break
        lines.append(line)
    return "\n".join(lines)


def _invoke_linear(graph: Any, raw_jd: str) -> dict:
    """Invoke the full linear pipeline (parse → ... → report)."""
    return graph.invoke({"raw_jd": raw_jd, "messages": []})


def _invoke_feedback(graph: Any, state: dict, human_feedback: str) -> dict:
    """Invoke the graph with a feedback message for the interactive loop.

    The previous state is carried forward so the human_feedback_loop node
    can classify intent and route to refine / compare / explain / questions.
    """
    payload: dict[str, Any] = {
        **{k: v for k, v in state.items() if k != "messages"},
        "human_feedback": human_feedback,
        "awaiting_human_feedback": True,
        "messages": state.get("messages", []),
    }
    return graph.invoke(payload)


# =====================================================================
# Sidebar-equivalent status print
# =====================================================================

def print_status(state: dict) -> None:
    """Print the current requirements + screening progress + shortlist."""
    print_divider()
    click.echo(colorize("=== Status ===", "bold"))

    # Requirements
    req = state.get("requirements") or {}
    if req:
        click.echo(colorize("Requirements:", "yellow"))
        for line in format_requirements_panel(req).splitlines():
            click.echo(f"  {line}")

    # Screening rounds
    rounds = state.get("screening_rounds") or []
    if rounds:
        click.echo("")
        click.echo(colorize("Screening:", "yellow"))
        for line in format_screening_progress(rounds).splitlines():
            click.echo(f"  {line}")

    # Shortlist
    shortlist = state.get("current_shortlist") or []
    if shortlist:
        click.echo("")
        click.echo(colorize(f"Shortlist ({len(shortlist)} candidates):", "yellow"))
        for line in format_shortlist_table(shortlist).splitlines():
            click.echo(f"  {line}")

    print_divider()


# =====================================================================
# Special commands handled in the CLI (not by the graph)
# =====================================================================

def _handle_special_command(
    cmd: str,
    state: dict,
    graph: Any,
) -> tuple[bool, dict | None]:
    """Handle CLI-only commands: status, help, reset, suggestions, show report, export.

    Returns:
        (handled, new_state_or_None)
        - If handled is True, the REPL should skip the graph invocation.
        - If new_state is not None, the REPL should update its state.
    """
    cmd_clean = cmd.strip().lower()
    tokens = cmd_clean.split()

    if cmd_clean in ("help", "?"):
        click.echo("")
        click.echo(colorize("Available commands:", "bold"))
        click.echo("  help / ?              Show this help")
        click.echo("  status                Re-print current state (reqs, rounds, shortlist)")
        click.echo("  suggestions           Show suggested prompts")
        click.echo("  show report for NAME  Print the full match report for a candidate")
        click.echo("  export [DIR]          Save all reports as Markdown to DIR (default: data/reports)")
        click.echo("  reset                 Clear state and return to JD-input mode")
        click.echo("  done / exit / quit    End the session")
        click.echo("")
        click.echo(colorize("Anything else is sent to the agent as natural language.", "dim"))
        click.echo("")
        return True, None

    if cmd_clean == "status":
        print_status(state)
        return True, None

    if cmd_clean in ("suggestions", "suggest"):
        click.echo(colorize("Suggested prompts:", "bold"))
        for s in get_suggested_prompts():
            click.echo(f"  - {colorize(s['label'], 'cyan')}: {s['message']}")
        click.echo("")
        return True, None

    if cmd_clean.startswith("show report"):
        # try to find the candidate name in the command
        name_part = cmd[len("show report"):].strip()
        name_part = name_part.replace("for", "").strip()
        reports = state.get("generated_reports") or {}
        shortlist = state.get("current_shortlist") or []
        if not reports:
            click.echo(colorize("No reports available yet. Run the pipeline first.", "red"))
            return True, None
        # Match by name (case-insensitive substring)
        target_cid = None
        for c in shortlist:
            cname = c.get("name", "")
            if name_part and name_part.lower() in cname.lower():
                target_cid = c.get("candidate_id")
                break
        # If only one candidate, just show that one
        if not target_cid and len(reports) == 1:
            target_cid = next(iter(reports))
        if not target_cid:
            click.echo(colorize(f"Could not find candidate matching '{name_part}'.", "red"))
            click.echo(colorize("Available reports: " + ", ".join(reports.keys()), "dim"))
            return True, None
        md = reports.get(target_cid, "")
        click.echo("")
        click.echo(md)
        click.echo("")
        return True, None

    if cmd_clean.startswith("export"):
        parts = cmd_clean.split(maxsplit=1)
        out_dir = parts[1] if len(parts) > 1 else "data/reports"
        written = export_reports_to_dir(state, out_dir)
        if written:
            click.echo(colorize(f"Wrote {len(written)} report(s):", "green"))
            for p in written:
                click.echo(f"  - {p}")
        else:
            click.echo(colorize("No reports to export. Run the pipeline first.", "yellow"))
        return True, None

    if cmd_clean in ("reset", "restart", "new"):
        click.echo(colorize("Session reset. Please provide a new JD.", "yellow"))
        return True, {"_reset": True}

    return False, None


# =====================================================================
# REPL
# =====================================================================

def run_repl(jd_path: str | None = None, jd_text: str | None = None) -> None:
    """Run the interactive agent REPL.

    Args:
        jd_path: Optional path to a JD file. If None, the user will be
            prompted to paste a JD interactively.
        jd_text: Optional JD text (overrides jd_path).
    """
    # Lazy import so --help is fast and any import error is reported cleanly
    try:
        from src.agent.graph import create_graph
    except Exception as e:
        click.echo(colorize(f"Failed to import the agent graph: {e}", "red"))
        click.echo(traceback.format_exc())
        sys.exit(1)

    print_banner()

    graph = create_graph()
    state: dict[str, Any] = {}

    # --- Initial pipeline run (if JD provided via flags) ---
    if jd_path or jd_text:
        try:
            raw_jd = _load_jd(jd_path, jd_text)
            click.echo(colorize(f"Loaded JD ({len(raw_jd)} chars). Running pipeline...", "dim"))
            state = _invoke_linear(graph, raw_jd)
            echo_agent(build_agent_response(state))
            print_status(state)
        except Exception as e:
            click.echo(colorize(f"Pipeline failed: {e}", "red"))
            click.echo(traceback.format_exc())

    # --- Main REPL loop ---
    max_turns = 20  # guard rail from architecture Section 15.2
    turn = 0
    while turn < max_turns:
        turn += 1
        try:
            user_input = prompt_user()
        except (KeyboardInterrupt, EOFError):
            click.echo("")
            click.echo(colorize("Exiting.", "yellow"))
            break

        if not user_input.strip():
            continue

        # Check for done/exit
        if user_input.strip().lower() in ("done", "exit", "quit"):
            click.echo(colorize("Session complete.", "green"))
            # Auto-export if any reports exist
            reports = state.get("generated_reports") or {}
            if reports:
                written = export_reports_to_dir(state, "data/reports")
                if written:
                    click.echo(colorize(f"Reports saved to data/reports/ ({len(written)} files).", "dim"))
            break

        # Special CLI commands
        handled, new_state = _handle_special_command(user_input, state, graph)
        if handled:
            if new_state and new_state.get("_reset"):
                state = {}
                # Prompt for new JD
                try:
                    raw_jd = _load_jd(None, None)
                    if raw_jd.strip():
                        state = _invoke_linear(graph, raw_jd)
                        echo_agent(build_agent_response(state))
                        print_status(state)
                except Exception as e:
                    click.echo(colorize(f"Pipeline failed: {e}", "red"))
            continue

        # Send to graph as feedback
        if not state:
            # No prior state — treat input as a JD
            click.echo(colorize("Treating input as a JD. Running pipeline...", "dim"))
            try:
                state = _invoke_linear(graph, user_input)
                echo_agent(build_agent_response(state))
                print_status(state)
            except Exception as e:
                click.echo(colorize(f"Pipeline failed: {e}", "red"))
                click.echo(traceback.format_exc())
            continue

        try:
            new_state = _invoke_feedback(graph, state, user_input)
            state = new_state
            echo_agent(build_agent_response(state))
        except Exception as e:
            click.echo(colorize(f"Agent error: {e}", "red"))
            click.echo(traceback.format_exc())

    else:
        click.echo(colorize(f"Reached max {max_turns} turns. Exiting to prevent infinite loop.", "yellow"))


# =====================================================================
# Click entrypoint
# =====================================================================

@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0", prog_name="agentic-profile-matching")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Agentic Profile Matching — CLI interface.

    Run without arguments to enter the interactive REPL. You will be
    prompted to paste a JD. Use 'python -m ui.cli_app start --jd <path>'
    to load a JD from a file at startup.
    """
    if ctx.invoked_subcommand is None:
        run_repl(jd_path=None, jd_text=None)


@cli.command("start")
@click.option("--jd", "jd_path", type=click.Path(exists=True), default=None,
              help="Path to a job description file (.txt or .pdf).")
@click.option("--jd-text", "jd_text", default=None,
              help="Inline JD text. Overrides --jd.")
def start(jd_path: str | None, jd_text: str | None) -> None:
    """Start the agent with a JD loaded from --jd or --jd-text."""
    run_repl(jd_path=jd_path, jd_text=jd_text)


@cli.command("info")
def info() -> None:
    """Print environment info (LLM status, RAG status)."""
    from src.llm.client import get_llm_status
    from src.rag.store import get_vector_store
    from src.rag.retriever import ResumeRetriever

    click.echo(colorize("Environment Info", "bold"))

    # LLM status (live check)
    try:
        status = get_llm_status()
        color = status.get("color", "gray")
        label = status.get("label", "Unknown")
        detail = status.get("detail", "")
        ansi_color = {"green": "green", "yellow": "yellow", "red": "red"}.get(color, "cyan")
        symbol = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(color, "⚪")
        click.echo(f"  {symbol} LLM              : {colorize(label, ansi_color)}")
        click.echo(colorize(f"     {detail}", "dim"))
        if status.get("model"):
            click.echo(f"     Model: {status['model']}")
    except Exception as e:
        click.echo(colorize(f"  ⚪ LLM              : check failed ({e})", "red"))

    # RAG status
    try:
        store = get_vector_store(os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db"))
        retriever = ResumeRetriever(store)
        results = retriever.search("developer", top_k=1)
        if results:
            click.echo(f"  ✅ RAG              : ok ({len(results)} result for 'developer')")
        else:
            click.echo(colorize("  ⚠️  RAG              : empty (run: python scripts/ingest_resumes.py)", "yellow"))
    except Exception as e:
        click.echo(colorize(f"  ❌ RAG              : error ({e})", "red"))


if __name__ == "__main__":
    cli()
