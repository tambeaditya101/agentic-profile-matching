"""
Agentic Profile Matching — Submission Entry Point.

This module is the canonical entry point required by the assignment:
"LangGraph-based agent implementation in ``matching_agent.py``".

It re-exports the compiled graph from ``src.agent.graph`` and provides
three convenience entry points:

  1. ``create_agent()``  — returns the compiled LangGraph StateGraph
  2. ``run()``           — launches the Streamlit chat UI
  3. ``run_cli(jd_path)``— launches the CLI REPL with an optional JD file

Usage:
    # As a library
    from matching_agent import create_agent
    graph = create_agent()
    result = graph.invoke({"raw_jd": "...", "messages": []})

    # As a Streamlit app
    streamlit run matching_agent.py

    # As a CLI
    python matching_agent.py cli --jd tests/fixtures/sample_jd.txt

Architecture Reference: architecture.md Section 4 (Graph Workflow)
Phase: 8 — Testing, Polish & Demo
"""

from __future__ import annotations

import sys
from typing import Any

# Re-export the graph factory so callers can do:
#   from matching_agent import create_agent
# instead of digging into src.agent.graph
from src.agent.graph import create_graph
from src.agent.graph import invoke_linear_pipeline

# Alias matching the assignment's expected name
create_agent = create_graph

__all__ = [
    "create_agent",
    "create_graph",
    "invoke_linear_pipeline",
    "run",
    "run_cli",
    "__version__",
]

__version__ = "0.1.0"


def run() -> None:
    """Launch the Streamlit chat UI.

    Equivalent to: ``streamlit run ui/streamlit_app.py``
    """
    import os
    import subprocess

    # Resolve ui/streamlit_app.py relative to this file so the entry
    # point works regardless of the caller's CWD.
    here = os.path.dirname(os.path.abspath(__file__))
    streamlit_app = os.path.join(here, "ui", "streamlit_app.py")

    if not os.path.exists(streamlit_app):
        print(f"ERROR: Streamlit app not found at {streamlit_app}", file=sys.stderr)
        sys.exit(1)

    # Set PYTHONPATH to include the project root so that `from ui.*` and
    # `from src.*` imports inside streamlit_app.py resolve correctly.
    # (streamlit_app.py also has a sys.path fix at the top, but setting
    # PYTHONPATH here is belt-and-suspenders in case the script's
    # bootstrap is bypassed.)
    env = os.environ.copy()
    if here not in env.get("PYTHONPATH", ""):
        env["PYTHONPATH"] = here + os.pathsep + env.get("PYTHONPATH", "")

    # Collect Streamlit flags from argv. Skip the first arg if it is the
    # literal "streamlit" / "ui" / "web" dispatch token (from _main()).
    extra_args = list(sys.argv[1:])
    if extra_args and extra_args[0] in ("streamlit", "ui", "web"):
        extra_args = extra_args[1:]

    # Prefer the `streamlit` executable on PATH; fall back to
    # `python -m streamlit` which works even when the script wrapper
    # isn't on PATH (e.g., when only pip-installed without --user).
    cmd = ["streamlit", "run", streamlit_app, *extra_args]
    print(f"Launching Streamlit: {' '.join(cmd)}")
    print(f"PYTHONPATH starts with: {here}")
    try:
        return_code = subprocess.call(cmd, env=env)
    except FileNotFoundError:
        # Fall back to `python -m streamlit`
        cmd = [sys.executable, "-m", "streamlit", "run", streamlit_app, *extra_args]
        print(f"streamlit not on PATH, falling back to: {' '.join(cmd)}")
        try:
            return_code = subprocess.call(cmd, env=env)
        except FileNotFoundError:
            print(
                "ERROR: streamlit is not installed.\n"
                "Install with: pip install streamlit",
                file=sys.stderr,
            )
            sys.exit(1)
    if return_code != 0:
        sys.exit(return_code)


def run_cli(jd_path: str | None = None, jd_text: str | None = None) -> None:
    """Launch the CLI REPL with an optional JD file or text.

    Args:
        jd_path: Optional path to a JD file (.txt or .pdf).
        jd_text: Optional inline JD text (overrides jd_path).
    """
    # Import here so the module-level import doesn't slow down library users
    from ui.cli_app import run_repl

    run_repl(jd_path=jd_path, jd_text=jd_text)


def _main() -> None:
    """CLI dispatch: choose UI based on argv[1].

    - ``streamlit`` (default) → launch Streamlit app
    - ``cli``                   → launch CLI REPL
    - ``--help`` / ``-h``       → print usage
    """
    args = sys.argv[1:]
    if not args or args[0] in ("streamlit", "ui", "web"):
        run()
        return

    if args[0] in ("cli", "repl"):
        rest = args[1:]
        jd_path = None
        jd_text = None
        i = 0
        while i < len(rest):
            if rest[i] == "--jd" and i + 1 < len(rest):
                jd_path = rest[i + 1]
                i += 2
            elif rest[i] == "--jd-text" and i + 1 < len(rest):
                jd_text = rest[i + 1]
                i += 2
            else:
                i += 1
        run_cli(jd_path=jd_path, jd_text=jd_text)
        return

    if args[0] in ("--help", "-h", "help"):
        print(__doc__)
        print("\nUsage:")
        print("  python matching_agent.py                  # launch Streamlit UI")
        print("  python matching_agent.py streamlit        # launch Streamlit UI")
        print("  python matching_agent.py cli              # launch CLI REPL")
        print("  python matching_agent.py cli --jd <path>  # launch CLI with JD file")
        print("  python matching_agent.py cli --jd-text <text>  # launch CLI with inline JD")
        print("  python matching_agent.py --help           # show this help")
        return

    print(f"Unknown command: {args[0]}", file=sys.stderr)
    print("Run 'python matching_agent.py --help' for usage.", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    _main()
