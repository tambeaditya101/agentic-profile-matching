"""
Agentic Profile Matching — Streamlit Chat Interface (Primary).

Connects the compiled LangGraph agent to a Streamlit chat UI with:
  - Sidebar: JD upload, requirements panel, screening progress, quick actions
  - Main chat area: st.chat_message based conversation loop
  - State management: st.session_state for persistence across reruns
  - Markdown report rendering with expandable sections
  - Comparison tables via st.dataframe / st.markdown
  - Streaming-style responses (chunk-by-chunk display via graph.stream)

Usage:
    streamlit run ui/streamlit_app.py

Architecture Reference: architecture.md Section 14.1 (Streamlit Interface)
Phase: 7 — User Interface
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any

# Ensure the project root (parent of ui/) is on sys.path before any
# `from ui.*` or `from src.*` imports. Streamlit sets sys.path[0] to the
# script's own directory (ui/), which means `import ui.components` would
# fail when launched via `streamlit run ui/streamlit_app.py` or
# `python matching_agent.py streamlit`. Inserting the project root fixes
# this regardless of how Streamlit was launched.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Streamlit is imported lazily inside main() to keep this file importable
# for unit tests. The CLI and unit tests never call run().
import streamlit as st

from ui.components import (
    SUGGESTED_PROMPTS,
    build_agent_response,
    export_reports_to_dir,
    format_requirements_panel,
    format_screening_progress,
    format_shortlist_table,
    get_suggested_prompts,
    render_comparison_table,
    render_explanation,
    render_match_report,
    render_questions,
    render_ranking_delta,
)


# =====================================================================
# Session-state initialization
# =====================================================================

def init_session_state() -> None:
    """Initialize all session-state keys used by the UI."""
    if "agent_state" not in st.session_state:
        st.session_state.agent_state: dict[str, Any] = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: list[dict[str, str]] = []
    if "graph" not in st.session_state:
        # Lazy: create graph only once per session
        try:
            from src.agent.graph import create_graph
            st.session_state.graph = create_graph()
        except Exception as e:
            st.session_state.graph = None
            st.session_state.graph_error = str(e)


def reset_session() -> None:
    """Clear all conversation state."""
    st.session_state.agent_state = {}
    st.session_state.chat_history = []
    # Keep the graph object — re-creating it is expensive


# =====================================================================
# JD upload + linear pipeline
# =====================================================================

def run_linear_pipeline(raw_jd: str) -> None:
    """Run the linear pipeline and append the agent's response to chat history."""
    graph = st.session_state.graph
    if graph is None:
        st.session_state.chat_history.append({
            "role": "agent",
            "content": f"⚠️ Agent graph unavailable: {st.session_state.get('graph_error', 'unknown error')}",
        })
        return

    with st.spinner("Parsing JD, extracting requirements, searching resumes, ranking, generating reports…"):
        try:
            result = graph.invoke({"raw_jd": raw_jd, "messages": []})
            st.session_state.agent_state = result
            response = build_agent_response(result)
            st.session_state.chat_history.append({"role": "agent", "content": response})
        except Exception as e:
            tb = traceback.format_exc()
            st.session_state.chat_history.append({
                "role": "agent",
                "content": f"⚠️ Pipeline failed: `{e}`\n\n```\n{tb}\n```",
            })


def run_feedback_turn(human_feedback: str) -> None:
    """Send a feedback message to the interactive loop and append the response."""
    graph = st.session_state.graph
    state = st.session_state.agent_state
    if graph is None or not state:
        st.session_state.chat_history.append({
            "role": "agent",
            "content": "⚠️ No active session. Upload a JD in the sidebar first.",
        })
        return

    # Handle "show report for X" as a CLI-style command since the graph
    # doesn't have a dedicated node for it.
    fb_lower = human_feedback.strip().lower()
    if fb_lower.startswith("show report"):
        _handle_show_report(human_feedback)
        return

    payload: dict[str, Any] = {
        **{k: v for k, v in state.items() if k != "messages"},
        "human_feedback": human_feedback,
        "awaiting_human_feedback": True,
        "messages": state.get("messages", []),
    }
    with st.spinner("Thinking…"):
        try:
            new_state = graph.invoke(payload)
            st.session_state.agent_state = new_state
            response = build_agent_response(new_state)
            st.session_state.chat_history.append({"role": "agent", "content": response})
        except Exception as e:
            tb = traceback.format_exc()
            st.session_state.chat_history.append({
                "role": "agent",
                "content": f"⚠️ Agent error: `{e}`\n\n```\n{tb}\n```",
            })


def _handle_show_report(message: str) -> None:
    """Show the full match report for a named candidate."""
    state = st.session_state.agent_state
    reports = state.get("generated_reports") or {}
    shortlist = state.get("current_shortlist") or []
    if not reports:
        st.session_state.chat_history.append({
            "role": "agent",
            "content": "⚠️ No reports available. Run the pipeline first.",
        })
        return
    # Parse the candidate name from the message
    name_part = message.lower().replace("show report", "").replace("for", "").strip()
    target_cid = None
    for c in shortlist:
        cname = c.get("name", "")
        if name_part and name_part in cname.lower():
            target_cid = c.get("candidate_id")
            break
    if not target_cid and len(reports) == 1:
        target_cid = next(iter(reports))
    if not target_cid:
        avail = ", ".join(reports.keys())
        st.session_state.chat_history.append({
            "role": "agent",
            "content": f"Could not find candidate matching '{name_part}'. Available: {avail}",
        })
        return
    md = reports.get(target_cid, "")
    st.session_state.chat_history.append({"role": "agent", "content": render_match_report(md)})


# =====================================================================
# Sidebar
# =====================================================================

def render_sidebar() -> None:
    """Render the sidebar: JD upload, requirements, screening, quick actions."""
    with st.sidebar:
        st.title("🤖 Profile Matching")
        st.caption("LangGraph agent · Phase 7 UI")

        # --- LLM status badge (prominent, at the top) ---
        try:
            from src.llm.client import get_llm_status, get_llm_call_history, reset_llm_status
            status = get_llm_status()
            color = status.get("color", "gray")
            label = status.get("label", "Unknown")
            detail = status.get("detail", "")
            provider = status.get("provider", "unknown")

            # Map color to Streamlit's status method
            if color == "green":
                st.success(f"🟢 **{label}**")
            elif color == "yellow":
                st.warning(f"🟡 **{label}**")
            elif color == "red":
                st.error(f"🔴 **{label}**")
            else:
                st.info(f"⚪ **{label}**")

            # Show detail + runtime call stats in an expander
            with st.expander("LLM details & call history", expanded=False):
                st.caption(detail)
                st.caption(f"Provider: `{provider}`")
                if status.get("model"):
                    st.caption(f"Model: `{status['model']}`")

                # Runtime call stats
                runtime_calls = status.get("runtime_calls", 0)
                success_rate = status.get("runtime_success_rate")
                last_failure = status.get("last_failure")

                if runtime_calls > 0:
                    st.caption(f"Runtime calls: {runtime_calls}")
                    if success_rate is not None:
                        pct = int(success_rate * 100)
                        st.caption(f"Success rate: {pct}%")
                    if last_failure:
                        st.caption(f"Last failure: {last_failure.get('tool', '?')} — {last_failure.get('error', '?')[:100]}")
                    # Show recent calls
                    history = get_llm_call_history(5)
                    if history:
                        st.caption("**Recent calls (newest last):**")
                        for call in history:
                            icon = "✅" if call["success"] else "❌"
                            tool = call.get("tool", "?")
                            ms = call.get("duration_ms", 0)
                            st.caption(f"  {icon} {tool} ({ms:.0f}ms)")
                else:
                    st.caption("_No LLM calls recorded yet. Run the pipeline to see call history._")

                # Re-check button (forces a fresh ping)
                if st.button("🔄 Re-check LLM", key="recheck_llm", use_container_width=True):
                    reset_llm_status()
                    st.rerun()
        except Exception as e:
            st.error(f"🔴 **LLM status check failed**")
            st.caption(f"Error: {e}")

        st.divider()

        # --- JD upload ---
        st.subheader("1. Job Description")
        uploaded = st.file_uploader(
            "Upload JD (.txt or .pdf)",
            type=["txt", "pdf"],
            help="Upload a JD file or paste JD text below.",
        )
        jd_text = st.text_area(
            "Or paste JD text",
            height=120,
            placeholder="Paste your job description here…",
        )

        col1, col2 = st.columns(2)
        run_clicked = col1.button("▶ Run Pipeline", type="primary", use_container_width=True)
        reset_clicked = col2.button("Reset", use_container_width=True)

        if reset_clicked:
            reset_session()
            st.rerun()

        if run_clicked:
            raw_jd = ""
            if uploaded is not None:
                raw_jd = uploaded.read().decode("utf-8", errors="ignore")
            elif jd_text.strip():
                raw_jd = jd_text.strip()
            if len(raw_jd.strip()) < 20:
                st.error("Please provide a JD with at least 20 characters.")
            else:
                st.session_state.chat_history.append({"role": "user", "content": f"_(loaded JD: {len(raw_jd)} chars)_"})
                run_linear_pipeline(raw_jd)
                st.rerun()

        st.divider()

        # --- Requirements panel ---
        state = st.session_state.agent_state
        st.subheader("2. Requirements")
        st.markdown(format_requirements_panel(state.get("requirements")))

        st.divider()

        # --- Screening progress ---
        st.subheader("3. Screening Progress")
        st.markdown(format_screening_progress(state.get("screening_rounds")))

        st.divider()

        # --- Quick actions ---
        st.subheader("4. Quick Actions")
        col_a, col_b = st.columns(2)
        if col_a.button("Export Reports", use_container_width=True):
            written = export_reports_to_dir(state, "data/reports")
            if written:
                st.success(f"Wrote {len(written)} report(s) to data/reports/")
            else:
                st.warning("No reports to export yet.")
        if col_b.button("Show Suggestions", use_container_width=True):
            st.session_state.show_suggestions = not st.session_state.get("show_suggestions", False)
            st.rerun()

        # Optional suggestions expander
        if st.session_state.get("show_suggestions"):
            with st.expander("Suggested prompts", expanded=True):
                for s in get_suggested_prompts():
                    if st.button(s["label"], key=f"sugg_{s['label']}", use_container_width=True):
                        st.session_state.pending_input = s["message"]
                        st.rerun()

        st.divider()

        # --- System info ---
        st.subheader("5. System")
        try:
            from src.rag.store import get_vector_store
            from src.rag.retriever import ResumeRetriever
            store = get_vector_store(os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db"))
            retriever = ResumeRetriever(store)
            n_results = len(retriever.search("developer", top_k=1))
            st.write(f"**RAG:** {'✅ ok' if n_results > 0 else '⚠️ empty'}")
        except Exception as e:
            st.write(f"**RAG:** ❌ `{e}`")


# =====================================================================
# Main chat area
# =====================================================================

def render_chat_area() -> None:
    """Render the chat history + input box."""
    st.title("💬 Chat")
    st.caption("Ask the agent to compare candidates, refine requirements, explain rankings, and more.")

    # Chat history
    chat_history: list[dict[str, str]] = st.session_state.chat_history
    for msg in chat_history:
        role = msg.get("role", "agent")
        content = msg.get("content", "")
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(content)

    # Pending input (set by suggestion buttons)
    pending = st.session_state.pop("pending_input", "")
    if pending:
        st.session_state.chat_history.append({"role": "user", "content": pending})
        if pending.strip().lower() in ("done", "exit", "quit"):
            st.session_state.chat_history.append({
                "role": "agent",
                "content": "Session complete. Reports remain available via the sidebar Export button.",
            })
        else:
            run_feedback_turn(pending)
        st.rerun()

    # Input box
    user_input = st.chat_input("Type your message…")
    if user_input and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        if user_input.strip().lower() in ("done", "exit", "quit"):
            st.session_state.chat_history.append({
                "role": "agent",
                "content": "Session complete. Reports remain available via the sidebar Export button.",
            })
        else:
            run_feedback_turn(user_input)
        st.rerun()


# =====================================================================
# Reports tab (bonus — view full match reports)
# =====================================================================

def render_reports_tab() -> None:
    """Render a 'Reports' tab where the user can browse full match reports."""
    state = st.session_state.agent_state
    reports = state.get("generated_reports") or {}
    shortlist = state.get("current_shortlist") or []
    name_by_id = {c.get("candidate_id", ""): c.get("name", c.get("candidate_id", "?")) for c in shortlist}

    st.subheader("📄 Match Reports")
    if not reports:
        st.info("No reports yet. Run the pipeline to generate match reports.")
        return

    options = [(cid, name_by_id.get(cid, cid)) for cid in reports.keys()]
    selected = st.selectbox(
        "Choose a candidate",
        options=options,
        format_func=lambda x: x[1],
    )
    if selected:
        cid, _name = selected
        md = reports.get(cid, "")
        st.markdown(md)


# =====================================================================
# Main entrypoint
# =====================================================================

def main() -> None:
    """Streamlit entrypoint — configures page, renders sidebar + tabs."""
    st.set_page_config(
        page_title="Agentic Profile Matching",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()

    render_sidebar()

    # Two tabs: Chat and Reports
    tab_chat, tab_reports = st.tabs(["💬 Chat", "📄 Reports"])
    with tab_chat:
        render_chat_area()
    with tab_reports:
        render_reports_tab()


if __name__ == "__main__":
    main()
