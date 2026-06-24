"""
Agentic Profile Matching — Shared LLM Client.

Provides a single ``get_llm()`` factory that returns a LangChain
``BaseChatModel`` with a 4-tier fallback chain:

  1. **Gemini 2.0 Flash** (free tier) — primary, high quality
  2. **Groq Llama 3.3 70B** (free tier) — secondary, very fast, high quota
  3. **Ollama gemma2:9b** (local) — offline fallback, no API key
  4. **Keyword fallback** — always available, lower quality

Also provides a runtime status tracker that records whether actual LLM
calls succeed or fail, so the UI can show a truly dynamic status badge.

Architecture Reference: architecture.md Section 12 (Technology Stack)
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

# Load .env once at import time
load_dotenv()

_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:9b").strip()


# =====================================================================
# Runtime LLM call tracker (thread-safe)
# =====================================================================

class _LLMCallTracker:
    """Thread-safe tracker for LLM call outcomes.

    Records the last N call outcomes so the UI can show whether the LLM
    is actually working (not just configured).
    """

    def __init__(self, max_history: int = 20) -> None:
        self._lock = threading.Lock()
        self._history: list[dict] = []
        self._max_history = max_history
        self._startup_ping_done = False
        self._startup_ping_result: dict | None = None

    def record_call(
        self,
        provider: str,
        success: bool,
        error: str | None = None,
        tool: str = "",
        duration_ms: float = 0.0,
    ) -> None:
        """Record a single LLM call outcome."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "success": success,
            "error": error,
            "tool": tool,
            "duration_ms": duration_ms,
        }
        with self._lock:
            self._history.append(entry)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

    def set_startup_ping(self, result: dict) -> None:
        """Record the result of the startup ping check."""
        with self._lock:
            self._startup_ping_done = True
            self._startup_ping_result = result

    def get_startup_ping(self) -> dict | None:
        with self._lock:
            return self._startup_ping_result.copy() if self._startup_ping_result else None

    def get_recent_calls(self, n: int = 5) -> list[dict]:
        with self._lock:
            return [c.copy() for c in self._history[-n:]]

    def get_success_rate(self) -> float | None:
        """Return the success rate of recent calls (0.0–1.0), or None if no calls."""
        with self._lock:
            if not self._history:
                return None
            successes = sum(1 for c in self._history if c["success"])
            return successes / len(self._history)

    def get_last_failure(self) -> dict | None:
        """Return the most recent failed call, or None if all succeeded."""
        with self._lock:
            for call in reversed(self._history):
                if not call["success"]:
                    return call.copy()
            return None


_tracker = _LLMCallTracker()


def record_llm_call(
    provider: str,
    success: bool,
    error: str | None = None,
    tool: str = "",
    duration_ms: float = 0.0,
) -> None:
    """Public API: record an LLM call outcome. Called by tools after each LLM invoke."""
    _tracker.record_call(provider, success, error, tool, duration_ms)


def get_llm_call_history(n: int = 5) -> list[dict]:
    """Public API: return the last N LLM call outcomes."""
    return _tracker.get_recent_calls(n)


def get_llm_success_rate() -> float | None:
    """Public API: return the success rate of recent LLM calls (0.0–1.0)."""
    return _tracker.get_success_rate()


def get_llm_last_failure() -> dict | None:
    """Public API: return the most recent failed LLM call, or None."""
    return _tracker.get_last_failure()


# =====================================================================
# LLM client factories (Gemini → Groq → Ollama)
# =====================================================================

def _create_gemini_llm() -> BaseChatModel | None:
    """Try to create a Gemini LLM client. Returns None if no API key."""
    if not _GEMINI_API_KEY:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=_GEMINI_MODEL,
            temperature=0,
            max_retries=3,
            timeout=60,
        )
    except Exception:
        return None


def _create_groq_llm() -> BaseChatModel | None:
    """Try to create a Groq LLM client. Returns None if no API key."""
    if not _GROQ_API_KEY:
        return None
    try:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=_GROQ_MODEL,
            temperature=0,
            max_retries=3,
            timeout=60,
        )
    except Exception:
        return None


def _create_ollama_llm() -> BaseChatModel | None:
    """Try to create an Ollama LLM client. Returns None if Ollama unavailable."""
    try:
        from langchain_community.chat_models import ChatOllama

        llm = ChatOllama(model=_OLLAMA_MODEL, temperature=0)
        # Quick connectivity check
        llm.invoke("ping")
        return llm
    except Exception:
        return None


# =====================================================================
# Live ping functions (actually invoke the LLM to verify it works)
# =====================================================================

def _ping_gemini() -> tuple[bool, str | None]:
    """Actually invoke Gemini with a tiny prompt to verify the key works.

    Returns (success, error_message).
    """
    if not _GEMINI_API_KEY:
        return False, "No GEMINI_API_KEY set"
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage

        llm = ChatGoogleGenerativeAI(
            model=_GEMINI_MODEL,
            temperature=0,
            max_retries=1,
            timeout=15,
        )
        # Tiny prompt — minimal token cost
        response = llm.invoke([HumanMessage(content="Reply with exactly: OK")])
        if response and hasattr(response, "content") and response.content:
            return True, None
        return False, "Empty response from Gemini"
    except Exception as e:
        return False, str(e)


def _ping_groq() -> tuple[bool, str | None]:
    """Actually invoke Groq with a tiny prompt to verify the key works."""
    if not _GROQ_API_KEY:
        return False, "No GROQ_API_KEY set"
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage

        llm = ChatGroq(
            model=_GROQ_MODEL,
            temperature=0,
            max_retries=1,
            timeout=15,
        )
        response = llm.invoke([HumanMessage(content="Reply with exactly: OK")])
        if response and hasattr(response, "content") and response.content:
            return True, None
        return False, "Empty response from Groq"
    except Exception as e:
        return False, str(e)


def _ping_ollama() -> tuple[bool, str | None]:
    """Actually invoke Ollama to verify it's running."""
    try:
        from langchain_community.chat_models import ChatOllama

        llm = ChatOllama(model=_OLLAMA_MODEL, temperature=0)
        llm.invoke("ping")
        return True, None
    except Exception as e:
        return False, str(e)


# =====================================================================
# get_llm() — 4-tier fallback chain (Groq → Gemini → Ollama)
# =====================================================================

@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Return a cached LLM instance. Tries Groq → Gemini → Ollama.

    IMPORTANT: This function does NOT just check if the API key is set —
    it actually pings each provider to verify it works. So if Groq's key
    is set and working, it returns Groq. If Groq fails (or has no key),
    it tries Gemini. If Gemini's quota is exhausted (429), it tries
    Ollama. This ensures the tools always get a *working* LLM, not a
    dead one.

    The LLM is cached via @lru_cache so the same instance is reused
    across all tools in a single process. If you need to force a
    re-check (e.g., after fixing your .env), call reset_llm_status()
    and get_llm.cache_clear().

    Raises:
        RuntimeError: If no provider works (all keys invalid/missing,
                      Ollama not running).
    """
    # Use the startup ping to find a working provider.
    # _do_startup_ping() actually invokes each LLM with a tiny prompt
    # to verify the key works and the API is reachable.
    status = _do_startup_ping()
    provider = status.get("provider")

    if provider == "groq":
        llm = _create_groq_llm()
        if llm is not None:
            return llm
    if provider == "gemini":
        llm = _create_gemini_llm()
        if llm is not None:
            return llm
    if provider == "ollama":
        llm = _create_ollama_llm()
        if llm is not None:
            return llm

    # If we get here, no provider works. Raise with a helpful message.
    raise RuntimeError(
        f"No LLM available (startup ping result: {provider}). "
        f"Details: {status.get('detail', 'unknown')}. "
        f"Fix: set GROQ_API_KEY (https://console.groq.com/keys) or "
        f"GEMINI_API_KEY (https://aistudio.google.com/apikey) in .env, "
        f"or install Ollama (https://ollama.com)."
    )


# =====================================================================
# Status reporting (live + runtime)
# =====================================================================

def get_llm_provider_name() -> str:
    """Return which provider will be used: 'groq', 'gemini', or 'ollama'.

    Priority order: Groq → Gemini → Ollama.
    Note: this returns the *configured* provider, not necessarily the one
    that's actually working. Use ``get_llm_status()`` for a live check.
    """
    if _GROQ_API_KEY:
        return "groq"
    if _GEMINI_API_KEY:
        return "gemini"
    return "ollama"


def _do_startup_ping() -> dict:
    """Perform a real LLM ping and cache the result. Called once at startup.

    This actually invokes the LLM with a tiny prompt to verify the key
    works and the API is reachable. The result is cached in the tracker.

    Fallback chain: Groq → Gemini → Ollama → keyword_fallback

    Groq is tried first because it has the highest free tier quota
    (14,400 req/day vs Gemini's 1,500) and is the fastest.
    """
    if _tracker.get_startup_ping() is not None:
        return _tracker.get_startup_ping()

    errors: list[str] = []  # collect errors for the final message if all fail

    # --- 1. Try Groq first (highest free quota, fastest) ---
    if _GROQ_API_KEY:
        success, error = _ping_groq()
        if success:
            result = {
                "provider": "groq",
                "label": f"Groq ({_GROQ_MODEL})",
                "status": "ok",
                "color": "green",
                "detail": (
                    f"Connected to Groq ({_GROQ_MODEL}). "
                    f"Verified with a test ping. Groq is the fastest provider "
                    f"with the highest free tier quota (14,400 req/day)."
                ),
                "model": _GROQ_MODEL,
                "ping_error": None,
            }
            _tracker.set_startup_ping(result)
            return result
        errors.append(f"Groq: {error}")

    # --- 2. Try Gemini (if Groq failed or no Groq key) ---
    if _GEMINI_API_KEY:
        success, error = _ping_gemini()
        if success:
            label = f"Gemini 2.0 Flash"
            if _GROQ_API_KEY:
                # Groq was configured but failed — note it
                label = f"Gemini (Groq failed, using Gemini)"
            result = {
                "provider": "gemini",
                "label": label,
                "status": "ok",
                "color": "green",
                "detail": (
                    f"Connected to Gemini ({_GEMINI_MODEL}). "
                    + ("Groq failed, fell back to Gemini. " if errors else "")
                    + "Verified with a test ping."
                ),
                "model": _GEMINI_MODEL,
                "ping_error": "; ".join(errors) if errors else None,
            }
            _tracker.set_startup_ping(result)
            return result
        errors.append(f"Gemini: {error}")

    # --- 3. Try Ollama (offline fallback) ---
    ollama_success, ollama_error = _ping_ollama()
    if ollama_success:
        result = {
            "provider": "ollama",
            "label": f"Ollama ({_OLLAMA_MODEL})",
            "status": "ok" if not errors else "degraded",
            "color": "yellow",
            "detail": (
                f"Connected to local Ollama ({_OLLAMA_MODEL}). "
                + (f"All cloud providers failed ({'; '.join(errors[:200])}). " if errors else "")
                + "LLM-powered analysis active (runs locally, no API key). "
                + "Tip: add a free Groq key for faster analysis."
            ),
            "model": _OLLAMA_MODEL,
            "ping_error": "; ".join(errors) if errors else None,
        }
        _tracker.set_startup_ping(result)
        return result
    errors.append(f"Ollama: {ollama_error}")

    # --- 4. All failed — keyword fallback ---
    all_errors = "; ".join(errors)
    result = {
        "provider": "keyword_fallback",
        "label": "Keyword Fallback (no working LLM)",
        "status": "degraded",
        "color": "red",
        "detail": (
            f"All LLM providers failed: {all_errors[:300]}. "
            f"Using keyword-based extraction (lower quality). "
            f"Fix: get a free Groq key at https://console.groq.com/keys (recommended), "
            f"or a free Gemini key at https://aistudio.google.com/apikey, "
            f"or install Ollama and run: ollama pull gemma2:9b"
        ),
        "model": None,
        "ping_error": all_errors,
    }
    _tracker.set_startup_ping(result)
    return result


def get_llm_status() -> dict:
    """Return a detailed LLM status dict for UI display.

    Combines the startup ping result with runtime call history to give
    a truly dynamic status. If the startup ping succeeded but runtime
    calls are failing (e.g., quota exhausted mid-session), the status
    will reflect that.

    Returns:
        {
            "provider": "gemini" | "groq" | "ollama" | "keyword_fallback",
            "label": str,
            "status": "ok" | "degraded" | "unavailable",
            "color": "green" | "yellow" | "red",
            "detail": str,
            "model": str | None,
            "ping_error": str | None,
            "runtime_calls": int,
            "runtime_success_rate": float | None,
            "last_failure": dict | None,
        }
    """
    base = _do_startup_ping()

    # Enrich with runtime call data
    recent = _tracker.get_recent_calls(20)
    success_rate = _tracker.get_success_rate()
    last_failure = _tracker.get_last_failure()

    result = {**base}
    result["runtime_calls"] = len(recent)
    result["runtime_success_rate"] = success_rate
    result["last_failure"] = last_failure

    # Dynamic status adjustment: if startup was OK but runtime calls
    # are failing, downgrade the status to warn the user.
    if base["status"] == "ok" and recent and success_rate is not None and success_rate < 0.5:
        # More than half of recent calls failed — likely quota or network issue
        result["status"] = "degraded"
        result["color"] = "yellow"
        result["label"] = f"{base['label']} (degraded)"
        result["detail"] = (
            f"{base['detail']} "
            f"⚠️ {int((1 - success_rate) * 100)}% of recent LLM calls failed "
            f"({len(recent)} calls recorded). "
            f"Last error: {last_failure['error'] if last_failure else 'unknown'}. "
            f"The agent is falling back to keyword-based analysis for failed calls."
        )
    elif base["status"] == "ok" and recent and success_rate is not None and success_rate < 1.0:
        # Some calls failing — show a warning but keep color
        result["detail"] = (
            f"{base['detail']} "
            f"Note: {int((1 - success_rate) * 100)}% of recent calls fell back to keywords "
            f"({len(recent)} calls, {sum(1 for c in recent if not c['success'])} failed)."
        )

    return result


def reset_llm_status() -> None:
    """Reset the cached startup ping so it re-checks on next status request.

    Useful after the user updates their .env file — call this to force
    a fresh ping. Also clears the get_llm() cache so the next call
    returns a freshly-evaluated LLM. Re-reads environment variables so
    changes to .env are picked up without restarting the process.
    """
    global _GEMINI_API_KEY, _GEMINI_MODEL, _GROQ_API_KEY, _GROQ_MODEL, _OLLAMA_MODEL
    # Re-read env vars (in case the user updated .env)
    load_dotenv(override=True)
    _GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
    _GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
    _GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
    _GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
    _OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:9b").strip()

    _tracker._startup_ping_done = False
    _tracker._startup_ping_result = None
    _tracker._history.clear()
    # Clear the get_llm cache so the next call re-evaluates providers
    get_llm.cache_clear()
