"""
Tool: rag_search — Semantic search over the resume vector store.

Wraps the ResumeRetriever from Phase 1 as a LangChain @tool.

Architecture Reference: architecture.md Section 6.2
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig

from src.rag.retriever import ResumeRetriever
from src.rag.store import get_vector_store

load_dotenv()

_DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")


# Module-level cache for the retriever to avoid re-initializing ChromaDB on every call
_retriever_cache: dict[str, ResumeRetriever] = {}


def _get_retriever(persist_dir: str = _DEFAULT_PERSIST_DIR) -> ResumeRetriever:
    """Return a cached ResumeRetriever instance."""
    if persist_dir not in _retriever_cache:
        collection = get_vector_store(persist_dir)
        _retriever_cache[persist_dir] = ResumeRetriever(collection)
    return _retriever_cache[persist_dir]


@tool
def rag_search(
    query: str,
    top_k: int = 20,
    filter: dict | None = None,
) -> list[dict]:
    """Search the resume vector store for candidates matching the query.

    Args:
        query: Natural language or keyword search string.
        top_k: Maximum number of candidate results to return (default 20).
        filter: Optional metadata filter, e.g. {"name": "Alice_Johnson"}.

    Returns:
        List of dicts, each with keys:
            - "candidate_id" (str)
            - "name" (str)
            - "score" (float) — distance (lower = better match)
            - "excerpt" (str) — matching chunk text
        Results are deduplicated by candidate (best chunk per candidate).
    """
    if not query or not query.strip():
        return []

    retriever = _get_retriever()
    results = retriever.search(query=query, top_k=top_k, filter=filter)
    return results


def get_full_resume_text(candidate_id: str) -> str | None:
    """Retrieve the full resume text for a candidate (not a @tool, used internally).

    Args:
        candidate_id: The candidate identifier from indexing.

    Returns:
        Full resume text, or None if not found.
    """
    retriever = _get_retriever()
    return retriever.get_full_resume(candidate_id)