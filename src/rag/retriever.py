"""
Agentic Profile Matching — Resume Retriever.

Searches the ChromaDB collection and reassembles full resume text
for individual candidates.

Architecture Reference: architecture.md Section 6.2 (rag_search tool)
"""

from __future__ import annotations

from typing import Any


class ResumeRetriever:
    """Query the resume vector store and retrieve full candidate resumes.

    Usage::

        from src.rag.store import get_vector_store
        from src.rag.retriever import ResumeRetriever

        collection = get_vector_store("data/chroma_db")
        retriever = ResumeRetriever(collection)

        results = retriever.search("React developer with 3 years experience", top_k=5)
        for r in results:
            print(r["name"], r["score"])

        full = retriever.get_full_resume("alice_johnson_12345678")
    """

    def __init__(self, collection: object) -> None:
        """Initialize with a ChromaDB Collection object.

        Args:
            collection: ChromaDB ``Collection`` (from ``get_vector_store()``).
        """
        self._collection = collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 20,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over the resume collection.

        Args:
            query:  Natural language or keyword search string.
            top_k:  Maximum number of results to return.
            filter: Optional ChromaDB metadata filter
                    (e.g. ``{"name": "Alice_Johnson"}``).

        Returns:
            List of dicts with keys:
                - ``candidate_id`` (str)
                - ``name`` (str)
                - ``score`` (float) — distance from query (lower = better)
                - ``excerpt`` (str) — the matching chunk text
            Deduplicated by candidate_id (best chunk per candidate).
        """
        query_kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": top_k,
        }
        if filter is not None:
            query_kwargs["where"] = filter

        results = self._collection.query(**query_kwargs)

        # ChromaDB returns: ids[0], documents[0], metadatas[0], distances[0]
        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        raw_ids = results["ids"][0]
        raw_docs = results.get("documents", [[]])[0]
        raw_metas = results.get("metadatas", [[]])[0]
        raw_dists = results.get("distances", [[]])[0]

        # Deduplicate: keep best (lowest distance) chunk per candidate
        best_per_candidate: dict[str, dict[str, Any]] = {}
        for i, cid in enumerate(raw_ids):
            meta = raw_metas[i] if i < len(raw_metas) else {}
            cand_id = meta.get("candidate_id", cid)
            distance = raw_dists[i] if i < len(raw_dists) else 1.0

            if cand_id not in best_per_candidate or distance < best_per_candidate[cand_id]["score"]:
                best_per_candidate[cand_id] = {
                    "candidate_id": cand_id,
                    "name": meta.get("name", "Unknown"),
                    "score": distance,
                    "excerpt": raw_docs[i] if i < len(raw_docs) else "",
                }

        # Sort by score (lowest distance = best match), return top_k
        sorted_results = sorted(best_per_candidate.values(), key=lambda x: x["score"])
        return sorted_results[:top_k]

    def get_full_resume(self, candidate_id: str) -> str | None:
        """Retrieve and reassemble the full resume text for a candidate.

        Looks up all chunks for the given candidate_id and concatenates
        them in chunk_index order. If the full text was stored on chunk 0
        (our indexing convention), returns that directly.

        Args:
            candidate_id: The candidate identifier used during indexing.

        Returns:
            The full resume text, or None if not found.
        """
        results = self._collection.get(
            where={"candidate_id": candidate_id},
            include=["documents", "metadatas"],
        )

        if not results or not results.get("ids"):
            return None

        metadatas = results.get("metadatas", [])
        documents = results.get("documents", [])

        if not metadatas:
            return None

        # Fast path: full_text stored on chunk 0
        for meta in metadatas:
            full = meta.get("full_text", "")
            if full:
                return full

        # Fallback: reassemble from all chunks in order
        indexed_chunks: list[tuple[int, str]] = []
        for i, meta in enumerate(metadatas):
            idx = meta.get("chunk_index", 0)
            doc = documents[i] if i < len(documents) else ""
            indexed_chunks.append((idx, doc))

        indexed_chunks.sort(key=lambda x: x[0])
        return "\n\n".join(text for _, text in indexed_chunks)

    def get_all_candidate_ids(self) -> list[str]:
        """Return a deduplicated list of all candidate IDs in the store."""
        results = self._collection.get(include=["metadatas"])
        if not results or not results.get("metadatas"):
            return []

        seen: set[str] = set()
        ids: list[str] = []
        for meta in results["metadatas"]:
            cid = meta.get("candidate_id", "")
            if cid and cid not in seen:
                seen.add(cid)
                ids.append(cid)
        return ids