"""
Agentic Profile Matching — Resume Indexer.

Ingests resume PDFs from a directory, extracts text via PyMuPDF,
splits into overlapping chunks, and upserts into a ChromaDB collection.

Architecture Reference: architecture.md Section 6.2, Section 11 (Data Flow)
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import fitz  # PyMuPDF


class ResumeIndexer:
    """Indexes resume PDFs into a ChromaDB vector collection.

    Usage::

        from src.rag.store import get_vector_store
        collection = get_vector_store("data/chroma_db")
        indexer = ResumeIndexer()
        stats = indexer.ingest_directory("data/resumes/", collection)
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_directory(
        self,
        dir_path: str | Path,
        collection: object,
    ) -> dict[str, int]:
        """Scan a directory for PDF files and index all of them.

        Args:
            dir_path:   Directory containing resume PDFs.
            collection: A ChromaDB ``Collection`` object.

        Returns:
            Dict with keys: ``pdfs_processed``, ``total_chunks``,
            ``collection_count``.
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Resume directory not found: {dir_path}")

        pdf_files = sorted(dir_path.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {dir_path}")

        total_chunks = 0
        for pdf_path in pdf_files:
            chunks = self._process_resume(pdf_path, collection)
            total_chunks += chunks

        return {
            "pdfs_processed": len(pdf_files),
            "total_chunks": total_chunks,
            "collection_count": collection.count(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_resume(
        self,
        pdf_path: Path,
        collection: object,
    ) -> int:
        """Extract text from one PDF, chunk it, and upsert into collection.

        Returns the number of chunks upserted.
        """
        candidate_name = pdf_path.stem  # e.g. "Alice_Johnson_React_Developer"
        candidate_id = self._make_candidate_id(candidate_name, pdf_path)

        full_text = self._extract_text(pdf_path)
        if not full_text.strip():
            return 0

        chunks = self._chunk_text(full_text)
        if not chunks:
            return 0

        self._upsert_chunks(
            collection=collection,
            candidate_id=candidate_id,
            name=candidate_name,
            full_text=full_text,
            chunks=chunks,
        )
        return len(chunks)

    @staticmethod
    def _make_candidate_id(stem: str, pdf_path: Path) -> str:
        """Derive a deterministic candidate_id from the filename.

        Strategy: slugify the stem (replace non-alphanum with _), then
        hash the full path for uniqueness if stems collide.
        """
        slug = re.sub(r"[^A-Za-z0-9]", "_", stem).strip("_")
        slug = re.sub(r"_+", "_", slug)
        # Append a short hash of the full path for global uniqueness
        path_hash = str(hash(str(pdf_path.resolve())))[:8]
        return f"{slug}_{path_hash}"

    @staticmethod
    def _extract_text(pdf_path: Path) -> str:
        """Extract all text from a PDF using PyMuPDF.

        Each page's text is separated by a double newline.
        """
        doc = fitz.open(str(pdf_path))
        pages: list[str] = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text.strip())
        doc.close()
        return "\n\n".join(pages)

    def _chunk_text(self, text: str) -> list[dict]:
        """Split text into overlapping chunks with metadata.

        Each chunk is a dict with keys:
            - ``text``:  the chunk content
            - ``index``: zero-based chunk index
            - ``start``: character offset in original text
            - ``end``:   character offset end

        Args:
            text: Full resume text.

        Returns:
            List of chunk dicts.
        """
        if not text or not text.strip():
            return []

        text = text.strip()
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks: list[dict] = []

        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # If this is not the last chunk, try to break at a newline
            # to avoid cutting mid-sentence.
            if end < len(text):
                # Look for the last newline within the chunk
                nl_pos = text.rfind("\n", start, end)
                if nl_pos > start + self.chunk_size // 2:
                    end = nl_pos + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "index": idx,
                        "start": start,
                        "end": end,
                    }
                )
                idx += 1

            # Advance by step, but ensure progress
            start = start + step
            if start >= len(text):
                break

        return chunks

    def _upsert_chunks(
        self,
        collection: object,
        candidate_id: str,
        name: str,
        full_text: str,
        chunks: list[dict],
    ) -> None:
        """Upsert chunks into the ChromaDB collection.

        ChromaDB upsert requires lists of ids, documents, and metadatas.

        We also store the full resume text on the *first* chunk's metadata
        so ``get_full_resume`` can retrieve it without reassembly.
        """
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for chunk in chunks:
            chunk_id = f"{candidate_id}__chunk_{chunk['index']:04d}"
            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append(
                {
                    "candidate_id": candidate_id,
                    "name": name,
                    "chunk_index": chunk["index"],
                    "char_start": chunk["start"],
                    "char_end": chunk["end"],
                    # Store full text only on chunk 0 to save storage
                    "full_text": full_text if chunk["index"] == 0 else "",
                }
            )

        # ChromaDB batch upsert
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )