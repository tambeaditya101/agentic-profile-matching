#!/usr/bin/env python3
"""
Standalone script to build/rebuild the resume vector index.

Usage:
    python scripts/ingest_resumes.py                  # uses defaults
    python scripts/ingest_resumes.py --dir data/resumes/ --persist data/chroma_db/
    python scripts/ingest_resumes.py --reset           # wipe and re-index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.rag.indexer import ResumeIndexer
from src.rag.store import get_vector_store, reset_collection


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest resume PDFs into ChromaDB vector store")
    parser.add_argument(
        "--dir",
        default="data/resumes/",
        help="Directory containing resume PDFs (default: data/resumes/)",
    )
    parser.add_argument(
        "--persist",
        default="data/chroma_db/",
        help="ChromaDB persistence directory (default: data/chroma_db/)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection and re-index from scratch",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Characters per chunk (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Character overlap between chunks (default: 200)",
    )
    args = parser.parse_args()

    resume_dir = Path(args.dir)
    persist_dir = Path(args.persist)

    # Validate
    if not resume_dir.is_dir():
        print(f"ERROR: Resume directory not found: {resume_dir}")
        sys.exit(1)

    pdf_count = len(list(resume_dir.glob("*.pdf")))
    if pdf_count == 0:
        print(f"ERROR: No PDF files found in {resume_dir}")
        sys.exit(1)

    print(f"Ingesting {pdf_count} PDFs from {resume_dir}")
    print(f"Persisting to {persist_dir}")
    print(f"Chunk size: {args.chunk_size}, overlap: {args.chunk_overlap}")

    # Get or reset collection
    if args.reset:
        print("\nResetting collection...")
        collection = reset_collection(persist_dir)
    else:
        collection = get_vector_store(persist_dir)

    # Index
    indexer = ResumeIndexer(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    stats = indexer.ingest_directory(resume_dir, collection)

    print(f"\n--- Ingestion Complete ---")
    print(f"  PDFs processed  : {stats['pdfs_processed']}")
    print(f"  Total chunks    : {stats['total_chunks']}")
    print(f"  Collection size : {stats['collection_count']} chunks")
    print(f"  Persist dir     : {persist_dir}")


if __name__ == "__main__":
    main()