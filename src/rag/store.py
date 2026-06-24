"""
Agentic Profile Matching — ChromaDB Vector Store Initialization.

Provides a persistent ChromaDB client with the built-in ONNX embedding
function (all-MiniLM-L6-v2). No external embedding model or PyTorch
needed — the model ships with ChromaDB.

Architecture Reference: architecture.md Section 6.2 (rag_search tool),
                         Section 12 (Technology Stack — ChromaDB)
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# Module-level singleton so the ONNX model is loaded only once per process.
_embedding_function: DefaultEmbeddingFunction | None = None
_collection_name = "resumes"


def _get_embedding_function() -> DefaultEmbeddingFunction:
    """Return a module-level singleton DefaultEmbeddingFunction.

    The first call downloads/loads the ONNX model (~79 MB) into
    ``~/.cache/chroma/onnx_models/``. Subsequent calls reuse it.
    """
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = DefaultEmbeddingFunction()
    return _embedding_function


def get_vector_store(persist_dir: str | Path = "data/chroma_db") -> chromadb.Collection:
    """Initialize (or load) the persistent ChromaDB resume collection.

    Args:
        persist_dir: Directory where ChromaDB stores its data on disk.
                     Created automatically if it does not exist.

    Returns:
        A ChromaDB ``Collection`` object ready for upsert/query.

    Example::

        collection = get_vector_store("data/chroma_db")
        collection.count()  # number of chunks indexed
    """
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(persist_path),
        settings=Settings(anonymized_telemetry=False),
    )

    collection = client.get_or_create_collection(
        name=_collection_name,
        embedding_function=_get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def reset_collection(persist_dir: str | Path = "data/chroma_db") -> chromadb.Collection:
    """Delete and recreate the collection. Useful for tests and re-ingestion.

    Args:
        persist_dir: Directory where ChromaDB stores its data.

    Returns:
        A fresh, empty ChromaDB ``Collection``.
    """
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(persist_path),
        settings=Settings(anonymized_telemetry=False),
    )

    # Delete if it exists, then recreate.
    try:
        client.delete_collection(name=_collection_name)
    except Exception:
        pass  # collection does not exist yet

    collection = client.get_or_create_collection(
        name=_collection_name,
        embedding_function=_get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
    return collection