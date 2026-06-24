"""
Tool: file_tools — File system utility tools.

Simple wrappers around os/pathlib for listing, reading, writing, searching,
and retrieving file metadata. No LLM needed.

Architecture Reference: architecture.md Section 6.5
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool


@tool
def list_files(directory: str) -> list[dict]:
    """List files in a directory with metadata.

    Args:
        directory: Path to the directory to list.

    Returns:
        List of dicts with keys: name, path, size (bytes), is_dir, modified.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return [{"error": f"Directory not found: {directory}"}]

    result = []
    for entry in sorted(dir_path.iterdir()):
        try:
            stat = entry.stat()
            result.append({
                "name": entry.name,
                "path": str(entry),
                "size": stat.st_size,
                "is_dir": entry.is_dir(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        except OSError:
            result.append({"name": entry.name, "path": str(entry), "error": "permission denied"})

    return result


@tool
def read_file(path: str, max_chars: int = 10000) -> dict:
    """Read the content of a file.

    Args:
        path: Path to the file.
        max_chars: Maximum characters to return (default 10000).

    Returns:
        Dict with keys: content, path, size, truncated (bool).
    """
    file_path = Path(path)
    if not file_path.is_file():
        return {"error": f"File not found: {path}"}

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        truncated = len(content) > max_chars
        if truncated:
            content = content[:max_chars] + "\n... [truncated]"
        return {
            "content": content,
            "path": str(file_path),
            "size": file_path.stat().st_size,
            "truncated": truncated,
        }
    except OSError as e:
        return {"error": f"Failed to read {path}: {e}"}


@tool
def write_file(path: str, content: str) -> dict:
    """Write content to a file. Creates parent directories if needed.

    Args:
        path: Path to the file to write.
        content: Text content to write.

    Returns:
        Dict with keys: path, size, success (bool).
    """
    file_path = Path(path)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return {
            "path": str(file_path),
            "size": len(content),
            "success": True,
        }
    except OSError as e:
        return {"error": f"Failed to write {path}: {e}", "success": False}


@tool
def search_files(query: str, directory: str = ".", max_results: int = 20) -> list[dict]:
    """Search for files whose names contain the query string (case-insensitive).

    Args:
        query: Search string to match against file names.
        directory: Root directory to search in (default: current).
        max_results: Maximum number of results (default 20).

    Returns:
        List of dicts with keys: name, path, size, is_dir.
    """
    query_lower = query.lower()
    root = Path(directory)
    if not root.is_dir():
        return [{"error": f"Directory not found: {directory}"}]

    results = []
    for entry in root.rglob("*"):
        if query_lower in entry.name.lower():
            try:
                results.append({
                    "name": entry.name,
                    "path": str(entry),
                    "size": entry.stat().st_size if entry.is_file() else 0,
                    "is_dir": entry.is_dir(),
                })
            except OSError:
                pass
            if len(results) >= max_results:
                break

    return results


@tool
def get_file_metadata(path: str) -> dict:
    """Retrieve file metadata (size, dates, type).

    Args:
        path: Path to the file or directory.

    Returns:
        Dict with keys: path, name, size, is_dir, is_file, modified, created, extension.
    """
    p = Path(path)
    if not p.exists():
        return {"error": f"Path not found: {path}"}

    try:
        stat = p.stat()
        return {
            "path": str(p),
            "name": p.name,
            "size": stat.st_size,
            "is_dir": p.is_dir(),
            "is_file": p.is_file(),
            "extension": p.suffix,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        }
    except OSError as e:
        return {"error": f"Failed to get metadata for {path}: {e}"}