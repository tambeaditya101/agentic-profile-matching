"""
Agentic Profile Matching — Phase 0 smoke test.
Verifies all core packages are importable.
"""

import sys


def test_import_langgraph():
    import langgraph
    assert langgraph is not None
    print(f"  langgraph        : installed")


def test_import_langchain_core():
    import langchain_core
    assert langchain_core.__version__
    print(f"  langchain-core   : {langchain_core.__version__}")


def test_import_langchain_google_genai():
    import langchain_google_genai
    assert langchain_google_genai is not None
    print("  langchain-google-genai : installed")


def test_import_chromadb():
    import chromadb
    assert chromadb.__version__
    print(f"  chromadb         : {chromadb.__version__}")


def test_import_pydantic():
    import pydantic
    assert pydantic.__version__
    print(f"  pydantic         : {pydantic.__version__}")


def test_import_streamlit():
    import streamlit
    assert streamlit.__version__
    print(f"  streamlit        : {streamlit.__version__}")


def test_import_click():
    import click
    assert click.__version__
    print(f"  click            : {click.__version__}")


def test_import_pymupdf():
    import fitz  # PyMuPDF
    assert fitz.version
    print(f"  PyMuPDF          : {fitz.version}")


def test_import_dotenv():
    from dotenv import load_dotenv
    assert callable(load_dotenv)
    print("  python-dotenv    : importable")


def test_import_project_packages():
    """Verify our own packages are importable."""
    import src.agent  # noqa: F401
    import src.tools  # noqa: F401
    import src.rag  # noqa: F401
    print("  src packages     : all importable")


if __name__ == "__main__":
    print("Phase 0 smoke test — verifying all dependencies...")
    all_passed = True
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
            except Exception as e:
                print(f"  FAILED: {name} — {e}")
                all_passed = False

    if all_passed:
        print("\nPhase 0 complete. All dependencies verified.")
        sys.exit(0)
    else:
        print("\nPhase 0 incomplete. Fix failures above.")
        sys.exit(1)