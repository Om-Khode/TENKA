"""
test_file_formats.py — Rich document reading in file_manager.read_file.

Verifies the generic extractor registry: dispatch by extension, graceful
missing-dependency messaging, extraction failure handling, and that plain-text
reading is unchanged. Extractor libraries (python-docx, pypdf, …) are optional,
so these tests stub the extractors rather than requiring the packages.

Run: python -m pytest tests/test_file_formats.py -v
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import assistant.file_manager as fm


# ─── 1. Registry shape ───────────────────────────────────────────────────────

def test_rich_extensions_cover_popular_formats():
    """The popular office/document formats are registered."""
    for ext in (".docx", ".pdf", ".xlsx", ".pptx", ".doc"):
        assert ext in fm.RICH_DOC_EXTENSIONS
        assert ext in fm._DOC_EXTRACTORS


def test_rich_and_plaintext_sets_are_disjoint():
    """A format is either plain-text or rich — never both (no ambiguous routing)."""
    assert fm.RICH_DOC_EXTENSIONS.isdisjoint(fm.READABLE_EXTENSIONS)


# ─── 2. Dispatch ─────────────────────────────────────────────────────────────

def test_read_file_routes_rich_to_extractor(tmp_path):
    """read_file on a rich format calls the registered extractor."""
    f = tmp_path / "doc.docx"
    f.write_bytes(b"PK\x03\x04")  # any bytes; extractor is stubbed
    with patch.dict(fm._DOC_EXTRACTORS, {".docx": ("python-docx", lambda p: "extracted body")}):
        assert fm.read_file(f) == "extracted body"


def test_read_file_rich_truncates(tmp_path):
    """Rich extraction is truncated to MAX_READ_CHARS like plain text."""
    f = tmp_path / "big.pdf"
    f.write_bytes(b"%PDF-1.4")
    big = "x" * (fm.MAX_READ_CHARS + 500)
    with patch.dict(fm._DOC_EXTRACTORS, {".pdf": ("pypdf", lambda p: big)}):
        out = fm.read_file(f)
        assert out.endswith("... (truncated)")
        assert len(out) <= fm.MAX_READ_CHARS + 20


# ─── 3. Graceful degradation ─────────────────────────────────────────────────

def test_missing_dependency_message(tmp_path):
    """A missing extractor library yields a clear pip-install message, not a crash."""
    f = tmp_path / "sheet.xlsx"
    f.write_bytes(b"PK\x03\x04")

    def _raise(_):
        raise ImportError("no module named openpyxl")

    with patch.dict(fm._DOC_EXTRACTORS, {".xlsx": ("openpyxl", _raise)}):
        out = fm.read_file(f)
        assert "openpyxl" in out
        assert "pip install" in out


def test_extraction_failure_message(tmp_path):
    """A corrupt/locked document yields a friendly message, never an exception."""
    f = tmp_path / "broken.pptx"
    f.write_bytes(b"not really pptx")

    def _boom(_):
        raise ValueError("bad zip")

    with patch.dict(fm._DOC_EXTRACTORS, {".pptx": ("python-pptx", _boom)}):
        out = fm.read_file(f)
        assert "couldn't read" in out.lower()


def test_empty_document_message(tmp_path):
    """A document with no extractable text reports that, rather than empty string."""
    f = tmp_path / "blank.docx"
    f.write_bytes(b"PK\x03\x04")
    with patch.dict(fm._DOC_EXTRACTORS, {".docx": ("python-docx", lambda p: "   ")}):
        out = fm.read_file(f)
        assert "no readable text" in out.lower()


# ─── 4. Unchanged behavior ───────────────────────────────────────────────────

def test_plaintext_read_unchanged(tmp_path):
    """Plain-text reading still works exactly as before."""
    f = tmp_path / "notes.txt"
    f.write_text("hello world", encoding="utf-8")
    assert fm.read_file(f) == "hello world"


def test_unsupported_extension_message(tmp_path):
    """A genuinely unsupported binary type still returns the can't-read message."""
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"\x00\x00\x00\x18ftyp")
    out = fm.read_file(f)
    assert "can't read" in out.lower()
    assert ".mp4" in out


def test_missing_file_message(tmp_path):
    """Nonexistent path reports not-found."""
    out = fm.read_file(tmp_path / "ghost.docx")
    assert "not found" in out.lower()
