"""
test_kg_noise_filters.py — KG hygiene: path/URL guard, confidence floor,
and silent-empty-extraction visibility.

Mirrors the facade-test pattern in test_knowledge_graph.py: a real tmp DB via
the `db` fixture, init_kg(), and a faked ask_for_entity_extraction so we drive
_persist_extraction with crafted payloads.

Run: python -m pytest tests/test_kg_noise_filters.py -v
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def db(tmp_path):
    from assistant.storage.db import _reset_for_testing, init_db
    _reset_for_testing()
    db_obj = init_db(tmp_path / "test.db")
    yield db_obj
    db_obj.close()
    _reset_for_testing()


def _fake_extract(payload):
    async def _inner(text, source, context_hint=None):
        return payload
    return _inner


# ─── Fix 1: path / URL guard ─────────────────────────────────────────────────

def test_looks_like_path_or_url_matches():
    from assistant.knowledge_graph import _looks_like_path_or_url
    assert _looks_like_path_or_url(r"D:\Code\TENKA\tenka\TENKA — print.pdf")
    assert _looks_like_path_or_url("d:/Code/x.docx")
    assert _looks_like_path_or_url(r"\\server\share\file.txt")
    assert _looks_like_path_or_url("https://example.com/a")
    assert _looks_like_path_or_url("http://x")
    assert _looks_like_path_or_url("www.example.com")


def test_looks_like_path_or_url_rejects_real_names():
    from assistant.knowledge_graph import _looks_like_path_or_url
    for name in ("Akihito Shirogane", "New York", "Figma", "Razorpay", "C-3PO"):
        assert not _looks_like_path_or_url(name)


def test_persist_drops_path_entity(db):
    """A file path leaked as a (valid-type) 'tool' entity must not be stored."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()

    payload = {
        "entities": [
            {"type": "tool", "name": r"d:\Code\TENKA\tenka\TENKA — print.pdf", "confidence": 0.9},
            {"type": "person", "name": "Akihito", "confidence": 1.0},
        ],
        "facts": [], "relationships": [],
    }
    with patch("assistant.knowledge_graph.ask_for_entity_extraction", _fake_extract(payload), create=True):
        _run(kg_mod.ingest_turn("Read the profile for Akihito please.", "user_msg"))

    repo = kg_mod._get_repo()
    assert repo.find_entities_by_name(r"d:\Code\TENKA\tenka\TENKA — print.pdf") == []
    assert len(repo.find_entities_by_name("Akihito")) == 1


def test_persist_drops_path_fact_subject(db):
    """A fact whose subject is a path must be dropped (no auto-upserted entity)."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()

    payload = {
        "entities": [],
        "facts": [
            {"subject": "https://example.com", "predicate": "is", "object": "site", "confidence": 0.9},
        ],
        "relationships": [],
    }
    with patch("assistant.knowledge_graph.ask_for_entity_extraction", _fake_extract(payload), create=True):
        _run(kg_mod.ingest_turn("Check https://example.com for details.", "user_msg"))

    repo = kg_mod._get_repo()
    assert repo.find_entities_by_name("https://example.com") == []


# ─── Fix 2: confidence floor ─────────────────────────────────────────────────

def test_persist_drops_low_confidence_entity(db):
    """A transient low-confidence concept ('depressed' style) is dropped; a
    high-confidence entity in the same payload survives."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()

    payload = {
        "entities": [
            {"type": "concept", "name": "lost", "confidence": 0.5},
            {"type": "project", "name": "Solaris", "confidence": 0.95},
        ],
        "facts": [], "relationships": [],
    }
    with patch("assistant.knowledge_graph.ask_for_entity_extraction", _fake_extract(payload), create=True):
        _run(kg_mod.ingest_turn("I feel lost about the Solaris project.", "user_msg"))

    repo = kg_mod._get_repo()
    assert repo.find_entities_by_name("lost") == []
    assert len(repo.find_entities_by_name("Solaris")) == 1


def test_confidence_floor_boundary(db):
    """Confidence exactly at the floor is kept; just below is dropped."""
    import assistant.knowledge_graph as kg_mod
    floor = kg_mod._MIN_ENTITY_CONFIDENCE
    kg_mod._repo = None
    kg_mod.init_kg()

    payload = {
        "entities": [
            {"type": "concept", "name": "AtFloor", "confidence": floor},
            {"type": "concept", "name": "BelowFloor", "confidence": floor - 0.01},
        ],
        "facts": [], "relationships": [],
    }
    with patch("assistant.knowledge_graph.ask_for_entity_extraction", _fake_extract(payload), create=True):
        _run(kg_mod.ingest_turn("Talking about AtFloor and BelowFloor things.", "user_msg"))

    repo = kg_mod._get_repo()
    assert len(repo.find_entities_by_name("AtFloor")) == 1
    assert repo.find_entities_by_name("BelowFloor") == []


# ─── Fix 3: silent empty extraction is logged ────────────────────────────────

def test_empty_extraction_logs_info(db, caplog):
    """When the pre-filter passes but extraction returns nothing, an INFO line
    surfaces the silent gap (e.g. degraded provider fallback)."""
    import logging
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()

    empty = {"entities": [], "facts": [], "relationships": [], "commitments": []}
    with patch("assistant.knowledge_graph.ask_for_entity_extraction", _fake_extract(empty), create=True):
        with caplog.at_level(logging.INFO, logger="assistant.knowledge_graph"):
            # Text with a capitalized noun → passes _has_entity_signal.
            _run(kg_mod.ingest_turn("I went to Mumbai yesterday afternoon.", "user_msg"))

    assert any("returned nothing despite entity signal" in r.message for r in caplog.records)


def test_nonempty_extraction_does_not_log_empty(db, caplog):
    """A productive extraction must NOT emit the empty-extraction notice."""
    import logging
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()

    payload = {"entities": [{"type": "place", "name": "Mumbai", "confidence": 1.0}],
               "facts": [], "relationships": [], "commitments": []}
    with patch("assistant.knowledge_graph.ask_for_entity_extraction", _fake_extract(payload), create=True):
        with caplog.at_level(logging.INFO, logger="assistant.knowledge_graph"):
            _run(kg_mod.ingest_turn("I went to Mumbai yesterday afternoon.", "user_msg"))

    assert not any("returned nothing" in r.message for r in caplog.records)
