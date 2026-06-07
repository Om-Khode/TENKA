"""Tests for knowledge-graph item A — temporal grounding.

Covers:
  * _relative_date pure helper (LLM never computes dates — Python does).
  * Repo: add_fact accepts event_at; get_facts_for_entity returns it.
  * Facade: _format_entity_block renders relative-date suffix when present.

Pure unit + SQLite — safe in isolation.
"""

from datetime import datetime
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.storage.db import _reset_for_testing, init_db
from assistant.storage.repos.knowledge_graph import KnowledgeGraphRepo


# ─── _relative_date helper (pure) ─────────────────────────────────────────


def _rd():
    """Lazy import so a NameError surfaces at call time, not collection."""
    from assistant.knowledge_graph import _relative_date
    return _relative_date


NOW = datetime(2026, 6, 3, 14, 30)


def test_relative_date_today():
    assert _rd()("2026-06-03", NOW) == "today"


def test_relative_date_yesterday_and_tomorrow():
    assert _rd()("2026-06-02", NOW) == "yesterday"
    assert _rd()("2026-06-04", NOW) == "tomorrow"


def test_relative_date_days_window():
    # 2-6 days produces "X days ago" / "in X days"
    assert _rd()("2026-06-01", NOW) == "2 days ago"
    assert _rd()("2026-05-29", NOW) == "5 days ago"
    assert _rd()("2026-06-07", NOW) == "in 4 days"


def test_relative_date_weeks_window():
    # 7-13 days = "last week"; 14-30 days = "N weeks ago"
    assert _rd()("2026-05-27", NOW) == "last week"
    assert _rd()("2026-05-13", NOW) == "3 weeks ago"


def test_relative_date_same_year_month():
    # >30 days but same year → month name
    assert _rd()("2026-03-15", NOW) == "in March"
    assert _rd()("2026-10-01", NOW) == "in October"


def test_relative_date_different_year():
    assert _rd()("2024-08-10", NOW) == "in 2024"
    assert _rd()("2025-01-01", NOW) == "in 2025"


def test_relative_date_partial_year_month():
    # ISO can be partial; helper handles year-only and year-month
    assert _rd()("2024", NOW) == "in 2024"
    assert _rd()("2026-03", NOW) == "in March"
    assert _rd()("2024-08", NOW) == "in August 2024"


def test_relative_date_invalid_or_empty_returns_none():
    assert _rd()("", NOW) is None
    assert _rd()(None, NOW) is None  # type: ignore[arg-type]
    assert _rd()("garbage", NOW) is None
    assert _rd()("not-a-date", NOW) is None


# ─── Repo: add_fact event_at ───────────────────────────────────────────────


def _noop_loader():
    class _M:
        def encode(self, *_a, **_kw):
            import numpy as np
            return np.zeros(3)
    return lambda: _M()


@pytest.fixture
def kg_repo(tmp_path):
    _reset_for_testing()
    db = init_db(tmp_path / "test.db")
    repo = KnowledgeGraphRepo(db, embed_model_loader=_noop_loader())
    yield repo
    db.close()
    _reset_for_testing()


def test_add_fact_persists_event_at(kg_repo):
    eid, _ = kg_repo.upsert_entity(entity_type="person", name="Aanya", source="user_msg")
    fid = kg_repo.add_fact(
        eid, "moved_to", "Bangalore", source="user_msg",
        event_at="2026-05-15",
    )
    facts = kg_repo.get_facts_for_entity(eid)
    assert len(facts) == 1
    assert facts[0]["id"] == fid
    assert facts[0]["event_at"] == "2026-05-15"


def test_add_fact_event_at_defaults_to_none(kg_repo):
    eid, _ = kg_repo.upsert_entity(entity_type="person", name="Aanya", source="user_msg")
    kg_repo.add_fact(eid, "is_a", "engineer", source="user_msg")
    facts = kg_repo.get_facts_for_entity(eid)
    assert facts[0]["event_at"] is None


# ─── Facade: _format_entity_block renders date suffix ─────────────────────


def test_format_entity_block_includes_relative_date():
    """When a fact has event_at, the rendered line includes a relative-date
    suffix produced by _relative_date (LLM doesn't compute dates)."""
    from assistant.knowledge_graph import _format_entity_block
    ctx = {
        "entity": {"display_name": "Aanya", "type": "person"},
        "facts": [
            {"predicate": "moved_to", "object": "Bangalore",
             "event_at": "2026-05-13", "confidence": 1.0},
        ],
        "neighbors": [],
    }
    # 3 weeks before NOW
    line = _format_entity_block(ctx, now=NOW)
    assert "Bangalore" in line
    assert "3 weeks ago" in line


def test_format_entity_block_omits_suffix_when_no_event_at():
    from assistant.knowledge_graph import _format_entity_block
    ctx = {
        "entity": {"display_name": "Aanya", "type": "person"},
        "facts": [
            {"predicate": "is_a", "object": "engineer",
             "event_at": None, "confidence": 1.0},
        ],
        "neighbors": [],
    }
    line = _format_entity_block(ctx, now=NOW)
    # No relative-date marker tokens present
    for marker in ("ago", "tomorrow", "yesterday", "today", "in 20", "last week"):
        assert marker not in line, f"unexpected date marker {marker!r} in: {line}"
