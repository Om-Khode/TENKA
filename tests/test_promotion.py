"""Tests for manifest-based promotion bookkeeping on automation-cache (schema v13)."""
from __future__ import annotations


# ─── Schema v13 ───────────────────────────────────────────────────────

def test_v13_adds_promoted_intent_id_column(tmp_path):
    from assistant.storage.db import Database

    db = Database(tmp_path / "test.db")
    cols = [r["name"] for r in db.fetchall("PRAGMA table_info(automation_cache)")]
    assert "promoted_intent_id" in cols, (
        f"automation_cache should have promoted_intent_id column; got {cols}"
    )


def test_v13_adds_promoted_index(tmp_path):
    from assistant.storage.db import Database

    db = Database(tmp_path / "test.db")
    row = db.fetchone(
        "SELECT name FROM sqlite_master "
        "WHERE type='index' AND name='idx_ac_promoted'"
    )
    assert row is not None, "idx_ac_promoted index should exist after v13"


# ─── find_unpromoted / mark_promoted ──────────────────────────────────

def _make_repo(tmp_path):
    from assistant.storage.db import Database
    from assistant.storage.repos.automation_cache import AutomationCacheRepo

    db = Database(tmp_path / "test.db")
    return AutomationCacheRepo(db), db


def test_find_unpromoted_returns_unclaimed_entries(tmp_path):
    repo, _ = _make_repo(tmp_path)
    steps = [{"action": "press_key", "params": {"key": "space"}}]
    repo.save("native", "test_app", "play_music", "play music", steps)
    repo.save("native", "test_app", "pause_music", "pause music", steps)

    rows = repo.find_unpromoted()
    assert len(rows) == 2
    slugs = {r["goal_slug"] for r in rows}
    assert slugs == {"play_music", "pause_music"}


def test_find_unpromoted_returns_expected_keys(tmp_path):
    repo, _ = _make_repo(tmp_path)
    steps = [{"action": "click", "params": {"selector": "name:Go"}}]
    repo.save("native", "test_app", "go_action", "go now", steps)

    rows = repo.find_unpromoted()
    assert len(rows) == 1
    entry = rows[0]
    for key in ("backend", "app_name", "goal_slug", "goal_text",
                "steps_json", "created_at"):
        assert key in entry, f"missing key {key!r} in {entry}"
    assert entry["backend"] == "native"
    assert entry["app_name"] == "test_app"
    assert entry["goal_slug"] == "go_action"
    assert entry["goal_text"] == "go now"


def test_mark_promoted_excludes_entry_from_find_unpromoted(tmp_path):
    repo, _ = _make_repo(tmp_path)
    steps = [{"action": "press_key", "params": {"key": "space"}}]
    repo.save("native", "test_app", "play_music", "play music", steps)

    repo.mark_promoted(
        "native", "test_app", "play_music", "test_app.desktop:play"
    )
    assert repo.find_unpromoted() == []


def test_mark_promoted_is_keyed_on_triple(tmp_path):
    repo, _ = _make_repo(tmp_path)
    steps = [{"action": "press_key", "params": {"key": "space"}}]
    repo.save("native", "test_app", "play_music", "play music", steps)
    repo.save("native", "test_app", "pause_music", "pause music", steps)

    repo.mark_promoted(
        "native", "test_app", "play_music", "test_app.desktop:play"
    )

    rows = repo.find_unpromoted()
    assert len(rows) == 1
    assert rows[0]["goal_slug"] == "pause_music"


def test_mark_promoted_persists_intent_ref(tmp_path):
    repo, db = _make_repo(tmp_path)
    steps = [{"action": "press_key", "params": {"key": "space"}}]
    repo.save("native", "test_app", "play_music", "play music", steps)

    repo.mark_promoted(
        "native", "test_app", "play_music", "test_app.desktop:play"
    )

    row = db.fetchone(
        "SELECT promoted_intent_id FROM automation_cache "
        "WHERE backend = ? AND app_name = ? AND goal_slug = ?",
        ("native", "test_app", "play_music"),
    )
    assert row is not None
    assert row["promoted_intent_id"] == "test_app.desktop:play"


def test_mark_promoted_logs_warning_on_miss(tmp_path):
    """A no-match mark_promoted must not silently no-op (raises no exception,
    leaves the table untouched). The repo also logs a warning for visibility;
    we assert structural state here rather than log capture to match the
    surrounding test conventions.
    """
    repo, _ = _make_repo(tmp_path)

    # No row matches this triple — must not raise.
    repo.mark_promoted(
        "native", "ghost_app", "nonexistent", "ghost.desktop:x"
    )

    # Table is unchanged: nothing to find as unpromoted, nothing was inserted.
    assert repo.find_unpromoted() == []


def test_save_leaves_promoted_intent_id_null(tmp_path):
    """New saves must not silently mark themselves promoted."""
    repo, db = _make_repo(tmp_path)
    steps = [{"action": "press_key", "params": {"key": "space"}}]
    repo.save("native", "test_app", "play_music", "play music", steps)

    row = db.fetchone(
        "SELECT promoted_intent_id FROM automation_cache "
        "WHERE backend = ? AND app_name = ? AND goal_slug = ?",
        ("native", "test_app", "play_music"),
    )
    assert row is not None
    assert row["promoted_intent_id"] is None
