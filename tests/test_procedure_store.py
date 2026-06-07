"""
test_procedure_store.py — TP-1a: procedures module unit tests

Run: python test_procedure_store.py
All tests operate on an in-memory SQLite DB via monkeypatching.
"""

import json
import sys
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

import assistant.procedures as ps
from assistant import config as _config_stub
from assistant.storage.db import init_db, _reset_for_testing


# ── Helper: re-init with fresh in-memory-like DB for each test ───────────────

def _fresh_db(test: unittest.TestCase) -> None:
    """Point the store at a fresh temp file DB, create schema.

    Post-RG-1 the procedures module is a thin facade over storage/repos —
    state lives on the shared Database singleton (storage.db._instance) and
    on procedures._repo. Reset BOTH so prior-test rows don't leak in.
    """
    tmp = Path(tempfile.mkdtemp()) / "test_personality.db"
    _config_stub.SANDBOX_DIR = tmp.parent.parent  # SANDBOX_DIR/memory/...
    (tmp.parent.parent / "memory").mkdir(parents=True, exist_ok=True)

    # Drop the Database singleton (closes prior connection) and the procedures
    # facade's cached repo so the next call rebuilds against the fresh path.
    _reset_for_testing()
    ps._repo = None

    # Bind the singleton to the fresh path, then create procedure tables. The
    # storage.db schema includes user_shortcuts already, so we don't need to
    # hand-CREATE it like the pre-RG-1 helper did.
    init_db(tmp)
    ps.init_procedure_db()


SAMPLE_STEPS = [
    {"type": "app", "action": "open", "params": {"name": "notepad"}},
    {"type": "app", "action": "press_key", "params": {"key": "ctrl+n"}},
]


class TestCreateAndGet(unittest.TestCase):
    def setUp(self):
        _fresh_db(self)

    def test_create_returns_id(self):
        pid = ps.create_procedure("open notepad workflow", "Open Notepad", SAMPLE_STEPS)
        self.assertIsInstance(pid, int)
        self.assertGreater(pid, 0)

    def test_get_by_trigger(self):
        ps.create_procedure("open notepad workflow", "Open Notepad", SAMPLE_STEPS)
        proc = ps.get_procedure("open notepad workflow")
        self.assertIsNotNone(proc)
        self.assertEqual(proc["name"], "Open Notepad")
        self.assertEqual(len(proc["steps"]), 2)
        self.assertEqual(proc["steps"][0]["action"], "open")

    def test_get_case_insensitive(self):
        ps.create_procedure("start coding session", "Coding Session", SAMPLE_STEPS)
        proc = ps.get_procedure("START CODING SESSION")
        self.assertIsNotNone(proc)

    def test_get_missing_returns_none(self):
        self.assertIsNone(ps.get_procedure("nonexistent trigger"))

    def test_create_with_description_and_backend(self):
        pid = ps.create_procedure(
            "daily standup", "Daily Standup", SAMPLE_STEPS,
            backend="native", description="Opens Zoom and starts standup"
        )
        proc = ps.get_procedure("daily standup")
        self.assertEqual(proc["backend"], "native")
        self.assertEqual(proc["description"], "Opens Zoom and starts standup")


class TestCreateValidation(unittest.TestCase):
    def setUp(self):
        _fresh_db(self)

    def test_reserved_word_rejected(self):
        with self.assertRaises(ValueError):
            ps.create_procedure("yes", "Yes", SAMPLE_STEPS)

    def test_trigger_too_short(self):
        with self.assertRaises(ValueError):
            ps.create_procedure("ab", "Short", SAMPLE_STEPS)

    def test_empty_steps_rejected(self):
        with self.assertRaises(ValueError):
            ps.create_procedure("valid trigger", "Proc", [])

    def test_too_many_steps_rejected(self):
        steps = [{"type": "app", "action": "wait", "params": {"seconds": 1}}] * 21
        with self.assertRaises(ValueError):
            ps.create_procedure("overloaded proc", "Overloaded", steps)

    def test_duplicate_trigger_rejected(self):
        ps.create_procedure("my workflow", "First", SAMPLE_STEPS)
        with self.assertRaises(ValueError):
            ps.create_procedure("my workflow", "Second", SAMPLE_STEPS)

    def test_shortcut_conflict_raises(self):
        from datetime import datetime
        from assistant.storage.db import get_db
        now = datetime.now().isoformat()
        get_db()._conn.execute(
            "INSERT INTO user_shortcuts (trigger, intent, params_json, description, created_at, updated_at) "
            "VALUES ('open browser', 'open_browser', '{}', '', ?, ?)", (now, now)
        )
        get_db()._conn.commit()
        with self.assertRaises(ValueError):
            ps.create_procedure("open browser", "Open Browser Proc", SAMPLE_STEPS)


class TestUpdate(unittest.TestCase):
    def setUp(self):
        _fresh_db(self)
        self.pid = ps.create_procedure("my workflow", "My Workflow", SAMPLE_STEPS)

    def test_update_name(self):
        ps.update_procedure(self.pid, name="Updated Name")
        proc = ps.get_procedure("my workflow")
        self.assertEqual(proc["name"], "Updated Name")

    def test_update_steps(self):
        new_steps = [{"type": "app", "action": "close", "params": {"name": "notepad"}}]
        ps.update_procedure(self.pid, steps=new_steps)
        proc = ps.get_procedure("my workflow")
        self.assertEqual(len(proc["steps"]), 1)
        self.assertEqual(proc["steps"][0]["action"], "close")

    def test_update_trigger(self):
        ps.update_procedure(self.pid, trigger="renamed workflow")
        self.assertIsNone(ps.get_procedure("my workflow"))
        proc = ps.get_procedure("renamed workflow")
        self.assertIsNotNone(proc)

    def test_update_reserved_trigger_rejected(self):
        with self.assertRaises(ValueError):
            ps.update_procedure(self.pid, trigger="yes")

    def test_update_nonexistent_returns_false(self):
        result = ps.update_procedure(9999, name="ghost")
        self.assertFalse(result)


class TestDelete(unittest.TestCase):
    def setUp(self):
        _fresh_db(self)
        self.pid = ps.create_procedure("my workflow", "My Workflow", SAMPLE_STEPS)

    def test_soft_delete(self):
        result = ps.delete_procedure(self.pid)
        self.assertTrue(result)
        # Should not appear in get_procedure (enabled_only by default)
        self.assertIsNone(ps.get_procedure("my workflow"))

    def test_still_retrievable_by_id(self):
        ps.delete_procedure(self.pid)
        proc = ps.get_procedure_by_id(self.pid)
        self.assertIsNotNone(proc)
        self.assertEqual(proc["enabled"], 0)

    def test_delete_nonexistent_returns_false(self):
        result = ps.delete_procedure(9999)
        self.assertFalse(result)


class TestList(unittest.TestCase):
    def setUp(self):
        _fresh_db(self)

    def test_list_empty(self):
        self.assertEqual(ps.list_procedures(), [])

    def test_list_multiple(self):
        ps.create_procedure("trigger one", "One", SAMPLE_STEPS)
        ps.create_procedure("trigger two", "Two", SAMPLE_STEPS)
        procs = ps.list_procedures()
        self.assertEqual(len(procs), 2)

    def test_list_excludes_disabled(self):
        pid = ps.create_procedure("trigger one", "One", SAMPLE_STEPS)
        ps.create_procedure("trigger two", "Two", SAMPLE_STEPS)
        ps.delete_procedure(pid)
        procs = ps.list_procedures(enabled_only=True)
        self.assertEqual(len(procs), 1)
        self.assertEqual(procs[0]["name"], "Two")

    def test_list_includes_disabled_when_asked(self):
        pid = ps.create_procedure("trigger one", "One", SAMPLE_STEPS)
        ps.delete_procedure(pid)
        procs = ps.list_procedures(enabled_only=False)
        self.assertEqual(len(procs), 1)


class TestMatchTrigger(unittest.TestCase):
    def setUp(self):
        _fresh_db(self)
        ps.create_procedure("start my coding session", "Coding Session", SAMPLE_STEPS)
        ps.create_procedure("open youtube", "YouTube", SAMPLE_STEPS)

    def test_exact_match(self):
        result = ps.match_trigger("start my coding session")
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Coding Session")

    def test_case_insensitive(self):
        result = ps.match_trigger("START MY CODING SESSION")
        self.assertIsNotNone(result)

    def test_trailing_filler_stripped(self):
        result = ps.match_trigger("start my coding session please")
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Coding Session")

    def test_leading_assistant_name_stripped(self):
        result = ps.match_trigger("tenka start my coding session")
        self.assertIsNotNone(result)

    def test_prefix_match(self):
        # User says the trigger plus extra words
        result = ps.match_trigger("open youtube and search for cats")
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "YouTube")

    def test_no_match_for_partial(self):
        # "notepad" alone should not match "open youtube" or "start my coding session"
        result = ps.match_trigger("notepad")
        self.assertIsNone(result)

    def test_no_match_empty(self):
        self.assertIsNone(ps.match_trigger(""))
        self.assertIsNone(ps.match_trigger("  "))

    def test_disabled_not_matched(self):
        pid = ps.create_procedure("daily report", "Daily Report", SAMPLE_STEPS)
        ps.delete_procedure(pid)
        self.assertIsNone(ps.match_trigger("daily report"))

    def test_subsequence_match(self):
        ps.create_procedure("search on youtube", "YT Search", SAMPLE_STEPS)
        result = ps.match_trigger("search mechanical keyboard on youtube")
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "YT Search")

    def test_subsequence_needs_first_word_match(self):
        ps.create_procedure("search on youtube", "YT Search", SAMPLE_STEPS)
        # "find" doesn't match first trigger word "search"
        result = ps.match_trigger("find cats on youtube")
        self.assertIsNone(result)

    def test_subsequence_single_word_trigger_ignored(self):
        result = ps.match_trigger("open something completely different")
        self.assertIsNone(result)

    def test_subsequence_preserves_priority_over_exact(self):
        ps.create_procedure("search on youtube", "YT Search", SAMPLE_STEPS)
        result = ps.match_trigger("search on youtube")
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "YT Search")

    def test_subsequence_with_multiple_gap_words(self):
        ps.create_procedure("message on whatsapp", "WA Message", SAMPLE_STEPS)
        result = ps.match_trigger("message john happy birthday on whatsapp")
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "WA Message")


class TestRecordUsage(unittest.TestCase):
    def setUp(self):
        _fresh_db(self)
        self.pid = ps.create_procedure("my workflow", "My Workflow", SAMPLE_STEPS)

    def test_usage_increments(self):
        ps.record_usage(self.pid)
        ps.record_usage(self.pid)
        proc = ps.get_procedure_by_id(self.pid)
        self.assertEqual(proc["use_count"], 2)

    def test_last_used_set(self):
        ps.record_usage(self.pid)
        proc = ps.get_procedure_by_id(self.pid)
        self.assertIsNotNone(proc["last_used"])


class TestConflictCheck(unittest.TestCase):
    def setUp(self):
        _fresh_db(self)

    def test_reserved_word_flagged(self):
        msg = ps.check_trigger_conflict("yes")
        self.assertIsNotNone(msg)
        self.assertIn("reserved", msg)

    def test_existing_procedure_flagged(self):
        ps.create_procedure("my workflow", "My Workflow", SAMPLE_STEPS)
        msg = ps.check_trigger_conflict("my workflow")
        self.assertIsNotNone(msg)
        self.assertIn("procedure", msg)

    def test_existing_shortcut_flagged(self):
        from datetime import datetime
        from assistant.storage.db import get_db
        now = datetime.now().isoformat()
        get_db()._conn.execute(
            "INSERT INTO user_shortcuts (trigger, intent, params_json, description, created_at, updated_at) "
            "VALUES ('open browser', 'open_browser', '{}', '', ?, ?)", (now, now)
        )
        get_db()._conn.commit()
        msg = ps.check_trigger_conflict("open browser")
        self.assertIsNotNone(msg)
        self.assertIn("shortcut", msg)

    def test_clean_trigger_returns_none(self):
        msg = ps.check_trigger_conflict("start my morning routine")
        self.assertIsNone(msg)


class TestStepCountWarning(unittest.TestCase):
    def test_no_warning_below_threshold(self):
        steps = [{}] * 5
        self.assertIsNone(ps.step_count_warning(steps))

    def test_warning_at_threshold(self):
        steps = [{}] * 10
        msg = ps.step_count_warning(steps)
        self.assertIsNotNone(msg)
        self.assertIn("10", msg)

    def test_hard_cap_message(self):
        steps = [{}] * 20
        msg = ps.step_count_warning(steps)
        self.assertIsNotNone(msg)
        self.assertIn("maximum", msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
