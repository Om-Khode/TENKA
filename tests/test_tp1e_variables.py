"""
test_tp1e_variables.py — TP-1e: Natural language variable resolution tests

Tests that natural language phrases during teaching are converted to {variable}
tokens, that the "paste" step pattern works, and that step descriptions show
human-friendly text for variable placeholders.

Run: python test_tp1e_variables.py
"""

import asyncio
import json
import re
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

import assistant.procedures as ps
import assistant.actions as actions
from assistant import config as _config_stub


def _fresh_db():
    tmp = Path(tempfile.mkdtemp()) / "test_personality.db"
    _config_stub.SANDBOX_DIR = tmp.parent.parent
    (tmp.parent.parent / "memory").mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(tmp), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS user_shortcuts (
            trigger TEXT PRIMARY KEY,
            intent TEXT NOT NULL,
            params_json TEXT NOT NULL DEFAULT '{}',
            description TEXT NOT NULL DEFAULT '',
            times_used INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)
    conn.commit()
    ps._DB_PATH = tmp
    ps._DB_DIR = tmp.parent
    ps._conn = conn
    ps.init_procedure_db()


def run_async(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────────────
# _detect_nl_variable
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectNLVariable(unittest.TestCase):

    def _detect(self, text):
        return actions._detect_nl_variable(text)

    # ── {user_input} ─────────────────────────────────────────────────────────

    def test_whatever_i_say(self):
        self.assertEqual(self._detect("whatever I say"), "{user_input}")

    def test_whatever_i_said(self):
        self.assertEqual(self._detect("whatever I said"), "{user_input}")

    def test_whatever_i_asked(self):
        self.assertEqual(self._detect("whatever I asked"), "{user_input}")

    def test_whatever_i_asked_for(self):
        self.assertEqual(self._detect("whatever I asked for"), "{user_input}")

    def test_what_i_want(self):
        self.assertEqual(self._detect("what I want"), "{user_input}")

    def test_what_i_mentioned(self):
        self.assertEqual(self._detect("what I mentioned"), "{user_input}")

    def test_whatever_i_search(self):
        self.assertEqual(self._detect("whatever I search"), "{user_input}")

    def test_what_i_type(self):
        self.assertEqual(self._detect("what I type"), "{user_input}")

    def test_rest_of_what_i_say(self):
        self.assertEqual(self._detect("the rest of what I say"), "{user_input}")

    def test_rest_of_my_input(self):
        self.assertEqual(self._detect("rest of my input"), "{user_input}")

    def test_my_input(self):
        self.assertEqual(self._detect("my input"), "{user_input}")

    def test_the_query(self):
        self.assertEqual(self._detect("the query"), "{user_input}")

    def test_my_search_term(self):
        self.assertEqual(self._detect("my search term"), "{user_input}")

    def test_user_input(self):
        self.assertEqual(self._detect("user input"), "{user_input}")

    # ── {date} ───────────────────────────────────────────────────────────────

    def test_todays_date(self):
        self.assertEqual(self._detect("today's date"), "{date}")

    def test_the_date(self):
        self.assertEqual(self._detect("date"), "{date}")

    def test_current_date(self):
        self.assertEqual(self._detect("the current date"), "{date}")

    def test_date_today(self):
        self.assertEqual(self._detect("the date today"), "{date}")

    def test_todays_no_apostrophe(self):
        self.assertEqual(self._detect("todays date"), "{date}")

    # ── {time} ───────────────────────────────────────────────────────────────

    def test_the_time(self):
        self.assertEqual(self._detect("the time"), "{time}")

    def test_current_time(self):
        self.assertEqual(self._detect("current time"), "{time}")

    def test_time_bare(self):
        self.assertEqual(self._detect("time"), "{time}")

    def test_what_time_it_is(self):
        self.assertEqual(self._detect("what time it is"), "{time}")

    # ── {clipboard} ──────────────────────────────────────────────────────────

    def test_clipboard(self):
        self.assertEqual(self._detect("clipboard"), "{clipboard}")

    def test_my_clipboard(self):
        self.assertEqual(self._detect("my clipboard"), "{clipboard}")

    def test_whats_in_clipboard(self):
        self.assertEqual(self._detect("what's in my clipboard"), "{clipboard}")

    def test_what_i_copied(self):
        self.assertEqual(self._detect("what I copied"), "{clipboard}")

    def test_the_copied_text(self):
        self.assertEqual(self._detect("the copied text"), "{clipboard}")

    def test_clipboard_contents(self):
        self.assertEqual(self._detect("clipboard contents"), "{clipboard}")

    # ── passthrough (no match) ───────────────────────────────────────────────

    def test_literal_text_passthrough(self):
        self.assertEqual(self._detect("hello world"), "hello world")

    def test_normal_word_passthrough(self):
        self.assertEqual(self._detect("meeting notes"), "meeting notes")

    def test_explicit_curly_passthrough(self):
        self.assertEqual(self._detect("{contact}"), "{contact}")

    def test_trailing_punctuation_stripped(self):
        self.assertEqual(self._detect("whatever I say."), "{user_input}")


# ─────────────────────────────────────────────────────────────────────────────
# Step parser: NL variable detection in "type" steps
# ─────────────────────────────────────────────────────────────────────────────

class TestParseStepNLVariables(unittest.TestCase):

    def _parse(self, text):
        return actions._parse_teaching_step(text)

    def test_type_whatever_i_say(self):
        s = self._parse("type whatever I say")
        self.assertEqual(s["params"]["text"], "{user_input}")

    def test_type_todays_date(self):
        s = self._parse("type today's date")
        self.assertEqual(s["params"]["text"], "{date}")

    def test_type_current_time(self):
        s = self._parse("type the current time")
        self.assertEqual(s["params"]["text"], "{time}")

    def test_type_clipboard(self):
        s = self._parse("type my clipboard")
        self.assertEqual(s["params"]["text"], "{clipboard}")

    def test_type_what_i_copied(self):
        s = self._parse("type what I copied")
        self.assertEqual(s["params"]["text"], "{clipboard}")

    def test_type_literal_unchanged(self):
        s = self._parse("type hello world")
        self.assertEqual(s["params"]["text"], "hello world")

    def test_type_whatever_i_want_in_window(self):
        s = self._parse("type whatever I want in notepad")
        self.assertEqual(s["params"]["text"], "{user_input}")
        self.assertEqual(s["params"]["window"], "notepad")

    def test_type_explicit_var_unchanged(self):
        s = self._parse("type {contact}")
        self.assertEqual(s["params"]["text"], "{contact}")

    def test_type_my_search_term(self):
        s = self._parse("type my search term")
        self.assertEqual(s["params"]["text"], "{user_input}")


# ─────────────────────────────────────────────────────────────────────────────
# Paste step pattern
# ─────────────────────────────────────────────────────────────────────────────

class TestPasteStep(unittest.TestCase):

    def _parse(self, text):
        return actions._parse_teaching_step(text)

    def test_paste_bare(self):
        s = self._parse("paste")
        self.assertEqual(s["action"], "press_key")
        self.assertEqual(s["params"]["key"], "ctrl+v")

    def test_paste_from_clipboard(self):
        s = self._parse("paste from clipboard")
        self.assertEqual(s["action"], "press_key")
        self.assertEqual(s["params"]["key"], "ctrl+v")

    def test_paste_clipboard(self):
        s = self._parse("paste the clipboard")
        self.assertEqual(s["action"], "press_key")
        self.assertEqual(s["params"]["key"], "ctrl+v")


# ─────────────────────────────────────────────────────────────────────────────
# Step description for variable steps
# ─────────────────────────────────────────────────────────────────────────────

class TestStepDescriptionVariables(unittest.TestCase):

    def _desc(self, step):
        return actions._step_description(step)

    def test_user_input_description(self):
        step = {"type": "app", "action": "type", "params": {"text": "{user_input}"}}
        desc = self._desc(step)
        self.assertIn("whatever you say", desc)
        self.assertNotIn("{user_input}", desc)

    def test_date_description(self):
        step = {"type": "app", "action": "type", "params": {"text": "{date}"}}
        desc = self._desc(step)
        self.assertIn("date", desc)
        self.assertNotIn("{date}", desc)

    def test_time_description(self):
        step = {"type": "app", "action": "type", "params": {"text": "{time}"}}
        desc = self._desc(step)
        self.assertIn("time", desc)
        self.assertNotIn("{time}", desc)

    def test_clipboard_description(self):
        step = {"type": "app", "action": "type", "params": {"text": "{clipboard}"}}
        desc = self._desc(step)
        self.assertIn("clipboard", desc)
        self.assertNotIn("{clipboard}", desc)

    def test_literal_text_description(self):
        step = {"type": "app", "action": "type", "params": {"text": "hello"}}
        desc = self._desc(step)
        self.assertEqual(desc, "type 'hello'")

    def test_named_slot_description(self):
        step = {"type": "app", "action": "type", "params": {"text": "{contact}"}}
        desc = self._desc(step)
        self.assertEqual(desc, "type '{contact}'")

    def test_user_input_with_window(self):
        step = {"type": "app", "action": "type",
                "params": {"text": "{user_input}", "window": "Notepad"}}
        desc = self._desc(step)
        self.assertIn("whatever you say", desc)
        self.assertIn("Notepad", desc)


# ─────────────────────────────────────────────────────────────────────────────
# Full teaching flow with NL variables
# ─────────────────────────────────────────────────────────────────────────────

class TestTeachingFlowWithVariables(unittest.TestCase):

    def setUp(self):
        _fresh_db()
        actions.teaching_session.clear()

    def tearDown(self):
        actions.teaching_session.clear()

    def _start(self, name_seed="search on youtube"):
        actions.start_teaching_session(name_seed)

    def _run(self, text):
        return run_async(actions.handle_pending_teaching(text))

    def test_teach_with_nl_user_input(self):
        self._start()
        self._run("open chrome")
        self._run("go to youtube.com")
        self._run("click search")
        self._run("type whatever I want")
        self._run("press enter")
        self._run("done")

        steps = actions.teaching_session.payload["steps"]
        type_step = next(s for s in steps if s.get("action") == "type")
        self.assertEqual(type_step["params"]["text"], "{user_input}")

    def test_teach_with_nl_date(self):
        self._start("write todays date")
        self._run("open notepad")
        self._run("type today's date")
        self._run("done")

        steps = actions.teaching_session.payload["steps"]
        type_step = next(s for s in steps if s.get("action") == "type")
        self.assertEqual(type_step["params"]["text"], "{date}")

    def test_teach_confirm_readback_shows_friendly_desc(self):
        self._start("quick note")
        self._run("open notepad")
        resp = self._run("type whatever I say")
        self.assertIn("whatever you say", resp)

    def test_full_flow_saves_with_variable(self):
        self._start("youtube search")
        self._run("open chrome")
        self._run("type whatever I search")
        self._run("press enter")
        self._run("done")
        self._run("yes")
        self._run("yes")

        self.assertFalse(actions.teaching_session.active)
        proc = ps.get_procedure("youtube search")
        self.assertIsNotNone(proc)
        type_step = next(s for s in proc["steps"] if s.get("action") == "type")
        self.assertEqual(type_step["params"]["text"], "{user_input}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
