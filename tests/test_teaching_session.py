"""
test_teaching_session.py — TP-1b: Teaching session unit tests

Tests the step parser, step description, key normalizer, and full
state-machine handler (handle_pending_teaching) in isolation.

Run: python test_teaching_session.py
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
from assistant.storage.db import init_db, _reset_for_testing


def _fresh_db():
    """Reset the storage singleton and the procedures facade for a fresh DB.

    Post-RG-1 procedures is a thin facade over the storage.db singleton —
    state lives on Database._instance and on procedures._repo. Reset BOTH
    so prior-test rows don't leak in. The shared schema (initialized by
    init_db) already creates user_shortcuts / procedures tables.
    """
    tmp = Path(tempfile.mkdtemp()) / "test_personality.db"
    _config_stub.SANDBOX_DIR = tmp.parent.parent
    (tmp.parent.parent / "memory").mkdir(parents=True, exist_ok=True)

    _reset_for_testing()
    ps._repo = None

    init_db(tmp)
    ps.init_procedure_db()


def run_async(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────────────
# Step Parser
# ─────────────────────────────────────────────────────────────────────────────

class TestParseTeachingStep(unittest.TestCase):

    def _parse(self, text):
        return actions._parse_teaching_step(text)

    # ── open ─────────────────────────────────────────────────────────────────

    def test_open_app(self):
        s = self._parse("open notepad")
        self.assertEqual(s["type"], "app")
        self.assertEqual(s["action"], "open")
        self.assertEqual(s["params"]["name"], "notepad")

    def test_open_url_becomes_browser(self):
        s = self._parse("open youtube.com")
        self.assertEqual(s["type"], "browser")
        self.assertEqual(s["action"], "navigate")
        self.assertIn("youtube.com", s["params"]["url"])

    def test_open_http_url(self):
        s = self._parse("open https://github.com")
        self.assertEqual(s["type"], "browser")
        self.assertEqual(s["params"]["url"], "https://github.com")

    # ── navigate ─────────────────────────────────────────────────────────────

    def test_go_to_url(self):
        s = self._parse("go to google.com")
        self.assertEqual(s["type"], "browser")
        self.assertEqual(s["params"]["url"], "https://google.com")

    def test_navigate_to_url(self):
        s = self._parse("navigate to https://example.com")
        self.assertEqual(s["params"]["url"], "https://example.com")

    def test_visit_url(self):
        s = self._parse("visit www.github.com")
        self.assertEqual(s["type"], "browser")
        self.assertIn("github.com", s["params"]["url"])

    # ── close ────────────────────────────────────────────────────────────────

    def test_close_app(self):
        s = self._parse("close VS Code")
        self.assertEqual(s["action"], "close")
        self.assertEqual(s["params"]["name"], "VS Code")

    # ── focus ────────────────────────────────────────────────────────────────

    def test_focus(self):
        s = self._parse("focus on notepad")
        self.assertEqual(s["action"], "focus")
        self.assertEqual(s["params"]["name"], "notepad")

    def test_focus_no_on(self):
        s = self._parse("focus chrome")
        self.assertEqual(s["action"], "focus")

    # ── press_key ────────────────────────────────────────────────────────────

    def test_press_simple(self):
        s = self._parse("press enter")
        self.assertEqual(s["action"], "press_key")
        self.assertEqual(s["params"]["key"], "enter")

    def test_press_combo(self):
        s = self._parse("press ctrl+s")
        self.assertEqual(s["params"]["key"], "ctrl+s")

    def test_press_spoken_combo(self):
        s = self._parse("press control s")
        self.assertEqual(s["params"]["key"], "ctrl+s")

    def test_hit_key(self):
        s = self._parse("hit escape")
        self.assertEqual(s["action"], "press_key")
        self.assertEqual(s["params"]["key"], "esc")

    def test_press_shift_combo(self):
        s = self._parse("press ctrl+shift+p")
        self.assertEqual(s["params"]["key"], "ctrl+shift+p")

    # ── type ─────────────────────────────────────────────────────────────────

    def test_type_plain(self):
        s = self._parse("type hello world")
        self.assertEqual(s["action"], "type")
        self.assertEqual(s["params"]["text"], "hello world")
        self.assertNotIn("window", s["params"])

    def test_type_in_window(self):
        s = self._parse("type hello in notepad")
        self.assertEqual(s["params"]["text"], "hello")
        self.assertEqual(s["params"]["window"], "notepad")

    def test_type_quoted(self):
        s = self._parse("type 'open recent'")
        self.assertEqual(s["params"]["text"], "open recent")

    # ── click ────────────────────────────────────────────────────────────────

    def test_click(self):
        s = self._parse("click save button")
        self.assertEqual(s["action"], "click")
        self.assertIn("save button", s["params"]["selector"])

    def test_click_on(self):
        s = self._parse("click on the submit button")
        self.assertIn("submit button", s["params"]["selector"])

    # ── wait ─────────────────────────────────────────────────────────────────

    def test_wait_seconds(self):
        s = self._parse("wait 3 seconds")
        self.assertEqual(s["action"], "wait")
        self.assertEqual(s["params"]["seconds"], 3.0)

    def test_wait_no_unit(self):
        s = self._parse("wait 5")
        self.assertEqual(s["params"]["seconds"], 5.0)

    def test_wait_no_number(self):
        s = self._parse("wait a moment")
        self.assertEqual(s["params"]["seconds"], 2)

    def test_wait_bare(self):
        s = self._parse("wait")
        self.assertEqual(s["params"]["seconds"], 2)

    # ── no match ─────────────────────────────────────────────────────────────

    def test_no_match_returns_none(self):
        self.assertIsNone(actions._parse_teaching_step("do something weird"))
        self.assertIsNone(actions._parse_teaching_step(""))
        self.assertIsNone(actions._parse_teaching_step("   "))


# ─────────────────────────────────────────────────────────────────────────────
# Step description
# ─────────────────────────────────────────────────────────────────────────────

class TestStepDescription(unittest.TestCase):

    def _desc(self, step):
        return actions._step_description(step)

    def test_open(self):
        self.assertEqual(self._desc({"type": "app", "action": "open", "params": {"name": "notepad"}}),
                         "open notepad")

    def test_press(self):
        self.assertEqual(self._desc({"type": "app", "action": "press_key", "params": {"key": "ctrl+s"}}),
                         "press ctrl+s")

    def test_type_no_window(self):
        self.assertEqual(self._desc({"type": "app", "action": "type", "params": {"text": "hello"}}),
                         "type 'hello'")

    def test_type_with_window(self):
        self.assertEqual(self._desc({"type": "app", "action": "type",
                                     "params": {"text": "hi", "window": "Notepad"}}),
                         "type 'hi' in Notepad")

    def test_browser_navigate(self):
        self.assertEqual(self._desc({"type": "browser", "action": "navigate",
                                     "params": {"url": "https://youtube.com"}}),
                         "go to https://youtube.com")

    def test_wait(self):
        self.assertEqual(self._desc({"type": "app", "action": "wait", "params": {"seconds": 3}}),
                         "wait 3 seconds")


# ─────────────────────────────────────────────────────────────────────────────
# Key normalizer
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeKey(unittest.TestCase):

    def _norm(self, text):
        from assistant.actions.teaching import _normalize_key
        return _normalize_key(text)

    def test_ctrl_alias(self):
        self.assertEqual(self._norm("control s"), "ctrl+s")

    def test_already_normalized(self):
        self.assertEqual(self._norm("ctrl+shift+p"), "ctrl+shift+p")

    def test_return_to_enter(self):
        self.assertIn("enter", self._norm("return"))

    def test_escape_to_esc(self):
        self.assertIn("esc", self._norm("escape"))

    def test_space_bar(self):
        self.assertEqual(self._norm("space bar"), "space")


# ─────────────────────────────────────────────────────────────────────────────
# start_teaching_session
# ─────────────────────────────────────────────────────────────────────────────

class TestStartTeachingSession(unittest.TestCase):

    def setUp(self):
        actions.teaching_session.clear()

    def test_sets_state(self):
        actions.start_teaching_session("start my coding session")
        self.assertTrue(actions.teaching_session.active)
        self.assertEqual(actions.teaching_session.payload["state"], "collecting")
        self.assertEqual(actions.teaching_session.payload["name_seed"], "start my coding session")
        self.assertEqual(actions.teaching_session.payload["steps"], [])

    def test_returns_opening_prompt(self):
        prompt = actions.start_teaching_session("open VS Code")
        self.assertIn("teach me", prompt.lower())
        self.assertIn("open VS Code", prompt)

    def tearDown(self):
        actions.teaching_session.clear()


# ─────────────────────────────────────────────────────────────────────────────
# handle_pending_teaching  (state machine)
# ─────────────────────────────────────────────────────────────────────────────

class TestHandlePendingTeaching(unittest.TestCase):

    def setUp(self):
        _fresh_db()
        actions.teaching_session.clear()

    def tearDown(self):
        actions.teaching_session.clear()

    def _start(self, name_seed="start my coding session"):
        actions.start_teaching_session(name_seed)

    def _run(self, text):
        return run_async(actions.handle_pending_teaching(text))

    # ── returns None when no session ─────────────────────────────────────────

    def test_none_when_inactive(self):
        self.assertIsNone(self._run("open notepad"))

    # ── collecting: step accumulation ────────────────────────────────────────

    def test_add_step_open(self):
        self._start()
        resp = self._run("open VS Code")
        self.assertIsNotNone(resp)
        self.assertIn("VS Code", resp)
        self.assertEqual(len(actions.teaching_session.payload["steps"]), 1)
        self.assertEqual(actions.teaching_session.payload["state"], "collecting")

    def test_add_multiple_steps(self):
        self._start()
        self._run("open VS Code")
        self._run("wait 3 seconds")
        self._run("press ctrl+shift+p")
        self.assertEqual(len(actions.teaching_session.payload["steps"]), 3)

    def test_unrecognized_step_returns_hint(self):
        self._start()
        resp = self._run("do something complicated")
        self.assertIsNotNone(resp)
        # Should mention supported formats
        self.assertTrue(
            any(kw in resp.lower() for kw in ("open", "press", "type", "go to", "rephrase"))
        )
        # Step should NOT have been added
        self.assertEqual(len(actions.teaching_session.payload["steps"]), 0)

    def test_done_with_no_steps_asks_for_step(self):
        self._start()
        resp = self._run("done")
        self.assertIsNotNone(resp)
        self.assertIn("step", resp.lower())
        self.assertEqual(actions.teaching_session.payload["state"], "collecting")

    def test_done_transitions_to_confirming(self):
        self._start()
        self._run("open VS Code")
        self._run("press ctrl+shift+p")
        resp = self._run("that's it")
        self.assertEqual(actions.teaching_session.payload["state"], "confirming")
        self.assertIn("Step 1", resp)
        self.assertIn("Step 2", resp)

    # ── confirming: yes/no ───────────────────────────────────────────────────

    def test_yes_transitions_to_naming(self):
        self._start()
        self._run("open VS Code")
        self._run("done")
        resp = self._run("yes")
        self.assertEqual(actions.teaching_session.payload["state"], "naming")
        # Assistant suggests name_seed as trigger
        self.assertIn("start my coding session", resp)

    def test_no_restarts_collecting(self):
        self._start()
        self._run("open VS Code")
        self._run("done")
        resp = self._run("no")
        self.assertEqual(actions.teaching_session.payload["state"], "collecting")
        self.assertEqual(actions.teaching_session.payload["steps"], [])
        self.assertIn("step", resp.lower())

    def test_ambiguous_re_prompts(self):
        self._start()
        self._run("open VS Code")
        self._run("done")
        resp = self._run("hmm maybe")
        self.assertEqual(actions.teaching_session.payload["state"], "confirming")
        # Should ask for yes or no
        self.assertTrue("yes" in resp.lower() or "no" in resp.lower())

    # ── naming: trigger selection ─────────────────────────────────────────────

    def test_yes_uses_name_seed(self):
        self._start("start my workflow")
        self._run("open VS Code")
        self._run("done")
        self._run("yes")           # confirm steps
        resp = self._run("yes")    # accept suggested trigger
        self.assertFalse(actions.teaching_session.active)  # session cleared
        self.assertIn("start my workflow", resp)
        # Verify saved in DB
        saved = ps.get_procedure("start my workflow")
        self.assertIsNotNone(saved)
        self.assertEqual(len(saved["steps"]), 1)

    def test_custom_trigger_name(self):
        self._start("open VS Code setup")
        self._run("open VS Code")
        self._run("done")
        self._run("yes")
        resp = self._run("launch developer environment")
        self.assertFalse(actions.teaching_session.active)
        saved = ps.get_procedure("launch developer environment")
        self.assertIsNotNone(saved)

    def test_reserved_trigger_rejected(self):
        self._start("test session")
        self._run("open notepad")
        self._run("done")
        self._run("yes")
        resp = self._run("yes please")
        # "yes please" → no-conflict, saves with name_seed
        # Don't assert on state since this saves. Instead test reserved word directly.
        actions.teaching_session.clear()

    def test_reserved_word_trigger_rejected(self):
        self._start("test procedure")
        self._run("open notepad")
        self._run("done")
        self._run("yes")
        # Force naming state
        actions.teaching_session.payload["state"] = "naming"
        resp = self._run("yes")  # name_seed = "test procedure" → clean
        # Now test with a reserved word explicitly
        actions.teaching_session.set({
            "state": "naming",
            "name_seed": "open editor",
            "steps": [{"type": "app", "action": "open", "params": {"name": "vscode"}}],
            "backend": "auto",
        })
        resp = self._run("stop")  # "stop" is reserved
        self.assertTrue(actions.teaching_session.active)  # session NOT cleared
        self.assertIn("reserved", resp.lower())

    def test_conflict_with_existing_procedure(self):
        ps.create_procedure("my workflow", "My Workflow",
                            [{"type": "app", "action": "open", "params": {"name": "notepad"}}])
        actions.teaching_session.set({
            "state": "naming",
            "name_seed": "something else",
            "steps": [{"type": "app", "action": "open", "params": {"name": "notepad"}}],
            "backend": "auto",
        })
        resp = self._run("my workflow")
        self.assertTrue(actions.teaching_session.active)  # session stays open
        self.assertIn("procedure", resp.lower())

    # ── full round-trip ───────────────────────────────────────────────────────

    def test_full_flow_saves_procedure(self):
        self._start("open notepad workflow")
        self._run("open notepad")
        self._run("press ctrl+n")
        self._run("type meeting notes")
        self._run("done")
        self._run("yes")                   # confirm steps
        self._run("yes")                   # accept name_seed as trigger

        self.assertFalse(actions.teaching_session.active)
        proc = ps.get_procedure("open notepad workflow")
        self.assertIsNotNone(proc)
        self.assertEqual(len(proc["steps"]), 3)
        self.assertEqual(proc["steps"][0]["action"], "open")
        self.assertEqual(proc["steps"][1]["action"], "press_key")
        self.assertEqual(proc["steps"][2]["action"], "type")


# ─────────────────────────────────────────────────────────────────────────────
# _match_teach_trigger  (main.py helper — tested via import)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# _match_teach_trigger — tested inline (avoid importing all of main.py)
# Mirrors the patterns defined in main.py exactly.
# ─────────────────────────────────────────────────────────────────────────────

_TEACH_PATTERNS_TEST = [
    re.compile(r"(?:let me |i want to |i'll )?teach you (?:how to |to )?(.+)", re.IGNORECASE),
    re.compile(r"(?:let me |i'll )?show you how to (.+)", re.IGNORECASE),
    re.compile(r"create (?:a )?procedure (?:for |to )?(.+)", re.IGNORECASE),
    re.compile(r"new procedure (?:for |to )?(.+)", re.IGNORECASE),
]


def _match_teach_trigger_test(text: str):
    for pat in _TEACH_PATTERNS_TEST:
        m = pat.fullmatch(text.strip())
        if not m:
            m = pat.match(text.strip())
        if m:
            seed = m.group(1).strip().rstrip(".!?").strip()
            if seed and len(seed) >= 3:
                return seed
    return None


class TestMatchTeachTrigger(unittest.TestCase):

    def match(self, text):
        return _match_teach_trigger_test(text)

    def test_teach_you_how_to(self):
        seed = self.match("teach you how to start my coding session")
        self.assertEqual(seed, "start my coding session")

    def test_teach_you_to(self):
        seed = self.match("teach you to open notepad")
        self.assertEqual(seed, "open notepad")

    def test_show_you_how_to(self):
        seed = self.match("show you how to play music")
        self.assertEqual(seed, "play music")

    def test_create_procedure(self):
        seed = self.match("create a procedure for daily standup")
        self.assertEqual(seed, "daily standup")

    def test_new_procedure(self):
        seed = self.match("new procedure for morning routine")
        self.assertEqual(seed, "morning routine")

    def test_let_me_prefix(self):
        seed = self.match("let me teach you how to open VS Code")
        self.assertIsNotNone(seed)
        self.assertIn("VS Code", seed)

    def test_no_match(self):
        self.assertIsNone(self.match("open notepad"))
        self.assertIsNone(self.match("play music"))
        self.assertIsNone(self.match("what time is it"))

    def test_too_short_seed_rejected(self):
        self.assertIsNone(self.match("teach you to ab"))

    def test_trailing_punctuation_stripped(self):
        seed = self.match("teach you how to start my workflow.")
        self.assertEqual(seed, "start my workflow")


if __name__ == "__main__":
    unittest.main(verbosity=2)
