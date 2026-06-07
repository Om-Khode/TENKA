"""
test_plan_and_execute.py — PE-1: Plan-and-Execute TODO tracking in computer_agent.

Covers:
  - _TaskState TODO list state mgmt: set_initial_todos, add_todo, mark_todo_done,
    all_todos_done, todo_progress_str, reset
  - 15-item cap enforced by both set_initial_todos and add_todo
  - Dedupe in add_todo (case-insensitive)
  - _parse_todo_list: bare list, fenced JSON, object wrapper, prose-wrapped, junk
  - _parse_todo_update: normal, fenced, missing keys, bad-id filtering
  - _generate_initial_todos: vision OK, vision returns junk, LLM unavailable
  - _update_todos_after_batch: short-circuits when no TODOs, marks+adds on
    success, fail-open on parse error / LLM unavailable
  - Disagreement path is not asserted here (it's a control-flow concern in
    the orchestrator) — manual / live test covers that. Unit tests focus on
    the deterministic state and parsing helpers.

Run: python test_plan_and_execute.py
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.vision as ca


# ─── Helpers ────────────────────────────────────────────────────────────────


def _run(coro):
    return asyncio.run(coro)


def _install_llm_stub(
    *,
    vision_response="[\"a\", \"b\", \"c\"]",
    text_response='{"completed":[],"new":[]}',
):
    """Install a fake assistant.llm module with controllable async returns.

    ``get_vision_response`` now returns ``LLMResult`` objects so each mock
    value is wrapped in ``SimpleNamespace(text=...)`` so callers can do
    ``.text`` on the result.  ``get_llm_response`` is not called directly
    by the code-under-test (it goes through contracts), so its mock value
    is left as a plain string for assertion convenience.
    """
    llm_mod = types.ModuleType("assistant.llm")
    if isinstance(vision_response, Exception):
        llm_mod.get_vision_response = AsyncMock(side_effect=vision_response)
    elif isinstance(vision_response, list):
        llm_mod.get_vision_response = AsyncMock(
            side_effect=[SimpleNamespace(text=v) for v in vision_response]
        )
    else:
        llm_mod.get_vision_response = AsyncMock(
            return_value=SimpleNamespace(text=vision_response)
        )
    llm_mod.get_llm_response = AsyncMock(return_value=text_response)
    sys.modules["assistant.llm"] = llm_mod
    return llm_mod


# ─── _TaskState TODO helpers ────────────────────────────────────────────────


class TestTaskStateTodos(unittest.TestCase):
    def setUp(self):
        self.state = ca._TaskState()

    def test_initial_todos_assigns_sequential_ids(self):
        n = self.state.set_initial_todos(["a", "b", "c"])
        self.assertEqual(n, 3)
        self.assertEqual([t["id"] for t in self.state.todo_list], [1, 2, 3])
        self.assertEqual([t["task"] for t in self.state.todo_list], ["a", "b", "c"])
        self.assertTrue(all(t["done"] is False for t in self.state.todo_list))

    def test_initial_todos_filters_empty_and_non_strings(self):
        self.state.set_initial_todos(["", "  ", None, 0, "real", False])
        self.assertEqual([t["task"] for t in self.state.todo_list], ["real"])

    def test_initial_todos_caps_at_max(self):
        items = [f"task {i}" for i in range(50)]
        n = self.state.set_initial_todos(items)
        self.assertEqual(n, ca._TaskState.TODO_MAX)
        self.assertEqual(len(self.state.todo_list), ca._TaskState.TODO_MAX)

    def test_initial_todos_resets_state_on_recall(self):
        self.state.set_initial_todos(["x", "y"])
        self.state.mark_todo_done(1)
        self.state.set_initial_todos(["fresh"])
        self.assertEqual(len(self.state.todo_list), 1)
        self.assertEqual(self.state.todo_list[0]["id"], 1)
        self.assertFalse(self.state.todo_list[0]["done"])

    def test_add_todo_returns_new_id(self):
        self.state.set_initial_todos(["a", "b"])
        new_id = self.state.add_todo("c")
        self.assertEqual(new_id, 3)
        self.assertEqual(self.state.todo_list[-1]["task"], "c")

    def test_add_todo_dedupes_case_insensitive(self):
        self.state.set_initial_todos(["Type Email"])
        self.assertIsNone(self.state.add_todo("type email"))
        self.assertIsNone(self.state.add_todo("  Type Email  "))
        self.assertEqual(len(self.state.todo_list), 1)

    def test_add_todo_skips_empty(self):
        self.state.set_initial_todos(["a"])
        self.assertIsNone(self.state.add_todo(""))
        self.assertIsNone(self.state.add_todo("   "))
        self.assertIsNone(self.state.add_todo(None))
        self.assertEqual(len(self.state.todo_list), 1)

    def test_add_todo_respects_cap(self):
        self.state.set_initial_todos([f"t{i}" for i in range(ca._TaskState.TODO_MAX)])
        self.assertIsNone(self.state.add_todo("overflow"))
        self.assertEqual(len(self.state.todo_list), ca._TaskState.TODO_MAX)

    def test_mark_todo_done_returns_true_on_match(self):
        self.state.set_initial_todos(["a", "b"])
        self.assertTrue(self.state.mark_todo_done(2))
        self.assertTrue(self.state.todo_list[1]["done"])

    def test_mark_todo_done_returns_false_for_unknown_id(self):
        self.state.set_initial_todos(["a"])
        self.assertFalse(self.state.mark_todo_done(99))

    def test_mark_todo_done_idempotent(self):
        self.state.set_initial_todos(["a"])
        self.assertTrue(self.state.mark_todo_done(1))
        self.assertTrue(self.state.mark_todo_done(1))  # already done — still True

    def test_all_todos_done_empty_list_is_false(self):
        self.assertFalse(self.state.all_todos_done())

    def test_all_todos_done_partial_is_false(self):
        self.state.set_initial_todos(["a", "b"])
        self.state.mark_todo_done(1)
        self.assertFalse(self.state.all_todos_done())

    def test_all_todos_done_all_done_is_true(self):
        self.state.set_initial_todos(["a", "b"])
        self.state.mark_todo_done(1)
        self.state.mark_todo_done(2)
        self.assertTrue(self.state.all_todos_done())

    def test_todo_progress_str_empty_is_empty(self):
        self.assertEqual(self.state.todo_progress_str(), "")

    def test_todo_progress_str_marks_first_unchecked_as_next(self):
        self.state.set_initial_todos(["a", "b", "c"])
        self.state.mark_todo_done(1)
        out = self.state.todo_progress_str()
        self.assertIn("1 of 3 done", out)
        self.assertIn("✓ a", out)
        self.assertIn("✗ b", out)
        self.assertIn("← NEXT", out)
        # Only the first ✗ gets the NEXT marker
        self.assertEqual(out.count("← NEXT"), 1)

    def test_reset_clears_todos(self):
        self.state.set_initial_todos(["a", "b"])
        self.state.mark_todo_done(1)
        self.state.reset()
        self.assertEqual(self.state.todo_list, [])
        # Next id resets so a fresh task starts at 1
        self.assertEqual(self.state._next_todo_id, 1)


# ─── Parsing helpers ────────────────────────────────────────────────────────


class TestParseTodoList(unittest.TestCase):
    def test_bare_list(self):
        self.assertEqual(ca._parse_todo_list('["a","b"]'), ["a", "b"])

    def test_fenced_list(self):
        self.assertEqual(ca._parse_todo_list('```json\n["x","y"]\n```'), ["x", "y"])

    def test_object_wrapper_todos_key(self):
        self.assertEqual(ca._parse_todo_list('{"todos":["p","q"]}'), ["p", "q"])

    def test_object_wrapper_items_key(self):
        self.assertEqual(ca._parse_todo_list('{"items":["m","n"]}'), ["m", "n"])

    def test_prose_wrapped_list(self):
        self.assertEqual(
            ca._parse_todo_list('Sure! Here is the list: ["a","b"] — done.'),
            ["a", "b"],
        )

    def test_junk_returns_empty(self):
        self.assertEqual(ca._parse_todo_list("not json"), [])

    def test_empty_string_returns_empty(self):
        self.assertEqual(ca._parse_todo_list(""), [])

    def test_non_string_returns_empty(self):
        self.assertEqual(ca._parse_todo_list(None), [])

    def test_filters_empty_strings(self):
        self.assertEqual(ca._parse_todo_list('["a","","  ","b"]'), ["a", "b"])


class TestParseTodoUpdate(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(
            ca._parse_todo_update('{"completed":[1,3],"new":["foo"]}'),
            ([1, 3], ["foo"]),
        )

    def test_fenced(self):
        self.assertEqual(
            ca._parse_todo_update('```\n{"completed":[2],"new":[]}\n```'),
            ([2], []),
        )

    def test_missing_keys(self):
        self.assertEqual(ca._parse_todo_update("{}"), ([], []))

    def test_junk_returns_empty_tuple(self):
        self.assertEqual(ca._parse_todo_update("???"), ([], []))

    def test_bad_ids_filtered(self):
        # "abc" is not int-castable — drop it; 4 is kept
        completed, new = ca._parse_todo_update(
            '{"completed":["abc",4],"new":["x",null,1]}'
        )
        self.assertEqual(completed, [4])
        # null is filtered; "x" and 1 (stringified) kept
        self.assertEqual(new, ["x", "1"])

    def test_empty_string(self):
        self.assertEqual(ca._parse_todo_update(""), ([], []))


# ─── Async helpers (LLM-stubbed) ────────────────────────────────────────────


class TestGenerateInitialTodos(unittest.TestCase):
    def test_vision_success_returns_list(self):
        _install_llm_stub(vision_response='["task A","task B"]')
        out = _run(ca._generate_initial_todos("goal", "fake-b64"))
        self.assertEqual(out, ["task A", "task B"])

    def test_vision_returns_junk_yields_empty_list(self):
        _install_llm_stub(vision_response="not parseable at all")
        out = _run(ca._generate_initial_todos("goal", "fake-b64"))
        self.assertEqual(out, [])

    def test_llm_unavailable_yields_empty_list(self):
        _install_llm_stub(vision_response="__LLM_UNAVAILABLE__")
        out = _run(ca._generate_initial_todos("goal", "fake-b64"))
        self.assertEqual(out, [])

    def test_no_screenshot_falls_back_to_text_llm(self):
        llm_stub = _install_llm_stub(text_response='["from text"]')
        out = _run(ca._generate_initial_todos("goal", None))
        self.assertEqual(out, ["from text"])
        llm_stub.get_llm_response.assert_awaited_once()
        llm_stub.get_vision_response.assert_not_called()

    def test_llm_crash_yields_empty_list_fail_open(self):
        llm_mod = types.ModuleType("assistant.llm")
        llm_mod.get_vision_response = AsyncMock(side_effect=RuntimeError("boom"))
        llm_mod.get_llm_response = AsyncMock(return_value="")
        sys.modules["assistant.llm"] = llm_mod
        out = _run(ca._generate_initial_todos("goal", "fake-b64"))
        self.assertEqual(out, [])


class TestUpdateTodosAfterBatch(unittest.TestCase):
    def setUp(self):
        ca._task_state.reset()

    def tearDown(self):
        ca._task_state.reset()

    def test_short_circuits_when_no_todos(self):
        llm_stub = _install_llm_stub(text_response='{"completed":[1],"new":[]}')
        marked, added = _run(
            ca._update_todos_after_batch(
                actions=[{"type": "mouse_click"}], results=["Clicked"], plan_thinking=""
            )
        )
        self.assertEqual((marked, added), (0, 0))
        llm_stub.get_llm_response.assert_not_called()

    def test_marks_and_adds_on_success(self):
        ca._task_state.set_initial_todos(["a", "b", "c"])
        _install_llm_stub(
            text_response='{"completed":[1,2],"new":["new task"]}'
        )
        marked, added = _run(
            ca._update_todos_after_batch(
                actions=[{"type": "keyboard_type", "text": "x"}],
                results=["Typed x"],
                plan_thinking="filling form",
            )
        )
        self.assertEqual((marked, added), (2, 1))
        self.assertTrue(ca._task_state.todo_list[0]["done"])
        self.assertTrue(ca._task_state.todo_list[1]["done"])
        self.assertFalse(ca._task_state.todo_list[2]["done"])
        self.assertEqual(ca._task_state.todo_list[-1]["task"], "new task")

    def test_unknown_id_in_completed_silently_skipped(self):
        ca._task_state.set_initial_todos(["a", "b"])
        _install_llm_stub(text_response='{"completed":[99],"new":[]}')
        marked, added = _run(
            ca._update_todos_after_batch(
                actions=[{"type": "x"}], results=["ok"], plan_thinking=""
            )
        )
        self.assertEqual((marked, added), (0, 0))
        self.assertFalse(any(t["done"] for t in ca._task_state.todo_list))

    def test_parse_failure_fail_open(self):
        ca._task_state.set_initial_todos(["a"])
        _install_llm_stub(text_response="garbage out")
        marked, added = _run(
            ca._update_todos_after_batch(
                actions=[{"type": "x"}], results=["ok"], plan_thinking=""
            )
        )
        self.assertEqual((marked, added), (0, 0))
        self.assertFalse(ca._task_state.todo_list[0]["done"])

    def test_llm_unavailable_fail_open(self):
        ca._task_state.set_initial_todos(["a"])
        _install_llm_stub(text_response="__LLM_UNAVAILABLE__")
        marked, added = _run(
            ca._update_todos_after_batch(
                actions=[{"type": "x"}], results=["ok"], plan_thinking=""
            )
        )
        self.assertEqual((marked, added), (0, 0))

    def test_llm_crash_fail_open(self):
        ca._task_state.set_initial_todos(["a"])
        llm_mod = types.ModuleType("assistant.llm")
        llm_mod.get_llm_response = AsyncMock(side_effect=RuntimeError("rate limit"))
        llm_mod.get_vision_response = AsyncMock(return_value=SimpleNamespace(text=""))
        sys.modules["assistant.llm"] = llm_mod
        marked, added = _run(
            ca._update_todos_after_batch(
                actions=[{"type": "x"}], results=["ok"], plan_thinking=""
            )
        )
        self.assertEqual((marked, added), (0, 0))

    def test_empty_actions_short_circuits(self):
        ca._task_state.set_initial_todos(["a"])
        llm_stub = _install_llm_stub(text_response='{"completed":[1],"new":[]}')
        marked, added = _run(
            ca._update_todos_after_batch(actions=[], results=[], plan_thinking="")
        )
        self.assertEqual((marked, added), (0, 0))
        llm_stub.get_llm_response.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
