"""test_vision_outcome_honesty.py — TENKA-v2 §17.P13, loop 3.

The vision agent is the loop that did **not** migrate its internals, and the
reason is measured rather than asserted: it carries two lists, not one.
`actions` are the executable units (planned per loop, dispatched by
`execute_all_actions`); `todo_list` is a completeness checklist derived from
the goal by a separate vision call and matched to actions after the fact. §28
read `todo_list` as "a `TaskStep` with a verification state machine", but
`TaskStep` maps to `actions`, and `todo_list` has no counterpart in the
contracts at all -- it is a verification structure. Typing it as a step would
be wrong; migrating `actions` means rewriting the 466-line dispatch table,
which is the rewrite §17.P13 forbids.

What this file pins is the boundary, which is where the loop was actually
dishonest:

  1. It already discloses uncertainty in speech (`_append_abandoned_suffix`,
     shipped before §11 specified it) and stored `action_outcome = "success"`
     for the same turn.
  2. Two of its three give-up exits told telemetry nothing, so a task that
     produced no actions at all was stored as one that worked.
  3. A check on `"ABORTED" in result` reported the user's own abort to them --
     for a string no abort ever produces.

All-stub. Imports the vision package, which proxies to `agent`; nothing here
reaches pyautogui, the screen or a model.

Run: py -3.11 -m pytest tests/test_vision_outcome_honesty.py -q
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.vision as ca
import assistant.telemetry as telemetry

_AGENT_PY = Path(__file__).parent.parent / "assistant" / "automation" / "vision" / "agent.py"
_MAIN_PY = Path(__file__).parent.parent / "assistant" / "main.py"


class _Tracker:
    """The two fields the marking functions touch. A stand-in rather than a
    real `TurnTracker` so these tests cannot be made to pass by a change in
    how a tracker is constructed."""

    def __init__(self, outcome: str = "skipped") -> None:
        self.action_outcome = outcome
        self.error_class = ""


class _tracker_ctx:
    def __init__(self, outcome: str = "skipped") -> None:
        self.tracker = _Tracker(outcome)

    def __enter__(self) -> _Tracker:
        self._token = telemetry.set_current_tracker(self.tracker)
        return self.tracker

    def __exit__(self, *exc) -> None:
        telemetry.reset_current_tracker(self._token)


# ─── The new channel ─────────────────────────────────────────────────────────


class TestMarkActionUncertain(unittest.TestCase):
    def test_it_records_uncertain_not_failure(self):
        with _tracker_ctx() as t:
            telemetry.mark_action_uncertain("VisionConfirmAbandoned", "two fields")
        self.assertEqual(t.action_outcome, "uncertain")
        self.assertEqual(t.error_class, "VisionConfirmAbandoned")

    def test_a_failure_overrides_an_earlier_uncertain(self):
        # Positive evidence against outranks no evidence either way --
        # `core/verdict.roll_up`'s precedence, at the turn level.
        with _tracker_ctx() as t:
            telemetry.mark_action_uncertain("Unconfirmed", "")
            telemetry.mark_action_failure("RealFailure", "")
        self.assertEqual(t.action_outcome, "failure")
        self.assertEqual(t.error_class, "RealFailure")

    def test_an_uncertain_never_downgrades_a_failure(self):
        with _tracker_ctx() as t:
            telemetry.mark_action_failure("RealFailure", "")
            telemetry.mark_action_uncertain("Unconfirmed", "")
        self.assertEqual(t.action_outcome, "failure")
        self.assertEqual(t.error_class, "RealFailure")

    def test_the_first_uncertain_keeps_its_error_class(self):
        with _tracker_ctx() as t:
            telemetry.mark_action_uncertain("First", "")
            telemetry.mark_action_uncertain("Second", "")
        self.assertEqual(t.error_class, "First")

    def test_no_tracker_is_a_no_op_not_a_crash(self):
        # Background tasks and the scheduler run with no tracker installed.
        telemetry.mark_action_uncertain("Nobody", "listening")

    def test_handler_reported_covers_both_channels(self):
        # The set main.py reads. If a channel is added to telemetry and left
        # out of here, main.py overwrites it with "success" on the next line.
        self.assertEqual(telemetry.HANDLER_REPORTED, frozenset({"failure", "uncertain"}))


class TestMainDoesNotClobberHandlerOutcomes(unittest.TestCase):
    """Structural, because the failure is silent: the handler sets the field
    correctly and the next line overwrites it, so every unit test of the
    handler passes and every stored row is wrong."""

    def test_no_post_dispatch_assignment_compares_against_a_bare_literal(self):
        """Only the stamps that *read* `action_outcome` before overwriting it.

        The first draft of this collected every `if ...: action_outcome =
        "success"` and failed on eight pre-dispatch branches, which is a
        different thing: those set the field for a turn that never dispatched,
        so there is no handler-reported value for them to clobber. The property
        is about the two sites that deliberately check the field first -- they
        are the ones a handler's report has to survive.
        """
        tree = ast.parse(_MAIN_PY.read_text(encoding="utf-8"))

        guarded = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            # The distinguishing feature: the guard consults the field.
            if "action_outcome" not in ast.unparse(node.test):
                continue
            assigns_success = any(
                isinstance(s, ast.Assign)
                and any(isinstance(t, ast.Attribute)
                        and t.attr == "action_outcome" for t in s.targets)
                and isinstance(s.value, ast.Constant)
                and s.value.value == "success"
                for s in node.body
            )
            if assigns_success:
                guarded.append(node)

        assert guarded, (
            "walked nothing: no `if <x>.action_outcome ...: <y>.action_outcome "
            "= \"success\"` in main.py. If the post-dispatch stamp moved, "
            "re-point this sweep -- an empty walk is not a pass."
        )
        self.assertGreaterEqual(len(guarded), 2, "expected the planner and "
                                                 "dispatch sites, found fewer")

        for node in guarded:
            rendered = ast.unparse(node.test)
            with self.subTest(line=node.lineno, test=rendered):
                self.assertIn(
                    "HANDLER_REPORTED", rendered,
                    f"line {node.lineno} guards the success stamp with "
                    f"{rendered!r} instead of the HANDLER_REPORTED set. A "
                    f"comparison against one literal is how mark_action_"
                    f"uncertain would be overwritten the line after it ran.",
                )


# ─── Disclosure and the stored row must agree ────────────────────────────────


class TestDisclosureIsAlsoRecorded(unittest.TestCase):
    """The reply already said "couldn't visually confirm"; the row said
    success. These are the same fact, so one function reports both."""

    def setUp(self):
        ca._task_state.reset()

    def tearDown(self):
        ca._task_state.reset()

    def _abandon_one(self) -> None:
        ca._task_state.set_initial_todos([
            "Type 'John' in First Name",
            "Select '1-50' from Staff Size dropdown",
        ])
        todo = ca._task_state.todo_list[1]
        todo["field"] = "Staff Size"
        todo["value"] = "1-50"
        ca._task_state.mark_todo_done(todo["id"])
        todo["confirm_abandoned"] = True
        ca._task_state.confirm_abandoned_count = 1

    def test_an_abandoned_todo_marks_the_turn_uncertain(self):
        self._abandon_one()
        with _tracker_ctx() as t:
            out = ca._append_abandoned_suffix("Filled the form.")
        self.assertIn("couldn't visually confirm", out)
        self.assertEqual(t.action_outcome, "uncertain")

    def test_the_spoken_disclosure_and_the_row_never_disagree(self):
        """The property, stated over both directions at once.

        Either the sentence hedges and the row says uncertain, or the sentence
        does not hedge and the row is untouched. A disclosure with a `success`
        row is the defect; a hedge-free reply with an `uncertain` row would be
        the opposite one.
        """
        for abandon in (False, True):
            with self.subTest(abandoned=abandon):
                ca._task_state.reset()
                if abandon:
                    self._abandon_one()
                else:
                    ca._task_state.set_initial_todos(["Type 'John' in First Name"])
                with _tracker_ctx() as t:
                    out = ca._append_abandoned_suffix("Filled the form.")
                hedged = "couldn't visually confirm" in out
                self.assertEqual(hedged, abandon)
                self.assertEqual(t.action_outcome == "uncertain", abandon)

    def test_a_clean_task_is_left_alone(self):
        ca._task_state.set_initial_todos(["Type 'John' in First Name"])
        with _tracker_ctx("skipped") as t:
            out = ca._append_abandoned_suffix("Filled the form.")
        self.assertEqual(out, "Filled the form.")
        self.assertEqual(t.action_outcome, "skipped")

    def test_telemetry_failing_does_not_cost_the_user_a_reply(self):
        self._abandon_one()
        with patch.object(telemetry, "mark_action_uncertain",
                          side_effect=RuntimeError("db gone")):
            out = ca._append_abandoned_suffix("Filled the form.")
        self.assertIn("couldn't visually confirm", out)


# ─── The give-up exits ───────────────────────────────────────────────────────


class TestEveryGiveUpExitTellsTelemetry(unittest.TestCase):
    """Three exits give up and only one used to say so.

    Structural rather than driven, because reaching the no-LLM and
    unparseable-plan exits means standing up the whole planner call chain --
    and what matters is that no *fourth* graceful give-up gets added without a
    marking, which a driven test of the existing three would never catch.
    """

    # The graceful sentences this loop returns when it accomplished nothing.
    _GIVE_UP_MARKERS = (
        "VisionAgentNoLLM",
        "VisionAgentUnparseablePlan",
        "VisionAgentMaxLoops",
    )

    def test_all_three_give_up_exits_mark_a_failure(self):
        src = _AGENT_PY.read_text(encoding="utf-8")
        tree = ast.parse(src)
        marked = {
            n.args[0].value
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "_mark_agent_failure"
            and n.args and isinstance(n.args[0], ast.Constant)
        }
        assert marked, "walked nothing: no _mark_agent_failure call in agent.py"
        self.assertEqual(set(self._GIVE_UP_MARKERS), marked)

    def test_no_sorry_return_is_unmarked(self):
        """Every "Sorry, I couldn't ..." return must have a marking above it.

        The two exits fixed here both begin that way, and it is the phrasing a
        third one would copy.
        """
        src = _AGENT_PY.read_text(encoding="utf-8")
        lines = src.splitlines()
        sorry = [i for i, ln in enumerate(lines)
                 if "return \"Sorry, I couldn't" in ln]
        assert sorry, "walked nothing: no `Sorry, I couldn't` return in agent.py"
        for i in sorry:
            window = "\n".join(lines[max(0, i - 4):i])
            with self.subTest(line=i + 1):
                self.assertIn(
                    "_mark_agent_failure", window,
                    f"line {i + 1} gives up gracefully with no telemetry "
                    f"marking within four lines above it, so main.py will "
                    f"stamp the turn 'success'.",
                )


# ─── The false abort ─────────────────────────────────────────────────────────


class TestFocusRefusalIsNotAnAbort(unittest.TestCase):
    """`ABORTED_WRONG_FOCUS` is a safety refusal, not the user pressing ESC.

    A genuine abort *raises* `_UserAborted`, so it never becomes a result
    string. The deleted check could therefore only ever fire for the focus
    refusal -- telling the user they had aborted something they had not, and
    cutting off the recovery the planner prompt explicitly asks for.
    """

    def test_no_result_string_check_returns_an_abort_message(self):
        src = _AGENT_PY.read_text(encoding="utf-8")
        tree = ast.parse(src)

        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            test_src = ast.unparse(node.test)
            if "ABORTED" not in test_src:
                continue
            for stmt in node.body:
                if (isinstance(stmt, ast.Return)
                        and isinstance(stmt.value, ast.Constant)
                        and "abort" in str(stmt.value.value).lower()):
                    offenders.append((node.lineno, test_src))

        self.assertFalse(offenders, (
            f"a check on an ABORTED result string returns an abort message at "
            f"{offenders}. The only ABORTED strings in this module are "
            f"ABORTED_WRONG_FOCUS, a recoverable focus refusal; a real abort "
            f"raises UserAborted and never reaches a result list."
        ))

    def test_the_only_aborted_strings_are_the_focus_refusal(self):
        """The premise of the test above, asserted so it cannot rot.

        If a genuine user abort ever starts producing an `ABORTED` result
        string, the deleted check becomes defensible and this test is where
        that gets noticed.
        """
        src = _AGENT_PY.read_text(encoding="utf-8")
        tree = ast.parse(src)
        # `startswith`, not `in`. The planner and TODO-update prompts both
        # *mention* ABORTED in prose ("ABORTED / FAILED action results -> do
        # NOT mark the item done"), and a prompt is not a result string. Only a
        # value a step returns can be matched by a check on `results`, and
        # every such value in this module leads with the tag. The first draft
        # used `in` and failed on the prompt, which is the right kind of
        # over-broad to catch here rather than in review.
        produced = {
            v.value for v in ast.walk(tree)
            if isinstance(v, ast.Constant) and isinstance(v.value, str)
            and v.value.lstrip().startswith("ABORTED")
        }
        assert produced, (
            "walked nothing: no ABORTED-leading literal in agent.py. If the "
            "focus refusal was reworded, re-point this sweep."
        )
        for text in produced:
            with self.subTest(text=text[:60]):
                self.assertTrue(
                    text.lstrip().startswith("ABORTED_WRONG_FOCUS"),
                    f"a new ABORTED result string exists: {text[:80]!r}. If it "
                    f"represents a user abort it must raise UserAborted "
                    f"instead -- see .claude/rules/automation.md.",
                )

    def test_the_planner_prompt_still_asks_for_focus_recovery(self):
        """The reason deleting the early return matters.

        `VISION_PLANNER_SYSTEM_PROMPT` tells the model to add a
        `focus_application` step when it sees `ABORTED_WRONG_FOCUS`. That
        instruction was unreachable while the loop returned first, so the
        prompt rule and the deletion are one change: if the rule ever goes, the
        deletion loses its justification.
        """
        src = _AGENT_PY.read_text(encoding="utf-8")
        self.assertIn("ABORTED_WRONG_FOCUS", src)
        self.assertIn("focus_application", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
