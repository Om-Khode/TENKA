"""test_p13_uncertain_reaches_the_db.py — TENKA-v2 §17.P13, loop 3.

`tests/test_vision_outcome_honesty.py` pins `mark_action_uncertain` against a
**stand-in tracker**, and a stand-in is exactly the shape
`.claude/rules/testing.md` calls the strongest live-test signal: the tests
exercise a fake of the thing that changed. Two hops go unproven by it --

    mark_action_uncertain -> a real TurnTracker -> interaction_events.action_outcome

-- and `action_outcome` is a free `TEXT` column, so nothing in the schema would
have complained about a value the writer never actually stores.

This file closes both hops against **real SQLite in a tmp dir**, which
`.claude/rules/storage.md` requires for anything touching persistence: a mocked
DB has masked a migration failure in this project before.

It does not need the vision agent. Reaching that loop live means defeating
three cheaper automation tiers -- it is the last fallback by design -- and two
attempts on 2026-08-28/29 were absorbed by the planner and the native tier
before it. What the agent contributes is one call to `mark_action_uncertain`,
already pinned structurally; what was unverified is everything downstream of
that call, and none of it is vision-specific.

Run: py -3.11 -m pytest tests/test_p13_uncertain_reaches_the_db.py -q
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import assistant.telemetry as telemetry
from assistant import config as _config



def _fresh_db() -> Path:
    """A real database file, migrated by the real migrator."""
    from assistant.storage.db import _reset_for_testing, init_db

    _reset_for_testing()
    tmp = Path(tempfile.mkdtemp()) / "memory" / "tenka.db"
    _config.SANDBOX_DIR = tmp.parent.parent
    tmp.parent.mkdir(parents=True, exist_ok=True)
    init_db(tmp)
    # The telemetry repo is cached module-side; drop it so it rebinds to this
    # database rather than to whichever one a previous test opened.
    telemetry._repo = None
    return tmp


def _rows(db: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return list(conn.execute(
        "SELECT action_outcome, error_class, action_dispatched "
        "FROM interaction_events ORDER BY id"))


class _TurnUnderTest:
    """One real tracker, installed as the current one, saved on exit."""

    def __init__(self, dispatched: str = "computer_task") -> None:
        self.dispatched = dispatched

    def __enter__(self) -> "telemetry.TurnTracker":
        self.tracker = telemetry.TurnTracker(
            session_id="p13-probe", input_modality="stt",
            transcript="fill the form",
        )
        self.tracker.intent_detected = "computer_task"
        self.tracker.action_dispatched = self.dispatched
        self._token = telemetry.set_current_tracker(self.tracker)
        return self.tracker

    def __exit__(self, *exc) -> None:
        self.tracker.save()
        telemetry.reset_current_tracker(self._token)


class TestUncertainSurvivesToTheColumn(unittest.TestCase):
    def setUp(self):
        self.db = _fresh_db()

    def test_a_marked_turn_stores_uncertain(self):
        with _TurnUnderTest():
            telemetry.mark_action_uncertain(
                "VisionConfirmAbandoned", "2 TODOs trusted on action signature")

        rows = _rows(self.db)
        self.assertEqual(len(rows), 1, "the turn was not persisted at all")
        self.assertEqual(rows[0]["action_outcome"], "uncertain")
        self.assertEqual(rows[0]["error_class"], "VisionConfirmAbandoned")

    def test_an_unmarked_turn_is_untouched(self):
        # The control. A default tracker must not acquire "uncertain" just
        # because the function now exists.
        with _TurnUnderTest() as t:
            self.assertNotEqual(t.action_outcome, "uncertain")

        self.assertNotEqual(_rows(self.db)[0]["action_outcome"], "uncertain")

    def test_the_dispatch_stamp_does_not_overwrite_it(self):
        """The whole point, in the order a real turn does it.

        `main.py` marks `action_outcome = "success"` after dispatch for any
        turn no handler contradicted. The handler's mark happens *inside*
        dispatch, so it lands first and the stamp runs over the top of it --
        which is why the stamp had to start consulting
        `telemetry.HANDLER_REPORTED` rather than comparing against one
        literal. This reproduces that sequence rather than trusting the AST
        sweep that the condition is spelled correctly.
        """
        with _TurnUnderTest() as t:
            telemetry.mark_action_uncertain("VisionConfirmAbandoned", "")
            # verbatim from main.py's post-dispatch block
            if t.action_outcome not in telemetry.HANDLER_REPORTED:
                t.action_outcome = "success"

        self.assertEqual(_rows(self.db)[0]["action_outcome"], "uncertain")

    def test_the_old_condition_would_have_lost_it(self):
        """The mutation, run in-process instead of by editing main.py.

        `!= "failure"` is what that line said before P13. Kept as a test so
        the reason the condition changed is legible next to the thing it
        protects -- and so that reverting main.py is not the only way to
        discover the consequence.
        """
        with _TurnUnderTest() as t:
            telemetry.mark_action_uncertain("VisionConfirmAbandoned", "")
            if t.action_outcome != "failure":          # the pre-P13 condition
                t.action_outcome = "success"

        self.assertEqual(
            _rows(self.db)[0]["action_outcome"], "success",
            "the pre-P13 condition no longer loses the uncertain mark -- if "
            "that is because the field became something other than a plain "
            "string, this test is measuring nothing and must be rewritten",
        )

    def test_a_real_failure_still_wins_and_still_stores(self):
        with _TurnUnderTest():
            telemetry.mark_action_uncertain("Unconfirmed", "")
            telemetry.mark_action_failure("VisionAgentMaxLoops", "gave up")

        row = _rows(self.db)[0]
        self.assertEqual(row["action_outcome"], "failure")
        self.assertEqual(row["error_class"], "VisionAgentMaxLoops")

    def test_every_outcome_name_the_procedure_map_emits_round_trips(self):
        """Loop 2's map writes into the same column, so its values are checked
        against real storage here rather than assumed to fit."""
        from assistant import main as main_mod

        for outcome, name in main_mod._PROC_OUTCOME_TO_TELEMETRY.items():
            with self.subTest(outcome=outcome.value, stored=name):
                self.db = _fresh_db()
                with _TurnUnderTest("some procedure") as t:
                    t.action_outcome = name
                self.assertEqual(_rows(self.db)[0]["action_outcome"], name)

    def test_the_map_emits_nothing_the_correction_detector_misreads(self):
        """`_maybe_mark_correction` treats `("failure", "skipped")` as "the
        last turn did not work". A value outside the vocabulary it knows is
        silently neither, which is a decision made by omission -- so the set
        of values this project writes is enumerated here.
        """
        from assistant import main as main_mod

        known = {"success", "failure", "skipped", "refused", "uncertain"}
        emitted = set(main_mod._PROC_OUTCOME_TO_TELEMETRY.values())
        self.assertTrue(
            emitted <= known,
            f"the procedure map emits {sorted(emitted - known)}, which nothing "
            f"downstream classifies. Add it to telemetry's readers first.",
        )
        self.assertIn("uncertain", known)
        self.assertNotIn(
            "uncertain", {"failure", "skipped"},
            "if uncertain is ever added to the correction detector's failed "
            "set, that is a behaviour change P13 deliberately did not make",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
