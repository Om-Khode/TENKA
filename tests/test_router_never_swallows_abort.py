"""test_router_never_swallows_abort.py — KI-36, and the class KI-39 was in.

`"__FALLBACK__"` is not an error string. It is an instruction to escalate to a
more expensive tier. A handler that swallows `UserAborted` into it therefore
does not merely lose the abort, it **re-triggers work**: the vision loop starts,
TTS says "Working on it", terminal windows are minimized and a vision call is
billed — all while the user is holding the key that means stop.

`automation/router.py` has nine broad handlers that return that sentinel. Two
were guarded by hand: one on the type-shortcut path after the bug bit there,
one on the DOM path after it bit again during P13. **The other seven were
not**, and nobody noticed either time, because fixing the instance in front of
you leaves the class alone.

So the rule has one implementation, `_reraise_if_user_aborted`, and this file
fails on a handler that returns the sentinel without calling it. The count is
asserted too: a sweep whose target set silently shrinks passes forever.

Pure AST plus one behavioural check. Nothing here opens a browser.

Run: py -3.11 -m pytest tests/test_router_never_swallows_abort.py -q
"""

from __future__ import annotations

import ast
import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent))

from assistant.automation import router
from assistant.core.abort import UserAborted

_ROUTER_PY = Path(__file__).parent.parent / "assistant" / "automation" / "router.py"
_GUARD = "_reraise_if_user_aborted"



def _skip_verify(*a, **kw):
    """A post-verify that decides nothing, so the loop moves on to step 2."""
    from assistant.automation import verification
    from assistant.core.verdict import Outcome
    return verification.VerifyResult(
        outcome=Outcome.UNVERIFIED, observation="stub", tier="skipped",
        confidence=1.0)


def _fallback_handlers() -> list[ast.ExceptHandler]:
    """Every broad `except` in router.py that can return `__FALLBACK__`."""
    tree = ast.parse(_ROUTER_PY.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            broad = (handler.type is None
                     or getattr(handler.type, "id", None)
                     in ("Exception", "BaseException"))
            if not broad:
                continue
            returns = [ast.unparse(n.value) for n in ast.walk(handler)
                       if isinstance(n, ast.Return) and n.value is not None]
            if any("__FALLBACK__" in r for r in returns):
                found.append(handler)
    return found


class TestEveryFallbackHandlerLetsAnAbortPast(unittest.TestCase):
    def test_the_sweep_walks_something(self):
        # A structural test whose target set is empty passes forever.
        handlers = _fallback_handlers()
        self.assertGreaterEqual(
            len(handlers), 9,
            f"only {len(handlers)} fallback-returning handlers found in "
            f"router.py, expected at least 9. If the tiers were restructured, "
            f"re-point this sweep rather than lowering the floor — an empty "
            f"walk is not a pass.",
        )

    def test_none_of_them_swallows_an_abort(self):
        bad = [h.lineno for h in _fallback_handlers()
               if _GUARD not in ast.unparse(h)]
        self.assertFalse(
            bad,
            f"handlers at lines {bad} return __FALLBACK__ without calling "
            f"{_GUARD}(e). That sentinel escalates a tier, so an abort landing "
            f"there starts the vision loop, speaks, and bills a model call "
            f"while the user is trying to stop.",
        )

    def test_the_guard_is_not_hand_copied(self):
        """One implementation, not nine.

        Two sites used to inline `isinstance(e, UserAborted): raise` with a
        hand-written comment, which is exactly how seven others went without.
        A second inline copy means the rule has started spreading again.
        """
        tree = ast.parse(_ROUTER_PY.read_text(encoding="utf-8"))
        inline = [
            n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and "UserAborted" in ast.unparse(n.test)
            and any(isinstance(s, ast.Raise) for s in n.body)
        ]
        # The one inside `_reraise_if_user_aborted` itself is the definition.
        self.assertLessEqual(
            len(inline), 1,
            f"UserAborted re-raises written inline at lines {inline}. There is "
            f"one helper for this; a second copy is how the other seven were "
            f"missed.",
        )


class TestTheGuardActuallyReRaises(unittest.TestCase):
    """The behaviour, because the sweep only proves the call is written."""

    def test_it_reraises_a_user_abort(self):
        with self.assertRaises(UserAborted):
            router._reraise_if_user_aborted(UserAborted("esc_hold"))

    def test_it_leaves_everything_else_alone(self):
        # Returning normally is what lets the handler log and fall back.
        for exc in (RuntimeError("boom"), ValueError("nope"), OSError("io")):
            with self.subTest(exc=type(exc).__name__):
                self.assertIsNone(router._reraise_if_user_aborted(exc))

    def test_the_original_exception_object_survives(self):
        # `raise e`, not `raise UserAborted(...)`: the reason the user's abort
        # carried must reach whoever reports it.
        original = UserAborted("esc_hold")
        with self.assertRaises(UserAborted) as ctx:
            router._reraise_if_user_aborted(original)
        self.assertIs(ctx.exception, original)


class TestTheDomPathStillPropagates(unittest.TestCase):
    """The P13 loop-1 property, re-asserted through the shared helper.

    That fix inlined its own guard; this converts it to the helper, and a
    conversion that quietly dropped the behaviour would look like tidying.
    """

    def test_an_abort_from_the_orchestrator_is_not_a_fallback(self):
        from assistant.automation.browser import cdp as bcdp
        from assistant.automation.browser import dom_orchestrator as bdo
        from unittest.mock import MagicMock

        handle = MagicMock()
        handle.kind = "cdp"
        handle.attachment = MagicMock()
        page = MagicMock()
        page.url = "https://example.invalid/form"

        with patch.object(bcdp, "get_or_attach_browser",
                          new=AsyncMock(return_value=handle)), \
             patch.object(router, "_pick_active_page",
                          new=AsyncMock(return_value=page)), \
             patch.object(bdo, "run_dom_task",
                          new=AsyncMock(side_effect=UserAborted("esc_hold"))):
            with self.assertRaises(UserAborted):
                asyncio.run(router._execute_dom_task("do the thing"))


class TestTheBrowserTierChecksBetweenSteps(unittest.TestCase):
    """KI-36 itself: the boundary the bundled-browser tier did not have.

    `handle_browser_action` checked `abort.is_aborted()` once on the way in and
    nothing looked again, so ESC held after the handler was entered was ignored
    for the rest of the task. From the live log:

        21:58:44  [BROWSER] Launching Chromium (headless=False)...
        21:58:47  [abort]   requested: esc_hold
        21:58:50  [BROWSER] Step 1: navigate - {...}
        21:58:57  [llm]     Vision (Gemini) OK (788 tokens)
        21:58:57  Executed 'browser_action': Navigated to https://...

    The abort is requested, then a navigation runs, then a vision call is
    billed, then the tier reports **success**.

    `native.py:run_app_steps` has had this check for a while, which is what
    made the gap a defect rather than a design: one tier honoured the contract
    and its sibling did not.

    Playwright is never reached -- the abort fires before the first step, which
    is the whole point of it being a boundary.
    """

    def setUp(self):
        from assistant.core.abort import abort
        abort.reset()

    def tearDown(self):
        from assistant.core.abort import abort
        abort.reset()

    def _steps(self):
        return [{"action": "navigate", "params": {"url": "https://example.invalid"}},
                {"action": "click", "params": {"selector": "#go"}}]

    def test_an_abort_raises_instead_of_returning_a_string(self):
        from assistant.automation.browser import automation as ba
        from assistant.core.abort import abort

        abort.request_abort("esc_hold")
        with patch.object(ba, "PLAYWRIGHT_AVAILABLE", True):
            with self.assertRaises(UserAborted):
                asyncio.run(ba.run_browser_steps(self._steps()))

    def test_the_reason_the_user_gave_survives(self):
        from assistant.automation.browser import automation as ba
        from assistant.core.abort import abort

        abort.request_abort("esc_hold")
        with patch.object(ba, "PLAYWRIGHT_AVAILABLE", True):
            with self.assertRaises(UserAborted) as ctx:
                asyncio.run(ba.run_browser_steps(self._steps()))
        self.assertIn("esc_hold", str(ctx.exception))

    def test_the_tail_handler_does_not_convert_it_to_a_result(self):
        """The other half. `run_browser_steps` wraps its loop in a broad
        `except` that returns `"Error running steps: ..."`, which the caller
        then turns into `__FALLBACK__`. An abort raised inside the loop has to
        pass through that untouched.
        """
        from assistant.automation.browser import automation as ba

        tree = ast.parse(
            (Path(__file__).parent.parent / "assistant" / "automation"
             / "browser" / "automation.py").read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.AsyncFunctionDef)
                  and n.name == "run_browser_steps")
        broad = [h for t in ast.walk(fn) if isinstance(t, ast.Try)
                 for h in t.handlers
                 if (h.type is None
                     or getattr(h.type, "id", None) in ("Exception", "BaseException"))
                 and any(isinstance(n, ast.Return) for n in ast.walk(h))]
        assert broad, "walked nothing: no returning broad handler in run_browser_steps"
        for h in broad:
            with self.subTest(line=h.lineno):
                self.assertIn(
                    "UserAborted", ast.unparse(h),
                    f"the handler at line {h.lineno} returns a string for any "
                    f"exception, so an abort raised in the step loop becomes "
                    f"prose and then __FALLBACK__",
                )

    def test_no_abort_means_the_loop_proceeds(self):
        """The control. A boundary that stops everything is not a boundary.

        `ensure_browser` is stubbed as well as `pre_check`: the first draft
        patched only the latter and **launched a real Chromium**, taking 5.6
        seconds and leaving a Playwright subprocess behind. A unit test of an
        abort check has no business starting a browser, and
        `.claude/rules/testing.md` is explicit that finding out by watching the
        clock is not the plan.
        """
        from assistant.automation.browser import automation as ba
        from assistant.automation import verification

        reached = []

        async def _precheck(*a, **kw):
            reached.append(1)
            raise RuntimeError("stop here — reaching step 1 is the assertion")

        # A harmless stand-in, not a refusal: with no abort set the loop is
        # *supposed* to get a browser. Making this raise would test that the
        # loop never launches one, which is the opposite property.
        async def _fake_browser(*a, **kw):
            from unittest.mock import MagicMock
            return MagicMock(name="browser")

        with patch.object(ba, "PLAYWRIGHT_AVAILABLE", True),              patch.object(ba, "ensure_browser", new=_fake_browser),              patch.object(verification, "pre_check", new=_precheck):
            asyncio.run(ba.run_browser_steps(self._steps()))
        self.assertEqual(reached, [1], "the loop refused to start with no abort set")

    def test_an_abort_raised_mid_run_stops_the_next_step(self):
        """The per-step check, which the other tests do not reach.

        Removing it was a **green mutant**: every test above sets the abort
        before calling, so the pre-launch check fires first and the in-loop one
        is never exercised. But the real case is ESC pressed *during* a
        multi-step run -- the operator's log shows exactly that, an abort at
        21:58:47 and a navigation at 21:58:50 -- so the check that matters is
        the one between steps.

        Here step 1 sets the abort, as a real keypress would mid-run, and step
        2 must not start.
        """
        from assistant.automation.browser import automation as ba
        from assistant.automation import verification
        from assistant.core.abort import abort

        started = []

        async def _precheck(verify_step, *a, **kw):
            started.append(verify_step.get("action"))
            if len(started) == 1:
                abort.request_abort("esc_hold")   # the user presses ESC now
            from assistant.core.verdict import Outcome
            return verification.VerifyResult(
                outcome=Outcome.UNVERIFIED, observation="stub",
                tier="pre", confidence=1.0)

        # `AsyncMock`, so every awaited call on the page/context chain
        # resolves instead of handing back a bare MagicMock.
        async def _fake_browser(*a, **kw):
            return AsyncMock(name="browser")

        with patch.object(ba, "PLAYWRIGHT_AVAILABLE", True),              patch.object(ba, "ensure_browser", new=_fake_browser),              patch.object(verification, "post_verify",
                          new=AsyncMock(side_effect=_skip_verify)),              patch.object(verification, "pre_check", new=_precheck):
            with self.assertRaises(UserAborted):
                asyncio.run(ba.run_browser_steps(self._steps()))

        self.assertEqual(
            started, ["navigate"],
            f"step 2 started after the abort: {started}. The in-loop check is "
            f"the one that stops a run already under way.")

    def test_the_abort_path_never_launches_a_browser(self):
        """Cheaper than the clock, and the property that matters: the check is
        a *boundary*, so it fires before anything expensive."""
        from assistant.automation.browser import automation as ba
        from assistant.core.abort import abort

        async def _no_browser(*a, **kw):
            raise AssertionError("a browser was launched despite the abort")

        abort.request_abort("esc_hold")
        with patch.object(ba, "PLAYWRIGHT_AVAILABLE", True),              patch.object(ba, "ensure_browser", new=_no_browser):
            with self.assertRaises(UserAborted):
                asyncio.run(ba.run_browser_steps(self._steps()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
