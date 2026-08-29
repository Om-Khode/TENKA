"""
test_page_adapter.py — Latch Task 1: the PageAdapter/LocatorAdapter seam.

The DOM tier touches a Playwright ``Page`` in exactly three places
(``dom.py:972`` ``url``, ``:979`` ``evaluate``, ``:1055`` ``locator``) and a
``Locator`` through nine methods. This file pins that seam so a second
implementation — the Latch extension — can satisfy it without the tier
noticing.

What is tested:

  - PageAdapter / LocatorAdapter are runtime-checkable and accept a fake that
    implements every member, and REJECT one missing any single member. The
    rejection half is the point: a Protocol that accepts everything documents
    nothing.
  - PlaywrightPage / PlaywrightLocator forward every member with arguments
    UNCHANGED. Asserted on recorded call args, not on "was called". A wrapper
    that silently reorders `select_option(label=...)` into a positional, or
    coerces a millisecond int, changes Playwright's semantics while every
    "was it called" test stays green.
  - PlaywrightPage.locator() returns something satisfying LocatorAdapter, so
    the chain from dom.py:1055 into PageDomTree.ref_to_locator stays typed
    end to end.

Run: py -3.11 -m pytest tests/test_page_adapter.py -v
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant.automation.browser.page_adapter import (  # noqa: E402
    LocatorAdapter,
    PageAdapter,
    PlaywrightLocator,
    PlaywrightPage,
)


def _run(coro):
    return asyncio.run(coro)


# ─── The member sets, declared once ──────────────────────────────────────
# Every conformance test derives from these, so adding a member to a Protocol
# without adding it here fails loudly rather than going unchecked.

_PAGE_MEMBERS = ("url", "evaluate", "locator")
_LOCATOR_MEMBERS = (
    "click", "fill", "input_value", "check", "uncheck", "is_checked",
    "press", "select_option", "evaluate",
)


# ─── Recording fakes ─────────────────────────────────────────────────────


class _Recorder:
    """Records (name, args, kwargs) for every call made through it."""

    def __init__(self):
        self.calls: list[tuple[str, tuple, dict]] = []

    def record(self, name, args, kwargs):
        self.calls.append((name, args, kwargs))

    def only(self) -> tuple[str, tuple, dict]:
        assert len(self.calls) == 1, f"expected exactly one call, got {self.calls}"
        return self.calls[0]


class _FakeLocator:
    """A stand-in for a Playwright Locator that records how it was called."""

    def __init__(self, rec: _Recorder, ret=None):
        self._rec = rec
        self._ret = ret

    async def click(self, **kw):
        self._rec.record("click", (), kw)

    async def fill(self, value, **kw):
        self._rec.record("fill", (value,), kw)

    async def input_value(self, **kw):
        self._rec.record("input_value", (), kw)
        return self._ret

    async def check(self, **kw):
        self._rec.record("check", (), kw)

    async def uncheck(self, **kw):
        self._rec.record("uncheck", (), kw)

    async def is_checked(self, **kw):
        self._rec.record("is_checked", (), kw)
        return self._ret

    async def press(self, key, **kw):
        self._rec.record("press", (key,), kw)

    async def select_option(self, **kw):
        self._rec.record("select_option", (), kw)
        return self._ret

    async def evaluate(self, expression, *a, **kw):
        self._rec.record("evaluate", (expression,) + a, kw)
        return self._ret


class _FakePage:
    """A stand-in for a Playwright Page that records how it was called."""

    def __init__(self, rec: _Recorder, locator_rec: _Recorder | None = None, ret=None):
        self._rec = rec
        self._locator_rec = locator_rec or _Recorder()
        self._ret = ret
        self.url = "https://example.invalid/start"

    async def evaluate(self, expression, arg=None):
        self._rec.record("evaluate", (expression, arg), {})
        return self._ret

    def locator(self, selector):
        self._rec.record("locator", (selector,), {})
        return _FakeLocator(self._locator_rec)


def _strip(obj, member):
    """A copy of `obj`'s class with `member` removed — used to prove the
    Protocol check actually rejects. Subclassing and setting the attribute to
    a non-existent name is not enough: runtime_checkable uses hasattr, so the
    member has to genuinely be absent."""
    ns = {k: v for k, v in vars(type(obj)).items() if k != member}
    ns.pop("__dict__", None)
    ns.pop("__weakref__", None)
    stripped_cls = type(f"Stripped_{member}", (), ns)
    inst = stripped_cls.__new__(stripped_cls)
    inst.__dict__.update({k: v for k, v in vars(obj).items() if k != member})
    return inst


# ─── Protocol conformance ────────────────────────────────────────────────


class TestProtocolConformance(unittest.TestCase):

    def test_full_fake_page_satisfies_page_adapter(self):
        self.assertIsInstance(_FakePage(_Recorder()), PageAdapter)

    def test_full_fake_locator_satisfies_locator_adapter(self):
        self.assertIsInstance(_FakeLocator(_Recorder()), LocatorAdapter)

    def test_page_missing_any_member_is_rejected(self):
        for member in _PAGE_MEMBERS:
            with self.subTest(missing=member):
                crippled = _strip(_FakePage(_Recorder()), member)
                self.assertFalse(
                    isinstance(crippled, PageAdapter),
                    f"PageAdapter accepted an object with no {member!r} — "
                    f"the Protocol is not pinning that member",
                )

    def test_locator_missing_any_member_is_rejected(self):
        for member in _LOCATOR_MEMBERS:
            with self.subTest(missing=member):
                crippled = _strip(_FakeLocator(_Recorder()), member)
                self.assertFalse(
                    isinstance(crippled, LocatorAdapter),
                    f"LocatorAdapter accepted an object with no {member!r} — "
                    f"the Protocol is not pinning that member",
                )

    def test_wrappers_satisfy_their_protocols(self):
        self.assertIsInstance(PlaywrightPage(_FakePage(_Recorder())), PageAdapter)
        self.assertIsInstance(PlaywrightLocator(_FakeLocator(_Recorder())), LocatorAdapter)


# ─── Forwarding: arguments must arrive unchanged ─────────────────────────


class TestPageForwarding(unittest.TestCase):

    def test_evaluate_forwards_expression_and_arg(self):
        rec = _Recorder()
        page = PlaywrightPage(_FakePage(rec, ret={"elements": []}))
        arg = {"filter": "interactive", "openComboboxes": True}
        out = _run(page.evaluate("(cfg) => cfg", arg))
        name, args, kwargs = rec.only()
        self.assertEqual(name, "evaluate")
        self.assertEqual(args, ("(cfg) => cfg", arg))
        self.assertIs(args[1], arg, "the arg dict was copied or rebuilt, not forwarded")
        self.assertEqual(kwargs, {})
        self.assertEqual(out, {"elements": []})

    def test_evaluate_forwards_none_arg_rather_than_omitting_it(self):
        rec = _Recorder()
        page = PlaywrightPage(_FakePage(rec))
        _run(page.evaluate("document.body.innerText"))
        _, args, _ = rec.only()
        self.assertEqual(args, ("document.body.innerText", None))

    def test_locator_forwards_selector_verbatim(self):
        rec = _Recorder()
        page = PlaywrightPage(_FakePage(rec))
        sel = "[data-tenka-idx='7']"
        page.locator(sel)
        name, args, _ = rec.only()
        self.assertEqual(name, "locator")
        self.assertEqual(args, (sel,))

    def test_locator_returns_a_locator_adapter(self):
        page = PlaywrightPage(_FakePage(_Recorder()))
        self.assertIsInstance(page.locator("[data-tenka-idx='0']"), LocatorAdapter)

    def test_url_reads_through_to_the_wrapped_page_live(self):
        inner = _FakePage(_Recorder())
        page = PlaywrightPage(inner)
        self.assertEqual(page.url, "https://example.invalid/start")
        inner.url = "https://example.invalid/after-nav"
        self.assertEqual(
            page.url, "https://example.invalid/after-nav",
            "url was snapshotted at construction — navigation detection in "
            "dom_orchestrator compares urls across reads and would go blind",
        )


class TestLocatorForwarding(unittest.TestCase):

    def _wrap(self, ret=None):
        rec = _Recorder()
        return rec, PlaywrightLocator(_FakeLocator(rec, ret=ret))

    def test_click_forwards_timeout_as_int_keyword(self):
        rec, loc = self._wrap()
        _run(loc.click(timeout=10_000))
        name, args, kwargs = rec.only()
        self.assertEqual(name, "click")
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"timeout": 10_000})
        self.assertIsInstance(
            kwargs["timeout"], int,
            "timeout was coerced; Playwright takes milliseconds and the call "
            "sites pass ints",
        )

    def test_fill_forwards_value_positionally_and_timeout_by_keyword(self):
        rec, loc = self._wrap()
        _run(loc.fill("hello", timeout=30_000))
        name, args, kwargs = rec.only()
        self.assertEqual((name, args, kwargs), ("fill", ("hello",), {"timeout": 30_000}))

    def test_input_value_returns_the_wrapped_result(self):
        rec, loc = self._wrap(ret="read-back")
        out = _run(loc.input_value(timeout=5_000))
        self.assertEqual(out, "read-back")
        self.assertEqual(rec.only(), ("input_value", (), {"timeout": 5_000}))

    def test_check_forwards_timeout(self):
        rec, loc = self._wrap()
        _run(loc.check(timeout=10_000))
        self.assertEqual(rec.only(), ("check", (), {"timeout": 10_000}))

    def test_uncheck_forwards_timeout(self):
        rec, loc = self._wrap()
        _run(loc.uncheck(timeout=10_000))
        self.assertEqual(rec.only(), ("uncheck", (), {"timeout": 10_000}))

    def test_is_checked_returns_the_wrapped_boolean(self):
        # dom_filler.fill_checkbox short-circuits when the box is already in
        # the wanted state. A wrapper that returned the coroutine, or coerced
        # False to None, would make it re-click every checkbox.
        rec, loc = self._wrap(ret=False)
        self.assertIs(_run(loc.is_checked(timeout=10_000)), False)
        self.assertEqual(rec.only(), ("is_checked", (), {"timeout": 10_000}))

    def test_press_forwards_key_positionally(self):
        rec, loc = self._wrap()
        _run(loc.press("Enter", timeout=10_000))
        self.assertEqual(rec.only(), ("press", ("Enter",), {"timeout": 10_000}))

    def test_select_option_keeps_label_as_a_keyword(self):
        rec, loc = self._wrap()
        _run(loc.select_option(label="India", timeout=10_000))
        name, args, kwargs = rec.only()
        self.assertEqual(name, "select_option")
        self.assertEqual(
            args, (),
            "label was passed positionally; Playwright's first positional "
            "argument means match-by-value, so by-label selection would "
            "silently become by-value",
        )
        self.assertEqual(kwargs, {"label": "India", "timeout": 10_000})

    def test_select_option_keeps_value_as_a_keyword(self):
        rec, loc = self._wrap()
        _run(loc.select_option(value="IN", timeout=10_000))
        name, args, kwargs = rec.only()
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"value": "IN", "timeout": 10_000})

    def test_select_option_omits_the_key_the_caller_did_not_pass(self):
        # dom_executor tries label first, then falls back to value. If the
        # wrapper forwarded label=None alongside value=..., Playwright would
        # see two selectors and the fallback would not behave as the executor
        # expects.
        rec, loc = self._wrap()
        _run(loc.select_option(value="IN", timeout=10_000))
        _, _, kwargs = rec.only()
        self.assertNotIn("label", kwargs)

    def test_evaluate_forwards_expression_verbatim(self):
        rec, loc = self._wrap(ret="Option text")
        expr = "el => el.options[el.selectedIndex].text"
        out = _run(loc.evaluate(expr))
        self.assertEqual(out, "Option text")
        name, args, _ = rec.only()
        self.assertEqual((name, args), ("evaluate", (expr,)))


# ─── The wrappers must not invent behaviour ──────────────────────────────


class TestWrappersAreTransparent(unittest.TestCase):

    def test_page_evaluate_does_not_swallow_exceptions(self):
        # dom.py:979 relies on a raising evaluate to set evaluate_failed=True,
        # which the orchestrator reads as a navigation signal. A wrapper that
        # caught and returned None would erase that signal.
        class _Boom(_FakePage):
            async def evaluate(self, expression, arg=None):
                raise RuntimeError("Execution context was destroyed")

        page = PlaywrightPage(_Boom(_Recorder()))
        with self.assertRaises(RuntimeError):
            _run(page.evaluate("x", None))

    def test_locator_click_does_not_swallow_exceptions(self):
        class _Boom(_FakeLocator):
            async def click(self, **kw):
                raise RuntimeError("element is not visible")

        loc = PlaywrightLocator(_Boom(_Recorder()))
        with self.assertRaises(RuntimeError):
            _run(loc.click(timeout=10_000))

    def test_page_locator_is_not_async(self):
        # dom.py:1055 calls page.locator() without awaiting it. An async
        # wrapper here would return a coroutine that every downstream
        # ref_to_locator consumer would then try to .click().
        page = PlaywrightPage(_FakePage(_Recorder()))
        self.assertFalse(asyncio.iscoroutine(page.locator("[x]")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
