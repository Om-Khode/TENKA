"""test_cdp_owner_is_a_browser.py — KI-37.

`cdp_health_probe` answers "is this a Chromium DevTools endpoint". That is not
"is this a browser the user is using", and anything embedding a Chromium
webview and exposing a debug port satisfies the first.

**Observed on a real machine, twice.** Port 9222 was held by
`msedgewebview2.exe`, the embedded webview of a preinstalled system utility,
which identifies itself as `Edg/151.0.4129.107`. The second time was not a
test: an ordinary form-fill request routed correctly to DOM-mode against the
user's Chrome window, attached to the webview instead, perceived seven elements
of a settings panel, mapped zero fields, and said "I couldn't figure out which
fields to fill" while the form sat on screen.

Two harms, and the quieter one is the second: clicks land in an application the
user is not looking at, and its accessibility tree is read into a planner
prompt.

**The executable name is not the discriminator.** `"edge" in "msedgewebview2"`
is True, so matching the exe against the known-browser list accepts the thing
the check exists to reject. That is pinned below, because it is the obvious
first idea and someone will have it again.

What separates them is a window: a browser owns a visible top-level window
titled after itself, an embedded webview renders inside its host's.

Everything here is stubbed -- `psutil`, the Win32 enumeration and the app
registry. Nothing enumerates real windows or touches a real port.

Run: py -3.11 -m pytest tests/test_cdp_owner_is_a_browser.py -q
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

from assistant.automation.browser import cdp


class _Env:
    """A fake machine: one listening pid, a process tree, and some windows."""

    def __init__(self, *, pid, name, windows, parent=None, children=(),
                 browsers=("chrome", "edge", "brave", "firefox")):
        self.pid = pid
        self.name = name
        self.windows = windows          # {pid: [titles]}
        self.parent = parent
        self.children = children
        self.browsers = browsers
        self._patches: list = []

    def __enter__(self):
        proc = MagicMock()
        proc.name.return_value = self.name
        proc.parent.return_value = (
            MagicMock(pid=self.parent) if self.parent else None)
        proc.children.return_value = [MagicMock(pid=c) for c in self.children]

        psutil = types.ModuleType("psutil")
        psutil.Process = lambda _pid: proc
        psutil.net_connections = lambda kind=None: []

        self._patches = [
            patch.dict(sys.modules, {"psutil": psutil}),
            patch.object(cdp, "_owner_pid", return_value=self.pid),
            patch.object(cdp, "_visible_window_titles_by_pid",
                         return_value=self.windows),
            patch("assistant.core.known_apps.get_apps_by_category",
                  return_value=list(self.browsers)),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()


class TestTheEmbeddedWebviewIsRefused(unittest.TestCase):
    def test_a_webview_owning_no_window_is_refused(self):
        # The live case: the webview renders inside its host, so it owns
        # nothing of its own.
        with _Env(pid=27664, name="msedgewebview2.exe", windows={}):
            owned, why = cdp.cdp_owner_is_a_browser(9222)
        self.assertIs(owned, False)
        self.assertIn("msedgewebview2.exe", why)

    def test_a_webview_with_its_own_host_titled_window_is_refused(self):
        # A differently-hosted webview does own a window -- titled after the
        # host application, which is not a browser.
        with _Env(pid=27664, name="msedgewebview2.exe",
                  windows={27664: ["Vendor System Utility"]}):
            owned, why = cdp.cdp_owner_is_a_browser(9222)
        self.assertIs(owned, False)
        self.assertIn("Vendor System Utility", why)

    def test_the_executable_name_is_not_what_decides(self):
        """The obvious first idea, pinned as wrong.

        `"edge" in "msedgewebview2"` is True. Any check that matched the
        process name against the known-browser list would *accept* precisely
        the process this exists to reject -- so the refusal above must not be
        an accident of the name being unusual.
        """
        self.assertTrue(
            any(b in "msedgewebview2" for b in ("chrome", "edge", "brave")),
            "the premise no longer holds -- if no browser name is a substring "
            "of the webview's exe, an exe-name check would work and this "
            "window-based one is more machinery than the problem needs",
        )
        with _Env(pid=1, name="msedgewebview2.exe", windows={}):
            self.assertIs(cdp.cdp_owner_is_a_browser(9222)[0], False)


class TestARealBrowserIsAccepted(unittest.TestCase):
    """The control. A check that refuses everything is worse than no check --
    it turns one wrong tier into no tier at all."""

    def test_chrome_owning_its_own_window_is_accepted(self):
        with _Env(pid=100, name="chrome.exe",
                  windows={100: ["httpbin.org/forms/post - Google Chrome"]}):
            owned, why = cdp.cdp_owner_is_a_browser(9222)
        self.assertIs(owned, True)
        self.assertIn("Google Chrome", why)

    def test_a_window_owned_by_a_child_process_counts(self):
        # Which process in a browser's tree holds the socket versus the window
        # depends on how it was launched, so one generation either side is
        # searched.
        with _Env(pid=100, name="chrome.exe", windows={101: ["Gmail - Google Chrome"]},
                  children=(101,)):
            self.assertIs(cdp.cdp_owner_is_a_browser(9222)[0], True)

    def test_a_window_owned_by_the_parent_counts(self):
        with _Env(pid=100, name="chrome.exe", windows={99: ["Gmail - Google Chrome"]},
                  parent=99):
            self.assertIs(cdp.cdp_owner_is_a_browser(9222)[0], True)

    def test_every_known_browser_is_recognised(self):
        # The names come from `known_apps`, so a browser is added as data.
        # This fails if the matching stops being name-based.
        for name in ("chrome", "edge", "brave", "firefox", "opera", "vivaldi"):
            with self.subTest(browser=name):
                with _Env(pid=1, name=f"{name}.exe",
                          windows={1: [f"Some Page - {name.title()}"]},
                          browsers=(name,)):
                    self.assertIs(cdp.cdp_owner_is_a_browser(9222)[0], True)

    def test_an_unrelated_window_on_screen_does_not_grant_acceptance(self):
        # A browser open *somewhere* must not vouch for a webview holding the
        # port. Only the listening process's own tree counts.
        with _Env(pid=27664, name="msedgewebview2.exe",
                  windows={999: ["Gmail - Google Chrome"]}):
            self.assertIs(cdp.cdp_owner_is_a_browser(9222)[0], False)


class TestBlindIsNotTheSameAsNo(unittest.TestCase):
    """`None` attaches; `False` refuses. Collapsing them either disables
    DOM-mode wherever enumeration is restricted, or lets the webview through
    on any machine where the check throws."""

    def test_an_undeterminable_owner_returns_none(self):
        with patch.object(cdp, "_owner_pid", return_value=None):
            owned, why = cdp.cdp_owner_is_a_browser(9222)
        self.assertIsNone(owned)
        self.assertIn("could not be determined", why)

    def test_a_check_that_raises_returns_none_not_false(self):
        with patch.object(cdp, "_owner_pid", return_value=123), \
             patch.object(cdp, "_visible_window_titles_by_pid",
                          side_effect=OSError("no window station")):
            owned, why = cdp.cdp_owner_is_a_browser(9222)
        self.assertIsNone(owned)
        self.assertIn("check itself failed", why)

    def test_an_empty_browser_registry_returns_none(self):
        # No names to match means nothing can be concluded -- not that
        # everything is a webview.
        with _Env(pid=1, name="chrome.exe", windows={1: ["x - Google Chrome"]},
                  browsers=()):
            owned, why = cdp.cdp_owner_is_a_browser(9222)
        self.assertIsNone(owned)
        self.assertIn("no browser names", why)


class TestTheAttachPathActuallyConsultsIt(unittest.TestCase):
    """The wiring, not only the predicate.

    Bypassing the refusal in `connect_to_existing_chrome` left every test above
    green -- they exercise `cdp_owner_is_a_browser` directly and say nothing
    about whether anything calls it. That is the failure §16.2 records from
    P-1: deleting the durability gate's hook in `execute()` left all seventeen
    of its unit tests passing. Every check now gets a test at the point of use
    as well as a test of itself.
    """

    def _attach(self):
        return asyncio.run(cdp.connect_to_existing_chrome(port=9222))

    def test_a_refused_owner_stops_the_attach(self):
        probe = cdp.CdpProbeResult(available=True, browser="Edg/151.0",
                                   ws_endpoint="ws://x")
        with patch.object(cdp, "cdp_health_probe",
                          new=AsyncMock(return_value=probe)),              patch.object(cdp, "cdp_owner_is_a_browser",
                          return_value=(False, "a webview owns it")):
            self.assertIsNone(
                self._attach(),
                "the ownership check said no and the attach proceeded anyway")

    def test_a_refusal_never_reaches_playwright(self):
        # Cheaper and stricter than checking the return: if Playwright is
        # imported at all, the refusal happened too late to be the reason.
        probe = cdp.CdpProbeResult(available=True, browser="Edg/151.0",
                                   ws_endpoint="ws://x")
        started = []

        def _boom():
            started.append(1)
            raise AssertionError("connect_to_existing_chrome reached Playwright "
                                 "after the owner check refused")

        fake_pw = types.ModuleType("playwright")
        fake_async = types.ModuleType("playwright.async_api")
        fake_async.async_playwright = _boom
        with patch.object(cdp, "cdp_health_probe",
                          new=AsyncMock(return_value=probe)),              patch.object(cdp, "cdp_owner_is_a_browser",
                          return_value=(False, "a webview owns it")),              patch.dict(sys.modules, {"playwright": fake_pw,
                                      "playwright.async_api": fake_async}):
            self.assertIsNone(self._attach())
        self.assertEqual(started, [])

    def test_a_blind_check_does_not_stop_the_attach(self):
        """`None` must not behave like `False`.

        Refusing when the check could not run would disable DOM-mode wherever
        process enumeration is restricted -- a correctness fix turning into an
        availability bug on machines that never had the problem.
        """
        probe = cdp.CdpProbeResult(available=True, browser="Chrome/151.0",
                                   ws_endpoint="ws://x")
        reached = []

        def _mark():
            reached.append(1)
            raise RuntimeError("stop here -- getting this far is the assertion")

        fake_pw = types.ModuleType("playwright")
        fake_async = types.ModuleType("playwright.async_api")
        fake_async.async_playwright = _mark
        with patch.object(cdp, "cdp_health_probe",
                          new=AsyncMock(return_value=probe)),              patch.object(cdp, "cdp_owner_is_a_browser",
                          return_value=(None, "could not tell")),              patch.dict(sys.modules, {"playwright": fake_pw,
                                      "playwright.async_api": fake_async}):
            self._attach()
        self.assertEqual(reached, [1], "a blind check refused the attach")

    def test_an_accepted_owner_does_not_stop_the_attach(self):
        probe = cdp.CdpProbeResult(available=True, browser="Chrome/151.0",
                                   ws_endpoint="ws://x")
        reached = []

        def _mark():
            reached.append(1)
            raise RuntimeError("stop here")

        fake_pw = types.ModuleType("playwright")
        fake_async = types.ModuleType("playwright.async_api")
        fake_async.async_playwright = _mark
        with patch.object(cdp, "cdp_health_probe",
                          new=AsyncMock(return_value=probe)),              patch.object(cdp, "cdp_owner_is_a_browser",
                          return_value=(True, "chrome.exe owns 'x - Google Chrome'")),              patch.dict(sys.modules, {"playwright": fake_pw,
                                      "playwright.async_api": fake_async}):
            self._attach()
        self.assertEqual(reached, [1], "an accepted browser was refused")


if __name__ == "__main__":
    unittest.main(verbosity=2)
