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
import time
import urllib.error
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


class TestTheSocketWeActuallyUse(unittest.TestCase):
    """KI-37, second round: the owner of the *IPv4* listener is the one that
    matters, because that is where `http://127.0.0.1:<port>` lands.

    Windows lets two processes hold one port through different address
    families. Measured on the operator's machine after launching Chrome on a
    port a webview already held:

        ::1:9222        pid 9676   chrome
        127.0.0.1:9222  pid 24516  msedgewebview2

    The first `_owner_pid` returned the first LISTEN row it found, so the check
    reported "chrome.exe owns 'httpbin.org/forms/post - Google Chrome'" and
    approved -- while the probe and the attach both reached the webview and
    DOM-mode ran against a settings panel. The check was right about a socket
    nobody was going to use.
    """

    def _conns(self, rows):
        """rows: [(ip, port, status, pid)]"""
        psutil = types.ModuleType("psutil")
        psutil.net_connections = lambda kind=None: [
            types.SimpleNamespace(
                laddr=types.SimpleNamespace(ip=ip, port=port),
                status=status, pid=pid)
            for ip, port, status, pid in rows
        ]
        psutil.Process = MagicMock()
        return patch.dict(sys.modules, {"psutil": psutil})

    def test_the_ipv4_owner_wins_over_an_ipv6_listener(self):
        # The live layout, IPv6 first so the old "first row" behaviour would
        # pick Chrome.
        with self._conns([("::1", 9222, "LISTEN", 9676),
                          ("127.0.0.1", 9222, "LISTEN", 24516)]):
            self.assertEqual(cdp._owner_pid(9222), 24516)

    def test_ordering_does_not_change_the_answer(self):
        with self._conns([("127.0.0.1", 9222, "LISTEN", 24516),
                          ("::1", 9222, "LISTEN", 9676)]):
            self.assertEqual(cdp._owner_pid(9222), 24516)

    def test_a_wildcard_ipv4_bind_counts(self):
        # 0.0.0.0 serves loopback too.
        with self._conns([("0.0.0.0", 9222, "LISTEN", 500)]):
            self.assertEqual(cdp._owner_pid(9222), 500)

    def test_an_ipv6_only_listener_is_not_the_owner(self):
        # Nothing serves `http://127.0.0.1:9222`, so there is no owner to
        # report -- "blind", not "chrome".
        with self._conns([("::1", 9222, "LISTEN", 9676)]):
            self.assertIsNone(cdp._owner_pid(9222))

    def test_other_ports_and_non_listening_rows_are_ignored(self):
        with self._conns([("127.0.0.1", 9333, "LISTEN", 1),
                          ("127.0.0.1", 9222, "ESTABLISHED", 2)]):
            self.assertIsNone(cdp._owner_pid(9222))

    def test_two_ipv4_listeners_is_reported_as_undecidable(self):
        # Should not happen; if it does, nothing here can say which answers,
        # and guessing is how the first round of this bug worked.
        with self._conns([("127.0.0.1", 9222, "LISTEN", 1),
                          ("0.0.0.0", 9222, "LISTEN", 2)]):
            self.assertIsNone(cdp._owner_pid(9222))


class TestTheEndpointsOwnUserAgent(unittest.TestCase):
    """A second signal, and deliberately a weak one.

    `/json/version` carries the endpoint's User-Agent. The live webview
    answered `LenovoVantage/3.0.0.197` -- it said what it was, and nothing read
    it. Used **only** to decide the blind case: a browser launched with a
    custom `--user-agent` must not be locked out on this alone.
    """

    def _attach(self, ua, owned):
        probe = cdp.CdpProbeResult(available=True, browser="Edg/151.0",
                                   ws_endpoint="ws://x", user_agent=ua)
        reached = []

        def _mark():
            reached.append(1)
            raise RuntimeError("stop here")

        fake_pw = types.ModuleType("playwright")
        fake_async = types.ModuleType("playwright.async_api")
        fake_async.async_playwright = _mark
        with patch.object(cdp, "cdp_health_probe",
                          new=AsyncMock(return_value=probe)),              patch.object(cdp, "cdp_owner_is_a_browser",
                          return_value=(owned, "because")),              patch.dict(sys.modules, {"playwright": fake_pw,
                                      "playwright.async_api": fake_async}):
            result = asyncio.run(cdp.connect_to_existing_chrome(port=9222))
        return result, reached

    def test_a_blind_check_plus_a_host_user_agent_refuses(self):
        result, reached = self._attach("LenovoVantage/3.0.0.197", None)
        self.assertIsNone(result)
        self.assertEqual(reached, [], "it reached Playwright despite refusing")

    def test_a_blind_check_plus_a_browser_user_agent_attaches(self):
        result, reached = self._attach(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", None)
        self.assertEqual(reached, [1], "a real browser was refused on its UA")

    def test_a_blind_check_with_no_user_agent_attaches(self):
        # Absent evidence is not evidence.
        _, reached = self._attach("", None)
        self.assertEqual(reached, [1])

    def test_a_positive_owner_check_ignores_the_user_agent(self):
        # The process evidence is stronger, and a custom --user-agent is a
        # thing people set.
        _, reached = self._attach("SomethingCustom/1.0", True)
        self.assertEqual(reached, [1],
                         "a UA check overrode a positive ownership answer")


class TestScanningFindsTheBrowsersPort(unittest.TestCase):
    """9222 is two conventions at once: Chrome's debug port and the WebView2
    default. When something else holds it, TENKA walks a few ports along rather
    than making the user discover and reconfigure around a squatter.

    `probe_one_port` and `cdp_owner_is_a_browser` are both stubbed, so nothing
    opens a socket or enumerates a window.
    """

    def setUp(self):
        cdp.reset_state_for_test()

    def tearDown(self):
        cdp.reset_state_for_test()

    def _scan(self, *, available: set, browsers: set, base=9222, span=4):
        """available: ports that answer. browsers: ports that are a browser."""
        self.probed: list[int] = []

        async def _probe(port, timeout=0.5, use_cache=True):
            self.probed.append(port)
            return cdp.CdpProbeResult(
                available=port in available, port=port,
                browser="Chrome/1" if port in available else "",
                error="" if port in available else "closed",
                probed_at=999999.0)

        return (patch.object(cdp, "probe_one_port", new=_probe),
                patch.object(cdp, "cdp_owner_is_a_browser",
                             side_effect=lambda p: (p in browsers, f"port {p}")),
                patch.object(cdp.config, "BROWSER_CDP_PORT", base),
                patch.object(cdp.config, "BROWSER_CDP_PORT_SCAN", span))

    def _run_scan(self, **kw):
        patches = self._scan(**kw)
        for p in patches:
            p.start()
        try:
            return asyncio.run(cdp.cdp_health_probe())
        finally:
            for p in reversed(patches):
                p.stop()

    def test_the_configured_port_is_used_when_it_is_a_browser(self):
        r = self._run_scan(available={9222}, browsers={9222})
        self.assertTrue(r.available)
        self.assertEqual(r.port, 9222)
        self.assertEqual(self.probed, [9222], "it kept scanning after a hit")

    def test_it_walks_past_a_squatter_to_the_real_browser(self):
        # The live shape: a webview answers on 9222, Chrome is on 9223.
        r = self._run_scan(available={9222, 9223}, browsers={9223})
        self.assertTrue(r.available)
        self.assertEqual(r.port, 9223)

    def test_it_walks_past_closed_ports(self):
        r = self._run_scan(available={9225}, browsers={9225})
        self.assertEqual(r.port, 9225)

    def test_the_span_counts_ports_past_the_configured_one(self):
        # span=4 means four *past* 9222, so 9222..9226 inclusive are probed.
        # Stated as an assertion because "how many ports" is exactly the kind
        # of off-by-one a reader assumes rather than checks -- the first draft
        # of this test assumed the other one.
        self._run_scan(available=set(), browsers=set())
        self.assertEqual(self.probed, [9222, 9223, 9224, 9225, 9226])

    def test_it_stops_at_the_configured_span(self):
        # 9227 is one past base+span and must not be probed, however tempting.
        r = self._run_scan(available={9227}, browsers={9227})
        self.assertFalse(r.available)
        self.assertNotIn(9227, self.probed)

    def test_a_span_of_zero_probes_only_the_configured_port(self):
        r = self._run_scan(available={9223}, browsers={9223}, span=0)
        self.assertEqual(self.probed, [9222])
        self.assertFalse(r.available)

    def test_nothing_usable_reports_the_configured_port(self):
        # The error a caller logs should be about the port the user set, not
        # about the last one tried.
        r = self._run_scan(available=set(), browsers=set())
        self.assertFalse(r.available)
        self.assertEqual(r.port, 9222)

    def test_a_rejected_port_is_not_reported_as_available(self):
        """The live failure of the first version of this scan.

        With a webview answering on 9222 and nothing on the rest, the scan
        rejected 9222 and then returned that same result as its fallback --
        `available=True`, for a port it had just refused. `_choose_browser_mode`
        reads `cdp_state_snapshot().available` to decide whether DOM-mode is on
        the table, so that answer routes a task to a tier that then refuses it.

        "Something answered" is not the question. "Is there a browser to drive"
        is.
        """
        r = self._run_scan(available={9222}, browsers=set())
        self.assertFalse(
            r.available,
            "a port that failed the browser check was reported as usable")

    def test_the_rejection_reason_survives_into_the_error(self):
        # "closed" sends someone looking for a browser that is not running.
        # "9222 (port 9222)" sends them to the port that is actually occupied.
        r = self._run_scan(available={9222}, browsers=set())
        self.assertIn("9222", r.error)
        self.assertIn("no browser found", r.error)

    def test_nothing_answering_says_so_differently(self):
        # The two failures want different words: nothing listening is a
        # different thing to something listening that is not a browser.
        r = self._run_scan(available=set(), browsers=set())
        self.assertNotIn("no browser found", r.error)
        self.assertIn("9222..9226", r.error)

    def test_a_port_that_answers_but_is_not_a_browser_is_skipped(self):
        r = self._run_scan(available={9222, 9224}, browsers={9224})
        self.assertEqual(r.port, 9224)

    def test_an_explicit_port_is_never_scanned_past(self):
        """`port=` means that port. A caller naming one is not asking to
        search, and searching on their behalf would drive a browser they did
        not mean."""
        patches = self._scan(available={9223}, browsers={9223})
        for p in patches:
            p.start()
        try:
            r = asyncio.run(cdp.cdp_health_probe(port=9222))
        finally:
            for p in reversed(patches):
                p.stop()
        self.assertEqual(self.probed, [9222])
        self.assertFalse(r.available)


class TestTheProbeAndTheConnectionAgreeOnThePort(unittest.TestCase):
    """The invariant KI-37's second round was about.

    A probe that passes for one socket while the connection opens another is
    how a green check ended up above the same wrong behaviour. The result
    carries the port it describes, and the attach uses that.
    """

    def setUp(self):
        cdp.reset_state_for_test()

    def tearDown(self):
        cdp.reset_state_for_test()

    def test_the_attach_uses_the_port_the_scan_settled_on(self):
        opened: list[str] = []

        async def _probe(port, timeout=0.5, use_cache=True):
            return cdp.CdpProbeResult(
                available=(port == 9224), port=port, browser="Chrome/1",
                ws_endpoint="ws://x", probed_at=999999.0)

        class _FakePw:
            async def start(self):
                return self

            class chromium:
                @staticmethod
                async def connect_over_cdp(url):
                    opened.append(url)
                    raise RuntimeError("stop here — the URL is the assertion")

            async def stop(self):
                pass

        fake_async = types.ModuleType("playwright.async_api")
        fake_async.async_playwright = lambda: _FakePw()
        with patch.object(cdp, "probe_one_port", new=_probe),              patch.object(cdp, "cdp_owner_is_a_browser",
                          return_value=(True, "a browser")),              patch.object(cdp.config, "BROWSER_CDP_PORT", 9222),              patch.object(cdp.config, "BROWSER_CDP_PORT_SCAN", 4),              patch.dict(sys.modules, {"playwright.async_api": fake_async}):
            asyncio.run(cdp.connect_to_existing_chrome())

        self.assertEqual(
            opened, ["http://127.0.0.1:9224"],
            "the attach opened a different port from the one the probe passed")

    def test_a_cached_result_for_another_port_is_not_reused(self):
        """The cache is keyed on the port, not on age alone.

        Without this, a probe of 9333 could be answered by a cached 9222
        result -- the same shape of mistake one level down.
        """
        cdp._cdp_state = cdp.CdpProbeResult(
            available=True, port=9222, browser="Chrome/1",
            probed_at=time.monotonic())

        calls: list[int] = []

        def _urlopen(req, timeout=None):
            calls.append(int(req.full_url.rsplit(":", 1)[1].split("/")[0]))
            raise urllib.error.URLError("refused")

        with patch("urllib.request.urlopen", side_effect=_urlopen):
            asyncio.run(cdp.probe_one_port(port=9333))
        self.assertEqual(calls, [9333], "a cached 9222 result answered for 9333")


if __name__ == "__main__":
    unittest.main(verbosity=2)
