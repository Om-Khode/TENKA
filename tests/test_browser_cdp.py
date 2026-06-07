"""
test_browser_cdp.py — Phase 1A: CDP attach helper.

Covers:
  - cdp_health_probe: closed port (cheap, no log spam), open port returning
    Chrome JSON, open port returning non-Chrome (cuckoo on 9222), HTTP
    non-200, malformed JSON, network timeout, cache TTL behavior
  - connect_to_existing_chrome: probe-says-no short-circuits, attach
    timeout, attach exception, success path with stub Playwright
  - get_or_attach_browser: prefer_cdp=False bypasses, config flag off
    bypasses, CDP available → cdp handle, CDP unavailable → bundled
    handle, dead cached attachment is dropped and re-probed, concurrent
    callers don't double-attach
  - detach: idempotent, leaves Chrome alive (browser.close called but
    that's connect_over_cdp's "disconnect" semantics), clears module state
  - cdp_state_snapshot: returns last probe result
  - Ownership: detach does NOT close user's contexts/pages

Run: python test_browser_cdp.py
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.browser.cdp as cdp
import assistant.config as cfg


def _run(coro):
    return asyncio.run(coro)


# ─── Stub helpers ────────────────────────────────────────────────────────


class _FakeResponse:
    """Mimic urlopen's return value with a `status` and `read()`."""
    def __init__(self, status: int, body: bytes):
        self.status = status
        self._body = body

    def read(self, n: int = -1) -> bytes:
        return self._body if n < 0 else self._body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_urlopen(*, status: int = 200, body: bytes = b'', side_effect=None):
    """Patch urllib.request.urlopen used inside cdp_health_probe's
    threaded helper. Returns the patcher (caller calls .stop())."""
    if side_effect is not None:
        return patch.object(urllib.request, "urlopen", side_effect=side_effect)
    return patch.object(
        urllib.request, "urlopen",
        return_value=_FakeResponse(status, body),
    )


def _chrome_json_version_body(browser_str: str = "Chrome/123.0.6312.86") -> bytes:
    return (
        b'{"Browser":"' + browser_str.encode() + b'",'
        b'"Protocol-Version":"1.3",'
        b'"User-Agent":"Mozilla/5.0 ...",'
        b'"V8-Version":"12.3.219.9",'
        b'"WebKit-Version":"537.36 (@abcdef)",'
        b'"webSocketDebuggerUrl":"ws://127.0.0.1:9222/devtools/browser/abc-123"}'
    )


# ─── cdp_health_probe ────────────────────────────────────────────────────


class TestCdpHealthProbe(unittest.TestCase):
    def setUp(self):
        cdp.reset_state_for_test()

    def tearDown(self):
        cdp.reset_state_for_test()

    def test_closed_port_returns_unavailable(self):
        with _patch_urlopen(side_effect=urllib.error.URLError("Connection refused")):
            result = _run(cdp.cdp_health_probe(port=9222, timeout=0.1))
        self.assertFalse(result.available)
        self.assertEqual(result.browser, "")
        self.assertIn("connection failed", result.error)

    def test_open_port_returning_chrome_json(self):
        with _patch_urlopen(status=200, body=_chrome_json_version_body()):
            result = _run(cdp.cdp_health_probe(port=9222, timeout=0.5))
        self.assertTrue(result.available)
        self.assertIn("Chrome", result.browser)
        self.assertTrue(result.ws_endpoint.startswith("ws://"))

    def test_open_port_returning_non_chrome_treated_as_unavailable(self):
        # Some random app holding port 9222 returning JSON that's not Chrome.
        with _patch_urlopen(
            status=200,
            body=b'{"Browser":"my-app/1.0","other":"junk"}',
        ):
            result = _run(cdp.cdp_health_probe(port=9222))
        self.assertFalse(result.available)
        self.assertIn("non-chromium", result.error)

    def test_http_non_200_treated_as_unavailable(self):
        with _patch_urlopen(status=404, body=b"not found"):
            result = _run(cdp.cdp_health_probe(port=9222))
        self.assertFalse(result.available)
        self.assertIn("http 404", result.error)

    def test_malformed_json_treated_as_unavailable(self):
        with _patch_urlopen(status=200, body=b"<html>not json</html>"):
            result = _run(cdp.cdp_health_probe(port=9222))
        self.assertFalse(result.available)
        self.assertIn("non-json", result.error)

    def test_network_timeout_treated_as_unavailable(self):
        with _patch_urlopen(side_effect=TimeoutError("timed out")):
            result = _run(cdp.cdp_health_probe(port=9222, timeout=0.1))
        self.assertFalse(result.available)

    def test_cache_returns_within_ttl(self):
        # First probe populates cache.
        with _patch_urlopen(status=200, body=_chrome_json_version_body()):
            first = _run(cdp.cdp_health_probe(port=9222))
        self.assertTrue(first.available)
        # Second probe within TTL must NOT hit the network — patch with a
        # side_effect that would fail if called.
        called = {"n": 0}
        def _spy(*args, **kw):
            called["n"] += 1
            return _FakeResponse(200, _chrome_json_version_body())
        with patch.object(urllib.request, "urlopen", side_effect=_spy):
            second = _run(cdp.cdp_health_probe(port=9222))
        self.assertEqual(called["n"], 0, "cache hit should skip the network call")
        self.assertEqual(first.probed_at, second.probed_at)

    def test_use_cache_false_bypasses_cache(self):
        with _patch_urlopen(status=200, body=_chrome_json_version_body()):
            _run(cdp.cdp_health_probe(port=9222))
        # use_cache=False should re-probe even within TTL.
        called = {"n": 0}
        def _spy(*args, **kw):
            called["n"] += 1
            return _FakeResponse(200, _chrome_json_version_body())
        with patch.object(urllib.request, "urlopen", side_effect=_spy):
            _run(cdp.cdp_health_probe(port=9222, use_cache=False))
        self.assertEqual(called["n"], 1)

    def test_cdp_state_snapshot_reflects_last_probe(self):
        self.assertIsNone(cdp.cdp_state_snapshot())
        with _patch_urlopen(status=200, body=_chrome_json_version_body()):
            _run(cdp.cdp_health_probe(port=9222))
        snap = cdp.cdp_state_snapshot()
        self.assertIsNotNone(snap)
        self.assertTrue(snap.available)


# ─── connect_to_existing_chrome ──────────────────────────────────────────


class _FakeBrowser:
    def __init__(self, contexts=None, connected=True):
        self._contexts = contexts or []
        self._connected = connected
        self.closed = False

    @property
    def contexts(self):
        return self._contexts

    def is_connected(self):
        return self._connected

    async def close(self):
        # connect_over_cdp's .close() is documented as "disconnect" — Chrome
        # stays alive. Our test just records the call.
        self.closed = True


class _FakePlaywright:
    """Minimal stand-in for the playwright session returned by start()."""
    def __init__(self, browser=None, connect_raises=None):
        self.chromium = MagicMock()
        if connect_raises is not None:
            self.chromium.connect_over_cdp = AsyncMock(side_effect=connect_raises)
        else:
            self.chromium.connect_over_cdp = AsyncMock(return_value=browser)
        self.stopped = False

    async def stop(self):
        self.stopped = True


class _FakeAsyncPlaywrightCtx:
    """Mimic async_playwright() context manager — has .start() returning a session."""
    def __init__(self, session):
        self._session = session

    async def start(self):
        return self._session


def _patch_playwright(session):
    """Patch the lazy import inside connect_to_existing_chrome.
    Returns a teardown closure."""
    fake_module = types.ModuleType("playwright.async_api")
    fake_module.async_playwright = lambda: _FakeAsyncPlaywrightCtx(session)
    sys.modules["playwright.async_api"] = fake_module
    return lambda: sys.modules.pop("playwright.async_api", None)


class TestConnectToExistingChrome(unittest.TestCase):
    def setUp(self):
        cdp.reset_state_for_test()

    def tearDown(self):
        cdp.reset_state_for_test()
        sys.modules.pop("playwright.async_api", None)

    def test_short_circuits_when_probe_unavailable(self):
        # Probe says no — must not try to import Playwright.
        with _patch_urlopen(side_effect=urllib.error.URLError("refused")):
            result = _run(cdp.connect_to_existing_chrome(port=9222))
        self.assertIsNone(result)

    def test_success_path_returns_attachment(self):
        fake_browser = _FakeBrowser(contexts=["ctx0", "ctx1"])
        teardown = _patch_playwright(_FakePlaywright(browser=fake_browser))
        try:
            with _patch_urlopen(status=200, body=_chrome_json_version_body()):
                attachment = _run(cdp.connect_to_existing_chrome(port=9222))
            self.assertIsNotNone(attachment)
            self.assertIs(attachment.browser, fake_browser)
            self.assertEqual(len(attachment.contexts), 2)
            self.assertEqual(attachment.port, 9222)
            self.assertTrue(attachment.ws_endpoint.startswith("ws://"))
        finally:
            teardown()

    def test_attach_exception_returns_none(self):
        # connect_over_cdp raises (e.g. version mismatch).
        teardown = _patch_playwright(
            _FakePlaywright(connect_raises=RuntimeError("version mismatch"))
        )
        try:
            with _patch_urlopen(status=200, body=_chrome_json_version_body()):
                result = _run(cdp.connect_to_existing_chrome(port=9222))
            self.assertIsNone(result)
        finally:
            teardown()

    def test_attach_timeout_returns_none(self):
        teardown = _patch_playwright(
            _FakePlaywright(connect_raises=asyncio.TimeoutError())
        )
        try:
            with _patch_urlopen(status=200, body=_chrome_json_version_body()):
                result = _run(cdp.connect_to_existing_chrome(port=9222, timeout=0.05))
            self.assertIsNone(result)
        finally:
            teardown()


# ─── get_or_attach_browser ───────────────────────────────────────────────


class TestGetOrAttachBrowser(unittest.TestCase):
    def setUp(self):
        cdp.reset_state_for_test()

    def tearDown(self):
        cdp.reset_state_for_test()
        cfg.BROWSER_PREFER_CDP = True
        sys.modules.pop("playwright.async_api", None)

    def _stub_browser_automation(self):
        """Install a fake browser_automation module so the bundled fallback
        path doesn't try to start a real Chromium."""
        fake_browser = MagicMock(name="bundled-chromium")
        fake_module = types.ModuleType("assistant.automation.browser.automation")
        fake_module.ensure_browser = AsyncMock(return_value=fake_browser)
        sys.modules["assistant.automation.browser.automation"] = fake_module
        return fake_browser, fake_module

    def test_prefer_cdp_false_bypasses_cdp(self):
        fake_browser, fake_mod = self._stub_browser_automation()
        try:
            handle = _run(cdp.get_or_attach_browser(prefer_cdp=False))
            self.assertEqual(handle.kind, "bundled")
            self.assertIs(handle.browser, fake_browser)
            self.assertIsNone(handle.attachment)
            fake_mod.ensure_browser.assert_awaited_once_with(headless=True)
        finally:
            sys.modules.pop("assistant.automation.browser.automation", None)

    def test_config_flag_off_bypasses_cdp(self):
        fake_browser, _ = self._stub_browser_automation()
        cfg.BROWSER_PREFER_CDP = False
        try:
            handle = _run(cdp.get_or_attach_browser(prefer_cdp=True))
            self.assertEqual(handle.kind, "bundled")
        finally:
            cfg.BROWSER_PREFER_CDP = True
            sys.modules.pop("assistant.automation.browser.automation", None)

    def test_cdp_available_returns_cdp_handle(self):
        fake_browser = _FakeBrowser(contexts=["ctx0"])
        teardown = _patch_playwright(_FakePlaywright(browser=fake_browser))
        try:
            with _patch_urlopen(status=200, body=_chrome_json_version_body()):
                handle = _run(cdp.get_or_attach_browser(prefer_cdp=True))
            self.assertEqual(handle.kind, "cdp")
            self.assertIs(handle.browser, fake_browser)
            self.assertIsNotNone(handle.attachment)
        finally:
            teardown()

    def test_cdp_unavailable_falls_back_to_bundled(self):
        fake_browser, _ = self._stub_browser_automation()
        try:
            with _patch_urlopen(side_effect=urllib.error.URLError("refused")):
                handle = _run(cdp.get_or_attach_browser(prefer_cdp=True))
            self.assertEqual(handle.kind, "bundled")
            self.assertIs(handle.browser, fake_browser)
        finally:
            sys.modules.pop("assistant.automation.browser.automation", None)

    def test_attach_failure_falls_back_to_bundled(self):
        # Probe succeeds but Playwright connect raises.
        fake_browser, _ = self._stub_browser_automation()
        teardown = _patch_playwright(
            _FakePlaywright(connect_raises=RuntimeError("DevTools open"))
        )
        try:
            with _patch_urlopen(status=200, body=_chrome_json_version_body()):
                handle = _run(cdp.get_or_attach_browser(prefer_cdp=True))
            self.assertEqual(handle.kind, "bundled")
        finally:
            teardown()
            sys.modules.pop("assistant.automation.browser.automation", None)

    def test_dead_cached_attachment_is_dropped(self):
        # Pre-populate _cdp_attachment with a dead one; get_or_attach should
        # detect via is_connected() and re-probe.
        fake_browser_alive = _FakeBrowser(contexts=["ctx-new"])
        cdp._cdp_attachment = cdp.CdpAttachment(
            browser=_FakeBrowser(connected=False),
            contexts=["stale"],
            ws_endpoint="ws://stale",
            attached_at=0.0,
            port=9222,
        )
        teardown = _patch_playwright(_FakePlaywright(browser=fake_browser_alive))
        try:
            with _patch_urlopen(status=200, body=_chrome_json_version_body()):
                handle = _run(cdp.get_or_attach_browser(prefer_cdp=True))
            self.assertEqual(handle.kind, "cdp")
            self.assertIs(handle.browser, fake_browser_alive)
        finally:
            teardown()

    def test_concurrent_callers_share_one_attachment(self):
        # Two concurrent get_or_attach_browser calls — only ONE
        # connect_over_cdp should be issued, thanks to _attach_lock.
        fake_browser = _FakeBrowser(contexts=[])
        session = _FakePlaywright(browser=fake_browser)
        teardown = _patch_playwright(session)
        try:
            with _patch_urlopen(status=200, body=_chrome_json_version_body()):
                async def _twice():
                    h1, h2 = await asyncio.gather(
                        cdp.get_or_attach_browser(prefer_cdp=True),
                        cdp.get_or_attach_browser(prefer_cdp=True),
                    )
                    return h1, h2
                h1, h2 = _run(_twice())
            self.assertEqual(h1.kind, "cdp")
            self.assertEqual(h2.kind, "cdp")
            self.assertIs(h1.browser, h2.browser)
            self.assertEqual(session.chromium.connect_over_cdp.await_count, 1)
        finally:
            teardown()


# ─── detach ──────────────────────────────────────────────────────────────


class TestDetach(unittest.TestCase):
    def setUp(self):
        cdp.reset_state_for_test()

    def tearDown(self):
        cdp.reset_state_for_test()

    def test_idempotent_when_no_attachment(self):
        # Should not raise.
        _run(cdp.detach())
        self.assertIsNone(cdp._cdp_attachment)

    def test_detach_clears_module_state(self):
        fake_browser = _FakeBrowser()
        fake_pw = _FakePlaywright()
        attachment = cdp.CdpAttachment(
            browser=fake_browser, contexts=[], ws_endpoint="", attached_at=0.0, port=9222,
        )
        object.__setattr__(attachment, "_pw", fake_pw)
        cdp._cdp_attachment = attachment

        _run(cdp.detach())

        self.assertIsNone(cdp._cdp_attachment)
        # Browser.close was called (which is connect_over_cdp's "disconnect")
        self.assertTrue(fake_browser.closed)
        # Playwright session was stopped
        self.assertTrue(fake_pw.stopped)

    def test_detach_does_not_close_user_contexts(self):
        # Critical ownership rule: detach must NEVER close contexts/pages
        # that belong to the user. We check by instrumenting fake context.close.
        ctx = MagicMock()
        ctx.close = AsyncMock()
        fake_browser = _FakeBrowser(contexts=[ctx])
        attachment = cdp.CdpAttachment(
            browser=fake_browser, contexts=[ctx], ws_endpoint="", attached_at=0.0, port=9222,
        )
        object.__setattr__(attachment, "_pw", _FakePlaywright())
        cdp._cdp_attachment = attachment

        _run(cdp.detach())

        ctx.close.assert_not_called()


# ─── Config / module-level integration ───────────────────────────────────


class TestConfigIntegration(unittest.TestCase):
    def test_config_flags_present(self):
        # Phase 1A flags must be defined in config so callers can rely on them.
        self.assertTrue(hasattr(cfg, "BROWSER_PREFER_CDP"))
        self.assertTrue(hasattr(cfg, "BROWSER_CDP_PORT"))
        self.assertTrue(hasattr(cfg, "BROWSER_CDP_PROBE_TTL"))
        self.assertEqual(cfg.BROWSER_CDP_PORT, 9222)


if __name__ == "__main__":
    unittest.main(verbosity=2)
