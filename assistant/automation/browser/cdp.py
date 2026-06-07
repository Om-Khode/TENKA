"""
browser_cdp.py — Chrome CDP attach for TENKA.

Lets TENKA drive a user-opened Chrome window (forms, logins, etc.) instead of
always launching its own bundled Chromium. Required for the DOM-aware planner
mode to be useful — without CDP attach we can't operate on the tabs the user
already has open.

Architecture (load-bearing):
  - This module's CDP-attached browser lives in `_cdp_attachment` — a SEPARATE
    global from browser_automation.py's `_browser` / `_pages` singletons.
  - We NEVER take ownership of contexts or pages we didn't open. Cleanup
    paths in browser_automation.py iterate only `_pages` THEY created.
  - `detach()` calls `playwright.disconnect()` only — Chrome stays alive.

Pure I/O. No business logic above the attach mechanics. The DOM planner,
routing decisions, and session orchestration all live elsewhere and consume
this module via the public API:

  cdp_health_probe(port, timeout)      — cheap HTTP GET, never raises
  connect_to_existing_chrome(port)     — Playwright connect_over_cdp wrapper
  get_or_attach_browser(prefer_cdp)    — single entry point for callers
  detach()                             — disconnect cleanly, leaves Chrome alive
  cdp_state_snapshot()                 — read the cached probe result

Failure modes handled:
  - Port closed (Chrome not started with flag) → silent fallback to bundled
  - Port in use by non-Chrome → fallback, log a WARNING
  - Chrome version mismatch → fallback, log WARNING with both versions
  - DevTools open hogging the port → probe succeeds but attach fails → fallback
  - User closes Chrome mid-task → CdpDetachedError raised by callers
"""

from __future__ import annotations

import asyncio
import logging
import time
import urllib.request
import urllib.error
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from ... import config

logger = logging.getLogger("browser_cdp")


# ─── Result types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CdpProbeResult:
    """
    Outcome of a single HTTP probe to Chrome's /json/version endpoint.

    `available` is the only field callers normally inspect. The others are
    diagnostic — useful for logging, version-mismatch detection, and the
    DevTools-open heuristic ("probe succeeded but attach failed").
    """
    available: bool
    browser: str = ""           # e.g. "Chrome/123.0.6312.86"
    ws_endpoint: str = ""       # webSocketDebuggerUrl from /json/version
    error: str = ""             # short reason when not available
    probed_at: float = 0.0      # time.monotonic() when this probe ran


@dataclass
class CdpAttachment:
    """
    Live Playwright handle on a CDP-attached browser. Held only while a
    task is using it. `attached_at` is for telemetry / reattach decisions.
    """
    browser: Any                # playwright.async_api.Browser (after connect_over_cdp)
    contexts: list = field(default_factory=list)  # snapshot at attach time; pages live in browser.contexts
    ws_endpoint: str = ""
    attached_at: float = 0.0
    port: int = 9222


@dataclass
class BrowserHandle:
    """
    What `get_or_attach_browser` returns. The caller checks `kind` to know
    which path it got and whom to consult for pages.
    """
    kind: str                   # "cdp" or "bundled"
    browser: Any                # Playwright Browser (CDP-attached or bundled-launched)
    attachment: Optional[CdpAttachment] = None  # populated only when kind == "cdp"


class CdpDetachedError(RuntimeError):
    """Raised by callers when an action discovers the CDP attachment is dead
    (e.g. user closed Chrome mid-task). Caught one level up to fall back to
    bundled Chromium / vision-loop."""


# ─── Module-level cache (probe TTL) ───────────────────────────────────────────

_cdp_state: Optional[CdpProbeResult] = None
_cdp_attachment: Optional[CdpAttachment] = None
# Lock so concurrent task entries don't double-probe / double-attach.
_attach_lock = asyncio.Lock()


def cdp_state_snapshot() -> Optional[CdpProbeResult]:
    """Read-only view of the last probe result. None if never probed.

    Callers MUST NOT mutate the return value. Used by the routing decision
    in desktop_automation.py to ask "is CDP available right now without me
    paying for a fresh probe."
    """
    return _cdp_state


# ─── HTTP probe ───────────────────────────────────────────────────────────────


async def cdp_health_probe(
    port: Optional[int] = None,
    timeout: float = 0.5,
    *,
    use_cache: bool = True,
) -> CdpProbeResult:
    """
    Cheap HTTP GET on `http://127.0.0.1:<port>/json/version`. Returns a
    `CdpProbeResult`. NEVER raises.

    When `use_cache=True` (default) and a recent probe is cached within
    `BROWSER_CDP_PROBE_TTL`, returns the cached result without hitting
    the network.

    Cost in the closed-port case: ~5-10ms (TCP RST). Open-port case:
    ~20-40ms (HTTP round-trip + JSON parse). The cache is what keeps
    per-task overhead at zero on the steady state.
    """
    global _cdp_state

    p = port if port is not None else int(getattr(config, "BROWSER_CDP_PORT", 9222))
    ttl = float(getattr(config, "BROWSER_CDP_PROBE_TTL", 30.0))

    # Cache hit: return without probing.
    if use_cache and _cdp_state is not None:
        age = time.monotonic() - _cdp_state.probed_at
        if age < ttl:
            return _cdp_state

    url = f"http://127.0.0.1:{p}/json/version"
    now = time.monotonic()

    # urllib is intentional — this runs at warmup time and during routing
    # decisions on the hot path. Pulling in aiohttp/httpx for one GET would
    # add 40+ ms of import cost. We run it in a thread to keep the event
    # loop free.
    def _do_request() -> tuple[bool, str, str, str]:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    return False, "", "", f"http {resp.status}"
                body = resp.read(8192)  # /json/version is tiny; 8KB is plenty
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                return False, "", "", "non-json response (port held by something else)"
            browser = str(data.get("Browser", "") or "")
            ws_endpoint = str(data.get("webSocketDebuggerUrl", "") or "")
            # Sanity check: Chrome's response always identifies as Chrome/Chromium/Edg/Brave.
            # Anything else is a cuckoo on port 9222 — treat as unavailable.
            if not browser:
                return False, "", "", "missing Browser field"
            lower = browser.lower()
            if not any(tag in lower for tag in ("chrome", "chromium", "edg", "brave")):
                return False, browser, ws_endpoint, "non-chromium browser on port"
            return True, browser, ws_endpoint, ""
        except urllib.error.URLError as e:
            return False, "", "", f"connection failed: {e.reason}"
        except (TimeoutError, OSError) as e:
            return False, "", "", f"network error: {e}"
        except Exception as e:
            return False, "", "", f"unexpected: {type(e).__name__}: {e}"

    available, browser, ws_endpoint, error = await asyncio.to_thread(_do_request)
    result = CdpProbeResult(
        available=available,
        browser=browser,
        ws_endpoint=ws_endpoint,
        error=error,
        probed_at=now,
    )
    _cdp_state = result

    # Single-line log per probe — easy to grep. Probes are TTL-cached
    # (default 30s) so this fires at most a handful of times per session.
    # We log all three outcomes at INFO level so the user can verify CDP
    # status from the log without needing DEBUG mode — important because
    # "wrong Chrome (no flag)" is the #1 reason DOM-mode silently falls
    # back to vision-loop. Closed-port log is laconic on purpose.
    if available:
        logger.info(f"[CDP] probe OK port={p} browser={browser!r}")
    elif error and "connection failed" not in error:
        # Interesting failures: HTTP non-200, non-Chrome cuckoo, malformed JSON
        logger.warning(f"[CDP] probe FAIL port={p} reason={error!r}")
    else:
        # Common closed-port case (Chrome not launched with --remote-debugging-port).
        # Single line, INFO level so users can grep `[CDP]` to confirm status.
        logger.info(f"[CDP] probe unavailable port={p} (chrome not launched with --remote-debugging-port?)")
    return result


# ─── Attach ───────────────────────────────────────────────────────────────────


async def connect_to_existing_chrome(
    port: Optional[int] = None,
    *,
    timeout: float = 5.0,
) -> Optional[CdpAttachment]:
    """
    Attempt Playwright `chromium.connect_over_cdp()`. Returns a
    `CdpAttachment` on success, or None on any failure.

    Uses the cached probe result to short-circuit when the port is known
    closed. Probe-then-attach can race (DevTools opens between probe and
    attach), so we still return None and log a WARNING when the attach
    itself fails despite a prior available probe.

    Caller is responsible for tracking the attachment's lifetime — this
    module's `_cdp_attachment` is set so subsequent callers can find it,
    but the caller should `detach()` when done (or rely on
    `cleanup_on_exit`).
    """
    global _cdp_attachment

    p = port if port is not None else int(getattr(config, "BROWSER_CDP_PORT", 9222))

    # Cheap pre-flight: don't even try to import Playwright if probe says no.
    probe = await cdp_health_probe(port=p)
    if not probe.available:
        logger.info(f"[CDP] attach skipped — probe says unavailable ({probe.error!r})")
        return None

    try:
        # Defer import so non-CDP code paths don't pay the cost.
        from playwright.async_api import async_playwright
    except ImportError:
        logger.warning("[CDP] Playwright not installed — cannot attach")
        return None

    # connect_over_cdp wants an HTTP URL; Playwright resolves to the WS itself.
    cdp_http_url = f"http://127.0.0.1:{p}"

    # We need a Playwright session to call connect_over_cdp. We do NOT share
    # browser_automation._playwright — that module's session is bound to its
    # own bundled Chromium lifecycle. CDP attach gets its own fresh session
    # so cleanup of one doesn't affect the other. Cost: ~1 extra node.exe
    # process when CDP is in use; negligible.
    pw = None
    try:
        pw_ctx = async_playwright()
        pw = await asyncio.wait_for(pw_ctx.start(), timeout=timeout)
        browser = await asyncio.wait_for(
            pw.chromium.connect_over_cdp(cdp_http_url), timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.warning(
            f"[CDP] connect_over_cdp TIMEOUT after {timeout}s on port {p}. "
            f"Possible cause: DevTools open on a tab in this Chrome (only one "
            f"CDP client per port). Close DevTools and retry, or use bundled."
        )
        if pw is not None:
            try:
                await pw.stop()
            except Exception:
                pass
        return None
    except Exception as e:
        logger.warning(
            f"[CDP] connect_over_cdp FAILED: {type(e).__name__}: {e}. "
            f"probe.browser={probe.browser!r} — likely Chrome/Playwright "
            f"version mismatch. Falling back to bundled Chromium."
        )
        if pw is not None:
            try:
                await pw.stop()
            except Exception:
                pass
        return None

    # Snapshot contexts at attach time. We do NOT close them on detach —
    # they belong to the user, not us.
    contexts = list(getattr(browser, "contexts", []) or [])
    attachment = CdpAttachment(
        browser=browser,
        contexts=contexts,
        ws_endpoint=probe.ws_endpoint,
        attached_at=time.monotonic(),
        port=p,
    )
    # Stash for cleanup. Note: we also stash `pw` on the attachment via a
    # private attr so detach can stop the playwright instance. Keeping it
    # off the dataclass schema avoids leaking it into telemetry.
    object.__setattr__(attachment, "_pw", pw)
    _cdp_attachment = attachment

    page_count = sum(len(getattr(c, "pages", []) or []) for c in contexts)
    logger.info(
        f"[CDP] attached port={p} browser={probe.browser!r} "
        f"contexts={len(contexts)} pages={page_count}"
    )
    return attachment


# ─── Single entry point for callers ───────────────────────────────────────────


async def get_or_attach_browser(*, prefer_cdp: bool = True) -> BrowserHandle:
    """
    The main entry point. Return a `BrowserHandle` describing which browser
    the caller should use:

      handle.kind == "cdp"     → user's Chrome via CDP attach
      handle.kind == "bundled" → TENKA's own Chromium (existing behaviour)

    Decision tree:
      1. If `prefer_cdp` is False → always bundled.
      2. If config.BROWSER_PREFER_CDP is False → always bundled.
      3. Probe CDP. If available → connect_to_existing_chrome.
         If attach succeeds → return CDP handle.
         If attach fails → fall through to bundled with one INFO log.
      4. Bundled: delegate to browser_automation.ensure_browser(headless=True).

    Holds a lock around the attach attempt so concurrent task entries
    don't try to open two CDP connections to the same Chrome.
    """
    global _cdp_attachment

    cfg_prefer = bool(getattr(config, "BROWSER_PREFER_CDP", True))
    if not prefer_cdp or not cfg_prefer:
        return await _bundled_handle()

    async with _attach_lock:
        # Reuse an existing attachment if it's still alive. Cheap check:
        # `browser.is_connected()` returns False if Chrome closed.
        if _cdp_attachment is not None:
            try:
                if _cdp_attachment.browser.is_connected():
                    return BrowserHandle(
                        kind="cdp",
                        browser=_cdp_attachment.browser,
                        attachment=_cdp_attachment,
                    )
                # Stale attachment — Chrome closed since we attached. Drop it.
                logger.info("[CDP] cached attachment is dead — re-probing")
                _cdp_attachment = None
            except Exception:
                _cdp_attachment = None

        attachment = await connect_to_existing_chrome()
        if attachment is not None:
            return BrowserHandle(
                kind="cdp",
                browser=attachment.browser,
                attachment=attachment,
            )

    # Fall through to bundled. One log line so we know why DOM-mode isn't
    # using the user's Chrome.
    logger.info("[CDP] using bundled Chromium (CDP unavailable or attach failed)")
    return await _bundled_handle()


async def _bundled_handle() -> BrowserHandle:
    """Delegate to the bundled-Chromium path. Imports browser_automation
    lazily to avoid circular import at module load time."""
    from . import automation as browser_automation
    browser = await browser_automation.ensure_browser(headless=True)
    return BrowserHandle(kind="bundled", browser=browser, attachment=None)


# ─── Detach / cleanup ─────────────────────────────────────────────────────────


async def detach() -> None:
    """
    Disconnect from the user's Chrome. Chrome STAYS ALIVE — we only stop
    the Playwright session that was attached to it.

    Safe to call multiple times; no-op when no attachment exists.
    Never closes contexts or pages — those belong to the user.
    """
    global _cdp_attachment
    if _cdp_attachment is None:
        return

    attachment = _cdp_attachment
    _cdp_attachment = None  # Release first so a concurrent caller doesn't reuse.

    try:
        # Disconnect — sends CDP detach, leaves Chrome alone.
        if hasattr(attachment.browser, "close"):
            # Playwright's connect_over_cdp Browser uses .close() to disconnect.
            # This is documented as "disconnects without closing the browser"
            # for connect_over_cdp specifically. (Calling .close() on a
            # browser we LAUNCHED would close it; on connect_over_cdp it
            # only severs the connection.)
            try:
                await attachment.browser.close()
            except Exception as e:
                logger.debug(f"[CDP] browser.close() during detach: {e}")
    finally:
        pw = getattr(attachment, "_pw", None)
        if pw is not None:
            try:
                await pw.stop()
            except Exception as e:
                logger.debug(f"[CDP] playwright.stop() during detach: {e}")

    logger.info(f"[CDP] detached from port={attachment.port} (Chrome left running)")


def reset_state_for_test() -> None:
    """
    Test helper: clear module-level state. Should NEVER be called outside
    tests. Intentionally synchronous so it's awkward to call from prod
    async code paths.
    """
    global _cdp_state, _cdp_attachment
    _cdp_state = None
    _cdp_attachment = None
