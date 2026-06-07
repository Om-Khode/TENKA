"""
browser_automation.py — Playwright-based browser automation for TENKA.

Manages a singleton Chromium browser instance (bundled, not the user's browser).
Supports headless (background tasks) and headed (visible) modes.

Part of the Desktop Automation Layer.
"""

import asyncio
import logging
import os
import re
from pathlib import Path

from ... import config
from typing import Optional, Any, Dict, List

logger = logging.getLogger("browser_automation")

# ── Playwright cache isolation ────────────────────────────────────────────────
# Without this, TENKA and any other Playwright instance on the same
# machine (e.g. IDE-side @playwright/mcp servers) fight over the shared
# ~/AppData/Local/ms-playwright cache directory, causing async_playwright().start()
# to hang indefinitely on driver locks.
#
# We point Playwright at our own sandbox-scoped directory. setdefault means a
# user-supplied PLAYWRIGHT_BROWSERS_PATH in .env still wins.
_PLAYWRIGHT_CACHE = Path(config.SANDBOX_DIR) / "browser-cache"
try:
    _PLAYWRIGHT_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(_PLAYWRIGHT_CACHE))
except Exception as _e:
    logger.warning(f"[BROWSER] Could not isolate Playwright cache at {_PLAYWRIGHT_CACHE}: {_e}")

_ACTIVE_PLAYWRIGHT_PATH = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")


def _chromium_installed() -> bool:
    """Check if any chromium build is present in the active Playwright cache.
    Returns False if the cache is empty (first run) — caller should hint at
    `playwright install chromium` with the env var set."""
    if not _ACTIVE_PLAYWRIGHT_PATH:
        return True  # using system default — let Playwright handle it
    p = Path(_ACTIVE_PLAYWRIGHT_PATH)
    if not p.exists():
        return False
    return any(child.name.startswith("chromium") for child in p.iterdir() if child.is_dir())


# Lazy-loaded — Playwright is optional
_playwright = None
_browser = None
_browser_headless = True
_pages: list = []
_MAX_PAGES = 5
_planner_page = None
_planner_context = None

try:
    from playwright.async_api import async_playwright, Browser, Page, Playwright, Error as PlaywrightError
    PLAYWRIGHT_AVAILABLE = True
    logger.info(f"[BROWSER] Playwright cache: {_ACTIVE_PLAYWRIGHT_PATH or '(system default)'}")
    if _ACTIVE_PLAYWRIGHT_PATH and not _chromium_installed():
        logger.warning(
            f"[BROWSER] No Chromium found in isolated cache: {_ACTIVE_PLAYWRIGHT_PATH}\n"
            f"    First-time setup needed. Run this once:\n"
            f"      PowerShell:\n"
            f"        $env:PLAYWRIGHT_BROWSERS_PATH = \"{_ACTIVE_PLAYWRIGHT_PATH}\"\n"
            f"        python -m playwright install chromium\n"
            f"      cmd.exe:\n"
            f"        set PLAYWRIGHT_BROWSERS_PATH={_ACTIVE_PLAYWRIGHT_PATH}\n"
            f"        python -m playwright install chromium\n"
            f"    Note: in PowerShell, plain `set NAME=value` only sets a shell variable,\n"
            f"    not an environment variable — child processes won't see it.\n"
            f"    Until installed, browser tasks will fail with a launch error."
        )
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("[BROWSER] playwright not installed — pip install playwright && playwright install chromium")

async def ensure_browser(headless: bool = True) -> "Browser":
    """
    Launch or return existing Playwright browser.
    If current browser has different headless mode, close and relaunch.
    """
    if not PLAYWRIGHT_AVAILABLE:
        raise RuntimeError("Playwright is not available. Please install it with `pip install playwright`.")

    global _playwright, _browser, _browser_headless

    if _browser is not None:
        if _browser.is_connected() and _browser_headless == headless:
            return _browser
        else:
            logger.info(f"[BROWSER] Re-launching browser (headless={headless})...")
            await close_browser()

    if _playwright is None:
        logger.info("[BROWSER] Starting Playwright driver...")
        try:
            # Fail loud after 30s instead of hanging forever. Common cause: a
            # parallel Playwright instance (e.g. an IDE's @playwright/mcp node
            # process) holds a lock on the shared ms-playwright driver/profile
            # cache. Kill stale node.exe Playwright drivers in Task Manager,
            # or run `playwright install --force chromium` to restore the cache.
            _playwright = await asyncio.wait_for(async_playwright().start(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.error(
                "[BROWSER] Playwright driver startup timed out after 30s. "
                "Likely cause: another Playwright instance is holding the driver lock. "
                "Check Task Manager for node.exe processes running '@playwright/mcp' "
                "or stale playwright drivers, then retry."
            )
            raise RuntimeError(
                "Playwright driver startup timed out — another Playwright instance "
                "(IDE MCP, stale process) is likely holding the driver lock."
            )
        logger.info("[BROWSER] Playwright driver started")

    logger.info(f"[BROWSER] Launching Chromium (headless={headless})...")
    # Standard Chrome user agent to avoid looking like a bot
    try:
        _browser = await asyncio.wait_for(
            _playwright.chromium.launch(
                headless=headless,
                args=["--disable-blink-features=AutomationControlled"],
            ),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        logger.error("[BROWSER] Chromium launch timed out after 30s")
        raise RuntimeError("Chromium launch timed out — see logs for diagnosis hints.")
    _browser_headless = headless
    return _browser


def _diagnose_runtime() -> str:
    """Capture process state right before driver spawn — surfaces likely
    causes (wrong asyncio loop policy, thread count, sys.platform, etc.)
    when the spawn hangs. Best-effort, never raises."""
    try:
        import sys, threading, platform
        try:
            loop = asyncio.get_running_loop()
            loop_name = type(loop).__name__
        except RuntimeError:
            loop_name = "no-running-loop"
        try:
            policy = asyncio.get_event_loop_policy()
            policy_name = type(policy).__name__
        except Exception:
            policy_name = "unknown"
        thread_names = [t.name for t in threading.enumerate()]
        return (
            f"py={sys.version.split()[0]} platform={platform.system()} "
            f"loop={loop_name} policy={policy_name} "
            f"threads({len(thread_names)})={thread_names[:12]}"
        )
    except Exception as e:
        return f"diagnose_failed: {e}"


async def warmup_driver(timeout: float = 30.0) -> bool:
    """Eagerly start the Playwright driver at process startup.

    Returns True if driver is ready (or a prior call already started it),
    False on timeout / Playwright unavailable. Never raises.
    """
    global _playwright
    if not PLAYWRIGHT_AVAILABLE:
        logger.warning("[BROWSER] warmup_driver: Playwright not installed")
        return False
    if _playwright is not None:
        logger.info("[BROWSER] warmup_driver: already started, no-op")
        return True
    logger.info(f"[BROWSER] warmup_driver: starting. Runtime diag: {_diagnose_runtime()}")
    try:
        # Split the call so we know WHICH step hangs:
        #   _ctx_mgr = async_playwright()  — sync, instantiates the manager
        #   await _ctx_mgr.start()          — async, spawns node.exe driver
        logger.info("[BROWSER] warmup_driver: instantiating async_playwright()...")
        _ctx_mgr = async_playwright()
        logger.info("[BROWSER] warmup_driver: calling .start() (spawns node.exe driver)...")
        _playwright = await asyncio.wait_for(_ctx_mgr.start(), timeout=timeout)
        logger.info("[BROWSER] warmup_driver: SUCCESS — driver is up")
        return True
    except asyncio.TimeoutError:
        logger.warning(
            f"[BROWSER] warmup_driver: TIMEOUT after {timeout}s on .start(). "
            f"Runtime diag at timeout: {_diagnose_runtime()}. "
            "If loop != ProactorEventLoop on Windows, that is the likely cause "
            "(Playwright requires Proactor for subprocess IPC)."
        )
        return False
    except Exception as e:
        logger.warning(f"[BROWSER] warmup_driver: FAILED with {type(e).__name__}: {e}")
        return False


async def _evict_oldest_page():
    """Close the oldest non-planner page when _pages exceeds _MAX_PAGES."""
    global _pages
    for i, page in enumerate(_pages):
        if page is not _planner_page:
            _pages.pop(i)
            try:
                await page.context.close()
            except Exception as e:
                logger.debug(f"[BROWSER] Error closing evicted page: {e}")
            return


async def get_page(url: str, wait_until: str = "domcontentloaded") -> "Page":
    """
    Navigate to URL in a new page. Returns the Page object.
    Caller is responsible for closing the page when done.
    wait_until: "domcontentloaded", "load", "networkidle"
    """
    _effective_headless = _browser_headless if _browser is not None and _browser.is_connected() else True
    browser = await ensure_browser(headless=_effective_headless)
    from ...core.geolocation import get_cached_region
    _region = get_cached_region()
    _locale = f"en-{_region['country_code']}" if _region.get("country_code") else None
    _tz = _region.get("timezone") or None
    _ctx_kwargs: dict = {"user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    if _locale:
        _ctx_kwargs["locale"] = _locale
    if _tz:
        _ctx_kwargs["timezone_id"] = _tz
    context = await browser.new_context(**_ctx_kwargs)
    page = await context.new_page()

    global _pages
    _pages.append(page)
    if len(_pages) > _MAX_PAGES:
        await _evict_oldest_page()

    url = re.sub(r'[\]\)\}>]+$', '', url.strip())
    logger.info(f"[BROWSER] Navigating to {url} (wait_until={wait_until})")
    try:
        await page.goto(url, wait_until=wait_until, timeout=30000)
    except PlaywrightError as e:
        logger.error(f"[BROWSER] Navigation to {url} failed: {e}")
        await page.context.close()
        if page in _pages:
            _pages.remove(page)
        raise

    return page

async def extract_text(url: str) -> str:
    """
    Navigate to URL and extract all visible text content.
    Returns the text. Closes the page automatically.
    """
    if not PLAYWRIGHT_AVAILABLE:
        return "Error: Playwright is not available."
    
    page = None
    try:
        page = await get_page(url, wait_until="domcontentloaded")
        # Evaluate script to get visible text loosely
        text = await page.evaluate("document.body.innerText")
        return text.strip() if text else ""
    except Exception as e:
        logger.error(f"[BROWSER] Extract text error: {e}")
        return f"Error extracting text: {e}"
    finally:
        if page:
            if page in _pages:
                _pages.remove(page)
            try:
                await page.context.close()
            except Exception:
                pass

async def extract_structured(url: str, selectors: Dict[str, str]) -> Dict[str, str]:
    """
    Navigate to URL and extract text from specific CSS selectors.
    selectors: {"price": ".product-price", "title": "h1.title"}
    """
    if not PLAYWRIGHT_AVAILABLE:
        return {"error": "Playwright is not available."}

    page = None
    results = {}
    try:
        page = await get_page(url, wait_until="domcontentloaded")
        for key, selector in selectors.items():
            elements = await page.locator(selector).all_inner_texts()
            results[key] = " ".join(elements).strip() if elements else ""
        return results
    except Exception as e:
        logger.error(f"[BROWSER] Extract structured error: {e}")
        return {"error": str(e)}
    finally:
        if page:
            if page in _pages:
                _pages.remove(page)
            try:
                await page.context.close()
            except Exception:
                pass

async def fill_and_submit(url: str, fields: List[Dict], submit_selector: str = None) -> str:
    """
    Navigate to URL, fill form fields, optionally submit.
    """
    if not PLAYWRIGHT_AVAILABLE:
        return "Error: Playwright is not available."

    page = None
    try:
        page = await get_page(url, wait_until="load")
        
        for field in fields:
            selector = field.get("selector")
            action = field.get("action", "fill")
            value = field.get("value")
            
            if not selector:
                continue
                
            locator = page.locator(selector).first
            
            if action == "fill":
                await locator.fill(str(value), timeout=10000)
            elif action == "check":
                await locator.check(timeout=10000)
            elif action == "uncheck":
                await locator.uncheck(timeout=10000)
            elif action == "select":
                await locator.select_option(str(value), timeout=10000)
            elif action == "click":
                await locator.click(timeout=10000)
                
        if submit_selector:
            await page.locator(submit_selector).first.click(timeout=10000)
            await page.wait_for_load_state("networkidle", timeout=15000)
            
        text = await page.evaluate("document.body.innerText")
        return text.strip() if text else "Action completed, no text found."
    except Exception as e:
        logger.error(f"[BROWSER] Fill and submit error: {e}")
        return f"Error during fill and submit: {e}"
    finally:
        if page:
            if page in _pages:
                _pages.remove(page)
            try:
                await page.context.close()
            except Exception:
                pass

async def click_and_wait(url: str, click_selector: str, wait_selector: str = None) -> str:
    """
    Navigate to URL, click an element, optionally wait for another element.
    """
    if not PLAYWRIGHT_AVAILABLE:
        return "Error: Playwright is not available."

    page = None
    try:
        page = await get_page(url, wait_until="load")
        await page.locator(click_selector).first.click(timeout=10000)
        
        if wait_selector:
            await page.locator(wait_selector).first.wait_for(state="visible", timeout=15000)
        else:
            await page.wait_for_load_state("domcontentloaded", timeout=15000)
            
        text = await page.evaluate("document.body.innerText")
        return text.strip() if text else "Action completed, no text found."
    except Exception as e:
        logger.error(f"[BROWSER] Click and wait error: {e}")
        return f"Error during click and wait: {e}"
    finally:
        if page:
            if page in _pages:
                _pages.remove(page)
            try:
                await page.context.close()
            except Exception:
                pass

_CONSENT_SELECTORS = [
    "#onetrust-accept-btn-handler",
    "#onetrust-close-btn-container button",
    "[id*='onetrust'] button[id*='accept'], [id*='onetrust'] button.onetrust-close-btn-handler",
    "[id*='cookie'] button[id*='accept'], [id*='cookie'] button[id*='agree']",
    "[class*='consent'] button[class*='accept'], [class*='consent'] button[class*='agree']",
    "[id*='gdpr'] button[id*='accept']",
    "button[id*='cookie-accept'], button[id*='cookieAccept']",
    "[class*='cookie-banner'] button, [class*='cookie-notice'] button",
]


async def _dismiss_consent_banner(page) -> None:
    """Try to dismiss cookie/consent banners after navigation. Generic — no site-specific logic."""
    try:
        for selector in _CONSENT_SELECTORS:
            btn = page.locator(selector).first
            if await btn.is_visible(timeout=2500):
                await btn.click(timeout=3000)
                logger.info(f"[BROWSER] Dismissed consent banner via: {selector}")
                await page.wait_for_timeout(500)
                return
        removed = await page.evaluate("""() => {
            let removed = false;
            for (const sel of ['#onetrust-consent-sdk', '[class*="cookie-banner"]', '[class*="cookie-notice"]', '[id*="gdpr"]']) {
                const el = document.querySelector(sel);
                if (el) { el.remove(); removed = true; }
            }
            return removed;
        }""")
        if removed:
            logger.info("[BROWSER] Removed consent/cookie overlay via JS fallback")
    except Exception:
        pass


async def run_browser_steps(steps: List[Dict], *, _from_planner: bool = False, headless: bool = False) -> str:
    """
    Execute a sequence of browser actions.
    Keeps page open across steps, closes at the end.

    Each step is wrapped in pre_check → execute → post_verify. On a
    confident verification failure we short-circuit and return a structured
    string so the caller (planner / procedure_executor) can self-heal or
    surface to the user. Ambiguous results are escalated to a vision call.
    """
    if not PLAYWRIGHT_AVAILABLE:
        return "Error: Playwright is not available."

    from .. import verification

    global _planner_page, _planner_context, _pages
    page = None
    results = []

    try:
        # Clean up stale planner page before anything else
        if _from_planner and _planner_page is not None and _planner_page.is_closed():
            logger.warning("[BROWSER] Planner page is closed (crashed?), cleaning up")
            if _planner_page in _pages:
                _pages.remove(_planner_page)
            if _planner_context:
                try:
                    await _planner_context.close()
                except Exception:
                    pass
            _planner_page = None
            _planner_context = None

        browser = await ensure_browser(headless=headless)

        for i, step in enumerate(steps):
            action = step.get("action")
            params = step.get("params", {})
            verify_step = {"type": "browser", "action": action, "params": params}
            logger.info(f"[BROWSER] Step {i+1}: {action} - {params}")

            # ── Resolve planner page early so pre-check has it ──
            if page is None and _from_planner and _planner_page is not None and not _planner_page.is_closed():
                page = _planner_page
                logger.info(f"[BROWSER] Resolved planner page for step {i+1} (url={page.url})")

            # ── Tier 0: pre-check (target visible/enabled, no occluding modal) ──
            pre = await verification.pre_check(verify_step, page=page)
            if not pre.ok and pre.confidence >= config.VERIFY_MIN_CONFIDENCE:
                # After a click, new UI (overlay/modal) may still be rendering.
                # Wait briefly and retry before giving up.
                _prev_action = steps[i - 1].get("action") if i > 0 else None
                if _prev_action == "click" and action in ("fill", "click", "select") and page:
                    logger.info(f"[BROWSER] Pre-check failed after click, waiting for overlay...")
                    await page.wait_for_timeout(800)
                    pre = await verification.pre_check(verify_step, page=page)

            if not pre.ok and pre.confidence >= config.VERIFY_MIN_CONFIDENCE:
                msg = f"verify_failed (pre): step {i+1} {action} — {pre.observation}"
                logger.warning(f"[BROWSER] {msg}")
                results.append(msg)
                return f"VERIFY_FAILED|step={i+1}|tier=pre_check|obs={pre.observation}\n" + "\n".join(results)

            if action == "navigate":
                url = params.get("url")
                if url:
                    url = re.sub(r'[\]\)\}>]+$', '', url.strip())
                if not page:
                    if _from_planner and _planner_page is not None and not _planner_page.is_closed():
                        page = _planner_page
                        logger.info(f"[BROWSER] Reusing planner page (was at {page.url})")
                    else:
                        from ...core.geolocation import get_cached_region
                        _region = get_cached_region()
                        _locale = f"en-{_region['country_code']}" if _region.get("country_code") else None
                        _tz = _region.get("timezone") or None
                        _ctx_kwargs: dict = {"user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
                        if _locale:
                            _ctx_kwargs["locale"] = _locale
                        if _tz:
                            _ctx_kwargs["timezone_id"] = _tz
                        context = await browser.new_context(**_ctx_kwargs)
                        page = await context.new_page()
                        _pages.append(page)
                        if _from_planner:
                            _planner_page = page
                            _planner_context = context
                            logger.info("[BROWSER] Stored new planner page")
                await page.goto(url, wait_until=params.get("wait_until", "domcontentloaded"), timeout=30000)
                await _dismiss_consent_banner(page)
                results.append(f"Navigated to {url}")

            elif not page:
                if _from_planner and _planner_page is not None and not _planner_page.is_closed():
                    page = _planner_page
                    logger.info(f"[BROWSER] Reusing planner page for non-navigate action '{action}'")
                else:
                    raise ValueError(f"Cannot perform action '{action}' without an active page. 'navigate' must be the first step.")
                
            elif action == "click":
                selector = params.get("selector")
                await page.locator(selector).first.click(timeout=10000)
                results.append(f"Clicked {selector}")
                
            elif action == "fill":
                selector = params.get("selector")
                value = params.get("value", "")
                await page.locator(selector).first.fill(str(value), timeout=10000)
                results.append(f"Filled {selector}")
                
            elif action == "extract_text":
                ready = await page.evaluate("document.readyState") if not page.is_closed() else "closed"
                text = await page.evaluate("document.body.innerText")
                logger.info(f"[BROWSER] extract_text: readyState={ready}, len={len(text) if text else 0}, url={page.url!r}")
                results.append(f"Extracted Text: {text[:500]}..." if text else "Extracted empty text")
                
            elif action == "extract_selector":
                selector = params.get("selector")
                elements = await page.locator(selector).all_inner_texts()
                text = " ".join(elements)
                results.append(f"Extracted from {selector}: {text}")
                
            elif action == "screenshot":
                # Returns base64 but for run_steps we might just want to store it or note it
                path = params.get("path")
                if path:
                    await page.screenshot(path=path)
                    results.append(f"Saved screenshot to {path}")
                else:
                    results.append("Screenshot taken (base64 omitted from log)")
            
            elif action == "wait":
                selector = params.get("selector")
                time_ms = params.get("time_ms")
                if selector:
                    await page.locator(selector).first.wait_for(state="visible", timeout=30000)
                    results.append(f"Waited for visible {selector}")
                elif time_ms:
                    await page.wait_for_timeout(float(time_ms))
                    results.append(f"Waited for {time_ms}ms")
                else:
                    results.append("Wait action missing selector or time_ms")
            
            elif action == "press":
                key = params.get("key") # e.g., "Enter", "Tab"
                selector = params.get("selector")
                if selector:
                    await page.locator(selector).first.press(key, timeout=10000)
                    results.append(f"Pressed {key} on {selector}")
                else:
                    await page.keyboard.press(key)
                    results.append(f"Pressed {key}")
                    
            elif action == "select":
                selector = params.get("selector")
                value = params.get("value")
                await page.locator(selector).first.select_option(str(value), timeout=10000)
                results.append(f"Selected {value} in {selector}")
                
            else:
                results.append(f"Unknown action: {action}")

            # ── Tier 1: post-verify (URL matches, field reads back, etc.) ──
            post = await verification.post_verify(verify_step, page=page)
            if post.tier == "ambiguous" and config.VERIFY_VISION_FALLBACK:
                post = await verification.vision_verify(verify_step, post, page=page)
            if not post.ok and post.confidence >= config.VERIFY_MIN_CONFIDENCE and not post.skipped:
                msg = f"verify_failed (post): step {i+1} {action} — {post.observation}"
                logger.warning(f"[BROWSER] {msg}")

                # Try generic recovery before halting. The overlay_appeared
                # strategy does a bbox click on the affordance matching the
                # goal. error_shown / no_change stubs return False, so the
                # loop guard or max_attempts escalates cleanly and we fall
                # through to the VERIFY_FAILED return below.
                if getattr(config, "RECOVERY_ENABLED", True):
                    from .. import recovery
                    outcome = await recovery.attempt_recovery(
                        step=verify_step,
                        goal=step.get("goal", ""),
                        verify_result=post,
                        page=page,
                        max_attempts=getattr(config, "RECOVERY_MAX_ATTEMPTS", 3),
                    )
                    if outcome.succeeded:
                        last = outcome.attempts[-1]
                        logger.info(
                            f"[BROWSER] recovered step {i+1} via {last.action_taken} "
                            f"(class={last.diagnose_class}) after {len(outcome.attempts)} attempt(s)"
                        )
                        results.append(
                            f"recovered: step {i+1} {action} via {last.action_taken}"
                        )
                        continue

                    # Escalated — surface the diagnose-enriched observation
                    # so the user hears WHAT we got stuck on, not just the
                    # original verify failure.
                    enriched = outcome.final_observation or post.observation
                    logger.warning(
                        f"[BROWSER] recovery escalated step {i+1} after {len(outcome.attempts)} "
                        f"attempt(s): {enriched}"
                    )
                    results.append(msg)
                    return (
                        f"VERIFY_FAILED|step={i+1}|tier={post.tier}|obs={enriched}\n"
                        + "\n".join(results)
                    )

                results.append(msg)
                return f"VERIFY_FAILED|step={i+1}|tier={post.tier}|obs={post.observation}\n" + "\n".join(results)

        return "\n".join(results)

    except Exception as e:
        logger.exception(f"[BROWSER] Run steps error: {type(e).__name__}: {e}")
        return f"Error running steps: {e}\nCompleted so far: " + "\n".join(results)
    finally:
        if page and not _from_planner:
            if page in _pages:
                _pages.remove(page)
            try:
                await page.context.close()
            except Exception:
                pass

async def take_screenshot(url: str = None) -> str:
    """
    Take a screenshot of the current page or navigate to URL first.
    Returns base64-encoded PNG.
    """
    if not PLAYWRIGHT_AVAILABLE:
        return ""

    import base64
    page = None
    close_after = False
    
    try:
        global _pages
        if url:
            page = await get_page(url, wait_until="load")
            close_after = True
        elif _pages:
            page = _pages[-1]
        else:
            raise ValueError("No active page and no URL provided for screenshot.")
            
        screenshot_bytes = await page.screenshot()
        return base64.b64encode(screenshot_bytes).decode('utf-8')
    except Exception as e:
        logger.error(f"[BROWSER] Screenshot error: {e}")
        return ""
    finally:
        if close_after and page:
            if page in _pages:
                _pages.remove(page)
            try:
                await page.context.close()
            except Exception:
                pass

async def close_planner_page():
    """Close the persistent planner page and its context. Safe to call when no planner page exists."""
    global _planner_page, _planner_context, _pages
    if _planner_page is None:
        return
    page = _planner_page
    ctx = _planner_context
    _planner_page = None
    _planner_context = None
    if page in _pages:
        _pages.remove(page)
    if ctx:
        try:
            await ctx.close()
        except Exception as e:
            logger.debug(f"[BROWSER] Error closing planner context: {e}")


async def get_planner_page_info() -> dict | None:
    """Return URL and title of the live planner page, or None."""
    if _planner_page is None or _planner_page.is_closed():
        return None
    try:
        return {"url": _planner_page.url, "title": await _planner_page.title()}
    except Exception:
        return None


# ─── Interactive element scan ────────────────────────────────────────────────

_SCAN_INTERACTIVE_JS = """() => {
    const MAX = 25;
    const results = [];
    const seen = new Set();

    function isSimpleId(id) {
        return /^[a-zA-Z][a-zA-Z0-9_-]*$/.test(id);
    }

    function buildSelector(el) {
        const tag = el.tagName.toLowerCase();
        if (el.id && isSimpleId(el.id))
            return tag + '#' + el.id;
        if (el.getAttribute('name'))
            return tag + '[name="' + el.getAttribute('name') + '"]';
        if (el.getAttribute('placeholder'))
            return tag + '[placeholder="' + el.getAttribute('placeholder') + '"]';
        const aria = el.getAttribute('aria-label');
        if (aria)
            return tag + '[aria-label="' + aria + '"]';
        if (tag === 'input' && el.type && el.type !== 'text')
            return 'input[type="' + el.type + '"]';
        if (tag === 'button' && el.type)
            return 'button[type="' + el.type + '"]';
        return null;
    }

    function scan(root, depth) {
        if (depth > 5 || results.length >= MAX) return;
        const els = root.querySelectorAll(
            'input, textarea, select, button, a[href]'
        );
        for (const el of els) {
            if (results.length >= MAX) break;
            const rect = el.getBoundingClientRect();
            if (rect.width === 0 && rect.height === 0) continue;
            const style = window.getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden') continue;

            const selector = buildSelector(el);
            if (!selector || seen.has(selector)) continue;
            seen.add(selector);

            const tag = el.tagName.toLowerCase();
            const text = (el.innerText || el.textContent || '').trim().slice(0, 40);

            results.push({
                s: selector,
                tag: tag,
                type: el.type || null,
                ph: el.getAttribute('placeholder') || null,
                al: el.getAttribute('aria-label') || null,
                text: text || null
            });
        }
        const allEls = root.querySelectorAll('*');
        for (const el of allEls) {
            if (results.length >= MAX) break;
            if (el.shadowRoot) scan(el.shadowRoot, depth + 1);
        }
    }

    scan(document, 0);
    return results;
}"""


async def get_interactive_elements() -> list[dict] | None:
    """Scan the planner page for interactive elements (inputs, buttons, links).

    Returns a compact list of element descriptors with real CSS selectors,
    or None if no planner page is available.  Traverses open shadow roots
    so SPAs that use web-components are covered.
    """
    if _planner_page is None or _planner_page.is_closed():
        return None
    try:
        await _planner_page.wait_for_load_state("load", timeout=5000)
        elements = await _planner_page.evaluate(_SCAN_INTERACTIVE_JS)
        if elements:
            logger.info(f"[BROWSER] Scanned {len(elements)} interactive element(s)")
        return elements if elements else None
    except Exception as e:
        logger.debug(f"[BROWSER] Interactive element scan failed: {e}")
        return None


async def close_browser():
    """Close browser and Playwright instance. Called on shutdown."""
    global _browser
    if _browser:
        logger.info("[BROWSER] Closing browser...")
        try:
            await _browser.close()
        except Exception as e:
            logger.debug(f"Error closing browser: {e}")
        _browser = None

async def cleanup():
    """Full cleanup — close all pages, browser, and playwright. Called on app exit."""
    global _pages, _playwright, _browser
    
    for page in list(_pages):
        try:
            await page.context.close()
        except Exception:
            pass
    _pages.clear()
    
    await close_browser()
    
    if _playwright:
        logger.info("[BROWSER] Stopping Playwright...")
        try:
            await _playwright.stop()
        except Exception as e:
            logger.debug(f"Error stopping playwright: {e}")
        _playwright = None
