"""
verification.py — Tiered step verification.

Three tiers, escalating only when the cheaper one is inconclusive:

  tier 0 (pre-check)   ── is the world ready for this step?     code, ~30ms, $0
  step  (execute)      ── existing step execution
  tier 1 (post-verify) ── did the step actually take effect?    code, ~30ms, $0
  tier 2 (vision)      ── only when tier 1 is AMBIGUOUS         vision, ~600ms, ~$0.0005

Public surface:
    pre_check(step, *, page=None, active_window=None)  -> VerifyResult
    post_verify(step, *, page=None, active_window=None) -> VerifyResult

Both return a VerifyResult. Callers (step loops, procedure_executor) decide
whether to short-circuit, retry, or escalate based on .ok / .tier / .confidence.

Failure-open policy: any internal exception (locator timeout, missing accessibility
backend, JSON parse) returns skipped=True so verification never blocks execution
on infrastructure problems. We surface the exception in observation for diagnosis.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .. import config

logger = logging.getLogger("verification")


# ─── Result type ──────────────────────────────────────────────────────────────

@dataclass
class VerifyResult:
    ok: bool = True
    observation: str = ""           # planner-actionable failure detail
    confidence: float = 1.0         # 1.0 = code-deterministic, <1.0 = vision/heuristic
    skipped: bool = False           # non-verifiable step (wait, extract, etc.)
    tier: str = "code"              # "pre_check" | "code" | "vision" | "skipped" | "ambiguous"

    @classmethod
    def ok_(cls, tier: str = "code", observation: str = "") -> "VerifyResult":
        return cls(ok=True, observation=observation, confidence=1.0, tier=tier)

    @classmethod
    def fail(cls, observation: str, *, tier: str = "code", confidence: float = 1.0) -> "VerifyResult":
        return cls(ok=False, observation=observation, confidence=confidence, tier=tier)

    @classmethod
    def ambiguous(cls, observation: str = "") -> "VerifyResult":
        # Code can't decide — escalate to vision tier.
        return cls(ok=True, observation=observation, confidence=0.5, tier="ambiguous")

    @classmethod
    def skip(cls, reason: str = "") -> "VerifyResult":
        return cls(ok=True, observation=reason, confidence=1.0, skipped=True, tier="skipped")


# ─── Action classification ────────────────────────────────────────────────────

# State-changing actions need verification. Read-only steps don't.
_NON_VERIFIABLE_BROWSER = {"extract_text", "extract_selector", "screenshot", "wait"}
_NON_VERIFIABLE_APP = {"wait", "get_text", "list"}

# Per (step_type, action) → "verifiable" | "skip"
def _is_verifiable(step: dict) -> bool:
    stype = step.get("type", "app")
    action = step.get("action", "")
    if stype == "browser":
        return action not in _NON_VERIFIABLE_BROWSER
    return action not in _NON_VERIFIABLE_APP


# ─── Settings gate ────────────────────────────────────────────────────────────

def _gate(step: dict) -> Optional[VerifyResult]:
    """Return a skip result if settings say to bypass verification for this step."""
    if not getattr(config, "VERIFY_ENABLED", True):
        return VerifyResult.skip("verify_enabled=False")
    stype = step.get("type", "app")
    if stype == "browser" and not getattr(config, "VERIFY_BROWSER_STEPS", True):
        return VerifyResult.skip("verify_browser_steps=False")
    if stype == "app" and not getattr(config, "VERIFY_APP_STEPS", True):
        return VerifyResult.skip("verify_app_steps=False")
    if not _is_verifiable(step):
        return VerifyResult.skip(f"non-verifiable action: {step.get('action')}")
    return None


# ─── Text matching ────────────────────────────────────────────────────────────

def _text_matches(expected: str, actual: str) -> bool:
    """Loose by default (case-insensitive contains) — autocomplete-tolerant.
    Strict (exact ==) only when verify_strict_text_match is on."""
    if expected is None or actual is None:
        return False
    if getattr(config, "VERIFY_STRICT_TEXT_MATCH", False):
        return expected == actual
    return expected.strip().lower() in actual.strip().lower()


# Selectors / param hints that imply a password field — readback would just be dots
_PASSWORD_HINTS = ("password", "passwd", "pwd")

def _is_password_selector(selector: str) -> bool:
    s = (selector or "").lower()
    return any(h in s for h in _PASSWORD_HINTS)


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    u = url.strip().lower().rstrip("/")
    for prefix in ("https://", "http://"):
        if u.startswith(prefix):
            u = u[len(prefix):]
            break
    if u.startswith("www."):
        u = u[4:]
    return u


# ─── Browser checkers (Playwright) ────────────────────────────────────────────

async def _pre_browser_fill(page, params) -> VerifyResult:
    selector = params.get("selector")
    if not selector or page is None:
        return VerifyResult.skip("no selector or page")
    try:
        all_loc = page.locator(selector)
        count = await all_loc.count()
        page_url = page.url
        ready = await page.evaluate("document.readyState") if not page.is_closed() else "closed"
        logger.info(f"[PRE-CHECK] fill: selector={selector!r}, matches={count}, readyState={ready}, url={page_url!r}")
        if count > 0:
            try:
                bb = await all_loc.first.bounding_box(timeout=1000)
                logger.debug(f"[PRE-CHECK] fill: bounding_box={bb}, is_hidden={await all_loc.first.is_hidden()}")
            except Exception:
                pass
        loc = all_loc.first
        if not await loc.is_visible(timeout=2000):
            return VerifyResult.fail(f"target {selector!r} not visible before fill", tier="pre_check")
        if not await loc.is_enabled(timeout=1000):
            return VerifyResult.fail(f"target {selector!r} disabled before fill", tier="pre_check")
        return VerifyResult.ok_(tier="pre_check")
    except Exception as e:
        logger.info(f"[PRE-CHECK] fill exception: {type(e).__name__}: {e}")
        return VerifyResult.skip(f"pre-check exception: {e}")


async def _pre_browser_click(page, params) -> VerifyResult:
    selector = params.get("selector")
    if not selector or page is None:
        return VerifyResult.skip("no selector or page")
    try:
        all_loc = page.locator(selector)
        count = await all_loc.count()
        page_url = page.url
        ready = await page.evaluate("document.readyState") if not page.is_closed() else "closed"
        logger.info(f"[PRE-CHECK] click: selector={selector!r}, matches={count}, readyState={ready}, url={page_url!r}")
        if count > 0:
            try:
                bb = await all_loc.first.bounding_box(timeout=1000)
                logger.debug(f"[PRE-CHECK] click: bounding_box={bb}, is_hidden={await all_loc.first.is_hidden()}")
            except Exception:
                pass
        loc = all_loc.first
        if not await loc.is_visible(timeout=2000):
            return VerifyResult.fail(f"target {selector!r} not visible before click", tier="pre_check")
        if not await loc.is_enabled(timeout=1000):
            return VerifyResult.fail(f"target {selector!r} disabled before click", tier="pre_check")
        return VerifyResult.ok_(tier="pre_check")
    except Exception as e:
        logger.info(f"[PRE-CHECK] click exception: {type(e).__name__}: {e}")
        return VerifyResult.skip(f"pre-check exception: {e}")


async def _post_browser_navigate(page, params) -> VerifyResult:
    expected = _normalize_url(params.get("url", ""))
    actual = _normalize_url(getattr(page, "url", "") if page else "")
    if not expected:
        return VerifyResult.skip("no expected URL")
    if not actual:
        return VerifyResult.fail("page has no URL after navigate")
    if actual == expected or actual.startswith(expected) or expected in actual:
        return VerifyResult.ok_()
    # Host matches but path differs → likely a server-side redirect
    # (e.g. slug normalization: "spiderman" → "spider-man")
    actual_host = actual.split("/")[0]
    expected_host = expected.split("/")[0]
    if actual_host and actual_host == expected_host:
        return VerifyResult.ambiguous(
            f"URL redirected within same host ({actual_host})"
        )
    return VerifyResult.fail(f"URL is {actual!r}, expected {expected!r}")


async def _post_browser_fill(page, params) -> VerifyResult:
    selector = params.get("selector")
    expected = str(params.get("value", ""))
    if page is None or not selector:
        return VerifyResult.skip("no page or selector")
    try:
        loc = page.locator(selector).first
        if _is_password_selector(selector):
            actual = await loc.input_value(timeout=2000)
            if actual:
                return VerifyResult.ok_(observation="password field non-empty (value masked)")
            return VerifyResult.fail("password field is empty after fill")
        actual = await loc.input_value(timeout=2000)
        logger.info(f"[POST-CHECK] fill: selector={selector!r}, expected={expected!r}, actual={actual!r}, url={page.url!r}")
        if _text_matches(expected, actual):
            return VerifyResult.ok_()
        count = await page.locator(selector).count()
        logger.info(f"[POST-CHECK] fill mismatch: {count} element(s) match selector, page readyState={await page.evaluate('document.readyState')}")
        return VerifyResult.fail(f"field {selector!r} reads {actual!r}, typed {expected!r}")
    except Exception as e:
        logger.info(f"[POST-CHECK] fill exception: {type(e).__name__}: {e}")
        return VerifyResult.skip(f"post-verify exception: {e}")


async def _post_browser_click(page, params) -> VerifyResult:
    # Click outcomes are step-dependent — code can't tell "the right thing happened"
    # from generic state. Mark ambiguous so vision tier can answer.
    return VerifyResult.ambiguous("click effect not code-verifiable")


# ─── App checkers (Terminator) ────────────────────────────────────────────────

def _get_app_automation():
    """Lazy import — avoids circular imports at module load and keeps tests
    that don't touch app paths from needing pywinauto/terminator stubs."""
    try:
        from . import native as app_automation
        return app_automation
    except Exception as e:
        logger.debug(f"app_automation import failed: {e}")
        return None


async def _pre_app_type(params, active_window) -> VerifyResult:
    """Pre-check: is the foreground window the one we intend to act on?

    We deliberately do NOT verify element existence here — that would require
    walking the accessibility tree, which on Windows can take 30+ seconds when
    the target isn't immediately found (multiple stacked locator timeouts +
    tree walks in app_automation.get_text). The actual action verifies element
    existence during execution; pre-check's job is to catch focus drift cheaply.
    """
    if not params.get("selector"):
        # Typing at focus → can't pre-verify target without vision.
        return VerifyResult.ambiguous("typing at focus — cannot pre-verify target")

    window = params.get("window") or active_window
    if not window:
        # No window context → nothing to check cheaply. Skip; post-verify
        # readback will catch readback errors after the fact.
        return VerifyResult.skip("no window context for pre-check")

    try:
        import pygetwindow as gw
        active = gw.getActiveWindow()
        if active is None:
            return VerifyResult.skip("no active window — pre-check inconclusive")
        if window.lower() in (active.title or "").lower():
            return VerifyResult.ok_(tier="pre_check")
        return VerifyResult.fail(
            f"focus drift: active window is {active.title!r}, expected {window!r}",
            tier="pre_check",
        )
    except Exception as e:
        return VerifyResult.skip(f"pre-check exception: {e}")


async def _pre_app_click(params, active_window) -> VerifyResult:
    return await _pre_app_type(params, active_window)


async def _post_app_open(params) -> VerifyResult:
    name = (params.get("name") or "").strip()
    if not name:
        return VerifyResult.skip("no app name")
    try:
        import pygetwindow as gw  # already a transitive dep via app_automation
        for w in gw.getAllWindows():
            if name.lower() in (w.title or "").lower():
                return VerifyResult.ok_()
        return VerifyResult.fail(f"no window matching {name!r} after open")
    except Exception as e:
        return VerifyResult.skip(f"window enumeration failed: {e}")


async def _post_app_close(params) -> VerifyResult:
    name = (params.get("name") or "").strip()
    if not name:
        return VerifyResult.skip("no app name")
    try:
        import pygetwindow as gw
        for w in gw.getAllWindows():
            if name.lower() in (w.title or "").lower():
                return VerifyResult.fail(f"window {name!r} still present after close")
        return VerifyResult.ok_()
    except Exception as e:
        return VerifyResult.skip(f"window enumeration failed: {e}")


async def _post_app_focus(params) -> VerifyResult:
    name = (params.get("name") or "").strip()
    if not name:
        return VerifyResult.skip("no window name")
    try:
        import pygetwindow as gw
        active = gw.getActiveWindow()
        if active is None:
            return VerifyResult.fail(f"no active window after focus {name!r}")
        if name.lower() in (active.title or "").lower():
            return VerifyResult.ok_()
        return VerifyResult.fail(f"active window is {active.title!r}, expected to contain {name!r}")
    except Exception as e:
        return VerifyResult.skip(f"focus check failed: {e}")


async def _post_app_type(params, active_window) -> VerifyResult:
    selector = params.get("selector")
    window = params.get("window") or active_window
    expected = str(params.get("text", ""))
    if not selector:
        # No selector — typed at focus. Cannot read back deterministically.
        return VerifyResult.ambiguous("typed at focus — readback not available")
    if _is_password_selector(selector):
        # Most password fields don't expose value via accessibility tree anyway.
        return VerifyResult.ambiguous("password field — readback not reliable")
    aa = _get_app_automation()
    if aa is None:
        return VerifyResult.skip("app_automation unavailable")
    try:
        actual = await aa.get_text(selector, window) or ""
        if isinstance(actual, str) and actual.startswith("Error:"):
            return VerifyResult.fail(f"could not read {selector!r}: {actual}")
        if _text_matches(expected, actual):
            return VerifyResult.ok_()
        return VerifyResult.fail(f"element reads {actual!r}, typed {expected!r}")
    except Exception as e:
        return VerifyResult.skip(f"post-verify exception: {e}")


async def _post_app_click(params, active_window) -> VerifyResult:
    return VerifyResult.ambiguous("click effect not code-verifiable")


# ─── Public dispatcher ────────────────────────────────────────────────────────

async def pre_check(step: dict, *, page=None, active_window: Optional[str] = None) -> VerifyResult:
    """Run before executing a state-changing step. Catches focus drift, missing
    targets, occluded elements — the bug class where the agent would otherwise
    type/click into the wrong thing."""
    skip = _gate(step)
    if skip is not None:
        return skip

    stype = step.get("type", "app")
    action = step.get("action", "")
    params = step.get("params", {}) or {}

    try:
        if stype == "browser":
            if action == "fill":
                return await _pre_browser_fill(page, params)
            if action == "click":
                return await _pre_browser_click(page, params)
            # navigate / press / select → no meaningful pre-check
            return VerifyResult.skip(f"no pre-check for browser/{action}")

        # app
        if action == "type":
            return await _pre_app_type(params, active_window)
        if action == "click":
            return await _pre_app_click(params, active_window)
        # open / focus / close / press_key → no meaningful pre-check
        return VerifyResult.skip(f"no pre-check for app/{action}")

    except Exception as e:
        logger.warning(f"[verify] pre-check crashed for {stype}/{action}: {e}")
        return VerifyResult.skip(f"pre-check crash: {e}")


async def post_verify(step: dict, *, page=None, active_window: Optional[str] = None) -> VerifyResult:
    """Run after executing a state-changing step. Returns:
       - ok=True, tier=code         → confident success
       - ok=False, tier=code        → confident failure (caller surfaces directly)
       - ok=True, tier=ambiguous    → code can't decide (caller may escalate to vision)
       - skipped=True               → bypassed (settings, non-verifiable, infra error)
    """
    skip = _gate(step)
    if skip is not None:
        return skip

    stype = step.get("type", "app")
    action = step.get("action", "")
    params = step.get("params", {}) or {}

    try:
        if stype == "browser":
            if action == "navigate":
                return await _post_browser_navigate(page, params)
            if action == "fill":
                return await _post_browser_fill(page, params)
            if action == "click":
                return await _post_browser_click(page, params)
            # press / select → ambiguous (vision tier)
            return VerifyResult.ambiguous(f"no code post-check for browser/{action}")

        # app
        if action == "open":
            return await _post_app_open(params)
        if action == "close":
            return await _post_app_close(params)
        if action == "focus":
            return await _post_app_focus(params)
        if action == "type":
            return await _post_app_type(params, active_window)
        if action == "click":
            return await _post_app_click(params, active_window)
        # press_key → ambiguous (vision tier)
        return VerifyResult.ambiguous(f"no code post-check for app/{action}")

    except Exception as e:
        logger.warning(f"[verify] post-verify crashed for {stype}/{action}: {e}")
        return VerifyResult.skip(f"post-verify crash: {e}")


# ─── Vision tier ──────────────────────────────────────────────────────────────

_VISION_PROMPT = """You are verifying whether a desktop automation step produced its expected effect.

Step type: {step_type}
Action: {action}
Params: {params}
Code-tier observation (may be empty): {code_obs}

Look at the current screen. Answer STRICTLY in this JSON format and nothing else:
{{
  "ok": true | false,
  "observation": "<one short sentence describing what you actually see — the active window, the visible UI state, any error message, any modal dialog>",
  "confidence": <float 0.0-1.0>
}}

Rules:
- "ok": true only if the action's expected effect is clearly visible right now.
- For CLICK actions specifically: a click TARGET often disappears, toggles state,
  or changes appearance on success (e.g., a Play button becoming a Pause button,
  a menu closing after selection, a checkbox flipping, a tab switching). Return
  "ok": true if you see ANY visible response — a state toggle, a new view, an
  opened dialog, content updated, or playback state changed. Return "ok": false
  for a click ONLY if you see clear evidence the click had no effect (a visible
  error message, OR the click target is still in its original state AND no
  other UI change is visible).
- If the screen shows a LOADING state, return ok=true with confidence ~0.5 and "still loading" in observation.
- If a wrong window/page is shown, return ok=false and say WHICH window/page IS shown.
- If focus is clearly in the wrong place (e.g. browser omnibox vs page search box), that is ok=false.
- If you see an error message or validation failure, quote its text in observation so the agent can self-heal.
- Be literal. Do not guess what the user probably intended.
"""


async def vision_verify(step: dict, code_result: VerifyResult, *,
                        page=None, active_window: Optional[str] = None) -> VerifyResult:
    """Tier 2: escalate to Gemini Flash vision when code tier is ambiguous.

    Captures the primary monitor, asks Flash to judge whether the step's
    expected effect is visible, and returns a VerifyResult. Fail-open: any
    capture/LLM/parse error returns the original code_result unchanged so
    verification never blocks execution on infra issues.
    """
    if not getattr(config, "VERIFY_VISION_FALLBACK", True):
        return code_result

    # Prefer Playwright page screenshot (focus-independent, exact content)
    # over OS screen capture (sees whatever window is in foreground).
    image_b64 = None
    if page is not None:
        try:
            raw_bytes = await page.screenshot(timeout=5000)
            import base64
            image_b64 = base64.b64encode(raw_bytes).decode("ascii")
            logger.debug("[verify-vision] using Playwright page screenshot")
        except Exception as e:
            logger.debug(f"[verify-vision] page screenshot failed, falling back to OS: {e}")

    if not image_b64:
        try:
            from ..io import screen
        except Exception as e:
            logger.warning(f"[verify-vision] screen module unavailable: {e}")
            return code_result
        image_b64 = screen.capture_screenshot_base64()

    if not image_b64:
        logger.warning("[verify-vision] screenshot capture returned None")
        return code_result

    prompt = _VISION_PROMPT.format(
        step_type=step.get("type", "?"),
        action=step.get("action", "?"),
        params=json.dumps(step.get("params", {}), default=str)[:300],
        code_obs=code_result.observation or "(none)",
    )

    try:
        from .. import llm
    except Exception as e:
        logger.warning(f"[verify-vision] llm module unavailable: {e}")
        return code_result

    try:
        raw = (await llm.get_vision_response(
            image_base64=image_b64,
            prompt=prompt,
            system_prompt="You are a precise computer-vision verifier. Reply only with JSON.",
            json_mode=True,
        )).text
    except Exception as e:
        logger.warning(f"[verify-vision] vision call crashed: {e}")
        return code_result

    if not raw or raw == "__LLM_UNAVAILABLE__":
        logger.warning("[verify-vision] no vision response")
        return code_result

    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    try:
        data = json.loads(cleaned)
        return VerifyResult(
            ok=bool(data.get("ok", True)),
            observation=str(data.get("observation", "")),
            confidence=float(data.get("confidence", 0.5)),
            tier="vision",
        )
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning(f"[verify-vision] JSON parse failed: {e} — response head: {raw[:200]}")
        return code_result


# ─── Helpers for callers (planner / TTS surface) ──────────────────────────────

# Format emitted by run_browser_steps / run_app_steps on confident verification
# failure. Stable across versions — the planner and procedure_executor parse it
# to surface human-readable observations to the user.
_VERIFY_FAILED_RE = re.compile(
    r"^VERIFY_FAILED\|step=(?P<step>\d+)\|tier=(?P<tier>[^|]+)\|obs=(?P<obs>[^\n]*)"
)


def parse_verify_failed(text: str) -> Optional[dict]:
    """Parse the structured verify_failed prefix out of a step's output.
    Returns {"step": int, "tier": str, "observation": str} or None.

    Tolerant — searches the first 600 chars so trailing partial results
    after the newline don't matter. Returns None for unrelated output."""
    if not text or "VERIFY_FAILED" not in text[:600]:
        return None
    m = _VERIFY_FAILED_RE.search(text[:600])
    if not m:
        return None
    return {
        "step": int(m.group("step")),
        "tier": m.group("tier"),
        "observation": m.group("obs").strip(),
    }


def format_failure_for_user(parsed: dict) -> str:
    """Build a one-sentence user-facing summary for TTS / chat output."""
    obs = parsed.get("observation") or "no detail available"
    return f"Step {parsed['step']} didn't take — {obs}."
