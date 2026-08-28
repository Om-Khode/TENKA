"""Live check: an ESC hold during a DOM batch must not become `__FALLBACK__`.

TENKA-v2 §17.P13, loop 1. **This drives the real browser.** It is in `tools/`
and not `tests/` on purpose — no collector reaches it, so it can never be
swept up by a `pytest tests/` run the way the 2026-08-08 and 2026-08-23
incidents were.

What it pins, and why a unit test could not:

    `dom_executor.execute_dom_batch` raises `UserAborted` at every action
    boundary. `router._execute_dom_task` used to catch it with a bare
    `except Exception` and return `"__FALLBACK__"` — not an error string but an
    instruction to escalate a tier. The unit tests stub the executor and assert
    the guard; only a real ESC hold against a real CDP page proves the flag
    reaches the boundary that raises in the first place.

Safety. It will click and type on whatever page it attaches to, so
`--expect-url` is **required**: the active tab's URL must contain that
substring or nothing runs. Point it at a scratch page, never at anything
logged in.

    py -3.11 tools/live_p13_dom_abort.py --expect-url httpbin --abort \
        --goal "click the submit button"

    py -3.11 tools/live_p13_dom_abort.py --expect-url httpbin --control \
        --goal "click the submit button"

`--abort` expects you to hold ESC for >= 1s once it says GO. `--control` runs
the same goal untouched and expects it to finish — the answer, not the
refusal. Run both; a guard that aborts correctly while breaking the success
path passes every red-green check there is.

Prerequisites:
  1. Chrome launched with `--remote-debugging-port=9222` (see
     `browser_cdp_setup` in config, or launch it by hand).
  2. The scratch page open in the active tab.
  3. TENKA itself NOT running — it owns the ESC monitor as a session
     singleton and two monitors on one keyboard is not a test of either.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ─── Log capture ─────────────────────────────────────────────────────────────
#
# The criteria are as much about what is *logged* as about what is returned:
# "DOM-mode orchestrator crashed" was the line that made an abort look like a
# bug in the orchestrator. Captured rather than eyeballed so the script can
# answer instead of asking the operator to grep.


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(f"{record.name}: {record.getMessage()}")
        except Exception:
            pass

    def text(self) -> str:
        return "\n".join(self.lines)


def _install_capture() -> _Capture:
    cap = _Capture()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(cap)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(logging.StreamHandler(sys.stdout))
    return cap


# ─── The run ─────────────────────────────────────────────────────────────────


async def _verify_target(expect_url: str) -> str:
    """Attach, pick the active page, and refuse a tab we were not pointed at.

    Duplicates the first two steps of `_execute_dom_task` deliberately. That
    function picks its own page and this script is about to let it act on
    whatever that is; the gate has to happen before it, not inside it.

    Not hypothetical. The first run of this script found a CDP-capable **Edge**
    on 9222 sitting on a Lenovo system-utility page. `--expect-url` is what
    stopped a DOM batch from clicking around in it.

    Raises `SystemExit` with a REFUSED message on any of the three gates.
    """
    from assistant.automation.browser import cdp as browser_cdp
    from assistant.automation import router

    # Probe before attaching, and not for speed. `get_or_attach_browser`
    # falls through to `ensure_browser()` when CDP is down -- step 4 of its
    # own decision tree -- so calling it blind on a machine without a
    # debug-port Chrome *launches a Chromium* as a side effect of a script
    # whose whole job is to be careful about what it touches. The probe is a
    # plain HTTP GET on 127.0.0.1 and never raises.
    probe = await browser_cdp.cdp_health_probe(use_cache=False)
    if not probe.available:
        raise SystemExit(
            f"REFUSED: no CDP on port {getattr(browser_cdp.config, 'BROWSER_CDP_PORT', 9222)} "
            f"({probe.error or 'unavailable'}). Launch Chrome with "
            f"--remote-debugging-port=9222 first. Not attaching, because "
            f"get_or_attach_browser would launch a bundled Chromium instead."
        )

    handle = await browser_cdp.get_or_attach_browser(prefer_cdp=True)
    if handle.kind != "cdp":
        raise SystemExit(
            f"REFUSED: attach returned kind={handle.kind!r}, not 'cdp' — "
            f"the probe passed but the attach did not, which usually means "
            f"DevTools is open and holding the port."
        )
    page = await router._pick_active_page(handle.attachment)
    if page is None:
        raise SystemExit("REFUSED: no usable page in the attached browser.")
    url = getattr(page, "url", "") or ""
    if expect_url.lower() not in url.lower():
        raise SystemExit(
            f"REFUSED: active tab is {url!r}, which does not contain "
            f"{expect_url!r}. Not clicking on it."
        )
    return url


async def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-url", required=True,
                    help="substring the active tab's URL must contain")
    ap.add_argument("--goal", required=True,
                    help="the DOM goal to run; must NOT match the form-intent "
                         "regex, or the router hands it to run_dom_form_fill "
                         "instead of run_dom_task")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--abort", action="store_true",
                      help="expect a UserAborted; hold ESC when told")
    mode.add_argument("--control", action="store_true",
                      help="expect completion; do not touch the keyboard")
    args = ap.parse_args(argv)

    cap = _install_capture()

    from assistant.automation import router
    from assistant.automation.browser import dom_orchestrator as bdo
    from assistant.core.abort import abort, UserAborted
    from assistant.io.esc_monitor import esc_monitor

    # `run_dom_form_fill` has no abort source in it -- no `execute_dom_batch`,
    # no abort check anywhere in `dom_filler` or the Playwright locator calls.
    # A goal that trips the form-intent regex therefore tests nothing here,
    # silently, which is exactly the kind of green this phase is about.
    if router._FORM_INTENT_RE.search(args.goal):
        print(f"REFUSED: goal {args.goal!r} matches _FORM_INTENT_RE, so "
              f"router._execute_dom_task will call run_dom_form_fill -- which "
              f"never raises UserAborted. Reword it without "
              f"fill/set/choose/pick/submit/enter/select/type/register/"
              f"login/signin/checkout/book/schedule/subscribe.")
        return 2

    # Every exit past this point detaches. Playwright's subprocess transport
    # outlives a loop that closes under it and prints an "Event loop is
    # closed" traceback over the top of the verdict -- which on the first run
    # buried the REFUSED line under six frames of asyncio teardown. Chrome
    # itself is untouched; `detach()` only ends the Playwright session.
    from assistant.automation.browser import cdp as browser_cdp
    try:
        url = await _verify_target(args.expect_url)
    except SystemExit as e:
        print(f"\n{e}")
        await browser_cdp.detach()
        return 2

    print(f"\ntarget page: {url}")

    abort.reset()
    esc_monitor.start()
    try:
        if args.abort:
            print("\n>>> GO. Hold ESC for at least 1 second, NOW. <<<\n")
        else:
            print("\n>>> GO. Control run — do not touch the keyboard. <<<\n")

        raised: BaseException | None = None
        result: str | None = None
        try:
            result = await router._execute_dom_task(args.goal)
        except UserAborted as e:
            raised = e
        except Exception as e:                        # noqa: BLE001
            raised = e
    finally:
        esc_monitor.stop()
        abort.reset()
        await browser_cdp.detach()

    log = cap.text()
    print("\n" + "─" * 68)

    checks: list[tuple[str, bool, str]] = []
    reached_dom = "[DA] DOM-mode running on page:" in log
    checks.append((
        "the fixed path actually ran (`DOM-mode running on page`)",
        reached_dom,
        "the goal never reached _execute_dom_task's orchestrator call, so "
        "nothing below was tested",
    ))

    if args.abort:
        checks.append((
            "UserAborted propagated out of _execute_dom_task",
            isinstance(raised, UserAborted),
            f"got {type(raised).__name__ if raised else 'no exception'}, "
            f"return={result!r}",
        ))
        checks.append((
            "it did NOT decay into __FALLBACK__",
            result != "__FALLBACK__",
            "the abort became an instruction to escalate to the vision tier",
        ))
        checks.append((
            "the abort was not logged as a crash",
            "orchestrator crashed" not in log,
            "the `DOM-mode orchestrator crashed` line is back",
        ))
    else:
        checks.append((
            "the task ran to a verdict with no exception",
            raised is None,
            f"raised {type(raised).__name__ if raised else '-'}: {raised}",
        ))
        checks.append((
            "it produced a spoken reply, not __FALLBACK__",
            bool(result) and result != "__FALLBACK__",
            f"return={result!r} -- the control path is broken, which a "
            f"refusal-only test would never show",
        ))

    ok = True
    for label, passed, why in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        if not passed:
            print(f"         {why}")
            ok = False

    print("─" * 68)
    if args.abort and ok:
        print("abort path clean.")
    elif ok:
        print(f"control path clean. reply: {result!r}")
    else:
        print("FAILED. The captured log follows.\n")
        print(log)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(sys.argv[1:])))
