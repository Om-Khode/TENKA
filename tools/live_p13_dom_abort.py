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

Three modes, and the middle one is the one to reach for:

    --control          run the goal untouched; it must finish
    --simulate-abort   fire `abort.request_abort()` from a timer; deterministic
    --abort            hold ESC yourself; also tests the ESC monitor

**`--simulate-abort` is what tests the guard.** The guard's job is: given a
raised abort flag, propagate `UserAborted` instead of returning
`"__FALLBACK__"`. Where the flag came from is not part of that, and putting a
human's key-hold timing in the path adds a second failure mode with a
different owner. The first `--abort` run here reported "UserAborted did not
propagate" when `request_abort` had never been called at all — a keypress
finding wearing a propagation finding's clothes. Hence the pre-flight in
`--abort` mode, which confirms the monitor fires *before* the task starts and
reports that as its own line.

Run `--control` and `--simulate-abort` as the pair that proves the change:
a guard that aborts correctly while breaking the success path passes every
red-green check there is. Add `--abort` when you want the keyboard in it too.

Prerequisites:
  1. Chrome launched with `--remote-debugging-port=<port>` on a throwaway
     `--user-data-dir` (recent Chrome ignores the flag on a default profile).
     If the port is not 9222, set `BROWSER_CDP_PORT` in the same shell —
     `config` resolves DB, then env var, then default.
  2. The scratch page open in the active tab.
  3. TENKA itself NOT running — it owns the ESC monitor as a session
     singleton and two monitors on one keyboard is not a test of either.

Note on what the probe accepts: any Chromium answering on the port. On this
machine that was Lenovo Vantage's embedded `msedgewebview2`. `--expect-url`
is the only thing standing between that and a DOM batch clicking through a
system utility.
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

# A harness that dies before printing its verdict is worse than no harness.
# The box-drawing separator and the em dashes in the failure messages are
# undefined in cp1252, which is what stdout falls back to when this is piped
# rather than run in a console -- so the first simulated-abort run raised
# UnicodeEncodeError *after* the task had already aborted correctly, and
# reported nothing at all.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


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


def _esc_still_down() -> bool:
    """Is ESC physically held right now?

    Reuses the monitor's own probe rather than a second implementation, so
    "held" means the same thing to the pre-flight as it does to the thing
    being pre-flighted.
    """
    from assistant.io.esc_monitor import _is_esc_down
    return _is_esc_down()


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
                      help="expect a UserAborted; hold ESC when told. Runs a "
                           "monitor pre-flight first, so a keypress that never "
                           "registered is reported as its own failure")
    mode.add_argument("--simulate-abort", action="store_true",
                      help="same as --abort but fires abort.request_abort() "
                           "from a timer instead of the keyboard. Deterministic; "
                           "tests propagation only, not the ESC monitor")
    mode.add_argument("--control", action="store_true",
                      help="expect completion; do not touch the keyboard")
    ap.add_argument("--fire-after", type=float, default=2.0,
                    help="--simulate-abort only: seconds after GO to fire "
                         "(default 2.0, which lands inside the first batch)")
    args = ap.parse_args(argv)
    wants_abort = args.abort or args.simulate_abort

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

    # ── Pre-flight: does the keypress even reach the flag? ──
    #
    # The first --abort run reported "UserAborted did not propagate" when what
    # actually happened was that `request_abort` was never called at all: no
    # `[abort] requested: esc_hold` line anywhere in the captured log. Those
    # are two different failures with two different owners -- the ESC monitor
    # is pre-existing, shipped machinery, and the guard under test is not --
    # and a harness that renders them identically sends you to read the wrong
    # code. So the keypress is confirmed *before* the task starts.
    preflight_ok = True
    if args.abort:
        print("\n>>> PRE-FLIGHT: hold ESC now, for at least 1 second. <<<")
        deadline = 20.0
        waited = 0.0
        while not abort.is_aborted() and waited < deadline:
            await asyncio.sleep(0.1)
            waited += 0.1
        if abort.is_aborted():
            print(f"    monitor fired after {waited:.1f}s. Release ESC.")
            # The task must start from a clean flag, or `_execute_dom_task`
            # aborts before it ever reaches the orchestrator and the run
            # proves nothing about the fixed path.
            abort.reset()
            await asyncio.sleep(1.2)      # let the hold lapse past threshold
            while _esc_still_down():
                await asyncio.sleep(0.2)
            abort.reset()
        else:
            preflight_ok = False
            print(f"    monitor did NOT fire in {deadline:.0f}s.")

    try:
        if not preflight_ok:
            raised: BaseException | None = None
            result: str | None = None
        else:
            if args.abort:
                print("\n>>> GO. Hold ESC again, for at least 1 second. <<<\n")
            elif args.simulate_abort:
                print(f"\n>>> GO. Firing abort in {args.fire_after:.1f}s — "
                      f"keyboard not needed. <<<\n")
            else:
                print("\n>>> GO. Control run — do not touch the keyboard. <<<\n")

            fired = None
            if args.simulate_abort:
                async def _fire() -> None:
                    await asyncio.sleep(args.fire_after)
                    abort.request_abort("esc_hold")
                fired = asyncio.create_task(_fire())

            raised = None
            result = None
            try:
                result = await router._execute_dom_task(args.goal)
            except UserAborted as e:
                raised = e
            except Exception as e:                    # noqa: BLE001
                raised = e
            finally:
                if fired is not None and not fired.done():
                    fired.cancel()
    finally:
        esc_monitor.stop()
        abort.reset()
        await browser_cdp.detach()

    log = cap.text()
    print("\n" + "─" * 68)

    checks: list[tuple[str, bool, str]] = []

    if args.abort:
        # First, and separately from everything else: did the keyboard reach
        # the flag at all? A run that fails here has tested nothing about the
        # guard, and saying so is the difference between "read router.py" and
        # "hold the key down longer".
        checks.append((
            "the ESC monitor fired at all (pre-flight)",
            preflight_ok,
            "`request_abort` was never called, so nothing below was exercised. "
            "Hold ESC down continuously for a full second — the threshold is "
            "_HOLD_THRESHOLD_SECS = 1.0 and a released-and-repressed key "
            "restarts the count. Or use --simulate-abort, which fires the "
            "flag from a timer and takes the keyboard out of it.",
        ))

    reached_dom = "[DA] DOM-mode running on page:" in log
    checks.append((
        "the fixed path actually ran (`DOM-mode running on page`)",
        reached_dom,
        "the goal never reached _execute_dom_task's orchestrator call, so "
        "nothing below was tested",
    ))

    if wants_abort:
        checks.append((
            "the abort flag was set during the run",
            "[abort] requested" in log or args.simulate_abort,
            "no `[abort] requested` line -- the flag was never raised while "
            "the task was running, whatever the pre-flight showed",
        ))
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
    if wants_abort and ok:
        print("abort path clean."
              + ("  (flag fired by timer, not by ESC)"
                 if args.simulate_abort else ""))
    elif ok:
        print(f"control path clean. reply: {result!r}")
    else:
        print("FAILED. The captured log follows.\n")
        print(log)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(sys.argv[1:])))
