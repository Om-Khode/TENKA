"""Run the test suite one file at a time and record what happened.

`pytest tests/` is not an option here. It drives the real keyboard and mouse
for anything missing the `live_automation` marker -- on 2026-08-08 a bare run
typed into someone's foreground window while they were working -- and it is
too slow to finish, which is how a route-completeness sweep sat red on `main`
for days without anyone noticing.

So: one file per invocation, each with a timeout, and a committed ledger of the
result. The point is not to make the suite green. It is to know, before
touching anything, which files are red and why -- so that a failure during a
refactor can be told apart from a failure that was already there.

That distinction is not theoretical. On 2026-08-22 and -23, three separate
changes hit pre-existing failures (`test_schema_versioning`,
`test_verification_vision`, `test_repo_preference`) and each one cost a
revert-and-rerun to attribute. A ledger answers that in a second.

    py -3.11 scripts/baseline.py --files test_a.py test_b.py    # named files only
    py -3.11 scripts/baseline.py --all-i-know-what-this-does    # everything. see below.

**Running everything requires the long flag, and the default refuses.** On
2026-08-23 this script ran all 271 files as a background job while the operator
was working. A test seized the mouse and keyboard and typed into their active
window, and they had to interrupt to stop it. `CLAUDE.md` already said never run
the whole suite -- running them one at a time in a loop is still running the
whole suite, and the serial framing is what made it feel like something else.

The marker does not make it safe either. `-m 'not live_automation'` filters
AFTER collection, so import-time side effects run regardless, and a file can
reach the desktop through a module no signal list mentions. The marker audit
below is a hint, never a clearance -- it reported one file that day and the
suite still took the keyboard from a file it had not flagged.

So: name the files, having read them. The long flag is for a machine nobody is
using.

Exit codes:
    0  matches the ledger (green, or red exactly where the ledger says)
    1  a file got worse than the ledger says, or a new file is red
    2  the runner itself could not do its job

**What this does not prove.** Per-file is a safety requirement, not an
equivalence claim. The tree has process-wide singletons -- one SQLite
connection, `pending_registry`, `abort`, the contextvars, the provider registry
-- so a per-file pass cannot detect a test that only fails when another file
ran first, and cannot see cross-file state leakage at all. Two real examples
already: `test_repo_preference` shows 1 failure alone and 12 in company, and a
`monkeypatch.setitem(sys.modules, ...)` in a new test passed alone and failed
beside anything that had already imported the module.

So a phase that changes singleton lifetime must additionally run its own
affected files together in one pass, and say so. Presenting this as equivalent
to a full run would be the same class of error as a structural test that walks
nothing: a check that reads as coverage and is not.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parent.parent
LEDGER = _ROOT / "tests" / "BASELINE.md"

# Generous: the slowest honest file measured so far is ~90s (transport
# lifecycle, real sockets). Past this a file is hanging, not working, and a
# hang that is not reported as a failure is worse than a failure -- a test
# that stalls on a mutant never goes red.
PER_FILE_TIMEOUT = 180

_COUNT_RE = re.compile(
    r"(?:^|\s)(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed|deselected)"
)
_NO_TESTS = re.compile(r"no tests ran", re.I)
# All of a file's tests filtered out by `-m 'not live_automation'`.
# Not a failure and not a mystery -- the marker working as intended.
_DESELECTED = re.compile(r"\d+ deselected", re.I)


def _run_one(path: pathlib.Path) -> dict:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(path), "-q",
             "-p", "no:cacheprovider", "--tb=no"],
            cwd=_ROOT, capture_output=True, text=True,
            timeout=PER_FILE_TIMEOUT,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        rc = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or b"").decode(errors="replace")
               + (e.stderr or b"").decode(errors="replace"))
        rc = -1
        timed_out = True

    counts: dict[str, int] = {}
    for n, word in _COUNT_RE.findall(out):
        key = "errored" if word.startswith("error") else word
        counts[key] = counts.get(key, 0) + int(n)

    duration = round(time.monotonic() - started, 1)

    # A hang is a failure, stated explicitly rather than inferred from a
    # missing summary -- the whole reason the timeout exists.
    if timed_out:
        status = "HUNG"
    # Collected nothing. An import error that empties a file, or a parametrised
    # test over an empty list, both look like a clean run otherwise. This is the
    # anti-vacuity check for the suite itself.
    elif _NO_TESTS.search(out) or (not counts and rc == 5):
        status = "EMPTY"
    elif counts.get("failed") or counts.get("errored"):
        status = "RED"
    elif counts.get("passed"):
        status = "GREEN"
    elif counts.get("skipped"):
        status = "SKIPPED"
    # Everything deselected by `addopts`, which is the marker doing its job.
    # `test_computer_task_integration.py` -- which opens apps and clicks
    # buttons -- reported UNKNOWN for exactly this, and UNKNOWN is meant to
    # mean "we could not tell". Here we can: the file is correctly excluded.
    # A state that is permanently unknown for a benign reason is how a column
    # stops being read.
    elif _DESELECTED.search(out):
        status = "DESELECTED"
    else:
        # No recognisable summary at all: a crash before collection, or an
        # output format change. Fail closed -- never read as green.
        status = "UNKNOWN"

    return {
        "file": path.name,
        "status": status,
        "passed": counts.get("passed", 0),
        "failed": counts.get("failed", 0),
        "errored": counts.get("errored", 0),
        "skipped": counts.get("skipped", 0),
        "seconds": duration,
        "rc": rc,
    }


# ─── the marker audit ────────────────────────────────────────────────────────
#
# `addopts = "-m 'not live_automation'"` protects only tests that CARRY the
# marker. An unmarked desktop-driving test is collected by a per-file run and
# types into whatever window is in front. These are the imports that can do it.

_DESKTOP_SIGNALS = (
    "pyautogui", "pygetwindow", "terminator", "keyboard.press",
    "pyperclip", "win32gui", "ctypes.windll", "mss.mss",
    "sync_playwright", "chromium.launch",
)


def _marker_audit() -> list[tuple[str, list[str]]]:
    """Files that look like they drive the desktop without saying so.

    Two filters, because the naive version reported 25 files and 23 of them
    were noise -- a list that long gets ignored, which is worse than no list.

    A file is excused when it mocks (the signal is a patch target, or a string
    in a `sys.modules` stub list) or when it guards itself with a skip. Neither
    is as good as the marker: a `skipUnless(chromium)` file launches a real
    browser on a machine that has one. But both are deliberate, and the point
    of this list is the files where nobody thought about it at all.
    """
    offenders = []
    for f in sorted((_ROOT / "tests").glob("test_*.py")):
        src = f.read_text(encoding="utf-8", errors="replace")
        if "live_automation" in src:
            continue
        # The module has to be genuinely imported, not merely named. Two of the
        # three files the substring version reported were matching a quoted
        # string: a `sys.modules` stub list, and an AST walk searching source
        # text for the word. Neither can move a mouse.
        hits = []
        for sig in _DESKTOP_SIGNALS:
            if sig not in src:
                continue
            mod = sig.split(".")[0]
            imported = re.search(
                rf"^\s*(?:import\s+{mod}\b|from\s+{mod}[\w.]*\s+import)",
                src, re.M)
            called = "." in sig and re.search(rf"\b{re.escape(sig)}\s*\(", src)
            if imported or called:
                hits.append(sig)
        if not hits:
            continue
        mocked = re.search(r"monkeypatch|mock|patch\(|MagicMock|stub|fake",
                           src, re.I)
        guarded = re.search(r"skipUnless|skipif|pytest\.skip", src, re.I)
        if mocked or guarded:
            continue
        offenders.append((f.name, hits))
    return offenders


def _read_ledger() -> dict[str, str]:
    """Previous statuses, by filename. Empty when there is no ledger yet."""
    if not LEDGER.exists():
        return {}
    known = {}
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*`?(test_[\w.]+\.py)`?\s*\|\s*(\w+)\s*\|", line)
        if m:
            known[m.group(1)] = m.group(2)
    return known


def _write_ledger(rows: list[dict], offenders) -> None:
    by_status: dict[str, list[dict]] = {}
    for r in rows:
        by_status.setdefault(r["status"], []).append(r)

    total_time = round(sum(r["seconds"] for r in rows) / 60, 1)
    lines = [
        "# Test baseline",
        "",
        "Generated by `scripts/baseline.py`. **Do not hand-edit the table** — rerun it.",
        "",
        "This exists so a failure during a refactor can be told apart from one that was",
        "already there. Three changes in two days hit pre-existing failures and each cost a",
        "revert-and-rerun to attribute; this answers it in a second.",
        "",
        f"- files: **{len(rows)}**",
        f"- wall time, sequential: **~{total_time} min**",
        "",
        "| status | files |",
        "| --- | --- |",
    ]
    for status in ("GREEN", "RED", "HUNG", "EMPTY", "SKIPPED", "UNKNOWN"):
        if status in by_status:
            lines.append(f"| {status} | {len(by_status[status])} |")
    lines += [
        "",
        "**Per-file is a safety requirement, not an equivalence claim.** Process-wide",
        "singletons mean this cannot see a test that only fails when another file ran",
        "first. `test_repo_preference` shows 1 failure alone and 12 in company. A phase that",
        "changes singleton lifetime must also run its own affected files together, and say so.",
        "",
    ]

    if offenders:
        lines += [
            "## Marker audit — unmarked files that look like they drive the desktop",
            "",
            "`addopts` only protects tests that carry `live_automation`. These import",
            "something that can move the real mouse or keyboard and do not:",
            "",
            "| file | signals |",
            "| --- | --- |",
        ]
        lines += [f"| `{n}` | {', '.join(h)} |" for n, h in offenders]
        lines.append("")

    reds = by_status.get("RED", []) + by_status.get("HUNG", []) \
        + by_status.get("EMPTY", []) + by_status.get("UNKNOWN", [])
    if reds:
        lines += [
            "## Known red",
            "",
            "**Every row needs a reason.** A red file with no recorded reason is one nobody",
            "has looked at, and it blocks the next phase.",
            "",
            "| file | status | failed | errored | reason |",
            "| --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| `{r['file']}` | {r['status']} | {r['failed']} | {r['errored']} | _unexplained_ |"
            for r in sorted(reds, key=lambda r: r["file"])
        ]
        lines.append("")

    lines += [
        "## All files",
        "",
        "| file | status | passed | failed | errored | skipped | secs |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| `{r['file']}` | {r['status']} | {r['passed']} | {r['failed']} | "
        f"{r['errored']} | {r['skipped']} | {r['seconds']} |"
        for r in sorted(rows, key=lambda r: r["file"])
    ]
    LEDGER.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="*", help="specific files, else all of tests/")
    ap.add_argument("--all-i-know-what-this-does", action="store_true",
                    dest="all_i_know_what_this_does",
                    help="run every file. Only on a machine nobody is using — "
                         "a test WILL take the mouse and keyboard.")
    ap.add_argument("--check", action="store_true",
                    help="compare against the ledger and write nothing")
    ap.add_argument("--json", type=pathlib.Path, help="also dump raw results here")
    args = ap.parse_args()

    if args.files:
        paths = [pathlib.Path(f) if pathlib.Path(f).exists()
                 else _ROOT / "tests" / f for f in args.files]
    elif args.all_i_know_what_this_does:
        paths = sorted((_ROOT / "tests").glob("test_*.py"))
    else:
        print(
            "Refusing to run every test file.\n"
            "\n"
            "On 2026-08-23 this script ran all 271 files as a background job while\n"
            "the operator was working. A test seized the mouse and keyboard and\n"
            "typed into their active window. `CLAUDE.md` already said never run the\n"
            "whole suite; running them one at a time in a loop is still running the\n"
            "whole suite, and the serial framing is what made it feel otherwise.\n"
            "\n"
            "The marker does not make this safe. `-m 'not live_automation'` filters\n"
            "AFTER collection, so import-time side effects run regardless, and a\n"
            "file can reach the desktop through a module no signal list mentions.\n"
            "\n"
            "  py -3.11 scripts/baseline.py --files test_a.py test_b.py\n"
            "\n"
            "Name the files. Read them first. Run it on a machine nobody is using.\n"
            "`--all-i-know-what-this-does` exists for that machine and nowhere else.",
            file=sys.stderr)
        return 2

    if not paths:
        print("no test files found — refusing to write an empty ledger", file=sys.stderr)
        return 2

    # A partial run must never overwrite the ledger with a partial picture: the
    # whole value of the ledger is that a file missing from it means "unknown",
    # not "green".
    if args.files and not args.check:
        print("a named-file run does not rewrite the ledger; use --check",
              file=sys.stderr)
        args.check = True

    known = _read_ledger()
    rows, regressions = [], []

    for i, path in enumerate(paths, 1):
        r = _run_one(path)
        rows.append(r)
        was = known.get(r["file"])
        flag = ""
        if was and was == "GREEN" and r["status"] != "GREEN":
            flag = f"  <-- REGRESSION (was {was})"
            regressions.append(r)
        elif was is None and r["status"] != "GREEN" and known:
            flag = "  <-- NEW FILE, not green"
            regressions.append(r)
        print(f"[{i:3}/{len(paths)}] {r['status']:8} {r['file']:52} "
              f"{r['seconds']:6}s{flag}", flush=True)

    offenders = _marker_audit()

    if not args.check:
        _write_ledger(rows, offenders)
        print(f"\nledger written: {LEDGER.relative_to(_ROOT)}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    green = sum(1 for r in rows if r["status"] == "GREEN")
    print(f"\n{green}/{len(rows)} green")
    if offenders:
        print(f"marker audit: {len(offenders)} unmarked file(s) look desktop-driving")
    if regressions:
        print(f"\n{len(regressions)} file(s) worse than the ledger:")
        for r in regressions:
            print(f"  {r['file']} -> {r['status']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
