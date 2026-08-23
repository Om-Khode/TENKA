"""The baseline runner reports what happened, including when nothing did.

`scripts/baseline.py` is the thing that tells a refactor whether a red file is
its fault. If it misreports, every later phase inherits the lie -- so its
failure modes are worth pinning harder than most tooling.

Three of them matter, and all three are shapes this project has already been
bitten by:

* **a hang is not a pass.** A test that stalls on a mutant never goes red, so
  the timeout has to produce a status and not a silent skip.
* **collecting nothing is not passing.** An import error that empties a file,
  or a parametrised test over an empty list, reads as a clean run otherwise.
  This is the anti-vacuity rule applied to the suite itself.
* **an unrecognised summary is not green.** A crash before collection, or a
  pytest output format change, must fail closed.

Run with:  py -3.11 -m pytest tests/test_baseline_runner.py -v
"""
import pathlib
import sys
import textwrap

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import baseline  # noqa: E402


def _write(tmp_path: pathlib.Path, name: str, body: str) -> pathlib.Path:
    f = tmp_path / name
    f.write_text(textwrap.dedent(body), encoding="utf-8")
    return f


# ─── status classification ───────────────────────────────────────────────────

def test_a_passing_file_is_green(tmp_path):
    f = _write(tmp_path, "test_ok.py", """
        def test_one(): assert True
    """)
    assert baseline._run_one(f)["status"] == "GREEN"


def test_a_failing_file_is_red(tmp_path):
    f = _write(tmp_path, "test_bad.py", """
        def test_one(): assert False
    """)
    r = baseline._run_one(f)
    assert r["status"] == "RED"
    assert r["failed"] == 1, f"the failure count was not parsed: {r}"


def test_a_collection_error_is_red(tmp_path):
    """An import error is not a pass. It used to be easy to read as one,
    because pytest reports it as an error rather than a failure."""
    f = _write(tmp_path, "test_boom.py", """
        import a_module_that_does_not_exist  # noqa
        def test_one(): assert True
    """)
    r = baseline._run_one(f)
    assert r["status"] == "RED", f"a collection error read as {r['status']}"
    assert r["errored"] >= 1


def test_a_file_that_collects_nothing_is_reported(tmp_path):
    """The anti-vacuity check, applied to the suite. A file with no tests
    passes every assertion it does not contain."""
    f = _write(tmp_path, "test_empty.py", """
        # deliberately no tests
        VALUE = 1
    """)
    assert baseline._run_one(f)["status"] == "EMPTY", (
        "a file containing no tests was not distinguished from a passing one"
    )


def test_a_hanging_file_is_reported_not_awaited_forever(tmp_path, monkeypatch):
    """A hang has to become a status. The timeout is lowered here rather than
    waiting three minutes -- the property is that it produces HUNG, not what
    the limit is."""
    monkeypatch.setattr(baseline, "PER_FILE_TIMEOUT", 3)
    f = _write(tmp_path, "test_hang.py", """
        import time
        def test_one(): time.sleep(60)
    """)
    r = baseline._run_one(f)
    assert r["status"] == "HUNG", (
        f"a hanging file was reported as {r['status']}. A test that stalls on "
        f"a mutant never goes red, so this is the one that must not be a pass."
    )


def test_a_skip_only_file_is_not_green(tmp_path):
    """Skipped is not passed. A file that skips everything -- a missing
    optional dependency, say -- must not be counted as coverage."""
    f = _write(tmp_path, "test_skip.py", """
        import pytest
        @pytest.mark.skip(reason="not today")
        def test_one(): assert True
    """)
    assert baseline._run_one(f)["status"] == "SKIPPED"


# ─── the marker audit ────────────────────────────────────────────────────────

def test_the_marker_audit_finds_an_unmarked_desktop_driver(tmp_path, monkeypatch):
    """`addopts` protects only tests that carry the marker. On 2026-08-08 a
    bare run typed into someone's foreground window."""
    tests = tmp_path / "tests"
    tests.mkdir()
    _write(tests, "test_raw.py", """
        import pyautogui
        def test_one(): pyautogui.click(1, 1)
    """)
    monkeypatch.setattr(baseline, "_ROOT", tmp_path)

    flagged = {n for n, _ in baseline._marker_audit()}
    assert "test_raw.py" in flagged


@pytest.mark.parametrize("body,why", [
    ("""
        import pytest
        pyautogui = pytest.importorskip("pyautogui")
        @pytest.mark.live_automation
        def test_one(): pyautogui.click(1, 1)
     """, "carries the marker"),
    ("""
        def test_one(monkeypatch):
            import pyautogui
            monkeypatch.setattr(pyautogui, "click", lambda *a: None)
     """, "mocks it"),
    ("""
        for name in ["playwright.async_api", "pyautogui"]:
            pass
     """, "only names it in a string"),
])
def test_the_marker_audit_does_not_cry_wolf(tmp_path, monkeypatch, body, why):
    """The substring version of this reported 25 files and 23 were noise --
    a patch target, or a name in a `sys.modules` stub list. A list that long
    gets ignored, which is worse than no list."""
    tests = tmp_path / "tests"
    tests.mkdir()
    _write(tests, "test_fine.py", body)
    monkeypatch.setattr(baseline, "_ROOT", tmp_path)

    flagged = {n for n, _ in baseline._marker_audit()}
    assert "test_fine.py" not in flagged, f"false positive on a file that {why}"


# ─── the ledger contract ─────────────────────────────────────────────────────

def test_the_ledger_round_trips(tmp_path, monkeypatch):
    """`--check` reads back what a run wrote. If the table format and the
    parser ever drift, every regression check silently compares against
    nothing."""
    monkeypatch.setattr(baseline, "_ROOT", tmp_path)
    (tmp_path / "tests").mkdir()
    monkeypatch.setattr(baseline, "LEDGER", tmp_path / "tests" / "BASELINE.md")

    rows = [
        {"file": "test_a.py", "status": "GREEN", "passed": 3, "failed": 0,
         "errored": 0, "skipped": 0, "seconds": 1.0, "rc": 0},
        {"file": "test_b.py", "status": "RED", "passed": 1, "failed": 2,
         "errored": 0, "skipped": 0, "seconds": 2.0, "rc": 1},
    ]
    baseline._write_ledger(rows, [])
    known = baseline._read_ledger()

    assert known == {"test_a.py": "GREEN", "test_b.py": "RED"}, (
        f"the ledger did not round-trip: {known}. A parser that reads nothing "
        f"makes every --check pass."
    )


def test_a_red_file_is_listed_as_needing_a_reason(tmp_path, monkeypatch):
    """A red file with no recorded reason is one nobody has looked at. The
    ledger has to say so rather than just counting it."""
    monkeypatch.setattr(baseline, "_ROOT", tmp_path)
    (tmp_path / "tests").mkdir()
    ledger = tmp_path / "tests" / "BASELINE.md"
    monkeypatch.setattr(baseline, "LEDGER", ledger)

    baseline._write_ledger([
        {"file": "test_b.py", "status": "RED", "passed": 0, "failed": 1,
         "errored": 0, "skipped": 0, "seconds": 1.0, "rc": 1},
    ], [])
    text = ledger.read_text(encoding="utf-8")
    assert "Known red" in text and "_unexplained_" in text


def test_the_limitation_is_stated_in_the_ledger(tmp_path, monkeypatch):
    """Per-file is a safety requirement, not an equivalence claim. Presenting
    it as equivalent to a full run would be a check that reads as coverage and
    is not -- so the caveat is part of the artifact, not just the docstring."""
    monkeypatch.setattr(baseline, "_ROOT", tmp_path)
    (tmp_path / "tests").mkdir()
    ledger = tmp_path / "tests" / "BASELINE.md"
    monkeypatch.setattr(baseline, "LEDGER", ledger)

    baseline._write_ledger([
        {"file": "test_a.py", "status": "GREEN", "passed": 1, "failed": 0,
         "errored": 0, "skipped": 0, "seconds": 1.0, "rc": 0},
    ], [])
    text = ledger.read_text(encoding="utf-8")
    assert "singleton" in text.lower(), (
        "the ledger does not record that per-file cannot see cross-file "
        "leakage, so a reader will take it for more than it is"
    )


def test_an_unrecognisable_summary_is_not_green(monkeypatch, tmp_path):
    """The fail-closed branch, which nothing covered until a mutation went
    green and said so.

    A crash before collection, or a pytest whose summary wording changed,
    leaves no counts to parse. `import-linter` taught this project the lesson
    already: the tool's exit code lied, the summary line was the only reliable
    signal, and the hook fails closed when that line is missing. Same rule
    here -- no parseable result is UNKNOWN, never GREEN.
    """
    class _Proc:
        stdout = "pytest: something went very wrong before collection\n"
        stderr = ""
        returncode = 3

    monkeypatch.setattr(baseline.subprocess, "run", lambda *a, **k: _Proc())
    r = baseline._run_one(tmp_path / "test_whatever.py")

    assert r["status"] == "UNKNOWN", (
        f"output with no parseable summary was reported as {r['status']}. "
        f"A format change or an early crash would read as coverage."
    )
    assert r["passed"] == 0


def test_a_summary_pytest_does_produce_is_still_parsed(monkeypatch, tmp_path):
    """The other direction: the UNKNOWN branch must not swallow real results.
    A fail-closed default that fires on everything is not fail-closed, it is
    broken."""
    class _Proc:
        stdout = "....                        [100%]\n4 passed in 0.12s\n"
        stderr = ""
        returncode = 0

    monkeypatch.setattr(baseline.subprocess, "run", lambda *a, **k: _Proc())
    r = baseline._run_one(tmp_path / "test_whatever.py")
    assert r["status"] == "GREEN" and r["passed"] == 4, r
