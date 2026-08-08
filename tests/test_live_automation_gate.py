"""The gate that keeps `pytest tests/` from driving the real desktop.

On 2026-08-08 a bare full-suite run typed into the foreground window while
someone was working: the suite collects files that open applications, click,
and launch browsers. Three mechanisms now stop that, and this file asserts all
three stay in place, because a gate nobody tests is a gate that quietly opens.

1. `addopts = -m 'not live_automation'` in pyproject.toml — deselects marked
   tests by default.
2. `pytestmark = pytest.mark.live_automation` on the files that drive the
   desktop.
3. `collect_ignore` in conftest.py for files that act at *import* time, which a
   marker cannot protect: pytest imports a module to read its markers.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

TESTS_DIR = pathlib.Path(__file__).resolve().parent
PYPROJECT = TESTS_DIR.parent / "pyproject.toml"
CONFTEST = TESTS_DIR / "conftest.py"

# Files that drive the real desktop or a real browser and must stay gated.
LIVE_FILES = ("test_computer_task_integration.py",)

# Files that do their work at import and must never be collected at all.
IMPORT_UNSAFE_FILES = ("test_pw_standalone.py",)


def test_default_run_deselects_live_automation():
    config = PYPROJECT.read_text(encoding="utf-8")
    assert "[tool.pytest.ini_options]" in config, "pytest config block is gone"
    assert "not live_automation" in config, (
        "addopts no longer deselects live_automation — a bare `pytest tests/` "
        "would drive the desktop again"
    )


def test_the_marker_is_registered():
    config = PYPROJECT.read_text(encoding="utf-8")
    assert "live_automation:" in config, (
        "the marker is unregistered, so --strict-markers runs would error and "
        "a typo in a pytestmark would silently stop gating anything"
    )


@pytest.mark.parametrize("filename", LIVE_FILES)
def test_live_files_carry_the_marker(filename):
    source = (TESTS_DIR / filename).read_text(encoding="utf-8")
    assert "pytestmark = pytest.mark.live_automation" in source, (
        f"{filename} drives the real desktop but is no longer marked"
    )


@pytest.mark.parametrize("filename", IMPORT_UNSAFE_FILES)
def test_import_unsafe_files_are_never_collected(filename):
    source = CONFTEST.read_text(encoding="utf-8")
    assert filename in source, (
        f"{filename} acts at import time and is no longer in collect_ignore; "
        f"collecting it would launch a browser"
    )


def _executes_at_import(path: pathlib.Path) -> bool:
    """True when the module runs something at import outside a __main__ guard."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return False
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        target = call.func
        name = getattr(target, "attr", None) or getattr(target, "id", None)
        # asyncio.run(...), main(), sys.exit(...) at module level — anything
        # that does work rather than defining it.
        if name in {"run", "main", "exit"}:
            return True
    return False


def test_no_test_module_does_work_at_import():
    """A new file like test_pw_standalone would reopen the same hole.

    Collection imports every module it finds, so work at import happens even
    under --collect-only, and even for a test that is about to be deselected.
    """
    offenders = [
        path.name
        for path in sorted(TESTS_DIR.glob("test_*.py"))
        if path.name not in IMPORT_UNSAFE_FILES and _executes_at_import(path)
    ]
    assert offenders == [], (
        f"these modules do work at import: {offenders}. Put the call behind "
        f"`if __name__ == '__main__':`, or add the file to collect_ignore in "
        f"conftest.py."
    )
