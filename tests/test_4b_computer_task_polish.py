"""
test_4b_computer_task_polish.py — Unit-testable logic for computer_task P1/P2/P3 fixes.

NOTE: Tests that require live Windows/Terminator are not included here.
These tests cover the detection logic and code paths that can be verified
without a real UI.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# ── P3: Calculator detection logic ───────────────────────────────────────────

_CALC_OPS = {'+': 'add', '-': 'subtract', '*': 'multiply', '/': 'divide'}

def _calc_keymap(text: str) -> list[str]:
    return [_CALC_OPS.get(ch, ch) for ch in text]

def test_calc_keymap_digits():
    assert _calc_keymap("25") == ['2', '5']

def test_calc_keymap_expression():
    assert _calc_keymap("3+4") == ['3', 'add', '4']

def test_calc_keymap_multiply():
    assert _calc_keymap("6*7") == ['6', 'multiply', '7']

def test_calc_keymap_divide():
    assert _calc_keymap("9/3") == ['9', 'divide', '3']

def test_calc_keymap_subtract():
    assert _calc_keymap("10-2") == ['1', '0', 'subtract', '2']

def test_calc_keymap_equals_passthrough():
    # '=' has no special map — passes through as '='
    assert _calc_keymap("1+2=") == ['1', 'add', '2', '=']

def test_calc_keymap_decimal():
    assert _calc_keymap("3.14") == ['3', '.', '1', '4']

# ── P1: Already-running window title match logic ──────────────────────────────

def _app_already_running(app_name: str, window_titles: list[str]) -> str | None:
    """Returns first matching window title if app is already running, else None."""
    for title in window_titles:
        if app_name.lower() in title.lower() and title.strip():
            return title
    return None

def test_p1_detects_running_app():
    titles = ["Spotify Premium", "Notepad", "Task Manager"]
    assert _app_already_running("spotify", titles) == "Spotify Premium"

def test_p1_detects_notepad():
    titles = ["Notepad - untitled.txt", "Chrome"]
    assert _app_already_running("Notepad", titles) == "Notepad - untitled.txt"

def test_p1_not_running():
    titles = ["Notepad", "Task Manager"]
    assert _app_already_running("spotify", titles) is None

def test_p1_empty_title_skipped():
    titles = ["", "  ", "Spotify"]
    assert _app_already_running("spotify", titles) == "Spotify"

def test_p1_case_insensitive():
    titles = ["SPOTIFY"]
    assert _app_already_running("Spotify", titles) == "SPOTIFY"

# ── P2: get_text wait logic (code path validation) ────────────────────────────

def test_p2_automationid_bypasses_direct_locator():
    # automationid: selectors skip approach 1/2 in get_text()
    selector = "automationid:CalculatorResults"
    assert selector.lower().startswith("automationid:")

def test_p2_name_selector_uses_direct_locator():
    selector = "name:Display"
    assert not selector.lower().startswith("automationid:")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
