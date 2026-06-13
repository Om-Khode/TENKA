"""Verify detect_backend routes browser-content goals to vision while
preserving the existing rules for browser-launch goals, native-app goals,
and simple URL goals."""
from unittest.mock import patch
from assistant.automation import router as da


def trace(open_windows, expectations):
    failures = []
    with patch("assistant.io.screen.get_open_windows", return_value=open_windows):
        for goal, expected_backend, note in expectations:
            backend, meta = da.detect_backend(goal)
            ok = backend == expected_backend
            status = "OK  " if ok else "FAIL"
            print(f"{status}  expected={expected_backend:8s}  got={backend:8s}  reason={meta.get('reason')}")
            print(f"      goal: {goal}")
            print(f"      note: {note}")
            if not ok:
                failures.append((goal, expected_backend, backend))
    return failures


FIREFOX_OPEN = [
    "DummyForms - TENKA Testing — Mozilla Firefox",
    "Antigravity",
    "cmd",
]
CHROME_OPEN = [
    "YouTube - Google Chrome",
    "cmd",
]
NO_BROWSER = [
    "Antigravity",
    "Notepad",
    "cmd",
]

# Browser-content fixes — should now route to vision
FIREFOX_CASES = [
    ("fill out the form with random test data", "vision", "form-fill in running Firefox"),
    ("fill this form for testing", "vision", "form-fill alt phrasing"),
    ("submit form", "vision", "submit form intent"),
]

# Browser-launch goals must still go to native/browser, not vision
LAUNCH_CASES_CHROME = [
    ("open chrome", "native", "regression: open running Chrome -> native focus"),
    ("launch chrome", "native", "regression: launch verb -> native"),
    ("open chrome and youtube.com", "browser", "URL pattern still wins -> browser/Playwright"),
]

# Vanilla cases — should be unchanged
GENERIC_CASES = [
    ("type 'café résumé 5' in notepad", "native", "in-app context pattern"),
    ("multiply 3 and 4 on calculator", "native", "in-app context pattern"),
    ("search for cats on google", "browser", "search intent"),
    ("click the back button", "unknown", "no candidate match anywhere"),
]


def _main() -> int:
    all_failures = []
    print("=== Firefox open + browser-content goals ===")
    all_failures += trace(FIREFOX_OPEN, FIREFOX_CASES)
    print()
    print("=== Chrome open + launch goals ===")
    all_failures += trace(CHROME_OPEN, LAUNCH_CASES_CHROME)
    print()
    print("=== No browser open + generic goals ===")
    all_failures += trace(NO_BROWSER, GENERIC_CASES)

    print()
    if all_failures:
        print(f"{len(all_failures)} FAILED:")
        for g, e, a in all_failures:
            print(f"  - {g!r}  expected={e}  got={a}")
        return 1

    total = len(FIREFOX_CASES) + len(LAUNCH_CASES_CHROME) + len(GENERIC_CASES)
    print(f"All {total} routing cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
