"""Verify URL detection no longer mis-classifies emails as URLs."""
from assistant.automation.router import _URL_PATTERN

# (goal, should_match, note)
CASES = [
    # --- Emails must NOT match ---
    ("type test.dev20154@gmail.com in the email field", False, "regression: email with .dev local + gmail.com"),
    ("type alice@example.io in the email field", False, "email with .io domain"),
    ("send to john.doe@example.com please", False, "email with .com"),
    ("Type 'alice@example.one' in the email field", False, ".one not in TLD list anyway"),
    ("fill the form with user@company.org", False, "email with .org"),

    # --- Real URLs must still match ---
    ("open chrome and youtube.com", True, "bare TLD URL"),
    ("go to https://example.com/path", True, "https URL"),
    ("visit www.bbc.com/news", True, "www. URL"),
    ("open google.com", True, "bare .com"),
    ("check test.dev tomorrow", True, "bare .dev domain (no @)"),

    # --- Goals with neither ---
    ("type 'café résumé 5'", False, "quoted literal text"),
    ("type my email in the form", False, "instruction without URL"),
    ("multiply 3 and 4 on calculator", False, "no URL"),
]

failures = []
for goal, expected, note in CASES:
    matched = bool(_URL_PATTERN.search(goal))
    actual_match = _URL_PATTERN.search(goal).group(0) if matched else None
    status = "OK " if matched == expected else "FAIL"
    print(f"{status}  expected={expected}  got={matched}  match={actual_match!r:30s}  | {note}")
    print(f"      goal: {goal}")
    if matched != expected:
        failures.append((goal, expected, matched, actual_match, note))

print()
if failures:
    print(f"{len(failures)} FAILED")
    for g, e, m, mm, n in failures:
        print(f"  - {g!r} expected={e} got={m} (match={mm!r}) [{n}]")
    raise SystemExit(1)
else:
    print(f"All {len(CASES)} cases passed.")
