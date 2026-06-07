"""Verify policy DANGEROUS_PATTERNS use word boundaries — so 'form with'
no longer matches 'rm ', etc. Real dangerous strings must still block."""

from assistant.intent import IntentResult
from assistant import policy


def _check(goal: str) -> tuple[bool, str]:
    res = policy.evaluate(IntentResult(intent="computer_task", response="", params={"goal": goal}))
    return res.allowed, res.reason


CASES = [
    # ── Benign goals that USED to be falsely blocked by 'rm ' substring ──
    ("fill out the form with random test data", True, "regression: 'form with' falsely matched 'rm '"),
    ("fill this form for testing", True, "regression: 'form for' falsely matched 'rm '"),
    ("type 'café résumé 5' in notepad", True, "résumé and form-free goal"),
    ("open chrome and youtube.com", True, "no dangerous patterns"),
    ("Type test.dev20154@gmail.com in the email field", True, "form-free email goal"),
    ("perform a calculation", True, "'perform' contains 'rm' but not as a word"),
    ("warm up the model", True, "'warm' should not match 'rm'"),
    ("alarm the system", True, "'alarm' should not match 'rm'"),

    # ── Words containing 'kill' as substring should not block ──
    # \bkill\b requires non-word boundaries on BOTH sides — "killer" has 'e'
    # after 'kill' (word char), so no boundary, no match. Correct behaviour.
    ("show me a skillet recipe", True, "skillet contains 'kill' but not at boundary"),
    ("I have killer skills", True, "killer contains 'kill' but trailing 'e' breaks boundary"),

    # ── Real dangerous shell commands MUST still block ──
    ("rm -rf /", False, "shell rm command"),
    ("sudo apt install foo", False, "sudo invocation"),
    ("kill the process now", False, "literal kill verb"),
    ("format c:", False, "format disk command"),
    ("execute(payload)", False, "execute( with paren"),
    ("eval(user_input)", False, "eval( with paren"),
    ("shutdown -h now", False, "shutdown command"),
    ("taskkill /im chrome.exe", False, "taskkill"),
    ("rmdir /s tmp", False, "rmdir"),
    ("mkfs.ext4 /dev/sda1", False, "mkfs"),
]


def run():
    failures = []
    for goal, expected_allowed, note in CASES:
        allowed, reason = _check(goal)
        ok = allowed == expected_allowed
        status = "OK  " if ok else "FAIL"
        verdict = "ALLOW" if allowed else f"DENY ({reason})"
        exp = "ALLOW" if expected_allowed else "DENY"
        print(f"{status}  expected={exp:5s}  got={verdict:50s}")
        print(f"      goal: {goal}")
        print(f"      note: {note}")
        if not ok:
            failures.append(goal)

    print()
    if failures:
        print(f"{len(failures)} FAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print(f"All {len(CASES)} cases passed.")


if __name__ == "__main__":
    run()
