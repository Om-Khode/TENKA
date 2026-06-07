"""Verify _extract_target_app correctly identifies target apps from
'in/on/with/using <app>' suffixes and strips them from the planner goal.
Regression test for the 'type X in notepad → typed into IDE' bug."""

from assistant.automation.router import _extract_target_app


CASES = [
    # (goal, expected_target, expected_stripped)
    # ── Real targets that should be detected and stripped ──
    ("type café résumé 5 in notepad", "notepad", "type café résumé 5"),
    ("Type 'café résumé 5' in notepad", "notepad", "Type 'café résumé 5'"),
    ("write a poem in word", "word", "write a poem"),
    ("multiply 3 and 4 on calculator", "calculator", "multiply 3 and 4"),
    ("play music with spotify", "spotify", "play music"),
    ("search using chrome", "chrome", "search"),

    # ── Stop-words at end should NOT be treated as apps ──
    # (prevents noun-collisions like "in the form / in the field / in the input")
    ("type my email in the form", None, "type my email in the form"),
    ("click in the email field", None, "click in the email field"),
    ("paste it into the textbox", None, "paste it into the textbox"),
    ("search in the box", None, "search in the box"),

    # ── 2-letter or stop-word noun should not match ──
    ("type X in it", None, "type X in it"),
    ("write that in mode", None, "write that in mode"),

    # ── Goals without 'in/on/with/using' suffix at end ──
    ("type café résumé 5", None, "type café résumé 5"),
    ("open notepad", None, "open notepad"),

    # ── 'in' at end of a quoted phrase should not strip ──
    # The regex requires alpha word + END, so 'marketing"' has trailing quote.
    ("type 'I work in marketing'", None, "type 'I work in marketing'"),
]


def run():
    failures = []
    for goal, expected_target, expected_stripped in CASES:
        target, stripped = _extract_target_app(goal)
        ok = (target == expected_target) and (stripped == expected_stripped)
        status = "OK  " if ok else "FAIL"
        print(f"{status}  goal={goal!r}")
        print(f"      target={target!r}  stripped={stripped!r}")
        if not ok:
            print(f"      EXPECTED target={expected_target!r}, stripped={expected_stripped!r}")
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
