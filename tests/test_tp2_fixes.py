"""
Tests for TP-2 fixes:
  Bug 1: Trigger matching — longest match wins
  Bug 2: Slot confirmation prompt for suspect literals (replaces auto-slot)
  Feature: Batch/paste teaching mode
  Feature: Keyboard shortcut hint in teach-start
"""

import asyncio
import sys
import os
import traceback

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant import procedures as ps
from assistant.actions import (
    _parse_teaching_step,
    _extract_slots_from_steps,
    _find_suspect_literals,
    _split_pasted_steps,
    start_teaching_session,
    start_batch_teaching,
    handle_pending_teaching,
)

_loop = asyncio.new_event_loop()


def _run(coro):
    return _loop.run_until_complete(coro)


def setup():
    ps.init_procedure_db()
    # Clean up any leftover test procedures
    for proc in ps.list_procedures(enabled_only=False):
        if proc["id"] > 2:
            ps._get_conn().execute("DELETE FROM user_procedures WHERE id = ?", (proc["id"],))
    ps._get_conn().commit()


# ─── Bug 1: Longest-match-wins ─────────────────────────────────────────────


def test_longest_trigger_wins():
    short_id = ps.create_procedure(
        trigger="open whatsapp and send message",
        name="Short Trigger",
        steps=[{"type": "app", "action": "open", "params": {"name": "whatsapp"}}],
    )
    long_id = ps.create_procedure(
        trigger="open whatsapp and send messages",
        name="Long Trigger",
        steps=[{"type": "app", "action": "open", "params": {"name": "whatsapp"}}],
    )
    result = ps.match_trigger("open whatsapp and send messages to samyak saying hello")
    assert result is not None and result["id"] == long_id, (
        f"Expected id={long_id}, got {result}"
    )
    print(f"  PASS: matched id={result['id']} trigger='{result['trigger']}'")
    ps.delete_procedure(short_id)
    ps.delete_procedure(long_id)


def test_exact_match_still_works():
    short_id = ps.create_procedure(
        trigger="open notes",
        name="Short",
        steps=[{"type": "app", "action": "open", "params": {"name": "notes"}}],
    )
    long_id = ps.create_procedure(
        trigger="open notes and save",
        name="Long",
        steps=[{"type": "app", "action": "open", "params": {"name": "notes"}}],
    )
    result = ps.match_trigger("open notes")
    assert result is not None and result["id"] == short_id
    print(f"  PASS: exact match id={result['id']}")
    ps.delete_procedure(short_id)
    ps.delete_procedure(long_id)


# ─── Parser: no auto-slot (removed) ────────────────────────────────────────


def test_no_auto_slot():
    """Auto-slot was removed — single words should stay literal."""
    step = _parse_teaching_step("type contact in search")
    assert step["params"]["text"] == "contact", (
        f"Expected literal 'contact', got '{step['params']['text']}'"
    )
    print(f"  PASS: text='{step['params']['text']}' (literal, no auto-slot)")


def test_braced_stays_braced():
    step = _parse_teaching_step("type {name}")
    assert step["params"]["text"] == "{name}"
    print(f"  PASS: text='{step['params']['text']}'")


def test_numbered_step_parsing():
    """Steps with leading numbers like '1. open whatsapp' should parse."""
    cases = [
        ("1. open whatsapp", "open"),
        ("2) press ctrl+f", "press_key"),
        ("3- type {contact}", "type"),
        ("  4: click on save button", "click"),
        ("- open notepad", "open"),
        ("* press enter", "press_key"),
    ]
    for text, expected_action in cases:
        step = _parse_teaching_step(text)
        assert step is not None, f"Failed to parse: '{text}'"
        assert isinstance(step, dict), f"Expected dict for '{text}', got {type(step)}"
        assert step["action"] == expected_action, (
            f"'{text}' → action='{step['action']}', expected '{expected_action}'"
        )
    print(f"  PASS: all {len(cases)} numbered formats parsed")


def test_positional_click_expansion():
    """'click on first result' should expand to [press down, press enter]."""
    step = _parse_teaching_step("click on first result")
    assert isinstance(step, list), f"Expected list, got {type(step)}: {step}"
    assert len(step) == 2, f"Expected 2 steps (down+enter), got {len(step)}"
    assert step[0]["params"]["key"] == "down"
    assert step[1]["params"]["key"] == "enter"
    print(f"  PASS: 'click first result' → {len(step)} keyboard steps")


def test_positional_click_second():
    """'click second item' should expand to [down, down, enter]."""
    step = _parse_teaching_step("click on second item")
    assert isinstance(step, list) and len(step) == 3
    assert step[0]["params"]["key"] == "down"
    assert step[1]["params"]["key"] == "down"
    assert step[2]["params"]["key"] == "enter"
    print(f"  PASS: 'click second item' → {len(step)} keyboard steps")


def test_repeat_modifier_times():
    """'press down 4 times' should expand to 4 press_key steps."""
    step = _parse_teaching_step("press down 4 times")
    assert isinstance(step, list), f"Expected list, got {type(step)}"
    assert len(step) == 4, f"Expected 4, got {len(step)}"
    assert all(s["params"]["key"] == "down" for s in step)
    print(f"  PASS: 'press down 4 times' → {len(step)} steps")


def test_repeat_modifier_x():
    """'press tab x3' should expand to 3 press_key steps."""
    step = _parse_teaching_step("press tab x3")
    assert isinstance(step, list) and len(step) == 3
    assert all(s["params"]["key"] == "tab" for s in step)
    print(f"  PASS: 'press tab x3' → {len(step)} steps")


def test_repeat_modifier_single():
    """'press enter' (no repeat) should stay as a single dict."""
    step = _parse_teaching_step("press enter")
    assert isinstance(step, dict), f"Expected dict, got {type(step)}"
    assert step["params"]["key"] == "enter"
    print(f"  PASS: single press unchanged")


def test_regular_click_unchanged():
    """'click on save button' should stay as a normal click step."""
    step = _parse_teaching_step("click on save button")
    assert isinstance(step, dict), f"Expected dict, got {type(step)}"
    assert step["action"] == "click"
    print(f"  PASS: regular click unchanged")


def test_inline_paste_split():
    """Single-line numbered paste should split into individual steps."""
    text = "1. open whatsapp  2. press ctrl+f  3. type {contact}  4. click on first result"
    lines = _split_pasted_steps(text)
    assert len(lines) == 4, f"Expected 4 lines, got {len(lines)}: {lines}"
    print(f"  PASS: split into {len(lines)} lines")


def test_newline_paste_split():
    """Multi-line paste with newlines should split correctly."""
    text = "1. open whatsapp\n2. press ctrl+f\n3. type {contact}"
    lines = _split_pasted_steps(text)
    assert len(lines) == 3, f"Expected 3 lines, got {len(lines)}"
    print(f"  PASS: newline split into {len(lines)} lines")


def test_inline_paste_during_collecting():
    """Pasting numbered steps during collecting state should batch-add all.
    'click on first result' expands to 2 keyboard steps, so 6 lines → 7 steps."""
    import assistant.actions as _actions

    start_teaching_session("inline test")
    resp = _run(handle_pending_teaching(
        "1. open whatsapp  2. press ctrl+f  3. type {contact}  4. click on first result  5. type {message}  6. press enter"
    ))
    assert "7 steps" in resp.lower() or "got 7" in resp.lower(), (
        f"Expected 7-step confirmation (click first result → 2 steps), got: {resp}"
    )

    # Now say done
    resp = _run(handle_pending_teaching("done"))
    assert "step 1" in resp.lower() or "step 2" in resp.lower(), (
        f"Expected readback, got: {resp}"
    )

    # Confirm and save
    resp = _run(handle_pending_teaching("yes"))
    resp = _run(handle_pending_teaching("yes"))
    assert any(w in resp.lower() for w in ("saved", "done", "got it"))

    proc = ps.get_procedure("inline test")
    assert proc is not None
    assert len(proc["steps"]) == 7, f"Expected 7 steps, got {len(proc['steps'])}"
    slots = _extract_slots_from_steps(proc["steps"])
    assert "contact" in slots and "message" in slots
    print(f"  PASS: inline paste → {len(proc['steps'])} steps, slots={slots}")
    ps.delete_procedure(proc["id"])


# ─── Suspect literal detection ──────────────────────────────────────────────


def test_find_suspects():
    steps = [
        {"type": "app", "action": "open", "params": {"name": "whatsapp"}},
        {"type": "app", "action": "type", "params": {"text": "contact", "window": "search"}},
        {"type": "app", "action": "type", "params": {"text": "{message}"}},
        {"type": "app", "action": "type", "params": {"text": "hello world"}},
        {"type": "app", "action": "type", "params": {"text": "the"}},
    ]
    suspects = _find_suspect_literals(steps)
    assert len(suspects) == 1 and suspects[0] == (1, "contact"), (
        f"Expected [(1, 'contact')], got {suspects}"
    )
    print(f"  PASS: suspects={suspects}")


# ─── Slot confirmation flow ────────────────────────────────────────────────


def test_slot_confirm_flow():
    """Teaching session with slot_confirm state for suspect literals."""
    import assistant.actions as _actions

    start_teaching_session("send whatsapp message")

    for s in ["open whatsapp", "type contact in search", "click first result",
              "type message", "press enter"]:
        _run(handle_pending_teaching(s))

    # Say done — should enter slot_confirm, ask about 'contact'
    resp = _run(handle_pending_teaching("done"))
    assert "contact" in resp and "change" in resp.lower(), (
        f"Expected slot-confirm for 'contact', got: {resp}"
    )

    # Say yes — contact becomes {contact}, should ask about 'message'
    resp = _run(handle_pending_teaching("yes"))
    assert "message" in resp, f"Expected slot-confirm for 'message', got: {resp}"

    # Say yes — message becomes {message}, should enter confirming
    resp = _run(handle_pending_teaching("yes"))
    assert "step 1" in resp.lower() or "step 2" in resp.lower(), (
        f"Expected step readback, got: {resp}"
    )
    assert "{contact}" in resp and "{message}" in resp, (
        f"Expected slotted values in readback, got: {resp}"
    )

    # Confirm and accept trigger
    resp = _run(handle_pending_teaching("yes"))
    resp = _run(handle_pending_teaching("yes"))
    assert any(w in resp.lower() for w in ("saved", "done", "got it"))

    proc = ps.get_procedure("send whatsapp message")
    assert proc is not None
    slots = _extract_slots_from_steps(proc["steps"])
    assert "contact" in slots and "message" in slots, f"slots={slots}"
    print(f"  PASS: slot_confirm flow → slots={slots}")
    ps.delete_procedure(proc["id"])


def test_slot_confirm_say_no():
    """User says 'no' to slot-confirm — literal stays as-is."""
    import assistant.actions as _actions

    start_teaching_session("type test thing")
    _run(handle_pending_teaching("type hello"))
    _run(handle_pending_teaching("press enter"))
    resp = _run(handle_pending_teaching("done"))
    assert "hello" in resp

    # Say no — hello stays literal
    resp = _run(handle_pending_teaching("no"))
    assert "step 1" in resp.lower() or "step 2" in resp.lower()
    assert "hello" in resp and "{hello}" not in resp, (
        f"Expected literal 'hello', got: {resp}"
    )

    # Confirm, accept, verify
    _run(handle_pending_teaching("yes"))
    _run(handle_pending_teaching("yes"))
    proc = ps.get_procedure("type test thing")
    assert proc is not None
    assert proc["steps"][0]["params"]["text"] == "hello"
    print(f"  PASS: slot_confirm no → literal kept")
    ps.delete_procedure(proc["id"])


# ─── Teach-start hints ─────────────────────────────────────────────────────


def test_teach_start_hints():
    resp = start_teaching_session("test hints")
    assert "keyboard shortcut" in resp.lower(), f"Missing keyboard hint: {resp}"
    assert "curly bracket" in resp.lower(), f"Missing curly bracket hint: {resp}"
    print(f"  PASS: both hints present")
    # Clean up session
    import assistant.actions as _actions
    _actions.teaching_session.clear()


# ─── Batch teaching ────────────────────────────────────────────────────────


def test_batch_basic():
    """Batch teaching parses multi-line steps and enters confirmation."""
    body = """1. open whatsapp
2. press ctrl+f
3. type {contact}
4. click on first result
5. type {message}
6. press enter"""

    resp = start_batch_teaching("send whatsapp message", body)
    # 6 lines but 'click on first result' expands to 2, so 7 steps
    assert "7 steps" in resp or "step 1" in resp.lower(), (
        f"Expected step summary, got: {resp}"
    )

    # Should be in confirming state (no suspects since all braced)
    resp = _run(handle_pending_teaching("yes"))
    assert any(w in resp.lower() for w in ("trigger", "phrase", "call", "say"))

    resp = _run(handle_pending_teaching("yes"))
    assert any(w in resp.lower() for w in ("saved", "done", "got it"))

    proc = ps.get_procedure("send whatsapp message")
    assert proc is not None
    assert len(proc["steps"]) == 7
    slots = _extract_slots_from_steps(proc["steps"])
    assert "contact" in slots and "message" in slots
    print(f"  PASS: batch mode → {len(proc['steps'])} steps, slots={slots}")
    ps.delete_procedure(proc["id"])


def test_batch_with_suspects():
    """Batch teaching with literal words triggers slot_confirm."""
    body = """open notepad
type filename
press ctrl+s"""

    resp = start_batch_teaching("save a file", body)
    assert "filename" in resp and "change" in resp.lower(), (
        f"Expected slot-confirm for 'filename', got: {resp}"
    )

    resp = _run(handle_pending_teaching("yes"))
    assert "step 1" in resp.lower() or "step 2" in resp.lower()

    proc_steps = None
    import assistant.actions as _actions
    if _actions.teaching_session.active:
        proc_steps = _actions.teaching_session.payload["steps"]
    assert proc_steps is not None
    assert proc_steps[1]["params"]["text"] == "{filename}"
    print(f"  PASS: batch + slot_confirm → '{proc_steps[1]['params']['text']}'")

    # Clean up
    _run(handle_pending_teaching("yes"))  # confirm
    _run(handle_pending_teaching("yes"))  # accept trigger
    proc = ps.get_procedure("save a file")
    if proc:
        ps.delete_procedure(proc["id"])


def test_batch_bad_lines():
    """Batch mode skips unparseable lines and reports them."""
    body = """open chrome
do a backflip
type hello"""

    resp = start_batch_teaching("weird procedure", body)
    assert "skipped" in resp.lower() or "2 steps" in resp.lower(), (
        f"Expected skip notice or 2-step count, got: {resp}"
    )
    print(f"  PASS: bad lines handled")

    # Clean up
    import assistant.actions as _actions
    _actions.teaching_session.clear()


# ─── Batch trigger detection (main.py) ──────────────────────────────────────


def test_batch_trigger_detection():
    from assistant.main import _match_batch_teach

    result = _match_batch_teach("create procedure for send email\n1. open gmail\n2. press c")
    assert result is not None, f"Expected match, got None"
    assert result[0] == "send email", f"Expected 'send email', got '{result[0]}'"
    assert "open gmail" in result[1]
    print(f"  PASS: batch trigger detected seed='{result[0]}'")

    # Single line should NOT match
    assert _match_batch_teach("create procedure for open notepad") is None
    print(f"  PASS: single line returns None")


# ─── Slot extraction JSON parser ────────────────────────────────────────────


def test_json_parser_clean():
    from assistant.procedure_executor import _parse_json_from_llm
    result = _parse_json_from_llm('{"contact": "samyak", "message": "hello"}')
    assert result == {"contact": "samyak", "message": "hello"}
    print(f"  PASS: clean JSON parsed")


def test_json_parser_wrapped():
    from assistant.procedure_executor import _parse_json_from_llm
    result = _parse_json_from_llm('Here is the result: {"contact": "samyak", "message": "hello"} done')
    assert result is not None and result["contact"] == "samyak"
    print(f"  PASS: wrapped JSON extracted")


def test_json_parser_codeblock():
    from assistant.procedure_executor import _parse_json_from_llm
    result = _parse_json_from_llm('```json\n{"contact": "samyak", "message": "hello"}\n```')
    assert result is not None and result["message"] == "hello"
    print(f"  PASS: code-block JSON extracted")


def test_json_parser_garbage():
    from assistant.procedure_executor import _parse_json_from_llm
    result = _parse_json_from_llm("I don't know what you mean")
    assert result is None
    print(f"  PASS: garbage returns None")


# ─── Runner ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    setup()

    tests = [
        ("Bug 1: longest trigger wins",          test_longest_trigger_wins),
        ("Bug 1: exact match priority",           test_exact_match_still_works),
        ("Parser: no auto-slot",                  test_no_auto_slot),
        ("Parser: braced stays braced",           test_braced_stays_braced),
        ("Parser: numbered steps",                test_numbered_step_parsing),
        ("Parser: positional click → keys",       test_positional_click_expansion),
        ("Parser: second item → 3 keys",          test_positional_click_second),
        ("Parser: press N times",                 test_repeat_modifier_times),
        ("Parser: press xN",                      test_repeat_modifier_x),
        ("Parser: single press unchanged",        test_repeat_modifier_single),
        ("Parser: regular click unchanged",       test_regular_click_unchanged),
        ("Split: inline paste",                   test_inline_paste_split),
        ("Split: newline paste",                  test_newline_paste_split),
        ("Collecting: inline paste batch",        test_inline_paste_during_collecting),
        ("Suspect literal detection",             test_find_suspects),
        ("Slot confirm: yes flow",                test_slot_confirm_flow),
        ("Slot confirm: no flow",                 test_slot_confirm_say_no),
        ("Teach-start hints",                     test_teach_start_hints),
        ("Batch: basic flow",                     test_batch_basic),
        ("Batch: with suspects",                  test_batch_with_suspects),
        ("Batch: bad lines",                      test_batch_bad_lines),
        ("Batch: trigger detection",              test_batch_trigger_detection),
        ("JSON parser: clean",                    test_json_parser_clean),
        ("JSON parser: wrapped",                  test_json_parser_wrapped),
        ("JSON parser: codeblock",                test_json_parser_codeblock),
        ("JSON parser: garbage",                  test_json_parser_garbage),
    ]

    import assistant.actions as _actions

    passed = 0
    failed = 0
    for name, fn in tests:
        _actions.teaching_session.clear()  # Reset between tests
        print(f"\n[TEST] {name}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
