"""
test_tp1f_manage.py — Tests for TP-1f: Voice commands to manage procedures.

Covers:
  1. Regex pattern matching (match_procedure_command)
  2. find_by_name_or_trigger fuzzy lookup
  3. Handler behavior (list, delete, rename, edit)
  4. Edit teaching flow (confirming → update instead of create)
"""

import asyncio
import json
import sqlite3
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.regex_router import match_procedure_command
from assistant.intent import IntentResult


# ─── 1. Regex Pattern Tests ─────────────────────────────────────────────────

def test_list_patterns():
    """List commands should match with action=list."""
    cases = [
        "list my procedures",
        "list procedures",
        "show my procedures",
        "show procedures",
        "show my routines",
        "list my routines",
        "list all procedures",
        "what procedures do I have",
        "which procedures are there",
    ]
    for text in cases:
        result = match_procedure_command(text)
        assert result is not None, f"Expected match for: {text!r}"
        assert result.intent == "manage_procedure", f"Wrong intent for: {text!r}"
        assert result.params["action"] == "list", f"Wrong action for: {text!r}"
    print(f"  [PASS] test_list_patterns ({len(cases)} cases)")


def test_delete_patterns():
    """Delete commands should extract procedure name."""
    cases = [
        ("delete procedure send a whatsapp", "send a whatsapp"),
        ("remove procedure open coding session", "open coding session"),
        ("forget procedure morning routine", "morning routine"),
        ("drop procedure daily standup", "daily standup"),
        ("delete the procedure send a whatsapp", "send a whatsapp"),
        ("delete the morning routine procedure", "morning routine"),
        ("remove the setup procedure", "setup"),
    ]
    for text, expected_name in cases:
        result = match_procedure_command(text)
        assert result is not None, f"Expected match for: {text!r}"
        assert result.params["action"] == "delete", f"Wrong action for: {text!r}"
        assert result.params["name"] == expected_name, (
            f"Wrong name for: {text!r} — got {result.params['name']!r}, expected {expected_name!r}"
        )
    print(f"  [PASS] test_delete_patterns ({len(cases)} cases)")


def test_rename_patterns():
    """Rename commands should extract old name and new trigger."""
    cases = [
        ("rename procedure send whatsapp to message mom", "send whatsapp", "message mom"),
        ("change procedure morning setup to start my day", "morning setup", "start my day"),
        ("rename procedure open chrome into launch browser", "open chrome", "launch browser"),
        ("rename the procedure daily standup to standup", "daily standup", "standup"),
    ]
    for text, expected_name, expected_new in cases:
        result = match_procedure_command(text)
        assert result is not None, f"Expected match for: {text!r}"
        assert result.params["action"] == "rename", f"Wrong action for: {text!r}"
        assert result.params["name"] == expected_name, (
            f"Wrong name for: {text!r} — got {result.params['name']!r}"
        )
        assert result.params["new_trigger"] == expected_new, (
            f"Wrong new_trigger for: {text!r} — got {result.params['new_trigger']!r}"
        )
    print(f"  [PASS] test_rename_patterns ({len(cases)} cases)")


def test_edit_patterns():
    """Edit commands should extract procedure name."""
    cases = [
        ("edit procedure send a whatsapp", "send a whatsapp"),
        ("modify procedure morning routine", "morning routine"),
        ("update procedure daily standup", "daily standup"),
        ("reteach procedure open coding session", "open coding session"),
        ("re-teach procedure send email", "send email"),
        ("redo procedure weekly review", "weekly review"),
        ("edit the morning routine procedure", "morning routine"),
        ("modify the procedure setup workspace", "setup workspace"),
    ]
    for text, expected_name in cases:
        result = match_procedure_command(text)
        assert result is not None, f"Expected match for: {text!r}"
        assert result.params["action"] == "edit", f"Wrong action for: {text!r}"
        assert result.params["name"] == expected_name, (
            f"Wrong name for: {text!r} — got {result.params['name']!r}"
        )
    print(f"  [PASS] test_edit_patterns ({len(cases)} cases)")


def test_non_matching_patterns():
    """These should NOT match procedure management patterns."""
    non_matches = [
        "open procedure editor",
        "teach me how to send a whatsapp",
        "delete the file",
        "list my shortcuts",
        "rename the file",
        "send a whatsapp",
        "play some music",
        "what's the time",
        "edit my resume",
    ]
    for text in non_matches:
        result = match_procedure_command(text)
        assert result is None, f"Unexpected match for: {text!r} → {result}"
    print(f"  [PASS] test_non_matching_patterns ({len(non_matches)} cases)")


def test_list_word_order_variants():
    """Regression: 'list all my procedures' must match (any order of my/all)."""
    cases = [
        "list all my procedures",
        "list my all procedures",
        "list all procedures",
        "show all my routines",
        "show my all procedures",
    ]
    for text in cases:
        result = match_procedure_command(text)
        assert result is not None, f"Expected match for: {text!r}"
        assert result.params["action"] == "list", f"Wrong action for: {text!r}"
    print(f"  [PASS] test_list_word_order_variants ({len(cases)} cases)")


def test_case_insensitivity():
    """Patterns should be case-insensitive."""
    cases = [
        "LIST MY PROCEDURES",
        "Delete Procedure send whatsapp",
        "EDIT PROCEDURE morning routine",
        "Show My Routines",
    ]
    for text in cases:
        result = match_procedure_command(text)
        assert result is not None, f"Expected match for: {text!r}"
    print(f"  [PASS] test_case_insensitivity ({len(cases)} cases)")


# ─── 2. find_by_name_or_trigger Tests ───────────────────────────────────────

def _setup_test_db():
    """Create an in-memory DB with test procedures."""
    from assistant import procedures as ps
    ps._conn = sqlite3.connect(":memory:")
    ps._conn.row_factory = sqlite3.Row
    ps._conn.executescript("""
        CREATE TABLE IF NOT EXISTS user_procedures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trigger TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            steps TEXT NOT NULL,
            backend TEXT NOT NULL DEFAULT 'auto',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            use_count INTEGER NOT NULL DEFAULT 0,
            last_used TEXT DEFAULT NULL,
            enabled INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS user_shortcuts (
            trigger TEXT PRIMARY KEY,
            intent TEXT NOT NULL,
            params_json TEXT NOT NULL DEFAULT '{}',
            description TEXT NOT NULL DEFAULT '',
            times_used INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)
    ps._conn.commit()
    return ps


def test_find_exact_trigger():
    ps = _setup_test_db()
    ps.create_procedure("send a whatsapp", "Send A Whatsapp", [{"type": "app", "action": "open", "params": {"name": "whatsapp"}}])
    result = ps.find_by_name_or_trigger("send a whatsapp")
    assert result is not None
    assert result["trigger"] == "send a whatsapp"
    print("  [PASS] test_find_exact_trigger")


def test_find_exact_name():
    ps = _setup_test_db()
    ps.create_procedure("start coding", "Morning Coding Setup", [{"type": "app", "action": "open", "params": {"name": "code"}}])
    result = ps.find_by_name_or_trigger("Morning Coding Setup")
    assert result is not None
    assert result["name"] == "Morning Coding Setup"
    print("  [PASS] test_find_exact_name")


def test_find_substring_name():
    ps = _setup_test_db()
    ps.create_procedure("do coding", "Morning Coding Setup", [{"type": "app", "action": "open", "params": {"name": "code"}}])
    result = ps.find_by_name_or_trigger("coding")
    assert result is not None
    assert result["name"] == "Morning Coding Setup"
    print("  [PASS] test_find_substring_name")


def test_find_substring_trigger():
    ps = _setup_test_db()
    ps.create_procedure("send a whatsapp to mom", "Whatsapp Mom", [{"type": "app", "action": "open", "params": {"name": "whatsapp"}}])
    result = ps.find_by_name_or_trigger("whatsapp")
    assert result is not None
    assert result["trigger"] == "send a whatsapp to mom"
    print("  [PASS] test_find_substring_trigger")


def test_find_text_contains_trigger():
    ps = _setup_test_db()
    ps.create_procedure("send whatsapp", "Send Whatsapp", [{"type": "app", "action": "open", "params": {"name": "whatsapp"}}])
    result = ps.find_by_name_or_trigger("please send whatsapp now")
    assert result is not None
    assert result["trigger"] == "send whatsapp"
    print("  [PASS] test_find_text_contains_trigger")


def test_find_word_boundary_disambiguation():
    """Regression: 'open work setup' must match 'open my work setup', NOT 'open work set'."""
    ps = _setup_test_db()
    ps.create_procedure("open my work setup", "Open My Work Setup",
                        [{"type": "app", "action": "open", "params": {"name": "code"}},
                         {"type": "app", "action": "open", "params": {"name": "chrome"}},
                         {"type": "app", "action": "open", "params": {"name": "slack"}}])
    pid2 = ps.create_procedure("open work set 2", "Open Work Set",
                        [{"type": "app", "action": "open", "params": {"name": "notepad"}}] * 11)
    result = ps.find_by_name_or_trigger("open work setup")
    assert result is not None
    assert result["trigger"] == "open my work setup", (
        f"Expected 'open my work setup' but got '{result['trigger']}' — "
        f"substring 'open work set' falsely matched inside 'open work setup'"
    )
    print("  [PASS] test_find_word_boundary_disambiguation")


def test_find_not_found():
    ps = _setup_test_db()
    ps.create_procedure("send whatsapp", "Send Whatsapp", [{"type": "app", "action": "open", "params": {"name": "whatsapp"}}])
    result = ps.find_by_name_or_trigger("play music")
    assert result is None
    print("  [PASS] test_find_not_found")


def test_find_respects_enabled():
    ps = _setup_test_db()
    pid = ps.create_procedure("old trigger", "Old Proc", [{"type": "app", "action": "open", "params": {"name": "x"}}])
    ps.delete_procedure(pid)
    result = ps.find_by_name_or_trigger("old trigger")
    assert result is None
    result_all = ps.find_by_name_or_trigger("old trigger", enabled_only=False)
    assert result_all is not None
    print("  [PASS] test_find_respects_enabled")


# ─── 3. Handler Tests ───────────────────────────────────────────────────────

def test_handler_list_empty():
    ps = _setup_test_db()
    from assistant import actions
    result = asyncio.run(
        actions.handle_manage_procedure({"action": "list"}, "")
    )
    assert "don't have" in result.lower() or "no procedures" in result.lower()
    print("  [PASS] test_handler_list_empty")


def test_handler_list_with_items():
    ps = _setup_test_db()
    ps.create_procedure("send whatsapp", "Send Whatsapp",
                        [{"type": "app", "action": "open", "params": {"name": "whatsapp"}}])
    ps.create_procedure("morning setup", "Morning Setup",
                        [{"type": "app", "action": "open", "params": {"name": "code"}},
                         {"type": "app", "action": "open", "params": {"name": "chrome"}}])
    from assistant import actions
    result = asyncio.run(
        actions.handle_manage_procedure({"action": "list"}, "")
    )
    assert "2 procedure" in result
    assert "send whatsapp" in result
    assert "morning setup" in result
    print("  [PASS] test_handler_list_with_items")


def test_handler_delete_success():
    ps = _setup_test_db()
    ps.create_procedure("send whatsapp", "Send Whatsapp",
                        [{"type": "app", "action": "open", "params": {"name": "whatsapp"}}])
    from assistant import actions
    result = asyncio.run(
        actions.handle_manage_procedure({"action": "delete", "name": "send whatsapp"}, "")
    )
    assert "send whatsapp" in result.lower()
    assert "delet" in result.lower() or "forgot" in result.lower() or "removed" in result.lower() or "gone" in result.lower()
    remaining = ps.list_procedures(enabled_only=True)
    assert len(remaining) == 0
    print("  [PASS] test_handler_delete_success")


def test_handler_delete_not_found():
    ps = _setup_test_db()
    from assistant import actions
    result = asyncio.run(
        actions.handle_manage_procedure({"action": "delete", "name": "nonexistent"}, "")
    )
    assert "don't have" in result.lower() or "can't find" in result.lower()
    print("  [PASS] test_handler_delete_not_found")


def test_handler_rename_success():
    ps = _setup_test_db()
    ps.create_procedure("send whatsapp", "Send Whatsapp",
                        [{"type": "app", "action": "open", "params": {"name": "whatsapp"}}])
    from assistant import actions
    result = asyncio.run(
        actions.handle_manage_procedure(
            {"action": "rename", "name": "send whatsapp", "new_trigger": "message mom"}, ""
        )
    )
    assert "message mom" in result.lower()
    proc = ps.get_procedure("message mom")
    assert proc is not None
    assert ps.get_procedure("send whatsapp") is None
    print("  [PASS] test_handler_rename_success")


def test_handler_rename_conflict():
    ps = _setup_test_db()
    ps.create_procedure("send whatsapp", "Send Whatsapp",
                        [{"type": "app", "action": "open", "params": {"name": "whatsapp"}}])
    ps.create_procedure("message mom", "Message Mom",
                        [{"type": "app", "action": "open", "params": {"name": "contacts"}}])
    from assistant import actions
    result = asyncio.run(
        actions.handle_manage_procedure(
            {"action": "rename", "name": "send whatsapp", "new_trigger": "message mom"}, ""
        )
    )
    assert "already" in result.lower() or "conflict" in result.lower() or "different" in result.lower()
    print("  [PASS] test_handler_rename_conflict")


def test_handler_edit_starts_teaching():
    ps = _setup_test_db()
    ps.create_procedure("send whatsapp", "Send Whatsapp",
                        [{"type": "app", "action": "open", "params": {"name": "whatsapp"}}])
    from assistant import actions
    actions.teaching_session.clear()
    result = asyncio.run(
        actions.handle_manage_procedure({"action": "edit", "name": "send whatsapp"}, "")
    )
    assert actions.teaching_session.active
    assert actions.teaching_session.payload["state"] == "collecting"
    assert actions.teaching_session.payload.get("_editing_proc_id") is not None
    assert actions.teaching_session.payload.get("_editing_trigger") == "send whatsapp"
    assert "send whatsapp" in result.lower()
    actions.teaching_session.clear()
    print("  [PASS] test_handler_edit_starts_teaching")


# ─── 4. Edit Teaching Flow Tests ────────────────────────────────────────────

def test_edit_teaching_updates_procedure():
    """After edit → collect steps → confirm → should update (not create) the procedure."""
    ps = _setup_test_db()
    pid = ps.create_procedure(
        "send whatsapp", "Send Whatsapp",
        [{"type": "app", "action": "open", "params": {"name": "whatsapp"}}]
    )
    from assistant import actions

    actions.teaching_session.set({
        "state": "confirming",
        "name_seed": "Send Whatsapp",
        "steps": [
            {"type": "app", "action": "open", "params": {"name": "chrome"}},
            {"type": "app", "action": "press_key", "params": {"key": "ctrl+t"}},
        ],
        "slots": [],
        "backend": "auto",
        "_editing_proc_id": pid,
        "_editing_trigger": "send whatsapp",
    })

    result = asyncio.run(
        actions.handle_pending_teaching("yes")
    )
    assert not actions.teaching_session.active
    assert "send whatsapp" in result.lower()

    updated = ps.get_procedure_by_id(pid)
    assert updated is not None
    assert len(updated["steps"]) == 2
    assert updated["steps"][0]["action"] == "open"
    assert updated["steps"][0]["params"]["name"] == "chrome"

    all_procs = ps.list_procedures(enabled_only=True)
    assert len(all_procs) == 1
    print("  [PASS] test_edit_teaching_updates_procedure")


def test_edit_teaching_restart_keeps_edit_mode():
    """If user says 'no' during confirm in edit mode, should re-enter collecting but keep edit context."""
    ps = _setup_test_db()
    pid = ps.create_procedure(
        "send whatsapp", "Send Whatsapp",
        [{"type": "app", "action": "open", "params": {"name": "whatsapp"}}]
    )
    from assistant import actions

    actions.teaching_session.set({
        "state": "confirming",
        "name_seed": "Send Whatsapp",
        "steps": [{"type": "app", "action": "open", "params": {"name": "chrome"}}],
        "slots": [],
        "backend": "auto",
        "_editing_proc_id": pid,
        "_editing_trigger": "send whatsapp",
    })

    result = asyncio.run(
        actions.handle_pending_teaching("no")
    )
    assert actions.teaching_session.active
    assert actions.teaching_session.payload["state"] == "collecting"
    assert actions.teaching_session.payload.get("_editing_proc_id") == pid
    actions.teaching_session.clear()
    print("  [PASS] test_edit_teaching_restart_keeps_edit_mode")


# ─── 5. Integration: LLM fallback via goal param ────────────────────────────

def test_handler_parses_goal_param():
    """When called via LLM intent (action not pre-parsed), handler extracts from goal."""
    ps = _setup_test_db()
    ps.create_procedure("send whatsapp", "Send Whatsapp",
                        [{"type": "app", "action": "open", "params": {"name": "whatsapp"}}])
    from assistant import actions
    result = asyncio.run(
        actions.handle_manage_procedure(
            {"goal": "list my procedures"}, ""
        )
    )
    assert "1 procedure" in result
    assert "send whatsapp" in result
    print("  [PASS] test_handler_parses_goal_param")


# ─── 6. Implicit procedure command tests ────────────────────────────────────

def test_implicit_edit_matches_procedure():
    """'edit open work set' (no 'procedure' keyword) should match if procedure exists."""
    ps = _setup_test_db()
    ps.create_procedure("open work set", "Open Work Set",
                        [{"type": "app", "action": "open", "params": {"name": "chrome"}}])
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from assistant.main import _match_implicit_proc_command
    result = _match_implicit_proc_command("edit open work set")
    assert result is not None, "Expected implicit edit to match"
    assert result.params["action"] == "edit"
    assert result.params["name"] == "open work set"
    print("  [PASS] test_implicit_edit_matches_procedure")


def test_implicit_delete_matches_procedure():
    """'delete open work set' (no 'procedure' keyword) should match if procedure exists."""
    ps = _setup_test_db()
    ps.create_procedure("open work set", "Open Work Set",
                        [{"type": "app", "action": "open", "params": {"name": "chrome"}}])
    from assistant.main import _match_implicit_proc_command
    result = _match_implicit_proc_command("delete open work set")
    assert result is not None, "Expected implicit delete to match"
    assert result.params["action"] == "delete"
    print("  [PASS] test_implicit_delete_matches_procedure")


def test_implicit_no_match_without_procedure():
    """'edit my resume' should NOT match if no procedure with that name exists."""
    ps = _setup_test_db()
    ps.create_procedure("open work set", "Open Work Set",
                        [{"type": "app", "action": "open", "params": {"name": "chrome"}}])
    from assistant.main import _match_implicit_proc_command
    result = _match_implicit_proc_command("edit my resume")
    assert result is None, "Should not match — no procedure called 'my resume'"
    result2 = _match_implicit_proc_command("delete the file")
    assert result2 is None, "Should not match — no procedure called 'the file'"
    print("  [PASS] test_implicit_no_match_without_procedure")


# ─── Run all tests ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== TP-1f: Procedure Management Tests ===\n")

    print("1. Regex patterns:")
    test_list_patterns()
    test_delete_patterns()
    test_rename_patterns()
    test_edit_patterns()
    test_non_matching_patterns()
    test_list_word_order_variants()
    test_case_insensitivity()

    print("\n2. find_by_name_or_trigger:")
    test_find_exact_trigger()
    test_find_exact_name()
    test_find_substring_name()
    test_find_substring_trigger()
    test_find_text_contains_trigger()
    test_find_word_boundary_disambiguation()
    test_find_not_found()
    test_find_respects_enabled()

    print("\n3. Handler behavior:")
    test_handler_list_empty()
    test_handler_list_with_items()
    test_handler_delete_success()
    test_handler_delete_not_found()
    test_handler_rename_success()
    test_handler_rename_conflict()
    test_handler_edit_starts_teaching()

    print("\n4. Edit teaching flow:")
    test_edit_teaching_updates_procedure()
    test_edit_teaching_restart_keeps_edit_mode()

    print("\n5. Integration:")
    test_handler_parses_goal_param()

    print("\n6. Implicit procedure commands:")
    test_implicit_edit_matches_procedure()
    test_implicit_delete_matches_procedure()
    test_implicit_no_match_without_procedure()

    print(f"\n=== ALL TESTS PASSED ===\n")
