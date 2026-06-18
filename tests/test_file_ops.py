"""
test_file_ops.py — Tests for the file_ops handler extraction (S1c).

Verifies:
  1. Helpers (_resolve_file_path, _resolve_dest_folder, _set_pending_destructive)
  2. Pending file search flow
  3. Pending destructive op flow
  4. handle_read_file delegation
  5. PendingState-based state access via import assistant.actions pattern

Run: python -m pytest tests/test_file_ops.py -v
"""

import asyncio
import sys
import os
import types
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── 1. _set_pending_destructive ──────────────────────────────────────────────

def test_set_pending_destructive_sets_state():
    """_set_pending_destructive should set pending_destructive on the actions module."""
    import assistant.actions as actions
    from assistant.actions.file_ops import _set_pending_destructive

    try:
        _set_pending_destructive("delete", Path("C:/test.txt"), {})
        assert actions.pending_destructive.active
        assert actions.pending_destructive.payload["op"] == "delete"
        assert actions.pending_destructive.payload["path"] == Path("C:/test.txt")
        assert actions.pending_destructive._ts > 0
    finally:
        actions.pending_destructive.clear()


def test_set_pending_destructive_merges_extra():
    """Extra dict should be merged into pending state."""
    import assistant.actions as actions
    from assistant.actions.file_ops import _set_pending_destructive

    try:
        _set_pending_destructive("write", Path("C:/f.txt"), {"content": "hi"})
        assert actions.pending_destructive.payload["content"] == "hi"
        assert actions.pending_destructive.payload["op"] == "write"
    finally:
        actions.pending_destructive.clear()


# ─── 2. _resolve_file_path ───────────────────────────────────────────────────

def test_resolve_file_path_absolute():
    """Absolute path that exists should be returned directly."""
    from assistant.actions.file_ops import _resolve_file_path

    with patch("assistant.actions.file_ops.Path") as MockPath:
        mock_p = MagicMock()
        mock_p.is_absolute.return_value = True
        mock_p.exists.return_value = True
        MockPath.return_value = mock_p
        MockPath.home = Path.home

        result = _resolve_file_path("C:\\Users\\test\\file.txt")
        assert result is mock_p


def test_resolve_file_path_sandbox_match(tmp_path):
    """File in SANDBOX_DIR should be returned.

    Earlier revisions of this test patched ``SANDBOX_DIR.__truediv__`` to
    redirect the / operator — Python ≥3.11 makes dunder attrs on PurePath
    read-only, breaking the patch. Use a real sandbox directory instead;
    the resolver's behavior is the same and the test is more honest.
    """
    from assistant.actions.file_ops import _resolve_file_path

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    target = sandbox / "todo.txt"
    target.write_text("hello", encoding="utf-8")

    with patch("assistant.file_manager.find_files", return_value=[]), \
         patch("assistant.config.SANDBOX_DIR", sandbox):
        result = _resolve_file_path("todo.txt")
        assert result == target
        assert result.exists()


def test_resolve_file_path_returns_none():
    """Should return None when file not found anywhere."""
    from assistant.actions.file_ops import _resolve_file_path

    with patch("assistant.file_manager.find_files", return_value=[]), \
         patch("assistant.config.SANDBOX_DIR", Path("C:/sandbox")):
        with patch.object(Path, "is_absolute", return_value=False), \
             patch.object(Path, "exists", return_value=False):
            result = _resolve_file_path("nonexistent.txt")
            assert result is None


# ─── 2b. _extract_explicit_path ──────────────────────────────────────────────

def test_extract_explicit_path_quoted(tmp_path):
    """A quoted absolute path inside a goal string is honored verbatim."""
    from assistant.actions.file_ops import _extract_explicit_path

    target = tmp_path / "World Building - Final.txt"
    target.write_text("hi", encoding="utf-8")
    goal = f"read this file '{target}'"
    assert _extract_explicit_path(goal) == target


def test_extract_explicit_path_unquoted_with_spaces(tmp_path):
    """An unquoted absolute path containing spaces is recovered (exists guard)."""
    from assistant.actions.file_ops import _extract_explicit_path

    target = tmp_path / "MC Character Profile.docx"
    target.write_text("x", encoding="utf-8")
    goal = f"read file {target}"
    assert _extract_explicit_path(goal) == target


def test_extract_explicit_path_forward_slashes(tmp_path):
    """Forward-slash paths (as the STT/LLM often emit) resolve too."""
    from assistant.actions.file_ops import _extract_explicit_path

    target = tmp_path / "notes.txt"
    target.write_text("x", encoding="utf-8")
    goal = f"& '{str(target).replace(chr(92), '/')}' can you read this file?"
    assert _extract_explicit_path(goal) == target


def test_extract_explicit_path_nonexistent_returns_none():
    """A path that doesn't exist must not be returned (no false positives)."""
    from assistant.actions.file_ops import _extract_explicit_path

    assert _extract_explicit_path("read 'Z:/nope/missing.txt'") is None
    assert _extract_explicit_path("just some words, no path") is None


def test_resolve_file_path_cwd_match(tmp_path, monkeypatch):
    """A bare filename present in the current working dir resolves via cwd."""
    from assistant.actions.file_ops import _resolve_file_path

    target = tmp_path / "report.csv"
    target.write_text("a,b", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with patch("assistant.file_manager.find_files", return_value=[]), \
         patch("assistant.config.SANDBOX_DIR", tmp_path / "sandbox"):
        result = _resolve_file_path("report.csv")
        assert result == target


# ─── 3. _resolve_dest_folder ─────────────────────────────────────────────────

def test_resolve_dest_folder_known():
    """Known folder name like 'desktop' should resolve."""
    from assistant.actions.file_ops import _resolve_dest_folder

    mock_desktop = MagicMock(spec=Path)
    mock_desktop.exists.return_value = True
    mock_desktop.resolve.return_value = mock_desktop

    with patch("assistant.file_manager.get_user_folder") as mock_get:
        mock_get.return_value = mock_desktop
        with patch.object(Path, "home", return_value=Path("C:/Users/test")):
            results = _resolve_dest_folder("desktop")
            assert len(results) >= 1


def test_resolve_dest_folder_absolute():
    """Absolute path that is a directory should resolve to itself."""
    from assistant.actions.file_ops import _resolve_dest_folder

    with patch("assistant.file_manager.get_user_folder", return_value=Path("C:/fake")):
        with patch.object(Path, "is_absolute", return_value=True), \
             patch.object(Path, "is_dir", return_value=True):
            results = _resolve_dest_folder("C:\\Users\\Alex\\Custom")
            assert len(results) == 1


# ─── 4. Pending file search flow ─────────────────────────────────────────────

def test_pending_file_search_none_returns_none():
    """When no pending search, handler returns None."""
    import assistant.actions as actions
    from assistant.actions.file_ops import handle_pending_file_search

    try:
        actions.pending_file_search.clear()
        result = asyncio.run(
            handle_pending_file_search("fast search")
        )
        assert result is None
    finally:
        actions.pending_file_search.clear()


def test_pending_file_search_cancel():
    """Saying 'no' should cancel pending file search."""
    import assistant.actions as actions
    from assistant.actions.file_ops import handle_pending_file_search

    try:
        actions.pending_file_search.set({"name": "test.txt", "tier": 1})
        result = asyncio.run(
            handle_pending_file_search("no thanks")
        )
        assert result is not None
        assert actions.pending_file_search.payload is None
    finally:
        actions.pending_file_search.clear()


def test_pending_file_search_fast():
    """Saying 'fast' should start a tier-2 search.

    Current behavior: when the tier-2 search returns no results, the handler
    transitions pending_file_search to tier-2 (so a follow-up 'deeper' can
    escalate to tier-3). When results ARE found, pending state clears.
    """
    import assistant.actions as actions
    from assistant.actions.file_ops import handle_pending_file_search

    try:
        actions.pending_file_search.set({"name": "resume.pdf", "tier": 1})
        with patch("assistant.file_manager.find_files", return_value=[]), \
             patch("assistant.file_manager.TIER2_TIMEOUT", 10), \
             patch("assistant.file_manager.TIER3_TIMEOUT", 120):
            result = asyncio.run(
                handle_pending_file_search("fast search")
            )
            assert result is not None
            assert "fast" in result.lower() or "3-level" in result.lower()
            # Tier-1 state was cleared; on no-results the handler installs a
            # new tier-2 pending state so the user can escalate to tier-3.
            payload = actions.pending_file_search.payload
            assert payload is None or payload.get("tier") == 2
    finally:
        actions.pending_file_search.clear()


# ─── 5. Pending destructive op flow ──────────────────────────────────────────

def test_pending_destructive_none_returns_none():
    """When no pending destructive op, handler returns None."""
    import assistant.actions as actions
    from assistant.actions.file_ops import handle_pending_destructive

    try:
        actions.pending_destructive.clear()
        result = asyncio.run(
            handle_pending_destructive("confirm")
        )
        assert result is None
    finally:
        actions.pending_destructive.clear()


def test_pending_destructive_timeout():
    """Expired pending op should auto-clear and pass through to fresh routing.

    PendingState auto-clears expired payloads on read (see assistant/pending.py)
    so the handler sees payload=None and returns None — the user's input then
    falls through to normal intent routing rather than getting a stale
    'confirm/cancel' prompt. The invariant we care about: an expired pending
    op MUST be cleared and MUST NOT capture the next utterance.
    """
    import assistant.actions as actions
    from assistant.actions.file_ops import handle_pending_destructive

    try:
        actions.pending_destructive.set({"op": "delete", "path": Path("C:/x.txt")})
        actions.pending_destructive._ts = time.time() - 999
        result = asyncio.run(
            handle_pending_destructive("confirm")
        )
        # Expired state → handler does not handle, lets the normal pipeline run
        assert result is None
        assert actions.pending_destructive.payload is None
        assert not actions.pending_destructive.active
    finally:
        actions.pending_destructive.clear()


def test_pending_destructive_cancel():
    """Cancelling should clear pending state."""
    import assistant.actions as actions
    from assistant.actions.file_ops import handle_pending_destructive

    try:
        actions.pending_destructive.set({"op": "delete", "path": Path("C:/x.txt")})
        result = asyncio.run(
            handle_pending_destructive("cancel")
        )
        assert result is not None
        assert actions.pending_destructive.payload is None
    finally:
        actions.pending_destructive.clear()


def test_pending_destructive_confirm():
    """Confirming should execute the operation."""
    import assistant.actions as actions
    from assistant.actions.file_ops import handle_pending_destructive

    try:
        actions.pending_destructive.set({"op": "delete", "path": Path("C:/x.txt")})
        with patch("assistant.file_manager.delete_path", return_value="Deleted!"):
            result = asyncio.run(
                handle_pending_destructive("confirm")
            )
            assert result == "Deleted!"
            assert actions.pending_destructive.payload is None
    finally:
        actions.pending_destructive.clear()


def test_pending_destructive_reprompt():
    """Ambiguous input should re-prompt, not pass through."""
    import assistant.actions as actions
    from assistant.actions.file_ops import handle_pending_destructive

    try:
        actions.pending_destructive.set({"op": "rename", "path": Path("C:/x.txt")})
        result = asyncio.run(
            handle_pending_destructive("what is this about")
        )
        assert result is not None
        assert "confirm" in result.lower()
        assert actions.pending_destructive.active
    finally:
        actions.pending_destructive.clear()


# ─── 6. Module integration ───────────────────────────────────────────────────

def test_tools_dict_has_file_handlers():
    """Tool registry should resolve file_task / read_file to the file_ops handlers.

    Post-RG-1 the hardcoded ``_TOOLS`` dict was replaced with a typed
    ``tool_registry`` populated via @tool_registry.decorator("intent") on each
    handler module — actions/__init__.py dispatches via tool_registry.get().
    """
    from assistant.actions.registry import tool_registry
    from assistant.actions.file_ops import handle_file_task, handle_read_file

    assert tool_registry.get("file_task") is handle_file_task
    # handle_read_file is an internal helper invoked by handle_file_task —
    # it is exported but not registered as a top-level intent.
    assert callable(handle_read_file)


def test_file_ops_functions_importable():
    """All public file_ops functions should be importable from the submodule."""
    from assistant.actions.file_ops import (
        handle_read_file,
        handle_file_task,
        handle_pending_file_search,
        handle_pending_destructive,
        _execute_destructive_op,
        _set_pending_destructive,
        _resolve_file_path,
        _resolve_dest_folder,
    )
    assert callable(handle_read_file)
    assert callable(handle_file_task)
    assert callable(handle_pending_file_search)
    assert callable(handle_pending_destructive)
    assert callable(_execute_destructive_op)
    assert callable(_set_pending_destructive)
    assert callable(_resolve_file_path)
    assert callable(_resolve_dest_folder)


# ─── 7. _execute_destructive_op ──────────────────────────────────────────────

def test_execute_destructive_write():
    """Write op should call file_manager.write_file."""
    import assistant.actions as actions
    from assistant.actions.file_ops import _execute_destructive_op

    try:
        actions.pending_destructive.set({
            "op": "write",
            "path": Path("C:/sandbox/test.txt"),
            "content": "hello world",
        })
        with patch("assistant.file_manager.write_file", return_value="Created test.txt"):
            result = asyncio.run(_execute_destructive_op())
            assert result == "Created test.txt"
    finally:
        actions.pending_destructive.clear()


def test_execute_destructive_rename():
    """Rename op should call file_manager.rename_path."""
    import assistant.actions as actions
    from assistant.actions.file_ops import _execute_destructive_op

    try:
        actions.pending_destructive.set({
            "op": "rename",
            "path": Path("C:/test.txt"),
            "new_name": "test2.txt",
        })
        with patch("assistant.file_manager.rename_path", return_value=("Renamed!", Path("C:/test2.txt"))):
            result = asyncio.run(_execute_destructive_op())
            assert result == "Renamed!"
    finally:
        actions.pending_destructive.clear()


def test_execute_destructive_no_pending():
    """Should return error when no pending op."""
    import assistant.actions as actions
    from assistant.actions.file_ops import _execute_destructive_op

    try:
        actions.pending_destructive.clear()
        result = asyncio.run(_execute_destructive_op())
        assert "no pending" in result.lower()
    finally:
        actions.pending_destructive.clear()
