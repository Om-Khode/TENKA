"""
test_extension_setup_intent.py — Drover Task 16: the setup intent.

What replaces the Chrome-shortcut generator. That one wrote launchers into the
Desktop and Start Menu so the user could open Chrome with a debug flag; this
one touches nothing on the operating system, because the extension drives the
browser already open. All it does is mint the credential and say where to paste
it.

Two properties carry weight:

  - **`undo` removes exactly what `setup` created, and nothing else.** Pinned by
    snapshotting the marker directory before and after, not by asserting the one
    file is gone — a handler that cleared the whole directory would satisfy the
    narrow check and take a neighbour's state with it.
  - **`preview` mints nothing.** A preview that quietly replaced the live
    credential would disconnect a working extension to answer a question about
    what it would do.

Run: py -3.11 -m pytest tests/test_extension_setup_intent.py -v
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

from assistant.io.api import extension_ws as ews  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """Point the credential store at a temp dir for every call in this file."""
    real_token_path = ews.token_path

    def scoped(home_arg=None):
        return real_token_path(tmp_path if home_arg is None else home_arg)

    monkeypatch.setattr(ews, "token_path", scoped)
    return tmp_path


@pytest.fixture()
def handler():
    from assistant.actions.browser_extension_setup import handle_browser_extension_setup
    return handle_browser_extension_setup


def _snapshot(directory: Path) -> set[str]:
    return {p.name for p in directory.rglob("*") if p.is_file()}


# ─── setup ───────────────────────────────────────────────────────────────


def test_setup_mints_a_credential_and_shows_the_steps(home, handler):
    msg = _run(handler({"mode": "setup"}))
    token = ews.read_token(home)
    assert token, "setup did not store a credential"
    assert token in msg, "the token was minted but never shown to the user"
    assert "Connect" in msg or "paste" in msg.lower(), "no install steps in the reply"


def test_setup_is_the_default_mode(home, handler):
    _run(handler({}))
    assert ews.read_token(home), "an empty params dict did nothing"


def test_the_stored_marker_is_schema_versioned(home, handler):
    _run(handler({"mode": "setup"}))
    raw = json.loads(ews.token_path(home).read_text(encoding="utf-8"))
    assert raw["schema_version"] == ews.TOKEN_SCHEMA_VERSION, (
        "the credential file carries no current schema version. Standing "
        "project rule: schema-version every on-disk marker and reject older ones."
    )


def test_setup_twice_replaces_rather_than_accumulating(home, handler):
    _run(handler({"mode": "setup"}))
    first = ews.read_token(home)
    before = _snapshot(home)

    _run(handler({"mode": "setup"}))
    second = ews.read_token(home)

    assert first != second, "a second setup reused the old credential"
    assert _snapshot(home) == before, (
        f"a second setup left extra files behind: {_snapshot(home) - before}"
    )


def test_the_reply_does_not_overclaim_what_was_installed(home, handler):
    # The old setup edited desktop and Start-Menu shortcuts. This one edits
    # nothing, and a reply implying otherwise sends the user looking for a
    # launcher that was never created.
    msg = _run(handler({"mode": "setup"})).lower()
    for overclaim in ("shortcut", "start menu", "desktop", "--remote-debugging-port"):
        assert overclaim not in msg, f"the reply claims it created a {overclaim}"


# ─── status ──────────────────────────────────────────────────────────────


def test_status_with_nothing_connected_says_so_without_calling_it_an_error(home, handler):
    ews.reset_state_for_test()
    msg = _run(handler({"mode": "status"}))
    assert "no browser extension is connected" in msg.lower()
    # Not connected is the normal state before setup, and the browser tier
    # falls back cleanly. Calling it a failure would send the user debugging.
    assert "error" not in msg.lower()
    assert "failed" not in msg.lower()


def test_status_names_the_connected_browser(home, handler):
    ews.reset_state_for_test()

    async def _noop(_frame):
        return None

    ews.register(ews.DroverConnection(send_json=_noop, browser_name="firefox"))
    try:
        msg = _run(handler({"mode": "status"}))
        assert "firefox" in msg.lower()
    finally:
        ews.reset_state_for_test()


def test_status_mints_nothing(home, handler):
    ews.reset_state_for_test()
    _run(handler({"mode": "status"}))
    assert ews.read_token(home) is None, "asking a question created a credential"


# ─── preview ─────────────────────────────────────────────────────────────


def test_preview_mints_nothing_and_disturbs_nothing(home, handler):
    _run(handler({"mode": "setup"}))
    before_token = ews.read_token(home)
    before_files = _snapshot(home)

    msg = _run(handler({"mode": "preview"}))

    assert ews.read_token(home) == before_token, (
        "preview replaced the live credential. A working extension would be "
        "disconnected by a question about what setup would do."
    )
    assert _snapshot(home) == before_files
    assert "would" in msg.lower(), "preview does not describe itself as hypothetical"


def test_preview_on_a_clean_machine_creates_nothing(home, handler):
    _run(handler({"mode": "preview"}))
    assert ews.read_token(home) is None
    assert _snapshot(home) == set()


# ─── undo ────────────────────────────────────────────────────────────────


def test_undo_removes_exactly_what_setup_created(home, handler):
    neighbour = home / "something_else.json"
    neighbour.write_text('{"not": "ours"}', encoding="utf-8")
    before = _snapshot(home)

    _run(handler({"mode": "setup"}))
    assert ews.read_token(home)

    _run(handler({"mode": "undo"}))

    assert ews.read_token(home) is None, "undo left the credential behind"
    assert _snapshot(home) == before, (
        f"undo did not restore the directory to its prior state. "
        f"Missing: {before - _snapshot(home)}; extra: {_snapshot(home) - before}"
    )
    assert neighbour.exists(), "undo deleted a file it did not create"


def test_undo_with_nothing_to_remove_is_not_an_error(home, handler):
    msg = _run(handler({"mode": "undo"}))
    assert "no extension credential" in msg.lower()


def test_undo_is_idempotent(home, handler):
    _run(handler({"mode": "setup"}))
    _run(handler({"mode": "undo"}))
    _run(handler({"mode": "undo"}))
    assert ews.read_token(home) is None


def test_after_undo_the_socket_refuses_everything(home, handler):
    """The point of undo, asserted through the handshake rather than the file.

    A removed credential must actually close the door. `evaluate_handshake`
    refuses when `expected_token` is falsy, which is what an absent store
    yields — the fail-closed path, exercised end to end.
    """
    from assistant.core import drover_protocol as proto

    _run(handler({"mode": "setup"}))
    token = ews.read_token(home)
    _run(handler({"mode": "undo"}))

    verdict = ews.evaluate_handshake(
        {
            "type": proto.Frame.HELLO,
            "protocolVersion": proto.PROTOCOL_VERSION,
            "domQuerySha256": "a" * 64,
            "token": token,
        },
        origin="chrome-extension://abc",
        expected_token=ews.read_token(home),
        expected_digest="a" * 64,
        occupied=False,
    )
    assert verdict.ok is False
    assert verdict.code == proto.Err.UNAUTHORIZED


# ─── the intent is wired ─────────────────────────────────────────────────


def test_the_intent_is_registered_under_its_new_name():
    from assistant.actions.registry import tool_registry
    import assistant.actions  # noqa: F401 — registration side effect

    assert tool_registry.has("browser_extension_setup")
    assert not tool_registry.has("browser_cdp_setup"), (
        "the old intent name is still registered; the classifier could still "
        "route to a handler for a mechanism that no longer exists"
    )


def test_the_intent_requires_execute():
    from assistant.core.intent_capabilities import REQUIRED_CAPABILITY
    from assistant.io.api.vault import Capability

    assert REQUIRED_CAPABILITY["browser_extension_setup"] == Capability.EXECUTE, (
        "the setup intent mints a credential; it may not be reachable with "
        "anything less than EXECUTE"
    )


def test_the_intent_persists_authority_and_a_raise_cannot_spend_it():
    """It moved out of TRANSIENT_AUTHORITY, and the reason is not the credential.

    The credential carries no intent authority -- the extension listener's
    ceiling is empty. What persists is a capability: before setup the browser
    tier drives a bundled Chromium signed out of everything; after it, and from
    then on, it drives the browser the user is actually signed into.

    A raise is minted at the keyboard and expires. Left transient, a half-hour
    raise could buy permanent reach into the user's logged-in sessions, which
    is the same shape as converting a raise into permanent local execution by
    installing a monitor with it.
    """
    from assistant.core.intent_capabilities import (
        PERSISTS_AUTHORITY,
        TRANSIENT_AUTHORITY,
    )

    assert "browser_extension_setup" in PERSISTS_AUTHORITY
    assert "browser_extension_setup" not in TRANSIENT_AUTHORITY, (
        "the intent is in both sets. The pair is exhaustive with no default in "
        "either direction, precisely so an answer like this cannot be silent."
    )
