"""
test_browser_tabs.py — tab control through the extension.

The DOM tier drives one page. This is the other half, and its interesting
property is not that it works — it is what it does when it is *not sure*.

Closing the wrong tab loses a page the user was reading and there is no undo.
So an ambiguous request asks rather than guesses, and "close the tab" with
nothing to disambiguate is refused outright. The read-only verbs are free to be
generous; the destructive one is not.

**No app-specific matching.** A tab is found by comparing what the user said
against its own title and url. Nothing here knows what any site is, and adding
one changes nothing — THE rule applies to this file as much as to the router.

Run: py -3.11 -m pytest tests/test_browser_tabs.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

from assistant.core import latch_protocol as proto  # noqa: E402
from assistant.io.api import extension_ws as ews  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


TABS = [
    {"id": 1, "title": "Inbox — Mail", "url": "https://mail.example.invalid/u/0",
     "active": False, "windowId": 10},
    {"id": 2, "title": "Rust Tutorial — YouTube", "url": "https://youtube.example.invalid/watch?v=1",
     "active": True, "windowId": 10},
    {"id": 3, "title": "Docs", "url": "https://docs.example.invalid/mail-api",
     "active": False, "windowId": 10},
]


class FakeConnection:
    def __init__(self, tabs=None, error=None):
        self.tabs = TABS if tabs is None else tabs
        self.calls: list[tuple[str, dict]] = []
        self.error = error

    async def call(self, method, params=None, *, timeout=30.0):
        self.calls.append((method, dict(params or {})))
        if self.error is not None:
            raise self.error
        if method == proto.Rpc.TABS_LIST:
            return {"tabs": self.tabs}
        return {"ok": True}

    def methods(self):
        return [m for m, _ in self.calls]


@pytest.fixture()
def handler():
    from assistant.actions.browser_tabs import handle_browser_tabs
    return handle_browser_tabs


@pytest.fixture()
def connected(monkeypatch):
    conn = FakeConnection()
    monkeypatch.setattr(ews, "current_connection", lambda: conn)
    return conn


@pytest.fixture()
def disconnected(monkeypatch):
    monkeypatch.setattr(ews, "current_connection", lambda: None)


# ─── Not connected ───────────────────────────────────────────────────────


def test_with_no_extension_it_says_so_and_offers_the_fix(handler, disconnected):
    msg = _run(handler({"action": "list"}))
    assert "isn't connected" in msg
    assert "set up the browser extension" in msg, (
        "the reply names a problem and no way out of it"
    )


def test_it_does_not_pretend_a_bundled_browser_would_do(handler, disconnected):
    # Page tasks fall back to bundled Chromium. Tabs cannot: they are
    # inherently about the window the user is looking at.
    msg = _run(handler({"action": "close", "query": "mail"})).lower()
    assert "closed" not in msg


# ─── list ────────────────────────────────────────────────────────────────


def test_list_names_every_tab_and_marks_the_active_one(handler, connected):
    msg = _run(handler({"action": "list"}))
    assert "3 tabs open" in msg
    for title in ("Inbox — Mail", "Rust Tutorial — YouTube", "Docs"):
        assert title in msg
    assert "you're on Rust Tutorial — YouTube" in msg


def test_list_is_the_default_action(handler, connected):
    _run(handler({}))
    assert connected.methods() == [proto.Rpc.TABS_LIST]


def test_list_with_no_tabs_says_so(handler, monkeypatch):
    monkeypatch.setattr(ews, "current_connection", lambda: FakeConnection(tabs=[]))
    assert "No tabs open" in _run(handler({"action": "list"}))


def test_a_tab_with_no_title_falls_back_to_its_url(handler, monkeypatch):
    conn = FakeConnection(tabs=[{"id": 9, "title": "", "url": "https://x.invalid/p",
                                 "active": True, "windowId": 1}])
    monkeypatch.setattr(ews, "current_connection", lambda: conn)
    assert "https://x.invalid/p" in _run(handler({"action": "list"}))


# ─── close: the destructive one ──────────────────────────────────────────


def test_close_with_no_query_refuses_rather_than_guessing(handler, connected):
    """"Close the tab" is not a request TENKA can safely resolve.

    Defaulting to the active tab would mean a misheard sentence closes whatever
    the user is reading, with no undo. It asks instead.
    """
    msg = _run(handler({"action": "close"}))
    assert "Which tab" in msg
    assert proto.Rpc.TABS_CLOSE not in connected.methods(), "it closed something anyway"


def test_close_matches_a_tab_by_its_own_title(handler, connected):
    msg = _run(handler({"action": "close", "query": "youtube"}))
    assert "Closed" in msg
    method, params = connected.calls[-1]
    assert method == proto.Rpc.TABS_CLOSE
    assert params["tabId"] == 2


def test_close_prefers_a_title_word_over_a_url_substring(handler, connected):
    """"mail" appears in tab 1's title and inside tab 3's url path.

    A user saying "the mail tab" means the one called Mail. Scoring title hits
    above url hits is what makes that true without knowing what mail is.
    """
    _run(handler({"action": "close", "query": "mail"}))
    assert connected.calls[-1][1]["tabId"] == 1


def test_an_ambiguous_close_asks_instead_of_picking(handler, monkeypatch):
    conn = FakeConnection(tabs=[
        {"id": 1, "title": "Notes", "url": "https://a.invalid/", "active": False, "windowId": 1},
        {"id": 2, "title": "Notes", "url": "https://b.invalid/", "active": False, "windowId": 1},
    ])
    monkeypatch.setattr(ews, "current_connection", lambda: conn)

    msg = _run(handler({"action": "close", "query": "notes"}))
    assert "More than one tab matches" in msg
    assert proto.Rpc.TABS_CLOSE not in conn.methods(), (
        "a tie was resolved by picking one. Closing the wrong tab loses a page "
        "the user cannot get back."
    )


def test_a_clear_winner_is_not_treated_as_ambiguous(handler, connected):
    # The other half. A handler that asked whenever more than one tab scored
    # anything at all would refuse almost every real request.
    msg = _run(handler({"action": "close", "query": "youtube"}))
    assert "More than one" not in msg
    assert "Closed" in msg


def test_close_with_nothing_matching_says_so(handler, connected):
    msg = _run(handler({"action": "close", "query": "spreadsheet"}))
    assert "No open tab" in msg
    assert proto.Rpc.TABS_CLOSE not in connected.methods()


# ─── switch ──────────────────────────────────────────────────────────────


def test_switch_activates_the_matching_tab(handler, connected):
    msg = _run(handler({"action": "switch", "query": "docs"}))
    assert "Switched to Docs" in msg
    method, params = connected.calls[-1]
    assert method == proto.Rpc.TABS_ACTIVATE
    assert params["tabId"] == 3


def test_switch_with_no_query_also_refuses(handler, connected):
    _run(handler({"action": "switch"}))
    assert proto.Rpc.TABS_ACTIVATE not in connected.methods()


# ─── open is deliberately NOT here ───────────────────────────────────────


def test_open_is_not_an_action_this_intent_claims(handler, connected):
    """`open_browser` already opens a URL in the user's default browser.

    A second intent for the same request means two rows competing in the
    classifier's table and whichever it happens to prefer winning -- the exact
    over-claim shape that sends "open a tab for X" to a GUI vision loop, which
    is what it did before this was removed.
    """
    msg = _run(handler({"action": "open", "query": "example.invalid"}))
    assert "don't know how to" in msg
    assert connected.calls == [], "it opened a tab anyway"


# ─── Errors ──────────────────────────────────────────────────────────────


def test_an_unknown_action_is_refused(handler, connected):
    msg = _run(handler({"action": "levitate"}))
    assert "don't know how to" in msg
    assert connected.calls == []


def test_the_browsers_own_refusal_is_passed_through(handler, monkeypatch):
    """Its reason beats anything this layer could invent."""
    from assistant.io.api.extension_ws import LatchCallError

    conn = FakeConnection(error=LatchCallError(proto.Err.NO_TAB, "tab 2 no longer exists"))
    monkeypatch.setattr(ews, "current_connection", lambda: conn)

    msg = _run(handler({"action": "list"}))
    assert "tab 2 no longer exists" in msg


def test_an_unexpected_crash_does_not_escape(handler, monkeypatch):
    conn = FakeConnection(error=RuntimeError("socket exploded"))
    monkeypatch.setattr(ews, "current_connection", lambda: conn)
    msg = _run(handler({"action": "list"}))
    assert "Something went wrong" in msg
    assert "socket exploded" not in msg, "the raw exception reached the user"


# ─── No app-specific knowledge ───────────────────────────────────────────


def test_nothing_here_names_a_website():
    """THE rule, asserted rather than trusted.

    Matching is against whatever the tab calls itself. A brand name in this
    file would mean the handler works for the sites someone thought of and
    quietly fails for every other.
    """
    import re

    source = (Path(__file__).parent.parent / "assistant" / "actions"
              / "browser_tabs.py").read_text(encoding="utf-8")
    code = re.sub(r'"""[\s\S]*?"""', "", source)
    code = re.sub(r"#.*$", "", code, flags=re.M)

    for brand in ("youtube", "gmail", "google", "facebook", "twitter", "reddit",
                  "amazon", "netflix", "spotify", "github"):
        assert brand not in code.lower(), (
            f"{brand!r} appears in the handler's code. Tabs are matched on what "
            f"they call themselves; a site named here is a site that works and "
            f"a thousand that do not."
        )
