"""
test_browser_event_source.py — Drover Task 14: browser events reach the bus.

The failure this file is written against is documented in
`event_sources/base.py`'s `normalize_process_name` docstring, and it is the
quietest one available: a monitor whose filter cannot match the events its
source emits is created, loads, reports active, and never fires. No error
anywhere. A filter that matches nothing looks exactly like an event that never
happened.

So the central assertion runs a real monitor dict through the real
`event_bus.check_dispatch` against a real emitted event. Comparing the two by
hand — asserting `source_app == "firefox"` and separately that a filter says
"firefox" — passes just as well when the comparison in between is broken, which
is precisely the bug.

Run: py -3.11 -m pytest tests/test_browser_event_source.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

from assistant.automation.event_bus import check_dispatch, make_dedup_key  # noqa: E402
from assistant.automation.event_sources.base import EventSource  # noqa: E402
from assistant.automation.event_sources.browser import BrowserEventSource  # noqa: E402


class FakeConnection:
    def __init__(self, browser_name="firefox"):
        self.browser_name = browser_name
        self.callbacks: list = []

    def on_event(self, cb):
        self.callbacks.append(cb)

    def remove_event_callback(self, cb):
        if cb in self.callbacks:
            self.callbacks.remove(cb)

    def emit(self, frame):
        for cb in list(self.callbacks):
            cb(frame)


@pytest.fixture()
def wired():
    conn = FakeConnection()
    events: list[dict] = []
    source = BrowserEventSource(lambda: conn)
    source.start(events.append)
    return conn, source, events


def monitor(event_type: str, source_filter: str | None = None) -> dict:
    return {"event_type": event_type, "source_filter": source_filter, "cooldown_secs": 5}


# ─── Protocol ────────────────────────────────────────────────────────────


def test_it_satisfies_the_event_source_protocol():
    assert isinstance(BrowserEventSource(lambda: None), EventSource)


def test_it_declares_every_event_it_can_emit():
    assert BrowserEventSource.event_types == frozenset({
        "browser_tab_opened", "browser_tab_closed", "browser_navigated",
        "browser_download_started", "browser_download_finished",
    })


# ─── The matching property ───────────────────────────────────────────────


def test_a_monitor_filter_actually_matches_an_emitted_event(wired):
    """The whole point of the file.

    Asserted through the real `check_dispatch`, not by comparing fields by
    hand: a hand comparison passes just as well when the normalisation between
    them is broken, and a broken normalisation is a monitor that reports active
    and never fires.
    """
    conn, _, events = wired
    conn.emit({"event": "navigated", "tabId": 3,
               "url": "https://example.invalid/x", "title": "Example"})
    assert events, "nothing was emitted at all"

    assert check_dispatch(
        monitor("browser_navigated", "firefox"), events[0], now=1000.0,
    ) is True, (
        "a monitor filtering on the browser name does not match the event this "
        "source emits. That monitor would load, report active, and never fire."
    )


def test_a_filter_for_a_different_browser_does_not_match(wired):
    # The other half. A source whose events matched every filter would pass the
    # test above and make every monitor fire on everything.
    conn, _, events = wired
    conn.emit({"event": "navigated", "tabId": 1, "url": "https://example.invalid/", "title": "T"})
    assert check_dispatch(monitor("browser_navigated", "chrome"), events[0], now=1000.0) is False


def test_a_filter_written_with_an_exe_suffix_still_matches(wired):
    # `normalize_process_name` strips `.exe`, and the parse prompt's examples
    # are bare names while models sometimes write the suffix. Both must land on
    # the same comparison — this is the exact mismatch the docstring records.
    conn, _, events = wired
    conn.emit({"event": "navigated", "tabId": 1, "url": "https://example.invalid/", "title": "T"})
    assert check_dispatch(monitor("browser_navigated", "firefox.exe"), events[0], now=1000.0) is True


def test_a_monitor_with_no_filter_matches_any_browser(wired):
    conn, _, events = wired
    conn.emit({"event": "tab_opened", "tabId": 1, "url": "https://example.invalid/", "title": "T"})
    assert check_dispatch(monitor("browser_tab_opened"), events[0], now=1000.0) is True


def test_the_wrong_event_type_does_not_match(wired):
    conn, _, events = wired
    conn.emit({"event": "tab_opened", "tabId": 1, "url": "https://example.invalid/", "title": "T"})
    assert check_dispatch(monitor("browser_navigated", "firefox"), events[0], now=1000.0) is False


# ─── Dedup ───────────────────────────────────────────────────────────────


def test_distinct_pages_are_not_deduplicated_into_one(wired):
    """`window_title` is half the dedup key.

    Without it every event from one browser collapses to the same key, and the
    bus drops all but the first within the cooldown — so a monitor fires once
    per browser session instead of once per page.
    """
    conn, _, events = wired
    conn.emit({"event": "navigated", "tabId": 1, "url": "https://example.invalid/a", "title": "A"})
    conn.emit({"event": "navigated", "tabId": 1, "url": "https://example.invalid/b", "title": "B"})
    assert make_dedup_key(events[0]) != make_dedup_key(events[1])


def test_a_page_with_no_title_still_gets_a_distinct_key(wired):
    conn, _, events = wired
    conn.emit({"event": "navigated", "tabId": 1, "url": "https://example.invalid/a", "title": ""})
    conn.emit({"event": "navigated", "tabId": 1, "url": "https://example.invalid/b", "title": ""})
    assert make_dedup_key(events[0]) != make_dedup_key(events[1]), (
        "an untitled page produced the same dedup key as another. Plenty of "
        "pages have no title; the url is the fallback for exactly this."
    )


# ─── Translation ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("wire,expected", [
    ("tab_opened", "browser_tab_opened"),
    ("tab_closed", "browser_tab_closed"),
    ("navigated", "browser_navigated"),
    ("download_started", "browser_download_started"),
    ("download_finished", "browser_download_finished"),
])
def test_every_wire_name_maps_to_its_bus_name(wired, wire, expected):
    conn, _, events = wired
    conn.emit({"event": wire, "tabId": 1})
    assert events[-1]["event_type"] == expected


def test_an_unknown_wire_event_is_dropped_not_forwarded(wired):
    # Enumerated, not computed: an event a future extension invents must not
    # silently become a new monitor type nobody declared.
    conn, _, events = wired
    conn.emit({"event": "something_new", "tabId": 1})
    assert events == []


def test_the_payload_carries_url_title_and_tab(wired):
    conn, _, events = wired
    conn.emit({"event": "navigated", "tabId": 42,
               "url": "https://example.invalid/p", "title": "Page"})
    e = events[0]
    assert e["url"] == "https://example.invalid/p"
    assert e["title"] == "Page"
    assert e["tab_id"] == 42
    assert e["timestamp"]


# ─── Lifecycle ───────────────────────────────────────────────────────────


def test_start_is_idempotent_and_does_not_double_dispatch():
    conn = FakeConnection()
    events: list[dict] = []
    source = BrowserEventSource(lambda: conn)
    source.start(events.append)
    source.start(events.append)
    source.poll()

    conn.emit({"event": "tab_opened", "tabId": 1})
    assert len(events) == 1, (
        f"one event dispatched {len(events)} times. `start()` runs again on "
        f"every reconnect, and a doubled event reads downstream as the user "
        f"doing the thing twice."
    )


def test_start_is_idempotent_even_without_a_removable_callback():
    """The guard's real job.

    `_detach()` normally undoes the previous subscription, which makes a second
    `start()` harmless — so removing the guard leaves every other test green.
    But `remove_event_callback` is not part of the EventSource contract and a
    connection may not have one; then detach is a no-op and each `start()`
    stacks another subscription. `start()` runs again on every reconnect, so
    over a long session one page load becomes N dispatches, and downstream that
    reads as the user doing the thing N times.
    """
    class NoRemove(FakeConnection):
        remove_event_callback = None  # present but unusable, as getattr sees it

    conn = NoRemove()
    events: list[dict] = []
    source = BrowserEventSource(lambda: conn)
    source.start(events.append)
    source.start(events.append)
    source.start(events.append)

    conn.emit({"event": "tab_opened", "tabId": 1})
    assert len(events) == 1, (
        f"one event dispatched {len(events)} times against a connection with "
        f"no removable callback"
    )


def test_stop_detaches_and_later_events_are_dropped():
    conn = FakeConnection()
    events: list[dict] = []
    source = BrowserEventSource(lambda: conn)
    source.start(events.append)
    conn.emit({"event": "tab_opened", "tabId": 1})
    source.stop()
    conn.emit({"event": "tab_opened", "tabId": 2})
    assert len(events) == 1, "an event arrived after stop()"


def test_a_reconnect_moves_the_subscription_to_the_new_connection():
    """The source holds a getter, not a connection.

    The extension reconnects on its own schedule. A source bound to one
    instance would keep listening to a socket that closed an hour ago and
    report nothing, while the live one went unheard.
    """
    first = FakeConnection("firefox")
    second = FakeConnection("chrome")
    current = [first]
    events: list[dict] = []

    source = BrowserEventSource(lambda: current[0])
    source.start(events.append)

    current[0] = second
    source.poll()

    second.emit({"event": "tab_opened", "tabId": 9})
    assert len(events) == 1
    assert events[0]["source_app"] == "chrome"

    first.emit({"event": "tab_opened", "tabId": 10})
    assert len(events) == 1, "the source is still listening to the old connection"


def test_starting_with_no_connection_is_not_an_error():
    source = BrowserEventSource(lambda: None)
    source.start(lambda _e: None)
    source.stop()


# ─── The source is actually started ──────────────────────────────────────
#
# Everything above this line passed while `EventBus` never instantiated this
# class. The source was written, registered in the source registry, and tested
# — and no browser event ever reached a monitor, because nothing constructed it.
#
# Same shape as the extension listener that was never bound: the mechanism was
# right and the plug was missing. A unit test cannot see that gap, so it is
# asserted structurally here.


def test_the_event_bus_constructs_the_browser_source():
    import ast
    import inspect

    from assistant.automation.event_bus import EventBus

    tree = ast.parse(inspect.getsource(EventBus.start).lstrip())
    constructed = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "BrowserEventSource"
    ]
    assert constructed, (
        "EventBus.start() never constructs BrowserEventSource. The class can be "
        "correct, registered and fully tested while no browser event ever "
        "reaches a monitor."
    )


def test_the_bus_subscribes_rather_than_polls():
    """The extension connects and reconnects on its own schedule.

    A source attached only at startup listens to whatever was connected then --
    which, on a machine where the browser starts after the assistant, is
    nothing at all, forever.
    """
    import inspect

    from assistant.automation.event_bus import EventBus

    source = inspect.getsource(EventBus.start)
    assert "on_connect" in source, (
        "the bus does not subscribe to connections, so a browser that starts "
        "after the assistant is never heard"
    )


def test_a_failing_browser_source_does_not_stop_the_bus():
    """The browser is one source of three.

    A browser that is not running, or an import that fails, must not take
    window and media monitors down with it.
    """
    import inspect

    from assistant.automation.event_bus import EventBus

    source = inspect.getsource(EventBus.start)
    head = source[: source.index("BrowserEventSource")]
    assert "try:" in head[-500:], (
        "the browser source is constructed outside a try block; an import "
        "failure there would stop the whole event bus from starting"
    )


def test_the_bus_stops_the_browser_source():
    import inspect

    from assistant.automation.event_bus import EventBus

    source = inspect.getsource(EventBus.stop)
    assert "browser_source" in source and ".stop()" in source, (
        "EventBus.stop() leaves the browser source attached, so a stopped bus "
        "keeps dispatching events into a dispatcher it no longer owns"
    )
