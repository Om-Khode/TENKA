"""
test_drover_page.py — Drover Task 12: the DOM tier driven through the extension.

The properties here are mostly about **refusing rather than returning nothing**.
This driver genuinely cannot do some of what a Playwright page can — MV3's CSP
forbids evaluating JS that arrived over the wire — and every one of those gaps
has an obvious wrong answer that looks like success:

  - an unsupported `evaluate` returning `None` reaches the planner as "the page
    has nothing on it", which is a confident wrong answer, not an error;
  - an unrecognised selector matching nothing reads as "the element is gone";
  - an error frame flattened into a sentinel string is not a failure report but
    an instruction to escalate a tier (`.claude/rules/automation.md`).

The other half is an index-space property that is easy to get wrong and
invisible when you do: indices are assigned **per query**, so a read-back issued
with different query params re-stamps the page and index N becomes a different
element. The value read is real, from a real element, and from the wrong one.

Run: py -3.11 -m pytest tests/test_drover_page.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

from assistant.automation.browser.dom_query_vendor import DOM_QUERY_JS  # noqa: E402
from assistant.automation.browser.drover import (  # noqa: E402
    ExtensionLocator,
    ExtensionPage,
    UnsupportedByExtension,
)
from assistant.automation.browser.page_adapter import (  # noqa: E402
    LocatorAdapter,
    PageAdapter,
)
from assistant.core import drover_protocol as proto  # noqa: E402
from assistant.io.api.extension_ws import DroverCallError  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


class FakeConnection:
    """Records every call and answers from a scripted table."""

    def __init__(self, replies: dict | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.replies = replies or {}
        self.error: DroverCallError | None = None

    async def call(self, method, params=None, *, timeout=30.0):
        self.calls.append((method, dict(params or {}), timeout))
        if self.error is not None:
            raise self.error
        reply = self.replies.get(method)
        return reply({}) if callable(reply) else (reply or {})

    def last(self):
        return self.calls[-1]


def _tree(elements, url="https://example.invalid/page"):
    return {"elements": elements, "url": url, "viewport": [800, 600],
            "validation_errors": []}


def _page(replies=None):
    conn = FakeConnection(replies)
    return conn, ExtensionPage(conn)


# ─── Conformance ─────────────────────────────────────────────────────────


def test_the_driver_satisfies_both_protocols():
    conn = FakeConnection()
    assert isinstance(ExtensionPage(conn), PageAdapter)
    assert isinstance(ExtensionLocator(conn, 0), LocatorAdapter)


# ─── evaluate ────────────────────────────────────────────────────────────


def test_the_vendored_query_is_routed_to_page_query():
    conn, page = _page({proto.Rpc.QUERY: _tree([{"idx": 0, "value": "x"}])})
    result = _run(page.evaluate(DOM_QUERY_JS, {"filter": "form", "openComboboxes": True}))
    method, params, _ = conn.last()
    assert method == proto.Rpc.QUERY
    assert params == {"filter": "form", "openComboboxes": True}
    assert result["elements"] == [{"idx": 0, "value": "x"}]


def test_query_params_default_when_the_caller_passes_none():
    conn, page = _page({proto.Rpc.QUERY: _tree([])})
    _run(page.evaluate(DOM_QUERY_JS))
    _, params, _ = conn.last()
    assert params == {"filter": "interactive", "openComboboxes": False}


@pytest.mark.parametrize("expression", [
    "document.body.innerText",
    "() => 1 + 1",
    "el => el.value",
    DOM_QUERY_JS + "\n// tampered",
    DOM_QUERY_JS[:-5],
    "",
])
def test_any_other_expression_raises_rather_than_returning_none(expression):
    _, page = _page()
    with pytest.raises(UnsupportedByExtension) as excinfo:
        _run(page.evaluate(expression))
    assert "MV3" in str(excinfo.value) or "vendored" in str(excinfo.value)


def test_the_refusal_quotes_what_was_asked_for():
    # An operator reading the log needs to know which expression was refused,
    # or the message is just "no".
    _, page = _page()
    with pytest.raises(UnsupportedByExtension) as excinfo:
        _run(page.evaluate("document.title"))
    assert "document.title" in str(excinfo.value)


# ─── url ─────────────────────────────────────────────────────────────────


def test_url_is_refreshed_from_the_query_snapshot():
    conn, page = _page({proto.Rpc.QUERY: _tree([], url="https://example.invalid/after")})
    assert page.url == ""
    _run(page.evaluate(DOM_QUERY_JS))
    assert page.url == "https://example.invalid/after"


def test_url_is_read_synchronously():
    # dom.py:972 does `getattr(page, "url", "")` with no await. A coroutine
    # here would be truthy, never compared equal to anything, and navigation
    # detection would silently never fire.
    _, page = _page()
    assert isinstance(page.url, str)


def test_a_query_without_a_url_leaves_the_previous_one_alone():
    conn = FakeConnection({proto.Rpc.QUERY: {"elements": [], "viewport": [0, 0]}})
    page = ExtensionPage(conn, url="https://example.invalid/known")
    _run(page.evaluate(DOM_QUERY_JS))
    assert page.url == "https://example.invalid/known", (
        "a malformed reply blanked the url. An empty url reads downstream as "
        "'the page navigated to nowhere', which is a navigation event that "
        "never happened."
    )


# ─── locator ─────────────────────────────────────────────────────────────


def test_the_stamped_selector_resolves_to_a_locator():
    _, page = _page()
    loc = page.locator(f"[{proto.IDX_ATTR}='7']")
    assert isinstance(loc, LocatorAdapter)
    assert loc.idx == 7


@pytest.mark.parametrize("selector", [
    "#submit",
    "button.primary",
    "[data-tenka-idx='7']",                       # the old name
    f"[{proto.IDX_ATTR}]",                        # no index
    f"[{proto.IDX_ATTR}='abc']",                  # not a number
    f"div [{proto.IDX_ATTR}='7']",                # descendant, not the bare shape
    f"[{proto.IDX_ATTR}='7'] > span",             # child, not the bare shape
    "",
])
def test_any_other_selector_raises_with_the_selector_quoted(selector):
    _, page = _page()
    with pytest.raises(UnsupportedByExtension) as excinfo:
        page.locator(selector)
    assert repr(selector) in str(excinfo.value) or selector in str(excinfo.value)


# ─── Actions ─────────────────────────────────────────────────────────────


def test_click_sends_an_act_frame_for_that_index():
    conn, page = _page({proto.Rpc.ACT: {"ok": True}})
    _run(page.locator(f"[{proto.IDX_ATTR}='3']").click(timeout=10_000))
    method, params, timeout = conn.last()
    assert method == proto.Rpc.ACT
    assert params["idx"] == 3
    assert params["action"] == "click"
    assert timeout == pytest.approx(10.0), "milliseconds were not converted to seconds"


def test_fill_carries_its_value():
    conn, page = _page({proto.Rpc.ACT: {"ok": True}})
    _run(page.locator(f"[{proto.IDX_ATTR}='1']").fill("hello", timeout=30_000))
    _, params, timeout = conn.last()
    assert params == {"idx": 1, "action": "fill", "value": "hello"}
    assert timeout == pytest.approx(30.0)


def test_select_option_prefers_label_and_falls_back_to_value():
    conn, page = _page({proto.Rpc.ACT: {"ok": True}})
    loc = page.locator(f"[{proto.IDX_ATTR}='2']")

    _run(loc.select_option(label="India", timeout=10_000))
    assert conn.last()[1]["value"] == "India"

    _run(loc.select_option(value="in", timeout=10_000))
    assert conn.last()[1]["value"] == "in"


def test_select_option_with_neither_raises_rather_than_selecting_nothing():
    _, page = _page({proto.Rpc.ACT: {"ok": True}})
    with pytest.raises(UnsupportedByExtension):
        _run(page.locator(f"[{proto.IDX_ATTR}='2']").select_option(timeout=10_000))


def test_check_and_uncheck_are_distinct_actions():
    conn, page = _page({proto.Rpc.ACT: {"ok": True}})
    loc = page.locator(f"[{proto.IDX_ATTR}='4']")
    _run(loc.check(timeout=10_000))
    assert conn.last()[1]["action"] == "check"
    _run(loc.uncheck(timeout=10_000))
    assert conn.last()[1]["action"] == "uncheck"


# ─── Read-back and the index space ───────────────────────────────────────


def test_input_value_reads_the_element_back():
    conn, page = _page({proto.Rpc.QUERY: _tree([
        {"idx": 0, "value": "other"}, {"idx": 5, "value": "typed"},
    ])})
    _run(page.evaluate(DOM_QUERY_JS))
    got = _run(page.locator(f"[{proto.IDX_ATTR}='5']").input_value(timeout=10_000))
    assert got == "typed"


def test_a_read_back_uses_the_params_that_stamped_the_index():
    """The index-space property.

    Indices are assigned per query, in document order over whatever that query
    captured. A read-back issued with a different `filter` re-stamps the page,
    and index N becomes a different element — the value returned is real, from a
    real element, and from the wrong one. Nothing raises.
    """
    conn, page = _page({proto.Rpc.QUERY: _tree([{"idx": 5, "value": "typed"}])})
    _run(page.evaluate(DOM_QUERY_JS, {"filter": "form", "openComboboxes": True}))
    conn.calls.clear()

    _run(page.locator(f"[{proto.IDX_ATTR}='5']").input_value(timeout=10_000))

    _, params, _ = conn.last()
    assert params == {"filter": "form", "openComboboxes": True}, (
        f"the read-back queried with {params}, not the params that produced the "
        f"index. Index N under a different filter is a different element."
    )


def test_a_read_back_for_a_vanished_element_raises():
    conn, page = _page({proto.Rpc.QUERY: _tree([{"idx": 0, "value": "x"}])})
    _run(page.evaluate(DOM_QUERY_JS))
    with pytest.raises(DroverCallError) as excinfo:
        _run(page.locator(f"[{proto.IDX_ATTR}='9']").input_value(timeout=10_000))
    assert excinfo.value.code == proto.Err.BAD_SELECTOR


def test_is_checked_reads_a_boolean_from_the_same_snapshot():
    conn, page = _page({proto.Rpc.QUERY: _tree([{"idx": 2, "value": "true"}])})
    _run(page.evaluate(DOM_QUERY_JS))
    assert _run(page.locator(f"[{proto.IDX_ATTR}='2']").is_checked(timeout=10_000)) is True


def test_is_checked_is_false_for_an_unchecked_box():
    conn, page = _page({proto.Rpc.QUERY: _tree([{"idx": 2, "value": ""}])})
    _run(page.evaluate(DOM_QUERY_JS))
    assert _run(page.locator(f"[{proto.IDX_ATTR}='2']").is_checked(timeout=10_000)) is False


def test_locator_evaluate_raises_rather_than_reading_back_nothing():
    # dom_executor uses this to read a select's chosen option text. Returning
    # None would make the read-back "succeed" with an empty value, and the
    # executor treats a successful read-back as confirmation.
    _, page = _page()
    with pytest.raises(UnsupportedByExtension):
        _run(page.locator(f"[{proto.IDX_ATTR}='0']").evaluate("el => el.value"))


# ─── Errors ──────────────────────────────────────────────────────────────


def test_an_error_frame_propagates_with_its_code_and_is_not_a_sentinel():
    conn, page = _page({proto.Rpc.ACT: {"ok": True}})
    conn.error = DroverCallError(proto.Err.INJECTION_BLOCKED, "page CSP refused")

    with pytest.raises(DroverCallError) as excinfo:
        _run(page.locator(f"[{proto.IDX_ATTR}='0']").click(timeout=10_000))
    assert excinfo.value.code == proto.Err.INJECTION_BLOCKED


def test_an_error_during_evaluate_propagates_rather_than_becoming_an_empty_tree():
    """The most dangerous flattening available here.

    An empty element tree is indistinguishable from a page with nothing on it,
    so swallowing a transport error into `{"elements": []}` produces a confident
    "there is nothing to click" instead of a failure.
    """
    conn, page = _page()
    conn.error = DroverCallError(proto.Err.TIMEOUT, "no reply")
    with pytest.raises(DroverCallError):
        _run(page.evaluate(DOM_QUERY_JS))
