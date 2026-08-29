"""
latch_protocol.py — the Latch wire protocol, mirrored from the extension's `protocol.js`.

Latch is a browser driver: it exposes primitives over a loopback WebSocket and
holds no model, no planner and no task loop. This module is the Python half of
its contract. The JavaScript half is `src/shared/protocol.js` in the extension
repo, and the two are changed together with `PROTOCOL_VERSION` bumped.

Constants only. No transport, no state — so that a reader checking whether the
two halves agree has one short file to compare, and so that `test_latch_protocol`
can assert the mirroring without importing anything that opens a socket.

**It lives in `core/` because both sides of the layering need it.** The
WebSocket endpoint is in `io/api/`, which may import `core/` and `config` and
nothing else; the driver that speaks the protocol is in `automation/browser/`.
A shared vocabulary between two tiers that may not import each other belongs
below both of them, and a protocol table is exactly the kind of dependency-free
constant `core/` is for.
"""
from __future__ import annotations

from typing import Final

#: Bumped on ANY breaking change to a frame shape, a method name, or the element
#: schema. Compared in the `hello` frame and refused outright on a mismatch
#: rather than negotiated down — a driver that half-speaks a protocol fails in
#: the middle of a task, which is worse than not connecting at all.
PROTOCOL_VERSION: Final[int] = 1

#: The attribute the query stamps on every captured element, and the only way a
#: client addresses one afterwards. `dom.py` builds `[data-latch-idx='N']`
#: selectors from it; the shared JS writes it via `dataset.latchIdx`.
IDX_ATTR: Final[str] = "data-latch-idx"


class Frame:
    """`type` values. Every frame on the wire carries one."""

    HELLO: Final[str] = "hello"
    WELCOME: Final[str] = "welcome"
    REJECT: Final[str] = "reject"
    REQUEST: Final[str] = "request"
    RESPONSE: Final[str] = "response"
    EVENT: Final[str] = "event"
    #: Client -> server, periodically, while the socket is open. Carries
    #: nothing and expects no reply. It exists because an open socket does not
    #: keep an MV3 background context alive -- only traffic does -- so without
    #: it the extension is suspended after ~30s idle and the connection drops.
    PING: Final[str] = "ping"


class Rpc:
    """Method names, namespaced by what they touch."""

    GOTO: Final[str] = "page.goto"
    QUERY: Final[str] = "page.query"
    ACT: Final[str] = "page.act"
    SCREENSHOT: Final[str] = "page.screenshot"
    INFO: Final[str] = "page.info"
    WAIT_LOAD: Final[str] = "page.waitLoad"
    TABS_LIST: Final[str] = "tabs.list"
    TABS_ACTIVATE: Final[str] = "tabs.activate"
    TABS_OPEN: Final[str] = "tabs.open"
    TABS_CLOSE: Final[str] = "tabs.close"


class Err:
    """Error codes. Numeric and stable.

    A caller branches on these. Matching on message text is how a client breaks
    the day someone improves an error message.
    """

    NO_TAB: Final[int] = 1001
    TIMEOUT: Final[int] = 1002
    INJECTION_BLOCKED: Final[int] = 1003
    BAD_SELECTOR: Final[int] = 1004
    PROTOCOL_MISMATCH: Final[int] = 1005
    HASH_MISMATCH: Final[int] = 1006
    UNAUTHORIZED: Final[int] = 1007
    UNKNOWN_METHOD: Final[int] = 1008
    INTERNAL: Final[int] = 1009


#: Every key `page.query` puts on an element, and the complete set.
#:
#: Declared here rather than inferred from what `dom.py` happens to read,
#: because it is the one part of the protocol two codebases must agree on field
#: by field. Each side asserts against this list rather than against the other's
#: output: comparing two implementations to each other passes happily when both
#: are wrong the same way.
ELEMENT_KEYS: Final[frozenset[str]] = frozenset({
    "aria_invalid", "autocomplete", "bounds", "enabled", "form_id", "idx",
    "in_dialog", "name", "options", "placeholder", "role", "tag", "type",
    "value", "visible",
})

#: Top-level keys of a `page.query` result. `url` is read in the same snapshot
#: as the elements: comparing a URL fetched separately against elements fetched
#: a moment later is precisely the mistake navigation detection would make.
QUERY_RESULT_KEYS: Final[frozenset[str]] = frozenset({
    "elements", "url", "validation_errors", "viewport",
})

#: Actions `page.act` accepts.
ACTIONS: Final[frozenset[str]] = frozenset({
    "click", "fill", "press", "select", "check", "uncheck",
})

#: Event names the extension pushes. Neutral on the wire; the event source
#: prefixes them with `browser_` on the way in.
EVENTS: Final[frozenset[str]] = frozenset({
    "tab_opened", "tab_closed", "navigated",
    "download_started", "download_finished",
})

#: Origin schemes a browser extension can present. Anything else on the
#: extension listener is not an extension, whatever it claims in `hello`.
EXTENSION_ORIGIN_SCHEMES: Final[tuple[str, ...]] = (
    "chrome-extension://",
    "moz-extension://",
)
