"""
latch.py — the DOM tier, driven through the browser extension.

`ExtensionPage` satisfies `PageAdapter` (three members) and `ExtensionLocator`
satisfies `LocatorAdapter` (nine methods), so `dom.py` and everything above it
runs unchanged against a real browser the user already has open.

## `evaluate` only speaks one expression

The DOM tier calls `page.evaluate` exactly once, at `dom.py:979`, passing the
vendored query and a config dict. `ExtensionPage.evaluate` recognises that one
expression and routes it to `page.query`; **anything else raises**.

That refusal is the important part. MV3's content-script CSP forbids evaluating
JS that arrived over the wire, so there is no honest way to run an arbitrary
expression here — and the dishonest way is to return `None`. A silently-empty
evaluate reaches the planner as "the page has nothing on it", which is not an
error anyone sees; it is a confident wrong answer.

## `locator` only speaks one selector

Every locator in the tier originates at `dom.py:1055`, built from the index the
query just stamped. So `[data-latch-idx='N']` is the only shape that can arrive,
and any other selector is a caller that has invented one — refused with the
selector quoted rather than silently matching nothing.

## Errors are exceptions, never sentinels

An RPC error frame becomes a `LatchCallError` carrying its numeric code.
`.claude/rules/automation.md` records what the alternative cost:
`router._execute_dom_task` once turned an abort into `"__FALLBACK__"`, which is
not a failure report but an instruction to escalate a tier — so ESC re-triggered
TTS, minimised the operator's terminals and spent a vision call. A sentinel is
worse than a string error, and both are worse than raising.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from ...core import latch_protocol as proto
from ...io.api.extension_ws import LatchCallError, LatchConnection
from .dom_query_vendor import DOM_QUERY_JS
from .page_adapter import LocatorAdapter, PageAdapter

logger = logging.getLogger("browser.latch")

#: The only selector shape the DOM tier builds. Anchored at both ends: a
#: selector that merely *contains* the attribute is still one this driver did
#: not hand out, and matching it loosely would let an invented selector through.
_IDX_SELECTOR = re.compile(rf"^\[{re.escape(proto.IDX_ATTR)}='(\d+)'\]$")


class UnsupportedByExtension(NotImplementedError):
    """Something was asked of the extension driver that it cannot honestly do.

    Its own class so a caller can tell "this driver cannot" apart from "the page
    said no". They call for different responses: the first means fall back to a
    driver that can, the second means the plan was wrong.
    """


class ExtensionLocator:
    """One element, addressed by the index the query stamped on it.

    It carries the query params that produced that index, and every read-back
    re-queries with **those** params. Indices are assigned per query, in
    document order over whatever that query captured — so a read-back issued
    with a different `filter` re-stamps the page and index N becomes a
    different element entirely. The value read would be real, from a real
    element, and from the wrong one.
    """

    __slots__ = ("_conn", "_idx", "_query_params")

    def __init__(
        self, connection: LatchConnection, idx: int,
        query_params: dict | None = None,
    ) -> None:
        self._conn = connection
        self._idx = idx
        self._query_params = dict(query_params or {"filter": "interactive", "openComboboxes": False})

    async def _read_field(self, field: str, *, timeout: int) -> str:
        """Re-query with this locator's own params and read one field back."""
        result = await self._conn.call(
            proto.Rpc.QUERY, dict(self._query_params), timeout=timeout / 1000.0,
        )
        for element in result.get("elements") or []:
            if element.get("idx") == self._idx:
                return str(element.get(field, ""))
        raise LatchCallError(
            proto.Err.BAD_SELECTOR,
            f"element {self._idx} is no longer on the page",
            method=proto.Rpc.QUERY,
        )

    @property
    def idx(self) -> int:
        return self._idx

    async def _act(self, action: str, value: Any = None, *, timeout: int) -> Any:
        return await self._conn.call(
            proto.Rpc.ACT,
            {"idx": self._idx, "action": action, "value": value},
            timeout=timeout / 1000.0,
        )

    async def click(self, *, timeout: int = 10_000) -> None:
        await self._act("click", timeout=timeout)

    async def fill(self, value: str, *, timeout: int = 10_000) -> None:
        await self._act("fill", value, timeout=timeout)

    async def press(self, key: str, *, timeout: int = 10_000) -> None:
        await self._act("press", key, timeout=timeout)

    async def check(self, *, timeout: int = 10_000) -> None:
        await self._act("check", timeout=timeout)

    async def uncheck(self, *, timeout: int = 10_000) -> None:
        await self._act("uncheck", timeout=timeout)

    async def select_option(
        self,
        *,
        label: str | None = None,
        value: str | None = None,
        timeout: int = 10_000,
    ) -> Any:
        # `dom_executor` tries label first and falls back to value. The wire has
        # one `select` action which matches by label then by value, so a caller
        # that supplied neither is asking for nothing at all.
        wanted = label if label is not None else value
        if wanted is None:
            raise UnsupportedByExtension("select_option needs a label or a value")
        return await self._act("select", wanted, timeout=timeout)

    async def is_checked(self, *, timeout: int = 10_000) -> bool:
        value = await self._read_field("value", timeout=timeout)
        return value.lower() in ("true", "checked", "on")

    async def input_value(self, *, timeout: int = 10_000) -> str:
        return await self._read_field("value", timeout=timeout)

    async def evaluate(self, expression: str) -> Any:
        # `dom_executor` and `dom_filler` use this for one thing: reading back
        # the selected option's visible text. There is no way to run an
        # arbitrary expression under MV3's CSP, and returning None would let a
        # read-back silently succeed with nothing.
        raise UnsupportedByExtension(
            f"the extension driver cannot evaluate arbitrary JS on an element "
            f"(MV3 forbids it). Expression was: {expression[:120]!r}"
        )

    def __repr__(self) -> str:
        return f"ExtensionLocator(idx={self._idx})"


class ExtensionPage:
    """One tab, satisfying `PageAdapter`."""

    __slots__ = ("_conn", "_url", "_last_query_params")

    def __init__(self, connection: LatchConnection, url: str = "") -> None:
        self._conn = connection
        self._url = url
        self._last_query_params: dict = {"filter": "interactive", "openComboboxes": False}

    @property
    def url(self) -> str:
        """The URL as of the last query.

        `dom.py:972` reads this synchronously, so it cannot be an RPC. It is
        refreshed by `evaluate` from the *same* snapshot that produced the
        elements — which is why `page.query` returns a url at all. A url fetched
        in its own round trip could be from either side of a navigation the
        elements do not reflect, and comparing those two is exactly what
        `dom_orchestrator`'s navigation check does.
        """
        return self._url

    async def evaluate(self, expression: str, arg: Any = None) -> Any:
        if expression != DOM_QUERY_JS:
            raise UnsupportedByExtension(
                f"the extension driver can only run the vendored DOM query "
                f"(MV3 forbids evaluating JS sent over the wire). Asked for: "
                f"{expression[:120]!r}"
            )

        params = arg if isinstance(arg, dict) else {}
        query_params = {
            "filter": params.get("filter", "interactive"),
            "openComboboxes": bool(params.get("openComboboxes", False)),
        }
        # Remembered so every locator handed out from this tree reads back with
        # the same params that stamped its index.
        self._last_query_params = query_params
        result = await self._conn.call(proto.Rpc.QUERY, dict(query_params))
        url = result.get("url")
        if isinstance(url, str):
            self._url = url
        return result

    def locator(self, selector: str) -> LocatorAdapter:
        match = _IDX_SELECTOR.match(selector)
        if match is None:
            raise UnsupportedByExtension(
                f"the extension driver addresses elements only by the index the "
                f"query stamped, i.e. [{proto.IDX_ATTR}='N']. Asked for: {selector!r}"
            )
        return ExtensionLocator(self._conn, int(match.group(1)), self._last_query_params)

    def __repr__(self) -> str:
        return f"ExtensionPage(url={self._url!r})"
