"""
page_adapter.py — the seam between the DOM tier and whatever is driving the browser.

The perceive→plan→execute tier (`dom.py`, `dom_orchestrator.py`, `dom_executor.py`,
`dom_filler.py`, `dom_planner.py`, `dom_mapper.py`) is far less coupled to Playwright
than its size suggests. Measured by exhaustive grep, it touches a `Page` in three
places:

    dom.py:972   getattr(page, "url", "")
    dom.py:979   await page.evaluate(_DOM_QUERY_JS, {...})
    dom.py:1055  page.locator(f"[data-drover-idx='{idx}']")

Everything downstream consumes plain dicts produced by that one `evaluate`, and acts
through locators handed out by that one `locator` call and stored in
`PageDomTree.ref_to_locator`. There are no nested locators anywhere in the tier.

The locator surface is nine methods. Two of them -- `uncheck` and `is_checked`, both in
`dom_filler.fill_checkbox` -- were missed by the grep that produced the first draft of
this Protocol and were found only when pyright type-checked the retyped call sites.
Worth remembering: **a grep enumerates what you thought to search for; a type checker
enumerates what is actually called.** Run the checker before believing a surface count.

So the whole tier can run against any object exposing three members — which is what
`dom.py`'s own docstring has promised since it was written ("a Playwright `Page` (or
any object exposing `.evaluate(js, arg)` and `.locator(selector)`)"). This module
turns that prose promise into a type, so a second implementation can satisfy it and
be checked.

**What is deliberately NOT here.** `automation.py`'s `get_page` / `extract_text` /
`extract_structured` / `browse_url` family owns a page *lifecycle* —
`browser.new_context()`, `context.new_page()`, `page.context.close()`, `goto`,
`screenshot`, `wait_for_load_state`. A browser-extension driver has no "context": it
drives tabs the user already opened and must never close them. Those functions are
the bundled-Chromium path and stay typed to Playwright. Widening these Protocols to
cover them would force every future driver to fake a lifecycle it does not have.

**Timeouts are milliseconds**, matching Playwright and every call site in the tier.
The Protocol defaults are documentation only; the call sites always pass one.

The two `Playwright*` classes are pure delegation. They add no defaults, no retries,
no logging, and — importantly — they swallow nothing: `dom.py:979` relies on a
raising `evaluate` to set `evaluate_failed=True`, which `dom_orchestrator` reads as a
navigation signal. A wrapper that caught and returned `None` would erase it.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# ─── Protocols ────────────────────────────────────────────────────────────────


@runtime_checkable
class LocatorAdapter(Protocol):
    """The nine Locator methods the DOM tier actually calls.

    Every locator in the tier originates at `dom.py:1055` and is stored in
    `PageDomTree.ref_to_locator`; there are no chained or nested locators, which
    is why there is no `locator()` member here and no `.first`.
    """

    async def click(self, *, timeout: int = ...) -> None: ...

    async def fill(self, value: str, *, timeout: int = ...) -> None: ...

    async def input_value(self, *, timeout: int = ...) -> str: ...

    async def check(self, *, timeout: int = ...) -> None: ...

    async def uncheck(self, *, timeout: int = ...) -> None: ...

    async def is_checked(self, *, timeout: int = ...) -> bool: ...

    async def press(self, key: str, *, timeout: int = ...) -> None: ...

    async def select_option(
        self,
        *,
        label: str | None = None,
        value: str | None = None,
        timeout: int = ...,
    ) -> Any:
        """Keyword-only, deliberately.

        `dom_executor.py:202` tries `label=` and falls back to `value=` on
        failure. Playwright's *first positional* argument means match-by-value,
        so collapsing `label=` into a positional would silently turn by-label
        selection into by-value — and the fallback would then be indistinguishable
        from the primary attempt.
        """
        ...

    async def evaluate(self, expression: str) -> Any: ...


@runtime_checkable
class PageAdapter(Protocol):
    """The three Page members the DOM tier actually touches.

    `url` is read via `getattr(page, "url", "")` at `dom.py:972` and must stay a
    *live* read — `dom_orchestrator` compares urls across successive perceives to
    detect navigation, so a value snapshotted at construction would make that
    check go blind.
    """

    url: str

    async def evaluate(self, expression: str, arg: Any = None) -> Any: ...

    def locator(self, selector: str) -> LocatorAdapter: ...


# ─── Playwright pass-through ──────────────────────────────────────────────────


class PlaywrightLocator:
    """Wraps a Playwright `Locator`. Pure delegation.

    `select_option` forwards only the keyword the caller actually supplied.
    Passing `label=None` alongside `value=...` would hand Playwright two
    selectors at once, and `dom_executor`'s label→value fallback would stop
    behaving as written.
    """

    __slots__ = ("_locator",)

    def __init__(self, locator: Any) -> None:
        self._locator = locator

    async def click(self, *, timeout: int = 10_000) -> None:
        await self._locator.click(timeout=timeout)

    async def fill(self, value: str, *, timeout: int = 10_000) -> None:
        await self._locator.fill(value, timeout=timeout)

    async def input_value(self, *, timeout: int = 10_000) -> str:
        return await self._locator.input_value(timeout=timeout)

    async def check(self, *, timeout: int = 10_000) -> None:
        await self._locator.check(timeout=timeout)

    async def uncheck(self, *, timeout: int = 10_000) -> None:
        await self._locator.uncheck(timeout=timeout)

    async def is_checked(self, *, timeout: int = 10_000) -> bool:
        return await self._locator.is_checked(timeout=timeout)

    async def press(self, key: str, *, timeout: int = 10_000) -> None:
        await self._locator.press(key, timeout=timeout)

    async def select_option(
        self,
        *,
        label: str | None = None,
        value: str | None = None,
        timeout: int = 10_000,
    ) -> Any:
        kwargs: dict[str, Any] = {"timeout": timeout}
        if label is not None:
            kwargs["label"] = label
        if value is not None:
            kwargs["value"] = value
        return await self._locator.select_option(**kwargs)

    async def evaluate(self, expression: str) -> Any:
        return await self._locator.evaluate(expression)

    def __repr__(self) -> str:
        return f"PlaywrightLocator({self._locator!r})"


class PlaywrightPage:
    """Wraps a Playwright `Page`. Pure delegation.

    `locator()` stays synchronous because `dom.py:1055` calls it without
    awaiting; an async version would put a coroutine into `ref_to_locator` that
    every downstream consumer would then try to `.click()`.
    """

    __slots__ = ("_page",)

    def __init__(self, page: Any) -> None:
        self._page = page

    @property
    def url(self) -> str:
        return self._page.url

    async def evaluate(self, expression: str, arg: Any = None) -> Any:
        return await self._page.evaluate(expression, arg)

    def locator(self, selector: str) -> LocatorAdapter:
        return PlaywrightLocator(self._page.locator(selector))

    def __repr__(self) -> str:
        return f"PlaywrightPage({self._page!r})"
