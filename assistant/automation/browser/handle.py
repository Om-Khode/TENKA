"""
handle.py — which browser the DOM tier is about to drive, and why.

Two kinds:

  `drover`    — the browser the user already has open, through the extension.
               Their profile, their logins, no launch flags.
  `bundled`  — TENKA's own Chromium. Signed out of everything, and the fallback
               whenever the extension is not connected.

Both hand back a `PageAdapter`, so `dom.py` and everything above it neither
knows nor cares which one it got.

## The single log line

A downgrade to `bundled` emits exactly one INFO line naming why. This is not
decoration. The extension not being connected is invisible from the outside —
the task still runs, it just runs against a browser with none of the user's
sessions, and the first symptom is a login wall on a site they are signed into.
The CDP tier this replaces kept the same discipline, and it was the only thing
that made "why did it use the bundled browser?" answerable from a log.

The line is emitted per resolution, not per process: a run that silently
downgraded once at startup and then never mentioned it again would be worse
than one that says so every time.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from ... import config
from .page_adapter import PageAdapter

logger = logging.getLogger("browser.handle")

BrowserKind = Literal["drover", "bundled"]


@dataclass(frozen=True)
class BrowserHandle:
    """A page to drive, and the provenance of it.

    `connection` is present only for `drover`. Callers that need to know whether
    the browser is the user's own read `kind`; nothing should branch on
    `connection is None` as a proxy, because a future third driver would be
    neither.
    """

    kind: BrowserKind
    page: PageAdapter
    connection: object | None = None

    @property
    def is_user_browser(self) -> bool:
        """True when this is a browser the user opened themselves.

        The distinction that matters to a caller is not which driver it is but
        whether the session is the user's — that is what decides whether a
        login wall is expected.
        """
        return self.kind == "drover"


async def get_browser_handle(*, prefer_drover: bool = True) -> BrowserHandle:
    """Resolve a browser to drive.

    Order: the extension if it is connected and preferred, otherwise bundled.

    `prefer_drover=False` does not merely lose a race — it never touches the
    connection at all. A caller that has asked for the bundled browser wants a
    clean profile, and quietly handing it the user's session would be a
    different task than the one it asked for.
    """
    if not prefer_drover:
        return await _bundled("caller asked for the bundled browser")

    if not bool(getattr(config, "BROWSER_PREFER_EXTENSION", True)):
        return await _bundled("BROWSER_PREFER_EXTENSION is off")

    # Imported here rather than at module load: `automation` may import `io`,
    # but doing it eagerly would drag the whole API package into every import
    # of the browser tier, including tests that stub it out.
    from ...io.api.extension_ws import current_connection

    connection = current_connection()
    if connection is None:
        return await _bundled("no browser extension is connected")

    from .drover import ExtensionPage

    return BrowserHandle(
        kind="drover",
        page=ExtensionPage(connection),
        connection=connection,
    )


async def _bundled(reason: str) -> BrowserHandle:
    """The fallback, with the one line that explains it."""
    from . import automation as browser_automation
    from .page_adapter import PlaywrightPage

    logger.info(f"[BROWSER] using bundled Chromium: {reason}")
    browser = await browser_automation.ensure_browser(headless=True)
    context = await browser.new_context()
    page = await context.new_page()
    return BrowserHandle(kind="bundled", page=PlaywrightPage(page), connection=None)
