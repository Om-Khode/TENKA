"""Tab control, through whichever browser the extension is connected from.

The DOM tier drives one page: the active tab. This is the other half — what the
user means by "close that", "what am I looking at", "open X over there".

**Nothing here is app-specific.** A tab is named by matching the words the user
said against its title and url, so "close the youtube tab" works because the tab
says "YouTube", not because anything here knows what YouTube is. Add a site and
nothing changes.

## Why closing needs a name

`tabs.close` on the wire requires an explicit id, and this handler will not
supply the active tab as a default. "Close the tab" with nothing to disambiguate
is a request TENKA cannot safely guess at: the cost of guessing wrong is the
user losing a page they were reading, and there is no undo. It asks instead.

The one exception is an unambiguous match — exactly one tab whose title or url
contains what the user said. That is not a guess.
"""

import logging

from .registry import tool_registry

logger = logging.getLogger("actions")

#: Read-only verbs answer even when several tabs match; the rest need one.
#:
#: No `open`. `open_browser` already opens a URL in the user's default browser
#: with `webbrowser.open`, and a second intent for the same request is the
#: over-claim shape this project keeps paying for: two rows competing, and
#: whichever the classifier happens to prefer wins. What lands here is only
#: what nothing else can do -- seeing and steering tabs that already exist.
_ACTIONS = ("list", "close", "switch")


def _score(tab: dict, query: str) -> int:
    """How well this tab matches what the user said. 0 means not at all.

    Title before url, and a whole-word hit before a substring: "close the mail
    tab" should prefer a tab titled "Mail" over one whose url merely contains
    the letters m-a-i-l inside "gmail.com/u/0/#inbox".
    """
    if not query:
        return 0
    words = [w for w in query.lower().split() if len(w) > 2]
    if not words:
        return 0

    title = str(tab.get("title", "")).lower()
    url = str(tab.get("url", "")).lower()
    score = 0
    for word in words:
        if word in title.split():
            score += 3
        elif word in title:
            score += 2
        elif word in url:
            score += 1
    return score


def _describe(tab: dict) -> str:
    title = str(tab.get("title", "")).strip()
    return title or str(tab.get("url", "")).strip() or f"tab {tab.get('id')}"


@tool_registry.decorator("browser_tabs")
async def handle_browser_tabs(
    params: dict, llm_response: str = "", bridge=None, **kwargs
) -> str:
    """
    params: {"action": "list" | "close" | "switch", "query": str}

    `query` is what the user called the tab. It is matched against titles and
    urls, never against a list of known sites.
    """
    action = (params.get("action") or "list").lower()
    query = (params.get("query") or "").strip()

    if bridge:
        await bridge.send_thought("thinking")
        await bridge.send_keyboard(False)

    async def _reply(message: str) -> str:
        logger.info(f"[ACTIONS] browser_tabs action={action}: {message[:120]}")
        if bridge:
            await bridge.send_thought("done", message)
        return message

    if action not in _ACTIONS:
        return await _reply(f"I don't know how to {action!r} a tab.")

    try:
        from ..core import latch_protocol as proto
        from ..io.api.extension_ws import LatchCallError, current_connection
    except ImportError as e:
        return await _reply(f"Couldn't load the browser driver: {e}")

    connection = current_connection()
    if connection is None:
        # Named as a state, not a failure. The browser tier falls back to a
        # bundled browser for page tasks, but tabs are inherently about the
        # user's own window -- there is nothing to fall back to.
        return await _reply(
            "The browser extension isn't connected, so I can't see your tabs. "
            "Say 'set up the browser extension' and I'll walk you through it."
        )

    try:
        tabs = (await connection.call(proto.Rpc.TABS_LIST, {})).get("tabs") or []

        if action == "list":
            if not tabs:
                return await _reply("No tabs open.")
            active = next((t for t in tabs if t.get("active")), None)
            lines = [
                f"{'* ' if t is active else '  '}{_describe(t)}" for t in tabs
            ]
            head = f"{len(tabs)} tab{'s' if len(tabs) != 1 else ''} open"
            if active is not None:
                head += f", you're on {_describe(active)}"
            return await _reply(head + ":\n" + "\n".join(lines))

        # close / switch both need exactly one tab.
        if not query:
            return await _reply(
                "Which tab? Name something from its title and I'll find it."
            )

        scored = sorted(
            ((t, _score(t, query)) for t in tabs), key=lambda p: p[1], reverse=True
        )
        matches = [t for t, s in scored if s > 0]

        if not matches:
            return await _reply(f"No open tab looks like {query!r}.")

        # A clear winner counts as unambiguous; a tie does not. Closing the
        # wrong tab loses a page the user was reading and cannot be undone.
        if len(matches) > 1 and scored[0][1] == scored[1][1]:
            names = ", ".join(_describe(t) for t in matches[:4])
            return await _reply(
                f"More than one tab matches {query!r}: {names}. Which one?"
            )

        target = matches[0]
        if action == "close":
            await connection.call(proto.Rpc.TABS_CLOSE, {"tabId": target["id"]})
            return await _reply(f"Closed {_describe(target)}.")

        await connection.call(proto.Rpc.TABS_ACTIVATE, {"tabId": target["id"]})
        return await _reply(f"Switched to {_describe(target)}.")

    except LatchCallError as e:
        # The extension answered, and said no. Its reason is more useful than
        # anything this layer could invent.
        return await _reply(f"The browser refused: {e.raw_message}")
    except Exception as e:
        logger.warning(f"[ACTIONS] browser_tabs crashed: {type(e).__name__}: {e}")
        return await _reply("Something went wrong talking to the browser. See logs.")
