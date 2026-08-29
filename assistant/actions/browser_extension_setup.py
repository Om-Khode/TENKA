"""Browser extension setup handler — mint a credential and explain the install.

Replaces the Chrome-shortcut generator this supersedes. That one had to create a
desktop and Start-Menu launcher so the user could open Chrome with
`--remote-debugging-port`, and the whole point of the extension tier is that no
relaunch is needed at all: it drives whichever browser is already open, in the
profile the user is already signed into.

So there is nothing to install on the operating system. What this does is mint
the loopback credential the extension presents in its handshake, and tell the
user where to paste it.

Modes:
  `setup`   — mint a fresh token and print it, with the install steps.
  `status`  — is the extension connected, and from which browser.
  `preview` — what `setup` would do, without minting anything.
  `undo`    — remove the stored credential; the extension can no longer connect.
"""

import logging

from .registry import tool_registry

logger = logging.getLogger("actions")

_INSTALL_STEPS = (
    "1. Open your browser's extensions page and load the Drover extension.\n"
    "2. Click its toolbar icon.\n"
    "3. Paste the token below and press Connect."
)


@tool_registry.decorator("browser_extension_setup")
async def handle_browser_extension_setup(
    params: dict, llm_response: str = "", bridge=None, **kwargs
) -> str:
    """
    params: {"mode": "setup" | "status" | "preview" | "undo"}
    """
    mode = (params.get("mode") or "setup").lower()
    if bridge:
        await bridge.send_thought("thinking")
        await bridge.send_keyboard(False)

    try:
        from ..io.api import extension_ws as _ews
    except ImportError as e:
        msg = f"Couldn't load the extension module: {e}"
        if bridge:
            await bridge.send_thought("done", msg)
        return msg

    if mode == "status":
        snap = _ews.drover_state_snapshot()
        if snap.connected:
            msg = f"The browser extension is connected from {snap.browser_name}."
        else:
            # Named as a state, not an error: not being connected is the normal
            # condition before setup, and the browser tier falls back cleanly.
            msg = ("No browser extension is connected. Browser tasks will use the "
                   "bundled browser, which is signed out of everything.")

    elif mode == "undo":
        removed = _ews.clear_token()
        msg = ("Removed the extension credential. The extension can no longer connect "
               "until you run setup again."
               if removed else
               "There was no extension credential to remove.")

    elif mode == "preview":
        # Deliberately mints nothing. A preview that quietly replaced the live
        # credential would disconnect a working extension to answer a question.
        msg = ("Setup would mint a new loopback token and replace any existing one, "
               "then show you these steps:\n" + _INSTALL_STEPS)

    else:
        try:
            token = _ews.mint_token()
        except Exception as e:
            logger.warning(f"[ACTIONS] extension token mint failed: {e}")
            msg = "Couldn't create the extension credential. See logs."
        else:
            msg = (f"{_INSTALL_STEPS}\n\nToken: {token}\n\n"
                   f"It only works from this machine, and it grants no access to "
                   f"anything but the browser.")

    logger.info(f"[ACTIONS] browser_extension_setup mode={mode}")
    if bridge:
        await bridge.send_thought("done", msg)
    return msg
