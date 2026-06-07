"""Browser CDP setup handler."""

import logging

from .registry import tool_registry

logger = logging.getLogger("actions")


@tool_registry.decorator("browser_cdp_setup")
async def handle_browser_cdp_setup(params: dict, llm_response: str = "", bridge=None, **kwargs) -> str:
    """
    Configure (or undo) Chrome's --remote-debugging-port flag so the DOM-mode
    browser automation can attach to the user's running Chrome.

    params: {"mode": "setup" | "undo" | "preview"}
    """
    mode = (params.get("mode") or "setup").lower()
    if bridge:
        await bridge.send_thought("thinking")
        await bridge.send_keyboard(False)

    try:
        from ..automation.browser import setup as _bs
    except ImportError as e:
        msg = f"Couldn't load Chrome setup module: {e}"
        if bridge:
            await bridge.send_thought("done", msg)
        return msg

    if mode == "undo":
        try:
            result = _bs.undo_chrome_cdp_setup()
            msg = result.message
            if result.failed:
                msg += f" ({len(result.failed)} couldn't be restored — see logs.)"
        except Exception as e:
            logger.warning(f"[ACTIONS] Chrome CDP undo crashed: {e}")
            msg = "Couldn't undo Chrome setup. See logs."
    else:
        try:
            result = _bs.setup_chrome_cdp(dry_run=(mode == "preview"))
            msg = result.message
            if getattr(result, "skipped", None):
                logger.info(f"[ACTIONS] browser_cdp_setup skip details ({len(result.skipped)}):")
                for path, reason in result.skipped:
                    logger.info(f"[ACTIONS]   - {reason}: {path}")
        except Exception as e:
            logger.warning(f"[ACTIONS] Chrome CDP setup crashed: {e}")
            msg = "Couldn't run Chrome setup. See logs."

    logger.info(f"[ACTIONS] browser_cdp_setup mode={mode}: {msg}")
    if bridge:
        await bridge.send_thought("done", msg)
    return msg
