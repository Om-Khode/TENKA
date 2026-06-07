"""Per-personality response pools.

Random-pick pools for variety without LLM cost. Each personality folder
has a responses.json. Use personality_say(key, **kwargs) from handlers.
"""

import random as _random
import logging

_logger = logging.getLogger("responses")


def personality_say(key: str, **kwargs) -> str:
    from assistant.personalities import get_active_loader
    from assistant import config

    if "error" in kwargs:
        err = str(kwargs["error"]).split("\n")[0][:80]
        kwargs["error"] = err

    kwargs.setdefault("assistant_name_lower", config.ASSISTANT_NAME_LOWER)

    loader = get_active_loader()
    pool = loader.get_responses().get(key)
    if not pool:
        _logger.debug(f"[RESPONSES] No pool for key '{key}', falling back to key")
        return key
    return _random.choice(pool).format(**kwargs)
