"""Web search (Tavily) and URL browsing (Jina) handlers."""

import logging
import itertools as _itertools

from .. import config
from .registry import tool_registry
from .responses import personality_say

logger = logging.getLogger("actions")

# --- Tavily key rotation ---

_tavily_key_cycle = None
_current_tavily_key: str | None = None


def _init_tavily_keys():
    """Initialize Tavily key rotation cycle from config."""
    global _tavily_key_cycle, _current_tavily_key
    keys = getattr(config, "TAVILY_API_KEYS", [])
    if keys:
        _tavily_key_cycle = _itertools.cycle(keys)
        _current_tavily_key = next(_tavily_key_cycle)
        logger.info(f"[WEB_SEARCH] Tavily initialized with {len(keys)} key(s)")
    else:
        logger.warning("[WEB_SEARCH] No Tavily API keys found — web search will fall back to LLM")

_init_tavily_keys()


# --- Handlers ---


@tool_registry.decorator("web_search")
async def handle_web_search(params: dict, llm_response: str, bridge=None,
                           _from_planner: bool = False) -> str:
    """
    Search the web using Tavily API with key rotation.
    Falls back to LLM knowledge if search fails or no keys available.
    """
    global _current_tavily_key, _tavily_key_cycle
    from ..llm.contracts import ask_for_synthesis, ask_for_default
    from ..core.abort import abort, UserAborted
    from ..io.status_broadcaster import status, StatusPhase
    import asyncio
    import uuid as _uuid
    import requests as req

    query = params.get("query", "").strip()
    if not query:
        return personality_say("need_query")

    _task_id = f"web_search:{_uuid.uuid4().hex[:8]}"
    if not _from_planner:
        abort.reset()
        abort.register_task(_task_id)
    status.set(StatusPhase.BROWSING, detail=query[:40], cursor_follows=False)

    try:
        if abort.is_aborted():
            raise UserAborted(abort.reason)
        return await _web_search_body(params, llm_response, bridge, _from_planner,
                                      query=query)
    except UserAborted:
        if _from_planner:
            raise  # let planner see the abort, not a "Stopped." string
        try:
            return personality_say("stopped", default="Stopped.")
        except Exception:
            return "Stopped."
    finally:
        if not _from_planner:
            abort.unregister_task(_task_id)
            status.set(StatusPhase.IDLE)


async def _web_search_body(params: dict, llm_response: str, bridge,
                            _from_planner: bool, query: str) -> str:
    """Original web_search body extracted so the wrapper can do try/finally cleanly."""
    global _current_tavily_key, _tavily_key_cycle
    from ..llm.contracts import ask_for_synthesis, ask_for_default
    from ..core.abort import abort, UserAborted
    from ..io.status_broadcaster import status, StatusPhase
    import asyncio
    import requests as req

    from ..core.datetime_utils import date_context_line
    _date_ctx = date_context_line()

    if not _current_tavily_key:
        logger.warning("[WEB_SEARCH] No Tavily key available — falling back to LLM")
        if abort.is_aborted():
            raise UserAborted(abort.reason)
        status.set(StatusPhase.THINKING, detail="LLM fallback", cursor_follows=False)
        return await ask_for_synthesis(
            f"{_date_ctx}\nAnswer this as best you can from your training knowledge, "
            f"noting if the info might be outdated: {query}",
        )

    # — Step 1: Search via Tavily with key rotation —
    search_result = None
    last_error = None
    keys = getattr(config, "TAVILY_API_KEYS", [])
    attempts = min(len(keys), 3) if keys else 1

    for attempt in range(attempts):
        if abort.is_aborted():
            raise UserAborted(abort.reason)
        status.set(StatusPhase.BROWSING,
                   detail=f"search {attempt+1}/{attempts}", cursor_follows=False)
        current_key = _current_tavily_key
        try:
            def _do_search(api_key=current_key):
                response = req.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": api_key,
                        "query": query,
                        "max_results": 3,
                        "search_depth": "basic",
                        "include_answer": True,
                    },
                    timeout=10,
                )
                response.raise_for_status()
                return response.json()

            loop = asyncio.get_running_loop()
            search_result = await loop.run_in_executor(None, _do_search)
            logger.info(f"[WEB_SEARCH] Got results for: '{query}' (key #{attempt + 1})")
            break

        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            if any(code in error_str for code in ("429", "401", "403", "rate")):
                if _tavily_key_cycle and len(keys) > 1:
                    _current_tavily_key = next(_tavily_key_cycle)
                    logger.warning(
                        f"[WEB_SEARCH] Tavily key rate limited, rotating to next key..."
                    )
                    continue
            logger.warning(f"[WEB_SEARCH] Search attempt {attempt + 1} failed: {e}")
            break

    if not search_result:
        logger.warning(f"[WEB_SEARCH] All attempts failed: {last_error} — falling back to LLM")
        return await ask_for_default(
            f"{_date_ctx}\nAnswer this as best you can from your training knowledge, "
            f"noting if the info might be outdated: {query}",
        )

    # — Step 2: Extract context from Tavily response —
    tavily_answer = search_result.get("answer", "")
    results = search_result.get("results", [])[:3]

    context_parts = []
    if tavily_answer:
        context_parts.append(f"[Direct Answer]: {tavily_answer}")
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        snippet = r.get("content", "")[:400]
        url = r.get("url", "")
        context_parts.append(f"[Result {i}] {title}\n{snippet}\nSource: {url}")

    search_context = "\n\n".join(context_parts)

    # — Step 3: Synthesize a spoken answer via LLM —
    if _from_planner:
        return search_context

    if abort.is_aborted():
        raise UserAborted(abort.reason)
    status.set(StatusPhase.THINKING, detail="synthesizing", cursor_follows=False)

    synthesis_prompt = (
        f"{_date_ctx}\nThe user asked: \"{query}\"\n\n"
        f"Here are the web search results:\n\n"
        f"{search_context}\n\n"
        f"Give a concise natural spoken answer in 2-3 sentences maximum. "
        f"Do not say 'according to the search results' or mention sources — "
        f"just answer naturally and confidently. "
        f"If the results don't clearly answer the question, say so briefly."
    )

    answer = await ask_for_synthesis(synthesis_prompt, max_tokens=200)

    if answer == "__LLM_UNAVAILABLE__":
        return tavily_answer or results[0].get("content", "I found results but couldn't summarize them.")

    return answer


@tool_registry.decorator("browse_url")
async def handle_browse_url(params: dict, llm_response: str, bridge=None,
                            _from_planner: bool = False) -> str:
    """
    Fetch and summarize a webpage using the Jina reader API.
    No API key required. Falls back to web_search if Jina fails.
    """
    from ..llm.contracts import ask_for_synthesis
    from ..core.abort import abort, UserAborted
    from ..io.status_broadcaster import status, StatusPhase
    import uuid as _uuid
    import requests as req

    url = params.get("url", "").strip()
    if not url:
        return personality_say("need_browse_url")

    _task_id = f"browse_url:{_uuid.uuid4().hex[:8]}"
    if not _from_planner:
        abort.reset()
        abort.register_task(_task_id)
    status.set(StatusPhase.BROWSING, detail=url[:40], cursor_follows=False)

    try:
        if abort.is_aborted():
            raise UserAborted(abort.reason)
        return await _browse_url_body(params, llm_response, bridge, url=url)
    except UserAborted:
        if _from_planner:
            raise  # let planner see the abort, not a "Stopped." string
        try:
            return personality_say("stopped", default="Stopped.")
        except Exception:
            return "Stopped."
    finally:
        if not _from_planner:
            abort.unregister_task(_task_id)
            status.set(StatusPhase.IDLE)


async def _browse_url_body(params: dict, llm_response: str, bridge, url: str) -> str:
    """Original browse_url body extracted for clean try/finally wrapping."""
    from ..llm.contracts import ask_for_synthesis
    from ..core.abort import abort, UserAborted
    from ..io.status_broadcaster import status, StatusPhase
    import requests as req

    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    jina_url = f"https://r.jina.ai/{url}"
    page_text = None

    try:
        response = req.get(
            jina_url,
            headers={"Accept": "text/plain"},
            timeout=15,
        )
        response.raise_for_status()
        page_text = response.text.strip()
        if len(page_text) > 4000:
            page_text = page_text[:4000] + "\n... (page truncated)"
        logger.info(f"[BROWSE] Fetched {len(page_text)} chars from {url}")
    except Exception as e:
        logger.warning(f"[BROWSE] Jina fetch failed for {url}: {e}")

    if not page_text:
        return personality_say("page_unreadable")

    if abort.is_aborted():
        raise UserAborted(abort.reason)
    status.set(StatusPhase.THINKING, detail="summarizing", cursor_follows=False)

    synthesis_prompt = (
        f"The user asked about this page: {url}\n\n"
        f"Here is the page content:\n\n{page_text}\n\n"
        f"Give a concise, natural spoken summary in 3-5 sentences. "
        f"Focus on the most important or interesting content. "
        f"Do not mention the URL or say 'according to the page' — just summarize naturally."
    )

    answer = await ask_for_synthesis(synthesis_prompt, max_tokens=150)

    if answer == "__LLM_UNAVAILABLE__":
        return f"Here's what I found: {page_text[:300]}..."

    return answer
