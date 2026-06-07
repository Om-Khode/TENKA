"""_utils.py — Shared helpers for code_executor package."""

import logging
import re

logger = logging.getLogger("code_executor")

from .. import service_registry as _sr


_ERROR_PREFIXES = ("BLOCKED", "ERROR:", "Traceback", "Error:", "An unexpected error")
_FAILURE_LINE_PREFIXES = (
    "error occurred", "error fetching", "error retrieving",
    "failed to ", "could not ", "unable to ",
    "permission denied", "access denied", "unauthorized",
    "need ", "missing ",
    "no devices", "no active", "no matching", "no such",
    "no playback", "no results", "no response",
    "not found", "not available", "not supported",
)
_POWERSHELL_ERROR_MARKERS = ("CategoryInfo", "FullyQualifiedErrorId")
_SCOPE_ERROR_PHRASES = ("Insufficient client scope", "http status: 4")
# Phrases printed by service SDKs when the desktop app is not running.
# These are NOT code bugs — the code is correct, the precondition is missing.
_APP_NOT_RUNNING_RE = re.compile(
    r"no\s+active\s+(?:\w+\s+)?devices?"       # "no active device(s)" / "no active <app> devices"
    r"|player\s+command\s+failed"               # SDK-level player error
    r"|please\s+(?:start|open)\s+(?:the\s+)?(?:app|\w+)"  # "please open/start the app"
    r"|app(?:lication)?\s+is\s+not\s+running"   # "app is not running" / "application is not running"
    r"|no\s+(?:active\s+)?devices?\s+(?:found|available)"  # "no devices found/available"
    r"|no\s+active\s+playback",                 # "no active playback"
    re.IGNORECASE,
)


def _detect_app_not_running(result: str) -> bool:
    return bool(_APP_NOT_RUNNING_RE.search(result))


def _needs_retry(result: str) -> bool:
    """Determine if a result indicates failure that warrants a retry."""
    if not result:
        return True
    if result.startswith("NEEDS_OAUTH|"):
        return False
    if result.startswith("APP_NOT_READY|"):
        return True
    if result in ("TIMEOUT", "(no output)"):
        return True
    if any(result.startswith(p) for p in _ERROR_PREFIXES):
        return True
    if any(marker in result for marker in _POWERSHELL_ERROR_MARKERS):
        return True
    if any(phrase in result for phrase in _SCOPE_ERROR_PHRASES):
        return True

    lines = [ln.strip() for ln in result.strip().splitlines() if ln.strip()]
    if not lines:
        return True
    if 1 <= len(lines) <= 2 and all(ln.endswith(':') for ln in lines):
        return True

    result_lower = result.lower()
    if len(lines) <= 3:
        for phrase in _FAILURE_LINE_PREFIXES:
            if result_lower.startswith(phrase) or result_lower == phrase:
                return True

    if len(lines) >= 3:
        data_lines = [ln for ln in lines if not ln.lower().startswith("total")]
        if data_lines:
            # Check 1: Lines that are None/null/n/a/unknown or start with None/Unknown
            placeholder_count = sum(
                1 for ln in data_lines
                if ln.strip().lower() in ('none', 'null', 'n/a', 'unknown')
                or ln.strip().endswith(". None")
                or ln.strip().lower().startswith("none ")  # "None []", "None - artist"
                or ln.strip().lower().startswith("none,")
                or ln.strip().lower().startswith("unknown ")  # "Unknown -", "Unknown Artist"
                or ln.strip().lower().startswith("unknown,")
                or ln.strip().lower().startswith("unknown-")  # "Unknown-" no space
            )
            if placeholder_count / len(data_lines) > 0.4:
                logger.info(f"[CODE] Soft failure: {placeholder_count}/{len(data_lines)} placeholder lines")
                return True

            # Check 2: Lines with no meaningful content — just punctuation,
            # whitespace, separators like " - ", "--", etc. This catches
            # output where formatting printed but actual data is empty.
            empty_content_count = sum(
                1 for ln in data_lines
                if not re.search(r'[a-zA-Z0-9]{2,}', ln)  # no 2+ alphanumeric chars
            )
            if empty_content_count / len(data_lines) > 0.5:
                logger.info(f"[CODE] Soft failure: {empty_content_count}/{len(data_lines)} lines have no real content")
                return True
    return False


def _is_scope_error(result: str) -> bool:
    """
    Check if result indicates an OAuth SCOPE error specifically.
    Must NOT match plain 403 Forbidden (which could be a deprecated endpoint).
    Only matches when scope-specific keywords are present.
    """
    r = result.lower()
    # Definite scope errors — these phrases only appear in scope problems
    if any(p in r for p in ("insufficient client scope", "insufficient_scope",
                             "insufficient authentication scopes",
                             "missing required scope", "requires scope")):
        return True
    # 403 + scope-specific keyword (NOT just "forbidden" — that matches everything)
    if "403" in r and "scope" in r:
        return True
    return False


def _strip_code_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        elif lines[0].strip().startswith("```"):
            lines = lines[1:]
        code = "\n".join(lines)
    return code.strip()


def _syntax_check(code: str) -> str | None:
    try:
        compile(code, "<syntax_check>", "exec")
        return None
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"


_BANNED_OAUTH_IMPORT_RE = re.compile(
    r"^\s*from\s+(?:" + "|".join(re.escape(c).replace(r"\ ", r"\s+")
    for c in ["spotipy.oauth2"]) + r")\s+import\s+.*$", re.MULTILINE
)
_BANNED_AUTH_MANAGER_RE = re.compile(r"(client_credentials_manager|auth_manager)\s*=\s*[^,\)]+")

def _sanitize_oauth_imports(code: str) -> str:
    """Strip banned OAuth flow imports/assignments and all references to stripped variables."""
    orig = code
    orphaned_vars: set[str] = set()

    for cls_name in _sr.ALL_BANNED_AUTH_CLASSES:
        # Remove "from x import ClassName" lines
        code = re.sub(rf"^\s*from\s+\S+\s+import\s+.*\b{re.escape(cls_name)}\b.*$",
                       "", code, flags=re.MULTILINE)
        # Collect variable names assigned to banned class constructors BEFORE removing them
        for m in re.finditer(rf"^\s*(\w+)\s*=\s*{re.escape(cls_name)}\s*\(", code, re.MULTILINE):
            var = m.group(1)
            if var not in ("_", "self"):
                orphaned_vars.add(var)
        # Remove the constructor call assignments
        code = re.sub(rf"^\s*\w+\s*=\s*{re.escape(cls_name)}\s*\([^)]*\).*$",
                       "", code, flags=re.MULTILINE)

    code = _BANNED_AUTH_MANAGER_RE.sub("", code)

    # Remove any remaining lines that reference the stripped variables (e.g. sp_oauth.token_info = ...)
    for var in orphaned_vars:
        code = re.sub(rf"^[^\n]*\b{re.escape(var)}\b[^\n]*$", "", code, flags=re.MULTILINE)

    code = re.sub(r",\s*,", ",", code)
    code = re.sub(r"\(\s*,", "(", code)
    code = re.sub(r",\s*\)", ")", code)
    if code != orig:
        logger.warning(f"[CODE] Stripped banned OAuth imports/references (vars: {orphaned_vars or 'none'})")
    return code


def _parse_oauth_sentinel(s: str) -> dict | None:
    """
    Parse NEEDS_OAUTH|service|auth_url|token_url|scopes|redirect_uri.
    Tolerates trailing empty fields (LLM sometimes omits redirect_uri or
    produces trailing pipes like NEEDS_OAUTH|gmail|...|scopes||).
    """
    try:
        parts = s.split("|")
        # Strip empty trailing parts from extra pipes
        while len(parts) > 6 and parts[-1] == "":
            parts.pop()
        if len(parts) < 5 or parts[0] != "NEEDS_OAUTH":
            return None
        service = parts[1]
        auth_url = parts[2]
        token_url = parts[3]
        scopes = parts[4] if len(parts) > 4 else ""
        redirect_uri = parts[5] if len(parts) > 5 and parts[5] else "http://127.0.0.1:8888/callback"
        return {"service": service, "auth_url": auth_url, "token_url": token_url,
                "scopes": scopes, "redirect_uri": redirect_uri}
    except Exception as e:
        logger.debug(f"[CODE] OAuth sentinel parse failed: {e}")
        return None


def _process_oauth_sentinel(result: str) -> str | None:
    parsed = _parse_oauth_sentinel(result)
    if not parsed:
        return None
    from .. import credentials as cs
    service = parsed["service"]
    existing = set((cs.get_credential(service, "granted_scopes") or "").split())
    requested = set(parsed["scopes"].split()) if parsed["scopes"] else set()

    # Merge minimum scopes — ensures all needed permissions are requested
    # upfront so the user only authorizes once.
    min_scopes = set(_sr.OAUTH_MIN_SCOPES.get(service, []))
    merged = " ".join(sorted(existing | requested | min_scopes))

    # Normalize auth_url — LLMs sometimes include template params like
    # ?client_id={client_id}&redirect_uri={redirect_uri} in the auth URL.
    # Strip query params — actions.py adds real params during setup.
    auth_url = parsed["auth_url"].split("?")[0]

    # Normalize token_url — same issue, strip any query params.
    token_url = parsed["token_url"].split("?")[0]

    cs.set_credential(service, "token_url", token_url)

    # Normalize redirect_uri — LLMs often generate wrong ports or paths.
    # The project standard is http://127.0.0.1:8888/callback (registered in
    # every OAuth provider's dashboard). Override whatever the LLM produced.
    redirect_uri = "http://127.0.0.1:8888/callback"

    return (f"__NEEDS_OAUTH__|{service}|{auth_url}|"
            f"{token_url}|{merged}|{redirect_uri}")


def _parse_device_auth_sentinel(s: str) -> dict | None:
    """Parse NEEDS_DEVICE_AUTH|service|session_path."""
    try:
        parts = s.split("|")
        if len(parts) < 3 or parts[0] != "NEEDS_DEVICE_AUTH":
            return None
        return {"service": parts[1], "session_path": parts[2]}
    except Exception as e:
        logger.debug(f"[CODE] Device auth sentinel parse failed: {e}")
        return None


def _process_device_auth_sentinel(result: str) -> str | None:
    """Process a NEEDS_DEVICE_AUTH sentinel. Returns signal for actions.py or None."""
    parsed = _parse_device_auth_sentinel(result)
    if not parsed:
        return None
    service = parsed["service"]
    session_path = parsed["session_path"]
    return f"__NEEDS_DEVICE_AUTH__|{service}|{session_path}"


def _sanitize_debug_output(output: str) -> str:
    s = re.sub(r'(eyJ[A-Za-z0-9_-]{20,})', '[TOKEN]', output)
    s = re.sub(r'(BQ[A-Za-z0-9_-]{20,})', '[TOKEN]', s)
    s = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', s)
    s = re.sub(r'("(?:token|key|secret|password|auth)":\s*")([^"]+)"',
               r'\1[REDACTED]"', s, flags=re.IGNORECASE)
    return s


async def _clean_error_for_user(raw: str, llm_func) -> str:
    r = await llm_func(f"The command failed: {raw}\nExplain in one short sentence.", task_type="synthesis")
    return r.strip() if r != "__LLM_UNAVAILABLE__" else "Sorry, that command didn't work."


async def _pre_gen_search(requires: list[str], goal: str) -> str:
    """
    Search for API documentation before code generation.
    Two queries: quickstart + recent changes/migration.
    """
    _STDLIB_LIKE = frozenset({"requests", "beautifulsoup4", "httpx", "pandas",
                               "openpyxl", "python-docx", "psutil"})
    search_pkgs = [p for p in requires if p not in _STDLIB_LIKE]
    if not search_pkgs:
        return ""

    # Extract action keywords from goal for a more targeted search
    _service_stop_words = set(_sr.ALL_SERVICE_NAMES)
    goal_words = [w for w in goal.lower().split()
                  if len(w) > 3 and w not in ('from', 'list', 'show', 'tell',
                                               'with', 'that',
                                               'this', 'what', 'your', 'mine')
                  and w not in _service_stop_words]
    action_hint = ' '.join(goal_words[:3]) if goal_words else ''

    queries = [
        f"{search_pkgs[0]} python {action_hint} example 2026".strip(),
        f"{search_pkgs[0]} API migration deprecated endpoints 2026",
    ]
    logger.info(f"[CODE] Pre-gen search: {queries}")
    result = await _search_and_fetch(queries)
    if result:
        logger.info(f"[CODE] Pre-gen docs: {len(result)} chars")
    return result


async def _search_and_fetch(queries: list[str], direct_urls: list[str] | None = None) -> str:
    """Fetch documentation from direct URLs and/or web search results."""
    import asyncio
    import urllib.request

    _JUNK = frozenset({
        "scribd.com", "medium.com", "reddit.com", "youtube.com",
        "quora.com", "udemy.com", "coursera.org", "pinterest.com",
        "facebook.com", "twitter.com", "instagram.com", "tiktok.com",
    })
    collected: list[str] = []
    loop = asyncio.get_running_loop()

    if direct_urls:
        for url in direct_urls[:2]:
            try:
                def _fetch_url(u=f"https://r.jina.ai/{url}"):
                    req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/plain"})
                    with urllib.request.urlopen(req, timeout=12) as resp:
                        return resp.read().decode("utf-8", errors="replace")
                raw = await loop.run_in_executor(None, _fetch_url)
                if len(raw.strip()) >= 200:
                    collected.append(f"Source: {url}\n{'─' * 50}\n{raw[:1500].strip()}")
            except Exception as e:
                logger.debug(f"[CODE] Direct URL fetch failed for {url}: {e}")

    try:
        from .. import config as _cfg
        import requests as _req
        keys = getattr(_cfg, "TAVILY_API_KEYS", [])
        if keys and queries:
            for q in queries[:3]:
                try:
                    def _search(q=q):
                        r = _req.post("https://api.tavily.com/search",
                                      json={"api_key": keys[0], "query": q, "max_results": 3, "search_depth": "basic"},
                                      timeout=8)
                        r.raise_for_status()
                        return r.json()
                    res = await loop.run_in_executor(None, _search)
                    for r in res.get("results", []):
                        url = r.get("url", "")
                        domain = url.split("/")[2] if url.count("/") >= 2 else ""
                        if any(j in domain for j in _JUNK) or any(url in c for c in collected):
                            continue
                        try:
                            def _fetch_result(u=f"https://r.jina.ai/{url}"):
                                req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/plain"})
                                with urllib.request.urlopen(req, timeout=10) as resp:
                                    return resp.read().decode("utf-8", errors="replace")
                            raw = await loop.run_in_executor(None, _fetch_result)
                            if len(raw.strip()) >= 200:
                                collected.append(f"Source: {url}\n{'─' * 50}\n{raw[:1500].strip()}")
                                break
                        except Exception as e:
                            logger.debug(f"[CODE] Fetch failed for {url}: {e}")
                except Exception as e:
                    logger.debug(f"[CODE] Search failed for '{q}': {e}")
    except ImportError:
        logger.debug("[CODE] Web search unavailable (missing config or requests)")
    except Exception as e:
        logger.debug(f"[CODE] Web search error: {e}")

    if not collected:
        return ""
    return "API DOCUMENTATION:\n" + "\n\n".join(collected) + "\nUse the docs above when writing code.\n"


def _looks_truncated(code: str) -> bool:
    """
    Heuristic check for truncated code that still passes syntax check.
    Returns True if the code looks incomplete.
    """
    lines = [ln for ln in code.strip().splitlines() if ln.strip() and not ln.strip().startswith('#')]
    if not lines:
        return True

    # Too short — a working Tier 2 script needs at least ~8 lines
    # (imports + token check + client init + api call + print)
    if len(lines) < 6:
        logger.debug(f"[CODE] Truncation check: only {len(lines)} meaningful lines")
        return True

    # Last line looks like a cut-off expression
    last = lines[-1].strip()
    _INCOMPLETE_ENDINGS = ('=', '(', ',', '[', '{', '.', '+', '-', '*', '/', '\\',
                           'import', 'from', 'if', 'for', 'while', 'try:', 'except',
                           'elif', 'else:', 'def', 'class', 'with', 'return')
    if any(last.endswith(e) for e in _INCOMPLETE_ENDINGS):
        logger.debug(f"[CODE] Truncation check: last line ends with incomplete token: '{last[-20:]}'")
        return True

    # No print statement — script won't produce any output
    if 'print(' not in code and 'print (' not in code:
        logger.debug("[CODE] Truncation check: no print statement found")
        return True

    return False
