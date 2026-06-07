"""retry.py — Error classification, fix planning/execution, and knowledge extraction."""

import logging
import re
import json as _json
import textwrap

logger = logging.getLogger("code_executor")

from ..core.json_utils import extract_json_object as _extract_json
from .prompts import _FIX_PLAN_PROMPT, _FIX_EXECUTE_PROMPT
from ._utils import _syntax_check, _sanitize_oauth_imports, _strip_code_fences, _looks_truncated


_MAX_RETRIES = 3


def _classify_error(result: str) -> dict:
    """
    Classify an error deterministically from the script output.
    No LLM call — pure regex on actual output.

    Returns a dict with:
      category: str — error type
      diagnosis: str — human-readable one-liner
      needs_discovery: bool — whether to run mechanical discovery
    """
    if not result:
        return {"category": "unknown", "diagnosis": "empty output", "needs_discovery": True}

    r = result.lower()
    lines = [ln.strip() for ln in result.strip().splitlines() if ln.strip()]

    # ── Traceback: extract exception type ──
    if "Traceback" in result:
        # Find the last line which is typically the exception
        exc_line = lines[-1] if lines else ""

        if "UnicodeEncodeError" in result or "UnicodeDecodeError" in result:
            return {"category": "encoding", "diagnosis": "Unicode encoding error — stdout needs UTF-8",
                    "needs_discovery": False}
        if "ModuleNotFoundError" in result or "ImportError" in result:
            return {"category": "import", "diagnosis": f"Missing module: {exc_line[:80]}",
                    "needs_discovery": False}
        if "SyntaxError" in result:
            return {"category": "syntax", "diagnosis": "Script has syntax error (likely truncated)",
                    "needs_discovery": False}
        if "TypeError" in result and "argument" in r:
            return {"category": "api_endpoint", "diagnosis": f"Wrong arguments: {exc_line[:80]}",
                    "needs_discovery": True}
        if "AttributeError" in result:
            return {"category": "api_endpoint", "diagnosis": f"Wrong method/attribute: {exc_line[:80]}",
                    "needs_discovery": True}
        if "KeyError" in result:
            return {"category": "field_access", "diagnosis": f"Wrong dict key: {exc_line[:80]}",
                    "needs_discovery": True}
        if "401" in result or "Unauthorized" in result:
            return {"category": "auth", "diagnosis": "Authentication failed (401)",
                    "needs_discovery": False}
        if "403" in result or "Forbidden" in result:
            return {"category": "scope", "diagnosis": "Permission denied (403)",
                    "needs_discovery": False}
        if "404" in result or "Not Found" in result:
            return {"category": "api_endpoint", "diagnosis": "Endpoint not found (404)",
                    "needs_discovery": True}
        if "ConnectionRefusedError" in result and "7780" in result:
            return {"category": "blocked", "diagnosis": "Messaging bridge not running — restart assistant",
                    "needs_discovery": False}
        if "ConnectionError" in result or "URLError" in result or "Timeout" in result:
            return {"category": "network", "diagnosis": "Network/connection error",
                    "needs_discovery": False}
        # Generic traceback
        return {"category": "logic", "diagnosis": f"Runtime error: {exc_line[:80]}",
                "needs_discovery": True}

    # ── Majority None/placeholder output (the 11/11 None pattern) ──
    if lines:
        bad_count = sum(
            1 for ln in lines
            if ln.lower() in ('none', 'null', 'n/a', 'unknown')
            or ln.strip().endswith(". None")
            or ln.strip().lower().startswith("none ")  # "None []", "None - "
            or ln.strip().lower().startswith("none,")
            or ln.strip().lower().startswith("unknown ")  # "Unknown -", "Unknown Artist"
            or ln.strip().lower().startswith("unknown,")
            or ln.strip().lower().startswith("unknown-")  # "Unknown-" no space
        )
        if bad_count > 0 and bad_count / len(lines) > 0.4:
            return {"category": "field_access",
                    "diagnosis": f"Wrong field names: {bad_count}/{len(lines)} lines are None/placeholder",
                    "needs_discovery": True}

    # ── Lines with no meaningful content (e.g. " - ", "--", just punctuation) ──
    if len(lines) >= 3:
        empty_content = sum(
            1 for ln in lines
            if not re.search(r'[a-zA-Z0-9]{2,}', ln)
        )
        if empty_content / len(lines) > 0.5:
            return {"category": "field_access",
                    "diagnosis": f"Empty data: {empty_content}/{len(lines)} lines have no content",
                    "needs_discovery": True}

    # ── Auth/scope errors in output text ──
    if any(p in r for p in ("insufficient client scope", "insufficient_scope",
                             "insufficient authentication")):
        return {"category": "scope", "diagnosis": "Insufficient OAuth scopes",
                "needs_discovery": False}
    if "401" in r and any(w in r for w in ("unauthorized", "token", "expired")):
        return {"category": "auth", "diagnosis": "Auth token expired or invalid",
                "needs_discovery": False}
    if "403" in r and any(w in r for w in ("forbidden", "permission", "scope",
                                            "deprecated", "not allowed")):
        return {"category": "api_endpoint",
                "diagnosis": "403 — endpoint forbidden/deprecated, try SDK or different approach",
                "needs_discovery": True}

    # ── HTTP status codes in error output (from verbose error printing) ──
    http_match = re.search(r'Error:\s*(\d{3})\b', result)
    if http_match:
        status = int(http_match.group(1))
        if status == 401:
            return {"category": "auth", "diagnosis": "HTTP 401 — auth failed",
                    "needs_discovery": False}
        if status == 403:
            return {"category": "api_endpoint",
                    "diagnosis": "HTTP 403 — forbidden/deprecated endpoint, try SDK or different approach",
                    "needs_discovery": True}
        if status == 404:
            return {"category": "api_endpoint", "diagnosis": "HTTP 404 — endpoint not found",
                    "needs_discovery": True}
        if 400 <= status < 500:
            return {"category": "api_endpoint", "diagnosis": f"HTTP {status} client error",
                    "needs_discovery": True}
        if status >= 500:
            return {"category": "network", "diagnosis": f"HTTP {status} server error",
                    "needs_discovery": False}

    # ── Empty-looking output ──
    if result == "TIMEOUT":
        return {"category": "timeout", "diagnosis": "Script timed out", "needs_discovery": False}
    if result == "(no output)":
        # Script ran but printed nothing — likely an API call returned empty/error
        # data that the code didn't handle. Discovery will show what the API returned.
        return {"category": "no_output", "diagnosis": "Script produced no output — API may return empty or unexpected data",
                "needs_discovery": True}
    if result.startswith("BLOCKED"):
        return {"category": "blocked", "diagnosis": result[:80], "needs_discovery": False}

    # ── Short output with labels but no data ──
    if 1 <= len(lines) <= 2 and all(ln.endswith(':') for ln in lines):
        return {"category": "field_access", "diagnosis": "Output has labels but no data",
                "needs_discovery": True}

    # ── Encoding errors in output ──
    if "charmap" in r or "cp1252" in r or "codec can't encode" in r:
        return {"category": "encoding", "diagnosis": "Console encoding error",
                "needs_discovery": False}

    # ── Missing required parameter (template mismatch) ──
    if any(r.startswith(p) for p in ("need ", "missing ", "no param")):
        return {"category": "wrong_approach",
                "diagnosis": f"Script requires parameters not provided for this goal: {result[:80]}. "
                             f"Rewrite the script to accomplish the goal without requiring those params.",
                "needs_discovery": False}

    # ── SDK argument errors caught by generic except handler ──
    if "unexpected keyword argument" in r or "unexpected argument" in r:
        return {"category": "api_endpoint",
                "diagnosis": f"Wrong keyword argument: {result[:100]}",
                "needs_discovery": True}
    if "missing" in r and "required" in r and "argument" in r:
        return {"category": "api_endpoint",
                "diagnosis": f"Missing required argument: {result[:100]}",
                "needs_discovery": True}

    # ── Generic error prefixes ──
    if any(result.startswith(p) for p in ("ERROR:", "Error:")):
        return {"category": "logic", "diagnosis": result[:100],
                "needs_discovery": True}

    # ── Fallback ──
    return {"category": "unknown", "diagnosis": "Unrecognized error pattern",
            "needs_discovery": True}


async def _plan_fix(goal, code, result, diagnosis, discovery_data, web_context,
                    cred_hint, scope_hint, llm_func, service: str | None = None) -> str | None:
    """
    Single-pass fix: Big model (agent_plan → llama-3.3-70b) analyzes the
    failure and produces XML <fix><old>...</old><new>...</new></fix> blocks.
    We apply these mechanically — no second LLM call needed.
    """
    parts = [f"Goal: {goal}\nError classification: {diagnosis.get('diagnosis', '?')}\n"]

    # Inject service knowledge — avoid known-bad patterns during fix
    if service:
        from .. import knowledge as _ks
        _kctx = _ks.render_for_llm(service)
        if _kctx:
            parts.append(f"\n{_kctx}\n")

    parts.append(f"\nBroken code:\n```python\n{code[:800]}\n```\n\nError output:\n{result[:300]}\n")
    if discovery_data:
        parts.append(f"\nDISCOVERY DATA (raw API response structure):\n{discovery_data[:2000]}\n")
    if web_context:
        parts.append(f"\n{web_context[:1500]}\n")
    parts.append(f"\n{cred_hint}\n{scope_hint}\n")

    plan = await llm_func(
        "\n".join(parts), system_prompt=_FIX_PLAN_PROMPT,
        task_type="agent_plan", max_tokens=1500, temperature=0,
    )
    if plan == "__LLM_UNAVAILABLE__":
        logger.warning("[CODE] Plan: LLM unavailable")
        return None

    plan = plan.strip()
    if not plan or len(plan) < 10:
        logger.warning(f"[CODE] Plan: too short or empty: '{plan[:50]}'")
        return None

    logger.info(f"[CODE] ── Fix plan ──\n{plan}\n── End plan ──")
    return plan


async def _execute_fix_plan(code, plan, llm_func) -> str | None:
    """
    Apply the XML fix blocks from the plan directly to the code.
    The plan model already produced <fix><old>...</old><new>...</new></fix>
    blocks — we just parse and apply them mechanically.

    Falls back to asking the code_gen model if the plan isn't in XML format.
    """
    # Try to apply the plan directly as XML fix blocks
    fixed = _apply_replace_blocks(code, plan)
    if fixed is not None:
        # Plan was in XML format — applied directly, no second LLM call
        syn = _syntax_check(fixed)
        if syn:
            logger.warning(f"[CODE] Fix (from plan XML) has syntax error: {syn}")
            return None
        if _looks_truncated(fixed):
            logger.warning("[CODE] Fix (from plan XML) looks truncated")
            return None
        return fixed

    # Fallback: plan was in natural language (old bullet-point format).
    # Ask code_gen model to produce replace blocks.
    logger.info("[CODE] Plan not in XML format — forwarding to code_gen for replace blocks")
    prompt = f"PLAN:\n{plan}\n\nBroken code to fix:\n```python\n{code}\n```"

    raw = await llm_func(
        prompt, system_prompt=_FIX_EXECUTE_PROMPT,
        task_type="code_gen", max_tokens=1500,
    )
    if raw == "__LLM_UNAVAILABLE__":
        return None

    fixed = _apply_replace_blocks(code, raw)
    if fixed is None:
        logger.warning("[CODE] Fix response not in replace-block format — trying as full code")
        fixed = _strip_code_fences(raw)
        fixed = _sanitize_oauth_imports(fixed)

    syn = _syntax_check(fixed)
    if syn:
        logger.warning(f"[CODE] Fix (from plan) has syntax error: {syn}")
        return None
    if _looks_truncated(fixed):
        logger.warning("[CODE] Fix (from plan) looks truncated")
        return None
    return fixed


def _strip_blank_lines(text: str) -> str:
    lines = text.split('\n')
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return '\n'.join(lines)


def _reindent(text: str, target_indent: str) -> str:
    dedented = textwrap.dedent(text)
    lines = dedented.splitlines()
    return '\n'.join(
        (target_indent + line) if line.strip() else line
        for line in lines
    )


def _apply_replace_blocks(original_code: str, llm_response: str) -> str | None:
    """
    Parse XML fix/replace blocks from LLM response and apply them.

    Handles multiple tag formats that models produce:
      <fix><old>...</old><new>...</new></fix>
      <replace><old>...</old><new>...</new></replace>
      Malformed: <replace>old text</replace><new>new text</new>

    Returns the modified code, or None if no valid blocks found.
    """
    # Try standard format: <fix> or <replace> wrapping <old>/<new>
    blocks = re.findall(
        r'<(?:fix|replace)>\s*<old>(.*?)</old>\s*<new>(.*?)</new>\s*</(?:fix|replace)>',
        llm_response, re.DOTALL,
    )

    # Fallback: malformed format where <old>/<new> are siblings, not nested
    if not blocks:
        blocks = re.findall(
            r'<old>(.*?)</old>\s*<new>(.*?)</new>',
            llm_response, re.DOTALL,
        )

    if not blocks:
        return None

    fixed = original_code
    applied = 0

    for old_text, new_text in blocks:
        old_clean = _strip_blank_lines(old_text)
        new_clean = _strip_blank_lines(new_text)

        if not old_clean.strip():
            continue

        # Skip trivially short patterns — they match too broadly
        # (e.g. <old>sys</old> would corrupt "import sys")
        if len(old_clean.strip()) < 10:
            logger.warning(f"[CODE] Replace block skipped (pattern too short): '{old_clean.strip()}'")
            continue

        # Sanitize: strip any OAuth flow classes from new_text
        new_clean = _sanitize_oauth_imports(new_clean)

        # 1. Direct match (LLM preserved indentation correctly)
        if old_clean in fixed:
            fixed = fixed.replace(old_clean, new_clean, 1)
            applied += 1
            logger.info(f"[CODE] Replace block applied: '{old_clean.strip()[:60]}' → '{new_clean.strip()[:60]}'")
            continue

        # 2. Dedented multi-line match with re-indentation
        old_stripped_lines = [l.strip() for l in old_clean.splitlines() if l.strip()]
        if not old_stripped_lines:
            continue

        code_lines = fixed.splitlines()
        matched = False

        for i, code_line in enumerate(code_lines):
            if code_line.strip() == old_stripped_lines[0]:
                end = i + len(old_stripped_lines)
                if end <= len(code_lines) and all(
                    code_lines[i + j].strip() == old_stripped_lines[j]
                    for j in range(len(old_stripped_lines))
                ):
                    target_indent = code_line[:len(code_line) - len(code_line.lstrip())]
                    new_reindented = _reindent(new_clean, target_indent)
                    old_actual = '\n'.join(code_lines[i:end])
                    fixed = fixed.replace(old_actual, new_reindented, 1)
                    applied += 1
                    matched = True
                    logger.info(f"[CODE] Replace block applied (re-indented): '{old_stripped_lines[0][:60]}'")
                    break

        if matched:
            continue

        # 3. Single-line normalized fallback with re-indentation
        if len(old_stripped_lines) == 1:
            old_normalized = ' '.join(old_stripped_lines[0].split())
            found = False
            for line in fixed.splitlines():
                if old_normalized == ' '.join(line.strip().split()):
                    target_indent = line[:len(line) - len(line.lstrip())]
                    new_reindented = _reindent(new_clean, target_indent)
                    fixed = fixed.replace(line, new_reindented, 1)
                    applied += 1
                    found = True
                    logger.info(f"[CODE] Replace block applied (normalized): '{old_stripped_lines[0][:60]}'")
                    break
            if not found:
                logger.warning(f"[CODE] Replace block: old text not found in code: '{old_clean.strip()[:80]}'")
        else:
            logger.warning(f"[CODE] Replace block: old text not found in code: '{old_clean.strip()[:80]}'")

    if applied == 0:
        return None

    logger.info(f"[CODE] Applied {applied}/{len(blocks)} replace blocks")
    return fixed


async def _save_failure_knowledge(service: str, slug: str, history: list[dict],
                                   llm_func) -> None:
    """
    Extract and save lessons from structural failures.
    Called when all retries are exhausted. Saves automatically (no approval).
    Only saves entries for structural error categories.
    """
    from .. import knowledge

    structural_failures = [
        h for h in history
        if h.get("category") and knowledge.is_structural_error(h["category"])
        and h.get("code_snapshot")
    ]

    if not structural_failures:
        logger.info("[CODE] No structural failures to save as knowledge")
        return

    for h in structural_failures:
        snippet = knowledge.extract_relevant_snippet(
            h["code_snapshot"], h.get("error", "")
        )
        prompt = knowledge.build_failure_extraction_prompt(
            snippet, h.get("error", ""), h.get("category", "unknown"),
            discovery_data=h.get("discovery_data", ""),
        )

        raw = await llm_func(prompt, system_prompt="", task_type="agent_plan", max_tokens=200, temperature=0)
        if raw == "__LLM_UNAVAILABLE__":
            continue

        try:
            clean = _extract_json(raw, sanitize=True, repair=True) or "{}"
            lesson = _json.loads(clean)
            pattern = lesson.get("pattern", "")
            reason = lesson.get("reason", "")
            if pattern and reason:
                knowledge.add_never_entry(
                    service, slug, pattern, reason, h.get("category", "unknown")
                )
        except Exception as e:
            logger.warning(f"[CODE] Failed to parse failure lesson: {e}")


async def _save_success_knowledge(service: str, slug: str,
                                    broken_code: str, fixed_code: str,
                                    history: list[dict], llm_func) -> str | None:
    """
    Extract lesson from successful retry and queue for user approval.
    Also auto-saves the original broken approach as a "never" entry.
    """
    from .. import knowledge

    # Auto-save the original broken approach as "never" (structural failures only)
    first_failure = next(
        (h for h in history if h.get("category")
         and knowledge.is_structural_error(h["category"])),
        None
    )
    if first_failure and first_failure.get("code_snapshot"):
        snippet = knowledge.extract_relevant_snippet(
            first_failure["code_snapshot"], first_failure.get("error", "")
        )
        fail_prompt = knowledge.build_failure_extraction_prompt(
            snippet, first_failure.get("error", ""), first_failure.get("category", "unknown"),
            discovery_data=first_failure.get("discovery_data", ""),
        )
        fail_raw = await llm_func(fail_prompt, system_prompt="", task_type="agent_plan", max_tokens=200, temperature=0)
        if fail_raw != "__LLM_UNAVAILABLE__":
            try:
                clean = _extract_json(fail_raw, sanitize=True, repair=True) or "{}"
                lesson = _json.loads(clean)
                if lesson.get("pattern") and lesson.get("reason"):
                    knowledge.add_never_entry(
                        service, slug, lesson["pattern"], lesson["reason"],
                        first_failure.get("category", "unknown"),
                    )
            except Exception as e:
                logger.warning(f"[CODE] Failed to parse broken approach lesson: {e}")

    # Extract the working pattern using diff (deterministic, no LLM)
    diff = knowledge.compute_code_diff(broken_code, fixed_code)
    if not diff:
        logger.info("[CODE] No code diff detected — skipping success knowledge")
        return None

    # Find the last discovery data from history (most informative)
    last_discovery = ""
    for h in reversed(history):
        if h.get("discovery_data"):
            last_discovery = h["discovery_data"]
            break

    success_prompt = knowledge.build_success_extraction_prompt(
        diff, history[0].get("error", "") if history else "",
        discovery_data=last_discovery,
    )
    success_raw = await llm_func(success_prompt, system_prompt="", task_type="agent_plan", max_tokens=200, temperature=0)
    if success_raw == "__LLM_UNAVAILABLE__":
        return None

    try:
        clean = _extract_json(success_raw, sanitize=True, repair=True) or "{}"
        lesson = _json.loads(clean)
        pattern = lesson.get("pattern", "")
        reason = lesson.get("reason", "")
        if not pattern or not reason:
            return None

        # Check novelty — skip if already known
        if knowledge.has_knowledge(service):
            novelty_prompt = knowledge.build_novelty_check_prompt(service, pattern, reason)
            if novelty_prompt:
                novelty_raw = await llm_func(novelty_prompt, task_type="synthesis", max_tokens=10, temperature=0)
                if novelty_raw != "__LLM_UNAVAILABLE__" and "YES" in novelty_raw.upper():
                    logger.info("[CODE] Knowledge lesson already known — skipping")
                    return None

        # Return proposal for immediate mode (appended to task response)
        # Store the pending entry so actions.py can handle yes/no
        _pending_knowledge_queue.append({
            "service": service,
            "slug": slug,
            "pattern": pattern,
            "reason": reason,
        })
        from .. import knowledge
        return knowledge.render_for_user(pattern, reason)

    except Exception as e:
        logger.warning(f"[CODE] Failed to parse success lesson: {e}")
        return None


# Module-level queue for knowledge approval (consumed by actions.py)
_pending_knowledge_queue: list[dict] = []


# (Empty - _queue_knowledge_approval logic was merged into caller)


def pop_pending_knowledge() -> dict | None:
    """Pop the next pending knowledge entry. Called by actions.py."""
    if _pending_knowledge_queue:
        return _pending_knowledge_queue.pop(0)
    return None
