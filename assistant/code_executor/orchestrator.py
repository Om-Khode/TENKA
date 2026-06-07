"""orchestrator.py — Main entry point: natural-language goal -> code -> result."""

import logging
import re
import json as _json

logger = logging.getLogger("code_executor")


def _disambiguate_slug_on_mismatch(slug: str, params: dict | None) -> str:
    """When a cached template's goal doesn't match the current request, derive
    a new slug so the fresh template caches alongside the old one instead of
    overwriting.

    Uses the sorted param-key signature as the discriminator. A parameterized
    request (e.g. {"song_name": "Blinding Lights"}) gets `slug__song_name`,
    so a song-specific template doesn't clobber the generic "play music"
    template. Successive specific requests with the same param shape share
    one cached slug — no `_play_play_play` accumulation.

    Idempotent: if the slug already ends with `__{param_sig}` (e.g. router
    learned the disambiguated slug from the dynamic catalog and emitted it
    again for a different specific request), do NOT append a second time.
    Without this guard, slug grows __music_title__music_title__music_title
    on every mismatch with the same param shape.

    Generic requests (params={} or None) keep the original slug and overwrite
    the cached generic template (same shape, no information loss).
    """
    if not params:
        return slug
    param_sig = "_".join(sorted(params.keys()))
    suffix = f"__{param_sig}"
    if slug.endswith(suffix):
        return slug  # already disambiguated for this param shape
    return f"{slug}{suffix}"

from .. import service_registry as _sr
from .sandbox import _run_tier2, run_code
from .prompts import _CODE_GEN_SYSTEM_PROMPT_TIER1, _CODE_GEN_SYSTEM_PROMPT_TIER2, _FIX_EXECUTE_PROMPT
from .templates import (
    _load_template, _save_template, _delete_template,
    _goal_matches_template, _dump_code,
)
from .routing import (
    detect_service_from_packages, get_oauth_env_map, detect_messaging_service,
    _check_service_blocklist, _route_goal, _ensure_packages,
    TIER2_ALLOWED_PACKAGES, _IMPORT_TO_PACKAGE,
)
from .discovery import (
    _run_discovery, _enrich_discovery_with_key_analysis,
    _apply_key_fixes, _extract_kwarg_fixes, _apply_kwarg_fixes,
    _extract_api_urls,
    _strip_fields_param, _extract_api_calls, _find_client_var,
    _build_flat_setup, _build_discovery_script, _build_http_injection_script,
)
from .retry import (
    _MAX_RETRIES, _classify_error, _plan_fix, _execute_fix_plan,
    _apply_replace_blocks,
    _save_failure_knowledge, _save_success_knowledge,
    _pending_knowledge_queue,
)
from ._utils import (
    _needs_retry, _is_scope_error,
    _strip_code_fences, _syntax_check, _sanitize_oauth_imports,
    _process_oauth_sentinel, _process_device_auth_sentinel,
    _detect_app_not_running, _looks_truncated,
    _pre_gen_search, _search_and_fetch,
    _clean_error_for_user,
)


# ─── System Command Gate ─────────────────────────────────────────────────────

_ACTION_VERBS = ["disable", "enable", "turn on", "turn off",
                 "connect", "disconnect", "restart", "toggle"]
_QUERY_VERBS = ["list", "show", "get", "check", "scan"]
_SYSTEM_NOUNS = ["wifi", "wi-fi", "wi fi", "bluetooth", "network",
                 "adapter", "battery", "volume", "brightness"]


def _is_system_command(goal: str) -> bool:
    g = goal.lower()
    has_action = any(v in g for v in _ACTION_VERBS)
    if has_action:
        return True
    has_query = any(v in g for v in _QUERY_VERBS)
    has_noun = any(n in g for n in _SYSTEM_NOUNS)
    return has_query and has_noun


# ─── Synthesis Verification Gate ────────────────────────────────────────────

_COMPLETION_EVIDENCE_RE = re.compile(
    r"(https?://|status[:\s]*[245]\d{2}|successfully|success|created|sent\b"
    r"|booked|completed|downloaded|uploaded|scheduled|reserved|confirmed"
    r"|purchased|ordered|deleted|removed|installed|played|posted)",
    re.I,
)


def _has_completion_evidence(result: str) -> bool:
    return bool(_COMPLETION_EVIDENCE_RE.search(result))


# ─── Code Generation with Truncation Guard ───────────────────────────────────

_MAX_GEN_ATTEMPTS = 3  # max times to regenerate on syntax error (uses fallback chain)


async def _generate_code(gen_prompt: str, llm_func) -> str | None:
    """
    Generate code with truncation guard. If the model produces truncated
    output (syntax error OR signs of truncation), regenerate immediately.
    """
    for gen_attempt in range(1, _MAX_GEN_ATTEMPTS + 1):
        from ..core.abort import abort, UserAborted
        if abort.is_aborted():
            raise UserAborted(abort.reason)
        from ..io.status_broadcaster import status, StatusPhase
        from ..io.overlay.theme import CODE_GEN_LABELS, rotating_label
        status.set(StatusPhase.THINKING,
                   detail=rotating_label(CODE_GEN_LABELS, gen_attempt - 1),
                   step=(gen_attempt, _MAX_GEN_ATTEMPTS))
        code = await llm_func(
            gen_prompt, system_prompt=_CODE_GEN_SYSTEM_PROMPT_TIER2,
            task_type="code_gen", max_tokens=1500,
        )
        if code == "__LLM_UNAVAILABLE__":
            return None

        code = _strip_code_fences(code)
        code = _sanitize_oauth_imports(code)

        # Check 1: Syntax error (most common truncation signal)
        syn = _syntax_check(code)
        if syn:
            logger.warning(f"[CODE] Gen attempt {gen_attempt}/{_MAX_GEN_ATTEMPTS} syntax error: {syn}")
            if gen_attempt < _MAX_GEN_ATTEMPTS:
                gen_prompt = gen_prompt + "\nIMPORTANT: Keep code VERY short, under 30 lines."
            continue

        # Check 2: Code looks truncated even if syntax is valid
        # (e.g. code ends mid-expression inside a string, or is suspiciously short)
        if _looks_truncated(code):
            logger.warning(f"[CODE] Gen attempt {gen_attempt}/{_MAX_GEN_ATTEMPTS} looks truncated")
            if gen_attempt < _MAX_GEN_ATTEMPTS:
                gen_prompt = gen_prompt + "\nIMPORTANT: Keep code VERY short, under 30 lines."
            continue

        logger.info(f"[CODE] Generated (attempt {gen_attempt}):\n{code[:400]}")
        _dump_code("generated", code)
        return code

    logger.warning("[CODE] All generation attempts produced bad code")
    return None


# ─── Main Entry Point ────────────────────────────────────────────────────────

async def execute_code_task(goal: str, llm_func, tts_func=None,
                            _escalated: bool = False,
                            _hint_packages: list[str] | None = None,
                            _from_planner: bool = False,
                            preference_hints: str = "") -> str:
    """Main entry point: natural-language goal → code → result.

    Args:
        goal:             The user's natural language goal.
        llm_func:         The LLM call function.
        tts_func:         Optional TTS function for progress updates.
        _escalated:       If True, force tier 1 → 2.
        _hint_packages:   Package hints from tier 1 escalation.
        _from_planner:    If True, return raw output without synthesis.
        preference_hints: Preference context for routing
                          (e.g. "messaging_default=whatsapp, music_app=spotify").
    """
    logger.info(f'[CODE] Goal: "{goal}"')

    # ── Route ─────────────────────────────────────────────────────────────
    route = await _route_goal(goal, llm_func, preference_hints=preference_hints)
    tier, slug = route["tier"], route["template_slug"]
    requires, params = route["requires"], route["params"]
    verify_needed = route.get("verification_needed", False)
    logger.info(f"[CODE] Route → tier={tier}, slug={slug}, requires={requires}, params={params}, verify={verify_needed}")

    if _escalated and tier == 1:
        logger.info("[CODE] Escalated call — forcing tier 1 → 2")
        tier = 2

    if tier == "gui":
        return "__NEEDS_GUI__"

    if tier == 1 and _is_system_command(goal):
        from ..automation.system_commands import run_system_command
        return await run_system_command(goal, llm_func)

    # ── Tier 2 ────────────────────────────────────────────────────────────
    if tier == 2:
        # Merge hint packages from Tier 1 escalation
        if _hint_packages:
            for hp in _hint_packages:
                if hp not in requires:
                    requires.append(hp)
            logger.info(f"[CODE] Merged hint packages: {requires}")

        if requires:
            ok, err = _ensure_packages(requires)
            if not ok:
                return f"Package problem: {err}"

        env_vars = {f"PARAM_{k.upper()}": str(v) for k, v in params.items()}

        # OAuth pre-flight — convention-based
        _svc = detect_service_from_packages(requires)
        if _svc:
            from .. import oauth_helper, credentials as cs
            oauth_map = get_oauth_env_map(_svc)
            _tu = cs.get_credential(_svc, "token_url") or ""
            _oe = oauth_helper.get_env_vars(_svc, _tu, oauth_map)
            if _oe:
                env_vars.update(_oe)
                logger.info(f"[CODE] OAuth injected for '{_svc}'")

        # Package env var aliases — handle packages with their own naming conventions.
        # LLMs write SPOTIPY_CLIENT_ID even when told to use SPOTIFY_CLIENT_ID.
        # Injecting aliases means both names resolve to the same value at runtime.
        for _pkg in requires:
            for _alias_key, _alias_src in _sr.PACKAGE_ENV_ALIASES.get(_pkg, {}).items():
                if _alias_key not in env_vars:
                    if _alias_src.startswith("$"):
                        _src = _alias_src[1:]
                        if _src in env_vars:
                            env_vars[_alias_key] = env_vars[_src]
                    else:
                        env_vars[_alias_key] = _alias_src

        # Messaging bridge pre-flight — convention-based
        _msg_svc = detect_messaging_service(requires)
        if _msg_svc:
            from ..io import messaging_bridge
            from ..service_registry import get_service_packages as _get_svc_pkgs
            if not messaging_bridge.is_connected(_msg_svc):
                import asyncio
                _pkgs = _get_svc_pkgs(_msg_svc)
                sentinel = await asyncio.to_thread(messaging_bridge._connect_service, _msg_svc, _pkgs)
                if sentinel and sentinel.startswith("NEEDS_DEVICE_AUTH|"):
                    sig = _process_device_auth_sentinel(sentinel)
                    if sig:
                        return sig
            # Inject bridge port so Tier 2 scripts can call the HTTP API
            env_vars["MESSAGING_BRIDGE_PORT"] = str(messaging_bridge.get_port())
            env_vars["MESSAGING_SERVICE"] = _msg_svc

            # ── Messaging send interception ───────────────────────────────
            # If the router detected a send (params has 'text'), call the
            # bridge directly instead of generating a Tier 2 script.
            # This avoids the problem of LLM scripts not handling the
            # confirmation dict correctly.
            # Non-send actions (read_messages, list_chats, get_contacts)
            # still go through Tier 2 as normal.
            _send_text = params.get("text", "").strip()
            _send_target = params.get("contact_name", "") or params.get("phone", "")
            if _send_text and _send_target:
                logger.info(f"[CODE] Messaging send intercepted — calling bridge directly")
                send_params = {"text": _send_text}
                if params.get("contact_name"):
                    send_params["contact_name"] = params["contact_name"]
                elif params.get("phone"):
                    send_params["phone"] = params["phone"]

                try:
                    send_result = messaging_bridge.execute(_msg_svc, "send_message", send_params)
                    if send_result.get("ok"):
                        # Adapter returned a result — could be confirmation dict or success string
                        inner = send_result["result"]
                        if isinstance(inner, dict) and inner.get("needs_confirmation"):
                            # Return sentinel for actions.py to set pending state
                            import json as _json
                            return f"__CONFIRM_SEND__|{_json.dumps(inner)}"
                        else:
                            # Direct success (shouldn't happen without _confirmed, but handle it)
                            return str(inner)
                    else:
                        # Error from adapter (multiple matches, not found, group blocked, etc.)
                        error_msg = send_result.get("error", "Unknown error")
                        import json as _json
                        is_disambig = "multiple contacts" in error_msg.lower()
                        err_payload = _json.dumps({
                            "service": _msg_svc,
                            "text": _send_text,
                            "error": error_msg,
                            "is_disambiguation": is_disambig,
                        })
                        return f"__SEND_ERROR__|{err_payload}"
                except Exception as e:
                    logger.error(f"[CODE] Messaging send failed: {e}")
                    import json as _json
                    err_payload = _json.dumps({
                        "service": _msg_svc,
                        "text": _send_text,
                        "error": str(e),
                        "is_disambiguation": False,
                    })
                    return f"__SEND_ERROR__|{err_payload}"

        scope_hint = ""
        if _svc:
            from .. import credentials as cs
            _sc = cs.get_credential(_svc, "granted_scopes") or ""
            if _sc:
                scope_hint = f"Granted scopes for {_svc}: {_sc}."

        # Build cred_hint with explicit os.environ.get() lines so the LLM sees
        # the exact variable names to read — no ambiguity about naming conventions.
        # Exclude alias keys (they're redundant copies; showing only canonical names).
        _alias_keys = set()
        for _pkg in requires:
            _alias_keys.update(_sr.PACKAGE_ENV_ALIASES.get(_pkg, {}).keys())
        oauth_names = [k for k in env_vars
                       if not k.startswith("PARAM_") and k != "SANDBOX_DIR"
                       and k not in _alias_keys]
        if oauth_names:
            _access_key = next((k for k in oauth_names if "ACCESS_TOKEN" in k), None)
            _cred_lines = "\n".join(f"  {k} = os.environ.get('{k}')" for k in oauth_names)
            _check = (f"If {_access_key} is None, print NEEDS_OAUTH sentinel and exit. "
                      f"Do NOT check CLIENT_ID or CLIENT_SECRET — only ACCESS_TOKEN absence means re-auth needed."
                      if _access_key else "")
            cred_hint = f"Credentials injected — READ EXACTLY THESE NAMES:\n{_cred_lines}\n{_check}"
        else:
            cred_hint = "No credentials — use free APIs."

        # ── Pre-gen API doc search (Approach 1) ───────────────────────────
        # Load template first — used for both search decision and code
        code, template_was_cached = None, False
        if slug:
            cached, stored_goal = _load_template(slug)
            if cached:
                if _goal_matches_template(goal, stored_goal):
                    code = _sanitize_oauth_imports(cached)
                    template_was_cached = True
                else:
                    logger.info(f"[CODE] Template '{slug}' goal mismatch — generating fresh")
                    slug = _disambiguate_slug_on_mismatch(slug, params)

        api_docs = ""
        if code is None:
            # Skip pre-gen search for messaging bridge services — the generated
            # scripts only call the bridge HTTP API, not the service SDK directly.
            # Searching for "neonize python" docs is wasteful and confusing.
            if not _msg_svc:
                api_docs = await _pre_gen_search(requires, goal)

        if code is None:
            # Inject service knowledge — patterns that work/fail with this service.
            # Gated by config.CODE_EXECUTOR_INJECT_KNOWLEDGE (default OFF for v1.0).
            # WRITES (collection) continue regardless — only the prompt READ is
            # gated so v1.1 has historical data to analyze. See config.py for
            # the rationale (stale "never" entries become permanent dogma).
            knowledge_context = ""
            if _svc:
                from .. import config as _cfg
                if getattr(_cfg, "CODE_EXECUTOR_INJECT_KNOWLEDGE", False):
                    from .. import knowledge
                    knowledge_context = knowledge.render_for_llm(_svc)
                    if knowledge_context:
                        logger.info(f"[CODE] Knowledge injected for '{_svc}' ({len(knowledge_context)} chars)")
                else:
                    logger.debug(f"[CODE] Knowledge injection skipped for '{_svc}' (CODE_EXECUTOR_INJECT_KNOWLEDGE=False)")

            # Inject recent conversation context so the code gen model
            # can resolve back-references ("play that song", "same as last time").
            # 8 turns is enough for continuity without inflating tokens on every call.
            conv_context = ""
            try:
                from .. import memory
                from ..session import get_current_session_id
                conv_context = memory.build_recent_context(
                    limit=8,
                    header="RECENT CONVERSATION (for reference resolution only — do NOT replay these tasks):",
                    session_id=get_current_session_id(),
                )
            except Exception as e:
                logger.debug(f"[CODE] conversation context unavailable: {e}")

            from ..core.datetime_utils import date_context_line
            gen_prompt = (
                f"{date_context_line()}\nGoal: {goal}\nPackages: {requires}\n{cred_hint}\n{scope_hint}\n"
                f"Params: {', '.join(f'PARAM_{k.upper()}' for k in params) if params else 'none'}"
            )
            if conv_context:
                gen_prompt = f"{conv_context}\n\n{gen_prompt}"
            if knowledge_context:
                gen_prompt = f"{gen_prompt}\n\n{knowledge_context}"
            if api_docs:
                gen_prompt = (
                    f"{gen_prompt}\n\nAPI DOCUMENTATION (method names and parameters only — "
                    f"NEVER copy auth patterns from docs; always use auth=ACCESS_TOKEN as per system rules):\n{api_docs}"
                )

            code = await _generate_code(gen_prompt, llm_func)
            if code is None:
                return "Sorry, can't generate working code right now."

        # Service action blocklist — enforcement layer (blocks before execution)
        block_msg = _check_service_blocklist(code, _svc)
        if block_msg:
            logger.warning(f"[CODE] {block_msg}")
            return block_msg

        from ..core.abort import abort, UserAborted
        if abort.is_aborted():
            raise UserAborted(abort.reason)
        from ..io.status_broadcaster import status, StatusPhase
        status.set(StatusPhase.THINKING, detail="running the code")

        # Run
        history = []
        _original_broken_code = None
        result = _run_tier2(code, env_vars=env_vars)
        _dump_code("first_run", code, result)
        if result.startswith("NEEDS_OAUTH|"):
            sig = _process_oauth_sentinel(result)
            if sig:
                return sig

        if result.startswith("NEEDS_DEVICE_AUTH|"):
            sig = _process_device_auth_sentinel(result)
            if sig:
                return sig

        if "CLIENT_OUTDATED|" in result:
            _parts = result.split("CLIENT_OUTDATED|")
            _svc_name = _parts[1].split()[0].strip('"\'') if len(_parts) > 1 else "messaging"
            return f"CLIENT_OUTDATED|{_svc_name}"

        if _is_scope_error(result) and _svc:
            _u = _svc.upper()
            env_no = {k: v for k, v in env_vars.items() if k != f"{_u}_ACCESS_TOKEN"}
            r2 = _run_tier2(code, env_vars=env_no)
            if r2.startswith("NEEDS_OAUTH|"):
                sig = _process_oauth_sentinel(r2)
                if sig:
                    return sig
            return f"The {_svc} token needs more permissions. Try again to re-authorize."

        # ── App-not-running: sentinel + text fallback ─────────────────────
        # Two entry points — both trigger the same launch-and-poll logic:
        #   1. APP_NOT_READY|<service> — structured sentinel the LLM prints when
        #      no active device is found (reliable, unambiguous)
        #   2. _detect_app_not_running() — text fallback for code that predates
        #      the sentinel or ignores the instruction
        # Neither consumes a retry slot. Poll up to 4× every 5s (20s max).
        # Stop early: success, or error is no longer device-related.
        _app_launch_svc = None
        if result.startswith("APP_NOT_READY|"):
            _app_launch_svc = result.split("|")[1].strip() if "|" in result else _svc
            logger.info(f"[CODE] APP_NOT_READY sentinel received for '{_app_launch_svc}'")
        elif _detect_app_not_running(result) and _svc:
            _app_launch_svc = _svc
            logger.info(f"[CODE] App-not-running (text fallback) for '{_app_launch_svc}'")

        if _app_launch_svc:
            _app_opened = False
            import asyncio as _asyncio
            from ..automation import native as _aa

            # ALWAYS call open_app, even when the app is already running.
            # open_app is idempotent — it focuses the existing window if
            # running, otherwise launches. Focusing matters for services
            # like Spotify whose Web API only registers the desktop app as
            # a "device" when it's foregrounded or recently played. Earlier
            # version skipped open_app entirely on already-running, which
            # removed that side effect and left sp.devices() returning
            # empty for tray-minimized Spotify.
            #
            # What we DO skip on already-running:
            #   - The "Opening X, one moment..." TTS (was a lie when app
            #     was already up — bad UX).
            #   - The 20s 4-poll cycle (cold-launch needs time; focus is
            #     near-instant — short 2x2s poll suffices).
            _already_running = _aa.is_app_running(_app_launch_svc)
            if _already_running:
                logger.info(
                    f"[CODE] '{_app_launch_svc}' already running — focusing (no TTS, short poll)"
                )
            elif tts_func:
                await tts_func(f"Opening {_app_launch_svc}, one moment...")
            try:
                await _aa.open_app(_app_launch_svc)
                _app_opened = True
                _polls = 2 if _already_running else 4
                _interval = 2 if _already_running else 5
                for _poll in range(_polls):
                    await _asyncio.sleep(_interval)
                    result = _run_tier2(code, env_vars=env_vars)
                    _dump_code(f"post_launch_poll_{_poll}", code, result)
                    logger.info(
                        f"[CODE] Post-launch poll {_poll+1}/{_polls} "
                        f"(already_running={_already_running}): {result[:80]}"
                    )
                    _still_device = (
                        result.startswith("APP_NOT_READY|")
                        or _detect_app_not_running(result)
                        or "http status: 404" in result
                        or "no device" in result.lower()
                    )
                    if _still_device:
                        continue  # app not ready yet — keep polling
                    break  # success or different error
            except Exception as _le:
                logger.warning(f"[CODE] App launch failed: {_le}")

            if _app_opened and (result.startswith("APP_NOT_READY|") or _detect_app_not_running(result)):
                result = (
                    f"Error: {_app_launch_svc} is confirmed running but the code "
                    f"found no ACTIVE device/instance/session. The AVAILABLE list "
                    f"is likely non-empty. Use any available item (devices[0], "
                    f"sessions[0], etc.) and call the SDK's activate/transfer/focus "
                    f"method to make it active before the action call. Only treat "
                    f"this as truly unavailable if the listing API itself returns "
                    f"an empty list."
                )
                logger.info("[CODE] APP_NOT_READY→retryable after poll exhaustion")

        # ══════════════════════════════════════════════════════════════════
        # RETRY LOOP
        # ══════════════════════════════════════════════════════════════════
        if _needs_retry(result):
            if template_was_cached and slug:
                _delete_template(slug)
                template_was_cached = False

            history = [{"error": result[:200], "diagnosis": None}]
            _original_broken_code = code  # preserve for knowledge extraction

            for attempt in range(1, _MAX_RETRIES + 1):
                if not _needs_retry(result):
                    break
                logger.info(f"[CODE] ── Retry {attempt}/{_MAX_RETRIES} ──")

                # Step A: Classify error deterministically (no LLM call)
                diag = _classify_error(result)
                logger.info(f"[CODE] Classification: cat={diag['category']}, "
                             f"disc={diag['needs_discovery']}, diag={diag['diagnosis'][:80]}")

                if diag["category"] == "blocked":
                    logger.info("[CODE] Sandbox block is a capability gap — skipping fix attempts")
                    break

                # Step B: Recovery path
                discovery_data, web_context = None, None
                key_replacements: dict[str, str] = {}

                # Always try discovery if needed (it's free — no LLM call)
                if diag["needs_discovery"]:
                    logger.info("[CODE] Path: MECHANICAL DISCOVERY")
                    discovery_data = _run_discovery(code, env_vars, error_category=diag["category"])
                    if discovery_data:
                        logger.info(f"[CODE] Discovery data:\n{discovery_data[:500]}")
                        # Enrich: compare keys code expects vs what API returns
                        discovery_data, key_replacements = _enrich_discovery_with_key_analysis(code, discovery_data)
                    else:
                        logger.info("[CODE] Discovery produced no data")

                # Step B2: Deterministic fixes — apply discovered mismatches
                # directly to the code BEFORE involving the LLM. If this resolves
                # the issue, we skip the expensive two-pass LLM fix entirely.
                # Triggers on: key replacements OR empty collection items (fields= issue)
                _has_empty_items = discovery_data and "first_items_keys=[]" in discovery_data
                if key_replacements or _has_empty_items:
                    key_fixed = _apply_key_fixes(code, key_replacements,
                                                  discovery_data=discovery_data or "")
                    if key_fixed:
                        _dump_code(f"retry_{attempt}_keyfixed", key_fixed)
                        key_result = _run_tier2(key_fixed, env_vars=env_vars)
                        _dump_code(f"retry_{attempt}_keyfixed_result", key_fixed, key_result)

                        if key_result.startswith("NEEDS_OAUTH|"):
                            if _svc and f"{_svc.upper()}_ACCESS_TOKEN" in env_vars:
                                key_result = "Error: Script incorrectly requested re-auth. Credentials exist."
                            else:
                                sig = _process_oauth_sentinel(key_result)
                                if sig:
                                    return sig

                        if not _needs_retry(key_result):
                            logger.info(f"[CODE] Retry {attempt}: deterministic key fix resolved the issue!")
                            code = key_fixed
                            result = key_result
                            history.append({
                                "error": "SUCCESS",
                                "diagnosis": diag.get("diagnosis"),
                                "category": diag.get("category", "unknown"),
                                "code_snapshot": code[:500],
                                "discovery_data": discovery_data[:1000] if discovery_data else "",
                            })
                            break
                        else:
                            # Key fix didn't fully resolve — use the partially-fixed
                            # code as the base for the LLM fix. The LLM now handles
                            # remaining issues (encoding, pagination, etc.) without
                            # needing to also figure out key renames.
                            logger.info(f"[CODE] Key fix applied but still failing — passing cleaner code to LLM plan")
                            code = key_fixed
                            result = key_result

                # Step B3: Deterministic kwarg fixes — when discovery reveals
                # correct method signature, mechanically fix wrong kwargs.
                kwarg_fixes = _extract_kwarg_fixes(result, discovery_data or "")
                if kwarg_fixes:
                    kwarg_fixed = _apply_kwarg_fixes(code, kwarg_fixes)
                    if kwarg_fixed:
                        _dump_code(f"retry_{attempt}_kwargfixed", kwarg_fixed)
                        kwarg_result = _run_tier2(kwarg_fixed, env_vars=env_vars)
                        _dump_code(f"retry_{attempt}_kwargfixed_result", kwarg_fixed, kwarg_result)

                        if kwarg_result.startswith("NEEDS_OAUTH|"):
                            if _svc and f"{_svc.upper()}_ACCESS_TOKEN" in env_vars:
                                kwarg_result = "Error: Script incorrectly requested re-auth. Credentials exist."
                            else:
                                sig = _process_oauth_sentinel(kwarg_result)
                                if sig:
                                    return sig

                        if not _needs_retry(kwarg_result):
                            logger.info(f"[CODE] Retry {attempt}: deterministic kwarg fix resolved the issue!")
                            code = kwarg_fixed
                            result = kwarg_result
                            history.append({
                                "error": "SUCCESS",
                                "diagnosis": diag.get("diagnosis"),
                                "category": diag.get("category", "unknown"),
                                "code_snapshot": code[:500],
                                "discovery_data": discovery_data[:1000] if discovery_data else "",
                            })
                            break
                        else:
                            logger.info("[CODE] Kwarg fix applied but still failing — passing cleaner code to LLM plan")
                            code = kwarg_fixed
                            result = kwarg_result

                # On 403/deprecated errors, search for replacement endpoint
                if diag["category"] == "api_endpoint":
                    failing_urls = _extract_api_urls(code)
                    if failing_urls:
                        fail_url = failing_urls[-1]
                        search_q = [f"{fail_url} API deprecated replacement"]
                        logger.info(f"[CODE] Path: targeted endpoint search — {search_q}")
                        web_context = await _search_and_fetch(search_q)
                        diag["diagnosis"] += f". Failing URL: {fail_url}"

                # On retry 2+, also do generic web search for docs
                if attempt >= 2 and not web_context:
                    pkg_queries = [f"{p} python API example" for p in requires[:1]]
                    if pkg_queries:
                        logger.info(f"[CODE] Path: web search — {pkg_queries}")
                        web_context = await _search_and_fetch(pkg_queries)

                # Inject pre-gen docs if we had them and no other web context
                if api_docs and not web_context:
                    web_context = api_docs

                # Step C: TWO-PASS FIX — plan with big model, execute with code gen
                # Pass 1: Plan (agent_plan → llama-3.3-70b, ~100 tokens output)
                plan = await _plan_fix(goal, code, result, diag,
                                       discovery_data, web_context,
                                       cred_hint, scope_hint, llm_func,
                                       service=_svc)
                if not plan:
                    history.append({"error": "Plan gen failed", "diagnosis": diag.get("diagnosis")})
                    continue

                # Pass 2: Execute plan (code_gen → kimi-k2, follows instructions)
                fixed = await _execute_fix_plan(code, plan, llm_func)
                if not fixed:
                    history.append({"error": "Plan execute failed", "diagnosis": diag.get("diagnosis")})
                    continue

                # Step D: Run
                code = fixed
                _dump_code(f"retry_{attempt}_fix", fixed)

                # Service action blocklist — also check fixed code
                block_msg = _check_service_blocklist(fixed, _svc)
                if block_msg:
                    logger.warning(f"[CODE] Retry {attempt}: {block_msg}")
                    history.append({"error": block_msg, "diagnosis": diag.get("diagnosis")})
                    result = block_msg
                    continue

                result = _run_tier2(fixed, env_vars=env_vars)
                _dump_code(f"retry_{attempt}_result", fixed, result)

                if result.startswith("NEEDS_OAUTH|"):
                    if _svc and f"{_svc.upper()}_ACCESS_TOKEN" in env_vars:
                        result = "Error: Script incorrectly requested re-auth. Credentials exist."
                    else:
                        sig = _process_oauth_sentinel(result)
                        if sig:
                            return sig

                if _is_scope_error(result) and _svc:
                    env_no = {k: v for k, v in env_vars.items() if k != f"{_svc.upper()}_ACCESS_TOKEN"}
                    r2 = _run_tier2(fixed, env_vars=env_no)
                    if r2.startswith("NEEDS_OAUTH|"):
                        sig = _process_oauth_sentinel(r2)
                        if sig:
                            return sig

                history.append({
                    "error": result[:200] if _needs_retry(result) else "SUCCESS",
                    "diagnosis": diag.get("diagnosis"),
                    "category": diag.get("category", "unknown"),
                    "code_snapshot": code[:500],
                    "discovery_data": discovery_data[:1000] if discovery_data else "",
                })
                if not _needs_retry(result):
                    logger.info(f"[CODE] Retry {attempt} succeeded!")

        # Exhausted
        if _needs_retry(result):
            logger.warning(f"[CODE] All {_MAX_RETRIES} retries exhausted")

            # Save structural failure knowledge (auto, no approval needed)
            if _svc and slug:
                await _save_failure_knowledge(_svc, slug, history, llm_func)

            if _from_planner:
                return result[:200]

            if verify_needed:
                logger.info("[CODE] Retries exhausted on action goal → escalating to planner")
                return "__ESCALATE_PLANNER__"

            fb = await llm_func(
                f'The user asked: "{goal}"\n'
                f'The action FAILED after multiple attempts. Error: {result[:200]}\n'
                f'Apologize briefly and suggest the user try manually. Do NOT pretend the action succeeded.',
                task_type="synthesis", max_tokens=200)
            return fb if fb != "__LLM_UNAVAILABLE__" else "Sorry, I couldn't complete that task."

        # ── Synthesize result for spoken output ─────────────────────────
        # When called from the planner, skip synthesis — return raw code
        # output so downstream $step_N references get clean data, not
        # personality-wrapped spoken responses. Saves 1 API call per step.
        if _from_planner:
            if slug and not template_was_cached and not _needs_retry(result) and not result.startswith("NEEDS_OAUTH|"):
                _save_template(slug, code, goal=goal, params=params)
            return result

        # Verification gate: action goals must produce completion evidence.
        # When code ran but didn't actually accomplish anything, escalate
        # to the planner which can try browser-based automation instead.
        if verify_needed and not _has_completion_evidence(result):
            logger.info("[CODE] Verification gate: action goal with no completion evidence → escalating to planner")
            return "__ESCALATE_PLANNER__"

        # Success — save template (NEVER save if result is a sentinel or error)
        # Placed after verification gate so useless templates aren't cached.
        if slug and not template_was_cached and not _needs_retry(result) and not result.startswith("NEEDS_OAUTH|"):
            _save_template(slug, code, goal=goal, params=params)

        # Extract knowledge after successful retry. The proposal return value
        # is intentionally discarded — _save_success_knowledge writes the entry
        # itself; we do not append it to TTS output (would bloat spoken response).
        if _svc and slug and len(history) > 1:
            await _save_success_knowledge(
                _svc, slug, _original_broken_code, code, history, llm_func
            )

        synth = await llm_func(
            f'User asked: "{goal}"\nResult:\n{result}\n\nConcise spoken response (1-3 sentences). '
            f'Only describe what is in the Result.', task_type="synthesis", max_tokens=400)
        final = synth if synth != "__LLM_UNAVAILABLE__" else result

        return final

    # ── Tier 1 ────────────────────────────────────────────────────────────
    from ..core.datetime_utils import date_context_line
    code = await llm_func(f"{date_context_line()}\nGoal: {goal}", system_prompt=_CODE_GEN_SYSTEM_PROMPT_TIER1,
                          task_type="code_gen", max_tokens=400)
    if code == "__LLM_UNAVAILABLE__":
        return "Sorry, no LLM available."
    code = _strip_code_fences(code)
    from .. import config as _c
    _t = 3 if _c.CODE_EXECUTOR_POWER_MODE else 1
    result = run_code(code, tier=_t)

    if _needs_retry(result):
        # Check if failure is a capability limitation (not a code bug).
        # "BLOCKED:" means the sandbox rejected an import, call, or write —
        # the task needs more capabilities than Tier 1 offers.
        # Escalate to Tier 2 (same pattern as GUI handoff).
        if result.startswith("BLOCKED:") and not _escalated:
            logger.info(f"[CODE] Tier 1 blocked → escalating to Tier 2: {result[:80]}")
            _pkg_match = re.search(r"import of '(\w+)'", result)
            _blocked_name = _pkg_match.group(1) if _pkg_match else None
            _hint = _IMPORT_TO_PACKAGE.get(_blocked_name, _blocked_name) if _blocked_name else None
            _hints = [_hint] if _hint and _hint in TIER2_ALLOWED_PACKAGES else None
            if _hints:
                logger.info(f"[CODE] Carrying forward blocked package hint: {_hints}")
            return await execute_code_task(goal, llm_func, tts_func, _escalated=True, _hint_packages=_hints, _from_planner=_from_planner)

        fix_prompt = f"PLAN:\n1. Fix the error shown below.\n\nBroken code to fix:\n```python\n{code}\n```\nError:\n{result}"
        fixed = await llm_func(fix_prompt,
                               system_prompt=_FIX_EXECUTE_PROMPT, task_type="code_gen", max_tokens=400)
        if fixed != "__LLM_UNAVAILABLE__":
            result = run_code(_strip_code_fences(fixed), tier=_t)
        if _needs_retry(result):
            # Second failure — also check for BLOCKED before giving up
            if result.startswith("BLOCKED:") and not _escalated:
                logger.info(f"[CODE] Tier 1 retry blocked → escalating to Tier 2: {result[:80]}")
                _pkg_match = re.search(r"import of '(\w+)'", result)
                _blocked_name = _pkg_match.group(1) if _pkg_match else None
                _hint = _IMPORT_TO_PACKAGE.get(_blocked_name, _blocked_name) if _blocked_name else None
                _hints = [_hint] if _hint and _hint in TIER2_ALLOWED_PACKAGES else None
                if _hints:
                    logger.info(f"[CODE] Carrying forward blocked package hint: {_hints}")
                return await execute_code_task(goal, llm_func, tts_func, _escalated=True, _hint_packages=_hints, _from_planner=_from_planner)

            # When called from planner, return raw error without synthesis
            if _from_planner:
                return result[:200]

            fb = await llm_func(
                f'The user asked: "{goal}"\nCode execution failed. Error: {result[:200]}\n'
                f'If you can answer from knowledge, do so. Otherwise apologize briefly. '
                f'Do NOT pretend an action succeeded.',
                task_type="synthesis", max_tokens=200)
            return fb if fb != "__LLM_UNAVAILABLE__" else await _clean_error_for_user(result, llm_func)

    # Skip synthesis when called from planner — raw output is better data
    if _from_planner:
        return result

    if verify_needed and not _has_completion_evidence(result):
        logger.info("[CODE] Verification gate: action goal but no completion evidence in output")
        synth = await llm_func(
            f'User asked: "{goal}"\nCode output:\n{result}\n\n'
            f'The code computed data but did NOT actually perform the requested action. '
            f'Tell the user briefly that this action cannot be done through code alone '
            f'and suggest trying a different approach. Do NOT claim the action succeeded.',
            task_type="synthesis", max_tokens=200)
    else:
        synth = await llm_func(
            f'User asked: "{goal}"\nResult:\n{result}\n\nConcise spoken response.',
            task_type="synthesis", max_tokens=400)
    return synth if synth != "__LLM_UNAVAILABLE__" else result
