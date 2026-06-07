"""Desktop automation and code execution handlers."""

import logging

from .registry import tool_registry
from .responses import personality_say

logger = logging.getLogger("actions")


async def _llm_text(prompt, **kwargs):
    """Thin adapter: call get_llm_response and return the plain text string.

    Downstream automation consumers (system_commands, code_executor, planner)
    expect an ``llm_func`` that returns ``str``.  Since ``get_llm_response``
    now returns an ``LLMResult`` object, we unwrap ``.text`` here so every
    ``llm_func=`` call site stays clean.
    """
    from .. import llm as llm_module
    result = await llm_module.get_llm_response(prompt, **kwargs)
    return result.text


@tool_registry.decorator("computer_task")
async def handle_computer_task(params: dict, llm_response: str, bridge=None,
                               _from_planner: bool = False) -> str:
    """Run the agentic computer control loop to accomplish a goal."""
    import uuid as _uuid
    from ..core.abort import abort, UserAborted
    from ..io.status_broadcaster import status, StatusPhase
    _task_id = f"computer_task:{_uuid.uuid4().hex[:8]}"
    if not _from_planner:
        abort.reset()
        abort.register_task(_task_id)
    status.set(StatusPhase.THINKING, detail=str(params.get("goal", ""))[:40])

    try:
        from ..automation import vision as computer_agent
        from .. import llm as llm_module
        from ..io.audio import tts

        if bridge:
            await bridge.send_thought("thinking")
            await bridge.send_keyboard(True)

        goal = params.get("goal", "")
        if not goal:
            if bridge:
                await bridge.send_thought("done", "")
                await bridge.send_keyboard(False)
            return personality_say("need_query")

        action_keywords = ["disable", "enable", "turn on", "turn off", "stop", "start", "connect", "disconnect", "restart"]
        goal_lower = goal.lower()

        if any(kw in goal_lower for kw in action_keywords):
            logger.info(f"[ACTIONS] Computer task '{goal}' contains action keywords, trying Tier 2 System Command first.")
            from ..automation.system_commands import run_system_command

            system_result = await run_system_command(goal, _llm_text)
            if not system_result.startswith("Blocked:"):
                logger.info(f"[ACTIONS] System command succeeded/executed: {system_result}")
                if bridge:
                    await bridge.send_command("play_animation", name="task_complete")
                response_text = f"Executed system command: {system_result}"
                if bridge:
                    await bridge.send_thought("done", response_text)
                    await bridge.send_keyboard(False)
                return response_text
            else:
                logger.info(f"[ACTIONS] System command approach blocked, falling back to GUI agent loop: {system_result}")

        # Try deterministic automation before vision loop
        try:
            from ..automation import router as desktop_automation
            from .. import llm as _da_llm
            from ..llm.contracts import ask_for_synthesis

            try:
                from ..automation.browser import cdp as _bcdp
                await _bcdp.cdp_health_probe(use_cache=False)
            except Exception as _probe_err:
                logger.debug(f"[ACTIONS] CDP refresh probe failed: {_probe_err}")

            da_can, da_backend = await desktop_automation.can_handle(goal)
            if da_can:
                logger.info(f"[ACTIONS] Trying {da_backend} automation for '{goal}'")
                da_result = await desktop_automation.execute_automation(
                    goal=goal,
                    llm_func=_llm_text,
                )
                if da_result != "__FALLBACK__":
                    from ..automation import verification as _ver
                    _verify_failure = _ver.parse_verify_failed(da_result)

                    if _verify_failure:
                        logger.warning(
                            f"[ACTIONS] {da_backend} automation verify_failed "
                            f"(tier={_verify_failure['tier']}) — {_verify_failure['observation']}"
                        )
                        try:
                            synth = await ask_for_synthesis(
                                f'User asked: "{goal}"\n'
                                f'I tried, but verification caught a problem at step '
                                f'{_verify_failure["step"]}: {_verify_failure["observation"]}\n\n'
                                f'Tell the user honestly that it didn\'t actually work, in 1-2 sentences. '
                                f'Be specific about what you observed. Do NOT pretend the action succeeded.',
                                max_tokens=200,
                            )
                            if synth and synth != "__LLM_UNAVAILABLE__":
                                da_result = synth
                            else:
                                da_result = (
                                    f"That didn't actually take — {_verify_failure['observation']}"
                                )
                        except Exception as synth_err:
                            logger.debug(f"[ACTIONS] Verify-failure synthesis failed: {synth_err}")
                            da_result = (
                                f"That didn't actually take — {_verify_failure['observation']}"
                            )
                    else:
                        logger.info(f"[ACTIONS] {da_backend} automation succeeded")
                        try:
                            synth = await ask_for_synthesis(
                                f'User asked: "{goal}"\n'
                                f'Automation result:\n{da_result}\n\n'
                                f'Give a concise spoken response (1-2 sentences). '
                                f'Report the result naturally. If there is extracted data, include it. '
                                f'Do NOT describe the steps taken — just the outcome.',
                                max_tokens=200,
                            )
                            if synth and synth != "__LLM_UNAVAILABLE__":
                                da_result = synth
                        except Exception as synth_err:
                            logger.debug(f"[ACTIONS] Synthesis failed, using raw: {synth_err}")

                    if bridge:
                        await bridge.send_thought("done", da_result)
                        await bridge.send_keyboard(False)
                        await tts.speak(da_result, bridge)
                    return da_result
                else:
                    logger.info(f"[ACTIONS] {da_backend} automation returned fallback, continuing to vision loop")
        except UserAborted:
            raise  # let outer except handle it
        except Exception as da_err:
            logger.warning(f"[ACTIONS] Desktop automation error, falling to vision loop: {da_err}")

        async def agent_tts(text):
            if bridge:
                await tts.speak(text, bridge)

        async def agent_bridge(cmd, **kwargs):
            if bridge:
                await bridge.send_command(cmd, **kwargs)

        try:
            result = await computer_agent.run_computer_task(
                goal=goal,
                llm_func=_llm_text,
                tts_func=agent_tts,
                bridge_func=agent_bridge,
            )
        except UserAborted:
            raise  # let outer except handle it
        except Exception:
            if bridge:
                await bridge.send_thought("done", "")
                await bridge.send_keyboard(False)
            raise

        if bridge:
            await bridge.send_thought("done", result)
            await bridge.send_keyboard(False)
        return result

    except UserAborted:
        # When called from the planner, re-raise so the planner sees the abort
        # (not a "Stopped." string mistaken for a successful step output).
        if _from_planner:
            raise
        try:
            return personality_say("stopped", default="Stopped.")
        except Exception:
            return "Stopped."
    finally:
        if not _from_planner:
            abort.unregister_task(_task_id)
            status.set(StatusPhase.IDLE)


@tool_registry.decorator("read_screen")
async def handle_read_screen(params: dict, llm_response: str, bridge=None) -> str:
    """OCR the screen and return a natural language summary."""
    from ..io import screen
    from ..llm.contracts import ask_for_synthesis

    ocr_text = screen.ocr_screen()

    if not ocr_text:
        return "I can't read this. It's either too dark or you're blocking my view!"

    summary_prompt = (
        f"The user asked what's on their screen. Here's the OCR text:\n\n"
        f"{ocr_text[:3000]}\n\n"
        f"Provide a brief, helpful summary of what's currently on their screen."
    )

    summary = await ask_for_synthesis(summary_prompt)
    if summary == "__LLM_UNAVAILABLE__":
        if len(ocr_text) > 500:
            ocr_text = ocr_text[:500] + "..."
        return f"Here's what I can see on your screen: {ocr_text}"

    return summary


@tool_registry.decorator("find_and_click")
async def handle_find_and_click(params: dict, llm_response: str, bridge=None,
                                _from_planner: bool = False) -> str:
    """Find text on the screen and click on it."""
    import uuid as _uuid
    from ..core.abort import abort, UserAborted
    from ..io.status_broadcaster import status, StatusPhase
    _task_id = f"find_and_click:{_uuid.uuid4().hex[:8]}"

    target = params.get("target", params.get("text", ""))
    if not _from_planner:
        abort.reset()
        abort.register_task(_task_id)
    status.set(StatusPhase.THINKING, detail=target[:40])

    try:
        if abort.is_aborted():
            raise UserAborted(abort.reason)

        from ..io import screen

        text = params.get("text", params.get("target", ""))
        if not text:
            return personality_say("need_query")

        import pyautogui

        matches = screen.find_text_on_screen(text)

        if abort.is_aborted():
            raise UserAborted(abort.reason)

        if not matches:
            return f'I couldn\'t find "{text}" on the screen.'

        x, y = matches[0]
        status.set(StatusPhase.CLICKING, detail=text[:40], cursor_follows=True, tier="native")
        try:
            pyautogui.click(x, y)
            return f'Found and clicked on "{text}" at position ({x}, {y}).'
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return f'I found "{text}" at ({x}, {y}) but couldn\'t click on it: {e}'
    except UserAborted:
        # When called from the planner, re-raise so the planner sees the abort
        # (not a "Stopped." string mistaken for a successful step output).
        if _from_planner:
            raise
        try:
            return personality_say("stopped", default="Stopped.")
        except Exception:
            return "Stopped."
    finally:
        if not _from_planner:
            abort.unregister_task(_task_id)
            status.set(StatusPhase.IDLE)


# --- Planner Handler ---

@tool_registry.decorator("planner")
async def handle_planner(params: dict, llm_response: str, bridge=None) -> str:
    """Orchestrate a multi-step plan to accomplish a complex goal."""
    import uuid as _uuid
    from ..core.abort import abort, UserAborted
    from ..io.status_broadcaster import status, StatusPhase

    import assistant.actions as _act
    from .planner import planner
    from .. import llm as llm_module
    from ..io.audio import tts as tts_module

    goal = params.get("goal", "")
    if not goal:
        return "What do you want me to do? Give me a goal, dummy."

    _task_id = f"planner:{_uuid.uuid4().hex[:8]}"
    abort.reset()
    abort.register_task(_task_id)
    status.set(StatusPhase.THINKING, detail=str(goal)[:40])
    try:
        async def agent_tts(text):
            if bridge:
                await tts_module.speak(text, bridge, emotion="neutral")

        result = await planner.execute_plan(
            goal=goal,
            llm_func=_llm_text,
            tts_func=agent_tts,
            bridge=bridge,
        )

        if result is None:
            return await handle_code_executor(params, llm_response, bridge,
                                              _from_planner=True)

        if isinstance(result, dict) and "bypass" in result:
            tool_name = result["bypass"]
            bypass_params = {**params, "goal": result["goal"]}
            bypass_result = await _act.execute(tool_name, bypass_params, llm_response,
                                               bridge, _from_planner=True)
            from ..automation import verification as _ver
            parsed = _ver.parse_verify_failed(bypass_result or "")
            if parsed:
                return _ver.format_failure_for_user(parsed)
            from ..llm.contracts import ask_for_synthesis
            try:
                synth = await ask_for_synthesis(
                    f'User asked: "{goal}"\n'
                    f'Result:\n{(bypass_result or "")[:1500]}\n\n'
                    f'Concise spoken response (1-2 sentences). '
                    f'Summarize what happened and the key information.',
                    max_tokens=200,
                )
                if synth and synth != "__LLM_UNAVAILABLE__":
                    return synth
            except Exception:
                pass
            return bypass_result

        return result
    except UserAborted:
        try:
            return personality_say("stopped", default="Stopped.")
        except Exception:
            return "Stopped."
    finally:
        status.set(StatusPhase.IDLE)
        abort.unregister_task(_task_id)


# --- Code Executor Handler ---

@tool_registry.decorator("code_executor")
async def handle_code_executor(params: dict, llm_response: str, bridge=None,
                               _from_planner: bool = False) -> str:
    """Run LLM-generated Python code to answer a question or compute something."""
    import uuid as _uuid
    from ..core.abort import abort, UserAborted
    from ..io.status_broadcaster import status, StatusPhase
    _task_id = f"code_executor:{_uuid.uuid4().hex[:8]}"

    goal = params.get("goal", "")
    if not _from_planner:
        abort.reset()
        abort.register_task(_task_id)
    status.set(StatusPhase.THINKING, detail=goal[:40])

    try:
        if abort.is_aborted():
            raise UserAborted(abort.reason)

        import assistant.actions as _act
        from .. import code_executor
        from .. import llm as llm_module
        from ..llm.contracts import ask_for_synthesis
        from ..io.audio import tts
        from ..code_executor import GUI_HANDOFF_SIGNAL

        if not goal:
            return personality_say("need_query")

        # Multi-step detection — reroute to planner
        if not _from_planner:
            from .planner import planner
            if planner.needs_planning(goal):
                logger.info(f"[ACTIONS] Multi-step goal detected → delegating to planner")
                return await handle_planner(params, llm_response, bridge)

        async def agent_tts(text):
            if bridge:
                await tts.speak(text, bridge)

        pref_hints = params.get("_pref_hints", "")

        result = await code_executor.execute_code_task(
            goal=goal,
            llm_func=_llm_text,
            tts_func=agent_tts,
            _from_planner=_from_planner,
            preference_hints=pref_hints,
        )

        # OAuth setup signal
        if result.startswith("__NEEDS_OAUTH__|"):
            parts = result.split("|")
            if len(parts) == 6:
                _act.pending_oauth_setup.set({
                    "service":       parts[1],
                    "auth_url":      parts[2],
                    "token_url":     parts[3],
                    "scopes":        parts[4],
                    "redirect_uri":  parts[5],
                    "original_goal": goal,
                    "step":          "has_app",
                })
                service_name = parts[1].title()
                return f"I need to set up {service_name} first. Do you already have a {service_name} developer app?"

        # Device auth setup signal (WhatsApp, Telegram, etc.)
        if result.startswith("__NEEDS_DEVICE_AUTH__|"):
            parts = result.split("|")
            if len(parts) >= 3:
                service_name = parts[1]
                session_path = parts[2]

                from ..io import messaging_bridge
                messaging_bridge.connect_for_pairing(service_name)

                _act.pending_device_auth.set({
                    "service": service_name,
                    "session_path": session_path,
                    "original_goal": goal,
                })
                return (
                    f"I need to set up {service_name.title()} first. "
                    f"A QR code should appear in the terminal. "
                    f"Open {service_name.title()} on your phone, go to Settings, then Linked Devices, "
                    f"then Link a Device, and scan the QR code. Say 'done' when you've scanned it. "
                    f"Or say your phone number and I'll give you a link code instead."
                )

        # Client outdated signal
        if result.startswith("CLIENT_OUTDATED|"):
            parts = result.split("|")
            service_name = parts[1] if len(parts) >= 2 else "messaging"
            return (
                f"{service_name.title()} rejected the connection as outdated. "
                f"The library needs an upgrade — check the terminal for the exact command, then restart."
            )

        # Messaging send confirmation
        if result.startswith("__CONFIRM_SEND__|"):
            import json as _json
            try:
                payload_str = result[len("__CONFIRM_SEND__|"):]
                confirm_data = _json.loads(payload_str)
                _act.pending_messaging_send.set(confirm_data)

                resolved_name = confirm_data.get("resolved_name", confirm_data.get("phone", "someone"))
                text_preview = confirm_data.get("text", "")
                if len(text_preview) > 50:
                    text_preview = text_preview[:47] + "..."

                return (
                    personality_say("msg_confirm") + f" (Send '{text_preview}' to {resolved_name}?)"
                )
            except Exception as e:
                logger.error(f"[MESSAGING] Failed to parse confirmation signal: {e}")

        # Messaging send error
        if result.startswith("__SEND_ERROR__|"):
            import json as _json
            try:
                payload_str = result[len("__SEND_ERROR__|"):]
                err_data = _json.loads(payload_str)
                error_msg = err_data.get("error", "Unknown error")

                if err_data.get("is_disambiguation"):
                    _act.pending_messaging_disambig.set({
                        "service": err_data.get("service", ""),
                        "text": err_data.get("text", ""),
                    })

                return error_msg
            except Exception as e:
                logger.error(f"[MESSAGING] Failed to parse send error signal: {e}")
                return result.split("|", 1)[-1] if "|" in result else result

        # Planner escalation — code_executor couldn't fulfill an action goal via APIs
        from ..code_executor import PLANNER_ESCALATION_SIGNAL
        if result == PLANNER_ESCALATION_SIGNAL:
            logger.info(f"[ACTIONS] Code executor → planner escalation for '{goal}'")
            return await handle_planner(params, llm_response, bridge)

        # GUI handoff — try deterministic automation first, then vision loop
        if result == GUI_HANDOFF_SIGNAL:
            try:
                from ..automation import router as desktop_automation
                da_can, da_backend = await desktop_automation.can_handle(goal)
                if da_can:
                    logger.info(f"[ACTIONS] GUI handoff: trying {da_backend} for '{goal}'")
                    if bridge:
                        await tts.speak(
                            "Let me handle this.",
                            bridge,
                            emotion="neutral",
                        )
                    da_result = await desktop_automation.execute_automation(
                        goal=goal,
                        llm_func=_llm_text,
                    )
                    if da_result != "__FALLBACK__":
                        try:
                            synth = await ask_for_synthesis(
                                f'User asked: "{goal}"\n'
                                f'Automation result:\n{da_result}\n\n'
                                f'Concise spoken response (1-2 sentences). Report the outcome.',
                                max_tokens=200,
                            )
                            if synth and synth != "__LLM_UNAVAILABLE__":
                                da_result = synth
                        except Exception:
                            pass
                        return da_result
                    logger.info("[ACTIONS] GUI handoff: fallback to vision loop")
            except Exception as da_err:
                logger.warning(f"[ACTIONS] GUI handoff error: {da_err}")

            if bridge:
                await tts.speak(
                    "Let me do this through the screen.",
                    bridge,
                    emotion="neutral",
                )
            computer_params = {"goal": goal}
            return await handle_computer_task(computer_params, llm_response, bridge,
                                              _from_planner=_from_planner)

        return result
    except UserAborted:
        # When called from the planner, re-raise so the planner sees the abort
        # (not a "Stopped." string mistaken for a successful step output).
        if _from_planner:
            raise
        try:
            return personality_say("stopped", default="Stopped.")
        except Exception:
            return "Stopped."
    finally:
        if not _from_planner:
            abort.unregister_task(_task_id)
            status.set(StatusPhase.IDLE)


# --- Desktop Automation Handlers ---

@tool_registry.decorator("browser_action")
async def handle_browser_action(params: dict, llm_response: str, bridge=None,
                                _from_planner: bool = False) -> str:
    """Execute a browser automation task via Playwright."""
    import uuid as _uuid
    from ..core.abort import abort, UserAborted
    from ..io.status_broadcaster import status, StatusPhase
    _task_id = f"browser_action:{_uuid.uuid4().hex[:8]}"

    if not _from_planner:
        abort.reset()
        abort.register_task(_task_id)
    status.set(StatusPhase.THINKING, detail=params.get("url", "")[:40])

    try:
        if abort.is_aborted():
            raise UserAborted(abort.reason)

        from ..automation import router as desktop_automation
        from .. import llm as llm_module
        from ..llm.contracts import ask_for_synthesis

        goal = params.get("goal", "")
        if not goal:
            return personality_say("need_query")

        _planner_goal = params.pop("_planner_goal", "")

        logger.info(f"[ACTIONS] Browser action: {goal}")
        result = await desktop_automation._execute_browser_task(
            goal=goal,
            llm_func=_llm_text,
            _from_planner=_from_planner,
            _planner_goal=_planner_goal,
        )
        if result == "__FALLBACK__":
            logger.info("[ACTIONS] Browser action fallback → computer_task")
            return await handle_computer_task(
                {"goal": goal}, llm_response, bridge,
                _from_planner=_from_planner,
            )

        if not _from_planner:
            from ..automation import verification as _ver
            _vf = _ver.parse_verify_failed(result)
            if _vf:
                logger.warning(
                    f"[ACTIONS] browser_action verify_failed "
                    f"(tier={_vf['tier']}) — {_vf['observation']}"
                )
                try:
                    synth = await ask_for_synthesis(
                        f'User asked: "{goal}"\n'
                        f'I tried, but verification caught a problem at step '
                        f'{_vf["step"]}: {_vf["observation"]}\n\n'
                        f'Tell the user honestly that it didn\'t actually work, in 1-2 sentences. '
                        f'Be specific about what you observed. Do NOT pretend the action succeeded.',
                        max_tokens=200,
                    )
                    if synth and synth != "__LLM_UNAVAILABLE__":
                        result = synth
                    else:
                        result = f"That didn't work — {_vf['observation']}"
                except Exception:
                    result = f"That didn't work — {_vf['observation']}"
            else:
                try:
                    synth = await ask_for_synthesis(
                        f'User asked: "{goal}"\n'
                        f'Browser result:\n{result[:1500]}\n\n'
                        f'Concise spoken response (1-3 sentences). '
                        f'Summarize the key information from the page.',
                        max_tokens=400,
                    )
                    if synth and synth != "__LLM_UNAVAILABLE__":
                        result = synth
                except Exception:
                    pass

        return result
    except UserAborted:
        try:
            return personality_say("stopped", default="Stopped.")
        except Exception:
            return "Stopped."
    except Exception as e:
        logger.error(f"[ACTIONS] Browser action error: {e}")
        return f"Browser automation failed: {e}"
    finally:
        if not _from_planner:
            abort.unregister_task(_task_id)
            status.set(StatusPhase.IDLE)


@tool_registry.decorator("app_action")
async def handle_app_action(params: dict, llm_response: str, bridge=None,
                            _from_planner: bool = False) -> str:
    """Execute a native app automation task via Terminator."""
    import uuid as _uuid
    from ..core.abort import abort, UserAborted
    from ..io.status_broadcaster import status, StatusPhase
    _task_id = f"app_action:{_uuid.uuid4().hex[:8]}"

    if not _from_planner:
        abort.reset()
        abort.register_task(_task_id)
    status.set(StatusPhase.THINKING, detail=params.get("app", "")[:40])

    try:
        if abort.is_aborted():
            raise UserAborted(abort.reason)

        from ..automation import router as desktop_automation
        from .. import llm as llm_module
        from ..llm.contracts import ask_for_synthesis

        goal = params.get("goal", "")
        if not goal:
            return personality_say("need_query")

        logger.info(f"[ACTIONS] App action: {goal}")
        result = await desktop_automation._execute_native_task(
            goal=goal,
            llm_func=_llm_text,
        )
        if result == "__FALLBACK__":
            logger.info("[ACTIONS] App action fallback → computer_task")
            return await handle_computer_task(
                {"goal": goal}, llm_response, bridge,
                _from_planner=_from_planner,
            )

        if not _from_planner:
            from ..automation import verification as _ver
            _vf = _ver.parse_verify_failed(result)
            if _vf:
                logger.warning(
                    f"[ACTIONS] app_action verify_failed "
                    f"(tier={_vf['tier']}) — {_vf['observation']}"
                )
                try:
                    synth = await ask_for_synthesis(
                        f'User asked: "{goal}"\n'
                        f'I tried, but verification caught a problem at step '
                        f'{_vf["step"]}: {_vf["observation"]}\n\n'
                        f'Tell the user honestly that it didn\'t actually work, in 1-2 sentences. '
                        f'Be specific about what you observed. Do NOT pretend the action succeeded.',
                        max_tokens=200,
                    )
                    if synth and synth != "__LLM_UNAVAILABLE__":
                        result = synth
                    else:
                        result = f"That didn't work — {_vf['observation']}"
                except Exception:
                    result = f"That didn't work — {_vf['observation']}"
            else:
                try:
                    synth = await ask_for_synthesis(
                        f'User asked: "{goal}"\n'
                        f'App result:\n{result}\n\n'
                        f'Concise spoken response (1-2 sentences). Report the outcome.',
                        max_tokens=200,
                    )
                    if synth and synth != "__LLM_UNAVAILABLE__":
                        result = synth
                except Exception:
                    pass

        return result
    except UserAborted:
        try:
            return personality_say("stopped", default="Stopped.")
        except Exception:
            return "Stopped."
    except Exception as e:
        logger.error(f"[ACTIONS] App action error: {e}")
        return f"App automation failed: {e}"
    finally:
        if not _from_planner:
            abort.unregister_task(_task_id)
            status.set(StatusPhase.IDLE)
