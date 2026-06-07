"""Pending state handlers: device auth, OAuth, messaging, knowledge approval."""

import logging

from .responses import personality_say

logger = logging.getLogger("actions")


async def _llm_text(prompt, **kwargs):
    """Thin adapter: call get_llm_response and return the plain text string.

    Downstream code_executor consumers expect an ``llm_func`` that returns
    ``str``.  Since ``get_llm_response`` now returns an ``LLMResult`` object
    we unwrap ``.text`` here so every ``llm_func=`` call site stays clean.
    """
    from .. import llm as llm_module
    result = await llm_module.get_llm_response(prompt, **kwargs)
    return result.text


async def handle_pending_device_auth(text: str, bridge=None) -> str | None:
    """
    Handle pending device auth (QR code pairing) for messaging services.

    Two modes:
    1. Auto-detect: If the adapter connected while the user was scanning,
       auto-retry the original goal immediately — no need to say "done".
    2. Manual: If user says "done" or "cancel", handle accordingly.
    """
    import assistant.actions as _act

    if not _act.pending_device_auth.active:
        return None

    from ..io import messaging_bridge
    from .. import llm as llm_module

    service = _act.pending_device_auth.payload["service"]
    original_goal = _act.pending_device_auth.payload.get("original_goal", "")

    # — CLIENT_OUTDATED check
    if messaging_bridge.is_client_outdated(service):
        _act.pending_device_auth.clear()
        return (
            f"{service.title()} rejected the connection as outdated. "
            f"The library needs an upgrade — check the terminal for the exact command, then restart."
        )

    # — Phone number pairing (say your number instead of scanning QR)
    import re as _re
    _phone_match = _re.search(r'\b(\+?[\d\s\-]{7,15})\b', text)
    if _phone_match and not any(w in text.lower() for w in ("done", "cancel", "scanned")):
        phone_raw = _re.sub(r'[\s\-]', '', _phone_match.group(1)).lstrip('+')
        _digits_only = _re.sub(r'[^\d]', '', _phone_match.group(1))
        if phone_raw.isdigit() and 7 <= len(phone_raw) <= 15 and len(_digits_only) >= 7:
            try:
                code = messaging_bridge.pair_phone(service, phone_raw)
                return (
                    f"Enter this code in {service.title()} → Settings → Linked Devices "
                    f"→ Link with Phone Number: {code}"
                )
            except Exception as _pe:
                logger.warning(f"[DEVICE_AUTH] PairPhone failed: {_pe}")

    if messaging_bridge.is_connected(service):
        _act.pending_device_auth.clear()
        if original_goal:
            from .. import code_executor
            retry_result = await code_executor.execute_code_task(
                goal=original_goal,
                llm_func=_llm_text,
            )
            return f"{service.title()} is connected! {retry_result}"
        return f"{service.title()} is connected and ready!"

    text_low = text.strip().lower()

    is_done = any(w in text_low for w in (
        "done", "scanned", "linked", "connected", "ready", "did it",
        "ok", "okay", "all set", "finished", "yes", "yeah", "yep",
    ))
    is_cancel = any(w in text_low for w in (
        "cancel", "stop", "never mind", "forget it", "abort", "no", "nope",
    ))

    if is_done:
        import asyncio
        for _ in range(10):
            await asyncio.sleep(0.5)
            if messaging_bridge.is_connected(service):
                _act.pending_device_auth.clear()
                if original_goal:
                    from .. import code_executor
                    retry_result = await code_executor.execute_code_task(
                        goal=original_goal,
                        llm_func=_llm_text,
                    )
                    return f"{service.title()} is connected! {retry_result}"
                return f"{service.title()} is connected and ready!"

        return (
            f"I don't see the connection yet. Make sure you scanned the QR code "
            f"in {service.title()} under Settings, Linked Devices, Link a Device. "
            f"Say 'done' when you've scanned it, or 'cancel' to abort."
        )

    if is_cancel:
        _act.pending_device_auth.clear()
        return f"Alright, I'll skip the {service.title()} setup for now. Just ask me again when you're ready."

    if messaging_bridge.is_connected(service):
        _act.pending_device_auth.clear()
        if original_goal:
            from .. import code_executor
            retry_result = await code_executor.execute_code_task(
                goal=original_goal,
                llm_func=_llm_text,
            )
            return f"{service.title()} just connected! {retry_result}"
        return f"{service.title()} connected!"

    _act.pending_device_auth.clear()
    return None


async def handle_pending_oauth_setup(text: str, bridge=None) -> str | None:
    """
    Multi-step OAuth setup handler.
    Returns a response string if this pending state is active, None otherwise.
    """
    import assistant.actions as _act

    if not _act.pending_oauth_setup.active:
        return None

    from .. import credentials, oauth_helper
    from .. import llm as llm_module
    from ..io.audio import tts

    service  = _act.pending_oauth_setup.payload["service"]
    step     = _act.pending_oauth_setup.payload["step"]
    text_low = text.strip().lower()

    # — Step: has_app
    if step == "has_app":
        existing_cid = credentials.get_credential(service, "client_id")
        existing_cs  = credentials.get_credential(service, "client_secret")
        if existing_cid and existing_cs:
            auth_url     = _act.pending_oauth_setup.payload["auth_url"]
            scopes       = _act.pending_oauth_setup.payload["scopes"]
            redirect_uri = _act.pending_oauth_setup.payload["redirect_uri"]

            from ..service_registry import OAUTH_AUTH_EXTRAS
            extra_params = OAUTH_AUTH_EXTRAS.get(service)

            setup_url = oauth_helper.get_setup_url(service, auth_url, scopes, redirect_uri,
                                                    extra_params=extra_params)
            if setup_url:
                import webbrowser
                webbrowser.open(setup_url)
                _act.pending_oauth_setup.payload["step"] = "auth_code"
                return (
                    f"Your {service.title()} app credentials are already saved. "
                    f"I've opened the authorization page — log in and approve access, "
                    f"then paste the 'code' parameter from the redirect URL."
                )

        is_yes = any(w in text_low for w in ("yes", "yeah", "yep", "yup", "i do", "already", "have"))
        is_no  = any(w in text_low for w in ("no", "nope", "don't", "dont", "haven't", "nah"))

        if is_yes:
            _act.pending_oauth_setup.payload["step"] = "client_id"
            return f"Great! Please paste your {service.title()} app's client ID."

        elif is_no:
            from ..service_registry import DEVELOPER_URLS
            dev_url = DEVELOPER_URLS.get(service, "")
            if dev_url:
                import webbrowser
                webbrowser.open(dev_url)
            _act.pending_oauth_setup.payload["step"] = "client_id"
            return (
                f"I've opened the {service.title()} developer dashboard. "
                f"Create a new app, set the redirect URI to http://127.0.0.1:8888/callback, "
                f"then paste your client ID here."
            )
        else:
            return f"Sorry, do you already have a {service.title()} developer app? Please say yes or no."

    # — Step: client_id
    elif step == "client_id":
        client_id = text.strip()
        if len(client_id) < 8:
            return "That doesn't look like a valid client ID. Please paste it again."
        credentials.set_credential(service, "client_id", client_id)
        _act.pending_oauth_setup.payload["step"] = "client_secret"
        return f"Got it. Now paste your {service.title()} app's client secret."

    # — Step: client_secret
    elif step == "client_secret":
        client_secret = text.strip()
        if len(client_secret) < 8:
            return "That doesn't look like a valid client secret. Please paste it again."
        credentials.set_credential(service, "client_secret", client_secret)

        auth_url     = _act.pending_oauth_setup.payload["auth_url"]
        scopes       = _act.pending_oauth_setup.payload["scopes"]
        redirect_uri = _act.pending_oauth_setup.payload["redirect_uri"]

        from ..service_registry import OAUTH_AUTH_EXTRAS
        extra_params = OAUTH_AUTH_EXTRAS.get(service)

        setup_url = oauth_helper.get_setup_url(service, auth_url, scopes, redirect_uri,
                                                extra_params=extra_params)
        if setup_url:
            import webbrowser
            webbrowser.open(setup_url)
            _act.pending_oauth_setup.payload["step"] = "auth_code"
            return (
                "I've opened the authorization page in your browser. "
                "Log in and approve access. After approving, your browser will redirect "
                "to a localhost URL. Copy the value of the 'code' parameter from that URL "
                "and paste it here."
            )
        else:
            _act.pending_oauth_setup.clear()
            return "Something went wrong building the authorization URL. Please try again."

    # — Step: auth_code
    elif step == "auth_code":
        auth_code = text.strip()
        if "code=" in auth_code:
            import urllib.parse
            try:
                parsed_url = urllib.parse.urlparse(auth_code)
                params = urllib.parse.parse_qs(parsed_url.query)
                auth_code = params.get("code", [auth_code])[0]
            except Exception:
                pass

        token_url    = _act.pending_oauth_setup.payload["token_url"]
        redirect_uri = _act.pending_oauth_setup.payload["redirect_uri"]
        original_goal = _act.pending_oauth_setup.payload["original_goal"]

        success, error_detail = oauth_helper.exchange_code_for_tokens(
            service, auth_code, token_url, redirect_uri,
            scopes=_act.pending_oauth_setup.payload.get("scopes", ""),
        )

        if success:
            _act.pending_oauth_setup.clear()
            from .. import code_executor
            retry_result = await code_executor.execute_code_task(
                goal=original_goal,
                llm_func=_llm_text,
            )
            if retry_result and ("NEEDS_OAUTH" in retry_result or "__NEEDS_OAUTH__" in retry_result):
                return f"[happy] {service.title()} is all set! Just say that again and I'll run it now."
            return f"[happy] {service.title()} is all set! {retry_result}"
        else:
            if error_detail == "redirect_uri_mismatch":
                _act.pending_oauth_setup.payload["step"] = "auth_code"
                return (
                    f"The redirect URI doesn't match your {service.title()} app settings. "
                    f"Go to your developer console and add {redirect_uri} "
                    f"to the Authorized redirect URIs, then paste a new code."
                )
            elif error_detail == "invalid_client":
                _act.pending_oauth_setup.clear()
                from .. import credentials
                credentials.delete_credential(service)
                return (
                    f"The client ID or secret is invalid. I've cleared the saved credentials. "
                    f"Double-check them in your {service.title()} developer console, "
                    f"then say '{original_goal}' to start fresh."
                )
            else:
                _act.pending_oauth_setup.clear()
                return (
                    f"The authorization code didn't work. "
                    f"This sometimes happens if the code expired — they're only valid for a few minutes. "
                    f"Just ask me to {original_goal} again to restart the setup."
                )

    return None


async def handle_pending_messaging_disambig(text: str, bridge=None) -> str | None:
    """
    Handle disambiguation after a 'multiple contacts match' error.

    The user said "Sarvesh" and got multiple matches. Now they say "Sarvesh Koli".
    We retry the original send with the clarified name.

    Generic — works for any messaging service.
    """
    import assistant.actions as _act

    if not _act.pending_messaging_disambig.active:
        return None

    text_stripped = text.strip()

    if len(text_stripped.split()) > 5:
        _act.pending_messaging_disambig.clear()
        return None

    text_low = text_stripped.lower()
    if any(w in text_low for w in ("cancel", "never mind", "forget it", "stop", "abort")):
        _act.pending_messaging_disambig.clear()
        return personality_say("msg_cancelled")

    service = _act.pending_messaging_disambig.payload["service"]
    original_text = _act.pending_messaging_disambig.payload["text"]
    clarified_name = text_stripped
    _act.pending_messaging_disambig.clear()

    logger.info(f"[MESSAGING] Disambiguation: retrying with '{clarified_name}'")

    from ..io import messaging_bridge
    send_result = messaging_bridge.execute(service, "send_message", {
        "contact_name": clarified_name,
        "text": original_text,
    })

    if send_result.get("ok"):
        inner = send_result["result"]
        if isinstance(inner, dict) and inner.get("needs_confirmation"):
            _act.pending_messaging_send.set(inner)
            resolved_name = inner.get("resolved_name", inner.get("phone", "someone"))
            text_preview = inner.get("text", "")
            if len(text_preview) > 50:
                text_preview = text_preview[:47] + "..."
            return (
                personality_say("msg_confirm") + f" (Send '{text_preview}' to {resolved_name}?)"
            )
        else:
            return str(inner)
    else:
        error_msg = send_result.get("error", "Unknown error")
        return error_msg


async def handle_pending_messaging_send(text: str, bridge=None) -> str | None:
    """
    Handle user yes/no response to a pending messaging send confirmation.
    Returns response string if handled, None if not applicable.
    """
    import assistant.actions as _act

    if _act.pending_messaging_send.payload is None:
        return None

    lowered = text.strip().lower()
    is_yes = any(w in lowered for w in ("yes", "yeah", "yep", "sure", "ok", "okay", "send", "do it"))
    is_no = any(w in lowered for w in ("no", "nope", "nah", "cancel", "don't", "dont", "never mind", "forget it"))

    if is_yes:
        from ..io import messaging_bridge

        service = _act.pending_messaging_send.payload.get("service", "")
        phone = _act.pending_messaging_send.payload.get("phone", "")
        msg_text = _act.pending_messaging_send.payload.get("text", "")
        resolved_name = _act.pending_messaging_send.payload.get("resolved_name", phone)
        _act.pending_messaging_send.clear()

        logger.info(f"[MESSAGING] Confirmed — sending to {phone} via {service}")
        result = messaging_bridge.execute(service, "send_message", {
            "phone": phone,
            "text": msg_text,
            "_confirmed": "true",
        })

        if result.get("ok"):
            return personality_say("msg_sent", name=resolved_name)
        else:
            error = result.get("error", "Unknown error")
            return personality_say("msg_send_failed", name=resolved_name, error=error)

    if is_no:
        _act.pending_messaging_send.clear()
        return personality_say("msg_cancelled")

    if len(text.strip().split()) > 3:
        _act.pending_messaging_send.clear()
        return None

    return personality_say("msg_read_prompt")


async def handle_pending_incoming_message(text: str, bridge=None) -> str | None:
    """
    Handle user response after an incoming message notification.

    Triggers on: "read it", "what did they say", "read message", "tell me", etc.
    - 1-3 messages: read content directly
    - 4+ messages: LLM summarize via Cerebras synthesis

    Returns response string if handled, None if not applicable.
    """
    import assistant.actions as _act

    if not _act.pending_incoming_messages.active:
        return None

    text_low = text.strip().lower()

    read_triggers = (
        "read it", "read them", "read that", "read message", "read messages",
        "read the message", "read the messages",
        "what did they say", "what do they say",
        "what does it say", "what's it say", "what's the message",
        "what did he say", "what did she say",
        "tell me", "go ahead", "what is it",
        "yes", "yeah", "sure", "yep",
    )
    is_read = any(trigger in text_low for trigger in read_triggers)

    if not is_read:
        _msg_words = {"message", "messages", "msg", "msgs", "text", "texts"}
        _query_words = {"say", "said", "read", "what", "tell", "show", "hear"}
        _input_words = set(text_low.split())
        if _msg_words & _input_words and _query_words & _input_words:
            is_read = True

    if not is_read:
        cancel_triggers = ("no", "nope", "ignore", "skip", "cancel", "never mind", "leave it", "not now")
        is_cancel = any(trigger in text_low for trigger in cancel_triggers)
        if is_cancel:
            _act.pending_incoming_messages.clear()
            return personality_say("msg_ignore")

        if len(text.strip().split()) > 5:
            _act.pending_incoming_messages.clear()
            return None

        return personality_say("msg_read_prompt")

    # — Read or summarize the messages
    from .. import config as _config

    payload = _act.pending_incoming_messages.payload
    if not payload:
        _act.pending_incoming_messages.clear()
        return None
    batch = payload[-1]
    sender_name = batch.get("sender_name", "someone")
    messages = batch.get("messages", [])
    _act.pending_incoming_messages.clear()

    if not messages:
        return personality_say("msg_empty")

    threshold = getattr(_config, "INCOMING_READ_THRESHOLD", 3)

    if len(messages) <= threshold:
        if len(messages) == 1:
            return f"{sender_name} said: {messages[0]['text']}"
        else:
            parts = []
            for i, m in enumerate(messages, 1):
                parts.append(f"Message {i}: {m['text']}")
            joined = ". ".join(parts)
            return f"{sender_name} sent {len(messages)} messages. {joined}"
    else:
        from ..llm.contracts import ask_for_synthesis

        all_texts = "\n".join(f"- {m['text']}" for m in messages)
        summary_prompt = (
            f"The following are {len(messages)} messages from {sender_name}. "
            f"Summarize what they're saying in 1-2 natural sentences, as if you're "
            f"telling a friend what the messages are about:\n\n{all_texts}"
        )
        try:
            summary = await ask_for_synthesis(
                summary_prompt,
                system_prompt="You are a helpful assistant. Summarize the messages concisely in plain language.",
            )
            return f"{sender_name} sent {len(messages)} messages. {summary}"
        except Exception as e:
            logger.error(f"[INCOMING] Summary failed: {e}")
            parts = [f"Message {i+1}: {m['text']}" for i, m in enumerate(messages[:3])]
            joined = ". ".join(parts)
            return f"{sender_name} sent {len(messages)} messages. Here are the first few: {joined}"


async def handle_pending_knowledge_approval(text: str, bridge=None) -> str | None:
    """
    Handle user yes/no response to a knowledge proposal.
    The proposal was already spoken as part of the previous task response.
    Returns response string if handling approval, None if not active.
    """
    import assistant.actions as _act

    if _act.pending_knowledge_approval.payload is None:
        from ..code_executor import pop_pending_knowledge
        entry = pop_pending_knowledge()
        if entry is None:
            return None
        _act.pending_knowledge_approval.set(entry)

    text_low = text.strip().lower()

    is_yes = any(w in text_low for w in (
        "yes", "yeah", "yep", "sure", "save", "ok", "go ahead",
        "do it", "confirm", "approve", "remember",
    ))
    is_no = any(w in text_low for w in (
        "no", "nope", "skip", "don't", "dont", "nah", "cancel",
        "forget", "never mind", "not now",
    ))

    if is_yes:
        from .. import knowledge
        entry = _act.pending_knowledge_approval.payload
        _act.pending_knowledge_approval.clear()
        added = knowledge.add_works_entry(
            entry["service"], entry["slug"],
            entry["pattern"], entry["reason"],
        )
        if added:
            return f"Got it, I'll remember that for {entry['service'].title()} tasks."
        else:
            return "I already know that one, no need to save it again."

    elif is_no:
        _act.pending_knowledge_approval.clear()
        return "Alright, I won't save that."

    else:
        _act.pending_knowledge_approval.clear()
        return None
