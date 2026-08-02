"""
actions/backup_pending.py — Pending-state handlers for backup onboarding.

Two-stage flow, both driven from actions/backup.py's "enable" action:
  1. pending_backup_confirm_phrase — show the recovery phrase once, wait
     for the user to confirm they've saved it.
  2. pending_backup_oauth — walk through connecting Google Drive, using
     oauth_helper.py's primitives directly (not the code_executor OAuth
     retry flow — see Task 4's plan note for why).
"""
import logging

logger = logging.getLogger("backup_pending")


async def handle_pending_backup_confirm_phrase(text: str, bridge=None) -> str | None:
    import assistant.actions as _act

    if not _act.pending_backup_confirm_phrase.active:
        return None

    text_low = text.strip().lower()
    is_yes = any(w in text_low for w in ("yes", "saved", "done", "got it", "wrote it", "ok", "okay"))

    if not is_yes:
        return (
            "Take your time — write the phrase down somewhere safe, then say "
            "'saved it' when you're ready to continue."
        )

    _act.pending_backup_confirm_phrase.clear()

    from ..io.backup.google_drive import AUTH_URL, TOKEN_URL, SCOPES, REDIRECT_URI

    _act.pending_backup_oauth.set({
        "step": "has_app",
        "auth_url": AUTH_URL,
        "token_url": TOKEN_URL,
        "scopes": SCOPES,
        "redirect_uri": REDIRECT_URI,
    })
    return (
        "Good. Now let's connect Google Drive. Do you already have a "
        "Google Cloud OAuth app set up?"
    )


async def handle_pending_backup_oauth(text: str, bridge=None) -> str | None:
    import assistant.actions as _act

    if not _act.pending_backup_oauth.active:
        return None

    from .. import credentials, oauth_helper
    from ..io.backup.google_drive import SERVICE_NAME

    step = _act.pending_backup_oauth.payload["step"]
    text_low = text.strip().lower()

    if step == "has_app":
        is_yes = any(w in text_low for w in ("yes", "yeah", "yep", "have one", "already"))
        is_no = any(w in text_low for w in ("no", "nope", "don't", "dont", "haven't", "nah"))

        if is_yes:
            _act.pending_backup_oauth.payload["step"] = "client_id"
            return "Great — please paste your Google OAuth app's client ID."
        elif is_no:
            import webbrowser
            webbrowser.open("https://console.cloud.google.com/apis/credentials")
            _act.pending_backup_oauth.payload["step"] = "client_id"
            return (
                "I've opened the Google Cloud credentials page. Create an OAuth "
                "client, add the Drive API with the drive.appdata scope, set the "
                f"redirect URI to {_act.pending_backup_oauth.payload['redirect_uri']}, "
                "then paste the client ID here."
            )
        return "Just need a yes or no — do you already have a Google OAuth app?"

    if step == "client_id":
        client_id = text.strip()
        if len(client_id) < 8:
            return "That doesn't look like a valid client ID. Please paste it again."
        credentials.set_credential(SERVICE_NAME, "client_id", client_id)
        _act.pending_backup_oauth.payload["step"] = "client_secret"
        return "Got it. Now paste your Google OAuth app's client secret."

    if step == "client_secret":
        client_secret = text.strip()
        if len(client_secret) < 8:
            return "That doesn't look like a valid client secret. Please paste it again."
        credentials.set_credential(SERVICE_NAME, "client_secret", client_secret)

        payload = _act.pending_backup_oauth.payload
        setup_url = oauth_helper.get_setup_url(
            SERVICE_NAME, payload["auth_url"], payload["scopes"], payload["redirect_uri"],
            extra_params={"access_type": "offline", "prompt": "consent"},
        )
        if not setup_url:
            _act.pending_backup_oauth.clear()
            return "Something went wrong building the authorization URL. Please try again."

        import webbrowser
        webbrowser.open(setup_url)
        _act.pending_backup_oauth.payload["step"] = "auth_code"
        return (
            "I've opened the authorization page. Log in, approve access, then "
            "your browser will redirect to a URL with a 'code=' parameter — "
            "copy that code and paste it here."
        )

    if step == "auth_code":
        auth_code = text.strip()
        payload = _act.pending_backup_oauth.payload
        success, error_detail = oauth_helper.exchange_code_for_tokens(
            SERVICE_NAME, auth_code, payload["token_url"], payload["redirect_uri"],
            scopes=payload.get("scopes", ""),
        )
        _act.pending_backup_oauth.clear()
        if success:
            from ..storage.db import get_db
            from ..storage.repos.settings import SettingsRepo
            db = get_db()
            if db is not None:
                SettingsRepo(db).set("backup_enabled", True, source="backup_onboarding")
                SettingsRepo(db).set("backup_provider", "google_drive", source="backup_onboarding")
            return (
                "[happy] Google Drive is connected and backup is enabled. "
                "Say 'back up now' any time, or I'll back up automatically."
            )
        return (
            f"The authorization code didn't work ({error_detail}). "
            "Say 'enable backup' to try again."
        )

    return None
