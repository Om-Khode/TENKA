"""
actions/backup_pending.py — Pending-state handlers for backup onboarding.

Two-stage flow, both driven from actions/backup.py's "enable" action:
  1. pending_backup_confirm_phrase — show the recovery phrase once, wait
     for the user to confirm they've saved it, then cache the derived key.
  2. pending_backup_oauth — walk through connecting Google Drive, using
     oauth_helper.py's primitives directly (not the code_executor OAuth
     retry flow — see Task 4's plan note for why).

Two standalone phrase-entry flows, for a phrase the user already has:
  3. pending_backup_unlock_phrase — re-arm this session's key after a
     restart ("unlock backup"), without generating a new phrase.
  4. pending_backup_restore_phrase — pull the latest backup down and
     extract it over the current data ("restore backup").

No handler here ever puts the recovery phrase in its return value: that
string is spoken, logged, saved to the conversations table, and replayed
into later LLM prompts. See actions/backup.py::_show_phrase_privately.
"""
import logging
import re

logger = logging.getLogger("backup_pending")

_CANCEL_RE = re.compile(r"\b(cancel|never\s*mind|nevermind|forget it|stop|abort)\b")

# Wins over positive-keyword substring matches — catches negated phrasing like
# "not done yet" or "don't have one" that would otherwise false-positive on
# a bare "done" / "have one" substring match. Word-boundary regex (not
# .split()) so "no," with trailing punctuation still matches "no" as a word.
_NEGATION_RE = re.compile(r"\b(no|not|don'?t|dont|nope|haven'?t|havent|nah|yet)\b")


async def handle_pending_backup_confirm_phrase(text: str, bridge=None) -> str | None:
    import assistant.actions as _act

    if not _act.pending_backup_confirm_phrase.active:
        return None

    text_low = text.strip().lower()
    has_negation = bool(_NEGATION_RE.search(text_low))
    is_yes = (not has_negation) and any(
        w in text_low for w in ("yes", "saved", "done", "got it", "wrote it", "ok", "okay")
    )

    if not is_yes:
        _act.pending_backup_confirm_phrase.touch()
        return (
            "Take your time — write the phrase down somewhere safe, then say "
            "'saved it' when you're ready to continue."
        )

    # Only now is the key cached: until the user confirms, an abandoned or
    # timed-out flow must leave the session locked rather than armed with a
    # phrase nobody wrote down (which later backups would upload under).
    phrase = _act.pending_backup_confirm_phrase.payload["phrase"]
    from ..io.backup import crypto, orchestrator
    orchestrator.set_unlocked_key(crypto.derive_key(phrase))

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

    # Every step below either advances the flow or re-prompts, and each one
    # can take the user minutes (creating a Cloud app, enabling the Drive
    # API, approving a consent screen). Refresh the timeout on entry so the
    # clock measures user silence, not total flow duration.
    _act.pending_backup_oauth.touch()

    if step == "has_app":
        has_negation = bool(_NEGATION_RE.search(text_low))
        is_yes = (not has_negation) and any(
            w in text_low for w in ("yes", "yeah", "yep", "have one", "already")
        )
        is_no = has_negation

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
        redirect_uri = payload["redirect_uri"]
        success, error_detail = oauth_helper.exchange_code_for_tokens(
            SERVICE_NAME, auth_code, payload["token_url"], redirect_uri,
            scopes=payload.get("scopes", ""),
        )
        if success:
            _act.pending_backup_oauth.clear()
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

        # Raw error codes are never spoken (CLAUDE.md's TTS rule) — they go
        # to the log, the user gets the fix. Same branching as
        # pending_handlers.handle_pending_oauth_setup, with 'enable backup'
        # as the retry entry point instead of an original_goal.
        logger.warning(f"[BACKUP] OAuth token exchange failed: {error_detail}")
        if error_detail == "redirect_uri_mismatch":
            _act.pending_backup_oauth.payload["step"] = "auth_code"
            return (
                "That redirect URI doesn't match your Google app settings. "
                f"Add {redirect_uri} to the Authorized redirect URIs, "
                "then paste a new code."
            )
        _act.pending_backup_oauth.clear()
        if error_detail == "invalid_client":
            credentials.delete_credential(SERVICE_NAME)
            return (
                "That client ID or secret is invalid, so I've cleared them. "
                "Check them in the Google Cloud console, then say "
                "'enable backup' to start fresh."
            )
        return (
            "That authorization code didn't work — they expire after a few "
            "minutes. Say 'enable backup' to try again."
        )

    return None


# ─── Phrase entry — unlock / restore ────────────────────────────────────────


def _normalize_phrase(text: str) -> str:
    """Canonicalise typed or dictated phrase input.

    generate_recovery_phrase() emits lowercase words separated by single
    spaces, so folding case, dropping punctuation STT loves to add, and
    collapsing whitespace maps spoken input back onto the exact string the
    key was derived from. Both validation and derivation use this same
    normalised form — they must never diverge.
    """
    cleaned = re.sub(r"[^a-z\s]", " ", text.lower())
    return " ".join(cleaned.split())


async def handle_pending_backup_unlock_phrase(text: str, bridge=None) -> str | None:
    """Re-arm the in-memory backup key from a phrase the user already has."""
    import assistant.actions as _act

    if not _act.pending_backup_unlock_phrase.active:
        return None

    if _CANCEL_RE.search(text.strip().lower()):
        _act.pending_backup_unlock_phrase.clear()
        return "Okay, backup stays locked for now."

    from ..io.backup import crypto, orchestrator

    phrase = _normalize_phrase(text)
    if not crypto.is_valid_recovery_phrase(phrase):
        # Deliberately not cleared: a mistyped or misheard word should cost
        # a retry, not the whole flow. The state expires on its own.
        _act.pending_backup_unlock_phrase.touch()
        return (
            "That doesn't look like a valid recovery phrase. It's 12 words. "
            "Try again, or say 'cancel'."
        )

    orchestrator.set_unlocked_key(crypto.derive_key(phrase))
    _act.pending_backup_unlock_phrase.clear()
    return "Backup is unlocked for this session. Say 'back up now' whenever you like."


async def handle_pending_backup_restore_phrase(text: str, bridge=None) -> str | None:
    """Download, decrypt, and extract the latest backup over local data."""
    import assistant.actions as _act

    if not _act.pending_backup_restore_phrase.active:
        return None

    if _CANCEL_RE.search(text.strip().lower()):
        _act.pending_backup_restore_phrase.clear()
        return "Okay, nothing was restored."

    from ..io.backup import crypto, orchestrator
    from ..io.backup.provider import BackupProviderError

    phrase = _normalize_phrase(text)
    if not crypto.is_valid_recovery_phrase(phrase):
        _act.pending_backup_restore_phrase.touch()
        return (
            "That doesn't look like a valid recovery phrase. It's 12 words. "
            "Try again, or say 'cancel'."
        )

    _act.pending_backup_restore_phrase.clear()

    provider_name = "google_drive"
    from ..storage.db import get_db
    from ..storage.repos.settings import SettingsRepo
    db = get_db()
    if db is not None:
        provider_name = SettingsRepo(db).get("backup_provider", "google_drive")

    try:
        orchestrator.run_restore(phrase, provider_name)
    except (BackupProviderError, RuntimeError, OSError) as e:
        # Never echo the exception: it can carry URLs, paths, and status
        # codes. Log it, speak a short generic line (CLAUDE.md's TTS rule).
        logger.error(f"[BACKUP] restore failed: {e}")
        return "Restore failed — check the logs for details."

    return "Restore finished. Restart me so I load the restored data."
