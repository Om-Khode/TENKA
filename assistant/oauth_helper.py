import logging
import requests
import urllib.parse
from datetime import datetime, timedelta, timezone
from . import credentials

logger = logging.getLogger("oauth_helper")

def has_client_credentials(service: str) -> bool:
    """Returns True if both 'client_id' and 'client_secret' exist in credentials for this service."""
    client_id = credentials.get_credential(service, "client_id")
    client_secret = credentials.get_credential(service, "client_secret")
    
    logger.debug(f"Checking client credentials for {service}")
    return bool(client_id and client_secret)

def has_token(service: str) -> bool:
    """Returns True if 'access_token' exists in credentials for this service."""
    access_token = credentials.get_credential(service, "access_token")
    
    logger.debug(f"Checking token for {service}")
    return bool(access_token)

def is_token_expired(service: str) -> bool:
    """Returns True if expiry is missing OR if current UTC time >= expiry minus 60 seconds buffer."""
    token_expiry_str = credentials.get_credential(service, "token_expiry")
    if not token_expiry_str:
        logger.warning(f"No token expiry found for {service}")
        return True
    
    try:
        # Parse ISO8601 UTC string (e.g. 2023-10-25T14:30:00+00:00)
        expiry = datetime.fromisoformat(token_expiry_str.replace('Z', '+00:00'))
        if expiry.tzinfo is None:
             expiry = expiry.replace(tzinfo=timezone.utc)
             
        now = datetime.now(timezone.utc)
        
        # Buffer of 60 seconds
        is_expired = now >= (expiry - timedelta(seconds=60))
        if is_expired:
            logger.warning(f"Token expired for {service}")
        else:
            logger.debug(f"Token valid for {service}")
            
        return is_expired
    except ValueError as e:
        logger.warning(f"Failed to parse token expiry for {service}: {e}")
        return True

def refresh_token(service: str, token_url: str, client_id: str, client_secret: str) -> bool:
    """
    Refreshes the access token. 
    Returns True on success, False on any failure.
    """
    refresh_token_val = credentials.get_credential(service, "refresh_token")
    
    if not refresh_token_val:
        logger.warning(f"Missing refresh_token to refresh token for {service}")
        return False
        
    logger.info(f"Attempting token refresh for {service}")
    
    try:
        response = requests.post(
            token_url,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token_val,
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        new_access_token = data.get("access_token")
        new_refresh_token = data.get("refresh_token") # May be None if provider doesn't rotate it
        # Default to 3600 seconds as fallback
        expires_in = data.get("expires_in", 3600)
        
        if not new_access_token:
            logger.warning(f"Refresh response missing access_token for {service}")
            return False
            
        _save_tokens(service, new_access_token, new_refresh_token, expires_in)
        # Note: granted_scopes are NOT changed on refresh — they persist from
        # the original authorization. OAuth providers return the same scope set
        # on refresh unless the user re-authorizes with different scopes.
        logger.info(f"Successfully refreshed token for {service}")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to refresh token for {service}: {e}")
        return False

def get_env_vars(service: str, token_url: str, env_var_map: dict[str, str]) -> dict[str, str] | None:
    """
    Builds a dict of env var name -> decrypted value for all keys in env_var_map.
    """
    if not has_token(service):
        return None
        
    if is_token_expired(service):
        client_id = credentials.get_credential(service, "client_id")
        client_secret = credentials.get_credential(service, "client_secret")
        
        if not client_id or not client_secret:
            logger.warning(f"Missing client credentials needed to auto-refresh token for {service}")
            return None
            
        if not refresh_token(service, token_url, client_id, client_secret):
            return None
            
    env_vars = {}
    logger.info(f"Building env vars for {service}")
    
    for cred_key, env_name in env_var_map.items():
        value = credentials.get_credential(service, cred_key)
        if value:
            env_vars[env_name] = value
        else:
            logger.warning(f"Missing key '{cred_key}' in credentials for {service}")
            
    return env_vars

def get_setup_url(service: str, auth_url: str, scopes: str, redirect_uri: str,
                  extra_params: dict[str, str] | None = None) -> str:
    """
    Build the authorization URL to show the user during one-time OAuth setup.
    Reads client_id from credentials (user has already saved it by this point).

    Args:
        extra_params: Optional provider-specific params to include in the URL.
                      e.g. {"access_type": "offline", "prompt": "consent"} for Google.

    Returns the full URL string, or empty string if client_id is missing.
    """
    client_id = credentials.get_credential(service, "client_id")
    if not client_id:
        return ""
    
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scopes,
    }
    if extra_params:
        params.update(extra_params)
    return auth_url + "?" + urllib.parse.urlencode(params)

def exchange_code_for_tokens(
    service: str,
    code: str,
    token_url: str,
    redirect_uri: str,
    scopes: str = "",
) -> tuple[bool, str]:
    """
    Exchange an authorization code for access + refresh tokens.
    Reads client_id and client_secret from credentials.
    Saves tokens via _save_tokens().
    Saves granted_scopes to credentials so future actions can check
    whether re-authorization is needed for broader scope sets.
    Returns (True, "") on success, (False, error_detail) on failure.
    Never raises.
    """
    client_id = credentials.get_credential(service, "client_id")
    client_secret = credentials.get_credential(service, "client_secret")

    if not client_id or not client_secret:
        logger.warning(f"[OAUTH] Missing client credentials for {service} during code exchange")
        return False, "missing_credentials"

    try:
        response = requests.post(
            token_url,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=10,
        )

        if response.status_code != 200:
            body = {}
            try:
                body = response.json()
            except Exception:
                pass
            error = body.get("error", "unknown")
            desc = body.get("error_description", response.text[:200])
            logger.warning(
                f"[OAUTH] Code exchange failed for {service}: "
                f"{response.status_code} {error}: {desc}"
            )
            return False, error

        data = response.json()

        access_token = data.get("access_token")
        refresh_token_val = data.get("refresh_token")
        expires_in = data.get("expires_in", 3600)

        if not access_token:
            logger.warning(f"[OAUTH] No access_token in response for {service}")
            return False, "no_access_token"

        _save_tokens(service, access_token, refresh_token_val, expires_in)

        # ── Save granted scopes ───────────────────────────────────────────
        # Use the scopes from the token response if the provider returns them
        # (e.g., "scope" field). Fall back to the scopes we requested in the
        # authorization URL. This lets code_executor detect scope gaps later.
        granted = data.get("scope", "") or scopes
        if granted:
            # Normalize: some providers return comma-separated, some space-separated
            normalized = granted.replace(",", " ").strip()
            credentials.set_credential(service, "granted_scopes", normalized)
            logger.info(f"[OAUTH] Saved granted scopes for {service}: {normalized}")

        logger.info(f"[OAUTH] Token exchange successful for {service}")
        return True, ""

    except Exception as e:
        logger.warning(f"[OAUTH] Code exchange failed for {service}: {e}")
        return False, str(e)

def needs_reauth(service: str, required_scopes: str) -> bool:
    """
    Check if the existing token's granted scopes are insufficient for the
    required scopes. Generic — works for any OAuth service.

    Args:
        service:         Service name (e.g. "spotify", "gmail")
        required_scopes: Space-separated string of scopes needed for the current action.

    Returns:
        True if re-authorization is needed, False if existing scopes are sufficient.
        Also returns True if no granted_scopes are stored (conservative — assume insufficient).
    """
    if not required_scopes:
        return False

    stored_str = credentials.get_credential(service, "granted_scopes") or ""
    if not stored_str:
        # No record of what scopes were granted — assume insufficient
        logger.info(f"[OAUTH] No granted_scopes stored for {service} — assuming reauth needed")
        return True

    stored_set = set(stored_str.split())
    required_set = set(required_scopes.split())
    missing = required_set - stored_set

    if missing:
        logger.info(
            f"[OAUTH] Scope gap for {service}: "
            f"granted={stored_set}, required={required_set}, missing={missing}"
        )
        return True

    return False

def clear_token(service: str) -> None:
    """
    Clear the access token and expiry for a service, forcing re-authorization
    on the next attempt. Does NOT clear client_id, client_secret, or
    granted_scopes — those are preserved so the re-auth flow can merge scopes.
    """
    path = credentials._service_path(service)
    if not path.exists():
        return

    try:
        data = credentials._load_raw(path)
        for key_to_remove in ("access_token", "refresh_token", "token_expiry"):
            data.pop(key_to_remove, None)
        credentials._save_raw(path, data)
        logger.info(f"[OAUTH] Cleared token for {service} (kept client creds + granted_scopes)")
    except Exception as e:
        logger.warning(f"[OAUTH] Failed to clear token for {service}: {e}")

def _save_tokens(service: str, access_token: str, refresh_token: str | None = None, expires_in: int = 3600):
    """
    Saves access_token (and optionally refresh_token) to credentials.
    Computes ISO8601 UTC expiry string and saves it too.
    """
    credentials.set_credential(service, "access_token", access_token)
    
    if refresh_token is not None:
        credentials.set_credential(service, "refresh_token", refresh_token)
        
    expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
    # Ensure it's explicitly ISO8601 string
    expiry_str = expiry.isoformat()
    credentials.set_credential(service, "token_expiry", expiry_str)