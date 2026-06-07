"""
credentials.py — Encrypted per-service credential storage for TENKA.

Stores API keys and secrets as AES-encrypted JSON files.
One file per service: SANDBOX_DIR/credentials/{service_name}.json

Encryption key is derived from machine-specific data (Windows machine GUID,
falling back to username+hostname). Never hardcoded, never stored on disk.

Public API:
    has_credential(service)              -> bool
    get_credential(service, key)         -> str | None
    set_credential(service, key, value)  -> None
    delete_credential(service)           -> bool
    list_services()                      -> list[str]
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("credentials")

_SCHEMA_VERSION = 1

# ─── Storage Location ─────────────────────────────────────────────────────────

def _credentials_dir() -> Path:
    """Return the credentials directory, creating it if needed."""
    from . import config
    cred_dir = config.SANDBOX_DIR / "credentials"
    cred_dir.mkdir(parents=True, exist_ok=True)
    return cred_dir


def _service_path(service: str) -> Path:
    """Return the path for a service's credential file."""
    # Sanitize service name — alphanumeric + underscore + hyphen only
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in service.lower())
    return _credentials_dir() / f"{safe}.json"


# ─── Key Derivation ───────────────────────────────────────────────────────────

def _derive_key() -> bytes:
    """
    Derive a 32-byte AES key from machine-specific data.

    Primary:  Windows machine GUID from registry
              HKLM\\SOFTWARE\\Microsoft\\Cryptography\\MachineGuid
    Fallback: username + hostname
    
    The key is derived via SHA-256 so it's always exactly 32 bytes
    regardless of input length.
    """
    import hashlib

    machine_id = _get_machine_guid()

    # SHA-256 of machine_id gives a stable 32-byte key
    key = hashlib.sha256(machine_id.encode("utf-8")).digest()
    return key


def _get_machine_guid() -> str:
    """
    Read the Windows Machine GUID from the registry.
    Falls back to username+hostname if registry read fails.
    """
    try:
        import winreg
        reg_path = r"SOFTWARE\Microsoft\Cryptography"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path) as key:
            guid, _ = winreg.QueryValueEx(key, "MachineGuid")
            logger.debug("[CRED] Key derived from Windows Machine GUID")
            return str(guid)
    except Exception as e:
        logger.warning(f"[CRED] Registry read failed, falling back to username+hostname: {e}")
        import socket
        fallback = f"{os.environ.get('USERNAME', 'user')}@{socket.gethostname()}"
        logger.debug(f"[CRED] Key derived from fallback: {fallback}")
        return fallback


# ─── Encryption / Decryption ──────────────────────────────────────────────────

def _encrypt(data: str) -> bytes:
    """
    Encrypt a string using AES-GCM.
    Returns: nonce (12 bytes) + ciphertext + tag, base64-encoded.
    """
    import base64
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = _derive_key()
    nonce = os.urandom(12)  # 96-bit nonce, unique per encryption
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, data.encode("utf-8"), None)
    # Store as base64(nonce + ciphertext) for clean JSON embedding
    return base64.b64encode(nonce + ciphertext).decode("utf-8")


def _decrypt(token: str) -> str:
    """
    Decrypt a base64-encoded AES-GCM token.
    Returns the original plaintext string.
    Raises ValueError on decryption failure (wrong key or tampered data).
    """
    import base64
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = _derive_key()
    raw = base64.b64decode(token)
    nonce = raw[:12]
    ciphertext = raw[12:]
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return plaintext.decode("utf-8")


# ─── Public API ───────────────────────────────────────────────────────────────

def has_credential(service: str) -> bool:
    """
    Check if a credential file exists for the given service.
    Fast — no decryption performed.

    Args:
        service: Service name e.g. "spotify", "gmail", "openai"

    Returns:
        True if a credential file exists, False otherwise.
    """
    return _service_path(service).exists()


def _load_raw(path: Path) -> dict:
    """Load a credentials JSON file, handling versioned and legacy bare-dict formats."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "version" in raw:
            return raw.get("data", {})
        if isinstance(raw, dict):
            # Legacy bare dict — migrate on next write
            return raw
        return {}
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"[CRED] Failed to read {path}: {e}")
        return {}


def _save_raw(path: Path, data: dict) -> None:
    """Write a credentials dict in versioned envelope."""
    path.write_text(
        json.dumps({"version": _SCHEMA_VERSION, "data": data}, indent=2),
        encoding="utf-8",
    )


def get_credential(service: str, key: str) -> str | None:
    """
    Retrieve a single credential value for a service.

    Args:
        service: Service name e.g. "spotify"
        key:     Credential key e.g. "client_id", "api_key"

    Returns:
        The decrypted value string, or None if not found.
    """
    path = _service_path(service)
    if not path.exists():
        logger.debug(f"[CRED] No credentials file for '{service}'")
        return None

    try:
        data = _load_raw(path)
        token = data.get(key)
        if token is None:
            logger.debug(f"[CRED] Key '{key}' not found in '{service}' credentials")
            return None
        value = _decrypt(token)
        logger.info(f"[CRED] Retrieved '{key}' for service '{service}'")
        return value
    except Exception as e:
        logger.error(f"[CRED] Failed to read credential '{key}' for '{service}': {e}")
        return None


def set_credential(service: str, key: str, value: str) -> None:
    """
    Store a single credential value for a service, encrypted.
    Creates the credentials file if it doesn't exist.
    Updates the key if it already exists.

    Args:
        service: Service name e.g. "spotify"
        key:     Credential key e.g. "client_id"
        value:   Plaintext value to encrypt and store
    """
    path = _service_path(service)
    existing = _load_raw(path)
    existing[key] = _encrypt(value)
    _save_raw(path, existing)
    logger.info(f"[CRED] Saved '{key}' for service '{service}' at {path}")


def delete_credential(service: str) -> bool:
    """
    Delete all credentials for a service.

    Args:
        service: Service name to delete

    Returns:
        True if the file was deleted, False if it didn't exist.
    """
    path = _service_path(service)
    if not path.exists():
        logger.debug(f"[CRED] No credentials file for '{service}' to delete")
        return False

    try:
        path.unlink()
        logger.info(f"[CRED] Deleted credentials for '{service}'")
        return True
    except Exception as e:
        logger.error(f"[CRED] Failed to delete credentials for '{service}': {e}")
        return False


def list_services() -> list[str]:
    """
    List all services that have stored credentials.

    Returns:
        List of service name strings (derived from filenames).
    """
    cred_dir = _credentials_dir()
    services = [p.stem for p in cred_dir.glob("*.json")]
    logger.debug(f"[CRED] Found {len(services)} stored service(s): {services}")
    return services