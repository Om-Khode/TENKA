# assistant/io/api/vault.py
"""Device tokens and the per-installation instance secret.

The plaintext token exists in memory exactly once, at issue time, and is
returned to the caller. Disk holds only HMAC-SHA256(instance_secret, token),
so a stolen devices.json grants nothing.

Layering: io/api — core + config only.
"""
from __future__ import annotations

import enum
import hmac
import json
import logging
import os
import secrets
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

logger = logging.getLogger(__name__)

_SECRET_FILE = "instance_secret"
_DEVICES_FILE = "devices.json"
_SCHEMA_VERSION = 1


class Capability(str, enum.Enum):
    """What a device is allowed to ask for. Granted per device, never implied."""

    CHAT = "chat"
    SCREEN = "screen"
    FILES = "files"
    SYSTEM_CONTROL = "system_control"


@dataclass(frozen=True)
class Device:
    device_id: str
    label: str
    grants: frozenset[Capability]
    created_at: str


def _restrict_to_current_user(path: Path) -> None:
    """Windows ACL: owner only. Best-effort, logged when it fails."""
    if os.name != "nt":
        try:
            path.chmod(0o600)
        except OSError as exc:
            logger.warning(f"[API] could not restrict {path.name}: {exc}")
        return
    user = os.environ.get("USERNAME", "")
    if not user:
        logger.warning(f"[API] no USERNAME in environment; {path.name} left with inherited ACL")
        return
    try:
        subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:F"],
            check=True, capture_output=True, timeout=10,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning(f"[API] icacls failed for {path.name}: {exc}")


class TokenVault:
    """Owns the instance secret and the device records under `root`."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._secret: bytes | None = None

    # ─── instance secret ────────────────────────────────────────────────
    def instance_secret(self) -> bytes:
        env = os.getenv("TENKA_SECRET")
        if env:
            try:
                return bytes.fromhex(env.strip())
            except ValueError:
                return sha256(env.strip().encode("utf-8")).digest()

        if self._secret is not None:
            return self._secret

        path = self._root / _SECRET_FILE
        if path.exists():
            try:
                self._secret = bytes.fromhex(path.read_text(encoding="utf-8").strip())
                return self._secret
            except ValueError as exc:
                # Hand-edited or truncated secret file. A vault that raises here
                # takes the whole daemon down at startup; regenerate instead.
                logger.warning(f"[API] instance secret file is corrupt, regenerating: {exc}")

        self._root.mkdir(parents=True, exist_ok=True)
        secret = secrets.token_bytes(32)
        path.write_text(secret.hex(), encoding="utf-8")
        _restrict_to_current_user(path)
        self._secret = secret
        return secret

    # ─── device records ─────────────────────────────────────────────────
    def _hash(self, token: str) -> str:
        return hmac.new(self.instance_secret(), token.encode("utf-8"), sha256).hexdigest()

    def _load(self) -> dict:
        path = self._root / _DEVICES_FILE
        if not path.exists():
            return {"version": _SCHEMA_VERSION, "devices": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"[API] devices.json unreadable, starting empty: {exc}")
            return {"version": _SCHEMA_VERSION, "devices": []}
        if (
            not isinstance(data, dict)
            or data.get("version") != _SCHEMA_VERSION
            or not isinstance(data.get("devices"), list)
        ):
            logger.warning("[API] devices.json is malformed or has an unexpected schema; starting empty")
            return {"version": _SCHEMA_VERSION, "devices": []}
        return data

    def _save(self, data: dict) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._root / _DEVICES_FILE
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        _restrict_to_current_user(path)

    def _parse_device(self, entry: object) -> Device | None:
        """Fail closed: a malformed record is treated as absent, not trusted."""
        if not isinstance(entry, dict):
            return None
        try:
            return Device(
                device_id=entry["device_id"],
                label=entry["label"],
                grants=frozenset(Capability(g) for g in entry["grants"]),
                created_at=entry["created_at"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(f"[API] skipping malformed device record: {exc}")
            return None

    def issue(self, label: str, grants: frozenset[Capability]) -> str:
        token = secrets.token_urlsafe(32)
        data = self._load()
        data["devices"].append({
            "device_id": secrets.token_hex(8),
            "label": label,
            "grants": sorted(c.value for c in grants),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "token_hmac": self._hash(token),
        })
        self._save(data)
        return token

    def verify(self, token: str) -> Device | None:
        if not token or not token.strip():
            return None
        try:
            candidate = self._hash(token)
        except ValueError:
            return None
        match: dict | None = None
        for entry in self._load()["devices"]:
            # compare_digest on every well-formed entry: no early exit, no
            # timing signal between "unknown" and "revoked".
            if not isinstance(entry, dict):
                continue
            token_hmac = entry.get("token_hmac")
            if not isinstance(token_hmac, str):
                continue
            if hmac.compare_digest(token_hmac, candidate):
                match = entry
        if match is None:
            return None
        return self._parse_device(match)

    def revoke(self, device_id: str) -> bool:
        data = self._load()
        devices = data["devices"]
        remaining = [
            d for d in devices
            if not (isinstance(d, dict) and d.get("device_id") == device_id)
        ]
        if len(remaining) == len(devices):
            return False
        data["devices"] = remaining
        self._save(data)
        return True

    def devices(self) -> list[Device]:
        result = []
        for entry in self._load()["devices"]:
            device = self._parse_device(entry)
            if device is not None:
                result.append(device)
        return result

    def reset(self) -> None:
        """Rotate the instance secret. Every existing token stops verifying."""
        self._secret = None
        (self._root / _SECRET_FILE).unlink(missing_ok=True)
        (self._root / _DEVICES_FILE).unlink(missing_ok=True)
        self.instance_secret()
