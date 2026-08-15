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
import tempfile
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

    # Watching her work: status, telemetry, the live /v1/events stream, and
    # the routes that describe how she is configured (settings, personality,
    # the command catalogue, whether backups run). Everything here is about
    # the assistant herself, and none of it is something a user told her.
    OBSERVE = "observe"
    # Reading what she stored: conversation transcripts, the knowledge graph,
    # preferences, taught procedures, the names of the people she recognises.
    #
    # Split out of the old `CHAT`, which meant both of these at once. That
    # ambiguity let the `quick` ceiling -- the Cloudflare tunnel, where a
    # third party terminates TLS and reads the plaintext -- look like
    # "observation only" while actually admitting the entire knowledge graph
    # and every transcript. `read_screen` and `camera_look` are intents, so
    # her narration of what was on screen lands in a transcript: excluding
    # SCREEN from that ceiling while admitting RECALL was excluding the
    # photograph and shipping the description.
    #
    # Neither implies the other. A wall display may watch without reading a
    # word she was told; an archive tool may read history without a live view.
    RECALL = "recall"
    # POST /v1/chat hands text to the same pipeline voice uses, so it reaches
    # every intent -- code_executor, file_task, shutdown, manage_backup --
    # not just conversation. Neither read capability may carry that: both gate
    # routes a device should be able to hold without being able to drive her.
    # Split so a device can be trusted to read a transcript without being
    # trusted to act on the machine through one.
    CHAT_SEND = "chat_send"
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


def _atomic_write(path: Path, content: str) -> None:
    """Write `content` to `path` without ever leaving it truncated.

    A plain `write_text()` that is interrupted mid-write (power loss, a kill
    -9) leaves `path` truncated. For the instance secret specifically, a
    truncated file read before any secret is in memory -- i.e. at startup, the
    read that matters -- reads back as corrupt and silently regenerates the
    secret, which revokes every device (see `TokenVault.instance_secret`'s
    docstring). Writing to a same-directory temp file and swapping it in with
    `os.replace` makes the update atomic: a reader sees either the old
    content or the new content, never a partial write.
    """
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


class TokenVault:
    """Owns the instance secret and the device records under `root`."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._secret: bytes | None = None

    # ─── instance secret ────────────────────────────────────────────────
    def instance_secret(self) -> bytes:
        """Return the per-installation secret used to hash and verify tokens.

        Precedence: `TENKA_SECRET` env var, then the on-disk secret file,
        then a freshly generated secret persisted to that file.

        Disk is the truth, `self._secret` is only a fallback. The file is
        re-read on every call, so a rotation performed by any other vault
        instance -- or any other process -- is picked up immediately, by
        `verify()` and by `issue()` alike. That costs one ~64-byte read per
        HMAC, which is deliberate: `verify()` already re-reads and JSON-parses
        the whole of devices.json on every call via `_load()`, so the secret
        was the one input that could go stale while everything around it
        stayed live. A cache that can silently disagree with disk is how
        `issue()` came to mint tokens against a superseded secret -- valid
        until the process restarted, then permanently invalid, with no error
        anywhere.

        Side effect a caller must know about, unchanged in the case that
        matters: a stored secret file that is corrupt, empty, whitespace-only,
        or valid hex decoding to anything other than exactly 32 bytes is
        treated as no file at all *when nothing is cached yet* -- this method
        regenerates the secret and overwrites the file rather than raising.
        (`bytes.fromhex("")` returns `b""` without raising, so the length
        check is load-bearing, not redundant with the hex decode.) Every
        existing device's `token_hmac` was computed against the old secret, so
        regenerating silently revokes every device that was ever issued. That
        is the intended recovery path -- a vault that raises here takes the
        whole daemon down at startup, with no way back -- but it means this
        call, despite reading like a pure getter, can invalidate the whole
        device list. `_hash`, `verify`, `issue`, `revoke`, and `devices` all
        call this and inherit that risk on their first read.

        Once a secret *is* held in memory, a file that goes missing or corrupt
        mid-run does not regenerate: the cached secret is kept and a warning
        is logged instead. Every device stays valid. A vanished or mangled
        file is far more likely to be a backup tool, a sync client, or a stray
        delete than an instruction to invalidate every paired device, and
        revoking them all is not a decision to make from a failed read. It
        also stays out of the file: re-persisting the cached secret would hide
        the loss and bake in whichever process noticed first.

        `TENKA_SECRET` is handled differently, because the operator chose
        that value on purpose: an empty string is treated the same as the
        variable being unset (falls through to the file), but a non-empty
        value that decodes as hex to anything other than exactly 32 bytes
        raises `ValueError` immediately. There is nothing to regenerate for
        an explicit override -- silently accepting a weak key, or silently
        substituting a different secret than the one asked for, would both
        hide the operator's mistake instead of surfacing it.
        """
        env = os.getenv("TENKA_SECRET")
        if env:
            stripped = env.strip()
            try:
                secret = bytes.fromhex(stripped)
            except ValueError:
                return sha256(stripped.encode("utf-8")).digest()
            if len(secret) != 32:
                raise ValueError(
                    f"TENKA_SECRET decodes to {len(secret)} bytes; a 256-bit "
                    "secret needs exactly 32 (64 hex chars). Refusing to run "
                    "with an explicit secret that isn't the size it claims to be."
                )
            return secret

        path = self._root / _SECRET_FILE
        unusable: str | None = None  # set only when a file was there but unreadable
        if path.exists():
            try:
                secret = bytes.fromhex(path.read_text(encoding="utf-8").strip())
                if len(secret) != 32:
                    raise ValueError(f"decoded to {len(secret)} bytes, not 32")
            except (ValueError, OSError) as exc:
                # OSError matters more now than it did when this was read once
                # per process: a per-HMAC read is exposed to transient failures
                # (a locked file, a scanner holding a handle) that a single
                # startup read would simply never have met. `_load()` treats a
                # bad devices.json the same way.
                unusable = f"{exc}" or exc.__class__.__name__
            else:
                self._secret = secret
                return secret

        if self._secret is not None:
            logger.warning(
                f"[API] instance secret file is unusable ({unusable or 'it is gone'}); keeping the "
                "secret already in memory -- regenerating here would revoke every "
                "paired device. Restore the file before the next restart."
            )
            return self._secret

        if unusable is not None:
            logger.warning(
                f"[API] instance secret was unreadable; regenerated, all devices revoked ({unusable})"
            )
        self._root.mkdir(parents=True, exist_ok=True)
        secret = secrets.token_bytes(32)
        _atomic_write(path, secret.hex())
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
        _atomic_write(path, json.dumps(data, indent=2))
        _restrict_to_current_user(path)

    def _parse_device(self, entry: object) -> Device | None:
        """Fail closed: a malformed record is treated as absent, not trusted.

        An empty `grants` list is malformed in the same sense an unknown
        capability string is: `issue()` refuses to create one, so the only
        way one reaches here is a hand-edited devices.json. Treated the same
        way -- caught below, logged, no `Device` returned -- rather than
        parsing cleanly into a `Device(grants=frozenset())` that `verify()`
        would then hand to `authenticate()` unchanged. `devices()` calls this
        too, so a zero-grant entry drops out of the admin listing exactly
        like an unknown-capability one does -- but `revoke()` matches on the
        raw dict's device_id without going through this method, so an
        operator who reads devices.json directly can still find and revoke
        it by id even though it no longer verifies or lists.

        A record written before `CHAT` split into `OBSERVE`/`RECALL` carries
        `"chat"`, which `Capability("chat")` now rejects -- so it lands in
        exactly this branch and the device stops verifying. That is the
        intended outcome and there is deliberately no migration: mapping the
        old string onto `RECALL` would hand a device paired under the
        ambiguous grant the stored-data access the split exists to withhold.
        Note that it is *dropped*, not raised past -- one stale record must
        not take the whole store down.
        """
        if not isinstance(entry, dict):
            return None
        try:
            grants = frozenset(Capability(g) for g in entry["grants"])
            if not grants:
                raise ValueError("device has no grants")
            return Device(
                device_id=entry["device_id"],
                label=entry["label"],
                grants=grants,
                created_at=entry["created_at"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(f"[API] skipping malformed device record: {exc}")
            return None

    def issue(self, label: str, grants: frozenset[Capability]) -> str:
        # A device with no grants can do nothing useful, but it can still
        # authenticate: every route gated by `authenticate` alone (rather
        # than `require(capability)`) would answer it before any capability
        # is checked. For a route like POST /v1/commands/{id}/run, whose 404
        # (unknown command) and 403 (known, not granted) both come after
        # authentication, that is enough to let a zero-grant device tell the
        # two apart -- learning which command ids exist without ever holding
        # the OBSERVE grant that GET /v1/commands requires to read the same
        # list. Refusing to issue an empty grant set at all closes that
        # oracle at its source, for every current and future route shaped
        # this way, rather than patching each route that happens to need a
        # capability floor before its own logic runs.
        if not grants:
            raise ValueError("a device must be issued at least one capability")
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
            # instance_secret() no longer raises ValueError on a corrupt
            # secret file (it regenerates instead), so this no longer guards
            # that. What it still guards: a token containing a lone UTF-16
            # surrogate code point, which token.encode("utf-8") rejects with
            # UnicodeEncodeError -- a ValueError subclass. Untrusted input
            # must never raise out of verify().
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
        """Rotate the instance secret. Every existing token stops verifying.

        Cross-process note: revocation is visible to a *different* TokenVault
        instance -- e.g. the running daemon's, while this call is made from a
        slash command -- twice over. `_DEVICES_FILE` is deleted, and `verify()`
        calls `_load()`, which re-reads that file from disk on every call, so
        even a vault that somehow held the old secret would hash it against an
        empty device list and match nothing. Single-device `revoke()` is
        visible the same way; the device list is never cached.

        The secret rotation itself is now visible too, independently of the
        device list: `instance_secret()` re-reads the secret file on every
        call, so the next `verify()` or `issue()` on any instance, in any
        process, uses the secret written here. That closes the edge this
        docstring used to warn about -- `issue()` on a vault whose cached
        secret predated a rotation minted a token hashed against the stale
        secret, which verified for the rest of that process's lifetime and
        then silently and permanently stopped verifying once a fresh vault
        read the rotated secret off disk. The guarantee is now positive: a
        token that `issue()` returns after this call verifies on any vault
        pointed at the same root, including one started after a restart.
        """
        self._secret = None
        (self._root / _SECRET_FILE).unlink(missing_ok=True)
        (self._root / _DEVICES_FILE).unlink(missing_ok=True)
        self.instance_secret()
