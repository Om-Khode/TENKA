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
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

logger = logging.getLogger(__name__)

_SECRET_FILE = "instance_secret"
_DEVICES_FILE = "devices.json"
_SCHEMA_VERSION = 1

# "Last seen" carries no useful signal at sub-minute resolution, and
# TokenVault.touch() is wired into the authenticated request path (Milestone
# 6a Task 10): without a floor, every authenticated call pays a full
# _load + _save, and _save's _restrict_to_current_user spawns an `icacls`
# subprocess on Windows -- a process spawn per request, and a flood of cheap
# requests becomes a flood of subprocess spawns against the one API that is
# about to be reachable from a public URL. One write per device per window
# is enough for a revoke-list column and caps the cost regardless of request
# rate.
_TOUCH_THROTTLE = timedelta(seconds=60)


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
    last_seen_at: str | None = None


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


class VaultReadError(RuntimeError):
    """`devices.json` exists but could not be read or parsed -- its state is
    unknown, not empty.

    Distinct from an absent file, which `_load()` treats as a legitimate
    "nothing has ever been issued" without ever reaching this branch. Raised
    by `_load()` so that a mutating method (`issue`, `revoke`, `touch`) that
    is about to write the *entire* document back cannot mistake "I could not
    read this" for "there is nothing here" and silently overwrite whatever is
    actually on disk with a synthetic empty one -- the fix for a review
    finding that proved exactly that on Windows: a file locked by a scanner,
    a backup tool, or a second TENKA process raises `PermissionError`, and
    `issue()` used to treat that identically to a fresh install and save a
    document containing only the one device it was asked to add, destroying
    every other record.

    A caller that only *reads* (`verify`, `devices`) is not obligated to fail
    the same way -- there is nothing to overwrite -- so each decides for
    itself, and documents why, at its own call site.
    """


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
    """Owns the instance secret and the device records under `root`.

    `issue`, `revoke`, and `touch` are each a `_load()` ... mutate ... `_save()`
    sequence, and none of them held a lock until this pass. Two of those
    sequences interleaving on the same in-memory document is a lost update:
    whichever finishes last overwrites the other's write in full, silently.
    The scenario that made this more than a theoretical race: `touch()` now
    runs on the authenticated request path (Milestone 6a Task 10), so a
    misbehaving device's in-flight request can call `touch()` at the exact
    moment an operator calls `revoke()` on it -- which is exactly when
    anyone revokes anything. `touch()` loads a copy that still has the
    device, `revoke()` runs to completion and removes it, then `touch()`
    saves its stale copy back and un-revokes the device it was never told
    about. `issue()` racing `revoke()` is the same class of bug: whichever
    saves last wins, and the other caller's write vanishes with no error
    anywhere. `self._lock` (a plain `threading.Lock`, mirroring
    `PairCodeStore`'s in `pairing.py`) makes each of those three methods hold
    the lock across its *entire* load-mutate-save sequence, so no second
    mutator can observe -- or overwrite -- a snapshot the first is still
    working from.

    Deliberately not locked: `verify()`, `devices()`, and `instance_secret()`.
    All three only read; `_atomic_write`'s `os.replace()` already guarantees
    that any concurrent read of `devices.json` (or the secret file) sees a
    complete write or the previous complete write, never a torn one, so a
    lock buys a lock-free read nothing in exchange for making it wait behind
    whichever mutator got there first. `verify()` in particular runs on
    every authenticated request specifically so a revocation is never stale
    (see its own note about re-reading on every call); serialising it behind
    `touch()`'s throttled-but-still-real disk writes would tax the one path
    this design goes out of its way to keep live, for a torn-read hazard
    that does not exist.

    Cross-process concurrency is explicitly out of scope for this lock. Two
    separate TENKA processes -- the uvicorn daemon and, say, a `/studio
    revoke` invocation that opens its own `TokenVault` on the same root --
    are not serialised by a `threading.Lock` at all; it only ever coordinates
    threads *within one process*. That hazard is real and unfixed by anything
    here -- it would need a file lock (`msvcrt.locking` / `fcntl.flock`) or a
    single-writer process design, neither of which this pass adds.

    What that hazard actually looks like on Windows is not the same shape as
    the same-process race above, and an earlier version of this docstring
    named it as one: a silent lost update, one process's complete write
    overwriting another's. Measured under real contention (six concurrent
    readers, two concurrent writers, all against one `devices.json`), the
    dominant failure was `PermissionError [WinError 5]` -- Windows refuses a
    second handle on a file another process has open, so a write racing
    another process's read (or write) is far more likely to fail loudly than
    to succeed and silently clobber. `_load()` raising `VaultReadError` on
    exactly that failure (rather than swallowing it into a synthetic empty
    document, which is what made `issue()` dangerous under this same
    contention -- see `VaultReadError`'s own docstring) is what turns a cross-
    process collision into a request the caller sees fail and can retry,
    instead of a device list that quietly loses everything but the record
    just written. It does not make cross-process writes safe in general --
    a true interleaved lost update remains possible in principle, and nothing
    here adds real mutual exclusion across processes -- but "fails loudly" is
    the practical, observed Windows behaviour this pass leaves in place.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._secret: bytes | None = None
        # Guards `issue`/`revoke`/`touch`'s _load-through-_save sequence as
        # one critical section. See the class docstring for what this does
        # and does not cover.
        self._lock = threading.Lock()

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
        """Load the devices document, or raise `VaultReadError` if its actual
        content could not be determined.

        Absent is not the same as unreadable: a file that has never existed
        (fresh install, or `reset()` just deleted it) is a legitimate, known
        empty vault, handled below without ever reaching the `try`. A file
        that exists but raises `OSError` (locked by another process) or
        `json.JSONDecodeError` (not parseable as JSON at all -- a torn read,
        or genuine corruption) is a state this method does not actually know,
        and is no longer treated as "empty" the way it used to be: a mutator
        that received a synthetic empty document here could not tell it apart
        from a real one and would save straight over whatever is genuinely on
        disk. See `VaultReadError`'s docstring for what that cost in
        practice. A read-only caller may still choose to treat this the same
        way `verify()` and `devices()` do -- fail closed to "nothing verifies"
        or "nothing is listed" -- but that is now a decision each of them
        makes and documents for itself, not something this method decides
        for all of them by handing back a lie.
        """
        path = self._root / _DEVICES_FILE
        if not path.exists():
            return {"version": _SCHEMA_VERSION, "devices": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise VaultReadError(f"devices.json unreadable: {exc}") from exc
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
                # `.get`, not `[...]`: every record written before this task
                # has no such key at all, and that absence is not malformed --
                # it just means "never touched". Only a genuinely missing
                # required field (device_id, label, grants, created_at) should
                # drop a record; this one must default instead.
                last_seen_at=entry.get("last_seen_at"),
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
        # Locked for the whole load-append-save sequence: without it, a
        # concurrent revoke() that loads, filters, and saves entirely inside
        # this window would have its write clobbered the instant this method
        # saves its own copy back -- the new device would survive, but the
        # revocation that raced it would silently vanish.
        with self._lock:
            # Not caught here: a `VaultReadError` means the current document
            # is unknown, and appending to a synthetic empty one and saving
            # would destroy every other device on disk the moment the lock
            # releases. Propagating lets the caller (a route) answer "try
            # again" instead of the vault silently doing that. See
            # `VaultReadError`'s docstring.
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
        try:
            loaded = self._load()
        except VaultReadError as exc:
            # Decision: fail closed, same as an unknown or revoked token.
            # This is the call `authenticate()` makes on every single
            # request specifically so a revocation is never stale (see this
            # method's own docstring above) -- so the honest options are
            # "refuse everyone until the read recovers" or "let a locked
            # file quietly authenticate as whatever was last cached", and
            # the second one is not a thing this vault does anywhere else.
            # The cost is real: a transient lock (a scanner, a backup tool,
            # a second TENKA process) refuses every device for as long as it
            # holds the file, with no partial credit for "some devices would
            # still have matched." That is the trade-off deliberately taken
            # here rather than risk verifying a token that a stale or
            # partial read would have rejected.
            logger.warning(f"[API] verify() could not read devices.json, "
                           f"refusing every device until it recovers: {exc}")
            return None
        match: dict | None = None
        for entry in loaded["devices"]:
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
        # Locked for the whole load-filter-save sequence -- this is the kill
        # switch this whole design leans on (see class docstring). A revoke
        # that only holds the lock around `_load()` or only around `_save()`
        # is exactly as unsafe as no lock at all: the window a racing
        # touch()/issue() needs to observe a stale snapshot is *between*
        # those two calls, not during either one alone.
        with self._lock:
            # Not caught here either. A synthetic empty document would make
            # `device_id` look absent -- `revoke()` would return `False`,
            # "nothing was revoked", when the truth is "the vault could not
            # be read" -- and a caller relying on that boolean (an operator
            # trying to cut a device off) must not be told the device is
            # already gone when its actual state is unknown.
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

    def touch(self, device_id: str) -> None:
        """Record that `device_id` was just seen, for the revoke-list UI.

        Silent on an unknown id: a stale or already-revoked credential
        showing up on the request path is ordinary, not exceptional, and a
        vault method must not hand a caller a reason to surface an error for
        it. That silence covers "the device is not in the document" only.
        "The document could not be read at all" is a different fact --
        `_load()` raises `VaultReadError` for it instead of handing back a
        synthetic empty document, and this method does not catch that,
        because a target genuinely found in a stale-but-readable snapshot
        could otherwise be silently skipped as if it were merely unknown.
        `authenticate()` and the event socket, the only two callers, treat a
        failed touch as best-effort and do not fail the request over it --
        see their own call sites for why.

        Throttled to one disk write per `_TOUCH_THROTTLE` per device: this
        sits on the authenticated request path, and `_save` is not free (a
        full JSON rewrite plus an `icacls` subprocess via
        `_restrict_to_current_user`). Without a floor, a device polling every
        few seconds -- or an attacker flooding an authenticated route -- pays
        that cost on every single call. A device with no stored timestamp
        yet (first sighting, or a pre-this-task record) always writes; one
        within the window is skipped with no `_load`-then-`_save` round trip
        wasted beyond the read already done to check.

        A stored timestamp that is in the future -- clock skew, or a
        hand-edited file -- must not wedge the write off forever, so it is
        treated as stale rather than as "just touched": the age computed
        against it is negative, which falls outside the
        `0 <= age < _TOUCH_THROTTLE` window that suppresses the write, so the
        write proceeds and self-heals the bogus value.

        A stored timestamp with no `tzinfo` (naive) is treated the same way
        as unparseable garbage: overwritten, not assumed-UTC. A naive
        value's true offset is genuinely unknown -- guessing UTC would
        invent a fact this method has no basis for -- and subtracting an
        aware `now` from a naive `stored_at` raises `TypeError` outright, so
        this must not be allowed to reach the subtraction unguarded. That
        mirrors `_parse_device`'s posture elsewhere in this file: a
        malformed field is dropped/overwritten, never allowed to crash the
        caller.
        """
        # Locked for the whole load-check-save sequence, not just the save:
        # this method sits on the authenticated request path (Task 10), which
        # is exactly where it can race an operator's revoke() of the same
        # device -- the scenario the class docstring walks through. Every
        # `return` below exits the `with` block too, releasing the lock, so a
        # throttled no-op touch (the common case) never blocks anyone.
        with self._lock:
            data = self._load()
            target: dict | None = None
            for entry in data["devices"]:
                if isinstance(entry, dict) and entry.get("device_id") == device_id:
                    target = entry
                    break
            if target is None:
                return

            now = datetime.now(timezone.utc)
            stored = target.get("last_seen_at")
            if isinstance(stored, str):
                try:
                    stored_at = datetime.fromisoformat(stored)
                    age = now - stored_at
                except (ValueError, TypeError):
                    # ValueError: not a parseable ISO-8601 string at all.
                    # TypeError: parsed fine but naive -- `fromisoformat` does not
                    # raise on a string with no offset, and subtracting an aware
                    # `now` from a naive `stored_at` is what actually raises.
                    # Both are malformed in the sense that matters here: fall
                    # through and overwrite rather than let either one crash the
                    # request path that calls this.
                    pass
                else:
                    if timedelta(0) <= age < _TOUCH_THROTTLE:
                        return  # seen recently enough; skip the write entirely

            target["last_seen_at"] = now.isoformat()
            self._save(data)

    def devices(self) -> list[Device]:
        # Decision: fail closed to an empty listing, matching `verify()`.
        # This is an admin-only read (`GET /v1/devices`, `/studio devices`),
        # not a security decision -- reporting "none" while a lock clears is
        # a stale UI, not a privilege granted to anyone who shouldn't have
        # had it, so there is nothing here worth raising into a caller that
        # almost certainly cannot do anything about a locked file anyway.
        try:
            loaded = self._load()
        except VaultReadError as exc:
            logger.warning(f"[API] devices() could not read devices.json, "
                           f"reporting none: {exc}")
            return []
        result = []
        for entry in loaded["devices"]:
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
