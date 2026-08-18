# assistant/io/api/raises.py
"""In-memory record of a ceiling raise: who, what, on which transport, until when.

Spec §3.2 (`.superpowers/sdd/2026-08-16-milestone6b-transports/spec.md`): a
raise must expire on a clock the requester does not control, and must not
survive a daemon restart. `time.monotonic()` handles the first half -- a
wall-clock change cannot extend a raise, and on CPython 3.11/Windows it is
`GetTickCount64`, which keeps counting through sleep, so a suspended laptop
does not bank time either.

The second half is structural here, not a policy someone could later relax.
This module keeps records in a plain `dict`, in process memory, and never
writes one to disk: there is no `save`, `load`, `to_json`, `from_json` or
`__getstate__` anywhere below, and the module imports neither `json` nor
`pathlib`. A test asserts all of that, because "nothing currently calls the
save method" is a different guarantee from "there is no save method to call."

`ListenerPolicy.raisable` (`policy.py`, Task 1) is the fixed, per-transport
ceiling on what a raise could ever reach -- static module data, vetted once
by a human. This store holds the live, expiring, device-scoped record that
`effective()`'s third argument reads. It does not decide whether a raise is
*admissible* -- whether a capability sits in `raisable`, whether the device
holds it, whether the transport is even running -- that judgment needs the
policy table and the vault both in hand, and belongs to the route that mints
a raise (Task 10), not to the record it mints. Keeping that split is what
lets this module import `Capability` and nothing else from the API layer.

Layering: io/api -- core + config only.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from ...core.capabilities import Capability

logger = logging.getLogger(__name__)

# Spec §3.3: "Duration is caller-set, in minutes, clamped to a 7-day hard
# cap." A caller asking for longer than this gets the cap, silently -- the
# cap is the safety property, not a promise the caller kept their word.
MAX_RAISE_SECONDS = 7 * 24 * 60 * 60


@dataclass(frozen=True)
class RaiseGrant:
    """One live raise: what it lifted, until when, who granted it, and why.

    `expires_at` is a `time.monotonic()` reading -- see the module docstring
    for why it is never a wall-clock timestamp. Never serialised, never
    handed to a non-admin listener except as the summary spec §3.6 describes
    (a later task's concern, not this record's).
    """

    capabilities: frozenset[Capability]
    expires_at: float
    granted_by: str
    reason: str


class RaiseStore:
    """Live raises, keyed on `(device_id, policy_name)`.

    Guarded by a single `threading.Lock`: `authenticate()` reads this store
    from a worker thread via `asyncio.to_thread` on every request, while the
    admin raise route writes it from the event loop that daemon shares with
    the assistant, which must never block on one.

    Expiry is checked on read, never by a timer, and a read that finds an
    expired record deletes it there and then -- a stale entry must never be
    handed back as live, and nothing here schedules a callback to clean one
    up later.
    """

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], RaiseGrant] = {}
        self._lock = threading.Lock()

    # ─── Minting and dropping ─────────────────────────────────────────────

    def grant(
        self,
        device_id: str,
        policy_name: str,
        capabilities: frozenset[Capability],
        seconds: float,
        granted_by: str,
        reason: str,
    ) -> RaiseGrant:
        """Mint (or replace) the raise for one device on one transport.

        Admissibility -- is `capabilities` a subset of the transport's
        `raisable`, does the device hold them, is the transport running --
        is the caller's job (Task 10's route); this method only refuses a
        duration that cannot be a raise at all.
        """
        if seconds <= 0:
            raise ValueError("raise duration must be positive")
        clamped = min(seconds, MAX_RAISE_SECONDS)
        record = RaiseGrant(
            capabilities=frozenset(capabilities),
            expires_at=time.monotonic() + clamped,
            granted_by=granted_by,
            reason=reason,
        )
        with self._lock:
            self._records[(device_id, policy_name)] = record
        return record

    def revoke(self, device_id: str, policy_name: str) -> bool:
        """Drop exactly one record. Returns whether one was there to drop."""
        with self._lock:
            return self._records.pop((device_id, policy_name), None) is not None

    def drop_device(self, device_id: str) -> None:
        """Drop every raise held by one device, across every transport.

        Called when the device itself is revoked: a raise outliving the
        device it was granted to would be absurd.
        """
        with self._lock:
            for key in [key for key in self._records if key[0] == device_id]:
                del self._records[key]

    def drop_policy(self, policy_name: str) -> None:
        """Drop every raise on one transport, across every device.

        Called when that transport's listener stops: a raise scoped to a
        tunnel that no longer exists can never be exercised, and must not
        be left behind for a future listener that reuses the same name.
        """
        with self._lock:
            for key in [key for key in self._records if key[1] == policy_name]:
                del self._records[key]

    def clear(self) -> None:
        """Drop every raise, for every device and transport.

        Called by `server.shutdown()`, alongside (not by) `vault.reset()` --
        the kill switch already revokes every device, and a raise surviving
        it would be absurd. Not called by `vault.reset()` itself: `TokenVault`
        has no reach into this store, and `shutdown()` is the one place that
        holds both.
        """
        with self._lock:
            self._records.clear()

    # ─── Reading ────────────────────────────────────────────────────────────

    def capabilities_for(self, device_id: str, policy_name: str) -> frozenset[Capability]:
        """What `effective()` should fold in as `raised`.

        The hot path: `authenticate()` calls this on every request. Returns
        `frozenset()` for a miss -- never `None` -- so every call site can
        fold the result straight into a set operation without a null check.
        """
        record = self._read(device_id, policy_name)
        return record.capabilities if record is not None else frozenset()

    def get(self, device_id: str, policy_name: str) -> RaiseGrant | None:
        """The full record, for callers that need `expires_at` too (the
        session payload's "seconds remaining"), not just the capabilities."""
        return self._read(device_id, policy_name)

    def active(self) -> dict[tuple[str, str], RaiseGrant]:
        """A snapshot of every currently-live raise, for `GET /v1/devices`'s
        per-device summary.

        Reading the whole store is still a read: an expired record found
        along the way is dropped here too, never merely omitted from the
        snapshot handed back.
        """
        with self._lock:
            now = time.monotonic()
            expired = [key for key, record in self._records.items() if record.expires_at <= now]
            for key in expired:
                del self._records[key]
            live = dict(self._records)
        # Outside the lock, deliberately: a logging handler writes to a file,
        # and holding this lock across that write would put the admin route and
        # every authenticated request behind disk I/O.
        self._announce_expiry(expired)
        return live

    # ─── Internal ───────────────────────────────────────────────────────────

    def _read(self, device_id: str, policy_name: str) -> RaiseGrant | None:
        key = (device_id, policy_name)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return None
            if record.expires_at <= time.monotonic():
                del self._records[key]
                record = None
        if record is None:
            # Reached only on the delete-above branch: the `is None` return a
            # few lines up already left. See `_announce_expiry` for why this is
            # one line per record rather than one per read.
            self._announce_expiry([key])
            return None
        return record

    def _announce_expiry(self, keys: list[tuple[str, str]]) -> None:
        """Spec §3.6: a log line on mint, and one on expiry.

        Expiry has no timer to hang a line off -- a record dies on the read
        that notices it -- so this fires from the delete-on-read paths above.
        That is still **one line per record, not one per read**: the record is
        gone from the dict by the time this runs, so no later read can find it
        to announce a second time.

        Ids and the transport, and nothing else. The reason string and the
        capabilities are already on the mint line, and repeating free text the
        operator typed into a log file that outlives the raise buys nothing.
        """
        for device_id, policy_name in keys:
            logger.info(f"[API] ceiling raise expired (device={device_id} "
                        f"transport={policy_name})")
