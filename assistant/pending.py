"""Pending state — multi-turn dialog tracker.

A PendingState owns one interactive dialog state (destructive op confirmation,
oauth setup, messaging disambiguation, teaching session, etc.). It carries a
payload, a touch timestamp, a timeout, and — since KI-13 — the principal that
armed it; reading the payload after the timeout has elapsed auto-clears the
state.

Eventually relocates to `actions/pending/base.py` once `actions.py` is split
into a package.

Usage:

    from assistant.pending import PendingState, pending_registry

    destructive = pending_registry.register(
        PendingState[dict]("destructive", timeout=30.0)
    )

    # start a pending interaction
    destructive.set({"op": "delete", "path": some_path})

    # later, in the handler
    payload = destructive.payload   # None if cleared or expired
    if payload is None:
        return None                 # not in a destructive flow
    payload["dest_folder"] = candidate
    destructive.touch()             # reset the timeout
"""
import time
from typing import Generic, Optional, TypeVar

from .core.principal import current_principal

T = TypeVar("T")


# ─── "I did not name an owner" vs "this is owned by nobody" ──────────────

class _AmbientPrincipal:
    """Sentinel for `PendingState.set`'s `principal` argument.

    `set()` has to be able to say two different things, and before this
    sentinel existed it could only say one of them, because both were spelled
    `None`:

    - **"I am not naming an owner — use the turn's."**  The ~18 bare
      `state.set(payload)` calls in `actions/` mean this. They sit several
      frames below the turn that authorised them and inherit its identity.
    - **"This is owned by nobody."**  A lazily-arming row carrying a principal
      that was captured as `None` means this. `owned_by` refuses everyone for
      such a state, in both directions, which is the fail-closed answer.

    Collapsing them made `set(payload, principal=x)` silently *invert* when
    `x` was `None`: an unowned proposal became owned by whoever spoke next,
    and the very next line's `owned_by(current_principal.get())` then said yes
    to them. A caller passing a value it did not choose got the opposite of
    what it asked for. So the two meanings get two spellings, and the one that
    can only be written deliberately — omitting the argument — is the one that
    consults the ambient principal.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return "AMBIENT_PRINCIPAL"


AMBIENT_PRINCIPAL = _AmbientPrincipal()


class PendingState(Generic[T]):
    """Single multi-turn interaction state."""

    def __init__(self, name: str, timeout: float):
        self.name = name
        self.timeout = timeout
        self._payload: Optional[T] = None
        self._ts: float = 0.0
        self._principal: Optional[str] = None
        self._foreign_attempts: int = 0

    def set(self, payload: T, *,
            principal: "Optional[str] | _AmbientPrincipal" = AMBIENT_PRINCIPAL,
            ) -> None:
        """Start (or replace) the pending state, recording who armed it.

        Three spellings, three meanings:

        - `set(payload)` — inherit the turn in flight (`current_principal`).
          That default is the mechanism, not a convenience. Almost every
          arming site in the tree sits inside a handler several frames below
          the turn that authorised it; asking each of them to pass an identity
          down would put the whole property one forgotten argument away from
          an unowned confirmation, which is a silent dead end rather than a
          loud one. Sites that arm *outside* a turn -- `main.py`'s
          notification flusher, where no principal is installed -- have to
          state one, and
          `tests/test_6b_principal.py::test_every_arming_site_records_a_principal`
          walks `main.py`'s AST to make sure they do.
        - `set(payload, principal="device:x")` — owned by that principal.
        - `set(payload, principal=None)` — **explicitly owned by nobody.**
          `owned_by` then refuses everyone, including the caller doing the
          arming. This is the spelling a lazily-arming row needs when the
          owner it carries turns out to be unknown: see `_AmbientPrincipal`
          for why it cannot share the "not specified" spelling.
        """
        self._payload = payload
        self._ts = time.time()
        self._principal = (current_principal.get()
                           if isinstance(principal, _AmbientPrincipal)
                           else principal)
        self._foreign_attempts = 0

    def touch(self) -> None:
        """Reset the timeout without changing the payload (for re-prompts).

        Deliberately does not re-read `current_principal`: a re-prompt is the
        same question asked again, not a new one, so it must not silently
        transfer ownership to whoever happened to trigger the re-prompt.
        """
        if self._payload is not None:
            self._ts = time.time()

    def clear(self) -> None:
        """End the pending state."""
        self._payload = None
        self._ts = 0.0
        self._principal = None
        self._foreign_attempts = 0

    def note_foreign_attempt(self) -> None:
        """Somebody who does not own this state just tried to answer it.

        The counter exists because the refusal has to reach the *owner*, and
        the owner is not in the conversation where the attempt happened. KI-13
        asks that a mismatch be loud "so the operator sees that something else
        tried to answer" -- telling only the one who tried satisfies the
        letter and misses the point, and a WARNING in `debug.log` is forensics
        rather than a person being told.

        So the fact is parked on the state itself and collected by
        `take_foreign_attempts()` when the owner next answers. It counts
        rather than flags: "somebody kept trying" and "somebody tried once"
        are different things to learn, even if the sentence the operator hears
        does not currently distinguish them.

        A no-op when the state is not armed. An attempt on a state nobody is
        waiting on is not something to report to a later, unrelated question.
        """
        if self._payload is not None:
            self._foreign_attempts += 1

    def take_foreign_attempts(self) -> int:
        """How many foreign attempts have piled up, and reset the counter.

        Read-and-clear rather than a plain property, so the owner is told
        exactly once per batch. Called only on the path that actually delivers
        something to the owner: if her answer was not understood, the count
        survives to be reported on the next one rather than being burned on a
        turn she never saw it in.
        """
        count = self._foreign_attempts
        self._foreign_attempts = 0
        return count

    @property
    def payload(self) -> Optional[T]:
        """Current payload, or None if inactive/expired.

        Reading an expired state clears it as a side effect.
        """
        if self._payload is None:
            return None
        if time.time() - self._ts > self.timeout:
            self.clear()
            return None
        return self._payload

    @property
    def principal(self) -> Optional[str]:
        """Who armed this state, or None if nobody said.

        Deliberately not folded into `payload`: ownership and expiry are
        different questions, and answering them through one property would
        make "someone else is answering" indistinguishable from "you took too
        long" -- which is the silent failure KI-13 exists to end.
        """
        return self._principal

    def owned_by(self, principal: Optional[str]) -> bool:
        """May `principal` answer this state?

        True only when both sides are set and equal. An unset principal owns
        nothing and an unowned state is answerable by nobody -- in both
        directions, the absence of a decision is not a decision to allow,
        exactly as `current_grants`' default of `None` refuses everything.

        Note what this deliberately does *not* do: `LOCAL_PRINCIPAL` is not a
        master key. A confirmation armed from a phone is the phone's to
        answer, and a "yes" typed at the console is as much a different voice
        as a remote one is.
        """
        return (self._principal is not None
                and principal is not None
                and self._principal == principal)

    @property
    def active(self) -> bool:
        """True iff payload is set and not expired. Clears on expiry."""
        return self.payload is not None

    @property
    def age(self) -> float:
        """Seconds since the state was last touched. 0.0 if inactive."""
        if self._payload is None:
            return 0.0
        return time.time() - self._ts


# ─── Arming without clobbering another principal's open question ─────────

def try_arm(state: "PendingState", payload, *,
            principal: "Optional[str] | _AmbientPrincipal" = AMBIENT_PRINCIPAL,
            ) -> bool:
    """Arm `state` with `payload`, unless that would replace another
    principal's still-active state.

    KI-13 closed the *answer* side of this: `main.py`'s dispatch loop skips a
    caller who does not own an active state (`state.active and not
    state.owned_by(principal)`) before it ever reaches a handler. Nothing
    mirrored that check on the *arm* side, so a foreign caller who was
    correctly skipped as an answer would fall through to ordinary
    classification, reach a handler that arms the very same state, and
    `PendingState.set()` — which unconditionally replaces, by design, so a
    caller mid-flow can always refine or restart its own question — would
    hand that caller ownership. The operator's own next "yes" then reads as a
    foreign answer against her own request: a denial of service on her
    confirmation, delivered by the mechanism meant to protect it.

    This is that mirror. Same test, same shape, opposite side:

    - **the same principal** re-arming its own active state (refining an open
      question, or abandoning it for a new one) passes `owned_by` and
      overwrites exactly as `set()` always has -- nothing about the ~14
      legitimate multi-step flows in `actions/` changes.
    - **a different principal** — or nobody, since an unowned active state is
      owned_by nobody in either direction — is refused. The existing payload,
      timestamp and owner are left untouched, and the refusal is parked on
      the state via `note_foreign_attempt()` so the real owner learns about
      it the same way she already learns about a foreign *answer*: on her
      next turn, read-and-cleared by `take_foreign_attempts()`.

    Deliberately not folded into `PendingState.set()` itself: several tests
    (`test_a_bare_arm_inherits_the_turns_principal` and its neighbours) pin
    `set()` as an unconditional replace at the mechanism layer, with ownership
    enforced only where a caller is identified -- the dispatch loop for
    answers, this function for arms. Collapsing the two would turn those
    pinned, deliberate overwrites into refusals.

    Returns True if armed, False if refused. Most callers do not need to look
    at the return value: the caller that was refused gets back exactly the
    same response text an ordinary first arm would have produced (whether it
    succeeded or not is never disclosed), and a "yes" from that caller later
    is skipped by the ordinary owner check exactly as it is today -- no new
    disclosure, just a confirmation that quietly stayed the real owner's.
    """
    resolved = (current_principal.get()
                if isinstance(principal, _AmbientPrincipal)
                else principal)
    if state.active and not state.owned_by(resolved):
        state.note_foreign_attempt()
        return False
    state.set(payload, principal=principal)
    return True


# ─── Clearing without discarding another principal's open question ───────

def try_clear(state: "PendingState", *,
              principal: "Optional[str] | _AmbientPrincipal" = AMBIENT_PRINCIPAL,
              ) -> bool:
    """Clear `state`, unless that would discard another principal's still-
    active state.

    The third door into the same room `try_arm` closed. KI-13's answer side
    (the dispatch loop) and `try_arm`'s arm side both already refuse a foreign
    principal; a bare `.clear()` reachable outside either check is the same
    denial of service spelled a third way -- a foreign caller who cannot
    answer the operator's confirmation and cannot re-arm over it can still
    make it vanish, and the operator finds her open question gone with no
    "something else tried" to explain why. Discarding a confirmation nobody
    but its owner should be able to end is exactly as bad as overwriting it or
    answering it.

    Same test as `try_arm`, same shape, one more time:

    - **the same principal** clearing its own active state passes `owned_by`
      and tears it down exactly as a bare `.clear()` always has -- every
      legitimate self-teardown in `actions/` (the flows already gated by the
      dispatch loop's `state.active and not state.owned_by(principal)` check
      before their handler is ever called, so the only principal that can
      reach their `.clear()` calls is already the owner) keeps working
      unchanged.
    - **a different principal** -- or nobody, since an unowned active state is
      owned_by nobody in either direction -- is refused. The state is left
      untouched and the refusal is parked via `note_foreign_attempt()`, read
      the same way a foreign answer or a foreign arm attempt already is: on
      the owner's next answer, read-and-cleared by `take_foreign_attempts()`.

    Most of the tree's `.clear()` call sites never needed this: they sit
    inside a `handle_pending_*` function the dispatch loop (or, for
    `teaching_session`, `main.py`'s own answer site; or, for
    `knowledge_approval`, the handler's own lazily-armed `owned_by` check)
    already refuses to reach for a non-owner, so by the time the body runs the
    only caller who could still be executing it is the owner -- guarding
    there again would be a no-op, not a hardening. This exists for the one
    shape that check cannot reach: a clear sitting inside an *ordinary* tool
    handler (`camera_look`'s tidy-up of a stale `pending_camera_settings`
    offer, run for every caller of the `camera_look` intent, not just the one
    who was asked about camera settings) rather than inside the pending-answer
    chain itself.

    Returns True if cleared, False if refused.
    """
    resolved = (current_principal.get()
                if isinstance(principal, _AmbientPrincipal)
                else principal)
    if state.active and not state.owned_by(resolved):
        state.note_foreign_attempt()
        return False
    state.clear()
    return True


class PendingRegistry:
    """Registry of all PendingStates so the planner can snapshot them.

    Replaces the reflection-based `_PENDING_VARS` list of string identifiers.
    """

    def __init__(self):
        self._states: dict[str, PendingState] = {}

    def register(self, state: PendingState) -> PendingState:
        """Register a state under its `name`. Returns the state for chaining.

        Raises ValueError if a state with the same name is already registered.
        """
        if state.name in self._states:
            raise ValueError(f"PendingState '{state.name}' already registered")
        self._states[state.name] = state
        return state

    def get(self, name: str) -> Optional[PendingState]:
        return self._states.get(name)

    def snapshot(self) -> dict[str, bool]:
        """Return {name: active} for every registered state.

        The planner calls this before and after a step to detect whether the
        step triggered a user-interaction flow (a state that flipped from
        inactive to active).
        """
        return {name: s.active for name, s in self._states.items()}

    def any_active(self, exclude: set[str] | None = None) -> bool:
        """True if any registered state is active (optionally excluding some)."""
        for name, state in self._states.items():
            if exclude and name in exclude:
                continue
            if state.active:
                return True
        return False

    def names(self) -> list[str]:
        return list(self._states.keys())


# Module-level singleton — the shared registry every PendingState joins.
pending_registry = PendingRegistry()
