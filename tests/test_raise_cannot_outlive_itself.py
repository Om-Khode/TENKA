"""A raise cannot install something that keeps running after it expires.

A capability raise is deliberately time-bounded: minted at the keyboard,
`require_admin(SYSTEM_CONTROL)`, scoped to one device and one transport,
expiring. `manage_monitor`, `manage_schedule`, `manage_procedure`,
`manage_shortcut` and `manage_backup` all install something that runs *later*,
and `automation/event_bus.py` and `scheduler.py` run it with `LOCAL_GRANTS` on
the stated argument that whoever installed it already held `EXECUTE`.

That argument is sound only if "held" means **durably**. Spend a thirty-minute
raise on installing a monitor and the expiry stops mattering: the row fires on
a cadence forever, attributed in every log to `local`. The bound the raise
exists to provide is defeated by the artefact it was spent on.

The chain, every link verified in the tree on 2026-08-22:

  1. a device pairs over `tailnet` with EXECUTE ticked -- 6b's issue-time fix
     stores it in the vault rather than stripping it (`routes/pairing.py`);
  2. ordinary requests narrow it away: `effective(issued, policy, raised=∅)`;
  3. the operator mints a raise at the keyboard;
  4. during the window the device reaches `manage_monitor`, which is gated on
     EXECUTE and therefore now passes;
  5. `handle_manage_monitor` has no other guard -- verified, it goes straight
     to `event_monitoring.create_monitor`;
  6. the raise expires;
  7. the row still fires, with `LOCAL_GRANTS` and `LOCAL_PRINCIPAL`.

Both directions are pinned here. The refusal matters, but so does the
permission: a fix that simply refused every install would pass every test
about step 7 while breaking the operator's own keyboard. Live-test the answer,
not the refusal.

Run with:  py -3.11 -m pytest tests/test_raise_cannot_outlive_itself.py -v
"""
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.core.capabilities import Capability  # noqa: E402
from assistant.core.intent_capabilities import (  # noqa: E402
    PERSISTS_AUTHORITY, REQUIRED_CAPABILITY, TRANSIENT_AUTHORITY,
)

# What the tunnel ceilings actually carry (io/api/policy.py). EXECUTE and
# SYSTEM_CONTROL are the two they omit, and the two `tailnet` can raise.
TUNNEL_CEILING = frozenset({
    Capability.OBSERVE, Capability.RECALL, Capability.CHAT_SEND,
    Capability.SCREEN, Capability.FILES,
})


@pytest.fixture()
def turn():
    """Install and tear down one turn's authority, the way main.py does."""
    import assistant.actions as actions

    def _install(*, grants, principal="device:phone", issued=None,
                 raisable=None, ceiling=None):
        ctx = None
        if issued is not None:
            ctx = actions.RaiseContext(
                issued=issued,
                raisable=frozenset() if raisable is None else raisable,
                ceiling=frozenset() if ceiling is None else ceiling,
            )
        # Order matters in production (principal, then raise context, then
        # grants -- see main.py) and is kept here so a test can never pass
        # against an ordering the pipeline does not use.
        actions.set_principal(principal)
        actions.set_raise_context(ctx)
        actions.set_grants(grants)

    yield _install

    # Reset by assignment, not by token. `pytest-asyncio` runs the coroutine
    # in its own context, so a token minted in this synchronous fixture cannot
    # be reset from there -- `Token was created in a different Context`. The
    # values still propagate INTO the coroutine, which is why the async tests
    # see them; only the token is context-bound.
    #
    # Restoring the documented fail-closed defaults rather than whatever was
    # there before: `None` grants refuse everything and `None` principal owns
    # nothing, so a leak out of this fixture disables the next test rather
    # than silently authorising it.
    actions.set_grants(frozenset())
    actions.current_grants.set(None)
    actions.set_principal(None)
    actions.set_raise_context(None)


# ─── the hole ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("intent", sorted(PERSISTS_AUTHORITY))
def test_a_raised_capability_cannot_install_a_durable_trigger(turn, intent):
    """The whole point. The device holds EXECUTE *now*, only because a raise
    lifted the ceiling -- so the install is refused."""
    import assistant.actions as actions

    required = REQUIRED_CAPABILITY.get(intent, Capability.EXECUTE)
    turn(
        # What the raise makes effective this instant.
        grants=TUNNEL_CEILING | {required},
        issued=TUNNEL_CEILING | {required},   # the vault holds it
        raisable=frozenset({Capability.EXECUTE, Capability.SYSTEM_CONTROL}),
        ceiling=TUNNEL_CEILING,               # but the transport does not
    )

    assert actions.durable_capability_refusal(required) is not None, (
        f"{intent} would be installed on authority that expires. The row it "
        f"writes fires with LOCAL_GRANTS forever."
    )


@pytest.mark.parametrize("intent", sorted(PERSISTS_AUTHORITY))
def test_the_same_capability_held_durably_still_installs(turn, intent):
    """The other half, and the mutation-catcher: a fix that refused
    unconditionally would pass every test above and break the operator."""
    import assistant.actions as actions

    required = REQUIRED_CAPABILITY.get(intent, Capability.EXECUTE)
    turn(
        grants=TUNNEL_CEILING | {required},
        issued=TUNNEL_CEILING | {required},
        raisable=frozenset(),
        ceiling=TUNNEL_CEILING | {required},   # the transport carries it
    )

    assert actions.durable_capability_refusal(required) is None, (
        f"{intent} was refused to a caller that holds {required.value} with "
        f"no raise in force. This refuses the operator's own keyboard."
    )


def test_the_local_caller_is_unaffected(turn):
    """`LOCAL_RAISE_CONTEXT` holds everything and can raise nothing, so the
    two predicates must agree at the keyboard. If they ever disagree there,
    the local path has been made worse by a remote-only fix."""
    import assistant.actions as actions

    turn(grants=actions.LOCAL_GRANTS, principal=actions.LOCAL_PRINCIPAL,
         issued=actions.LOCAL_GRANTS, raisable=frozenset(),
         ceiling=actions.LOCAL_GRANTS)

    for cap in Capability:
        assert actions.capability_refusal(cap) is None, f"local lost {cap}"
        assert actions.durable_capability_refusal(cap) is None, (
            f"the durability gate refused {cap.value} at the keyboard"
        )


def test_a_missing_raise_context_refuses_rather_than_guessing(turn):
    """`None` means nobody said. `_refuse` degrades to a vaguer sentence for
    that, because it is only choosing wording. This gate is choosing whether
    to permit a durable effect, so it fails closed instead."""
    import assistant.actions as actions

    turn(grants=actions.LOCAL_GRANTS, issued=None)   # no context installed

    assert actions.durable_capability_refusal(Capability.EXECUTE) is not None, (
        "a turn with no raise context was allowed to install a durable "
        "trigger -- durable authority was assumed rather than established"
    )


def test_the_refusal_says_a_raise_will_not_help(turn):
    """The operator minted the raise and needs to know why it did not carry.
    A generic 'this device doesn't have it' would send them to mint another."""
    import assistant.actions as actions

    turn(grants=TUNNEL_CEILING | {Capability.EXECUTE},
         issued=TUNNEL_CEILING | {Capability.EXECUTE},
         raisable=frozenset({Capability.EXECUTE}),
         ceiling=TUNNEL_CEILING)

    msg = actions.durable_capability_refusal(Capability.EXECUTE)
    assert msg and "keyboard" in msg.lower(), (
        f"the refusal does not point anywhere useful: {msg!r}"
    )
    assert len(msg) < 120, f"spoken aloud, so under 120 chars: {len(msg)}"


# ─── the classification is exhaustive ───────────────────────────────────────

def test_every_intent_is_classified_exactly_once():
    """No default is safe here: 'persists' would refuse code_executor to a
    raised device and destroy the raise's purpose; 'transient' fails open for
    a future intent that installs something. So there is no default, and this
    is what closes the second direction."""
    from assistant import config

    intents = set(config.INTENTS)
    assert intents, "config.INTENTS is empty -- this test would pass vacuously"

    unclassified = intents - PERSISTS_AUTHORITY - TRANSIENT_AUTHORITY
    assert not unclassified, (
        f"intents in neither durability set: {sorted(unclassified)}. Decide "
        f"whether each installs something that runs after the turn ends."
    )

    both = PERSISTS_AUTHORITY & TRANSIENT_AUTHORITY
    assert not both, f"intents in both sets: {sorted(both)}"

    strays = (PERSISTS_AUTHORITY | TRANSIENT_AUTHORITY) - intents
    assert not strays, (
        f"durability sets name things that are not intents: {sorted(strays)}"
    )


def test_the_intents_that_install_something_are_all_execute_gated():
    """Durability is a second gate, not a replacement. Anything that installs
    a trigger must already cost EXECUTE, or the durability check is guarding a
    door that was open."""
    for intent in PERSISTS_AUTHORITY:
        assert REQUIRED_CAPABILITY.get(intent) == Capability.EXECUTE, (
            f"{intent} installs something durable but does not require "
            f"EXECUTE: {REQUIRED_CAPABILITY.get(intent)}"
        )


def test_code_executor_is_transient_deliberately():
    """Stated as a test because it looks like an omission and is not.
    `code_executor` can do anything a shell can inside the raise window, and
    no in-process check changes that -- which is what granting EXECUTE means.
    What the durability gate stops is TENKA's *own* machinery being used to
    make the window permanent. Classifying it as persistent would refuse the
    one thing a raise exists to permit."""
    assert "code_executor" in TRANSIENT_AUTHORITY
    assert "code_executor" not in PERSISTS_AUTHORITY


# ─── the audit column ───────────────────────────────────────────────────────

def test_the_tables_that_fire_later_record_who_installed_them(tmp_path):
    """Schema v21. The gate is at dispatch, not at fire time -- this column is
    the record, so the audit trail stops saying the machine did it to itself."""
    from assistant.storage.db import Database

    db = Database(tmp_path / "t.db")
    try:
        assert db._get_version() >= 21
        for table in ("event_monitors", "schedules",
                      "user_procedures", "user_shortcuts"):
            cols = {r[1]: r for r in db._conn.execute(
                f"PRAGMA table_info({table})")}
            assert "installed_by" in cols, f"{table} has no installed_by"
            row = cols["installed_by"]
            assert row[3] == 1, f"{table}.installed_by is nullable"
            assert row[4] == "'local'", (
                f"{table}.installed_by default is {row[4]!r}; existing rows "
                f"must read as keyboard-installed without a backfill"
            )
    finally:
        db._conn.close()


# ─── the gate is actually wired ──────────────────────────────────────────────
#
# Everything above calls `durable_capability_refusal` directly. A perfect
# predicate that nothing calls refuses nothing, and deleting the hook in
# `execute()` left all seventeen of those tests green -- measured, which is why
# these exist. These go through dispatch.

@pytest.mark.asyncio
@pytest.mark.parametrize("intent", sorted(PERSISTS_AUTHORITY))
async def test_dispatch_refuses_a_raised_install_before_the_handler_runs(
        turn, intent, monkeypatch):
    """The refusal has to happen at the choke point, before
    `tool_registry.get(intent)` -- so the handler is never entered and no row
    is written. Asserted by making the handler explode if reached."""
    import assistant.actions as actions

    required = REQUIRED_CAPABILITY.get(intent, Capability.EXECUTE)
    turn(grants=TUNNEL_CEILING | {required},
         issued=TUNNEL_CEILING | {required},
         raisable=frozenset({Capability.EXECUTE, Capability.SYSTEM_CONTROL}),
         ceiling=TUNNEL_CEILING)

    def _boom(*a, **k):
        raise AssertionError(
            f"{intent}'s handler ran. The durability gate is not wired into "
            f"execute(), or it sits after handler resolution."
        )

    monkeypatch.setattr(actions.tool_registry, "get", lambda _i: _boom)

    result = await actions.execute(intent, {"goal": "watch for something"})
    assert isinstance(result, str) and result, "dispatch returned nothing"
    assert "keyboard" in result.lower(), (
        f"dispatch did not return the durability refusal: {result!r}"
    )


@pytest.mark.asyncio
async def test_dispatch_still_runs_a_transient_intent_under_a_raise(
        turn, monkeypatch):
    """The gate must not become a blanket refusal for raised callers. Running
    code is exactly what a raise exists to permit."""
    import assistant.actions as actions

    turn(grants=TUNNEL_CEILING | {Capability.EXECUTE},
         issued=TUNNEL_CEILING | {Capability.EXECUTE},
         raisable=frozenset({Capability.EXECUTE}),
         ceiling=TUNNEL_CEILING)

    ran = []
    monkeypatch.setattr(actions.tool_registry, "get",
                        lambda _i: (lambda *a, **k: ran.append(1) or "done"))

    result = await actions.execute("code_executor", {"goal": "print(1)"})
    assert ran, (
        f"code_executor was refused to a raised caller: {result!r}. The raise "
        f"exists to permit exactly this."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("intent", sorted(PERSISTS_AUTHORITY))
async def test_dispatch_allows_a_durable_install(turn, intent, monkeypatch):
    """And the operator's own path still works, through dispatch."""
    import assistant.actions as actions

    required = REQUIRED_CAPABILITY.get(intent, Capability.EXECUTE)
    turn(grants=actions.LOCAL_GRANTS, principal=actions.LOCAL_PRINCIPAL,
         issued=actions.LOCAL_GRANTS, raisable=frozenset(),
         ceiling=actions.LOCAL_GRANTS)

    ran = []
    monkeypatch.setattr(actions.tool_registry, "get",
                        lambda _i: (lambda *a, **k: ran.append(1) or "ok"))

    await actions.execute(intent, {"goal": "every morning"})
    assert ran, f"{intent} was refused at the keyboard, holding {required.value}"
