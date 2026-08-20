# assistant/io/api/routes/transports.py
"""Which doors exist, opening one, and closing one.

Loopback only, the same gate `routes/devices.py` sits behind and for the same
reason: managing which transports run is the same class of thing as managing
which devices hold a credential, and both stay at the keyboard. `policy.admin`
is `True` for `local` alone, so all three routes here are unreachable from any
tunnel -- a compromised remote session cannot open a fourth door for itself,
nor close the one it is using.

`transport_registry.names()` is the only source of truth for what a name may
be: never `POLICIES` directly, and never a route-level `if name == ...`. Both
would reintroduce KI-17 through this route rather than a stray tunnel --
`POLICIES` has a `"local"` key with no adapter behind it, and
`transport_registry.register` already refuses to let anything claim that name
(`transports/__init__.py`). Checking membership in the *registry* is what
makes `"local"` a 404 here, the same way an unregistered name is, rather than
reaching `TransportManager.start`'s own (409) refusal of it.

Layering: io/api -- core + config only.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..events import notify_invalidate
from ..payloads import TransportPayload, TransportsPayload
from ..policy import POLICIES
from ..schemas import Envelope
from ..security import require_admin
from ..transports import transport_registry
from ..transports.manager import TransportError
from ..vault import Capability, Device

router = APIRouter()


def _manager(request: Request):
    """The live transport manager, or `None` when nothing wired one.

    `app.state.transports` is populated by `create_app`'s caller in a sibling
    task (Task 9's `TransportManager`). Read defensively, the same shape
    `routes/devices.py` already uses for `app.state.raises`: an app built
    before that wiring lands has no attribute at all, and "nothing running,
    nothing startable" is the correct fail-closed answer either way.
    """
    return getattr(request.app.state, "transports", None)


def _row(name: str, session) -> TransportPayload:
    policy = POLICIES[name]
    return TransportPayload(
        name=name,
        running=session is not None,
        url=session.url if session is not None else None,
        ceiling=sorted(c.value for c in policy.ceiling),
        raisable=sorted(c.value for c in policy.raisable),
        # Off the policy, never inferred from `name` -- see
        # `TransportPayload`'s docstring for why a client must not have to
        # guess this from a transport's name.
        pairable=policy.pairable,
    )


@router.get("/transports")
async def list_transports(
    request: Request,
    _: Device = Depends(require_admin(Capability.SYSTEM_CONTROL)),
) -> Envelope[TransportsPayload]:
    """Every registered transport, running or not, with the capability set
    it may ever carry and the capability set it may carry if temporarily
    raised -- so a caller can explain why a control is unavailable on a given
    transport."""
    # `ceiling`/`raisable` are read straight off `POLICIES` in `_row` rather
    # than hand-copied here, so this listing can never drift from what the
    # daemon actually enforces.
    manager = _manager(request)
    running = manager.running() if manager is not None else {}
    return Envelope(data=TransportsPayload(
        transports=[_row(name, running.get(name))
                    for name in transport_registry.names()]))


@router.post("/transports/{name}")
async def start_transport(
    name: str, request: Request,
    _: Device = Depends(require_admin(Capability.SYSTEM_CONTROL)),
) -> Envelope[TransportPayload]:
    """Start transport *name*. An unrecognised name -- including `"local"`,
    which is never a startable transport -- is a 404. A conflict that
    prevents the transport from starting is a 409, with a message describing
    what went wrong."""
    # `"local"` is never in the registry (it is the loopback listener itself,
    # with no adapter behind it -- see `transports/__init__.py`'s KI-17
    # guard), so checking membership here is what makes it a 404 the same way
    # an unregistered name is, rather than reaching `TransportManager.start`'s
    # own refusal of it.
    if name not in transport_registry.names():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    manager = _manager(request)
    if manager is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="try again")
    try:
        session = await manager.start(name)
    except TransportError as exc:
        # A preflight conflict, a tunnel that never announced a hostname, one
        # already running or already starting -- `TransportError`'s message is
        # already written to name a misconfiguration and never a hostname,
        # token or path, so it is passed straight through.
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=str(exc)) from exc
    # After `manager.start()` above has actually succeeded: the transports
    # list any admin viewer holds open (the pair dialog among them) just
    # went stale.
    notify_invalidate(getattr(request.app.state, "hub", None), "transports")
    return Envelope(data=_row(name, session))


@router.delete("/transports/{name}")
async def stop_transport(
    name: str, request: Request,
    _: Device = Depends(require_admin(Capability.SYSTEM_CONTROL)),
) -> Envelope[TransportPayload]:
    """Stop transport *name*. Not currently running -- including an unknown
    name -- is a 404: "nothing happened" has to be distinguishable from "it
    is gone now". A failure to confirm the stop actually took effect is a
    409, with a message describing what went wrong."""
    manager = _manager(request)
    running = manager.running() if manager is not None else {}
    if name not in running:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    try:
        await manager.stop(name)
    except TransportError as exc:
        # An unverified provider-side stop, most notably: the message is
        # already written to name a misconfiguration and never a hostname,
        # token or path, so it is passed straight through rather than
        # replaced with this route's own words.
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=str(exc)) from exc
    # After `manager.stop()` above has actually succeeded, same reasoning as
    # `start_transport`.
    notify_invalidate(getattr(request.app.state, "hub", None), "transports")
    return Envelope(data=_row(name, None))
