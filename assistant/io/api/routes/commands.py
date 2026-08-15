# assistant/io/api/routes/commands.py
"""Run an OS-level capability. The grant is declared by the command itself, so
adding one to the catalogue cannot accidentally widen what a device may do."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ..payloads import CommandDefPayload, CommandRunPayload, CommandsPayload
from ..schemas import Envelope
from ..security import authenticate, device_key, require
from ..vault import Capability

router = APIRouter()

# A command's own grant, not one fixed capability, decides *whether* a
# device may run it -- which is exactly why this route can't reuse
# security.throttle() (a `require(capability)` dependency baked in at
# declaration time). Bounding *how often*, regardless of which command, still
# matters: a SCREEN-granted device left running "screenshot" in a loop is the
# concrete case this closes, but the budget applies to every command_id
# alike, the same way the shared limiter applies to every route.
_RUN_MAX_PER_WINDOW = 20
_RUN_WINDOW_SECONDS = 60.0


# OBSERVE: the catalogue is a fixed list of what she *can* be asked to do,
# declared in code. It says nothing about what she has been asked or told, so
# it is observation of the assistant, not a read of stored data. Running one
# is gated by the command's own `required_grant` below, never by this.
@router.get("/commands")
async def list_commands(request: Request,
                        _=Depends(require(Capability.OBSERVE))) -> Envelope[CommandsPayload]:
    catalogue = await request.app.state.runtime.commands.catalogue()
    return Envelope(data=CommandsPayload(commands=[
        CommandDefPayload(
            command_id=command.command_id,
            label=command.label,
            description=command.description,
            destructive=command.destructive,
            required_grant=command.required_grant,
        )
        for command in catalogue
    ]))


@router.post("/commands/{command_id}/run")
async def run_command(command_id: str, request: Request,
                      device=Depends(authenticate)) -> Envelope[CommandRunPayload]:
    state = request.app.state.auth
    key = f"commands_run:{device_key(device)}"
    if not state.limiter.check(key, max_per_window=_RUN_MAX_PER_WINDOW,
                               window_seconds=_RUN_WINDOW_SECONDS):
        raise HTTPException(status_code=429, detail="too many requests")

    catalogue = await request.app.state.runtime.commands.catalogue()
    match = next((c for c in catalogue if c.command_id == command_id), None)
    if match is None:
        # `detail=` is dead on a 404 (app.py's status-code handler discards
        # it), so it is omitted here rather than left reading as live.
        raise HTTPException(status_code=404)

    try:
        needed = Capability(match.required_grant)
    except ValueError:
        raise HTTPException(status_code=500, detail="command declares an unknown grant")
    if needed not in device.grants:
        raise HTTPException(status_code=403, detail="capability not granted")

    outcome = await request.app.state.runtime.commands.run(command_id)
    if not outcome.ok:
        raise HTTPException(status_code=502, detail=outcome.message)
    return Envelope(data=CommandRunPayload(command_id=command_id, message=outcome.message))
