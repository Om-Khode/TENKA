# assistant/io/api/routes/commands.py
"""Run an OS-level capability. The grant is declared by the command itself, so
adding one to the catalogue cannot accidentally widen what a device may do."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ..schemas import Envelope
from ..security import authenticate, require
from ..vault import Capability

router = APIRouter()


@router.get("/commands")
async def list_commands(request: Request,
                        _=Depends(require(Capability.CHAT))) -> Envelope:
    catalogue = await request.app.state.runtime.commands.catalogue()
    return Envelope(data={"commands": [
        {
            "command_id": command.command_id,
            "label": command.label,
            "description": command.description,
            "destructive": command.destructive,
            "required_grant": command.required_grant,
        }
        for command in catalogue
    ]})


@router.post("/commands/{command_id}/run")
async def run_command(command_id: str, request: Request,
                      device=Depends(authenticate)) -> Envelope:
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
    return Envelope(data={"command_id": command_id, "message": outcome.message})
