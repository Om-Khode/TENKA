# assistant/io/api/routes/status.py
"""Liveness and identity — authenticated like everything else."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ..schemas import Envelope
from ..security import require
from ..vault import Capability

router = APIRouter()


@router.get("/status")
async def get_status(request: Request, _=Depends(require(Capability.CHAT))) -> Envelope:
    info = await request.app.state.runtime.system.status()
    return Envelope(data={
        "assistant_name": info.assistant_name,
        "version": info.version,
        "active_model": info.active_model,
        "personality": info.personality,
        "busy": info.busy,
    })
