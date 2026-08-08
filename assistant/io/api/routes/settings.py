# assistant/io/api/routes/settings.py
"""The settings registry and the personality state, as data."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ..schemas import Envelope, PersonalityPatch, SettingsPatch
from ..security import require
from ..vault import Capability

router = APIRouter()


@router.get("/settings")
async def list_settings(request: Request,
                        _=Depends(require(Capability.CHAT))) -> Envelope:
    rows = await request.app.state.runtime.settings.all()
    return Envelope(data={"rows": [
        {
            "key": row.key,
            "group": row.group,
            "description": row.description,
            "kind": row.kind,
            "value": row.value,
            "default": row.default,
            "needsRestart": row.needs_restart,
            "source": row.source,
            "options": row.options,
        }
        for row in rows
    ]})


@router.get("/personality")
async def get_personality(request: Request,
                          _=Depends(require(Capability.CHAT))) -> Envelope:
    state = await request.app.state.runtime.personality.state()
    return Envelope(data={
        "base": state.base,
        "available": state.available,
        "traits": state.traits,
        "sample_line": state.sample_line,
    })


@router.patch("/settings")
async def save_settings(body: SettingsPatch, request: Request,
                        _=Depends(require(Capability.CHAT))) -> Envelope:
    outcome = await request.app.state.runtime.settings.save(body.changes)
    return Envelope(data={
        "saved": outcome.saved,
        "rejected": outcome.rejected,
        "restart_required": outcome.restart_required,
    })


@router.patch("/personality")
async def set_personality(body: PersonalityPatch, request: Request,
                          _=Depends(require(Capability.CHAT))) -> Envelope:
    state = await request.app.state.runtime.personality.set_base(body.base)
    return Envelope(data={
        "base": state.base,
        "available": state.available,
        "traits": state.traits,
        "sample_line": state.sample_line,
    })
