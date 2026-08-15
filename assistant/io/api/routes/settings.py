# assistant/io/api/routes/settings.py
"""The settings registry and the personality state, as data."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ..payloads import PersonalityPayload, SaveOutcomePayload, SettingRowPayload, SettingsPayload
from ..schemas import Envelope, PersonalityPatch, SettingsPatch
from ..security import require
from ..vault import Capability

router = APIRouter()


def _personality_body(state) -> PersonalityPayload:
    return PersonalityPayload(
        base=state.base,
        available=state.available,
        traits=state.traits,
        sample_line=state.sample_line,
    )


@router.get("/settings")
async def list_settings(request: Request,
                        _=Depends(require(Capability.CHAT))) -> Envelope[SettingsPayload]:
    rows = await request.app.state.runtime.settings.all()
    return Envelope(data=SettingsPayload(rows=[
        SettingRowPayload(
            key=row.key,
            group=row.group,
            description=row.description,
            kind=row.kind,
            value=row.value,
            default=row.default,
            needs_restart=row.needs_restart,
            source=row.source,
            options=row.options,
        )
        for row in rows
    ]))


@router.get("/personality")
async def get_personality(request: Request,
                          _=Depends(require(Capability.CHAT))) -> Envelope[PersonalityPayload]:
    state = await request.app.state.runtime.personality.state()
    return Envelope(data=_personality_body(state))


@router.patch("/settings")
async def save_settings(body: SettingsPatch, request: Request,
                        # SYSTEM_CONTROL, not CHAT: the same reasoning that
                        # moved forget-all off of chat applies here, harder --
                        # a phone paired only for conversation must not be
                        # able to rewrite the daemon's own CORS allow-list,
                        # switch the camera on, or flip any other setting the
                        # assistant trusts nobody but its owner to touch.
                        # Reading settings stays on chat; only writing them
                        # is gated up.
                        _=Depends(require(Capability.SYSTEM_CONTROL))) -> Envelope[SaveOutcomePayload]:
    outcome = await request.app.state.runtime.settings.save(body.changes)
    return Envelope(data=SaveOutcomePayload(
        saved=outcome.saved,
        rejected=outcome.rejected,
        restart_required=outcome.restart_required,
    ))


@router.patch("/personality")
async def set_personality(body: PersonalityPatch, request: Request,
                          # SYSTEM_CONTROL, not CHAT, for the same reason
                          # PATCH /settings above is: the personality base is
                          # a stored, restart-surviving configuration value,
                          # and switching it changes how she answers *every*
                          # caller on this machine, not just the one that
                          # asked. It is a setting that happens to have its
                          # own route, not a conversational act -- which is
                          # what separates it from CHAT_SEND. Reading the
                          # personality stays on CHAT.
                          _=Depends(require(Capability.SYSTEM_CONTROL))) -> Envelope[PersonalityPayload]:
    # switch_personality() reports an unknown base as a return string, not an
    # exception -- left unchecked, that string was silently discarded and the
    # route answered 200 with the *previous* (unchanged) state, giving a
    # caller no way to tell "switched" from "rejected, nothing happened".
    # Validated against the runtime's own `available` list, the same set the
    # client renders as choices, rather than duplicating a hardcoded
    # personality catalogue here -- THE rule again: no app/persona names in
    # this layer.
    current = await request.app.state.runtime.personality.state()
    if body.base not in current.available:
        raise HTTPException(status_code=400, detail="unknown personality")
    state = await request.app.state.runtime.personality.set_base(body.base)
    return Envelope(data=_personality_body(state))


@router.post("/personality/reset")
async def reset_personality(request: Request,
                            # SYSTEM_CONTROL, matching PATCH above: this is
                            # the *destructive* half of the same setting. It
                            # discards the trait state her reflection loop
                            # accumulated over every past conversation, and
                            # there is no undo. A route that can only destroy
                            # cannot sit below the one that merely changes.
                            _=Depends(require(Capability.SYSTEM_CONTROL))) -> Envelope[PersonalityPayload]:
    state = await request.app.state.runtime.personality.reset()
    return Envelope(data=_personality_body(state))
