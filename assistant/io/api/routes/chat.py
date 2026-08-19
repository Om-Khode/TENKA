# assistant/io/api/routes/chat.py
"""Chat is another input source, not another brain.

POST /v1/chat hands text to the same pipeline voice uses. Tokens come back over
the WebSocket, not in this response, so a request never blocks on a full turn.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..payloads import (
    AbortPayload, ChatMessagePayload, ChatSendPayload, ConversationDetailPayload,
    ConversationRefPayload, ConversationsPayload,
)
from ..schemas import ChatRequest, Envelope
from ..security import require
from ..vault import Capability, Device

router = APIRouter()


# ─── Sending a turn ──────────────────────────────────────────────────────
@router.post("/chat", status_code=status.HTTP_202_ACCEPTED)
async def send_chat(body: ChatRequest, request: Request,
                    device: Device = Depends(require(Capability.CHAT_SEND))
                    ) -> Envelope[ChatSendPayload]:
    # `device.grants` is already `effective(issued, listener ceiling)` --
    # `authenticate()` narrows before `require()` hands the Device back -- so
    # this is the intersection the ceiling exists to enforce, not the device's
    # issued set. It is passed rather than recomputed: a second computation is
    # a second chance to compute it differently.
    #
    # CHAT_SEND alone used to reach every intent through this route, because
    # the pipeline behind it applied no further check. It is the entry
    # permission now, and what the turn may actually *do* travels with it.
    #
    # The principal is built here for the same reason the grants are passed
    # here: this is the only place that knows *which* device authenticated,
    # and by the time the turn runs the request is gone. The `device:` prefix
    # is added on this side, never taken from the caller, so a device cannot
    # name itself `"local"` and inherit the operator's own confirmations.
    # `device_id` is the vault's identifier for the pairing, not anything the
    # request supplied. See core/principal.py and KI-13.
    # `issued` and `raisable` travel alongside the effective grants so a
    # refusal deep in the pipeline can say whether a raise at the keyboard
    # would fix it -- `authenticate()` already stashed the first on
    # `request.state` and the second is the listener's own fixed policy, so
    # this is two more already-known facts, not a new read. Neither is new
    # information to the device that owns it: both are visible on its own
    # `GET /v1/session` response already.
    ref = await request.app.state.runtime.chat.send(
        body.text, device.grants, f"device:{device.device_id}",
        issued=request.state.issued_grants, raisable=request.state.policy.raisable)
    if not ref.accepted:
        # Deliberately generic: a caller that cannot authenticate any further
        # than "holds a read token" should not learn *what* she is doing --
        # only that she isn't free. `ref.reason` is never interpolated here.
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="busy")
    return Envelope(data=ChatSendPayload(turn_id=ref.turn_id,
                                         conversation_id=ref.conversation_id))


# ─── Conversation history ────────────────────────────────────────────────
# RECALL, not OBSERVE. A transcript is what she was *told*, not what she is
# doing: it holds whatever the user typed or said, and -- because `read_screen`
# and `camera_look` are intents like any other -- her description of what was
# on the screen or in front of the camera. Watching her work must not carry it.
@router.get("/chat/conversations")
async def list_conversations(request: Request,
                             _=Depends(require(Capability.RECALL))) -> Envelope[ConversationsPayload]:
    conversations = await request.app.state.runtime.chat.conversations()
    return Envelope(data=ConversationsPayload(conversations=[
        ConversationRefPayload(
            conversation_id=c.conversation_id,
            title=c.title,
            updated_at=c.updated_at,
            message_count=c.message_count,
        )
        for c in conversations
    ]))


@router.get("/chat/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, request: Request,
                           _=Depends(require(Capability.RECALL))) -> Envelope[ConversationDetailPayload]:
    detail = await request.app.state.runtime.chat.conversation(conversation_id)
    if detail is None:
        raise HTTPException(status_code=404)  # detail is dead on a 404 -- see app.py's handler
    return Envelope(data=ConversationDetailPayload(
        conversation_id=detail.conversation_id,
        title=detail.title,
        messages=[
            ChatMessagePayload(
                message_id=m.message_id,
                role=m.role,
                text=m.text,
                created_at=m.created_at,
                intent=m.intent,
            )
            for m in detail.messages
        ],
    ))


# ─── Abort ────────────────────────────────────────────────────────────────
@router.post("/abort")
async def abort(request: Request,
                _=Depends(require(Capability.CHAT_SEND))) -> Envelope[AbortPayload]:
    stopped = await request.app.state.runtime.chat.abort()
    return Envelope(data=AbortPayload(aborted=stopped))
