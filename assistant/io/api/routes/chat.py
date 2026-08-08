# assistant/io/api/routes/chat.py
"""Chat is another input source, not another brain.

POST /v1/chat hands text to the same pipeline voice uses. Tokens come back over
the WebSocket, not in this response, so a request never blocks on a full turn.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..schemas import ChatRequest, Envelope
from ..security import require
from ..vault import Capability

router = APIRouter()


# ─── Sending a turn ──────────────────────────────────────────────────────
@router.post("/chat", status_code=status.HTTP_202_ACCEPTED)
async def send_chat(body: ChatRequest, request: Request,
                    _=Depends(require(Capability.CHAT))) -> Envelope:
    ref = await request.app.state.runtime.chat.send(body.text)
    if not ref.accepted:
        # Deliberately generic: a caller that cannot authenticate any further
        # than "holds a CHAT token" should not learn *what* she is doing --
        # only that she isn't free. `ref.reason` is never interpolated here.
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="busy")
    return Envelope(data={"turnId": ref.turn_id,
                          "conversationId": ref.conversation_id})


# ─── Conversation history ────────────────────────────────────────────────
@router.get("/chat/conversations")
async def list_conversations(request: Request,
                             _=Depends(require(Capability.CHAT))) -> Envelope:
    conversations = await request.app.state.runtime.chat.conversations()
    return Envelope(data={"conversations": [
        {
            "conversationId": c.conversation_id,
            "title": c.title,
            "updatedAt": c.updated_at,
            "messageCount": c.message_count,
        }
        for c in conversations
    ]})


@router.get("/chat/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, request: Request,
                           _=Depends(require(Capability.CHAT))) -> Envelope:
    detail = await request.app.state.runtime.chat.conversation(conversation_id)
    if detail is None:
        raise HTTPException(status_code=404)  # detail is dead on a 404 -- see app.py's handler
    return Envelope(data={
        "conversationId": detail.conversation_id,
        "title": detail.title,
        "messages": [
            {
                "messageId": m.message_id,
                "role": m.role,
                "text": m.text,
                "createdAt": m.created_at,
                "intent": m.intent,
            }
            for m in detail.messages
        ],
    })


# ─── Abort ────────────────────────────────────────────────────────────────
@router.post("/abort")
async def abort(request: Request, _=Depends(require(Capability.CHAT))) -> Envelope:
    stopped = await request.app.state.runtime.chat.abort()
    return Envelope(data={"aborted": stopped})
