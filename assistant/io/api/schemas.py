# assistant/io/api/schemas.py
"""Wire shapes. Every response is {data, meta}; a sealed form is reserved.

Milestone 6 adds application-layer encryption because a Cloudflare tunnel
terminates TLS at their edge. The `sealed` field exists now and is rejected
now, so bodies never have to be reshaped for it.

Layering: io/api — core + config only.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Meta(BaseModel):
    request_id: str = ""
    generated_at: str = ""


class Envelope(BaseModel):
    """Deliberately not generic.

    `Envelope[T]` as a FastAPI response annotation forces a distinct model per
    payload shape and buys nothing here: the payload types already live in
    runtime.py, and the routes build plain dicts from them. `data: Any` keeps
    one envelope across every route, which is what the client parses against.
    """

    data: Any
    meta: Meta = Field(default_factory=Meta)


class SealedEnvelope(BaseModel):
    """Reserved for Milestone 6. Accepted by the parser, refused by the app."""

    sealed: str
    nonce: str


class ErrorBody(BaseModel):
    error: str
    detail: str = ""


class ChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8_000)


class SettingsPatch(BaseModel):
    changes: dict[str, Any] = Field(default_factory=dict)


class PersonalityPatch(BaseModel):
    base: str = Field(min_length=1, max_length=64)


class RenameRequest(BaseModel):
    path: str = Field(min_length=1, max_length=1_024)
    new_name: str = Field(min_length=1, max_length=255)


class DeleteRequest(BaseModel):
    path: str = Field(min_length=1, max_length=1_024)


class RestoreRequest(BaseModel):
    recovery_phrase: str = Field(min_length=1, max_length=512)
