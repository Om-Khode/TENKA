# assistant/io/api/schemas.py
"""Wire shapes. Every response is {data, meta}; a sealed form is reserved.

Milestone 6 adds application-layer encryption because a Cloudflare tunnel
terminates TLS at their edge. The `sealed` field exists now and is rejected
now, so bodies never have to be reshaped for it.

Layering: io/api — core + config only.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


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


# A settings page realistically edits a handful of rows at once and every key
# is a short snake_case identifier; these bounds are generous for that and
# absurd for a body built to exhaust memory or a database column.
MAX_SETTINGS_KEYS = 200
MAX_SETTINGS_KEY_LENGTH = 200
MAX_SETTINGS_STRING_VALUE_LENGTH = 4_096


class SettingsPatch(BaseModel):
    changes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("changes")
    @classmethod
    def _bound_changes(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > MAX_SETTINGS_KEYS:
            raise ValueError(
                f"too many settings in one patch: {len(value)} > {MAX_SETTINGS_KEYS}"
            )
        for key, item in value.items():
            if len(key) > MAX_SETTINGS_KEY_LENGTH:
                raise ValueError(
                    f"setting key too long: {len(key)} > {MAX_SETTINGS_KEY_LENGTH} chars"
                )
            if isinstance(item, str) and len(item) > MAX_SETTINGS_STRING_VALUE_LENGTH:
                raise ValueError(
                    f"value for {key!r} too long: {len(item)} > "
                    f"{MAX_SETTINGS_STRING_VALUE_LENGTH} chars"
                )
        return value


class PersonalityPatch(BaseModel):
    base: str = Field(min_length=1, max_length=64)


class RenameRequest(BaseModel):
    path: str = Field(min_length=1, max_length=1_024)
    new_name: str = Field(min_length=1, max_length=255)


class DeleteRequest(BaseModel):
    path: str = Field(min_length=1, max_length=1_024)


class RestoreRequest(BaseModel):
    recovery_phrase: str = Field(min_length=1, max_length=512)
