# assistant/io/api/schemas.py
"""Wire shapes. Every response is {data, meta}; a sealed form is reserved.

Milestone 6 adds application-layer encryption because a Cloudflare tunnel
terminates TLS at their edge. The `sealed` field exists now and is rejected
now, so bodies never have to be reshaped for it.

Layering: io/api — core + config only.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .context import request_id_var
from .runtime import SettingValue


class Meta(BaseModel):
    # default_factory, not a fixed "" -- a Meta built with no arguments (the
    # common case: every route does `Envelope(data=...)` and leaves `meta` on
    # its default) used to ship two permanently empty strings on every single
    # response. request_id_var is set by app.py's `audit_and_tag` middleware
    # before the router ever runs, so it is already populated by the time a
    # route constructs its Envelope.
    #
    # Aliased to camelCase, like every other wire key in this schema: Studio
    # generates its TypeScript types from app.openapi(), and a mix of
    # request_id/requestId across the same contract is exactly the
    # inconsistency the wire-naming fix wave closed everywhere else.
    request_id: str = Field(
        default_factory=lambda: request_id_var.get(), alias="requestId"
    )
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        alias="generatedAt",
    )

    model_config = ConfigDict(populate_by_name=True)


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
    # extra="forbid": a Milestone-6 client sending {text, sealed, nonce} (a
    # sealed-envelope body against a route that doesn't understand sealing
    # yet) must be refused outright, not have "sealed"/"nonce" silently
    # dropped while "text" is accepted -- half-honouring a body shaped for a
    # different protocol version is worse than rejecting it.
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=8_000)


# A settings page realistically edits a handful of rows at once and every key
# is a short snake_case identifier; these bounds are generous for that and
# absurd for a body built to exhaust memory or a database column. The value
# type is SettingValue (str | int | float | bool) -- the same union
# SettingsRuntime.save() actually stores -- rather than Any, so Pydantic
# itself rejects a list, a dict, or None with a 422 before any length check
# below ever runs; a nested structure or null can no longer arrive at all.
MAX_SETTINGS_KEYS = 200
MAX_SETTINGS_KEY_LENGTH = 200
MAX_SETTINGS_STRING_VALUE_LENGTH = 4_096
# Nesting is now impossible, but a bare int is still unbounded in principle
# (Python ints have no width limit). Bounding the magnitude also bounds the
# digit count Pydantic/json has to convert -- 10**12 is a terabyte in bytes,
# comfortably past anything a real setting (a timer, a count, a byte size)
# stores, and 13 digits nowhere near where int-to-str conversion cost would
# start to matter.
MAX_SETTINGS_INT_MAGNITUDE = 1_000_000_000_000


class SettingsPatch(BaseModel):
    changes: dict[str, SettingValue] = Field(default_factory=dict)

    @field_validator("changes")
    @classmethod
    def _bound_changes(cls, value: dict[str, SettingValue]) -> dict[str, SettingValue]:
        # Every `raise ValueError` below embeds the raw settings `key` in its
        # message. That is safe only because the 422 handler in app.py drops
        # Pydantic's `msg` app-wide (it rebuilds every validation error body
        # from `loc`/`type` alone) -- if a future change ever puts `msg` back
        # for debuggability, these messages start leaking key names into a
        # response again. Bound the value, not what you say about it, if
        # that coupling ever gets undone.
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
            if (
                isinstance(item, int)
                and not isinstance(item, bool)
                and abs(item) > MAX_SETTINGS_INT_MAGNITUDE
            ):
                raise ValueError(
                    f"value for {key!r} out of range: magnitude exceeds "
                    f"{MAX_SETTINGS_INT_MAGNITUDE}"
                )
            # A non-finite float is the risk here, not its magnitude: `inf`
            # or `nan` reaching the settings store or a UI that renders it
            # (a slider, a percentage) is a different failure mode than an
            # oversized-but-ordinary number, and no magnitude bound catches
            # it -- `float("inf") > MAX` is true, but so is `float("nan") >
            # MAX` being *false*, silently passing a magnitude check that
            # was never meant to guard against non-finite values at all.
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError(f"value for {key!r} must be finite")
        return value


class PersonalityPatch(BaseModel):
    base: str = Field(min_length=1, max_length=64)


class RenameRequest(BaseModel):
    # No populate_by_name: openapi() advertises only "newName", and nothing
    # outside this repo has ever consumed "new_name" (openapi.json is
    # gitignored -- nothing generated from the old shape has shipped). A
    # silent second accepted spelling is complexity with no consumer.
    path: str = Field(min_length=1, max_length=1_024)
    new_name: str = Field(min_length=1, max_length=255, alias="newName")


class DeleteRequest(BaseModel):
    path: str = Field(min_length=1, max_length=1_024)


class RestoreRequest(BaseModel):
    # No populate_by_name -- see RenameRequest's comment above. Doubly true
    # on a recovery-phrase field: a second, undocumented accepted spelling
    # is not a thing a security-relevant body should carry for free.
    recovery_phrase: str = Field(
        min_length=1, max_length=512, alias="recoveryPhrase"
    )
