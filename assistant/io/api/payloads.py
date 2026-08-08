# assistant/io/api/payloads.py
"""One Pydantic model per response shape, so `Envelope[T]` has a `T`.

`schemas.py` is already the home of every *request* body (`ChatRequest`,
`SettingsPatch`, ...) and the envelope itself (`Envelope`, `Meta`). Response
payloads are a second, larger family -- one model per route, roughly thirty of
them once every nested list item (an `Entity`, a `FileEntry`, an
`EnrolledItem`, ...) gets its own model too -- and putting all of them into
schemas.py would bury the request side that already lives there under a much
bigger response side. Splitting by direction (request vs response) keeps each
file's job legible; this module is the response half.

Every model here is a wire-shape twin of a dataclass in `runtime.py`: same
fields, same nesting, but camelCased for the client and constructed from
Python's own snake_case names via `populate_by_name` -- a route builds
`EntityPayload(canonical_name=entity.canonical_name, ...)` from the runtime
dataclass's own attributes, and `CamelModel`'s alias generator handles the
rewrite to `canonicalName` on the way out. `by_alias=True` is FastAPI's
default for a response model, so nothing extra has to be set per-route for
that -- it only has to not be turned *off* (see `Envelope`'s docstring in
schemas.py for why `exclude_none` in particular must never be set).

Layering: io/api — core + config only.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue
from pydantic.alias_generators import to_camel

from .runtime import FileKind, SettingValue


class CamelModel(BaseModel):
    """Shared base for every payload: camelCase on the wire, snake_case in
    Python. `populate_by_name=True` so a route can construct one from a
    runtime dataclass's own (snake_case) attribute names without spelling out
    every alias by hand."""

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


# ─── Chat ────────────────────────────────────────────────────────────────
class ChatSendPayload(CamelModel):
    turn_id: str
    conversation_id: str


class ConversationRefPayload(CamelModel):
    conversation_id: str
    title: str
    updated_at: str
    message_count: int


class ConversationsPayload(CamelModel):
    conversations: list[ConversationRefPayload]


class ChatMessagePayload(CamelModel):
    message_id: str
    role: Literal["user", "assistant"]
    text: str
    created_at: str
    intent: str


class ConversationDetailPayload(CamelModel):
    conversation_id: str
    title: str
    messages: list[ChatMessagePayload]


class AbortPayload(CamelModel):
    aborted: bool


# ─── Memory ──────────────────────────────────────────────────────────────
class EntityPayload(CamelModel):
    id: int
    type: str
    canonical_name: str
    display_name: str
    # `JsonValue`, not `str`: a taught entity's properties are whatever
    # `_load_properties()` (studio_runtime.py) parsed out of a JSON column --
    # a number, a bool, a nested object, `null`, all legal. `dict[str, str]`
    # here used to make response validation 400 the *entire scope* the
    # moment one taught property held anything but a string (Finding 1,
    # 2026-08-08 review) -- ResponseValidationError subclasses ValueError,
    # so errors.py mapped it to a 400 that looked like a bad client request
    # for what was actually a server-side type mismatch.
    properties: dict[str, JsonValue]
    source: str
    confidence: float
    created_at: str
    updated_at: str
    source_turn_id: str | None


class FactPayload(CamelModel):
    id: int
    subject_id: int
    predicate: str
    object: str
    confidence: float
    source: str
    event_at: str | None
    invalid_at: str | None
    expires_at: str | None
    verified_at: str | None
    created_at: str
    source_turn_id: str | None


class RelationshipPayload(CamelModel):
    id: int
    from_id: int
    to_id: int
    type: str
    # Same reasoning as EntityPayload.properties above.
    properties: dict[str, JsonValue]
    confidence: float
    source: str
    source_turn_id: str | None


class KnowledgeGraphPayload(CamelModel):
    entities: list[EntityPayload]
    facts: list[FactPayload]
    relationships: list[RelationshipPayload]


class PreferenceChangePayload(CamelModel):
    value: str
    changed_at: str


class PreferenceRecordPayload(CamelModel):
    key: str
    value: str
    updated_at: str
    history: list[PreferenceChangePayload]


class PreferencesPayload(CamelModel):
    preferences: list[PreferenceRecordPayload]


class ProcedureRecordPayload(CamelModel):
    id: int
    name: str
    steps: list[str]
    taught_at: str
    run_count: int


class ProceduresPayload(CamelModel):
    procedures: list[ProcedureRecordPayload]


class ForgottenPayload(CamelModel):
    forgotten: str


class RemovedPayload(CamelModel):
    removed: int


# ─── Settings and personality ────────────────────────────────────────────
class SettingRowPayload(CamelModel):
    key: str
    group: str
    description: str
    kind: Literal["toggle", "slider", "select", "number", "text"]
    value: SettingValue
    default: SettingValue
    needs_restart: bool
    source: Literal["db", "env", "default"]
    options: list[str]


class SettingsPayload(CamelModel):
    rows: list[SettingRowPayload]


class SaveOutcomePayload(CamelModel):
    saved: list[str]
    rejected: dict[str, str]
    restart_required: list[str]


class PersonalityPayload(CamelModel):
    base: str
    available: list[str]
    traits: dict[str, float]
    sample_line: str


# ─── Files ───────────────────────────────────────────────────────────────
class RootsPayload(CamelModel):
    roots: list[str]


class FileEntryPayload(CamelModel):
    id: str
    name: str
    kind: Literal["dir", "file"]
    size_bytes: int
    modified_at: str
    content_kind: FileKind | None


class FilesListingPayload(CamelModel):
    path: str
    entries: list[FileEntryPayload]


class FileContentPayload(CamelModel):
    id: str
    content_kind: FileKind
    content: str
    language: str
    truncated: bool


class DeletedPayload(CamelModel):
    deleted: str


# ─── Commands ────────────────────────────────────────────────────────────
class CommandDefPayload(CamelModel):
    command_id: str
    label: str
    description: str
    destructive: bool
    required_grant: str


class CommandsPayload(CamelModel):
    commands: list[CommandDefPayload]


class CommandRunPayload(CamelModel):
    command_id: str
    message: str


# ─── System ──────────────────────────────────────────────────────────────
class StatusPayload(CamelModel):
    assistant_name: str
    active_model: str
    personality: str
    busy: bool


class TelemetryPayload(CamelModel):
    cpu_percent: float
    ram_percent: float
    battery_percent: float | None
    active_model: str
    uptime_seconds: int


class BackupStatePayload(CamelModel):
    enabled: bool
    provider: str
    last_backup_at: str
    last_result: str
    size_bytes: int


class RestorePayload(CamelModel):
    restored: bool


class EnrolledItemPayload(CamelModel):
    item_id: str
    name: str
    enrolled_at: str
    count: int | None
    last_seen_at: str


class EnrollmentPayload(CamelModel):
    voices: list[EnrolledItemPayload]
    faces: list[EnrolledItemPayload]


class ForgetEnrolledPayload(CamelModel):
    forgotten: str
    kind: Literal["voice", "face"]


class AuditEntryPayload(CamelModel):
    at: str
    device_id: str
    method: str
    path: str
    outcome: str


class AuditPayload(CamelModel):
    entries: list[AuditEntryPayload]
