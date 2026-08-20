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


# ─── Session ─────────────────────────────────────────────────────────────
class SessionPayload(CamelModel):
    """Who is calling, and what may this connection carry.

    Three lists, deliberately, not one -- each answers a different reason a
    control might be greyed out in Studio, and collapsing any two would make
    that reason unexplainable at the UI:

    - `grants` is what the device was issued at pairing. A control disabled
      here means the device itself was never handed that capability, on any
      connection, ever.
    - `effective` is what survives this listener's fixed ceiling on this
      connection (`policy.py`'s `effective()`). A control disabled only here
      means the device holds the capability, but the transport it arrived on
      -- a Cloudflare tunnel that can read the plaintext, say -- refuses to
      carry it.
    - `raised` is what Milestone 6b's third case, the ceiling lifting, is
      reporting: a live, expiring, per-device raise minted by the operator at
      the keyboard, on top of whatever `effective` already allows. A control
      enabled only because of a raise is not the same story as one always
      enabled -- Studio's banner needs to say a floor was deliberately and
      temporarily raised, not render an ordinary control that looks
      permanent. Empty, never omitted, when no raise is live: a payload
      whose shape varied by transport would be worse for a client to read
      than one whose values are simply empty.

    `raise_expires_in_seconds` is `None` exactly when `raised` is empty, and
    otherwise counts down a live raise -- converted from the store's
    `time.monotonic()` reading at request time, never recomputed as
    `now + duration`, which would drift if the store's cap ever changed.
    """

    device_id: str
    label: str
    grants: list[str]
    effective: list[str]
    raised: list[str]
    raise_expires_in_seconds: int | None
    policy: str


# ─── Listener discovery ──────────────────────────────────────────────────
class ListenerPayload(CamelModel):
    """The three facts a caller needs before it holds any credential at all.

    Served by `GET /v1/listener`, the one unauthenticated read in this API
    (`POST /v1/pair` is the one unauthenticated *write*) -- Studio's
    `/connect` screen has no other way to learn which listener served it, and
    that has to be known before it can decide whether to offer the
    bearer-token exchange at all. Deliberately nothing beyond these three:
    no device data, no hostname, no ceiling, no capability list. The caller
    already knows which port it reached -- it chose it -- so none of this is
    a secret; it is Studio being told what it could otherwise only guess by
    trying the bearer exchange and reading the failure.

    `can_pair` is `pairing_denied_by_transport()`'s own answer negated, read
    from `routes/pairing.py` rather than re-derived here, so this field and
    `pair_device`'s actual refusal cannot drift apart.
    """

    policy: str
    allow_bearer: bool
    can_pair: bool


# ─── Pairing and devices ─────────────────────────────────────────────────
class PairCodePayload(CamelModel):
    """What the laptop puts on screen. The code is in here exactly once, and
    it is the reason nothing on this route is logged.

    `qr_svg` encodes `<endpoint>/pair#<code>` -- the code in the URL
    *fragment*, which a browser never sends to a server, so it lands in no
    access log on the way. `endpoints` is a list because a phone may have to
    be told more than one way to reach this machine once transports exist; in
    6a it holds the loopback origin alone.
    """

    code: str
    expires_at: str
    endpoints: list[str]
    qr_svg: str


class RaisePayload(CamelModel):
    """One live ceiling raise, as an admin listener reports it.

    Four of the five fields describe the raise; the fifth, `reason`, is the
    only thing that makes a seven-day window reviewable a week later. It is
    free text the operator typed at mint time, echoed back verbatim -- which
    is safe here and only here: this payload is served by
    `require_admin(SYSTEM_CONTROL)` routes, so the only reader is the loopback
    caller who wrote it.

    `granted_by` is deliberately absent. The record holds it (`RaiseGrant`) and
    the log line names it, but a device row in the revoke list is not the place
    to publish which *other* device authorised something -- the same reasoning
    that keeps a listener and a source address off `DevicePayload`.

    `expires_in_seconds` counts down, converted from the store's
    `time.monotonic()` reading at request time. Never a wall-clock timestamp: a
    monotonic reading means nothing to a client, and recomputing one as
    `now + duration` would drift the moment the store's cap changed.
    """

    device_id: str
    transport: str
    capabilities: list[str]
    expires_in_seconds: int
    reason: str


class DevicePayload(CamelModel):
    """One row of the revoke list, and deliberately nothing more.

    This response describes every credential that reaches this machine, so it
    carries only what a person needs to decide what to kill: which row, what
    it is called, what it can do, and when it was last used. No token, no
    token hash, and no live address -- a device's grants say what it may do,
    and where it happens to be connecting *from right now* is not a fact this
    list is allowed to make somebody act on.

    `raises` is Milestone 6b's one addition, and it belongs to the same
    question rather than widening it: a device that can currently do more than
    its grants column suggests is precisely the row somebody reading this list
    needs to see. Empty, never omitted, for a device with no live raise --
    and there is deliberately no route that lists raises on their own, so
    nothing here becomes a second oracle for which device ids exist.

    `paired_on` is a second Milestone 6b addition and a different kind of fact
    from `raises`: not a live connection address, but which listener policy --
    `"local"`, `"tailnet"`, or `"funnel"` -- this credential was redeemed over
    at pairing time, recorded once and never updated again. With three
    transports carrying three different capability ceilings, which door a
    device's credential came through is exactly what someone deciding whether
    to trust or cut off a row needs. `None` for a device paired before this
    field existed, or issued outside the pairing route entirely -- a genuinely
    unknown origin, never guessed as `"local"` for convenience.
    """

    device_id: str
    label: str
    grants: list[str]
    created_at: str
    last_seen_at: str | None
    raises: list[RaisePayload]
    paired_on: str | None


class DevicesPayload(CamelModel):
    devices: list[DevicePayload]


class RevokedPayload(CamelModel):
    revoked: str


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
    # Whether the encryption key is armed in this process. See BackupState's
    # own comment in runtime.py: `enabled` without this describes a machine
    # that intends to back up, not one that can.
    unlocked: bool


class RestorePayload(CamelModel):
    restored: bool


class UnlockPayload(CamelModel):
    unlocked: bool


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


# ─── Transports ──────────────────────────────────────────────────────────
class TransportPayload(CamelModel):
    """One transport, running or not.

    `ceiling` and `raisable` are read straight off `policy.py`'s `POLICIES`
    for this transport's name, never hand-copied here -- so Studio can explain
    *why* a control is unavailable on a given transport without carrying a
    second copy of the table that could drift from the one the daemon
    actually enforces. `url` is `None` whenever `running` is `False`: a
    session with no announced hostname does not serve
    (`transports/manager.py`'s `is_serving`), so there is nothing to show.

    `pairable` is read the same way, off `policy.pairable`, and it exists for
    the transport that someday is not. All three standing transports are
    pairable, so today this field is uniformly `True` -- but a client that
    infers pairability from a transport's *name* is a client that will offer a
    QR nobody can redeem the day a TLS-terminating transport ships, which is
    exactly the shape the daemon removed from its own code when it replaced
    `policy.name == "quick"` with `policy.pairable`. Studio should not be left
    holding the name check the daemon just deleted.
    """

    name: str
    running: bool
    url: str | None
    ceiling: list[str]
    raisable: list[str]
    pairable: bool


class TransportsPayload(CamelModel):
    transports: list[TransportPayload]


class AuditEntryPayload(CamelModel):
    at: str
    device_id: str
    method: str
    path: str
    outcome: str


class AuditPayload(CamelModel):
    entries: list[AuditEntryPayload]
