# assistant/io/api/runtime.py
"""What the daemon is allowed to know about the assistant.

Seven small protocols, one per domain, bundled into StudioRuntime. Routes touch
these and nothing else, which is what lets io/api stay clear of storage/,
actions/ and automation/ while still serving real data.

The concrete implementation lives at assistant/actions/studio_runtime.py, where
those imports are legal. main.py builds it and injects it.

Layering: io/api — core + config only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

MemoryScope = Literal["knowledge", "preferences", "procedures"]
SettingValue = str | int | float | bool


# ─── Chat ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TurnRef:
    turn_id: str
    conversation_id: str
    accepted: bool
    reason: str = ""


@dataclass(frozen=True)
class ChatMessage:
    message_id: str
    role: Literal["user", "assistant"]
    text: str
    created_at: str
    intent: str = ""


@dataclass(frozen=True)
class ConversationRef:
    conversation_id: str
    title: str
    updated_at: str
    message_count: int


@dataclass(frozen=True)
class ConversationDetail:
    conversation_id: str
    title: str
    messages: list[ChatMessage] = field(default_factory=list)


# ─── Memory ──────────────────────────────────────────────────────────────
# These mirror the assistant's kg_entities / kg_facts / kg_relationships /
# user_preferences / user_procedures columns, because Studio's types/memory.ts
# already does. Flattening the graph into title/subtitle would take provenance,
# supersession and the ego graph away from a page that renders all three today.
@dataclass(frozen=True)
class Entity:
    id: int
    type: str
    canonical_name: str
    display_name: str
    properties: dict[str, str]
    source: str
    confidence: float
    created_at: str
    updated_at: str
    source_turn_id: str | None


@dataclass(frozen=True)
class Fact:
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


@dataclass(frozen=True)
class Relationship:
    id: int
    from_id: int
    to_id: int
    type: str
    confidence: float
    source: str
    source_turn_id: str | None
    # Mirrors kg_relationships.properties_json, the way Entity.properties
    # mirrors kg_entities' — defaulted empty so existing positional
    # construction sites (built before this field existed) stay valid.
    properties: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeGraph:
    entities: list[Entity] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)


@dataclass(frozen=True)
class PreferenceChange:
    value: str
    changed_at: str


@dataclass(frozen=True)
class PreferenceRecord:
    key: str
    value: str
    updated_at: str
    history: list[PreferenceChange] = field(default_factory=list)


@dataclass(frozen=True)
class ProcedureRecord:
    id: int
    name: str
    steps: list[str]
    taught_at: str
    run_count: int


# ─── Settings and personality ────────────────────────────────────────────
@dataclass(frozen=True)
class SettingRow:
    key: str
    group: str
    description: str
    kind: Literal["toggle", "slider", "select", "number", "text"]
    value: SettingValue
    default: SettingValue
    needs_restart: bool
    # Which layer owns the current value, mirroring the assistant's
    # DB -> env -> default resolution and Studio's SettingSource union. A row
    # owned by "env" is not user-editable, which a boolean "overridden" cannot
    # say without conflating "an environment variable owns this" with "this
    # differs from its default".
    source: Literal["db", "env", "default"]
    options: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SaveOutcome:
    saved: list[str] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)
    restart_required: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PersonalityState:
    base: str
    available: list[str] = field(default_factory=list)
    traits: dict[str, float] = field(default_factory=dict)
    sample_line: str = ""


# ─── Files ───────────────────────────────────────────────────────────────
# Path-keyed, because Studio's types/file.ts already is: a FileNode's id is its
# path ("downloads/invoices/inv-0001.pdf"), the breadcrumb is a split() of it,
# and its own comment says spec 5's path parameter maps straight onto it.
FileKind = Literal["text", "code", "image", "binary"]


@dataclass(frozen=True)
class FileEntry:
    path: str                       # "desktop" | "desktop/notes.md"
    name: str
    kind: Literal["dir", "file"]
    size_bytes: int                 # directories report 0
    modified_at: str
    content_kind: FileKind | None = None


@dataclass(frozen=True)
class FileContent:
    path: str
    content_kind: FileKind
    text: str = ""                  # text/code, or a data URI for an image
    language: str = ""              # shiki language, code only
    truncated: bool = False


# ─── Commands ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CommandDef:
    command_id: str
    label: str
    description: str
    destructive: bool
    required_grant: str


@dataclass(frozen=True)
class CommandOutcome:
    ok: bool
    message: str


# ─── System ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TelemetrySnapshot:
    cpu_percent: float
    ram_percent: float
    battery_percent: float | None
    active_model: str
    uptime_seconds: int


@dataclass(frozen=True)
class BackupState:
    enabled: bool
    provider: str
    last_backup_at: str
    last_result: str
    size_bytes: int


@dataclass(frozen=True)
class EnrolledItem:
    item_id: str
    name: str
    enrolled_at: str
    # Deliberately one generic field, not "encoding_count" for faces and
    # "sample_count" for voices -- the panel renders both kinds the same
    # way. None means the assistant has no accessor for this item's count
    # (speaker_verify exposes none today); a real 0 would be a wrong fact
    # stated confidently about an enrollment that plainly exists.
    count: int | None = None
    # Last time this item was recognised/heard, if the assistant tracks
    # one. "" means it genuinely doesn't know, not "never".
    last_seen_at: str = ""


@dataclass(frozen=True)
class EnrollmentState:
    voices: list[EnrolledItem] = field(default_factory=list)
    faces: list[EnrolledItem] = field(default_factory=list)


@dataclass(frozen=True)
class StatusInfo:
    # No `version` field: no app version constant exists anywhere in the
    # codebase (config.py, assistant/__init__.py, pyproject.toml all checked),
    # so the choice was between wiring a real one and shipping an eternal ""
    # a Studio client has no honest use for. Dropped from the wire rather
    # than faked; add it back if a real version string ever exists to report.
    assistant_name: str
    active_model: str
    personality: str
    busy: bool


# ─── Protocols ───────────────────────────────────────────────────────────
@runtime_checkable
class ChatRuntime(Protocol):
    async def send(self, text: str) -> TurnRef: ...
    async def conversations(self) -> list[ConversationRef]: ...
    async def conversation(self, conversation_id: str) -> ConversationDetail | None: ...
    async def abort(self) -> bool: ...


@runtime_checkable
class MemoryRuntime(Protocol):
    async def knowledge(self) -> KnowledgeGraph: ...
    async def preferences(self) -> list[PreferenceRecord]: ...
    async def procedures(self) -> list[ProcedureRecord]: ...
    async def forget(self, scope: MemoryScope, item_id: str) -> bool: ...
    async def forget_all(self) -> int: ...


@runtime_checkable
class SettingsRuntime(Protocol):
    async def all(self) -> list[SettingRow]: ...
    async def save(self, changes: dict[str, SettingValue]) -> SaveOutcome: ...


@runtime_checkable
class PersonalityRuntime(Protocol):
    async def state(self) -> PersonalityState: ...
    async def set_base(self, base: str) -> PersonalityState: ...
    async def reset(self) -> PersonalityState: ...


@runtime_checkable
class FileRuntime(Protocol):
    async def roots(self) -> list[str]: ...
    async def listing(self, path: str) -> list[FileEntry]: ...
    async def read(self, path: str) -> FileContent: ...
    async def rename(self, path: str, new_name: str) -> FileEntry: ...
    async def delete(self, path: str) -> bool: ...


@runtime_checkable
class CommandRuntime(Protocol):
    async def catalogue(self) -> list[CommandDef]: ...
    async def run(self, command_id: str) -> CommandOutcome: ...


@runtime_checkable
class SystemRuntime(Protocol):
    async def status(self) -> StatusInfo: ...
    async def telemetry(self) -> TelemetrySnapshot: ...
    async def backup_state(self) -> BackupState: ...
    async def run_backup(self) -> BackupState: ...
    async def restore_backup(self, recovery_phrase: str) -> bool: ...
    async def enrollment(self) -> EnrollmentState: ...
    async def forget_enrolled(self, kind: str, item_id: str) -> bool: ...


@dataclass(frozen=True)
class StudioRuntime:
    chat: ChatRuntime
    memory: MemoryRuntime
    settings: SettingsRuntime
    personality: PersonalityRuntime
    files: FileRuntime
    commands: CommandRuntime
    system: SystemRuntime
