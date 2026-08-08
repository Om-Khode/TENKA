"""A StudioRuntime that behaves, so route tests need no DB and no assistant.

Mutations mutate. A fake whose forget() returns True without removing anything
cannot distinguish a correct route from a broken one.
"""
from __future__ import annotations

from assistant.io.api.runtime import (
    BackupState, ChatMessage, CommandDef, CommandOutcome, ConversationDetail,
    ConversationRef, EnrolledItem, EnrollmentState, Entity, Fact, FileContent,
    FileEntry, KnowledgeGraph, PersonalityState, PreferenceChange, PreferenceRecord,
    ProcedureRecord, Relationship, SaveOutcome, SettingRow, StatusInfo,
    StudioRuntime, TelemetrySnapshot, TurnRef,
)


class FakeChatRuntime:
    def __init__(self) -> None:
        self.busy = False
        self.sent: list[str] = []
        self.aborted = 0
        self._messages = [
            ChatMessage("m1", "user", "what did I ask you yesterday", "2026-08-07T09:00:00Z"),
            ChatMessage("m2", "assistant", "Three things. Want them in order?",
                        "2026-08-07T09:00:04Z", intent="memory_query"),
        ]

    async def send(self, text: str) -> TurnRef:
        if self.busy:
            return TurnRef("", "c1", accepted=False, reason="busy")
        self.sent.append(text)
        return TurnRef(f"t{len(self.sent)}", "c1", accepted=True)

    async def conversations(self) -> list[ConversationRef]:
        return [ConversationRef("c1", "Yesterday's list", "2026-08-07T09:00:04Z", 2)]

    async def conversation(self, conversation_id: str) -> ConversationDetail | None:
        if conversation_id != "c1":
            return None
        return ConversationDetail("c1", "Yesterday's list", list(self._messages))

    async def abort(self) -> bool:
        self.aborted += 1
        return True


class FakeMemoryRuntime:
    """Two entities, a superseded fact, one relationship, one dangling edge.

    The dangling relationship (to entity 99, which does not exist) is
    deliberate: real graphs point at rows that are gone, and the Memory page
    already drops those rather than crashing. A fixture with a clean graph
    cannot tell a correct client from one that never handles it.
    """

    def __init__(self) -> None:
        self._entities = [
            Entity(1, "person", "sister", "Sister", {"relation": "family"},
                   "conversation", 0.82, "2026-07-19T10:11:00Z",
                   "2026-07-30T09:00:00Z", "turn-8812"),
            Entity(2, "event", "thesis defence", "Thesis defence", {},
                   "conversation", 0.64, "2026-07-22T18:02:00Z",
                   "2026-07-22T18:02:00Z", "turn-8840"),
        ]
        self._facts = [
            # superseded: she moved cities, the old fact is kept, not overwritten
            Fact(10, 1, "lives_in", "Pune", 0.7, "conversation", None,
                 "2026-07-25T12:00:00Z", None, None, "2026-07-19T10:11:00Z", "turn-8812"),
            Fact(11, 1, "lives_in", "Bengaluru", 0.9, "conversation", None,
                 None, None, "2026-07-30T09:00:00Z", "2026-07-25T12:00:00Z", "turn-9001"),
            Fact(12, 2, "happens_on", "2026-09-04", 0.95, "conversation",
                 "2026-09-04T00:00:00Z", None, None, None, "2026-07-22T18:02:00Z", "turn-8840"),
        ]
        self._relationships = [
            Relationship(20, 1, 2, "attending", 0.6, "conversation", "turn-8840",
                         properties={"role": "plus one"}),
            Relationship(21, 1, 99, "mentions", 0.4, "conversation", None),
        ]
        self._preferences = [
            PreferenceRecord("reading_pace", "1.5x", "2026-07-02T08:00:00Z",
                             [PreferenceChange("1.25x", "2026-06-10T08:00:00Z")]),
        ]
        self._procedures = [
            ProcedureRecord(30, "wind down",
                            ["dim the lights", "close the browser", "start the playlist"],
                            "2026-06-28T22:40:00Z", 12),
        ]

    async def knowledge(self) -> KnowledgeGraph:
        return KnowledgeGraph(list(self._entities), list(self._facts),
                              list(self._relationships))

    async def preferences(self) -> list[PreferenceRecord]:
        return list(self._preferences)

    async def procedures(self) -> list[ProcedureRecord]:
        return list(self._procedures)

    async def forget(self, scope: str, item_id: str) -> bool:
        if scope == "knowledge":
            remaining = [e for e in self._entities if str(e.id) != item_id]
            if len(remaining) == len(self._entities):
                return False
            gone = {e.id for e in self._entities} - {e.id for e in remaining}
            self._entities = remaining
            self._facts = [f for f in self._facts if f.subject_id not in gone]
            self._relationships = [
                r for r in self._relationships
                if r.from_id not in gone and r.to_id not in gone
            ]
            return True
        if scope == "preferences":
            remaining = [p for p in self._preferences if p.key != item_id]
            if len(remaining) == len(self._preferences):
                return False
            self._preferences = remaining
            return True
        if scope == "procedures":
            remaining = [p for p in self._procedures if str(p.id) != item_id]
            if len(remaining) == len(self._procedures):
                return False
            self._procedures = remaining
            return True
        return False

    async def forget_all(self) -> int:
        removed = len(self._entities) + len(self._preferences) + len(self._procedures)
        self._entities, self._facts, self._relationships = [], [], []
        self._preferences, self._procedures = [], []
        return removed


class FakeSettingsRuntime:
    def __init__(self) -> None:
        self._rows = {
            "followup_timer": SettingRow(
                "followup_timer", "Conversation", "Seconds to keep listening after a reply.",
                "slider", 4.5, 3.0, False, "db",
            ),
            "tts_speed": SettingRow(
                "tts_speed", "Voice", "Playback rate for spoken replies.",
                "slider", 1.0, 1.0, False, "default",
            ),
            "active_personality": SettingRow(
                "active_personality", "Personality", "Which base persona she speaks from.",
                "select", "warm", "warm", True, "default", options=["warm", "dry", "brisk"],
            ),
            "camera_enabled": SettingRow(
                "camera_enabled", "Perception", "Allow the camera to be opened on request.",
                "toggle", True, False, True, "env",
            ),
        }

    async def all(self) -> list[SettingRow]:
        return list(self._rows.values())

    async def save(self, changes: dict) -> SaveOutcome:
        saved, rejected, restart = [], {}, []
        for key, value in changes.items():
            row = self._rows.get(key)
            if row is None:
                rejected[key] = "unknown setting"
                continue
            # A row currently sourced from "env" is still saveable: the
            # assistant resolves DB before env, so writing a stored value takes
            # precedence from then on. The row flips to "db".
            self._rows[key] = SettingRow(
                row.key, row.group, row.description, row.kind, value, row.default,
                row.needs_restart, "db", row.options,
            )
            saved.append(key)
            if row.needs_restart:
                restart.append(key)
        return SaveOutcome(saved, rejected, restart)


class FakePersonalityRuntime:
    def __init__(self) -> None:
        self._state = PersonalityState(
            base="warm",
            available=["warm", "dry", "brisk"],
            traits={"warmth": 0.72, "wit": 0.55, "candour": 0.61,
                    "patience": 0.48, "curiosity": 0.66, "restraint": 0.39},
            sample_line="You left the tab open again. I closed it. You're welcome.",
        )

    async def state(self) -> PersonalityState:
        return self._state

    async def set_base(self, base: str) -> PersonalityState:
        self._state = PersonalityState(base, self._state.available,
                                       self._state.traits, self._state.sample_line)
        return self._state

    async def reset(self) -> PersonalityState:
        self._state = PersonalityState("warm", self._state.available,
                                       {k: 0.5 for k in self._state.traits},
                                       self._state.sample_line)
        return self._state


class FakeFileRuntime:
    """One nested directory, because the breadcrumb and the path-keyed ids are
    the whole reason this is not a flat listing."""

    def __init__(self) -> None:
        self._entries: dict[str, list[FileEntry]] = {
            "desktop": [
                FileEntry("desktop/notes.md", "notes.md", "file", 2_048,
                          "2026-08-06T21:24:00Z", "text"),
                FileEntry("desktop/captures", "captures", "dir", 0,
                          "2026-08-01T10:00:00Z"),
            ],
            "desktop/captures": [
                FileEntry("desktop/captures/shot.png", "shot.png", "file", 41_002,
                          "2026-08-01T10:00:00Z", "image"),
            ],
            "downloads": [
                FileEntry("downloads/statement.pdf", "statement.pdf", "file", 190_211,
                          "2026-08-04T12:02:00Z", "binary"),
            ],
            "documents": [],
        }
        self._contents = {
            "desktop/notes.md": FileContent("desktop/notes.md", "text",
                                            "# notes\n\nbuy cable\n"),
        }

    async def roots(self) -> list[str]:
        return ["desktop", "documents", "downloads"]

    async def listing(self, path: str) -> list[FileEntry]:
        if ".." in path or path.startswith(("/", "\\")):
            raise ValueError("path escapes its root")
        if path not in self._entries:
            raise KeyError(path)
        return list(self._entries[path])

    async def read(self, path: str) -> FileContent:
        if ".." in path or path.startswith(("/", "\\")):
            raise ValueError("path escapes its root")
        if path not in self._contents:
            raise FileNotFoundError(path)
        return self._contents[path]

    async def rename(self, path: str, new_name: str) -> FileEntry:
        if ".." in path or "/" in new_name or "\\" in new_name:
            raise ValueError("invalid path or name")
        parent = path.rsplit("/", 1)[0]
        for index, entry in enumerate(self._entries.get(parent, [])):
            if entry.path == path:
                renamed = FileEntry(f"{parent}/{new_name}", new_name, entry.kind,
                                    entry.size_bytes, entry.modified_at,
                                    entry.content_kind)
                self._entries[parent][index] = renamed
                return renamed
        raise FileNotFoundError(path)

    async def delete(self, path: str) -> bool:
        if ".." in path:
            raise ValueError("path escapes its root")
        parent = path.rsplit("/", 1)[0]
        entries = self._entries.get(parent, [])
        remaining = [e for e in entries if e.path != path]
        if len(remaining) == len(entries):
            return False
        self._entries[parent] = remaining
        return True


class FakeCommandRuntime:
    def __init__(self) -> None:
        self.ran: list[str] = []
        self._catalogue = [
            CommandDef("lock_workstation", "Lock PC", "Locks the desktop session.",
                       True, "system_control"),
            CommandDef("volume_up", "Volume Up", "Raises the system volume.",
                       False, "system_control"),
        ]

    async def catalogue(self) -> list[CommandDef]:
        return list(self._catalogue)

    async def run(self, command_id: str) -> CommandOutcome:
        if command_id not in [c.command_id for c in self._catalogue]:
            return CommandOutcome(False, "unknown command")
        self.ran.append(command_id)
        return CommandOutcome(True, "done")


class FakeSystemRuntime:
    def __init__(self) -> None:
        self.backups_run = 0
        self.restored_with: list[str] = []
        self._backup = BackupState(True, "google_drive", "2026-08-07T04:00:00Z", "ok", 18_432_112)
        self._enrollment = EnrollmentState(
            voices=[EnrolledItem("v1", "primary", "2026-07-01T08:00:00Z",
                                 count=6, last_seen_at="2026-08-07T22:10:00Z")],
            faces=[EnrolledItem("f1", "Om", "2026-07-03T19:20:00Z",
                                count=14, last_seen_at="2026-08-06T21:24:00Z")],
        )

    async def status(self) -> StatusInfo:
        return StatusInfo("TENKA", "1.0.0", "gemini-flash-lite", "warm", busy=False)

    async def telemetry(self) -> TelemetrySnapshot:
        return TelemetrySnapshot(21.5, 63.0, 88.0, "gemini-flash-lite", 4_512)

    async def backup_state(self) -> BackupState:
        return self._backup

    async def run_backup(self) -> BackupState:
        self.backups_run += 1
        self._backup = BackupState(True, "google_drive", "2026-08-08T09:15:00Z", "ok", 18_500_000)
        return self._backup

    async def restore_backup(self, recovery_phrase: str) -> bool:
        self.restored_with.append(recovery_phrase)
        return len(recovery_phrase.split()) == 8

    async def enrollment(self) -> EnrollmentState:
        return self._enrollment

    async def forget_enrolled(self, kind: str, item_id: str) -> bool:
        if kind == "voice":
            remaining = [v for v in self._enrollment.voices if v.item_id != item_id]
            changed = len(remaining) != len(self._enrollment.voices)
            self._enrollment = EnrollmentState(remaining, self._enrollment.faces)
            return changed
        if kind == "face":
            remaining = [f for f in self._enrollment.faces if f.item_id != item_id]
            changed = len(remaining) != len(self._enrollment.faces)
            self._enrollment = EnrollmentState(self._enrollment.voices, remaining)
            return changed
        return False


def build_fake_runtime() -> StudioRuntime:
    return StudioRuntime(
        chat=FakeChatRuntime(),
        memory=FakeMemoryRuntime(),
        settings=FakeSettingsRuntime(),
        personality=FakePersonalityRuntime(),
        files=FakeFileRuntime(),
        commands=FakeCommandRuntime(),
        system=FakeSystemRuntime(),
    )
