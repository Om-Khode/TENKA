# assistant/actions/studio_runtime.py
"""The concrete StudioRuntime.

io/api declares what the daemon may know; this module supplies it. Living in
actions/ is what makes importing storage/, automation/ and llm/ legal, and what
keeps io/api clean of all three.

Every facade behind this module is synchronous, so every call goes through
asyncio.to_thread. Calling one directly would block the assistant's event loop
for the duration of a SQLite read.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Protocol

from ..core import runtime_config
from ..core.capabilities import Capability
from ..io.api.runtime import (
    ChatMessage, ConversationDetail, ConversationRef, Entity, Fact,
    KnowledgeGraph, MemoryScope, PersonalityState, PreferenceChange,
    PreferenceRecord, ProcedureRecord, Relationship, SaveOutcome, SettingRow,
    StudioRuntime, TurnRef,
)

_SCOPES = ("knowledge", "preferences", "procedures")

_KIND_BY_CAST = {bool: "toggle", int: "number", float: "slider", str: "text"}


class ChatDispatch(Protocol):
    """Supplied by main.py — the only path from a request into the pipeline."""

    # `grants` is positional and has no default, deliberately. A caller that
    # forgets it raises TypeError at the call site instead of silently
    # inheriting whatever the pipeline would otherwise have run with -- and
    # what it would otherwise have run with, before 6a.5, was everything.
    # "Forgot to say what this device may do" must never be spelled the same
    # way as "this device may do anything".
    async def submit(self, text: str,
                     grants: frozenset[Capability]) -> tuple[str, str, bool, str]: ...
    async def abort(self) -> bool: ...
    # Whether a submitted turn is currently in flight -- read by
    # LiveSystemRuntime for StatusInfo.busy. A plain attribute/property, not
    # async: main.py's `_StudioDispatch` only ever flips a bool, no I/O.
    busy: bool


def _parse_int_id(item_id: str) -> int | None:
    """`int(item_id)` on a non-numeric route parameter used to raise
    ValueError straight out of `_forget_sync`, which nothing here caught --
    it reached the client as a bare 500. A knowledge/procedure item_id that
    isn't a number is exactly as absent as one that parses but doesn't
    exist, so it is treated the same way `forget()` already treats an
    unknown id: caught here and reported as "not found" (via a plain
    `False`) rather than surfacing what shape the id-space happens to have.
    """
    try:
        return int(item_id)
    except (TypeError, ValueError):
        return None


def _load_properties(raw: Any) -> dict:
    """kg_entities/kg_relationships store properties as a properties_json
    TEXT column. Missing or corrupt JSON must not take the whole page down --
    it degrades to an empty dict instead."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


# ─── Chat ────────────────────────────────────────────────────────────────
class LiveChatRuntime:
    def __init__(self, dispatch: ChatDispatch) -> None:
        self._dispatch = dispatch

    async def send(self, text: str, grants: frozenset[Capability]) -> TurnRef:
        # `grants` travels with the text rather than being read from anywhere
        # here: the route is the only place that knows *which device* asked,
        # and the turn runs later, on the queue consumer's task, where the
        # request is long gone.
        turn_id, conversation_id, accepted, reason = await self._dispatch.submit(text, grants)
        return TurnRef(turn_id, conversation_id, accepted, reason)

    async def conversations(self) -> list[ConversationRef]:
        from .. import memory
        # list_conversation_sessions groups the `conversations` table (chat
        # turns) by session_id. memory.list_sessions() is a different
        # feature -- it lists recording_sessions, the screen/audio-capture
        # transcript table -- and would silently show the wrong history here.
        sessions = await asyncio.to_thread(memory.list_conversation_sessions, 20)
        return [
            ConversationRef(
                conversation_id=str(s.get("session_id", "")),
                title=str(s.get("last_input") or s.get("started_at") or "Conversation")[:80],
                updated_at=str(s.get("ended_at") or s.get("started_at") or ""),
                message_count=int(s.get("turn_count") or 0),
            )
            for s in sessions
        ]

    async def conversation(self, conversation_id: str) -> ConversationDetail | None:
        from .. import memory
        turns = await asyncio.to_thread(memory.get_recent, 200, conversation_id)
        if not turns:
            return None
        messages: list[ChatMessage] = []
        for turn in turns:
            turn_id = str(turn.get("id", ""))
            messages.append(ChatMessage(
                f"{turn_id}-u", "user", str(turn.get("user_input", "")),
                str(turn.get("timestamp", "")), intent=str(turn.get("intent", "")),
            ))
            messages.append(ChatMessage(
                f"{turn_id}-a", "assistant", str(turn.get("response", "")),
                str(turn.get("timestamp", "")), intent=str(turn.get("intent", "")),
            ))
        return ConversationDetail(conversation_id, conversation_id, messages)

    async def abort(self) -> bool:
        return await self._dispatch.abort()


# ─── Memory ──────────────────────────────────────────────────────────────
class LiveMemoryRuntime:
    async def knowledge(self) -> KnowledgeGraph:
        return await asyncio.to_thread(self._knowledge_sync)

    @staticmethod
    def _knowledge_sync() -> KnowledgeGraph:
        from .. import knowledge_graph
        return KnowledgeGraph(
            entities=[
                Entity(
                    id=int(row["id"]),
                    type=str(row.get("type") or ""),
                    canonical_name=str(row.get("canonical_name") or ""),
                    display_name=str(row.get("display_name") or row.get("canonical_name") or ""),
                    properties=_load_properties(row.get("properties_json")),
                    source=str(row.get("source") or ""),
                    confidence=float(row.get("confidence") or 0.0),
                    created_at=str(row.get("created_at") or ""),
                    updated_at=str(row.get("updated_at") or ""),
                    source_turn_id=row.get("source_turn_id"),
                )
                for row in knowledge_graph.list_entities()
            ],
            facts=[
                Fact(
                    id=int(row["id"]),
                    subject_id=int(row["subject_id"]),
                    predicate=str(row.get("predicate") or ""),
                    object=str(row.get("object") or ""),
                    confidence=float(row.get("confidence") or 0.0),
                    source=str(row.get("source") or ""),
                    event_at=row.get("event_at"),
                    invalid_at=row.get("invalid_at"),
                    expires_at=row.get("expires_at"),
                    verified_at=row.get("verified_at"),
                    created_at=str(row.get("created_at") or ""),
                    source_turn_id=row.get("source_turn_id"),
                )
                for row in knowledge_graph.list_facts()
            ],
            relationships=[
                Relationship(
                    id=int(row["id"]),
                    from_id=int(row["from_id"]),
                    to_id=int(row["to_id"]),
                    type=str(row.get("type") or ""),
                    confidence=float(row.get("confidence") or 0.0),
                    source=str(row.get("source") or ""),
                    source_turn_id=row.get("source_turn_id"),
                    properties=_load_properties(row.get("properties_json")),
                )
                for row in knowledge_graph.list_relationships()
            ],
        )

    async def preferences(self) -> list[PreferenceRecord]:
        return await asyncio.to_thread(self._preferences_sync)

    @staticmethod
    def _preferences_sync() -> list[PreferenceRecord]:
        from .. import preferences as preferences_facade
        # get_all_preferences() first, and short-circuit on empty: with no
        # preferences there is no key to attach history to, and skipping
        # get_preference_history() here avoids a second query the result
        # would never use.
        prefs = preferences_facade.get_all_preferences()
        if not prefs:
            return []
        history_by_key: dict[str, list[PreferenceChange]] = {}
        for entry in preferences_facade.get_preference_history(days=3_650):
            key = str(entry.get("key", ""))
            history_by_key.setdefault(key, []).append(PreferenceChange(
                value=str(entry.get("value", "")),
                changed_at=str(entry.get("changed_at") or entry.get("timestamp") or ""),
            ))
        return [
            PreferenceRecord(
                key=str(pref.get("key", "")),
                value=str(pref.get("value", "")),
                updated_at=str(pref.get("updated_at") or ""),
                history=history_by_key.get(str(pref.get("key", "")), []),
            )
            for pref in prefs
        ]

    async def procedures(self) -> list[ProcedureRecord]:
        return await asyncio.to_thread(self._procedures_sync)

    @staticmethod
    def _procedures_sync() -> list[ProcedureRecord]:
        from .. import procedures as procedures_facade
        records = []
        for proc in procedures_facade.list_procedures(enabled_only=False):
            steps = proc.get("steps") or []
            records.append(ProcedureRecord(
                id=int(proc["id"]),
                name=str(proc.get("name") or proc.get("trigger") or ""),
                steps=[str(s if isinstance(s, str) else s.get("text", s)) for s in steps],
                taught_at=str(proc.get("created_at") or ""),
                run_count=int(proc.get("use_count") or 0),
            ))
        return records

    async def forget(self, scope: MemoryScope, item_id: str) -> bool:
        if scope not in _SCOPES:
            raise ValueError(f"unknown scope: {scope}")
        return await asyncio.to_thread(self._forget_sync, scope, item_id)

    @staticmethod
    def _forget_sync(scope: str, item_id: str) -> bool:
        if scope == "knowledge":
            from .. import knowledge_graph
            entity_id = _parse_int_id(item_id)
            if entity_id is None:
                return False
            # Forgetting an entity takes its facts and edges with it. Leaving
            # orphaned facts behind would keep answering questions about
            # something the user asked her to forget.
            return knowledge_graph.delete_entity(entity_id)
        if scope == "preferences":
            from .. import preferences
            return preferences.delete_preference(item_id)
        from .. import procedures
        procedure_id = _parse_int_id(item_id)
        if procedure_id is None:
            return False
        return procedures.delete_procedure(procedure_id)

    async def forget_all(self) -> int:
        return await asyncio.to_thread(self._forget_all_sync)

    @staticmethod
    def _forget_all_sync() -> int:
        from .. import knowledge_graph, preferences, procedures
        removed = 0
        for row in knowledge_graph.list_entities():
            if knowledge_graph.delete_entity(int(row["id"])):
                removed += 1
        preferences.reset_preferences()
        for proc in procedures.list_procedures(enabled_only=False):
            procedures.delete_procedure(int(proc["id"]))
            removed += 1
        return removed


def _cast_env_value(cast: type, raw: str) -> Any:
    """Apply the same cast `runtime_config.setting()` applies when it
    resolves an env-sourced value, rather than reporting the raw os.getenv()
    string verbatim while `default` (below) keeps its real type -- the
    mismatch Studio would otherwise render as `value: "true"` next to
    `default: false`. Returns None (never a legal SettingValue) on a cast
    that would fail, exactly the case runtime_config.setting() itself falls
    through to the hardcoded default for.
    """
    try:
        if cast is bool:
            return raw.strip().lower() in ("true", "1", "yes", "on")
        return cast(raw)
    except (TypeError, ValueError):
        return None


def _matches_cast(cast: type, value: Any) -> bool:
    """Whether `value` is a legitimate value for a setting whose registry
    entry declares `cast`. Guards the write path the read path
    (_cast_env_value / runtime_config.setting()) both already have to
    tolerate: a value that fails its cast was being stored as-is, then
    silently served back as `default` by runtime_config the next time
    anything actually read it -- Studio would show the submitted value while
    the assistant used something else entirely.

    `bool` is checked by identity, not by attempting the cast: Python's
    `bool("anything non-empty")` never raises, so a bare try/except would
    accept literally any non-empty string for a toggle setting. `bool` is
    also a subclass of `int`, so True/False must be excluded before an
    int/float/str setting's own try/except gets a chance to accept them.
    """
    if cast is bool:
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    try:
        cast(value)
    except (TypeError, ValueError):
        return False
    return True


# ─── Settings ────────────────────────────────────────────────────────────
class LiveSettingsRuntime:
    async def all(self) -> list[SettingRow]:
        return await asyncio.to_thread(self._all_sync)

    @staticmethod
    def _all_sync() -> list[SettingRow]:
        from .. import settings as settings_facade
        stored = settings_facade.list_all()
        rows: list[SettingRow] = []
        for key, meta in sorted(runtime_config.REGISTRY.items()):
            cast = meta.get("cast", str)
            env_value = os.getenv(key.upper())
            # Same precedence runtime_config.setting() resolves with, reported
            # rather than reinvented. Read it there before changing this: the DB
            # wins over the environment, not the other way round, so a stored
            # value is reported as "db" even when the env var is also set.
            if key in stored:
                value, source = stored[key], "db"
            elif env_value is not None:
                casted = _cast_env_value(cast, env_value)
                if casted is None:
                    # The same cast failure runtime_config.setting() itself
                    # falls through on -- reported as "default", matching
                    # what the assistant actually resolves at runtime rather
                    # than the raw, wrongly-typed env string.
                    value, source = meta["default"], "default"
                else:
                    value, source = casted, "env"
            else:
                value, source = meta["default"], "default"
            rows.append(SettingRow(
                key=key,
                group=key.split("_")[0].title(),
                description=meta.get("description", ""),
                kind=_KIND_BY_CAST.get(cast, "text"),
                value=value,
                default=meta["default"],
                needs_restart=bool(meta.get("needs_restart")),
                source=source,
                options=[],
            ))
        return rows

    async def save(self, changes: dict) -> SaveOutcome:
        return await asyncio.to_thread(self._save_sync, changes)

    @staticmethod
    def _save_sync(changes: dict) -> SaveOutcome:
        from .. import settings as settings_facade
        saved, rejected, restart = [], {}, []
        for key, value in changes.items():
            meta = runtime_config.REGISTRY.get(key)
            if meta is None:
                rejected[key] = "unknown setting"
                continue
            # No env check. runtime_config resolves DB before env, so a stored
            # value legitimately takes precedence over an environment one --
            # refusing the save here would invent a rule the assistant does not
            # have.
            #
            # Type-checked against the setting's own cast before it ever
            # reaches storage: SettingsPatch's Pydantic type
            # (str | int | float | bool) only proves `value` is *a* legal
            # SettingValue, not that it is the *right one* for this key. A
            # value that fails its cast used to be stored as-is, then
            # silently served back as `meta["default"]` by
            # runtime_config.setting() the next time anything actually read
            # it -- Studio would show `value: "banana"` while the assistant
            # used something else entirely.
            cast = meta.get("cast", str)
            if not _matches_cast(cast, value):
                rejected[key] = "value does not match the setting's type"
                continue
            try:
                settings_facade.set(key, value, source="studio")
            except Exception as exc:  # storage failure, not user error
                rejected[key] = f"could not save: {exc}"
                continue
            saved.append(key)
            if meta.get("needs_restart"):
                restart.append(key)
        return SaveOutcome(saved, rejected, restart)


# ─── Personality ─────────────────────────────────────────────────────────
class LivePersonalityRuntime:
    async def state(self) -> PersonalityState:
        return await asyncio.to_thread(self._state_sync)

    @staticmethod
    def _state_sync() -> PersonalityState:
        from .. import personality
        # The active personality id lives on assistant.personalities'
        # module-level loader (get_active_personality_id), not on
        # assistant.personality -- that module only ever imports the name
        # locally inside _get_repo(); it never re-exports it. This is the
        # live source switch_personality() actually writes through, unlike
        # config.ACTIVE_PERSONALITY, which is frozen at import time.
        from ..personalities import get_active_personality_id
        return PersonalityState(
            base=get_active_personality_id(),
            available=sorted(personality.list_personalities()),
            traits=personality.get_current_traits(),
            sample_line=personality.get_metadata("sample_line") or "",
        )

    async def set_base(self, base: str) -> PersonalityState:
        await asyncio.to_thread(self._set_base_sync, base)
        return await self.state()

    @staticmethod
    def _set_base_sync(base: str) -> None:
        from .. import personality
        personality.switch_personality(base)

    async def reset(self) -> PersonalityState:
        await asyncio.to_thread(self._reset_sync)
        return await self.state()

    @staticmethod
    def _reset_sync() -> None:
        from .. import personality
        personality.reset_traits()


def build_studio_runtime(dispatch: ChatDispatch) -> StudioRuntime:
    """Assemble the runtime. Files, commands and system land in Task 8."""
    from .studio_runtime_system import (  # local import: same package, later task
        LiveCommandRuntime, LiveFileRuntime, LiveSystemRuntime,
    )
    return StudioRuntime(
        chat=LiveChatRuntime(dispatch),
        memory=LiveMemoryRuntime(),
        settings=LiveSettingsRuntime(),
        personality=LivePersonalityRuntime(),
        files=LiveFileRuntime(),
        commands=LiveCommandRuntime(),
        # dispatch, not zero-arg: StatusInfo.busy reads the same in-flight
        # state `dispatch.submit()`/the queue consumer's completion hook
        # maintain, rather than a hardcoded False.
        system=LiveSystemRuntime(dispatch),
    )
