# assistant/actions/studio_runtime_system.py
"""Files, commands, and system telemetry for the concrete StudioRuntime.

Task 8 fills these in. For now every method raises NotImplementedError --
build_studio_runtime needs these three classes to exist so studio_runtime.py
imports, but a half-real implementation (a files browser that silently
returns nothing, a backup command that silently no-ops) is worse than an
obvious hole.
"""
from __future__ import annotations

from ..io.api.runtime import (
    BackupState, CommandDef, CommandOutcome, EnrollmentState, FileContent,
    FileEntry, StatusInfo, TelemetrySnapshot,
)


class LiveFileRuntime:
    async def roots(self) -> list[str]:
        raise NotImplementedError("Task 8")

    async def listing(self, path: str) -> list[FileEntry]:
        raise NotImplementedError("Task 8")

    async def read(self, path: str) -> FileContent:
        raise NotImplementedError("Task 8")

    async def rename(self, path: str, new_name: str) -> FileEntry:
        raise NotImplementedError("Task 8")

    async def delete(self, path: str) -> bool:
        raise NotImplementedError("Task 8")


class LiveCommandRuntime:
    async def catalogue(self) -> list[CommandDef]:
        raise NotImplementedError("Task 8")

    async def run(self, command_id: str) -> CommandOutcome:
        raise NotImplementedError("Task 8")


class LiveSystemRuntime:
    async def status(self) -> StatusInfo:
        raise NotImplementedError("Task 8")

    async def telemetry(self) -> TelemetrySnapshot:
        raise NotImplementedError("Task 8")

    async def backup_state(self) -> BackupState:
        raise NotImplementedError("Task 8")

    async def run_backup(self) -> BackupState:
        raise NotImplementedError("Task 8")

    async def restore_backup(self, recovery_phrase: str) -> bool:
        raise NotImplementedError("Task 8")

    async def enrollment(self) -> EnrollmentState:
        raise NotImplementedError("Task 8")

    async def forget_enrolled(self, kind: str, item_id: str) -> bool:
        raise NotImplementedError("Task 8")
