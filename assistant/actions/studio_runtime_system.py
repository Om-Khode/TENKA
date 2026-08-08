# assistant/actions/studio_runtime_system.py
"""Files, commands and system halves of the StudioRuntime.

Confinement lives in resolve_within, not in a route, so every caller gets it
whether or not a future route remembers to ask. Resolve first, then check: the
other order lets a symlink, a "sub/./../../x" walk, an alternate data stream,
or a reserved device name straight through.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from ..io.api.runtime import (
    BackupState, CommandDef, CommandOutcome, EnrolledItem, EnrollmentState,
    FileContent, FileEntry, StatusInfo, TelemetrySnapshot,
)

logger = logging.getLogger(__name__)

_ROOTS = ("desktop", "documents", "downloads")
_STARTED_AT = time.monotonic()

# Windows redirects these to hardware/pseudo devices regardless of which real
# directory addresses them -- "C:\Users\x\Desktop\CON" still opens the console
# device, not a file named CON inside Desktop. Checked against the resolved
# path's stem so both "CON" and "CON.txt" are caught.
_RESERVED_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def resolve_within(root_path: Path, relative_path: str) -> Path:
    """Resolve `relative_path` under `root_path`, or raise ValueError.

    Resolve-then-check. Checking a string for ".." before resolving misses
    symlinks, UNC paths, and every encoding trick; resolving first and then
    asking whether the real path is under the real root misses none of them.

    A colon anywhere in the input is refused outright, before resolving: on
    Windows it can only mean a drive letter or an alternate-data-stream
    separator, neither of which is a legal character in a plain relative
    path, and letting one through would just mean the join silently drops
    or twists it. After resolving, the candidate's final path is also
    checked against Windows' reserved device names, which get redirected to
    hardware even when addressed through an ordinary-looking directory path.
    """
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("empty path")
    stripped = relative_path.strip()
    if stripped in (".", ".."):
        raise ValueError("path is not a file")
    if ":" in relative_path:
        raise ValueError(f"path contains a reserved character: {relative_path}")

    root_real = Path(root_path).resolve(strict=False)
    candidate = (root_real / relative_path).resolve(strict=False)
    if candidate == root_real or root_real not in candidate.parents:
        raise ValueError(f"path escapes its root: {relative_path}")
    if candidate.stem.upper() in _RESERVED_DEVICE_NAMES:
        raise ValueError(f"path names a reserved device: {relative_path}")
    return candidate


_TEXT_SUFFIXES = {".md", ".txt", ".log", ".csv", ".json", ".yaml", ".yml", ".ini"}
_CODE_SUFFIXES = {
    ".py": "python", ".ts": "typescript", ".tsx": "tsx", ".js": "javascript",
    ".jsx": "jsx", ".css": "css", ".html": "html", ".sh": "bash", ".sql": "sql",
    ".rs": "rust", ".go": "go", ".java": "java", ".toml": "toml",
}
_IMAGE_SUFFIXES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                   ".gif": "image/gif", ".webp": "image/webp"}
_MAX_PREVIEW_BYTES = 512_000


def _classify(path: Path) -> tuple[str, str]:
    """Return (content_kind, language). Extension-driven, brand-free."""
    suffix = path.suffix.lower()
    if suffix in _CODE_SUFFIXES:
        return ("code", _CODE_SUFFIXES[suffix])
    if suffix in _TEXT_SUFFIXES:
        return ("text", "")
    if suffix in _IMAGE_SUFFIXES:
        return ("image", "")
    return ("binary", "")


class LiveFileRuntime:
    """Path-keyed: "desktop" and "desktop/captures" are both listable, and a
    node's id is its path, which is what the client's breadcrumb splits."""

    async def roots(self) -> list[str]:
        return sorted(_ROOTS)

    @staticmethod
    def _resolve(path: str) -> tuple[str, Path]:
        """Split "<root>/<rest>" into its root name and its real absolute path."""
        if not isinstance(path, str) or not path.strip():
            raise ValueError("empty path")
        normalised = path.replace("\\", "/").strip("/")
        root, _, rest = normalised.partition("/")
        if root not in _ROOTS:
            raise KeyError(root)
        from .. import file_manager
        root_path = file_manager.get_user_folder(root)
        if not rest:
            return (root, Path(root_path).resolve(strict=False))
        return (root, resolve_within(root_path, rest))

    async def listing(self, path: str) -> list[FileEntry]:
        root, target = self._resolve(path)
        return await asyncio.to_thread(self._listing_sync, root, target)

    @staticmethod
    def _listing_sync(root: str, target: Path) -> list[FileEntry]:
        from .. import file_manager
        root_path = Path(file_manager.get_user_folder(root)).resolve(strict=False)
        entries: list[FileEntry] = []
        for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            stat = child.stat()
            relative = child.relative_to(root_path).as_posix()
            content_kind = None if child.is_dir() else _classify(child)[0]
            entries.append(FileEntry(
                path=f"{root}/{relative}" if relative else root,
                name=child.name,
                kind="dir" if child.is_dir() else "file",
                size_bytes=0 if child.is_dir() else stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                content_kind=content_kind,
            ))
        return entries

    async def read(self, path: str) -> FileContent:
        _root, target = self._resolve(path)
        return await asyncio.to_thread(self._read_sync, path, target)

    @staticmethod
    def _read_sync(path: str, target: Path) -> FileContent:
        import base64
        content_kind, language = _classify(target)
        size = target.stat().st_size

        if content_kind == "image":
            if size > _MAX_PREVIEW_BYTES:
                return FileContent(path, "image", truncated=True)
            mime = _IMAGE_SUFFIXES[target.suffix.lower()]
            encoded = base64.b64encode(target.read_bytes()).decode("ascii")
            return FileContent(path, "image", f"data:{mime};base64,{encoded}")

        if content_kind == "binary":
            # file_manager.read_file already extracts text from documents; if it
            # cannot, the client renders "no preview" from an empty body.
            from .. import file_manager
            try:
                text = file_manager.read_file(target)
            except Exception:
                return FileContent(path, "binary")
            return FileContent(path, "binary", text[:_MAX_PREVIEW_BYTES],
                               truncated=len(text) > _MAX_PREVIEW_BYTES)

        # Bounded read: a multi-gigabyte .log under a listable root must not
        # be loaded whole before being truncated. read() on a text-mode file
        # object stops at the character cap itself, unlike read_text().
        with target.open("r", encoding="utf-8", errors="replace") as handle:
            text = handle.read(_MAX_PREVIEW_BYTES + 1)
        truncated = len(text) > _MAX_PREVIEW_BYTES
        return FileContent(path, content_kind, text[:_MAX_PREVIEW_BYTES], language,
                           truncated=truncated)

    async def rename(self, path: str, new_name: str) -> FileEntry:
        if "/" in new_name or "\\" in new_name or new_name in (".", ".."):
            raise ValueError("new name must be a bare file name")
        root, target = self._resolve(path)
        return await asyncio.to_thread(self._rename_sync, root, target, new_name)

    @staticmethod
    def _rename_sync(root: str, target: Path, new_name: str) -> FileEntry:
        from .. import file_manager
        if file_manager.is_protected_path(target):
            raise PermissionError(f"protected path: {target.name}")
        _message, new_path = file_manager.rename_path(target, new_name)
        root_path = Path(file_manager.get_user_folder(root)).resolve(strict=False)
        stat = new_path.stat()
        return FileEntry(
            path=f"{root}/{new_path.relative_to(root_path).as_posix()}",
            name=new_path.name,
            kind="dir" if new_path.is_dir() else "file",
            size_bytes=0 if new_path.is_dir() else stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            content_kind=None if new_path.is_dir() else _classify(new_path)[0],
        )

    async def delete(self, path: str) -> bool:
        _root, target = self._resolve(path)
        return await asyncio.to_thread(self._delete_sync, target)

    @staticmethod
    def _delete_sync(target: Path) -> bool:
        from .. import file_manager
        if file_manager.is_protected_path(target):
            raise PermissionError(f"protected path: {target.name}")
        if not target.exists():
            return False
        file_manager.delete_path(target)
        return True


class LiveCommandRuntime:
    """OS-level capabilities, not applications. Opening an app is a chat turn."""

    _CATALOGUE = (
        CommandDef("lock_workstation", "Lock PC", "Locks the desktop session.",
                   True, "system_control"),
        CommandDef("volume_up", "Volume Up", "Raises the system volume.",
                   False, "system_control"),
        CommandDef("volume_down", "Volume Down", "Lowers the system volume.",
                   False, "system_control"),
        CommandDef("screenshot", "Take Screenshot", "Captures the current screen.",
                   False, "screen"),
    )

    async def catalogue(self) -> list[CommandDef]:
        return list(self._CATALOGUE)

    async def run(self, command_id: str) -> CommandOutcome:
        known = {c.command_id for c in self._CATALOGUE}
        if command_id not in known:
            return CommandOutcome(False, "unknown command")
        try:
            return await asyncio.to_thread(self._run_sync, command_id)
        except Exception as exc:
            logger.warning(f"[API] command '{command_id}' failed: {exc}")
            return CommandOutcome(False, "command failed")

    @staticmethod
    def _run_sync(command_id: str) -> CommandOutcome:
        import ctypes
        if command_id == "lock_workstation":
            ctypes.windll.user32.LockWorkStation()
            return CommandOutcome(True, "locked")
        if command_id in ("volume_up", "volume_down"):
            key = 0xAF if command_id == "volume_up" else 0xAE
            ctypes.windll.user32.keybd_event(key, 0, 0, 0)
            ctypes.windll.user32.keybd_event(key, 0, 2, 0)
            return CommandOutcome(True, "volume changed")

        # screenshot: io/screen.py has no "capture_screen" -- the real
        # entry point is capture_screenshot(), which returns a PIL Image
        # (or None on failure) rather than writing a file itself. Saved
        # under the same SANDBOX_DIR/captures directory the vision loop
        # already uses -- io/backup/orchestrator.py excludes it from
        # archives for exactly this reason (regenerable, large).
        from .. import config
        from ..io import screen
        image = screen.capture_screenshot()
        if image is None:
            return CommandOutcome(False, "screenshot failed")
        capture_dir = config.SANDBOX_DIR / "captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        image.save(capture_dir / f"studio_{stamp}.png")
        return CommandOutcome(True, "captured")


class LiveSystemRuntime:
    async def status(self) -> StatusInfo:
        return await asyncio.to_thread(self._status_sync)

    @staticmethod
    def _status_sync() -> StatusInfo:
        from .. import config
        from ..llm.router import TASK_MODEL_MAP
        # config.ACTIVE_PERSONALITY is frozen at import time; the live switch
        # target is personalities.get_active_personality_id() (see
        # LivePersonalityRuntime._state_sync in studio_runtime.py).
        from ..personalities import get_active_personality_id
        default_chain = TASK_MODEL_MAP.get("default", [])
        active_model = str(default_chain[0][1]) if default_chain else ""
        return StatusInfo(
            assistant_name=str(config.ASSISTANT_NAME),
            # No app version constant exists anywhere in the codebase (checked
            # config.py, assistant/__init__.py, pyproject.toml). An empty
            # string is an honest "unknown"; inventing "1.0.0" would not be.
            version="",
            active_model=active_model,
            personality=get_active_personality_id(),
            busy=False,
        )

    async def telemetry(self) -> TelemetrySnapshot:
        return await asyncio.to_thread(self._telemetry_sync)

    @staticmethod
    def _telemetry_sync() -> TelemetrySnapshot:
        import psutil
        from ..llm.router import TASK_MODEL_MAP
        battery = psutil.sensors_battery()
        default_chain = TASK_MODEL_MAP.get("default", [])
        active_model = str(default_chain[0][1]) if default_chain else ""
        return TelemetrySnapshot(
            cpu_percent=psutil.cpu_percent(interval=None),
            ram_percent=psutil.virtual_memory().percent,
            battery_percent=battery.percent if battery else None,
            active_model=active_model,
            uptime_seconds=int(time.monotonic() - _STARTED_AT),
        )

    async def backup_state(self) -> BackupState:
        return await asyncio.to_thread(self._backup_state_sync)

    @staticmethod
    def _backup_state_sync() -> BackupState:
        from ..io.backup import orchestrator
        state = orchestrator.get_state()
        return BackupState(
            enabled=bool(state.get("enabled", False)),
            provider=str(state.get("provider", "google_drive")),
            last_backup_at=str(state.get("last_backup_at", "")),
            last_result=str(state.get("last_result", "")),
            size_bytes=int(state.get("size_bytes", 0)),
        )

    async def run_backup(self) -> BackupState:
        from ..io.backup import orchestrator
        await asyncio.to_thread(orchestrator.run_backup)
        return await self.backup_state()

    async def restore_backup(self, recovery_phrase: str) -> bool:
        from ..io.backup import crypto, orchestrator
        if not crypto.is_valid_recovery_phrase(recovery_phrase):
            return False
        await asyncio.to_thread(orchestrator.run_restore, recovery_phrase)
        return True

    async def enrollment(self) -> EnrollmentState:
        return await asyncio.to_thread(self._enrollment_sync)

    @staticmethod
    def _enrollment_sync() -> EnrollmentState:
        from .. import faces
        from ..io.audio import speaker_verify
        voices = [EnrolledItem("primary", "primary", "")] if speaker_verify.is_enrolled() else []
        known = [
            # faces.load_encodings() entries key their enrollment date as
            # "added" (and "updated"), not "added_at" -- there is no
            # "added_at" key anywhere in faces.py.
            EnrolledItem(entry.get("name", ""), entry.get("name", ""),
                        str(entry.get("added", "")))
            for entry in faces.load_encodings()
        ]
        seen, unique = set(), []
        for item in known:
            if item.item_id in seen:
                continue
            seen.add(item.item_id)
            unique.append(item)
        return EnrollmentState(voices=voices, faces=unique)

    async def forget_enrolled(self, kind: str, item_id: str) -> bool:
        return await asyncio.to_thread(self._forget_enrolled_sync, kind, item_id)

    @staticmethod
    def _forget_enrolled_sync(kind: str, item_id: str) -> bool:
        if kind == "voice":
            from ..io.audio import speaker_verify
            speaker_verify.clear_enrollment()
            return True
        if kind == "face":
            from .. import faces
            return faces.forget_face(item_id)
        return False
