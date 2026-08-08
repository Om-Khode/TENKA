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


# Every extension file_manager.READABLE_EXTENSIONS accepts must land in one
# of these two sets, not the "binary" default -- read_file()'s plain-text
# fallback for a READABLE_EXTENSIONS suffix is an unbounded path.read_text(),
# and only MAX_PREVIEW_BYTES-capped _classify branches (text/code/image) read
# through the bounded path in _read_sync below. Missing one here means a
# large file of that type gets read into memory whole before truncation --
# exactly the memory spike the bounded read exists to avoid.
_TEXT_SUFFIXES = {
    ".md", ".txt", ".log", ".csv", ".json", ".yaml", ".yml", ".ini",
    ".cfg", ".env", ".rst",
}
_CODE_SUFFIXES = {
    ".py": "python", ".ts": "typescript", ".tsx": "tsx", ".js": "javascript",
    ".jsx": "jsx", ".css": "css", ".html": "html", ".htm": "html",
    ".sh": "bash", ".sql": "sql", ".rs": "rust", ".go": "go", ".java": "java",
    ".toml": "toml", ".xml": "xml", ".bat": "bat",
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
    node's id is its path, which is what the client's breadcrumb splits.

    Every method below does its path resolution *inside* the to_thread call,
    not before it: file_manager.get_user_folder() is a shell32 COM call with
    a filesystem-probing fallback, and resolve_within() calls Path.resolve(),
    a syscall. Both are blocking I/O; running either one directly on the
    caller's coroutine would block the assistant's event loop for as long as
    they take, the same rule Task 7 has a test for on the settings/memory
    facades.
    """

    async def roots(self) -> list[str]:
        return sorted(_ROOTS)

    @staticmethod
    def _resolve(path: str, *, allow_bare_root: bool = False) -> tuple[str, Path]:
        """Split "<root>/<rest>" into its root name and its real absolute path.

        allow_bare_root=False (the default, and every caller except
        listing()) refuses a path that names the root directory itself --
        directly ("desktop") or via a traversal that cancels back down to it
        ("desktop/sub/.."; resolve_within's own containment check already
        catches this shape once there is a non-empty rest, since the
        resolved candidate equals root_real). Without this, delete("desktop")
        sends the user's entire Desktop to the Recycle Bin in one call, and
        rename("desktop", "x") renames the folder out from under every other
        listing -- file_manager.is_protected_path() guards Windows, Program
        Files and drive roots, never these three user-data roots, so nothing
        downstream catches this on its own.
        """
        if not isinstance(path, str) or not path.strip():
            raise ValueError("empty path")
        normalised = path.replace("\\", "/").strip("/")
        root, _, rest = normalised.partition("/")
        if root not in _ROOTS:
            raise KeyError(root)
        from .. import file_manager
        root_path = file_manager.get_user_folder(root)
        if not rest:
            if not allow_bare_root:
                raise ValueError(f"path must not be the root itself: {path}")
            return (root, Path(root_path).resolve(strict=False))
        return (root, resolve_within(root_path, rest))

    async def listing(self, path: str) -> list[FileEntry]:
        return await asyncio.to_thread(self._listing_thread, path)

    @classmethod
    def _listing_thread(cls, path: str) -> list[FileEntry]:
        root, target = cls._resolve(path, allow_bare_root=True)
        return cls._listing_sync(root, target)

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
        return await asyncio.to_thread(self._read_thread, path)

    @classmethod
    def _read_thread(cls, path: str) -> FileContent:
        _root, target = cls._resolve(path)
        return cls._read_sync(path, target)

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
            # Every plain-text READABLE_EXTENSIONS suffix is classified as
            # "text" or "code" above and never reaches here -- this branch is
            # now only rich documents (file_manager's own extractors: pypdf /
            # python-docx / openpyxl / python-pptx, each already bounded by
            # MAX_READ_CHARS after parsing) and genuinely unreadable
            # extensions, for which read_file() returns a short message
            # without reading the file at all. Neither does an unbounded
            # plain read_text() the way the six-suffix gap used to.
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
        return await asyncio.to_thread(self._rename_thread, path, new_name)

    @classmethod
    def _rename_thread(cls, path: str, new_name: str) -> FileEntry:
        root, target = cls._resolve(path)
        return cls._rename_sync(root, target, new_name)

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
        return await asyncio.to_thread(self._delete_thread, path)

    @classmethod
    def _delete_thread(cls, path: str) -> bool:
        _root, target = cls._resolve(path)
        return cls._delete_sync(target)

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
            # LockWorkStation is documented to return a nonzero BOOL on
            # success and 0 on failure (call GetLastError for why) --
            # unlike keybd_event below, it actually has a checkable failure
            # signal. This command is destructive=True in the catalogue;
            # reporting "locked" without checking it would be a false
            # positive on a security-relevant action.
            ok = bool(ctypes.windll.user32.LockWorkStation())
            return CommandOutcome(ok, "locked" if ok else "lock failed")
        if command_id in ("volume_up", "volume_down"):
            # keybd_event is documented VOID -- Win32 gives it no return
            # value to check at all (unlike LockWorkStation above), so there
            # is nothing honest to verify here without switching to the
            # Core Audio API (IAudioEndpointVolume), which is a bigger
            # change than this fix. Flagged in the task report rather than
            # fabricating a check against a value the API doesn't provide.
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
        # speaker_verify exposes is_enrolled()/clear_enrollment() only -- no
        # public accessor for a sample count or a last-heard timestamp
        # (_enrolled_embeddings is module-private). count=None and
        # last_seen_at="" are the honest values, not zeros: a real "0
        # samples" would misstate an enrollment that plainly exists.
        voices = (
            [EnrolledItem("primary", "primary", "", count=None, last_seen_at="")]
            if speaker_verify.is_enrolled() else []
        )
        known = [
            # faces.load_encodings() entries key their enrollment date as
            # "added" (and "updated"), not "added_at" -- there is no
            # "added_at" key anywhere in faces.py. encoding_count() is a
            # real public accessor; "updated" (last time this person's
            # stored encoding set changed) is the closest honest signal for
            # "last seen" that exists -- there is no live-recognition log to
            # draw an exact one from.
            EnrolledItem(entry.get("name", ""), entry.get("name", ""),
                        str(entry.get("added", "")),
                        count=faces.encoding_count(entry.get("name", "")),
                        last_seen_at=str(entry.get("updated", "")))
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
