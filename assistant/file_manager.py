"""
file_manager.py — File search and read operations for TENKA.
Windows-aware: uses SHGetKnownFolderPath to get real folder locations.
"""

import os
import logging
import ctypes
import ctypes.wintypes
from pathlib import Path

logger = logging.getLogger("file_manager")

READABLE_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".json", ".csv", ".log",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env", ".xml",
    ".html", ".htm", ".css", ".rst", ".bat", ".sh"
}

MAX_READ_CHARS = 3000

# Tier search timeout in seconds
TIER2_TIMEOUT = 15.0
TIER3_TIMEOUT = 90.0


def get_user_folder(name: str) -> Path:
    """
    Get real Windows user folder path using shell API.
    Falls back to Path.home() / name if shell API fails.
    Handles OneDrive redirection automatically.
    """
    # Known folder GUIDs for Windows shell folders
    FOLDER_GUIDS = {
        "desktop":   "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}",
        "documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
        "downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
        "pictures":  "{33E28130-4E1E-4676-835A-98395C3BC3BB}",
        "music":     "{4BD8D571-6D19-48D3-BE97-422220080E43}",
        "videos":    "{18989B1D-99B5-455B-841C-AB7C74E4DDFC}",
    }

    guid = FOLDER_GUIDS.get(name.lower())
    if guid:
        try:
            # Use SHGetKnownFolderPath via ctypes
            from ctypes import windll, wintypes
            # No comtypes import here as it wasn't mentioned in user's prompt but ctypes works fine

            # Parse GUID
            import uuid
            folder_id = uuid.UUID(guid)
            guid_bytes = (ctypes.c_byte * 16)(*folder_id.bytes_le)

            buf = ctypes.c_wchar_p()
            result = windll.shell32.SHGetKnownFolderPath(
                ctypes.byref(guid_bytes), 0, None, ctypes.byref(buf)
            )
            if result == 0 and buf.value:
                return Path(buf.value)
        except Exception as e:
            logger.debug(f"[FILE] SHGetKnownFolderPath failed for {name}: {e}")

    # Fallback: try common OneDrive path first, then plain home
    home = Path.home()
    onedrive = home / "OneDrive"
    candidates = {
        "desktop":   [home / "Desktop", onedrive / "Desktop"],
        "documents": [onedrive / "Documents", home / "Documents"],
        "downloads": [home / "Downloads"],
        "pictures":  [onedrive / "Pictures", home / "Pictures"],
        "music":     [home / "Music"],
        "videos":    [home / "Videos"],
    }
    for candidate in candidates.get(name.lower(), [home / name]):
        if candidate.exists():
            return candidate

    return home / name


def _get_tier1_folders() -> list[Path]:
    """Get all standard user folders that actually exist."""
    names = ["desktop", "documents", "downloads", "pictures", "music", "videos"]
    folders = []
    for name in names:
        p = get_user_folder(name)
        if p.exists():
            folders.append(p)
    return folders


def find_files(
    name: str,
    tier: int = 1,
    limit: int = 10,
    timeout_seconds: float = 90.0,
) -> list[Path]:
    """
    Search for files matching a name pattern across tiers.

    Tier 1 — Instant (~0.5s):
        Desktop, Documents, Downloads, Pictures, Music, Videos.
        Recursive, no timeout needed.

    Tier 2 — Fast (~5-15s):
        All available drives, 3 levels deep only.
        Runs with timeout safety.

    Tier 3 — Deep (~30-120s):
        All available drives, fully recursive.
        Stops at timeout_seconds or when limit reached.

    Args:
        name:            Filename or partial name (case-insensitive).
        tier:            Search depth tier (1, 2, or 3).
        limit:           Max results to return.
        timeout_seconds: Max seconds to spend searching (Tier 2/3 only).

    Returns:
        List of matching Path objects, up to limit results.
    """
    import time
    import string

    name_lower = name.lower().strip()
    matches = []
    deadline = time.time() + timeout_seconds

    def _add(p: Path) -> bool:
        """Add a match. Returns True if limit reached."""
        matches.append(p)
        return len(matches) >= limit

    # ── Tier 1: common user folders, fully recursive ──────────────────────
    for folder in _get_tier1_folders():
        try:
            for p in folder.rglob("*"):
                try:
                    if p.is_file() and name_lower in p.name.lower():
                        if _add(p):
                            return matches
                except (PermissionError, OSError):
                    continue
        except (PermissionError, OSError):
            continue

    if tier == 1 or matches:
        return matches

    # ── Tier 2: all drives, 3 levels deep ────────────────────────────────
    drives = [
        Path(f"{d}:\\")
        for d in string.ascii_uppercase
        if Path(f"{d}:\\").exists()
    ]

    # Skip drives already covered by Tier 1 (avoid re-scanning home drive root)
    tier1_roots = {p.anchor for p in _get_tier1_folders()}

    def _scan_dir(folder: Path, current_depth: int, max_depth: int) -> bool:
        """Recursively scan up to max_depth. Returns True if limit reached."""
        if time.time() > deadline:
            return False
        try:
            for item in folder.iterdir():
                if time.time() > deadline:
                    return False
                try:
                    if item.is_file() and name_lower in item.name.lower():
                        if _add(item):
                            return True
                    elif item.is_dir() and current_depth < max_depth:
                        if _scan_dir(item, current_depth + 1, max_depth):
                            return True
                except (PermissionError, OSError):
                    continue
        except (PermissionError, OSError):
            pass
        return False

    max_depth = 3 if tier == 2 else 999  # 999 = effectively unlimited for Tier 3

    for drive in drives:
        if time.time() > deadline:
            logger.info(f"[FILE] Tier {tier} search timed out after {timeout_seconds}s")
            break
        try:
            for item in drive.iterdir():
                if time.time() > deadline:
                    break
                try:
                    if item.is_file() and name_lower in item.name.lower():
                        if _add(item):
                            return matches
                    elif item.is_dir():
                        if _scan_dir(item, 1, max_depth):
                            return matches
                except (PermissionError, OSError):
                    continue
        except (PermissionError, OSError):
            continue

    return matches


def read_file(path: Path) -> str:
    """Read a text file and return its contents."""
    if not path.exists():
        return f"File not found: {path}"
    if not path.is_file():
        return f"That path is a folder, not a file."
    if path.suffix.lower() not in READABLE_EXTENSIONS:
        return (
            f"I can't read {path.suffix} files — only text formats like "
            f".txt, .md, .py, .json, .csv, .log and similar."
        )
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            content = path.read_text(encoding=encoding)
            if len(content) > MAX_READ_CHARS:
                content = content[:MAX_READ_CHARS] + f"\n... (truncated)"
            return content
        except UnicodeDecodeError:
            continue
        except PermissionError:
            return "I don't have permission to read that file."
        except Exception as e:
            return f"Error reading file: {e}"
    return "Could not decode the file — it may be binary."


def list_folder(path: Path, extensions: list[str] | None = None) -> list[dict]:
    """List contents of a folder, safely skipping broken symlinks."""
    if not path.exists() or not path.is_dir():
        return []
    results = []
    try:
        for item in sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
            try:
                # Skip broken symlinks and shell shortcuts
                if not item.exists():
                    continue
                if extensions and item.is_file() and item.suffix.lower() not in extensions:
                    continue
                stat = item.stat()
                results.append({
                    "name": item.name,
                    "type": "file" if item.is_file() else "folder",
                    "size_kb": round(stat.st_size / 1024, 1) if item.is_file() else None,
                    "modified": stat.st_mtime,
                    "path": item,
                })
            except (PermissionError, OSError):
                continue
    except (PermissionError, OSError):
        return []
    return results[:50]


def open_path(path: Path) -> str:
    """Open a file or folder with its default application."""
    if not path.exists():
        return f"Could not find: {path}"
    try:
        os.startfile(str(path))
        return f"Opened {path.name}"
    except Exception as e:
        return f"Failed to open {path.name}: {e}"


def get_file_info(path: Path) -> dict:
    """Get metadata about a file."""
    if not path.exists():
        return {}
    try:
        from datetime import datetime
        stat = path.stat()
        return {
            "name": path.name,
            "extension": path.suffix.lower(),
            "size_kb": round(stat.st_size / 1024, 1),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            "created": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M"),
            "path": str(path),
        }
    except Exception as e:
        logger.warning(f"[FILE] get_file_info failed: {e}")
        return {}


def write_file(path: Path, content: str) -> str:
    """Create or overwrite a file with UTF-8 encoding."""
    logger.info(f"[FILE] write_file: {path}")
    try:
        is_overwrite = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        content_bytes = content.encode("utf-8")
        path.write_bytes(content_bytes)
        
        action = "Overwrote" if is_overwrite else "Created"
        return f"{action} {path.name} in {path.parent} ({len(content_bytes)} bytes)"
    except Exception as e:
        return f"Error: {e}"


def rename_path(path: Path, new_name: str) -> tuple[str, Path]:
    """Rename a file or folder, preserving original extension if not provided."""
    logger.info(f"[FILE] rename_path: {path} -> {new_name}")
    try:
        final_name = new_name
        if path.is_file() and "." not in new_name and path.suffix:
            final_name = f"{new_name}{path.suffix}"
            
        new_path = path.with_name(final_name)
        new_path = path.rename(new_path)
        return (f"Renamed '{path.name}' to '{new_path.name}'", new_path)
    except Exception as e:
        return (f"Error: {e}", path)


def move_path(src: Path, dest_folder: Path) -> tuple[str, Path]:
    """Move a file or folder into an existing destination folder."""
    logger.info(f"[FILE] move_path: {src} -> {dest_folder}")
    try:
        if not dest_folder.exists() or not dest_folder.is_dir():
            return (f"Error: Destination folder is missing or invalid: {dest_folder}", src)
            
        new_path = dest_folder / src.name
        if new_path.exists():
            if src.is_file():
                new_path = dest_folder / f"{src.stem} (moved){src.suffix}"
            else:
                new_path = dest_folder / f"{src.name} (moved)"
                
        import shutil
        shutil.move(str(src), str(new_path))
        return (f"Moved '{src.name}' to {dest_folder}", new_path)
    except Exception as e:
        return (f"Error: {e}", src)


def delete_path(path: Path) -> str:
    """Send a file or folder to the Recycle Bin using send2trash."""
    logger.info(f"[FILE] delete_path: {path}")
    try:
        from send2trash import send2trash
        send2trash(str(path.resolve()))
        return f"Deleted '{path.name}' — it's in your Recycle Bin if you need it back."
    except Exception as e:
        return f"Error: {e}"


def is_protected_path(path: Path) -> bool:
    """Check if a path is in a protected system location or is a drive root."""
    logger.info(f"[FILE] is_protected_path: {path}")
    try:
        resolved = path.resolve()
        
        if resolved == Path(resolved.anchor):
            return True
            
        protected_locations = [
            Path(r"C:\Windows").resolve(),
            Path(r"C:\Program Files").resolve(),
            Path(r"C:\Program Files (x86)").resolve(),
            Path(r"C:\System Volume Information").resolve(),
            (Path.home() / "AppData").resolve(),
            Path(__file__).resolve().parent,
        ]
        
        for protected in protected_locations:
            if resolved.is_relative_to(protected):
                return True
                
        return False
    except Exception:
        return True
