"""
browser_setup.py — Chrome shortcut auto-setup for CDP.

One-time setup that lets TENKA's DOM-mode attach to the
user's Chrome via `--remote-debugging-port=9222`. The script:

  1. Detects Chrome's installation path (Windows registry + Program Files
     fallbacks)
  2. Creates a user-owned shortcut "Chrome (TENKA-CDP).lnk" on the user's
     Desktop AND in the user's Start Menu, both pointing at chrome.exe
     with `--remote-debugging-port=<port>`. Both locations are under the
     user profile — no admin needed even when Chrome is system-installed.
  3. The user's existing Chrome shortcuts are NOT touched. The user
     launches Chrome from "Chrome (TENKA-CDP)" when they want CDP enabled.
  4. Writes a marker file at `~/.tenka/chrome_cdp_setup.json` so future
     startups know setup is done.

Idempotent + reversible: `undo_chrome_cdp_setup()` deletes the created
shortcuts and the marker.

Pure stdlib + PowerShell subprocess — no pywin32 dependency. PowerShell ships
with every modern Windows.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("browser_setup")


# ─── Paths + constants ────────────────────────────────────────────────────────


def _user_dir() -> Path:
    """The .tenka home where the marker lives. ~/.tenka/."""
    return Path.home() / ".tenka"


def _marker_path() -> Path:
    return _user_dir() / "chrome_cdp_setup.json"


_BACKUP_SUFFIX = ".tenka.bak"
_DEFAULT_PORT = 9222

# Marker schema version. Bump whenever the shortcut args / setup behaviour
# changes in a way that requires re-creating an existing shortcut. is_setup_done
# returns False when the on-disk marker's version is older than this, which
# forces setup_chrome_cdp to rewrite the shortcut with the current args.
#
# v1: --remote-debugging-port=PORT only
# v2: --remote-debugging-port=PORT --user-data-dir=<dedicated profile>
#     (without --user-data-dir, opening the shortcut while user's main Chrome
#      runs merges into the existing flag-less process and CDP is dropped)
_SETUP_SCHEMA_VERSION = 2

# Chrome (and Chromium-based: Edge, Brave, etc.) Application directories
# we probe when the registry doesn't give a hit.
_DEFAULT_CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
]


# ─── Result types ─────────────────────────────────────────────────────────────


@dataclass
class ShortcutModification:
    """One shortcut that was (or would have been) modified."""
    path: str
    backup_path: str
    original_args: str = ""
    new_args: str = ""


@dataclass
class SetupResult:
    """
    Outcome of `setup_chrome_cdp()`.

    `ok`             — True iff at least one shortcut was modified successfully
                       OR the marker indicates setup was already done
    `message`        — TTS-friendly summary the caller can speak verbatim
    `modified`       — list of ShortcutModification for shortcuts we changed
    `skipped`        — list of (path, reason) for shortcuts we skipped
    `chrome_running` — True if we refused because Chrome was running
    `chrome_exe`     — detected Chrome executable path (informational)
    `port`           — the debug port we configured for
    """
    ok: bool
    message: str
    modified: list[ShortcutModification] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    chrome_running: bool = False
    chrome_exe: str = ""
    port: int = _DEFAULT_PORT


@dataclass
class UndoResult:
    """Outcome of `undo_chrome_cdp_setup()`."""
    ok: bool
    message: str
    restored: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)


# ─── Marker file helpers ──────────────────────────────────────────────────────


def is_setup_done(port: int = _DEFAULT_PORT) -> bool:
    """True iff:
      - a marker exists for the given port, AND
      - the marker's schema version matches `_SETUP_SCHEMA_VERSION` (otherwise
        the on-disk shortcut was created with an older args format and needs
        rewriting), AND
      - at least one of the shortcuts it references still exists on disk.

    Returning False forces `setup_chrome_cdp` to re-create the shortcut with
    the current args. This is the upgrade path when we change the shortcut
    contract (e.g. adding `--user-data-dir`).
    """
    p = _marker_path()
    if not p.is_file():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if int(data.get("port", 0)) != port:
            return False
        if int(data.get("version", 0)) < _SETUP_SCHEMA_VERSION:
            logger.info(
                f"[BROWSER_SETUP] marker schema version "
                f"{data.get('version')} < current {_SETUP_SCHEMA_VERSION} — "
                f"forcing re-setup to apply updated shortcut args"
            )
            return False
        entries = data.get("shortcuts_modified", []) or []
        if not entries:
            return False
        for entry in entries:
            sc = Path(entry.get("path", ""))
            if sc.is_file():
                return True
        return False
    except (OSError, json.JSONDecodeError, ValueError):
        return False


def _write_marker(result: SetupResult) -> None:
    """Persist the marker after a successful setup."""
    payload = {
        "version": _SETUP_SCHEMA_VERSION,
        "configured_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "port": result.port,
        "chrome_executable": result.chrome_exe,
        "shortcuts_modified": [asdict(m) for m in result.modified],
    }
    _user_dir().mkdir(parents=True, exist_ok=True)
    _marker_path().write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _read_marker() -> Optional[dict]:
    p = _marker_path()
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _delete_marker() -> bool:
    p = _marker_path()
    try:
        if p.is_file():
            p.unlink()
        return True
    except OSError:
        return False


# ─── Chrome detection ─────────────────────────────────────────────────────────


def find_chrome_executable() -> Optional[Path]:
    """
    Locate Chrome's chrome.exe via Windows registry first, then well-known
    Program Files paths. Returns None if no Chromium-based browser is found.
    """
    # Registry probe: HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe
    if sys.platform == "win32":
        try:
            import winreg
            for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                try:
                    with winreg.OpenKey(
                        hive,
                        r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe",
                    ) as key:
                        val, _ = winreg.QueryValueEx(key, "")
                        path = Path(val)
                        if path.is_file():
                            return path
                except OSError:
                    continue
        except ImportError:
            pass

    for candidate in _DEFAULT_CHROME_PATHS:
        p = Path(candidate)
        if p.is_file():
            return p
    return None


def is_chrome_running() -> bool:
    """
    True if any chrome.exe / msedge.exe / brave.exe process is alive.
    Used by `setup_chrome_cdp` to refuse setup while a browser is up
    (modifying a shortcut while the browser is running and then relaunching
    forwards args to the existing process — the flag would silently drop).
    """
    try:
        import psutil
    except ImportError:
        # Without psutil we can't tell; assume running to be safe.
        return True

    target_names = {"chrome.exe", "msedge.exe", "brave.exe"}
    try:
        for proc in psutil.process_iter(["name"]):
            try:
                name = (proc.info.get("name") or "").lower()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if name in target_names:
                return True
    except Exception:
        return True
    return False


# ─── Shortcut discovery ───────────────────────────────────────────────────────


def find_chrome_shortcuts() -> list[Path]:
    """
    Locate all .lnk files on disk that point at a Chromium-based browser
    executable. Searches:
      - User Desktop (%USERPROFILE%\\Desktop)
      - Public Desktop (%PUBLIC%\\Desktop)
      - User Start Menu (%APPDATA%\\Microsoft\\Windows\\Start Menu)
      - Common Start Menu (%PROGRAMDATA%\\Microsoft\\Windows\\Start Menu)

    Pinned-taskbar shortcuts are intentionally NOT modified — Windows has
    a separate `PinnedFiles` cache that re-syncs from the original target,
    making taskbar tweaks fragile. Users who want CDP via the taskbar can
    unpin → relaunch from the modified Desktop shortcut → repin.

    Returns deduplicated paths (in case the same shortcut appears under
    multiple roots via symlinks).
    """
    if sys.platform != "win32":
        return []

    candidates: set[Path] = set()
    roots: list[Path] = []

    home = Path.home()
    public = Path(os.environ.get("PUBLIC", r"C:\Users\Public"))
    appdata = Path(os.environ.get("APPDATA", str(home / "AppData" / "Roaming")))
    programdata = Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData"))

    roots.append(home / "Desktop")
    roots.append(public / "Desktop")
    roots.append(appdata / "Microsoft" / "Windows" / "Start Menu")
    roots.append(programdata / "Microsoft" / "Windows" / "Start Menu")

    chrome_exe_lower_keys = ("chrome.exe", "msedge.exe", "brave.exe")

    for root in roots:
        if not root.is_dir():
            continue
        try:
            # Recursive walk for Start Menu (which has nested folders).
            for path in root.rglob("*.lnk"):
                target = _read_lnk_target(path)
                if not target:
                    continue
                target_lower = target[0].lower()
                if any(target_lower.endswith(k) for k in chrome_exe_lower_keys):
                    candidates.add(path.resolve())
        except OSError:
            continue

    return sorted(candidates)


# ─── PowerShell-based .lnk read/write ─────────────────────────────────────────


def _ps_quote(value: str) -> str:
    """
    Wrap `value` as a PowerShell single-quoted literal string. The only
    escape inside single quotes is doubling embedded apostrophes — no
    other character is interpreted. Bulletproof against paths containing
    parentheses, spaces, ampersands, etc. that break `-Command "$args[0]"`
    passing on Windows.
    """
    return "'" + value.replace("'", "''") + "'"


def _ps_run(script: str, *, timeout: float = 10.0) -> Optional[str]:
    """
    Run a PowerShell script (as a single string) via subprocess. Returns
    stdout on success or None on any failure. Uses `-NoProfile` for speed
    and reproducibility.

    The script must be self-contained — embed any path/value via
    `_ps_quote()` rather than `$args[N]` (PowerShell's `-Command` arg-passing
    has parser quirks with parentheses in paths, e.g. `Python 3.11 (64-bit)`,
    that we side-step entirely with literal substitution).
    """
    if sys.platform != "win32":
        return None
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy", "Bypass",
        "-Command", script,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning(f"[BROWSER_SETUP] PowerShell run failed: {e}")
        return None
    if result.returncode != 0:
        # Demoted to debug — find_chrome_shortcuts iterates over hundreds of
        # .lnk files in Start Menu, many of which are non-shortcut junk
        # (Python 3.11 (64-bit).lnk pointing at a folder, etc.). Logging
        # each as a warning would spam the console with non-actionable noise.
        logger.debug(
            f"[BROWSER_SETUP] PowerShell exit={result.returncode} "
            f"stderr={result.stderr.strip()[:120]!r}"
        )
        return None
    return result.stdout


def _read_lnk_target(path: Path) -> Optional[tuple[str, str, str]]:
    """
    Read a .lnk file's (target_exe, arguments, working_directory).
    Returns None on failure (PowerShell error, file not a shortcut, etc.).
    """
    quoted = _ps_quote(str(path))
    script = (
        f"$path = {quoted}; "
        '$shell = New-Object -ComObject WScript.Shell; '
        '$sc = $shell.CreateShortcut($path); '
        'Write-Output $sc.TargetPath; '
        'Write-Output $sc.Arguments; '
        'Write-Output $sc.WorkingDirectory'
    )
    out = _ps_run(script)
    if not out:
        return None
    # PowerShell returns the three Write-Outputs separated by newlines.
    lines = out.replace("\r\n", "\n").rstrip("\n").split("\n")
    if len(lines) < 1:
        return None
    target = lines[0].strip() if len(lines) > 0 else ""
    arguments = lines[1].strip() if len(lines) > 1 else ""
    workdir = lines[2].strip() if len(lines) > 2 else ""
    if not target:
        return None
    return target, arguments, workdir


def _write_lnk_arguments(path: Path, new_arguments: str) -> bool:
    """Set the Arguments field of a .lnk. Returns True on success."""
    quoted_path = _ps_quote(str(path))
    quoted_args = _ps_quote(new_arguments)
    script = (
        f"$path = {quoted_path}; "
        f"$newArgs = {quoted_args}; "
        '$shell = New-Object -ComObject WScript.Shell; '
        '$sc = $shell.CreateShortcut($path); '
        '$sc.Arguments = $newArgs; '
        '$sc.Save(); '
        'Write-Output OK'
    )
    out = _ps_run(script)
    return out is not None and out.strip().endswith("OK")


def _create_lnk(
    path: Path, *, target: str, arguments: str, workdir: str, icon: str = "",
) -> bool:
    """
    Create (or overwrite) a .lnk at `path` pointing at `target` with the
    given arguments + working directory. Optional `icon` lets the shortcut
    inherit Chrome's own icon so it looks identical to a normal Chrome
    shortcut. Returns True on success.
    """
    quoted_path = _ps_quote(str(path))
    quoted_target = _ps_quote(target)
    quoted_args = _ps_quote(arguments)
    quoted_workdir = _ps_quote(workdir)
    quoted_icon = _ps_quote(icon)
    script = (
        f"$path = {quoted_path}; "
        f"$target = {quoted_target}; "
        f"$lnkArgs = {quoted_args}; "
        f"$wd = {quoted_workdir}; "
        f"$icon = {quoted_icon}; "
        '$shell = New-Object -ComObject WScript.Shell; '
        '$sc = $shell.CreateShortcut($path); '
        '$sc.TargetPath = $target; '
        '$sc.Arguments = $lnkArgs; '
        '$sc.WorkingDirectory = $wd; '
        'if ($icon) { $sc.IconLocation = $icon }; '
        '$sc.Save(); '
        'Write-Output OK'
    )
    out = _ps_run(script)
    return out is not None and out.strip().endswith("OK")


# ─── Args manipulation ────────────────────────────────────────────────────────


_REMOTE_DEBUG_RE = re.compile(r"--remote-debugging-port=\d+", re.IGNORECASE)


def _ensure_cdp_flag(current_args: str, port: int) -> tuple[str, bool]:
    """
    Return (new_args, was_modified). Idempotent: if `--remote-debugging-port=`
    is already present, the flag's value is replaced if it differs and
    was_modified=True; otherwise returns the input unchanged with False.

    Preserves any other flags (e.g. `--profile-directory=Default`).
    """
    desired = f"--remote-debugging-port={port}"
    cur = current_args or ""
    match = _REMOTE_DEBUG_RE.search(cur)
    if match:
        existing = match.group(0)
        if existing.lower() == desired.lower():
            return cur, False  # already configured
        # Replace the existing flag's value
        new_args = _REMOTE_DEBUG_RE.sub(desired, cur).strip()
        return new_args, True
    # Append. Strip + " " + flag.
    if cur.strip():
        return f"{cur.strip()} {desired}", True
    return desired, True


# ─── Main entry points ────────────────────────────────────────────────────────


_CDP_LNK_NAME = "Chrome (TENKA-CDP).lnk"


def _cdp_profile_dir() -> Path:
    """Dedicated Chrome user-data-dir for CDP sessions.

    CRITICAL — this is what makes the TENKA shortcut reliable. Without a
    separate profile dir, launching Chrome from the shortcut while the
    user's main Chrome is running just adds a tab to the existing
    (flag-less) process and the `--remote-debugging-port=9222` arg is
    silently dropped. With a dedicated profile, the shortcut always spawns
    its own Chrome instance and CDP is reliably exposed.
    """
    return _user_dir() / "chrome-profile"


def _resolve_known_folder(folder: str) -> Optional[Path]:
    """
    Resolve a Windows known folder via [Environment]::GetFolderPath. This
    respects OneDrive redirection (where the user's Desktop lives at
    `~/OneDrive/Desktop` instead of `~/Desktop`). Returns None if PowerShell
    is unavailable or the folder can't be resolved.

    Valid `folder` values include 'Desktop', 'Programs', 'StartMenu'.
    """
    if sys.platform != "win32":
        return None
    quoted = _ps_quote(folder)
    out = _ps_run(f"[Environment]::GetFolderPath({quoted})")
    if not out:
        return None
    path = out.strip().splitlines()[0].strip() if out.strip() else ""
    if not path:
        return None
    p = Path(path)
    return p if p.is_dir() else None


def _shortcut_targets() -> list[Path]:
    """
    Return user-writable locations where the TENKA Chrome shortcut should
    be placed: user Desktop + user Start Menu. Both are always under the
    user's profile — no admin needed even when Chrome is system-installed.

    Uses Windows' Known Folders API (via PowerShell) so OneDrive folder
    redirection is handled correctly. Falls back to env vars / Path.home()
    if the API call fails.
    """
    home = Path.home()
    appdata = Path(os.environ.get("APPDATA", str(home / "AppData" / "Roaming")))

    desktop = _resolve_known_folder("Desktop")
    if desktop is None:
        # Probe the common locations (plain Desktop + a couple of OneDrive
        # variants) so the fallback handles real-world OneDrive setups.
        for candidate in (
            home / "Desktop",
            home / "OneDrive" / "Desktop",
            Path(os.environ.get("OneDrive", "")) / "Desktop" if os.environ.get("OneDrive") else None,
        ):
            if candidate and candidate.is_dir():
                desktop = candidate
                break

    start_programs = _resolve_known_folder("Programs")
    if start_programs is None:
        candidate = appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        if candidate.is_dir():
            start_programs = candidate

    targets: list[Path] = []
    if desktop is not None:
        targets.append(desktop / _CDP_LNK_NAME)
    if start_programs is not None:
        targets.append(start_programs / _CDP_LNK_NAME)
    return targets


def setup_chrome_cdp(
    *,
    port: int = _DEFAULT_PORT,
    dry_run: bool = False,
    require_chrome_closed: bool = False,  # No longer needed — we don't touch existing shortcuts
) -> SetupResult:
    """
    Create a user-owned Chrome shortcut that launches Chrome with
    `--remote-debugging-port=<port>`. Placed on the user's Desktop and in
    the user's Start Menu (both user-writable — no admin required).

    The user's existing Chrome shortcuts are NOT touched. The user launches
    Chrome from "Chrome (TENKA-CDP)" when they want CDP enabled, otherwise
    uses their normal shortcut.

    `dry_run=True` reports what WOULD be created without writing files.
    Marker file is NOT written in dry-run mode.

    Idempotent: if the marker says setup is already done for this port and
    at least one created shortcut still exists, returns ok=True with an
    "already configured" message.
    """
    chrome_exe = find_chrome_executable()
    if chrome_exe is None:
        return SetupResult(
            ok=False,
            message="Couldn't find Chrome. Install it first.",
            port=port,
        )

    if is_setup_done(port=port):
        return SetupResult(
            ok=True,
            message=f"Chrome's already set up on port {port}.",
            chrome_exe=str(chrome_exe),
            port=port,
        )

    targets = _shortcut_targets()
    if not targets:
        return SetupResult(
            ok=False,
            message="Couldn't find your Desktop or Start Menu folder.",
            chrome_exe=str(chrome_exe),
            port=port,
        )

    profile = _cdp_profile_dir()
    profile.mkdir(parents=True, exist_ok=True)
    # Both flags matter: --user-data-dir guarantees Chrome spawns a new
    # process (so the CDP flag isn't dropped when user's main Chrome is
    # running), and --remote-debugging-port exposes the CDP endpoint.
    args = (
        f'--remote-debugging-port={port} '
        f'--user-data-dir="{profile}"'
    )
    workdir = str(chrome_exe.parent)
    icon = f"{chrome_exe},0"  # use chrome.exe's own icon

    created: list[ShortcutModification] = []
    failed: list[tuple[str, str]] = []

    for sc_path in targets:
        if dry_run:
            created.append(ShortcutModification(
                path=str(sc_path), backup_path="",
                original_args="", new_args=args,
            ))
            continue
        if _create_lnk(
            sc_path, target=str(chrome_exe),
            arguments=args, workdir=workdir, icon=icon,
        ):
            created.append(ShortcutModification(
                path=str(sc_path), backup_path="",
                original_args="", new_args=args,
            ))
        else:
            failed.append((str(sc_path), "couldn't create shortcut"))

    if not created and not dry_run:
        return SetupResult(
            ok=False,
            message="Couldn't create the TENKA Chrome shortcut.",
            modified=created, skipped=failed,
            chrome_exe=str(chrome_exe), port=port,
        )

    if dry_run:
        message = f"Would create {len(created)} TENKA Chrome shortcut(s)."
    else:
        # Short, TTS-friendly. No paths, no error codes.
        message = "Done. Launch Chrome from the new TENKA shortcut on your Desktop."

    result = SetupResult(
        ok=True, message=message,
        modified=created, skipped=failed,
        chrome_exe=str(chrome_exe), port=port,
    )

    if not dry_run:
        try:
            _write_marker(result)
        except OSError as e:
            logger.warning(f"[BROWSER_SETUP] could not write marker: {e}")

    return result


def undo_chrome_cdp_setup() -> UndoResult:
    """
    Reverse `setup_chrome_cdp`. Deletes the TENKA-created shortcuts and
    the marker. Safe to call when no marker exists.
    """
    marker = _read_marker()
    if marker is None:
        return UndoResult(ok=True, message="Chrome setup wasn't configured.")

    restored: list[str] = []
    failed: list[tuple[str, str]] = []
    for entry in marker.get("shortcuts_modified", []) or []:
        sc = Path(entry.get("path", ""))
        if not entry.get("path"):
            continue
        if not sc.is_file():
            # Already gone — count as success
            restored.append(str(sc))
            continue
        try:
            sc.unlink()
            restored.append(str(sc))
        except OSError as e:
            failed.append((str(sc), f"delete failed: {e}"))

    _delete_marker()
    n = len(restored)
    return UndoResult(
        ok=True,
        message=(
            f"Removed {n} TENKA Chrome shortcut(s)."
            if n else "Nothing to remove."
        ),
        restored=restored,
        failed=failed,
    )


# ─── CLI entry point ─────────────────────────────────────────────────────────


def _main_cli() -> int:
    """Run from terminal: `python -m assistant.browser_setup [--undo|--dry-run]`."""
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    undo = "--undo" in args

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if undo:
        result = undo_chrome_cdp_setup()
        print(result.message)
        if result.failed:
            for path, reason in result.failed:
                print(f"  ! {path}: {reason}")
        return 0 if result.ok else 1

    result = setup_chrome_cdp(dry_run=dry_run)
    print(result.message)
    for m in result.modified:
        verb = "Would create" if dry_run else "Created"
        print(f"  {verb}: {m.path}")
    for path, reason in result.skipped:
        print(f"  Failed: {path} ({reason})")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(_main_cli())
