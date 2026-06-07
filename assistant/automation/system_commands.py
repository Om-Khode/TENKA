"""
system_commands.py — System-level shell and known-command execution.

Extracted from code_executor.py.
Handles bluetooth/wifi toggles, shell command validation, and elevated PowerShell.
"""

import logging
import os
import re
import subprocess
import tempfile
import uuid

logger = logging.getLogger("code_executor")


# ─── Allowed System Commands ─────────────────────────────────────────────────

ALLOWED_EXECUTABLES = frozenset({
    "netsh", "net", "ipconfig",
    "tasklist", "sc", "reg", "ping",
    "systeminfo", "shutdown",
})

_SHELL_METACHAR_RE = re.compile(r'[&|;`$\n\r]')

# ─── Banned Command Patterns ────────────────────────────────────────────────
# Commands that can brick hardware, destroy data, or make the system unrecoverable.
# Blocked in ALL execution paths — known commands AND LLM-generated commands.

_BANNED_CMD_PATTERNS: list[tuple[re.Pattern, str]] = [
    # --- Hardware / PnP device manipulation ---
    (re.compile(r'\bDisable-PnpDevice\b', re.IGNORECASE),
     "Disable-PnpDevice can brick hardware at driver level"),
    (re.compile(r'\bEnable-PnpDevice\b', re.IGNORECASE),
     "Enable-PnpDevice can leave devices in broken state"),
    (re.compile(r'\bDisable-NetAdapter\b', re.IGNORECASE),
     "Disable-NetAdapter disables networking at driver level"),
    (re.compile(r'\bEnable-NetAdapter\b', re.IGNORECASE),
     "Enable-NetAdapter can leave adapters in broken state"),

    # --- Disk / volume / partition destruction ---
    (re.compile(r'\bFormat-Volume\b', re.IGNORECASE),
     "Format-Volume destroys all data on a volume"),
    (re.compile(r'\bClear-Disk\b', re.IGNORECASE),
     "Clear-Disk destroys all disk data"),
    (re.compile(r'\bInitialize-Disk\b', re.IGNORECASE),
     "Initialize-Disk destroys partition table"),
    (re.compile(r'\bRemove-Partition\b', re.IGNORECASE),
     "Remove-Partition destroys data"),
    (re.compile(r'\bdiskpart\b', re.IGNORECASE),
     "diskpart can destroy partitions and wipe disks"),

    # --- Boot / OS integrity ---
    (re.compile(r'\bbcdedit\b', re.IGNORECASE),
     "bcdedit can make system unbootable"),
    (re.compile(r'\bDisable-WindowsOptionalFeature\b', re.IGNORECASE),
     "Disable-WindowsOptionalFeature removes OS components"),

    # --- File/directory deletion via PowerShell ---
    (re.compile(r'\bRemove-Item\b', re.IGNORECASE),
     "Remove-Item can delete critical system files"),

    # --- Service manipulation ---
    (re.compile(r'\bStop-Service\b', re.IGNORECASE),
     "Stop-Service can disable critical system services"),
    (re.compile(r'\bSet-Service\b', re.IGNORECASE),
     "Set-Service can permanently disable services"),
    (re.compile(r'\bsc\s+(delete|stop|config)\b', re.IGNORECASE),
     "sc delete/stop/config can disable or remove services"),
    (re.compile(r'\bnet\s+stop\b', re.IGNORECASE),
     "net stop can disable critical services"),

    # --- User / privilege escalation ---
    (re.compile(r'\bNew-LocalUser\b', re.IGNORECASE),
     "New-LocalUser creates system accounts"),
    (re.compile(r'\bAdd-LocalGroupMember\b', re.IGNORECASE),
     "Add-LocalGroupMember escalates privileges"),
    (re.compile(r'\bnet\s+user\b', re.IGNORECASE),
     "net user manipulates system accounts"),
    (re.compile(r'\bnet\s+localgroup\b', re.IGNORECASE),
     "net localgroup modifies group membership"),

    # --- Registry modification ---
    (re.compile(r'\breg\s+delete\b', re.IGNORECASE),
     "reg delete can corrupt the registry"),
    (re.compile(r'\breg\s+add\b', re.IGNORECASE),
     "reg add can modify critical registry settings"),

    # --- Arbitrary code execution / payload download ---
    (re.compile(r'\bInvoke-Expression\b', re.IGNORECASE),
     "Invoke-Expression executes arbitrary code"),
    (re.compile(r'\bIEX\b', re.IGNORECASE),
     "IEX is Invoke-Expression alias"),
    (re.compile(r'\bInvoke-WebRequest\b', re.IGNORECASE),
     "Invoke-WebRequest can download payloads"),
    (re.compile(r'\bInvoke-RestMethod\b', re.IGNORECASE),
     "Invoke-RestMethod can download payloads"),
    (re.compile(r'\bDownloadString\b', re.IGNORECASE),
     "DownloadString can download and execute payloads"),
    (re.compile(r'\bDownloadFile\b', re.IGNORECASE),
     "DownloadFile can download payloads"),
    (re.compile(r'\bStart-BitsTransfer\b', re.IGNORECASE),
     "Start-BitsTransfer can download payloads"),

    # --- Security software manipulation ---
    (re.compile(r'\bSet-MpPreference\b', re.IGNORECASE),
     "Set-MpPreference can disable Windows Defender"),

    # --- PowerShell encoded command bypass ---
    (re.compile(r'\b-EncodedCommand\b', re.IGNORECASE),
     "EncodedCommand can hide any payload in base64"),
    (re.compile(r'\s-Enc\b', re.IGNORECASE),
     "Abbreviated -EncodedCommand bypass"),
]


def _check_banned_patterns(cmd_text: str) -> str | None:
    """Check command text against banned patterns. Returns error message or None."""
    for pattern, reason in _BANNED_CMD_PATTERNS:
        if pattern.search(cmd_text):
            return f"BANNED: {reason}"
    return None


# ─── Bluetooth Radio Toggle (WinRT Radio API) ──────────────────────────────
# Uses Windows.Devices.Radios — same mechanism as Settings / Quick Settings.
# Safe: toggles the radio soft-switch only, never touches the PnP device stack.
# No elevation required.

_BT_RADIO_PREAMBLE = (
    "Add-Type -AssemblyName System.Runtime.WindowsRuntime\n"
    "$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {\n"
    "    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and\n"
    "    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'\n"
    "})[0]\n"
    "function Await($op, $type) {\n"
    "    $t = $asTask.MakeGenericMethod($type).Invoke($null, @($op))\n"
    "    $t.Wait(-1) | Out-Null\n"
    "    $t.Result\n"
    "}\n"
    "[Windows.Devices.Radios.Radio,Windows.System.Devices,ContentType=WindowsRuntime] | Out-Null\n"
    "$radios = Await ([Windows.Devices.Radios.Radio]::GetRadiosAsync()) "
    "([System.Collections.Generic.IReadOnlyList[Windows.Devices.Radios.Radio]])\n"
    "$bt = $radios | Where-Object { $_.Kind -eq 'Bluetooth' }\n"
    "if (-not $bt) { throw 'No Bluetooth radio found' }\n"
)

# ─── Known-Good Commands ─────────────────────────────────────────────────────

KNOWN_COMMANDS = {
    "bluetooth_on": {
        "script": _BT_RADIO_PREAMBLE + (
            "Await ($bt.SetStateAsync([Windows.Devices.Radios.RadioState]::On)) "
            "([Windows.Devices.Radios.RadioAccessStatus]) | Out-Null"
        ),
        "elevated": False,
    },
    "bluetooth_off": {
        "script": _BT_RADIO_PREAMBLE + (
            "Await ($bt.SetStateAsync([Windows.Devices.Radios.RadioState]::Off)) "
            "([Windows.Devices.Radios.RadioAccessStatus]) | Out-Null"
        ),
        "elevated": False,
    },
    "wifi_on": {
        "cmd": "netsh interface set interface Wi-Fi enabled",
        "elevated": False,
    },
    "wifi_off": {
        "cmd": "netsh interface set interface Wi-Fi disabled",
        "elevated": False,
    },
    "wifi_list": {
        "cmd": "netsh wlan show profiles",
        "elevated": False,
    },
}

# Safety audit: ensure no known command contains a banned pattern.
for _cmd_name, _cmd_entry in KNOWN_COMMANDS.items():
    _cmd_text = " ".join(filter(None, [_cmd_entry.get("script"), _cmd_entry.get("cmd")]))
    _violation = _check_banned_patterns(_cmd_text)
    if _violation:
        raise RuntimeError(
            f"KNOWN_COMMANDS['{_cmd_name}'] contains banned pattern: {_violation}"
        )


# ─── Sync error fallback ──────────────────────────────────────────────────────

def _clean_error_for_user_sync(raw: str) -> str:
    """Sync fallback for when we can't call the async LLM error cleaner."""
    return "Sorry, that command didn't work."


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _match_known_command(goal):
    g = goal.lower()
    if "bluetooth" in g:
        if any(w in g for w in ("enable", "turn on", "on", "start", "connect")):
            return KNOWN_COMMANDS["bluetooth_on"]
        if any(w in g for w in ("disable", "turn off", "off", "stop", "disconnect")):
            return KNOWN_COMMANDS["bluetooth_off"]
    if "wi-fi" in g or "wifi" in g or "wi fi" in g:
        if any(w in g for w in ("enable", "turn on", "on", "start", "connect")):
            return KNOWN_COMMANDS["wifi_on"]
        if any(w in g for w in ("disable", "turn off", "off", "stop", "disconnect")):
            return KNOWN_COMMANDS["wifi_off"]
        if any(w in g for w in ("list", "show", "scan")):
            return KNOWN_COMMANDS["wifi_list"]
    return None


def _validate_shell_command(cmd: str) -> str | None:
    cmd = cmd.strip()
    if not cmd:
        return "Empty command"
    banned = _check_banned_patterns(cmd)
    if banned:
        return banned
    if _SHELL_METACHAR_RE.search(cmd):
        return "Blocked: compound commands not allowed"
    tok = cmd.split()[0].lower().rstrip(".exe")
    if tok not in ALLOWED_EXECUTABLES:
        return f"Blocked: '{tok}' not allowed."
    return None


def _run_shell(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=10)
        if r.returncode == 0:
            return True, r.stdout.strip() or "Done."
        return False, (r.stderr or r.stdout or "Error").strip()
    except subprocess.TimeoutExpired:
        return False, "Timeout."
    except Exception as e:
        return False, f"Error: {e}"


def _run_elevated_ps(ps):
    tmp = tempfile.gettempdir()
    unique = uuid.uuid4().hex[:8]
    sp = os.path.join(tmp, f"mate_admin_{unique}.ps1")
    op = os.path.join(tmp, f"mate_admin_out_{unique}.txt")
    try:
        with open(sp, "w", encoding="utf-8") as f:
            f.write(f"try {{\n    {ps}\n    'SUCCESS' | Out-File '{op}' -Encoding UTF8\n"
                    f"}} catch {{\n    $_.Exception.Message | Out-File '{op}' -Encoding UTF8\n}}\n")
        if os.path.exists(op):
            os.remove(op)
        try:
            subprocess.run(
                f'powershell -WindowStyle Hidden -Command "Start-Process powershell '
                f'-Verb RunAs -Wait -ArgumentList \'-ExecutionPolicy Bypass -WindowStyle Hidden '
                f'-File \\\"{sp}\\\"\'"',
                shell=True, capture_output=True, text=True,
                encoding='utf-8', errors='replace', timeout=30)
        except Exception as e:
            logger.warning(f"[CODE] Elevated PS failed: {e}")
            return False, "Elevated command failed."
        if os.path.exists(op):
            with open(op, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            return (True, "Done.") if "SUCCESS" in txt.lstrip("﻿") else (False, txt or "Failed.")
        return True, "Done."
    finally:
        for path in (sp, op):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass


def _run_ps_script(ps: str) -> tuple[bool, str]:
    """Run a PowerShell script without elevation, via temp file."""
    tmp = tempfile.gettempdir()
    unique = uuid.uuid4().hex[:8]
    sp = os.path.join(tmp, f"mate_ps_{unique}.ps1")
    try:
        with open(sp, "w", encoding="utf-8") as f:
            f.write(ps)
        r = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File", sp],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15,
        )
        if r.returncode == 0:
            return True, r.stdout.strip() or "Done."
        return False, (r.stderr or r.stdout or "Error").strip()
    except subprocess.TimeoutExpired:
        return False, "Timeout."
    except Exception as e:
        return False, f"Error: {e}"
    finally:
        try:
            if os.path.exists(sp):
                os.remove(sp)
        except OSError:
            pass


def _run_known_command(e):
    cmd_text = e.get("script") or e.get("cmd", "")
    banned = _check_banned_patterns(cmd_text)
    if banned:
        logger.error(f"[CMD] Known command blocked at runtime: {banned}")
        return False, banned
    if e.get("elevated"):
        return _run_elevated_ps(e["script"])
    if "script" in e:
        return _run_ps_script(e["script"])
    return _run_shell(e["cmd"])


# ─── Public API ───────────────────────────────────────────────────────────────

async def run_system_command(goal, llm_func):
    known = _match_known_command(goal)
    if known:
        ok, out = _run_known_command(known)
        if ok:
            return out
        if out.startswith("BANNED:"):
            return "Blocked: command is banned for safety."
    goal_safe = goal.replace('\n', ' ').replace('\r', ' ')
    cmd = await llm_func(f"Generate one Windows shell command to: {goal_safe}\nOnly the command.", task_type="code_gen")
    if cmd == "__LLM_UNAVAILABLE__":
        return "No LLM available."
    cmd = cmd.strip().strip("`")
    err = _validate_shell_command(cmd)
    if err:
        if err.startswith("BANNED:"):
            return "Blocked: command is banned for safety."
        return err
    ok, out = _run_shell(cmd)
    if ok:
        return out
    retry = await llm_func(f"Failed: {out}\nCorrected command for: {goal_safe}\nOnly the command.", task_type="code_gen")
    if retry == "__LLM_UNAVAILABLE__":
        return _clean_error_for_user_sync(out)
    retry = retry.strip().strip("`")
    err2 = _validate_shell_command(retry)
    if err2:
        if err2.startswith("BANNED:"):
            return "Blocked: command is banned for safety."
        return _clean_error_for_user_sync(out)
    ok2, out2 = _run_shell(retry)
    return out2 if ok2 else _clean_error_for_user_sync(out2)
