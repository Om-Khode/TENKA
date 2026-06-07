"""sandbox.py — Tiered code execution sandboxes with AST-based security scanning."""

import ast
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import uuid

logger = logging.getLogger("code_executor")


_BANNED_CALLS_TIER1: frozenset[tuple[str, str]] = frozenset({
    ("os", "remove"), ("os", "rmdir"), ("os", "unlink"), ("os", "rename"),
    ("os", "system"), ("os", "popen"), ("os", "execv"), ("os", "execve"),
    ("shutil", "rmtree"), ("shutil", "move"),
    ("subprocess", "call"), ("subprocess", "Popen"), ("subprocess", "run"),
    ("subprocess", "check_output"), ("subprocess", "check_call"),
})

_BANNED_CALLS_TIER2: frozenset[tuple[str, str]] = frozenset({
    # Tier 2 allows os.remove/unlink/rename — scripts run in SANDBOX_DIR.
    # Still block process/shell spawning and recursive deletion.
    ("os", "system"), ("os", "popen"), ("os", "execv"), ("os", "execve"),
    ("shutil", "rmtree"), ("shutil", "move"),
    ("subprocess", "call"), ("subprocess", "Popen"), ("subprocess", "run"),
    ("subprocess", "check_output"), ("subprocess", "check_call"),
})

_BANNED_BUILTINS: frozenset[str] = frozenset({"eval", "exec", "compile", "__import__"})

_BANNED_IMPORTS_TIER1: frozenset[str] = frozenset({
    "subprocess", "shutil", "socket", "http", "urllib",
    "requests", "httpx", "ctypes", "multiprocessing",
})


def _ast_scan(code: str, tier: int) -> str | None:
    """Parse code into AST and scan for dangerous patterns. Returns error or None."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _regex_scan_fallback(code, tier)

    banned_calls = _BANNED_CALLS_TIER1 if tier == 1 else _BANNED_CALLS_TIER2

    for node in ast.walk(tree):
        if tier == 1 and isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split('.')[0] for alias in node.names]
            elif node.module:
                names = [node.module.split('.')[0]]
            for name in names:
                if name in _BANNED_IMPORTS_TIER1:
                    return f"BLOCKED: import of '{name}' not allowed in Tier 1"

        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                pair = (func.value.id, func.attr)
                if pair in banned_calls:
                    return f"BLOCKED: call to {pair[0]}.{pair[1]}() not allowed"
            if isinstance(func, ast.Name) and func.id in _BANNED_BUILTINS:
                return f"BLOCKED: call to {func.id}() not allowed"
            if isinstance(func, ast.Name) and func.id == "open" and tier <= 2:
                mode_val = None
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    mode_val = node.args[1].value
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode_val = kw.value
                if isinstance(mode_val, str) and any(m in mode_val for m in ('w', 'a', 'x')):
                    if tier == 1:
                        return "BLOCKED: file writes not allowed in Tier 1"
    return None


def _regex_scan_fallback(code: str, tier: int) -> str | None:
    """Fallback regex scan for code with syntax errors."""
    _PATTERNS_TIER1 = [
        r'\bos\.remove\b', r'\bos\.rmdir\b', r'\bos\.unlink\b', r'\bos\.system\b',
        r'\bshutil\.rmtree\b', r'\bsubprocess\.\w+\b',
        r'\beval\s*\(', r'\bexec\s*\(', r'\b__import__\s*\(',
        r'\brequests\.', r'\burllib\.', r'\bhttpx\.', r'\bsocket\.',
    ]
    _PATTERNS_TIER2 = [
        r'\bos\.remove\b', r'\bos\.rmdir\b', r'\bos\.unlink\b', r'\bos\.system\b',
        r'\bshutil\.rmtree\b', r'\bsubprocess\.\w+\b',
        r'\beval\s*\(', r'\bexec\s*\(', r'\b__import__\s*\(',
    ]
    patterns = _PATTERNS_TIER1 if tier == 1 else _PATTERNS_TIER2
    for pat in patterns:
        if re.search(pat, code):
            return "BLOCKED: unsafe code detected"
    return None


_TIER1_SAFE_BUILTINS = {
    "print": print, "int": int, "float": float, "str": str, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "bytes": bytes, "bytearray": bytearray,
    "range": range, "enumerate": enumerate, "zip": zip,
    "map": map, "filter": filter, "sorted": sorted, "reversed": reversed,
    "len": len, "sum": sum, "min": min, "max": max,
    "isinstance": isinstance, "issubclass": issubclass, "type": type,
    "abs": abs, "round": round, "pow": pow, "divmod": divmod,
    "hex": hex, "oct": oct, "bin": bin, "ord": ord, "chr": chr,
    "repr": repr, "format": format, "hash": hash,
    "any": any, "all": all, "next": next, "iter": iter,
    "dir": dir, "vars": vars, "getattr": getattr, "hasattr": hasattr,
    "Exception": Exception, "ValueError": ValueError,
    "TypeError": TypeError, "KeyError": KeyError,
    "IndexError": IndexError, "AttributeError": AttributeError,
    "RuntimeError": RuntimeError, "StopIteration": StopIteration,
    "True": True, "False": False, "None": None,
}

_TIER1_ALLOWED_MODULES = frozenset({
    "math", "statistics", "decimal", "fractions",
    "datetime", "time", "calendar",
    "string", "re", "textwrap",
    "json", "csv",
    "pathlib", "os.path",
    "collections", "itertools", "functools",
    "psutil", "platform", "sys",
})


class _ReadOnlyEnvProxy:
    """Read-only proxy for os.environ that hides sensitive keys."""
    _SENSITIVE_PATTERNS = ('TOKEN', 'SECRET', 'KEY', 'PASSWORD', 'CREDENTIAL')

    def __init__(self, real_environ):
        self._env = real_environ

    def _is_sensitive(self, key: str) -> bool:
        return any(p in key.upper() for p in self._SENSITIVE_PATTERNS)

    def get(self, key, default=None):
        if self._is_sensitive(key):
            return default
        return self._env.get(key, default)

    def __getitem__(self, key):
        if self._is_sensitive(key):
            raise KeyError(key)
        return self._env[key]

    def __contains__(self, key):
        if self._is_sensitive(key):
            return False
        return key in self._env

    def __iter__(self):
        return (k for k in self._env if not self._is_sensitive(k))

    def keys(self):
        return [k for k in self._env if not self._is_sensitive(k)]

    def values(self):
        return [self._env[k] for k in self._env if not self._is_sensitive(k)]

    def items(self):
        return [(k, self._env[k]) for k in self._env if not self._is_sensitive(k)]

    def __len__(self):
        return sum(1 for k in self._env if not self._is_sensitive(k))

    def __repr__(self):
        return "<ReadOnlyEnvProxy>"


def _run_tier1(code: str, timeout: int = 15) -> str:
    """Run code in a restricted in-process sandbox."""
    import io

    block = _ast_scan(code, tier=1)
    if block:
        return block

    _real_import = __import__

    def _safe_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if name == "os":
            import os as _os
            import types
            safe_os = types.ModuleType("os")
            safe_os.path = _os.path
            safe_os.sep = _os.sep
            safe_os.getcwd = _os.getcwd
            safe_os.listdir = _os.listdir
            safe_os.environ = _ReadOnlyEnvProxy(_os.environ)
            return safe_os
        if top not in _TIER1_ALLOWED_MODULES and name not in _TIER1_ALLOWED_MODULES:
            raise ImportError(f"Module '{name}' not allowed. Allowed: {sorted(_TIER1_ALLOWED_MODULES)}")
        return _real_import(name, *args, **kwargs)

    captured = io.StringIO()
    safe_builtins = {**_TIER1_SAFE_BUILTINS, "__import__": _safe_import}
    safe_globals = {"__builtins__": safe_builtins, "__name__": "__main__"}
    holder: dict[str, str | None] = {"result": None, "error": None}

    def _exec():
        import sys as _s
        old = _s.stdout
        _s.stdout = captured
        try:
            exec(compile(code, "<sandbox>", "exec"), safe_globals)
            holder["result"] = captured.getvalue().strip()
        except ImportError as e:
            holder["error"] = f"BLOCKED: {e}"
        except Exception as e:
            holder["error"] = f"ERROR: {type(e).__name__}: {e}"
        finally:
            _s.stdout = old

    t = threading.Thread(target=_exec, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        return "TIMEOUT"
    if holder["error"]:
        return holder["error"]
    return holder["result"] or "(no output)"


def _run_tier2(code: str, env_vars: dict | None = None, timeout: int = 30) -> str:
    """Run code in a subprocess sandbox with network access.

    Spawned via Popen and polled at 100ms intervals; if the global abort
    flag is raised mid-execution, the subprocess is terminated cleanly and
    we return ``ABORTED`` so callers can short-circuit.
    """
    from .. import config
    from ..core.abort import abort, UserAborted

    block = _ast_scan(code, tier=2)
    if block:
        return block

    tmp_id = uuid.uuid4().hex[:12]
    tmp = os.path.join(tempfile.gettempdir(), f"mate_tier2_{tmp_id}.py")
    proc = None
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(code)
        env = os.environ.copy()
        env["SANDBOX_DIR"] = str(config.SANDBOX_DIR)
        if env_vars:
            for k, v in env_vars.items():
                env[str(k)] = str(v)
        logger.info(f"[CODE] Tier2 running ({len(code)} chars, {timeout}s)...")

        proc = subprocess.Popen(
            [sys.executable, tmp],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace', env=env,
        )

        poll_interval = 0.1
        elapsed = 0.0
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=poll_interval)
                break  # process exited
            except subprocess.TimeoutExpired:
                pass
            elapsed += poll_interval
            if abort.is_aborted():
                logger.info("[CODE] Tier2 aborted by user — terminating subprocess")
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise UserAborted(abort.reason or "esc_hold")
            if elapsed >= timeout:
                proc.kill()
                return "TIMEOUT"

        stdout, stderr = (stdout or "").strip(), (stderr or "").strip()
        for line in stdout.splitlines():
            if line.strip().startswith("NEEDS_OAUTH|"):
                return line.strip()
            if line.strip().startswith("NEEDS_DEVICE_AUTH|"):
                return line.strip()
        if proc.returncode == 0:
            # Success — return stdout, or a success indicator if empty.
            # Action commands (pause, play, mute, etc.) often produce no output.
            # Previously "(no output)" caused _needs_retry to retry after success.
            return stdout if stdout else "(completed successfully)"
        elif stderr:
            return f"ERROR: {stderr}"
        return stdout or "(no output)"
    except UserAborted:
        raise  # propagate to handler-level catch
    except Exception as e:
        return f"Error: {e}"
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def run_code(code: str, timeout: int = 15, tier: int = 1) -> str:
    """Public entry point to run code at a given tier."""
    from .. import config as _c
    if tier == 3 and not _c.CODE_EXECUTOR_POWER_MODE:
        tier = 1
    if tier == 1:
        return _run_tier1(code, timeout=timeout)
    if tier == 2:
        return _run_tier2(code, timeout=timeout)
    block = _ast_scan(code, tier=2)
    if block:
        return block
    try:
        r = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            encoding='utf-8', errors='replace', timeout=timeout,
        )
        if r.returncode == 0:
            return r.stdout.strip() if r.stdout.strip() else "(completed successfully)"
        elif r.stderr.strip():
            return f"ERROR: {r.stderr.strip()}"
        return r.stdout.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"Error: {e}"
