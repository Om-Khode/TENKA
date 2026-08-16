"""sandbox.py — Tiered code execution sandboxes with AST-based security scanning."""

import ast
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import types
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

# ─── Process-spawn calls, banned at every tier and from every module ─────────
# `_BANNED_CALLS_TIER*` bans a (module, attribute) *pair*, which bans a spelling
# rather than a capability. `subprocess.Popen` is refused while `psutil.Popen`
# — the same thing, from a package requirements.txt already installs into the
# very interpreter `_run_tier2` spawns via sys.executable — was not. Verified:
# Tier 1 ran `psutil.Popen(['cmd','/c','ver'])` and printed the Windows version.
#
# These names are refused wherever they are reached from, so a wrapper, an
# alias, or a package nobody has heard of yet is caught by the same rule. No
# package is named anywhere here: the ban is on the capability.
_BANNED_CALL_ATTRS: frozenset[str] = frozenset({
    "Popen", "popen", "startfile", "check_output", "check_call",
    # Process control, banned as one set. Banning `kill`/`killpg` while leaving
    # `terminate`/`send_signal` reachable was an inconsistency in the set, not a
    # boundary: the same object offers all four, and the sandbox proved it by
    # handing back a bound terminate for its own interpreter.
    "fork", "forkpty", "kill", "killpg", "terminate", "send_signal",
    "execl", "execle", "execlp", "execlpe",
    "execv", "execve", "execvp", "execvpe",
    "spawnl", "spawnle", "spawnlp", "spawnlpe",
    "spawnv", "spawnve", "spawnvp", "spawnvpe",
    "create_subprocess_exec", "create_subprocess_shell",
    "launch", "launch_persistent_context",
})
# Deliberately NOT prefix-matched on "exec": `cursor.execute(...)` is the
# ordinary way to use a database, and a prefix rule would refuse every script
# that touches one.
#
# "system" is not in the set either, because `platform.system()` returns a
# string and is one of the most common calls there is. The discriminator is the
# argument, not the name: a shell-out always takes one, `platform.system()`
# never does. Handled separately in `_ast_scan`.
_ARGUMENT_SENSITIVE_CALL_ATTRS: frozenset[str] = frozenset({"system"})

_BANNED_IMPORTS_TIER1: frozenset[str] = frozenset({
    "subprocess", "shutil", "socket", "http", "urllib",
    "requests", "httpx", "ctypes", "multiprocessing",
    # `inspect` is allowed at Tier 2 and refused here — see the note on
    # `_TIER2_ALLOWED_MODULES` for why the two tiers differ. The runtime
    # allow-list in `_safe_import` already refused it; naming it here makes the
    # split visible at scan time too, and gives the clearer message.
    "inspect",
})

# ─── Tier 2 import allow-list ────────────────────────────────────────────────
# Tier 2 had no import restriction at all: the scan branch read `if tier == 1`,
# so `import ctypes` reached a subprocess holding the whole environment.
#
# The restriction is deny-by-default *over the standard library*, which is
# where every escape primitive lives (ctypes, multiprocessing, importlib,
# winreg, subprocess). It cannot be a flat allow-list of names, because Tier 2
# exists to run SDK work: `_ensure_packages` pip-installs whatever the route's
# `requires` names — spotipy, google-api-python-client, python-docx — and a
# fixed name list would refuse all of it. So a third-party import is admitted.
#
# What that admission is NOT: a claim that a third-party import is safe. It is
# not "only what _ensure_packages installed" either — requirements.txt is
# installed into this same interpreter, so psutil and playwright are importable
# with no install step at all, and psutil.Popen spawns processes just as well as
# subprocess.Popen does. The capability is therefore blocked at the *call*, by
# name, wherever it is reached from — see `_BANNED_CALL_ATTRS`. This allow-list
# governs which module may be named; that set governs what may be done with it.
_TIER2_ALLOWED_MODULES: frozenset[str] = frozenset({
    # Everything Tier 1 may have.
    "math", "statistics", "decimal", "fractions",
    "datetime", "time", "calendar", "zoneinfo",
    "string", "re", "textwrap", "unicodedata",
    "json", "csv", "xml", "html",
    "pathlib", "os", "io", "sys", "tempfile", "glob", "fnmatch", "shutil",
    "collections", "itertools", "functools", "operator", "heapq", "bisect",
    "psutil", "platform",
    # The network Tier 2 is chosen for.
    "requests", "httpx", "urllib", "http", "socket", "ssl", "email",
    # Encoding / identity helpers generated scripts routinely reach for.
    "base64", "binascii", "hashlib", "hmac", "secrets", "uuid", "struct",
    "random", "typing", "dataclasses", "enum", "copy", "traceback",
    "logging", "warnings", "pprint", "argparse", "mimetypes",
    "zipfile", "tarfile", "sqlite3",
    "asyncio", "queue", "threading", "contextlib", "abc",
    # `inspect` — Tier 2 only, and the asymmetry is deliberate. TENKA's own
    # discovery probe (`discovery.py`) reads an SDK method's signature after a
    # TypeError, so refusing it here silently broke first-party SDK discovery.
    #
    # It stays banned at Tier 1 because the tiers are not the same kind of
    # thing: Tier 1 is an in-process exec sharing the daemon's interpreter, so
    # frame walking there reaches TENKA's own stack and its live globals. Tier 2
    # is a separate subprocess with a scrubbed environment — there is nothing of
    # TENKA's in it to introspect, so the same capability reaches only the
    # script's own frames and the SDK it just imported.
    "inspect",
})
# Deliberately absent, and each absence is the point: subprocess, ctypes,
# multiprocessing, importlib, winreg, pty, signal, gc, inspect, marshal,
# pickle, code, codeop, runpy, sysconfig, site, webbrowser, concurrent.

# Names the running interpreter reports as standard library. Used to decide
# whether an unrecognised import is stdlib (deny) or a deliberately installed
# package (allow). Read from the interpreter rather than hardcoded so it stays
# correct across Python versions.
_STDLIB_MODULE_NAMES: frozenset[str] = frozenset(sys.stdlib_module_names)


def _import_violation(name: str, tier: int) -> str | None:
    """Return a BLOCKED message if `name` may not be imported at `tier`."""
    top = name.split(".")[0]
    if tier == 1:
        if top in _BANNED_IMPORTS_TIER1:
            return f"BLOCKED: import of '{top}' not allowed in Tier 1"
        return None
    if tier >= 2:
        if top in _TIER2_ALLOWED_MODULES or name in _TIER2_ALLOWED_MODULES:
            return None
        if top in _STDLIB_MODULE_NAMES:
            return f"BLOCKED: import of '{top}' not allowed in Tier 2"
    return None


def _imported_names(node) -> list[str]:
    """Every module name an Import / ImportFrom node brings into scope."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.module:
        return [node.module]
    return []


# Attribute names that hand generated code a live reference to the interpreter
# it is running inside. `_ast_scan`'s call check cannot see them: it requires an
# `ast.Name` in the value position, and `os.path.join.__globals__["os"]` holds
# an `ast.Attribute` there.
_TIER1_ESCAPE_ATTRS: frozenset[str] = frozenset({
    "__globals__", "__class__", "__bases__", "__base__", "__subclasses__",
    "__mro__", "__builtins__", "__code__", "__closure__", "__func__",
    "__self__", "__wrapped__", "__dict__", "__getattribute__",
    "__reduce__", "__reduce_ex__", "__init_subclass__", "__subclasshook__",
    "__loader__", "__spec__", "__import__",
})


def _ast_scan(code: str, tier: int) -> str | None:
    """Parse code into AST and scan for dangerous patterns. Returns error or None."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _regex_scan_fallback(code, tier)

    banned_calls = _BANNED_CALLS_TIER1 if tier == 1 else _BANNED_CALLS_TIER2

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for name in _imported_names(node):
                violation = _import_violation(name, tier)
                if violation:
                    return violation
            # `from x import Popen as q` renames the primitive, so the call site
            # cannot be checked by name. Refusing the import is what catches it.
            for alias in node.names:
                if alias.name.split(".")[-1] in _BANNED_CALL_ATTRS:
                    return f"BLOCKED: import of '{alias.name}' not allowed"

        if isinstance(node, ast.Attribute):
            if tier == 1 and node.attr in _TIER1_ESCAPE_ATTRS:
                return f"BLOCKED: attribute '{node.attr}' not allowed in Tier 1"
            # Checked on the *reference*, not only the call. Binding the bound
            # method to a local first — `t = p.terminate` then `t()` — leaves an
            # ast.Name at the call site, which no name-based check can see.
            if node.attr in _BANNED_CALL_ATTRS:
                return (f"BLOCKED: attribute '{node.attr}' not allowed — "
                        f"process spawning or control")

        if isinstance(node, ast.Call):
            func = node.func
            called = getattr(func, "attr", None) or getattr(func, "id", None)
            if called in _BANNED_CALL_ATTRS:
                return f"BLOCKED: call to {called}() not allowed — process spawning"
            if called in _ARGUMENT_SENSITIVE_CALL_ATTRS and (node.args or node.keywords):
                return f"BLOCKED: call to {called}() with arguments not allowed"
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


_IMPORT_CLAUSE_RE = re.compile(r'^[ \t]*(import|from)[ \t]+([^\n#;]+)', re.MULTILINE)


def _regex_imported_names(code: str) -> list[str]:
    """Best-effort module names from source the parser refused.

    The fallback runs on code that raised `SyntaxError`, so there is no tree to
    walk — but the import restriction has to apply here too. A fallback weaker
    than the AST path is a hole in its own right: emit one stray bracket and
    the scan you route around is the only one that checks imports.
    """
    names: list[str] = []
    for keyword, clause in _IMPORT_CLAUSE_RE.findall(code):
        if keyword == "from":
            head = clause.split(" import ")[0].strip()
            if head:
                names.append(head.split()[0])
            continue
        for part in clause.split(","):
            token = part.strip().split(" as ")[0].strip()
            if token:
                names.append(token.split()[0])
    return names


_BANNED_CALL_ATTR_RE = re.compile(
    r'\b(?:' + '|'.join(sorted(_BANNED_CALL_ATTRS)) + r')\s*\(')


def _regex_scan_fallback(code: str, tier: int) -> str | None:
    """Fallback regex scan for code with syntax errors."""
    for name in _regex_imported_names(code):
        violation = _import_violation(name, tier)
        if violation:
            return violation
    if _BANNED_CALL_ATTR_RE.search(code):
        return "BLOCKED: unsafe code detected — process spawning"

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


# ─── Tier 1 module shims ─────────────────────────────────────────────────────
# Modules re-export each other. Verified on this machine against every entry of
# `_TIER1_ALLOWED_MODULES`:
#
#   os.path.os          -> os      (os.path is the real ntpath; ntpath.os is os)
#   platform.os         -> os
#   pathlib.os          -> os      (also pathlib.ntpath, pathlib.fnmatch)
#   psutil.os           -> os      (also psutil.subprocess -> subprocess)
#   statistics.random._os -> os
#
# So an allow-list of module *names* is not a boundary while the module objects
# themselves are handed over. `_SafeModule` hands out a submodule only when that
# submodule is itself allow-listed, and shims it in turn; private attributes are
# refused outright, because a module's underscore-prefixed names are its
# internals, never its API.

_TIER1_OS_PATH_ATTRS: frozenset[str] = frozenset({
    "join", "exists", "basename", "dirname", "splitext", "split",
    "splitdrive", "abspath", "normpath", "normcase", "realpath",
    "isabs", "isdir", "isfile", "getsize", "getmtime", "commonprefix",
    "sep", "altsep", "extsep", "pathsep", "curdir", "pardir",
})

# Public module attributes that hand out live interpreter machinery. Data, not
# code: a new one is a new row here.
_TIER1_BLOCKED_MODULE_ATTRS: frozenset[str] = frozenset({
    "modules", "meta_path", "path_hooks", "path_importer_cache",
    "environ", "environb", "settrace", "setprofile", "gettrace", "getprofile",
    "audit", "addaudithook", "excepthook", "unraisablehook", "breakpointhook",
})


class _SafeModule:
    """Attribute proxy over an allow-listed module, used only at Tier 1."""

    __slots__ = ("_module", "_label")

    def __init__(self, module, label: str):
        object.__setattr__(self, "_module", module)
        object.__setattr__(self, "_label", label)

    def __getattr__(self, attr: str):
        label = object.__getattribute__(self, "_label")
        if attr.startswith("_"):
            raise AttributeError(
                f"BLOCKED: '{label}.{attr}' is module-private and not available in Tier 1")
        if attr in _TIER1_BLOCKED_MODULE_ATTRS:
            raise AttributeError(
                f"BLOCKED: '{label}.{attr}' exposes interpreter internals")
        if attr in _BANNED_CALL_ATTRS:
            # Runtime half of the call-name ban. `psutil.Popen` is a public
            # class attribute, not a submodule, so the submodule rule below
            # never sees it — and Tier 1 really did spawn cmd.exe through it.
            raise AttributeError(
                f"BLOCKED: '{label}.{attr}' spawns or controls processes")
        value = getattr(object.__getattribute__(self, "_module"), attr)
        if isinstance(value, types.ModuleType):
            name = getattr(value, "__name__", attr)
            top = name.split(".")[0]
            if top not in _TIER1_ALLOWED_MODULES and name not in _TIER1_ALLOWED_MODULES:
                raise AttributeError(
                    f"BLOCKED: '{label}.{attr}' re-exports module "
                    f"'{name}', which is not allowed in Tier 1")
            return _SafeModule(value, name)
        return value

    def __setattr__(self, attr, value):
        raise AttributeError("BLOCKED: Tier 1 modules are read-only")

    def __delattr__(self, attr):
        raise AttributeError("BLOCKED: Tier 1 modules are read-only")

    def __dir__(self):
        return [a for a in dir(object.__getattribute__(self, "_module"))
                if not a.startswith("_") and a not in _TIER1_BLOCKED_MODULE_ATTRS]

    def __repr__(self):
        return f"<module '{object.__getattribute__(self, '_label')}' (tier1)>"


def _build_safe_os_path() -> types.ModuleType:
    """A path module holding only pure path functions, never the real ntpath.

    Copying the functions off the real module rather than exposing the module
    object is the whole fix: `ntpath.os is os`, so handing over `ntpath` handed
    over `os.environ` and `os.system` one attribute away.
    """
    import os.path as _real_path
    shim = types.ModuleType("path")
    for attr in _TIER1_OS_PATH_ATTRS:
        if hasattr(_real_path, attr):
            setattr(shim, attr, getattr(_real_path, attr))
    return shim


# Returned when a run produced nothing and the only thing it asked for was a
# withheld variable. Deliberately in `_ast_scan`'s `BLOCKED:` shape, because the
# orchestrator already understands that prefix and will not retry it.
#
# The wording is identical whether the variable is set or not, and it has to
# stay that way: a message that distinguished the two cases would be an oracle
# for probing which credentials this machine holds. That property is structural,
# not careful phrasing — `_withhold` is called on the *key name* matching a
# sensitive pattern, before the real environment is consulted at all, so there
# is nothing in the code path that could tell the two apart.
_ENV_WITHHELD_MESSAGE = (
    "BLOCKED: secret-looking environment variables are hidden from the sandbox, set or not."
)


_SENSITIVE_ENV_PATTERNS = ('TOKEN', 'SECRET', 'KEY', 'PASSWORD', 'CREDENTIAL')

# Where a script asks for a named environment variable. Used by Tier 2, which
# has no proxy to record withholding — it is a subprocess, and B3 scrubbed the
# variable out of its environment before it ever started.
_ENV_LOOKUP_RE = re.compile(
    r"""(?:environ\s*\.\s*get\s*\(|environ\s*\[|getenv\s*\()\s*['"]([A-Za-z_]\w*)['"]""")


def _is_sensitive_env_name(key) -> bool:
    return any(p in str(key).upper() for p in _SENSITIVE_ENV_PATTERNS)


def _withheld_env_names(code: str, provided) -> list[str]:
    """Sensitive-looking variables the script asked for and did not receive."""
    return [n for n in dict.fromkeys(_ENV_LOOKUP_RE.findall(code))
            if _is_sensitive_env_name(n) and n not in provided]


class _ReadOnlyEnvProxy:
    """Read-only proxy for os.environ that hides sensitive keys."""
    _SENSITIVE_PATTERNS = _SENSITIVE_ENV_PATTERNS

    def __init__(self, real_environ):
        self._env = real_environ
        # Names this run asked for and did not get. Names only — never values,
        # and never whether the name actually exists.
        self.withheld: list[str] = []

    def _is_sensitive(self, key: str) -> bool:
        return _is_sensitive_env_name(key)

    def _withhold(self, key) -> None:
        if key not in self.withheld:
            self.withheld.append(key)

    def get(self, key, default=None):
        if self._is_sensitive(key):
            self._withhold(key)
            return default
        return self._env.get(key, default)

    def __getitem__(self, key):
        if self._is_sensitive(key):
            self._withhold(key)
            raise KeyError(key)
        return self._env[key]

    def __contains__(self, key):
        if self._is_sensitive(key):
            self._withhold(key)
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
    # Built once for the whole run, so the withheld-name record survives past
    # the `import os` that created it.
    env_proxy = _ReadOnlyEnvProxy(os.environ)

    def _safe_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if top == "os":
            # `import os.path` used to match the "os.path" allow-list entry and
            # fall through to the real importer, which returns the *top-level*
            # os module and binds it to the name `os`. That handed over the
            # whole real module — a shorter escape than os.path.os.
            import os as _os
            safe_os = types.ModuleType("os")
            safe_os.path = _build_safe_os_path()
            safe_os.sep = _os.sep
            safe_os.getcwd = _os.getcwd
            safe_os.listdir = _os.listdir
            safe_os.environ = env_proxy
            fromlist = args[2] if len(args) >= 3 else kwargs.get("fromlist")
            if name != "os" and fromlist:
                # `from os.path import join` expects the submodule itself.
                return safe_os.path
            return safe_os
        if top not in _TIER1_ALLOWED_MODULES and name not in _TIER1_ALLOWED_MODULES:
            raise ImportError(f"Module '{name}' not allowed. Allowed: {sorted(_TIER1_ALLOWED_MODULES)}")
        module = _real_import(name, *args, **kwargs)
        if isinstance(module, types.ModuleType):
            return _SafeModule(module, getattr(module, "__name__", name))
        return module

    def _safe_getattr(obj, name, *default):
        # The AST guard sees `x.__globals__`; it cannot see
        # getattr(x, "__" + "globals__"). This closes the runtime half.
        if isinstance(name, str) and name.startswith("__"):
            raise AttributeError(f"BLOCKED: attribute '{name}' not allowed in Tier 1")
        return getattr(obj, name, *default)

    captured = io.StringIO()
    safe_builtins = {**_TIER1_SAFE_BUILTINS,
                     "__import__": _safe_import,
                     "getattr": _safe_getattr}
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
    if not holder["result"] and env_proxy.withheld:
        # A run that produced nothing after asking for a hidden variable was
        # reported as an empty result or a KeyError — both of which read as
        # "the script is broken" rather than "the sandbox refused". The
        # orchestrator retried on that, the model guessed a different key name,
        # and the user was handed code to try it again themselves. The control
        # worked; only the way it reported itself did not.
        return _ENV_WITHHELD_MESSAGE
    if holder["error"]:
        return holder["error"]
    return holder["result"] or "(no output)"


# ─── Tier 2 environment allow-list ───────────────────────────────────────────
# Names only — no values, and nothing here carries a credential. If a real run
# fails for a missing variable, add the name with a comment saying which run
# needed it. Never widen this back to a copy of the process environment.
_TIER2_ENV_PASSTHROUGH: frozenset[str] = frozenset({
    # What the interpreter itself needs to start and find its own files.
    "PATH", "PATHEXT", "COMSPEC", "SYSTEMROOT", "WINDIR", "SYSTEMDRIVE",
    "TEMP", "TMP", "PYTHONPATH", "PYTHONHOME",
    "PYTHONIOENCODING", "PYTHONUTF8", "PYTHONHASHSEED",
    "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS",
    # Path anchors installed SDKs resolve against — pathlib.Path.home(),
    # per-user token caches, certifi's data directory.
    "LOCALAPPDATA", "APPDATA", "PROGRAMDATA", "USERPROFILE",
    "HOMEDRIVE", "HOMEPATH", "PROGRAMFILES", "PROGRAMFILES(X86)",
    "COMMONPROGRAMFILES",
})


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
        # Never `os.environ.copy()`: that handed generated code every provider
        # key and OAuth token in the process, one `requests.post` from leaving
        # the machine. Only the variables the interpreter genuinely needs cross
        # the boundary, plus the credentials the orchestrator injected *for this
        # goal* — which is the mechanism, not the leak.
        env = {k: v for k, v in os.environ.items()
               if k.upper() in _TIER2_ENV_PASSTHROUGH}
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
            if not stdout and _withheld_env_names(code, env):
                # B3 scrubbed the variable out of this subprocess's environment,
                # so the script read nothing and exited 0. Reporting that as
                # success is worse than Tier 1's old empty string: `_needs_retry`
                # is False for it, so a refused read became a confident empty
                # answer. Tier 1 escalates its BLOCKED here, so both tiers have
                # to say the same thing or the escalation just relocates the lie.
                return _ENV_WITHHELD_MESSAGE
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
