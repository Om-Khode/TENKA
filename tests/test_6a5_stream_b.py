"""Milestone 6a.5 — Stream B: sandbox and shell hardening.

Four findings, each with its own control test proving the fix did not collapse
into refusing everything:

* B1 — Tier 1's synthetic ``os`` module handed out the *real* ``ntpath`` as
  ``os.path``, and ``ntpath.os is os``. One attribute chain
  (``os.path.os.environ``) walked straight back out of the sandbox.
* B2 — ``_ast_scan`` applied no import restriction at Tier 2 at all
  (``if tier == 1``).
* B3 — Tier 2 ran with ``os.environ.copy()``: every provider key and OAuth
  token in the process.
* B4 — the shell allow-list used ``rstrip(".exe")`` (a character *set*, not a
  suffix), missed ``>``/``<``, missed ``sc create``, and ran through
  ``shell=True``.
"""

import asyncio
import os

import pytest

from assistant.automation import system_commands as sc
from assistant.code_executor import sandbox
from assistant.code_executor.sandbox import _ast_scan, run_code


# ─── B1: the Tier 1 os.path escape ───────────────────────────────────────────

def test_tier1_cannot_reach_the_real_environ_through_os_path():
    """safe_os.path was the real ntpath, and ntpath.os is os. One attribute
    chain walks straight back out of the sandbox."""
    out = run_code("import os\nprint(len(os.path.os.environ))", tier=1)
    assert "BLOCKED" in out or "ERROR" in out or "AttributeError" in out, out


def test_tier1_cannot_reach_os_system_through_os_path():
    out = run_code("import os\nprint(os.path.os.system)", tier=1)
    assert "BLOCKED" in out or "ERROR" in out or "AttributeError" in out, out


def test_tier1_can_still_join_a_path():
    """The fix must not remove the path functions legitimate code uses."""
    out = run_code("import os\nprint(os.path.join('a', 'b'))", tier=1)
    assert "a" in out and "b" in out, out


def test_tier1_can_still_use_the_other_pure_path_helpers():
    """Control, widened: basename/splitext/isabs are the rest of what generated
    Tier 1 code actually calls."""
    code = (
        "import os\n"
        "p = os.path.join('c:', 'tmp', 'notes.txt')\n"
        "print(os.path.basename(p), os.path.splitext(p)[1], os.path.isabs(p))\n"
    )
    out = run_code(code, tier=1)
    assert "notes.txt" in out and ".txt" in out, out


def test_the_readonly_env_proxy_still_hides_secrets():
    out = run_code("import os\nprint(os.environ.get('TENKA_SECRET_X', 'hidden'))", tier=1)
    assert "hidden" in out, out


def test_tier1_cannot_reach_the_real_os_by_importing_os_path_directly():
    """Found during the B1 audit: `_safe_import` special-cased only the exact
    name "os". `import os.path` matched the "os.path" allow-list entry, so
    `_real_import` returned the *real* top-level os module and bound it to the
    name `os` -- a shorter escape than the one the spec found. On the baseline
    this printed 86: the real environment, every key of it."""
    out = run_code(
        "import os.path\n"
        "print(hasattr(os, 'system'), hasattr(os, 'remove'), hasattr(os, 'popen'))\n"
        "print(os.environ.get('TENKA_SECRET_X', 'hidden'))\n",
        tier=1)
    assert "False False False" in out, out
    assert "hidden" in out, out


@pytest.mark.parametrize("module_name", ["platform", "pathlib", "psutil", "statistics"])
def test_no_allowed_tier1_module_re_exports_the_real_os(module_name):
    """The B1 audit: every one of these reaches the real `os` through a plain
    attribute, and `_ast_scan` cannot see it because line 63 requires an
    ast.Name where `platform.os.system(...)` holds an ast.Attribute."""
    out = run_code(f"import {module_name}\nprint(len({module_name}.os.environ))", tier=1)
    assert "BLOCKED" in out or "ERROR" in out, out


def test_the_allowed_tier1_modules_still_work():
    """Control for the audit fix: shimming submodule access must not break the
    modules themselves."""
    code = (
        "import platform, pathlib, statistics, json, collections\n"
        "print(bool(platform.system()))\n"
        "print(pathlib.PurePath('a', 'b').name)\n"
        "print(statistics.mean([1, 2, 3]))\n"
        "print(json.dumps({'k': 1}))\n"
        "print(collections.Counter('aab')['a'])\n"
    )
    out = run_code(code, tier=1)
    assert "True" in out, out
    assert '{"k": 1}' in out, out


def test_tier1_cannot_walk_out_through_a_function_globals():
    """Residual of the same class: every function object exposes __globals__,
    and ntpath.join.__globals__['os'] is the real os."""
    out = run_code("import os\nprint(os.path.join.__globals__['os'].environ)", tier=1)
    assert "BLOCKED" in out or "ERROR" in out, out


def test_tier1_cannot_walk_out_through_the_object_subclass_graph():
    out = run_code("print(().__class__.__bases__[0].__subclasses__())", tier=1)
    assert "BLOCKED" in out or "ERROR" in out, out


def test_tier1_cannot_launder_the_dunder_through_getattr():
    out = run_code("import os\nprint(getattr(os.path.join, '__globals__'))", tier=1)
    assert "BLOCKED" in out or "ERROR" in out, out


def test_tier1_still_runs_ordinary_code():
    """The broadest control: the dunder guard must not break normal programs.

    No `class` statement here on purpose -- `__build_class__` is absent from
    `_TIER1_SAFE_BUILTINS`, so Tier 1 could never define a class. That is a
    pre-existing limitation, observed on the baseline, not something this
    stream introduced."""
    code = (
        "import math, json\n"
        "def double(x):\n"
        "    return x * 2\n"
        "vals = [double(i) for i in range(4)]\n"
        "print(sum(vals), math.floor(2.7), json.dumps(vals))\n"
    )
    out = run_code(code, tier=1)
    assert "12" in out and "[0, 2, 4, 6]" in out, out


# ─── B2: the Tier 2 import allow-list ────────────────────────────────────────

def test_tier2_blocks_an_import_nobody_vetted():
    assert _ast_scan("import ctypes\nctypes.windll", tier=2) is not None


def test_tier2_blocks_multiprocessing():
    assert _ast_scan("import multiprocessing", tier=2) is not None


def test_tier2_blocks_a_from_import_of_the_same_module():
    assert _ast_scan("from ctypes import windll", tier=2) is not None


def test_tier2_still_allows_the_network_it_exists_for():
    """Tier 2 is chosen precisely for goals needing an API call. A fix that
    breaks that has broken the feature, not secured it."""
    assert _ast_scan("import requests\nrequests.get('https://x')", tier=2) is None


def test_tier2_still_allows_json_and_pathlib():
    assert _ast_scan("import json, pathlib", tier=2) is None


def test_tier2_still_allows_os_and_sys():
    """Control: essentially every generated Tier 2 script reads an injected
    credential out of os.environ."""
    assert _ast_scan("import os, sys\nprint(os.environ['X'])", tier=2) is None


def test_tier2_still_allows_a_deliberately_installed_sdk():
    """Control, and the reason the allow-list is scoped to the standard library:
    `_ensure_packages` pip-installs whatever the route's `requires` names --
    spotipy, google-api-python-client, python-docx. A flat stdlib-shaped
    allow-list would refuse every SDK task Tier 2 exists to run."""
    assert _ast_scan("import spotipy\nfrom docx import Document", tier=2) is None


def test_tier1_is_unchanged():
    """Control: the tier-specific gap must close without loosening tier 1."""
    assert _ast_scan("import socket", tier=1) is not None


def test_the_regex_fallback_applies_the_same_tier2_restriction():
    """A SyntaxError routes around the AST path entirely. The fallback was the
    weaker of the two, which is a hole in its own right."""
    broken = "import ctypes\nthis is not python("
    assert _ast_scan(broken, tier=2) is not None
    assert sandbox._regex_scan_fallback(broken, tier=2) is not None


def test_the_regex_fallback_still_admits_requests():
    broken = "import requests\nthis is not python("
    assert sandbox._regex_scan_fallback(broken, tier=2) is None


# ─── B3: the Tier 2 environment allow-list ───────────────────────────────────

def test_tier2_does_not_copy_the_real_process_environment():
    from unittest.mock import patch
    with patch.dict(os.environ, {"GEMINI_API_KEY": "sk-real-secret-value"}):
        out = sandbox.run_code(
            "import os\nprint(os.environ.get('GEMINI_API_KEY', 'ABSENT'))", tier=2)
    assert "sk-real-secret-value" not in out, out
    assert "ABSENT" in out, out


def test_tier2_still_receives_deliberately_injected_credentials():
    """The orchestrator injects service credentials for the goal at hand. That
    is the mechanism, not the leak -- it must keep working."""
    out = sandbox._run_tier2(
        "import os\nprint(os.environ.get('SERVICE_TOKEN', 'ABSENT'))",
        env_vars={"SERVICE_TOKEN": "injected-on-purpose"})
    assert "injected-on-purpose" in out, out


def test_tier2_still_receives_the_variables_python_needs_to_run():
    out = sandbox.run_code("import os\nprint(bool(os.environ.get('PATH')))", tier=2)
    assert "True" in out, out


def test_tier2_still_receives_sandbox_dir():
    out = sandbox.run_code("import os\nprint(bool(os.environ.get('SANDBOX_DIR')))", tier=2)
    assert "True" in out, out


def test_tier2_can_still_import_and_use_a_third_party_library():
    """The widest control on B2+B3 together: a real subprocess run that imports
    the network library and resolves a home directory. If the env allow-list is
    too tight, `pathlib.Path.home()` is where it shows up first on Windows."""
    code = (
        "import pathlib, requests\n"
        "print(bool(pathlib.Path.home()))\n"
        "print(hasattr(requests, 'get'))\n"
    )
    out = sandbox.run_code(code, tier=2)
    assert out.count("True") == 2, out


# ─── B4: the shell validator ─────────────────────────────────────────────────

def test_the_allow_list_is_not_satisfied_by_an_executable_not_on_it():
    """rstrip takes a character set, not a suffix: 'regex.exe' loses any
    trailing run of '.', 'e', 'x' and normalises to 'reg', which is allowed."""
    assert sc._validate_shell_command("regex.exe --help") is not None


def test_a_real_exe_suffix_is_still_stripped():
    """Control: the normalisation the buggy call was reaching for must survive."""
    assert sc._validate_shell_command("ping.exe 127.0.0.1") is None


def test_output_redirection_is_rejected():
    """'>' is absent from _SHELL_METACHAR_RE, and _run_shell uses shell=True.
    Arbitrary file create/overwrite with the compound-command guard satisfied."""
    assert sc._validate_shell_command(
        r"ping 127.0.0.1 > C:\Users\Public\evil.bat") is not None


def test_input_redirection_is_rejected():
    assert sc._validate_shell_command("ping < payload.txt") is not None


def test_environment_variable_expansion_is_rejected():
    assert sc._validate_shell_command("ping %USERPROFILE%") is not None


def test_service_creation_is_rejected():
    """sc is allow-listed and the banned list covers delete/stop/config but
    not create -- service-based persistence."""
    assert sc._validate_shell_command(
        r"sc create tenkasvc binPath= C:\payload.exe") is not None


def test_remote_share_mount_is_rejected():
    assert sc._validate_shell_command(r"net use \\attacker\share") is not None


def test_registry_import_is_rejected():
    assert sc._validate_shell_command("reg import evil.reg") is not None


def test_an_ordinary_allowed_command_still_passes():
    """The validator must not collapse into refusing everything."""
    assert sc._validate_shell_command("ipconfig /all") is None
    assert sc._validate_shell_command("tasklist") is None
    assert sc._validate_shell_command("netsh wlan show profiles") is None
    assert sc._validate_shell_command("sc query bthserv") is None
    assert sc._validate_shell_command(r"reg query HKLM\SOFTWARE\Microsoft") is None


def test_run_shell_does_not_use_the_shell():
    """The validator forbids compound commands, so shell=True buys nothing and
    costs the whole metacharacter class. An argv list deletes the class."""
    import inspect
    src = inspect.getsource(sc._run_shell)
    assert "shell=True" not in src


def test_run_shell_still_returns_real_output():
    """Windows argv splitting is where this breaks if it breaks. Assert on real
    output from a real process, not on the validator agreeing."""
    ok, out = sc._run_shell(["ipconfig", "/all"])
    assert ok, out
    assert "Windows IP Configuration" in out, out[:200]


def test_run_system_command_end_to_end_still_returns_real_output():
    """The whole B4 path: validator -> shlex split -> argv -> subprocess. No LLM
    call; the generator is stubbed with the command a real route would emit."""
    async def fake_llm(prompt, task_type=None):
        return "ipconfig /all"

    out = asyncio.run(sc.run_system_command("show my ip configuration", fake_llm))
    assert "Windows IP Configuration" in out, out[:200]


def test_a_known_command_still_runs_through_the_argv_path():
    """`_run_known_command` hands `_run_shell` a plain string today. The argv
    change must carry that call site with it."""
    ok, out = sc._run_known_command({"cmd": "netsh wlan show profiles",
                                     "elevated": False})
    assert ok, out


def test_an_encoded_command_is_still_banned():
    r"""Pre-existing failure found on the baseline, in a file this stream owns:
    `\b-EncodedCommand\b` never matches, because the character before '-' is a
    space and neither is a word character, so there is no boundary there."""
    assert sc._check_banned_patterns(
        "powershell -EncodedCommand ZABpAHMAYQBiAGwAZQ") is not None
