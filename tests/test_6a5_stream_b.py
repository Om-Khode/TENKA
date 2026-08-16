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


# ─── B5: a working control must not report itself as a broken script ─────────
# Live test: the operator asked TENKA to print GEMINI_API_KEY. Tier 1 hid it
# correctly and `os.environ.get(key, '')` returned ''. Empty output tripped the
# orchestrator's retry, the model guessed another key, that was empty too, and
# the user was told the script "didn't quite work" -- along with code to run it
# again. A guard that reports itself as a bug teaches people to route around it,
# and it burned two LLM calls on a retry that could never have succeeded.

def _withheld_probe(default_expr: str = "''") -> str:
    return f"import os\nprint(os.environ.get('GEMINI_API_KEY', {default_expr}))"


def test_tier1_says_a_secret_was_withheld_instead_of_returning_nothing():
    from unittest.mock import patch
    with patch.dict(os.environ, {"GEMINI_API_KEY": "sk-real-secret-value"}):
        out = run_code(_withheld_probe(), tier=1)
    assert "sk-real-secret-value" not in out, out
    assert out.startswith("BLOCKED:"), out
    assert "hidden" in out.lower(), out


def test_the_withheld_message_does_not_reveal_whether_the_variable_is_set():
    """The message must not become an oracle for probing which credentials this
    machine holds. Set and unset have to be indistinguishable."""
    from unittest.mock import patch
    with patch.dict(os.environ, {"GEMINI_API_KEY": "sk-real-secret-value"}):
        when_set = run_code(_withheld_probe(), tier=1)
    env_without = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
    with patch.dict(os.environ, env_without, clear=True):
        when_unset = run_code(_withheld_probe(), tier=1)
    assert when_set == when_unset, (when_set, when_unset)
    assert when_set.startswith("BLOCKED:"), when_set


def test_a_subscript_lookup_of_a_secret_reports_the_same_way():
    """`os.environ['SECRET']` raised KeyError, which reads as 'not set' and is
    the same misleading shape."""
    out = run_code("import os\nprint(os.environ['SOME_API_KEY'])", tier=1)
    assert out.startswith("BLOCKED:"), out


def test_the_withheld_message_is_short_enough_to_speak():
    """It can reach TTS: under 120 chars, no paths, no error codes."""
    from unittest.mock import patch
    with patch.dict(os.environ, {"GEMINI_API_KEY": "x"}):
        out = run_code(_withheld_probe(), tier=1)
    assert len(out) < 120, (len(out), out)


# Controls: the refusal must not fire on anything but a withheld secret.

def test_a_script_reading_an_ordinary_variable_and_printing_nothing_is_unchanged():
    out = run_code("import os\nv = os.environ.get('PATH')", tier=1)
    assert out == "(no output)", out


def test_a_script_that_touches_no_environment_at_all_is_unchanged():
    out = run_code("x = 1 + 1", tier=1)
    assert out == "(no output)", out


def test_a_script_that_prints_after_touching_a_secret_keeps_its_output():
    """Only an empty run is reinterpreted. A script that produced real output
    must return it."""
    out = run_code(
        "import os\nprint('ok:', os.environ.get('SOME_API_KEY', 'hidden'))", tier=1)
    assert "ok: hidden" in out, out
    assert "BLOCKED" not in out, out


def test_the_readonly_proxy_still_hides_the_value_itself():
    """The original control, restated: the fix is about the *message*, and the
    value must still never appear."""
    from unittest.mock import patch
    with patch.dict(os.environ, {"GEMINI_API_KEY": "sk-real-secret-value"}):
        out = run_code(
            "import os\nprint('v=', os.environ.get('GEMINI_API_KEY', 'hidden'))", tier=1)
    assert "sk-real-secret-value" not in out, out
    assert "hidden" in out, out


# ─── B2b: a spawn primitive is a capability, not a spelling ──────────────────
# The stdlib scoping in B2 admitted any non-stdlib import on the reasoning that
# it could only resolve to something `_ensure_packages` deliberately installed.
# That reasoning was wrong: requirements.txt is installed into the same
# interpreter `_run_tier2` spawns via sys.executable, so every TENKA dependency
# is importable with no install step at all. `psutil.Popen` is a subprocess
# .Popen subclass, and `_BANNED_CALLS_TIER2` bans the *pair* ("subprocess",
# "Popen") -- a spelling, not the capability.

def test_a_spawn_primitive_is_importable_without_any_install_step():
    """The premise of the finding, asserted rather than assumed: psutil ships in
    requirements.txt, so it is already in the interpreter `_run_tier2` spawns --
    no `_ensure_packages` step, no route that asked for it."""
    import importlib
    import psutil
    assert importlib.util.find_spec("psutil") is not None
    assert importlib.util.find_spec("playwright") is not None
    assert callable(psutil.Popen)


def test_tier2_blocks_a_spawn_primitive_reached_through_a_third_party_wrapper():
    assert _ast_scan("import psutil\npsutil.Popen(['cmd', '/c', 'ver'])", tier=2) is not None


def test_tier2_blocks_a_spawn_primitive_under_a_module_alias():
    """An alias defeats any check keyed on the module name."""
    assert _ast_scan("import psutil as p\np.Popen(['cmd'])", tier=2) is not None


def test_tier2_blocks_a_spawn_primitive_imported_by_name():
    assert _ast_scan("from psutil import Popen\nPopen(['cmd'])", tier=2) is not None


def test_tier2_blocks_a_spawn_primitive_imported_under_an_alias():
    """`from x import Popen as q` cannot be caught at the call site, so the
    import of the name is what has to be refused."""
    assert _ast_scan("from psutil import Popen as q\nq(['cmd'])", tier=2) is not None


def test_tier2_blocks_a_browser_launch():
    """playwright is installed for TENKA's own automation tier and launches
    browser processes. Same class as psutil.Popen."""
    code = ("from playwright.sync_api import sync_playwright\n"
            "sync_playwright().start().chromium.launch()\n")
    assert _ast_scan(code, tier=2) is not None


def test_tier2_blocks_a_shell_out_however_it_is_spelled():
    for snippet in ("import os\nos.startfile('x.exe')",
                    "import os\nos.execv('cmd', [])",
                    "import os\nos.spawnv(0, 'cmd', [])",
                    "import asyncio\nasyncio.create_subprocess_shell('dir')"):
        assert _ast_scan(snippet, tier=2) is not None, snippet


def test_tier1_cannot_reach_a_spawn_primitive_through_an_allowed_module():
    """psutil is on the Tier 1 allow-list, and Popen is a public class
    attribute -- so _SafeModule's submodule rule never sees it."""
    out = run_code("import psutil\nprint(psutil.Popen)", tier=1)
    assert "BLOCKED" in out or "ERROR" in out, out


def test_tier1_really_cannot_spawn_a_process():
    """Behavioural, not just scanner-level: `cmd /c ver` is harmless, and a
    returncode printed here is a process this sandbox started."""
    out = run_code(
        "import psutil\np = psutil.Popen(['cmd', '/c', 'ver'])\nprint(p.wait())",
        tier=1)
    assert "BLOCKED" in out or "ERROR" in out, out
    assert "\n0" not in out, out


def test_the_regex_fallback_also_blocks_a_spawn_primitive():
    broken = "import psutil\npsutil.Popen(['cmd'])\nthis is not python("
    assert _ast_scan(broken, tier=2) is not None
    assert sandbox._regex_scan_fallback(broken, tier=2) is not None


def test_process_control_names_are_banned_as_one_set():
    """`kill` and `killpg` were banned while `terminate` was not: an
    inconsistency in the set rather than a considered boundary. The primitive it
    left reachable stops TENKA's own process, or anything else the user's token
    can touch."""
    for name in ("kill", "killpg", "terminate", "send_signal"):
        assert _ast_scan(f"import psutil\npsutil.Process(1234).{name}()",
                         tier=2) is not None, name


def test_tier1_cannot_reach_a_process_control_primitive():
    out = run_code("import psutil\nprint(psutil.Process().terminate)", tier=1)
    assert "BLOCKED" in out or "ERROR" in out, out


def test_binding_the_primitive_to_a_local_first_does_not_evade_the_check():
    """Found while closing `terminate`: the call-site check reads a name, and
    `t = p.terminate` leaves a bare ast.Name at the call. So the *reference* is
    what has to be refused, not only the call."""
    for tier in (1, 2):
        assert _ast_scan("import psutil\np = psutil.Process(1)\n"
                         "t = p.terminate\nt()", tier=tier) is not None, tier
        assert _ast_scan("import psutil\nspawn = psutil.Popen\n"
                         "spawn(['cmd'])", tier=tier) is not None, tier


def test_the_regex_fallback_also_blocks_process_control():
    broken = "import psutil\npsutil.Process(1).terminate()\nthis is not python("
    assert sandbox._regex_scan_fallback(broken, tier=2) is not None


# Controls: over-blocking here breaks real Tier 2 goals.

def test_psutil_still_works_for_what_it_is_on_the_list_for():
    out = run_code("import psutil\nprint(type(psutil.cpu_percent()).__name__)", tier=1)
    assert "float" in out or "int" in out, out
    assert _ast_scan("import psutil\nprint(psutil.cpu_percent(), psutil.virtual_memory())",
                     tier=2) is None


def test_a_zero_argument_system_call_is_not_a_shell_out():
    """`platform.system()` returns the string 'Windows'. `os.system(cmd)` runs
    a command. The discriminator is the argument, not the name -- banning the
    name outright would break the single most common platform call there is."""
    assert _ast_scan("import platform\nprint(platform.system())", tier=2) is None
    assert _ast_scan("import platform\nprint(platform.system())", tier=1) is None
    assert _ast_scan("import os\nos.system('dir')", tier=2) is not None


def test_execute_is_not_exec():
    """No prefix matching on 'exec': `cursor.execute(...)` is the ordinary way
    to use sqlite3, and a prefix rule would refuse every database script."""
    assert _ast_scan("import sqlite3\nc = sqlite3.connect(':memory:')\n"
                     "c.execute('select 1')", tier=2) is None


def test_the_legitimate_sdk_imports_still_pass():
    """The B2 control, re-asserted after the call-name ban: importing an SDK
    was never the problem, spawning from one was."""
    assert _ast_scan("import spotipy\nfrom docx import Document\n"
                     "import requests\nrequests.get('https://x')", tier=2) is None


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
