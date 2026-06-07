"""discovery.py — Generic API discovery and deterministic fix infrastructure."""

import logging
import re

logger = logging.getLogger("code_executor")

from .. import service_registry as _sr
from .sandbox import _run_tier2
from ._utils import _syntax_check, _sanitize_debug_output


def _extract_api_urls(code: str) -> list[str]:
    """
    Extract API URL paths from generated code.
    Returns clean path portions like '/v1/playlists/{id}/tracks'.
    Strips query parameters to avoid garbage in search queries.
    """
    url_pattern = re.compile(r"""https?://[^'"\s]+?(/v\d+/[^'"\s,)]+)""")
    matches = url_pattern.findall(code)

    seen: set[str] = set()
    paths: list[str] = []
    for m in matches:
        # Strip query params
        path = m.split('?')[0].strip("'\"")
        # Normalize f-string variables to {id}
        path = re.sub(r'\{[^}]+\}', '{id}', path)
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths

def _extract_api_calls(code: str) -> list[str]:
    """Extract API method calls from code. Returns list of method names."""
    pattern = re.compile(r'\b(\w{1,30})\.([\w]+)\s*\(', re.MULTILINE)
    matches = pattern.findall(code)

    _SKIP_METHODS = frozenset({
        'get', 'set', 'pop', 'keys', 'values', 'items', 'append',
        'strip', 'lower', 'upper', 'split', 'join', 'format',
        'encode', 'decode', 'replace', 'startswith', 'endswith',
        'environ', 'path', 'exit', 'dumps', 'loads', 'getenv',
        'Client', 'write', 'read', 'close',
        'sleep', 'strftime', 'isoformat',
        'group', 'match', 'search', 'findall', 'sub', 'compile',
    } | frozenset(_sr.ALL_CONSTRUCTOR_NAMES))
    _SKIP_VARS = frozenset({
        'os', 'sys', 'json', 're', 'str', 'int', 'dict', 'list',
        'print', 'requests', 'httpx', 'time', 'datetime',
        'match', 'pattern', 'm',
        'stdout', 'stderr', 'stdin',
    })

    api_methods = []
    seen: set[str] = set()
    for var, method in matches:
        if var in _SKIP_VARS or method in _SKIP_METHODS or method.startswith('_'):
            continue
        key = f"{var}.{method}"
        if key not in seen:
            seen.add(key)
            api_methods.append(method)
    return api_methods


def _find_client_var(code: str) -> str | None:
    """Find the API client variable name from generated code."""
    _SKIP_VARS = frozenset({
        'os', 'sys', 'json', 'import', 'TOKEN', 'token', 'type',
        'access_token', 'refresh_token', 'creds', 'credentials',
        'response', 'result', 'results', 'data', 'output',
    })
    m = re.search(r'(\w+)\s*=\s*\w+\.([A-Z]\w+)\s*\(', code)
    if m and m.group(1) not in _SKIP_VARS:
        return m.group(1)
    m = re.search(r'(\w+)\s*=\s*(build|Client|' + '|'.join(re.escape(c) for c in _sr.ALL_CONSTRUCTOR_NAMES) + r')\s*\(', code)
    if m and m.group(1) not in _SKIP_VARS:
        return m.group(1)
    m = re.search(r'(\w+)\s*=\s*\w+\.\w+\(', code)
    if m and m.group(1) not in _SKIP_VARS:
        return m.group(1)
    return None


def _build_flat_setup(code: str, client_var: str) -> str:
    """Extract import + auth + client init from code, flattened to module level."""
    lines = code.splitlines()
    imports = []
    for line in lines:
        s = line.strip()
        if s.startswith(("import ", "from ")):
            if not any(ban in s for ban in _sr.ALL_BANNED_AUTH_CLASSES):
                imports.append(s)

    stripped_lines = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith(("def ", "class ", "@", "#", "import ", "from ")):
            continue
        if re.match(r'^\w+\(\)\s*$', s) or s in ("return", "pass", "break", "continue"):
            continue
        stripped_lines.append(s)

    _ENV_KW = ('os.environ', 'os.getenv', '_ACCESS_TOKEN', '_API_KEY',
               '_CLIENT_ID', '_CLIENT_SECRET', '_REFRESH_TOKEN')
    first_env_idx, client_init_idx = None, None
    for i, s in enumerate(stripped_lines):
        if first_env_idx is None and any(kw in s for kw in _ENV_KW):
            first_env_idx = i
        if re.match(rf'\b{re.escape(client_var)}\s*=', s):
            client_init_idx = i
            break

    if first_env_idx is not None and client_init_idx is not None:
        auth_lines = stripped_lines[first_env_idx:client_init_idx + 1]
    elif first_env_idx is not None:
        auth_lines = stripped_lines[first_env_idx:]
    elif client_init_idx is not None:
        auth_lines = [stripped_lines[client_init_idx]]
    else:
        auth_lines = []

    setup = []
    for s in auth_lines:
        if '=' in s and not s.startswith(('print(', 'results', 'response', 'data')):
            if 'NEEDS_OAUTH' not in s:
                # Validate line is syntactically complete — skip truncated lines
                # like "auth_url =" or "artists = ', '.join([a.ge"
                try:
                    compile(s, "<line_check>", "exec")
                    setup.append(s)
                except SyntaxError:
                    logger.debug(f"[CODE] Flat setup: skipping broken line: {s[:60]}")
    return "\n".join(imports + [""] + setup)


def _build_discovery_script(code: str, env_vars: dict, error_category: str = "") -> str | None:
    """
    Build a GENERIC discovery script. Handles TWO modes:

    1. SDK replay: sp.method() → call each method with no args, dump result.
       Good for understanding API structure (what methods return what shape).

    2. HTTP injection: take the original code up to its print loop, inject a
       JSON dump. Runs the FULL code path including ID resolution.
       Good for seeing what the API ACTUALLY returned in the failing call.

    Mode selection:
    - no_output / field_access errors → prefer HTTP injection (the API was
      called, we need to see what came back before the print loop mangled it)
    - Other errors → prefer SDK replay (understand API structure)
    - Falls back between modes if the preferred one can't build a script.
    """
    # ── Common _dump helper ──
    dump_helper = [
        "import json, os, requests, sys",
        "",
        "def _dump(label, obj):",
        "    try:",
        "        print(f'DISCOVERY:{label}:type={type(obj).__name__}')",
        "        if isinstance(obj, dict):",
        "            print(f'DISCOVERY:{label}:keys={list(obj.keys())}')",
        "            # Probe ALL list-valued keys — no hardcoded collection names",
        "            for coll_key, coll_val in obj.items():",
        "                if isinstance(coll_val, list) and len(coll_val) > 0:",
        "                    print(f'DISCOVERY:{label}:{coll_key}_count={len(coll_val)}')",
        "                    first = coll_val[0]",
        "                    if isinstance(first, dict):",
        "                        print(f'DISCOVERY:{label}:first_{coll_key}_keys={list(first.keys())}')",
        "                        # Deep dive: show keys of dict sub-values in first item",
        "                        for sub_key, sub_val in first.items():",
        "                            if isinstance(sub_val, dict) and sub_val:",
        "                                print(f'DISCOVERY:{label}:first_{coll_key}[{sub_key}]:keys={list(sub_val.keys())[:15]}')",
        "                                # Show first few values for identification",
        "                                _preview = {k: str(v)[:50] for k, v in list(sub_val.items())[:5]}",
        "                                print(f'DISCOVERY:{label}:first_{coll_key}[{sub_key}]:preview={_preview}')",
        "                    break",
        "        elif isinstance(obj, list) and len(obj) > 0:",
        "            print(f'DISCOVERY:{label}:list_len={len(obj)}')",
        "            if isinstance(obj[0], dict):",
        "                print(f'DISCOVERY:{label}:first_keys={list(obj[0].keys())}')",
        "                for sub_key, sub_val in obj[0].items():",
        "                    if isinstance(sub_val, dict) and sub_val:",
        "                        print(f'DISCOVERY:{label}:first[{sub_key}]:keys={list(sub_val.keys())[:15]}')",
        "    except Exception as e:",
        "        print(f'DISCOVERY:{label}:error={e}')",
        "",
    ]

    # ── Decide which mode to try first ──
    # For no_output / field_access: the API was called, we need to see the
    # actual response. HTTP injection runs the full code path and dumps.
    # For other errors: SDK replay shows API structure.
    prefer_injection = error_category in ("no_output", "field_access")

    if prefer_injection:
        script = _build_http_injection_script(code, dump_helper)
        if script:
            return script
        # Fall through to SDK mode

    # ── SDK-based discovery ──
    api_methods = _extract_api_calls(code)
    client_var = _find_client_var(code)

    if client_var and api_methods:
        logger.info(f"[CODE] Discovery: SDK mode — client='{client_var}', methods={api_methods}")
        setup = _build_flat_setup(code, client_var)
        parts = [setup, ""] + dump_helper

        for method in api_methods:
            if any(skip in method.lower() for skip in ('auth', 'token', 'login')):
                continue
            parts.extend([
                f"try:",
                f"    _r = {client_var}.{method}()",
                f"    _dump('{method}', _r)",
                f"except TypeError as e:",
                f"    print(f'DISCOVERY:{method}:needs_args={{e}}')",
                f"    try:",
                f"        import inspect as _insp",
                f"        _sig = _insp.signature({client_var}.{method})",
                f"        print(f'DISCOVERY:{method}:signature={{_sig}}')",
                f"    except (ValueError, TypeError):",
                f"        pass",
                f"except Exception as e:",
                f"    print(f'DISCOVERY:{method}:error={{e}}')",
                "",
            ])
        return "\n".join(parts)

    # ── Fallback: HTTP injection (if not already tried) ──
    if not prefer_injection:
        script = _build_http_injection_script(code, dump_helper)
        if script:
            return script

    logger.info("[CODE] Discovery: no SDK client and no injectable loop found")
    return None


def _build_http_injection_script(code: str, dump_helper: list[str]) -> str | None:
    """
    Build a discovery script that makes ONE API call and dumps the response.

    Strategy:
      1. Find the API call that produces the data being iterated.
      2. Extract setup code (imports, auth, client init, ID resolution)
         but EXCLUDE while/for loops to avoid pagination timeouts.
      3. Run the API call ONCE and dump whatever comes back.

    This avoids the fundamental problem with running the full code path:
    while-True pagination loops will timeout in the 30s discovery window.
    """
    lines = code.split('\n')

    # ── Step 1: Find the main data-fetching API call ──
    # Look for patterns like: res = sp.playlist_tracks(...)
    #                          batch = sp.playlist_items(...)
    # Inside or just before while/for loops that process items.
    _INIT_SKIP = re.compile(
        r'\b(' + _sr.CONSTRUCTOR_SKIP_PATTERN + r'|current_user_playlists|'
        r'current_user|reconfigure)\s*\(', re.IGNORECASE
    )
    # Skip dict access methods — these are NOT API calls
    _DICT_METHODS = frozenset({
        'get', 'keys', 'values', 'items', 'pop', 'update', 'setdefault',
        'append', 'extend', 'insert', 'remove', 'strip', 'lower', 'upper',
        'split', 'join', 'replace', 'format', 'encode', 'decode',
        'startswith', 'endswith', 'casefold', 'count', 'find', 'index',
        'getenv', 'environ',  # os.getenv(), os.environ.get()
    })
    api_call_line = None
    api_call_var = None

    for i, ln in enumerate(lines):
        stripped = ln.strip()
        # Match: var = object.method(...)
        m = re.match(r'(\w+)\s*=\s*(\w+)\.(\w+)\(([^)]*)\)', stripped)
        if not m:
            # Also match chained: var = object.method(...).get(...)
            m2 = re.match(r'(\w+)\s*=\s*(\w+)\.(\w+)\(([^)]*)\)\.get\(', stripped)
            if m2:
                m = m2
        if m:
            var_name = m.group(1)
            obj_name = m.group(2)
            method_name = m.group(3)
            call = f"{obj_name}.{method_name}({m.group(4)})"
            # Skip dict/string methods
            if method_name in _DICT_METHODS:
                continue
            # Skip client init and non-data calls
            if _INIT_SKIP.search(call):
                continue
            # Skip env var reads
            if 'environ' in call or 'getenv' in call:
                continue
            api_call_line = i
            api_call_var = var_name
            # Keep scanning — last data call wins (the main one)

    if api_call_line is None:
        logger.info("[CODE] Discovery: no injectable API call found")
        return None

    logger.info(f"[CODE] Discovery: found API call '{api_call_var} = ...' at line {api_call_line}")

    # ── Step 2: Build setup code — everything BEFORE the API call, ──
    # ── but SKIP while-True loops (pagination) that would timeout. ──
    # Keep for-loops (they resolve IDs, iterate playlists, etc.)
    setup_lines = []
    skip_until_dedent = False
    loop_indent = 0

    for i in range(api_call_line):
        ln = lines[i]
        stripped = ln.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith('#'):
            setup_lines.append(ln)
            continue

        indent = len(ln) - len(ln.lstrip())

        # If we're skipping a loop body, check if we've dedented
        if skip_until_dedent:
            if indent > loop_indent:
                continue  # still inside loop body
            else:
                skip_until_dedent = False

        # Only skip while-True loops (pagination) — they timeout.
        # Keep for-loops — they typically resolve IDs, iterate short lists.
        if stripped.startswith('while ') and stripped.endswith(':'):
            skip_until_dedent = True
            loop_indent = indent
            continue

        setup_lines.append(ln)

    # ── Step 3: Extract the API call itself ──
    # Take the exact line, but strip any .get() chain to get raw response
    call_line = lines[api_call_line].strip()
    chain_match = re.match(r'(\w+)\s*=\s*(.+?)\.get\s*\([^)]*\)\s*$', call_line)
    if chain_match:
        # Split: var = api.method(...).get('items', [])
        # → _raw = api.method(...)
        raw_call = chain_match.group(2)
        api_call_var = '_discovery_raw'
        call_code = f"{api_call_var} = {raw_call}"
        logger.info(f"[CODE] Discovery: stripped .get() chain — dumping raw response")
    else:
        call_code = call_line

    # ── Step 4: Assemble the discovery script ──
    setup = "\n".join(setup_lines)
    dump_code = "\n".join(dump_helper)

    script = f"""{setup}

# ── Discovery: single API call ──
{call_code}
import json
{dump_code}
try:
    _raw = {api_call_var} if isinstance({api_call_var}, dict) else {api_call_var}.json()
    _dump('response', _raw)
except Exception as e:
    print(f'DISCOVERY:response:error={{e}}')
"""

    logger.info(f"[CODE] Discovery: HTTP injection mode — single call to {api_call_var}")
    return script


def _enrich_discovery_with_key_analysis(code: str, discovery_data: str) -> tuple[str, dict[str, str]]:
    """
    Compare keys the code accesses via .get() or [] against keys discovery found.
    Appends explicit MISMATCH lines so the fix plan can't miss renamed fields.

    Returns:
        (enriched_discovery_text, key_replacements)
        key_replacements is a dict of {old_key: new_key} for deterministic fixing.
        Only populated when a confident mapping is found.
    """
    # Extract keys the code accesses on collection items
    # Patterns: item.get('track'), t['track'], t.get('track', {})
    code_keys = set()
    for m in re.finditer(r"\.get\(\s*['\"](\w+)['\"]", code):
        code_keys.add(m.group(1))
    for m in re.finditer(r"\['\s*(\w+)\s*'\]", code):
        code_keys.add(m.group(1))

    # Filter out env var names — ALL_UPPER_CASE keys are os.environ.get(),
    # not API field access. Also filter PARAM_* which are injected env vars.
    code_keys = {k for k in code_keys if not k.isupper() and not k.startswith('PARAM_')}

    # Extract keys discovery found in first collection item
    api_keys = set()
    for m in re.finditer(r'first_\w+_keys=\[([^\]]+)\]', discovery_data):
        for k in re.findall(r"'(\w+)'", m.group(1)):
            api_keys.add(k)

    # Skip common/structural keys that appear in both code and API responses
    _SKIP_KEYS = {'items', 'name', 'id', 'href', 'next', 'offset', 'limit',
                  'total', 'previous', 'type', 'uri'}

    # Find keys code expects but API doesn't have
    missing_in_api = (code_keys - api_keys) - _SKIP_KEYS
    # Find keys API has but code doesn't use (potential replacements)
    extra_in_api = (api_keys - code_keys) - _SKIP_KEYS

    key_replacements: dict[str, str] = {}

    if missing_in_api:
        mismatch_lines = []
        for mk in missing_in_api:
            mismatch_lines.append(
                f"DISCOVERY:KEY_MISMATCH: code uses .get('{mk}') but API response does NOT have '{mk}' key"
            )
        if extra_in_api:
            mismatch_lines.append(
                f"DISCOVERY:KEY_MISMATCH: API has these keys instead: {sorted(extra_in_api)}"
            )
            # Try to suggest the replacement and build the key map
            for mk in missing_in_api:
                # Find what fields the code accesses on the missing key's value
                # e.g. if code has `track.get('name')`, `track.get('artists')`
                # then accessed_fields = {'name', 'artists'}
                _mk_access_patterns = [
                    rf"\.get\('{mk}'[^)]*\)\s*\.\s*get\(\s*['\"](\w+)['\"]",  # item.get('track').get('name')
                    rf"(\w+)\s*=\s*\w+\.get\('{mk}'",  # track = item.get('track') → then track.get(X)
                ]
                accessed_fields = set()
                for pat in _mk_access_patterns:
                    for m in re.finditer(pat, code):
                        accessed_fields.add(m.group(1))
                # Also find: if track = item.get('track'), then find track.get('X')
                mk_var_match = re.search(rf"(\w+)\s*=\s*\w+\.get\('{mk}'", code)
                if mk_var_match:
                    mk_var = mk_var_match.group(1)
                    for m in re.finditer(rf"{mk_var}\.get\(\s*['\"](\w+)['\"]", code):
                        accessed_fields.add(m.group(1))

                candidates = []
                for ek in extra_in_api:
                    # Check if the extra key's sub-object has fields the code expects
                    ek_sub_keys = set()
                    for sm in re.finditer(rf"first_\w+\[{ek}\]:keys=\[([^\]]+)\]", discovery_data):
                        for sk in re.findall(r"'(\w+)'", sm.group(1)):
                            ek_sub_keys.add(sk)

                    if ek_sub_keys:
                        # Verify: do the sub-keys overlap with what code accesses?
                        if accessed_fields and (accessed_fields & ek_sub_keys):
                            candidates.append(ek)
                            mismatch_lines.append(
                                f"DISCOVERY:KEY_MISMATCH: '{mk}' should likely be replaced with '{ek}' (sub-keys {accessed_fields & ek_sub_keys} match code access pattern)"
                            )
                        elif not accessed_fields:
                            # Can't determine accessed fields — accept any dict sub-key
                            candidates.append(ek)
                            mismatch_lines.append(
                                f"DISCOVERY:KEY_MISMATCH: '{mk}' should likely be replaced with '{ek}' (it contains nested data)"
                            )

                # Build confident replacement: exactly one candidate matched
                if len(candidates) == 1:
                    key_replacements[mk] = candidates[0]
                elif len(candidates) == 0 and len(extra_in_api) == 1 and len(missing_in_api) == 1:
                    # Unambiguous 1:1 — only one missing, only one extra
                    sole_extra = next(iter(extra_in_api))
                    key_replacements[mk] = sole_extra
                    mismatch_lines.append(
                        f"DISCOVERY:KEY_MISMATCH: '{mk}' → '{sole_extra}' (only unmatched key in API response)"
                    )
                elif len(candidates) == 0 and len(missing_in_api) == 1:
                    # No sub-key confirmation (value might be None due to
                    # region/permission issues). Filter out obvious metadata
                    # field names — timestamps, booleans, thumbnails, etc.
                    # These patterns are common across all REST APIs.
                    _METADATA_PATTERNS = (
                        'added_', 'created_', 'updated_', 'modified_', 'deleted_',
                        'is_', 'has_', 'can_',
                        '_at', '_by', '_on', '_url', '_uri', '_id',
                        'thumbnail', 'color', 'preview', 'snapshot',
                        'primary_', 'secondary_',
                        'video_', 'image_', 'photo_',
                        'external_',
                    )
                    data_keys = [
                        ek for ek in extra_in_api
                        if not any(ek.startswith(p) or ek.endswith(p.lstrip('_'))
                                   for p in _METADATA_PATTERNS)
                    ]
                    if len(data_keys) == 1:
                        key_replacements[mk] = data_keys[0]
                        mismatch_lines.append(
                            f"DISCOVERY:KEY_MISMATCH: '{mk}' → '{data_keys[0]}' (only non-metadata key among extras)"
                        )

        enriched = discovery_data + "\n" + "\n".join(mismatch_lines)
        logger.info(f"[CODE] Key mismatch detected: code uses {missing_in_api}, API has {extra_in_api}")
        if key_replacements:
            logger.info(f"[CODE] Key replacements resolved: {key_replacements}")
        return enriched, key_replacements

    return discovery_data, key_replacements


# ─── Deterministic Kwarg Fixes ────────────────────────────────────────

def _extract_kwarg_fixes(result: str, discovery_data: str) -> dict[str, tuple[str, str]]:
    """Extract wrong→correct kwarg mappings from error output + discovery signatures.

    Parses "ClassName.method() got an unexpected keyword argument 'kwarg'"
    and matches against DISCOVERY:method:signature=(...) to find the real param.

    Returns: {method_name: (wrong_kwarg, correct_kwarg)}
    """
    if not discovery_data:
        return {}

    err_match = re.search(
        r"(\w+)\.(\w+)\(\) got an unexpected keyword argument ['\"](\w+)['\"]",
        result, re.IGNORECASE,
    )
    if not err_match:
        return {}

    method_name = err_match.group(2)
    wrong_kwarg = err_match.group(3)

    sig_match = re.search(
        rf"DISCOVERY:{re.escape(method_name)}:signature=\(([^)]+)\)",
        discovery_data,
    )
    if not sig_match:
        return {}

    sig_params = []
    for part in sig_match.group(1).split(","):
        param = part.strip().split("=")[0].strip()
        if param and param != "self":
            sig_params.append(param)

    for param in sig_params:
        if param != wrong_kwarg and wrong_kwarg in param:
            logger.info(f"[CODE] Kwarg fix found: {method_name}() {wrong_kwarg}= → {param}=")
            return {method_name: (wrong_kwarg, param)}

    return {}


def _apply_kwarg_fixes(code: str, kwarg_fixes: dict[str, tuple[str, str]]) -> str | None:
    """Mechanically replace wrong kwargs in specific method calls.

    Only modifies lines containing the target method call.
    Returns fixed code, or None if no fixes were needed/applied.
    """
    if not kwarg_fixes:
        return None

    fixed = code
    total = 0
    for method, (wrong, correct) in kwarg_fixes.items():
        lines = fixed.split("\n")
        for i, line in enumerate(lines):
            if f".{method}(" in line:
                new_line = re.sub(
                    rf"(?<![a-zA-Z_]){re.escape(wrong)}=",
                    f"{correct}=",
                    line,
                )
                if new_line != line:
                    lines[i] = new_line
                    total += 1
        fixed = "\n".join(lines)

    if total == 0:
        return None

    syn = _syntax_check(fixed)
    if syn:
        logger.warning(f"[CODE] Kwarg fix produced syntax error: {syn}")
        return None

    logger.info(f"[CODE] Kwarg fix applied: {total} replacement(s)")
    return fixed


def _apply_key_fixes(code: str, key_replacements: dict[str, str],
                     discovery_data: str = "") -> str | None:
    """
    Deterministic fixes based on discovery data. Two fix types:

    1. Key replacement — applies discovered key mismatches directly.
       Discovery found the old and new key names; this swaps them mechanically.

    2. Fields parameter removal — when discovery shows empty collection item
       keys (first_*_keys=[]), the fields= parameter on the API call is
       filtering out all data. Strip it so the full response comes back.

    Returns the fixed code, or None if no fixes were needed/applied.
    """
    fixed = code
    total_fixes = 0

    # ── Fix type 1: Key replacements ──
    if key_replacements:
        for old_key, new_key in key_replacements.items():
            count = 0

            # Pattern 1: .get('old_key'   → .get('new_key'
            old_p = f".get('{old_key}'"
            new_p = f".get('{new_key}'"
            c = fixed.count(old_p)
            if c:
                fixed = fixed.replace(old_p, new_p)
                count += c

            # Pattern 2: .get("old_key"   → .get("new_key"
            old_p = f'.get("{old_key}"'
            new_p = f'.get("{new_key}"'
            c = fixed.count(old_p)
            if c:
                fixed = fixed.replace(old_p, new_p)
                count += c

            # Pattern 3: ['old_key']      → ['new_key']
            old_p = f"['{old_key}']"
            new_p = f"['{new_key}']"
            c = fixed.count(old_p)
            if c:
                fixed = fixed.replace(old_p, new_p)
                count += c

            # Pattern 4: ["old_key"]      → ["new_key"]
            old_p = f'["{old_key}"]'
            new_p = f'["{new_key}"]'
            c = fixed.count(old_p)
            if c:
                fixed = fixed.replace(old_p, new_p)
                count += c

            if count:
                logger.info(f"[CODE] Key fix: '{old_key}' → '{new_key}' ({count} replacements)")
                total_fixes += count

    # ── Fix type 2: Strip broken fields= parameter ──
    # When discovery shows first_*_keys=[] (empty), it means the fields=
    # parameter is referencing fields that don't exist (e.g. the API renamed
    # them). Stripping fields= lets the full response come back so the key
    # fix above can work on actual data.
    if discovery_data and "first_items_keys=[]" in discovery_data:
        stripped = _strip_fields_param(fixed)
        if stripped != fixed:
            fixed = stripped
            total_fixes += 1

    if total_fixes == 0:
        return None

    # Verify the fix didn't break syntax
    syn = _syntax_check(fixed)
    if syn:
        logger.warning(f"[CODE] Deterministic fix produced syntax error: {syn} — reverting")
        return None

    logger.info(f"[CODE] Deterministic fix applied: {total_fixes} total fixes")
    return fixed


def _strip_fields_param(code: str) -> str:
    """
    Remove fields= parameter from API method calls.

    Handles patterns like:
      sp.playlist_items(id, fields='items(track(name))')
      sp.playlist_items(id, fields="items(track(name))")
      .method(id, fields='...').get(...)

    The fields= param is removed, other params are preserved.
    Generic — works for any SDK method that accepts fields=.
    """
    # Pattern: , fields='...'  or  , fields="..."
    # Captures the comma + space before fields= and the quoted value
    fixed = re.sub(r",\s*fields=['\"][^'\"]*['\"]", "", code)
    if fixed != code:
        logger.info("[CODE] Stripped broken fields= parameter from API call")
    return fixed


def _run_discovery(code: str, env_vars: dict, error_category: str = "") -> str | None:
    """Run the mechanical discovery script and return parsed results."""
    discovery_code = _build_discovery_script(code, env_vars, error_category=error_category)
    if discovery_code is None:
        logger.info("[CODE] Discovery: could not build script")
        return None

    syn_err = _syntax_check(discovery_code)
    if syn_err:
        logger.warning(f"[CODE] Discovery script has syntax error: {syn_err}")
        logger.debug(f"[CODE] Discovery code:\n{discovery_code}")
        return None

    result = _run_tier2(discovery_code, env_vars=env_vars, timeout=30)

    discovery_lines = [ln for ln in result.splitlines() if ln.strip().startswith("DISCOVERY:")]
    if not discovery_lines:
        logger.info(f"[CODE] Discovery produced no data. Output: {result[:300]}")
        return None

    raw = "\n".join(discovery_lines)
    sanitized = _sanitize_debug_output(raw)
    logger.info(f"[CODE] Discovery captured: {len(sanitized)} chars, {len(discovery_lines)} lines")

    # Enrich discovery with missing-key analysis if we have the original code
    return sanitized
