"""
test_drover_protocol.py — the two halves of the Drover protocol must agree.

The protocol is declared twice: `assistant/core/drover_protocol.py` here,
and `src/shared/protocol.js` in the extension repo. Two declarations is the
price of two runtimes, and the cost of that price is drift — a method renamed on
one side, an error code reused on the other, an element key added here and never
produced there. Every one of those failures is silent: the call simply returns
nothing useful, and the DOM tier reports an empty page.

So the halves are compared. This reads the JavaScript as text rather than
importing it: the point is that the *source of truth* files agree, not that some
build artefact does.

The extension repo is not a dependency and may not be checked out — a fresh
clone of this repo alone must still pass. When it is absent these tests SKIP,
loudly, rather than passing quietly. A skip is visible in the run; a vacuous
pass is not.

Run: py -3.11 -m pytest tests/test_drover_protocol.py -v
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

from assistant.core import drover_protocol as py_protocol  # noqa: E402
from assistant.automation.browser.dom_query_vendor import DOM_QUERY_SHA256  # noqa: E402

#: Sibling checkout. Not a dependency and not required — see the module
#: docstring for why absence is a skip and not a pass.
_EXTENSION_REPO = Path(__file__).resolve().parent.parent.parent / "TENKA-extension"
_PROTOCOL_JS = _EXTENSION_REPO / "src" / "shared" / "protocol.js"
_QUERY_JS = _EXTENSION_REPO / "src" / "shared" / "dom_query.js"


def _js() -> str:
    if not _PROTOCOL_JS.is_file():
        pytest.skip(
            f"the Drover extension repo is not checked out at {_EXTENSION_REPO}. "
            f"The protocol halves cannot be compared, so drift between them is "
            f"UNCHECKED in this run."
        )
    return _PROTOCOL_JS.read_text(encoding="utf-8")


def _js_string_list(source: str, const: str) -> list[str]:
    """Pull `export const NAME = Object.freeze([...])` out of the JS as a list."""
    m = re.search(rf"export const {const} = Object\.freeze\(\[(.*?)\]\)", source, re.S)
    assert m, f"{const} not found in protocol.js"
    return re.findall(r'"([^"]+)"', m.group(1))


def _js_object_values(source: str, const: str) -> dict[str, str | int]:
    """Pull `export const NAME = {...}` out of the JS as a dict."""
    m = re.search(rf"export const {const} = \{{(.*?)\n\}};", source, re.S)
    assert m, f"{const} not found in protocol.js"
    out: dict[str, str | int] = {}
    for key, value in re.findall(r"^\s*([A-Z_]+):\s*([^,\n]+),", m.group(1), re.M):
        raw = value.strip()
        out[key] = int(raw) if raw.isdigit() else raw.strip('"')
    return out


# ─── Version and the index attribute ─────────────────────────────────────


def test_protocol_versions_match():
    source = _js()
    m = re.search(r"export const PROTOCOL_VERSION = (\d+);", source)
    assert m, "PROTOCOL_VERSION not found in protocol.js"
    assert int(m.group(1)) == py_protocol.PROTOCOL_VERSION, (
        "the two halves declare different protocol versions. The handshake "
        "compares them and refuses a mismatch, so this ships as 'the extension "
        "will not connect' with no other symptom."
    )


def test_the_index_attribute_matches():
    source = _js()
    m = re.search(r'export const IDX_ATTR = "([^"]+)";', source)
    assert m, "IDX_ATTR not found in protocol.js"
    assert m.group(1) == py_protocol.IDX_ATTR, (
        "the halves stamp and select different attribute names. Every element "
        "resolves to nothing, and a page where nothing resolves is "
        "indistinguishable from a page with nothing on it."
    )


# ─── The tables ──────────────────────────────────────────────────────────


def test_element_keys_match():
    assert set(_js_string_list(_js(), "ELEMENT_KEYS")) == set(py_protocol.ELEMENT_KEYS)


def test_query_result_keys_match():
    assert set(_js_string_list(_js(), "QUERY_RESULT_KEYS")) == set(py_protocol.QUERY_RESULT_KEYS)


def test_actions_match():
    assert set(_js_string_list(_js(), "ACTIONS")) == set(py_protocol.ACTIONS)


def test_rpc_method_names_match():
    js = _js_object_values(_js(), "RPC")
    py = {k: v for k, v in vars(py_protocol.Rpc).items() if not k.startswith("_")}
    assert js == py, (
        "the RPC tables disagree. A name present on one side only answers "
        "UNKNOWN_METHOD, which is a reply — so the failure arrives as a task "
        "that quietly does less than it was asked."
    )


def test_error_codes_match():
    js = _js_object_values(_js(), "ERR")
    py = {k: v for k, v in vars(py_protocol.Err).items() if not k.startswith("_")}
    assert js == py, (
        "the error tables disagree. Callers branch on the numbers, so a code "
        "that means one thing here and another there is a caller retrying "
        "something that can never succeed."
    )


def test_event_names_match():
    js = _js_object_values(_js(), "EVENTS")
    assert set(js.values()) == set(py_protocol.EVENTS)


def test_frame_types_match():
    js = _js_object_values(_js(), "FRAME")
    py = {k: v for k, v in vars(py_protocol.Frame).items() if not k.startswith("_")}
    assert js == py


def test_the_tables_are_not_empty():
    # Every comparison above passes trivially if both sides parse as empty —
    # and the JS parsing here is regex, which is exactly the sort of thing that
    # silently returns nothing after an unrelated reformat.
    source = _js()
    assert len(_js_string_list(source, "ELEMENT_KEYS")) >= 15
    assert len(_js_object_values(source, "RPC")) >= 10
    assert len(_js_object_values(source, "ERR")) >= 9


# ─── The shared query file ───────────────────────────────────────────────


def test_the_vendored_query_matches_the_extensions_copy():
    if not _QUERY_JS.is_file():
        pytest.skip(f"the Drover extension repo is not checked out at {_EXTENSION_REPO}")
    import hashlib

    theirs = hashlib.sha256(_QUERY_JS.read_bytes()).hexdigest()
    assert theirs == DOM_QUERY_SHA256, (
        f"the vendored dom_query.js and the extension's copy differ.\n"
        f"  vendored here: {DOM_QUERY_SHA256}\n"
        f"  extension:     {theirs}\n"
        f"The handshake compares exactly these, so this ships as a refused "
        f"connection and a silent fall back to the bundled browser."
    )
