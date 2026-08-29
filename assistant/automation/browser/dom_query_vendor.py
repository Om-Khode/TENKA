"""
dom_query_vendor.py — the DOM extraction JS, loaded from disk and integrity-checked.

The query JS is one artifact with two copies. The source of truth lives in the Drover
extension repo (`src/shared/dom_query.js`); this package vendors a byte copy at
`vendor/dom_query.js` next to `vendor/dom_query.sha256`.

Two copies exist because Manifest V3's content-script CSP forbids evaluating JS that
arrived over the wire. The extension cannot be handed this string at call time — it has
to ship the file in its own bundle. So instead of pretending there is one copy, the two
are compared: the extension reports its file's SHA-256 in the WS `hello` frame, and a
mismatch refuses the connection and falls back to bundled Chromium, loudly.

That makes the digest load-bearing, which is why `_load` compares against a **recorded**
digest rather than recomputing one. A loader that hashed whatever it just read would
agree with itself no matter what the file contained, and the handshake would then be
comparing two tampered copies to each other.

**Windows note.** `core.autocrlf` rewrites line endings on checkout, and LF→CRLF changes
the digest with no code change involved — the handshake would start failing on a fresh
clone for a reason nobody would think to look for. The file is read as **bytes** and
decoded explicitly; `.gitattributes` marks it `binary` in this repo and in the extension
repo so git leaves the bytes alone. Both sides need that rule; one side having it is not
enough.

The file must stay a bare arrow function `(cfg) => {...}`. Playwright's `page.evaluate`
and the extension's `scripting.executeScript` both require exactly that shape; a wrapper
of any kind breaks one or both.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
_JS_PATH = _VENDOR_DIR / "dom_query.js"
_SHA_PATH = _VENDOR_DIR / "dom_query.sha256"


class DomQueryIntegrityError(RuntimeError):
    """The vendored JS does not match its recorded digest.

    Raised at import time, deliberately. A corrupted or half-synced query file
    produces an empty element tree rather than an error, and an empty tree is
    indistinguishable from a page with nothing on it — the DOM tier would report
    a confident "nothing to click" instead of failing.
    """


def _load(*, js_bytes: bytes, recorded: str) -> str:
    """Verify `js_bytes` against `recorded`, then decode.

    Split out from module import so tests can drive it with tampered bytes
    without touching the file on disk.
    """
    actual = hashlib.sha256(js_bytes).hexdigest()
    if actual != recorded:
        raise DomQueryIntegrityError(
            f"vendored dom_query.js does not match its recorded digest: "
            f"expected {recorded}, found {actual}. Either the JS was edited "
            f"without regenerating dom_query.sha256, or git rewrote its line "
            f"endings on checkout (see .gitattributes). The extension ships a "
            f"byte copy of this file and compares digests at handshake — both "
            f"copies must be updated together."
        )
    return js_bytes.decode("utf-8")


def _read_recorded() -> str:
    return _SHA_PATH.read_bytes().decode("utf-8").strip()


DOM_QUERY_SHA256: str = _read_recorded()
DOM_QUERY_JS: str = _load(js_bytes=_JS_PATH.read_bytes(), recorded=DOM_QUERY_SHA256)


def dom_query_source() -> tuple[str, str]:
    """The JS and its digest, together.

    Callers that send the digest over the wire should take both from here rather
    than pairing the module constant with a digest computed somewhere else — the
    pairing is the whole contract.
    """
    return DOM_QUERY_JS, DOM_QUERY_SHA256
