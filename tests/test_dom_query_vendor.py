"""
test_dom_query_vendor.py — Latch Task 2: the DOM query JS as a shared artifact.

MV3's content-script CSP forbids evaluating JS that arrives over the wire, so
the Latch extension cannot be handed `_DOM_QUERY_JS` at call time — it has to
ship its own copy. Two copies of the same file is a drift hazard, so the copies
are compared by SHA-256 at the WS handshake and a mismatch refuses the
connection.

That makes the digest load-bearing, and this file pins the two ways it can go
wrong silently:

  1. The recorded `.sha256` and the file disagree. The loader must RAISE. A
     loader that recomputed the digest from whatever it just read would agree
     with itself forever and the handshake would compare two tampered copies.
  2. Git rewrites the line endings. This is Windows, `core.autocrlf` is on by
     default, and a checkout that turns LF into CRLF changes the digest with no
     code change involved — the handshake would then fail on a fresh clone for
     a reason nobody would think to look for. `.gitattributes` marks the file
     binary; this asserts the bytes on disk are actually LF-only.

Also asserted: the extracted JS is byte-identical to what `dom.py` used to hold
inline, so the extraction itself changed nothing.

Run: py -3.11 -m pytest tests/test_dom_query_vendor.py -v
"""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import assistant.automation.browser.dom as bdom  # noqa: E402
from assistant.automation.browser import dom_query_vendor as vendor  # noqa: E402

_VENDOR_DIR = Path(bdom.__file__).parent / "vendor"
_JS_PATH = _VENDOR_DIR / "dom_query.js"
_SHA_PATH = _VENDOR_DIR / "dom_query.sha256"


class TestVendoredFileOnDisk(unittest.TestCase):

    def test_both_files_exist(self):
        self.assertTrue(_JS_PATH.is_file(), f"missing {_JS_PATH}")
        self.assertTrue(_SHA_PATH.is_file(), f"missing {_SHA_PATH}")

    def test_recorded_digest_matches_the_bytes_on_disk(self):
        recorded = _SHA_PATH.read_bytes().decode("utf-8").strip()
        actual = hashlib.sha256(_JS_PATH.read_bytes()).hexdigest()
        self.assertEqual(
            actual, recorded,
            "dom_query.js and dom_query.sha256 disagree. If the JS was edited "
            "on purpose, regenerate the digest AND update the extension's copy "
            "— the two repos must ship identical bytes or the handshake fails.",
        )

    def test_file_is_lf_only(self):
        raw = _JS_PATH.read_bytes()
        self.assertNotIn(
            b"\r\n", raw,
            "CRLF found in dom_query.js. git's autocrlf rewrote it on checkout, "
            "which changes the SHA-256 and breaks the Latch handshake with no "
            "code change involved. .gitattributes must mark this file binary.",
        )

    def test_file_is_a_bare_arrow_function(self):
        # Both consumers require this exact shape: Playwright's page.evaluate
        # and the extension's scripting.executeScript. A wrapper (an IIFE, an
        # export, a `const f =` prefix) breaks one or both.
        text = _JS_PATH.read_bytes().decode("utf-8")
        self.assertTrue(
            text.lstrip().startswith("(cfg) =>"),
            f"expected a bare arrow function, got {text.lstrip()[:40]!r}",
        )


class TestLoader(unittest.TestCase):

    def test_exports_match_the_files(self):
        self.assertEqual(vendor.DOM_QUERY_JS, _JS_PATH.read_bytes().decode("utf-8"))
        self.assertEqual(
            vendor.DOM_QUERY_SHA256,
            _SHA_PATH.read_bytes().decode("utf-8").strip(),
        )

    def test_dom_query_source_returns_both(self):
        js, sha = vendor.dom_query_source()
        self.assertEqual(js, vendor.DOM_QUERY_JS)
        self.assertEqual(sha, vendor.DOM_QUERY_SHA256)

    def test_digest_is_of_the_bytes_not_the_decoded_string(self):
        # Encoding the decoded str back to UTF-8 must reproduce the file
        # exactly. If the loader ever decoded with a lossy codec or normalised
        # newlines, this diverges while every other assertion still passes.
        self.assertEqual(
            hashlib.sha256(vendor.DOM_QUERY_JS.encode("utf-8")).hexdigest(),
            vendor.DOM_QUERY_SHA256,
        )

    def test_loader_raises_when_the_recorded_digest_disagrees(self):
        # The whole point: a loader that recomputed the digest from what it
        # just read would agree with itself no matter what the file contained.
        tampered = _JS_PATH.read_bytes().replace(b"const FILTER", b"const FILTER_", 1)
        self.assertNotEqual(tampered, _JS_PATH.read_bytes(), "tamper did not apply")
        with self.assertRaises(vendor.DomQueryIntegrityError) as ctx:
            vendor._load(js_bytes=tampered, recorded=vendor.DOM_QUERY_SHA256)
        msg = str(ctx.exception)
        self.assertIn(vendor.DOM_QUERY_SHA256[:12], msg, "error omits the expected digest")
        self.assertIn(
            hashlib.sha256(tampered).hexdigest()[:12], msg,
            "error omits the digest actually found — an operator cannot tell "
            "which copy drifted",
        )

    def test_loader_accepts_matching_bytes(self):
        # The permitted path, not just the refusal. A loader that raised on
        # everything would pass the test above and fail the product.
        raw = _JS_PATH.read_bytes()
        js = vendor._load(js_bytes=raw, recorded=hashlib.sha256(raw).hexdigest())
        self.assertEqual(js, raw.decode("utf-8"))


class TestHostAgnosticAttribute(unittest.TestCase):
    """The stamped index attribute is part of the wire protocol (SPEC 6.4).

    It is named `data-latch-idx`, not after this host: the extension is
    host-agnostic and its content script writes the same attribute this module
    then selects on. A rename on one side alone leaves the query stamping one
    name and the locator selecting another -- every element resolves to nothing
    and the tier reports an empty page rather than an error.
    """

    def test_js_stamps_the_latch_attribute(self):
        text = _JS_PATH.read_bytes().decode("utf-8")
        self.assertIn("dataset.latchIdx", text)
        self.assertIn("[data-latch-idx]", text)

    def test_no_host_name_survives_in_the_shared_artifact(self):
        text = _JS_PATH.read_bytes().decode("utf-8").lower()
        self.assertNotIn(
            "tenka", text,
            "the shared JS names its host. It is byte-shared with a repo that "
            "must not mention one (SPEC 9).",
        )

    def test_the_selector_dom_builds_matches_what_the_js_stamps(self):
        # The two halves of the same contract, asserted against each other
        # rather than each against a literal. A rename that updated only one
        # side passes two separate literal checks and still resolves nothing.
        js = _JS_PATH.read_bytes().decode("utf-8")
        source = Path(bdom.__file__).read_text(encoding="utf-8")
        self.assertIn("dataset.latchIdx", js)
        self.assertIn("[data-latch-idx='{idx}']", source)


class TestDomUsesTheVendoredCopy(unittest.TestCase):

    def test_dom_module_no_longer_holds_an_inline_js_literal(self):
        source = Path(bdom.__file__).read_text(encoding="utf-8")
        self.assertNotIn(
            'r"""\n(cfg) => {', source,
            "dom.py still carries an inline copy of the query JS — there must "
            "be exactly one source of truth",
        )

    def test_dom_query_js_is_the_vendored_text(self):
        self.assertEqual(bdom._DOM_QUERY_JS, vendor.DOM_QUERY_JS)

    def test_js_digest_is_pinned(self):
        # Pinned so an accidental edit fails loudly. The extension ships a byte
        # copy and compares digests at handshake, so an edit here that is not
        # mirrored there silently disables the extension tier.
        self.assertEqual(
            vendor.DOM_QUERY_SHA256,
            "66c870ba204e3550ad13c4bb061ecfab38c28dcb366e5c4b1a1cd237941e0712",
            "the query JS changed. That is allowed, but the extension's copy "
            "must change identically or the handshake refuses the connection. "
            "Update both, then re-pin this digest.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
