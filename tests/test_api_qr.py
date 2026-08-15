"""assistant/io/api/qr.py -- SVG QR rendering for the phone-pairing flow.

The pair code is a live credential for three minutes (see
`assistant.io.api.pairing.CODE_TTL_SECONDS`), so the interesting property
here is not "does this produce a QR" but "does the payload leak out of band".
An SVG built from `<path>` fill commands carries no text nodes at all --
`test_the_payload_is_not_embedded_as_text` is the test that would catch a
regression to a factory (or a hand-rolled label) that puts the payload
back in as a `<text>` element or an XML comment.
"""
import sys

import pytest

from assistant.io.api.qr import qr_svg


def test_returns_inline_svg():
    out = qr_svg("https://example.ts.net/pair#7K2M-9QX4")
    assert out.startswith("<svg") or out.startswith("<?xml")
    assert "</svg>" in out


def test_the_payload_is_not_embedded_as_text():
    """A QR is an image, not a label. The code must not be readable by
    grepping the SVG, or a screenshot of the DOM leaks it."""
    out = qr_svg("https://example.ts.net/pair#7K2M-9QX4")
    assert "7K2M-9QX4" not in out


def test_long_payloads_still_encode():
    long_url = "https://" + "a" * 200 + ".trycloudflare.com/pair#7K2M-9QX4"
    assert "</svg>" in qr_svg(long_url)


def test_empty_payload_raises():
    with pytest.raises(ValueError):
        qr_svg("")


def test_qr_svg_works_even_when_pillow_is_unimportable():
    """`qr_svg`'s own code path never touches pillow -- but `qrcode`'s
    compatibility shim (`qrcode/image/styles/moduledrawers/__init__.py`)
    imports PIL drawers in a bare `try/except ImportError`, so on an install
    where pillow happens to be present (as it is here, pulled in
    transitively by `easyocr`/`face_recognition`/`torchvision`), merely
    importing `qrcode.image.svg` loads PIL into the process as a side
    effect. `pip show qrcode` listing only `colorama` proves the
    *dependency graph* doesn't need pillow; it says nothing about the
    *runtime*. This test proves the runtime property instead: block every
    `PIL` import, force `qrcode` and this module to import fresh under that
    block, and confirm `qr_svg` still renders a valid SVG without PIL ever
    landing in `sys.modules`.
    """

    class _BlockPIL:
        def find_spec(self, name, path, target=None):
            if name == "PIL" or name.startswith("PIL."):
                raise ImportError(f"blocked for test: {name}")
            return None

    stale = [
        name
        for name in sys.modules
        if name == "PIL"
        or name.startswith("PIL.")
        or name == "qrcode"
        or name.startswith("qrcode.")
        or name == "assistant.io.api.qr"
    ]
    saved = {name: sys.modules.pop(name) for name in stale}
    blocker = _BlockPIL()
    sys.meta_path.insert(0, blocker)
    try:
        from assistant.io.api.qr import qr_svg as fresh_qr_svg

        assert "PIL" not in sys.modules, "qrcode.image.svg pulled in PIL despite the block"
        out = fresh_qr_svg("https://example.ts.net/pair#7K2M-9QX4")
        assert "</svg>" in out
    finally:
        # Restore real state so later tests in this process see the normal,
        # pillow-present modules rather than this test's blocked stand-ins.
        sys.meta_path.remove(blocker)
        for name in list(sys.modules):
            if (
                name == "PIL"
                or name.startswith("PIL.")
                or name == "qrcode"
                or name.startswith("qrcode.")
                or name == "assistant.io.api.qr"
            ):
                del sys.modules[name]
        sys.modules.update(saved)
