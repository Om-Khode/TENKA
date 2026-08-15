"""assistant/io/api/qr.py -- SVG QR rendering for the phone-pairing flow.

The pair code is a live credential for three minutes (see
`assistant.io.api.pairing.CODE_TTL_SECONDS`), so the interesting property
here is not "does this produce a QR" but "does the payload leak out of band".
An SVG built from `<path>` fill commands carries no text nodes at all --
`test_the_payload_is_not_embedded_as_text` is the test that would catch a
regression to a factory (or a hand-rolled label) that puts the payload
back in as a `<text>` element or an XML comment.
"""
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
