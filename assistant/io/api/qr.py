# assistant/io/api/qr.py
"""Render a pairing payload as an SVG QR code.

Task 10 builds the payload -- `https://<endpoint>/pair#<code>` -- and this
module's only job is turning that string into an image. Three choices are
load-bearing, not stylistic:

- SVG, not PNG. `qrcode`'s pure-Python SVG factories need no PIL, so one
  implementation serves the Studio dialog, the console, and the desktop
  overlay without pulling in an image codec anywhere.
- Path-based SVG, not a factory that labels modules with `<text>` or leaves
  the payload in an XML comment. The pair code is a live credential for
  `PairCode`'s TTL (see `pairing.py`), so the same rule that keeps it out of
  logs applies here: it must not be recoverable by grepping the rendered
  output, screenshotting the DOM, or viewing a saved page's source. A QR is
  a bitmap of black/white squares; `SvgPathImage` renders exactly that --
  one `<path>` of fill rectangles -- and never touches the source string
  after handing it to the encoder.
- Fit is auto-negotiated, not fixed at a version. A short local-network URL
  and a `*.trycloudflare.com` hostname plus path and fragment (over 200
  characters) both have to fit, and the QR spec's higher versions exist
  for exactly this range. Pinning a version would silently truncate or
  except on the longer payload the day the tunnel provider's hostnames grow.

Layering: io/api -- core + config only. `qrcode` is a third-party dependency,
not an assistant package, so it does not count against that boundary.
"""
from __future__ import annotations

import io

import qrcode
from qrcode.image.svg import SvgPathImage


def qr_svg(payload: str) -> str:
    """Render `payload` as a standalone `<svg>...</svg>` string.

    No PIL, no file written -- the caller decides whether the SVG is
    base64-embedded (`<img src="data:image/svg+xml;base64,...">`, Studio's
    approach) or piped straight to a renderer. It is deliberately never
    inlined into an HTML DOM by this function: inline SVG is an active
    document format that admits `<script>` and event handlers, and this
    module has no way to know whether its caller's page sanitizes that.

    Raises `ValueError` on an empty payload rather than handing back a QR
    that encodes nothing -- a scannable code with no destination is a
    silent failure, not a valid pairing artifact.
    """
    if not payload:
        raise ValueError("qr_svg requires a non-empty payload")

    # `fit=True` picks the smallest QR version that holds `payload` at this
    # error-correction level, so both a short LAN URL and a 200+ character
    # tunnel URL encode without the caller having to reason about capacity.
    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        image_factory=SvgPathImage,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    image = qr.make_image()

    buffer = io.BytesIO()
    image.save(buffer)
    return buffer.getvalue().decode("utf-8")
