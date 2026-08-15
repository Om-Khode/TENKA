"""A stand-in for a real `next build --output export`, as a zip or a directory.

Mirrors the layout Task 1 recorded against a live export, including the detail
the daemon's whole resolution order exists for: `app.html` is a *sibling* of
the `app/` directory, not `app/index.html`.

Shared rather than local to test_api_ui_serving.py because test_api_auth.py
needs a mountable bundle too -- its schema-sweep guard carries a named
exemption for the UI route, and an exemption that is never exercised against a
real mounted route is the same as no exemption at all.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from assistant.io.api.ui import MARKER_NAME, UI_MANIFEST_VERSION, UiBundle

INDEX = b"<html><head><title>TENKA</title></head><body>shell<script>1</script></body></html>"
APP = b"<html><body>app shell</body></html>"
CHAT = b"<html><body>chat page</body></html>"
SETTINGS = b"<html><body>settings page</body></html>"
CONNECT = b"<html><body>connect page</body></html>"
CHUNK = b"self.__next_f=[];console.log('chunk')"
FAVICON = b"\x00\x00\x01\x00fake-icon"

EXPORT: dict[str, bytes] = {
    "index.html": INDEX,
    "app.html": APP,
    "app/chat.html": CHAT,
    "app/settings.html": SETTINGS,
    "connect.html": CONNECT,
    "_next/static/chunks/main-deadbeef.js": CHUNK,
    "_next/static/css/site-deadbeef.css": b"body{color:#000}",
    "favicon.ico": FAVICON,
}


def marker(contract: str, *, version: int | object = UI_MANIFEST_VERSION) -> str:
    return json.dumps({"version": version, "contract": contract,
                       "builtAt": "2026-08-15T00:00:00Z"})


def write_ui_zip(path: Path, contract: str, *,
                 files: dict[str, bytes] | None = None,
                 version: int | object = UI_MANIFEST_VERSION) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MARKER_NAME, marker(contract, version=version))
        for name, body in (EXPORT if files is None else files).items():
            archive.writestr(name, body)
    return path


def write_ui_dir(root: Path, contract: str, *,
                 files: dict[str, bytes] | None = None,
                 version: int | object = UI_MANIFEST_VERSION) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / MARKER_NAME).write_text(marker(contract, version=version), encoding="utf-8")
    for name, body in (EXPORT if files is None else files).items():
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
    return root


def build_ui_bundle(tmp_path: Path, contract: str = "unused") -> UiBundle:
    """A mountable bundle for tests that only need the route to exist.

    The contract deliberately does not match any app: `mount_ui` registers the
    route either way, and a caller that cares about serving passes the real
    hash instead.
    """
    bundle = UiBundle.open(zip_path=write_ui_zip(tmp_path / "studio-ui.zip", contract),
                           dir_path=None)
    assert bundle is not None
    return bundle
