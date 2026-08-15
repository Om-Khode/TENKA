"""The packaging step that turns Studio's export into the zip TENKA ships.

Everything here is about one asymmetry: the packager runs once, by hand, on a
release, and its output is then trusted by a daemon on somebody else's machine
that has no way to check any of it. So each thing the daemon silently tolerates
is something this has to refuse loudly.

The two that would not be found by looking:

* **A bundle under the wrong marker name is not a broken bundle -- it is no
  bundle at all.** `UiBundle._from_zip` logs one WARNING, returns `None`, and
  the daemon comes up with a perfectly healthy API and no `/` route. Nothing
  crashes and nothing says why.

* **The API base is frozen into the JS at build time.** A bundle carrying
  Studio's absolute loopback default works on `next dev` and fails completely
  the first time it is served from a tunnel, because every call becomes
  cross-origin and the daemon's own `connect-src 'self'` refuses it. It reads
  as a broken WebSocket and is not one, so the base is asserted here from the
  shipped bytes rather than taken on faith from whoever ran the build.
"""
from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import pytest

from assistant.io.api.ui import (
    MARKER_NAME,
    MAX_MEMBER_BYTES,
    UI_MANIFEST_VERSION,
    UiBundle,
    contract_hash,
)
from assistant.io.api.vault import TokenVault
from tests.fakes.api_client import build_api_client
from tests.fakes.studio_runtime import build_fake_runtime
from tools.package_studio_ui import package

# The shape `apiBase()` minifies to. The packager reads the base out of this,
# so a fake export that omits it is -- correctly -- refused.
RELATIVE_BASE_CHUNK = b'function i(){return"/".replace(/\\/+$/,"")}'
ABSOLUTE_BASE_CHUNK = b'function i(){return"https://tunnel.example".replace(/\\/+$/,"")}'
LOOPBACK_BASE_CHUNK = b'function i(){return"http://127.0.0.1:8787".replace(/\\/+$/,"")}'


def _fake_export(tmp_path: Path, *, chunk: bytes = RELATIVE_BASE_CHUNK) -> Path:
    """A miniature `next build --output export`, in the shape Task 1 recorded:
    `app/chat.html` is a sibling of `app/`, not `app/chat/index.html`."""
    out = tmp_path / "out"
    (out / "app").mkdir(parents=True)
    (out / "_next" / "static" / "chunks").mkdir(parents=True)
    (out / "index.html").write_bytes(b"<html><body>shell</body></html>")
    (out / "app.html").write_bytes(b"<html><body>app</body></html>")
    (out / "app" / "chat.html").write_bytes(b"<html><body>chat</body></html>")
    (out / "_next" / "static" / "chunks" / "main-deadbeef.js").write_bytes(chunk)
    (out / "favicon.ico").write_bytes(b"\x00\x00\x01\x00icon")
    return out


def _reference_contract() -> str:
    from assistant.io.api.app import create_app
    return contract_hash(create_app(build_fake_runtime(),
                                    TokenVault(Path(tempfile.mkdtemp())),
                                    origins=["http://localhost:3000"]))


def _marker(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as archive:
        return json.loads(archive.read(MARKER_NAME))


# ─── the brief's contract ────────────────────────────────────────────────
def test_the_manifest_carries_the_live_contract_hash(tmp_path):
    out = _fake_export(tmp_path)
    zip_path = package(out, tmp_path / "ui.zip")
    manifest = _marker(zip_path)
    assert manifest["version"] == UI_MANIFEST_VERSION
    assert manifest["contract"] == _reference_contract()
    assert manifest["builtAt"]


def test_packaging_refuses_an_export_without_index(tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(ValueError):
        package(tmp_path / "out", tmp_path / "ui.zip")


def test_the_zip_contains_no_secrets(tmp_path):
    out = _fake_export(tmp_path)
    (out / "leak.js").write_text('const t="Bearer abc";', encoding="utf-8")
    with pytest.raises(ValueError):
        package(out, tmp_path / "ui.zip")


def test_packaging_is_reproducible_given_a_fixed_timestamp(tmp_path):
    out = _fake_export(tmp_path)
    a = package(out, tmp_path / "a.zip", built_at="2026-08-15T00:00:00+00:00",
                contract="c").read_bytes()
    b = package(out, tmp_path / "b.zip", built_at="2026-08-15T00:00:00+00:00",
                contract="c").read_bytes()
    assert a == b


# ─── the marker, which is the difference between a bundle and no bundle ──
def test_the_marker_is_the_name_the_daemon_reads(tmp_path):
    """Spelling it `manifest.json` would not produce a broken bundle. It would
    produce a daemon with no `/` route, one WARNING deep."""
    zip_path = package(_fake_export(tmp_path), tmp_path / "ui.zip", contract="c")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    assert MARKER_NAME in names
    assert "manifest.json" not in names


def test_the_daemon_opens_what_the_packager_wrote(tmp_path):
    """The whole point, end to end: the artefact this produces is one
    `UiBundle` accepts and serves, not merely one it can be made to parse."""
    zip_path = package(_fake_export(tmp_path), tmp_path / "ui.zip")
    bundle = UiBundle.open(zip_path=zip_path, dir_path=None)
    assert bundle is not None
    client = build_api_client(build_fake_runtime(),
                              TokenVault(Path(tempfile.mkdtemp())),
                              ui_bundle=bundle)
    assert client.get("/").status_code == 200
    assert client.get("/app/chat").content == b"<html><body>chat</body></html>"
    # ...and the marker it just wrote is still not reachable over HTTP.
    assert client.get(f"/{MARKER_NAME}").status_code == 404


def test_a_marker_the_daemon_would_reject_is_never_written(tmp_path):
    zip_path = package(_fake_export(tmp_path), tmp_path / "ui.zip", contract="c")
    version = _marker(zip_path)["version"]
    assert isinstance(version, int) and not isinstance(version, bool)


# ─── the API base ────────────────────────────────────────────────────────
def test_an_absolute_api_base_is_refused(tmp_path):
    """Not only the loopback default. Any absolute origin is cross-origin once
    the daemon serves the page from a tunnel, and `connect-src 'self'` refuses
    it before mixed content or Private Network Access are consulted."""
    out = _fake_export(tmp_path, chunk=ABSOLUTE_BASE_CHUNK)
    with pytest.raises(ValueError, match="tunnel.example"):
        package(out, tmp_path / "ui.zip", contract="c")


def test_studios_loopback_default_is_refused(tmp_path):
    out = _fake_export(tmp_path, chunk=LOOPBACK_BASE_CHUNK)
    with pytest.raises(ValueError):
        package(out, tmp_path / "ui.zip", contract="c")


def test_a_loopback_origin_anywhere_in_the_bundle_is_refused(tmp_path):
    """Studio's `apiBase()` is written so the unused default folds away
    entirely. A build where it survived is a build whose base check cannot be
    trusted, wherever the string turned up."""
    out = _fake_export(tmp_path)
    (out / "stray.js").write_bytes(b'const dev="http://localhost:8787";')
    with pytest.raises(ValueError):
        package(out, tmp_path / "ui.zip", contract="c")


def test_an_export_with_no_recognisable_base_is_refused(tmp_path):
    """Fails closed. If the minified shape changes and this check silently
    matches nothing, it becomes a check that can never fail -- the same trap as
    running `git grep` over a gitignored directory."""
    out = _fake_export(tmp_path, chunk=b"function i(){return window.__base}")
    with pytest.raises(ValueError, match="no API base literal"):
        package(out, tmp_path / "ui.zip", contract="c")


def test_legitimate_code_that_merely_mentions_bearer_still_packages(tmp_path):
    """Studio really does build an `Authorization: Bearer` header for the dev
    path, and it survives minification as a template literal. A rule that
    refused the word would refuse every export ever produced, and a check that
    always fails gets deleted rather than fixed."""
    out = _fake_export(tmp_path)
    (out / "auth.js").write_bytes(b'h.set("Authorization",`Bearer ${t}`);'
                                  b'g.set("Authorization","Bearer ".concat(t));')
    assert package(out, tmp_path / "ui.zip", contract="c").exists()


# ─── members the daemon would refuse to serve ────────────────────────────
def test_a_member_the_daemon_cannot_name_is_refused(tmp_path):
    """`UiBundle` drops a member whose stored name fails `normalise_member`,
    quietly. Refusing here turns "this asset 404s in production" into
    "packaging failed", which is the same fact where somebody is looking."""
    out = _fake_export(tmp_path)
    # Depth, rather than one of the name shapes -- Windows silently rewrites
    # most of those on creation (`trailing.js.` becomes `trailing.js`), so a
    # test built on one would pass here for the wrong reason.
    deep = out.joinpath(*("a" for _ in range(25)))
    deep.mkdir(parents=True)
    (deep / "buried.js").write_bytes(b"x")
    with pytest.raises(ValueError, match="serve"):
        package(out, tmp_path / "ui.zip", contract="c")


def test_a_member_the_daemon_would_refuse_for_size_is_refused(tmp_path):
    out = _fake_export(tmp_path)
    (out / "huge.bin").write_bytes(b"\0" * (MAX_MEMBER_BYTES + 1))
    with pytest.raises(ValueError, match="bytes"):
        package(out, tmp_path / "ui.zip", contract="c")


def test_every_member_is_deflated(tmp_path):
    """A stored bundle would roughly triple what git carries forever."""
    zip_path = package(_fake_export(tmp_path), tmp_path / "ui.zip", contract="c")
    with zipfile.ZipFile(zip_path) as archive:
        assert all(info.compress_type == zipfile.ZIP_DEFLATED
                   for info in archive.infolist())


def test_nested_directories_keep_their_posix_names(tmp_path):
    """Windows separators in a stored name would make `app/chat.html`
    unreachable on the very machine that built the bundle."""
    zip_path = package(_fake_export(tmp_path), tmp_path / "ui.zip", contract="c")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    assert "app/chat.html" in names
    assert "_next/static/chunks/main-deadbeef.js" in names
    assert not any("\\" in name for name in names)


# ─── the artefact that actually ships ────────────────────────────────────
# A fixture proves the packager. It does not prove the bundle: a fake export is
# whatever this file says it is, while the vendored zip was produced by a real
# `next build` and is the thing a user's daemon will open. Skipped rather than
# failed when absent, so a fresh clone that has not run the packager is not
# red -- but present in CI and on any checkout that has, which is every one
# that matters.
VENDORED = Path(__file__).resolve().parents[1] / "assistant" / "io" / "api" / "studio_ui.zip"
vendored_only = pytest.mark.skipif(not VENDORED.is_file(),
                                   reason="no vendored bundle in this checkout")


@vendored_only
def test_the_vendored_bundle_opens_and_serves():
    bundle = UiBundle.open(zip_path=VENDORED, dir_path=None)
    assert bundle is not None, "the vendored zip is not one the daemon accepts"
    client = build_api_client(build_fake_runtime(),
                              TokenVault(Path(tempfile.mkdtemp())),
                              ui_bundle=bundle)
    # 200 rather than 503 is also the contract assertion: a stale marker takes
    # the whole bundle dark, so a served page means the hash agrees.
    for path in ("/", "/app", "/app/chat", "/connect"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert b"<html" in response.content.lower(), path
    assert client.get("/v1/status").status_code == 401
    assert client.get(f"/{MARKER_NAME}").status_code == 404


@vendored_only
def test_the_vendored_bundle_was_built_for_this_api():
    bundle = UiBundle.open(zip_path=VENDORED, dir_path=None)
    assert bundle.manifest()["contract"] == _reference_contract(), (
        "the vendored bundle is stale -- rebuild Studio with "
        "`npm run build:bundled` and re-run tools/package_studio_ui.py")


@vendored_only
def test_the_vendored_bundle_carries_a_real_export_not_a_placeholder():
    """A one-file zip would pass every other test here."""
    with zipfile.ZipFile(VENDORED) as archive:
        names = archive.namelist()
    assert len(names) > 100
    assert any(n.startswith("_next/static/chunks/") and n.endswith(".js")
               for n in names)
    assert "app/chat.html" in names        # the sibling shape, not app/chat/index.html


def test_the_contract_hash_matches_the_daemons_own(tmp_path):
    """Both sides hash the schema object under the same canonical
    serialisation. Hashing the bytes of a schema *file* on either side would
    turn the staleness guard into a whitespace detector."""
    from tools.package_studio_ui import live_contract_hash
    assert live_contract_hash() == _reference_contract()
