"""The daemon serves Studio itself, and that route is deliberately public.

Two things make this file worth more than its line count.

**It is one of only two unauthenticated surfaces in the API.** The page has to
load before pairing exists, so it cannot demand a credential -- and from
Milestone 6b it is reachable from a public tunnel URL. Every escape shape a
path can take is therefore a live attack, not a hypothetical one.

**A fallback-only server would pass a casual look.** Task 1 established that
the export writes `app/chat.html`, not `app/chat/index.html`, and that
Studio's client router repairs the URL after load. So a server that answered
`index.html` for everything would *appear* to work while silently serving the
wrong document first. `test_a_route_is_served_from_its_own_prerendered_html`
is the half of the contract that catches that.
"""
from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

import pytest

from assistant.io.api.ui import (
    MARKER_NAME,
    UI_MANIFEST_VERSION,
    UiBundle,
    contract_hash,
    normalise_member,
)
from assistant.io.api.vault import TokenVault
from tests.fakes.api_client import ApiTestClient, build_api_client
from tests.fakes.studio_runtime import build_fake_runtime
from tests.fakes.studio_ui import CHAT, EXPORT, INDEX, write_ui_dir, write_ui_zip


def _reference_contract() -> str:
    """The contract hash of a daemon built exactly the way `_client` builds it.

    Computed off a throwaway app rather than hardcoded, because the hash is a
    function of the whole OpenAPI schema and every later task changes it.
    """
    return contract_hash(_app(ui_bundle=None))


def _write_zip(path: Path, files: dict[str, bytes], *,
               version: int | object = UI_MANIFEST_VERSION,
               contract: str | None = None) -> Path:
    return write_ui_zip(path,
                        _reference_contract() if contract is None else contract,
                        files=files, version=version)


def _member(zip_path: Path, name: str) -> bytes:
    with zipfile.ZipFile(zip_path) as archive:
        return archive.read(name)


@pytest.fixture()
def ui_zip(tmp_path) -> Path:
    return _write_zip(tmp_path / "studio-ui.zip", EXPORT)


@pytest.fixture()
def ui_dir(tmp_path) -> Path:
    """The dev path, built so it actually serves *through the route*.

    Every dev-directory test in the first round called `bundle.read()`
    directly against a `"contract": "any"` marker, which would have answered
    503 had it ever reached the route. That gap is the root cause of the
    case-sensitive deny-list surviving review -- not carelessness about case,
    but a whole mode of this module never once exercised over HTTP.
    """
    return write_ui_dir(tmp_path / "out", _reference_contract())


@pytest.fixture()
def ui_zip_wrong_contract(tmp_path) -> Path:
    return _write_zip(tmp_path / "stale-ui.zip", EXPORT, contract="0" * 64)


@pytest.fixture()
def ui_zip_v0(tmp_path) -> Path:
    return _write_zip(tmp_path / "old-ui.zip", EXPORT, version=0)


def _app(*, ui_bundle):
    from assistant.io.api.app import create_app
    return create_app(build_fake_runtime(), TokenVault(Path(tempfile.mkdtemp())),
                      origins=["http://localhost:3000"], ui_bundle=ui_bundle)


def _client(*, ui_bundle) -> ApiTestClient:
    return build_api_client(build_fake_runtime(),
                            TokenVault(Path(tempfile.mkdtemp())),
                            ui_bundle=ui_bundle)


# ─── the brief's contract ────────────────────────────────────────────────
def test_index_is_served_without_authentication(ui_zip):
    client = _client(ui_bundle=UiBundle.open(zip_path=ui_zip, dir_path=None))
    r = client.get("/")
    assert r.status_code == 200 and b"<html" in r.content.lower()


def test_a_traversal_member_is_refused(ui_zip):
    """Reachable from a public URL in 6b. Classic hole."""
    client = _client(ui_bundle=UiBundle.open(zip_path=ui_zip, dir_path=None))
    for path in ("/../instance_secret", "/..%2Finstance_secret",
                 "/_next/../../devices.json", "//etc/passwd"):
        assert client.get(path).status_code in (403, 404), path


def test_api_routes_are_not_shadowed(ui_zip):
    client = _client(ui_bundle=UiBundle.open(zip_path=ui_zip, dir_path=None))
    assert client.get("/v1/status").status_code == 401     # auth, not the SPA


def test_a_route_is_served_from_its_own_prerendered_html(ui_zip):
    """Task 1 established the export's shape: `/app/chat` lands on disk as
    `app/chat.html`, not `app/chat/index.html` and not only as a client-side
    route. Resolution order is therefore `<path>` exactly, then `<path>.html`,
    then `<path>/index.html`, and only then the index fallback. Falling
    straight back to index.html would 'work' -- the client router would repair
    the URL -- while silently throwing away every prerendered page and making
    a deep link flash the wrong document first."""
    client = _client(ui_bundle=UiBundle.open(zip_path=ui_zip, dir_path=None))
    body = client.get("/app/chat").content
    assert body == _member(ui_zip, "app/chat.html")


def test_an_unknown_path_falls_back_to_index(ui_zip):
    client = _client(ui_bundle=UiBundle.open(zip_path=ui_zip, dir_path=None))
    r = client.get("/app/no-such-page")
    assert r.status_code == 200
    assert r.content == _member(ui_zip, "index.html")


def test_a_contract_mismatch_refuses_to_serve(ui_zip_wrong_contract):
    bundle = UiBundle.open(zip_path=ui_zip_wrong_contract, dir_path=None)
    client = _client(ui_bundle=bundle)
    r = client.get("/")
    assert r.status_code == 503
    assert "stale" in r.text.lower()


def test_an_older_manifest_version_is_rejected(ui_zip_v0):
    assert UiBundle.open(zip_path=ui_zip_v0, dir_path=None) is None


def test_the_dev_path_wins_over_the_zip(tmp_path, ui_zip):
    # The brief spelled the marker `manifest.json`. It is `.tenka-ui.json`
    # now: `manifest.json` is a name a Next app may legitimately ship as a PWA
    # web app manifest, and one name for two different files is a collision
    # waiting to be found in production rather than in review.
    (tmp_path / "index.html").write_text("<html>dev</html>", encoding="utf-8")
    (tmp_path / MARKER_NAME).write_text(
        json.dumps({"version": 1, "contract": "any", "builtAt": "now"}), encoding="utf-8")
    bundle = UiBundle.open(zip_path=ui_zip, dir_path=tmp_path)
    assert b"dev" in bundle.read("index.html")[0]


def test_security_headers_are_present(ui_zip):
    client = _client(ui_bundle=UiBundle.open(zip_path=ui_zip, dir_path=None))
    h = client.get("/").headers
    assert "default-src 'self'" in h["content-security-policy"]
    assert h["x-content-type-options"] == "nosniff"
    assert h["referrer-policy"] == "no-referrer"
    assert "no-store" in h["cache-control"]


def test_no_bundle_means_no_ui_route_and_a_working_api(ui_zip):
    client = _client(ui_bundle=None)
    assert client.get("/").status_code == 404
    assert client.get("/v1/status").status_code == 401


# ─── traversal, past the brief's floor ───────────────────────────────────
# The brief's four inputs are a floor for a reason that only shows up when you
# run them: `httpx` normalises dot segments on the client, so three of the four
# never arrive as traversal at all -- `/../instance_secret` reaches the daemon
# as the perfectly ordinary path `/instance_secret`. That is why the SPA
# fallback here is not a blanket one (see `test_an_unknown_top_level_name_is_
# not_the_spa_shell`), and it is why the normaliser is also exercised directly
# below: a unit test is the only place an encoding the client rewrites can be
# put in front of the code that has to refuse it.

_ESCAPES = [
    "..",
    "../instance_secret",
    "../../devices.json",
    "a/../../b",
    "..\\instance_secret",                  # Windows separator
    "a\\..\\..\\b",
    "/etc/passwd",                          # absolute, posix
    "//etc/passwd",                         # network-path form
    "\\\\server\\share\\secret",            # UNC
    "//server/share/secret",                # UNC, folded
    "C:/Windows/win.ini",                   # drive letter
    "C:\\Windows\\win.ini",
    "c:x.html",                             # drive-relative
    "app/../../devices.json",
    "app/./../../devices.json",
    ".",
    "./devices.json",
    "index.html\x00.js",                    # NUL truncation
    "index.html.",                          # Windows strips a trailing dot
    "index.html ",                          # ...and trailing space
    " index.html",
    "a//b",                                 # empty segment
    "\x00",
    "app/chat.html:stream",                 # NTFS alternate data stream
]


@pytest.mark.parametrize("raw", _ESCAPES)
def test_the_member_normaliser_refuses_every_escape_shape(raw):
    assert normalise_member(raw) is None, raw


@pytest.mark.parametrize("raw,expected", [
    ("", ""),
    ("/", ""),
    ("index.html", "index.html"),
    ("app/chat", "app/chat"),
    ("app/", "app"),
    ("_next/static/chunks/main-deadbeef.js", "_next/static/chunks/main-deadbeef.js"),
])
def test_the_member_normaliser_keeps_a_legitimate_path(raw, expected):
    assert normalise_member(raw) == expected


def test_a_traversal_survives_the_client_only_when_encoded(ui_zip):
    """The encodings `httpx` does *not* rewrite, sent over the wire for real."""
    client = _client(ui_bundle=UiBundle.open(zip_path=ui_zip, dir_path=None))
    for path in ("/..%2Finstance_secret",
                 "/%2e%2e/instance_secret",
                 "/..%5Cinstance_secret",
                 "/app%2F..%2F..%2Fdevices.json",
                 "/%2e%2e%2f%2e%2e%2fdevices.json",
                 "/..%252Fdevices.json",          # double-encoded
                 "/app/chat.html%00.js",
                 "/C:%2FWindows%2Fwin.ini"):
        assert client.get(path).status_code in (403, 404), path


def test_an_unknown_top_level_name_is_not_the_spa_shell(ui_zip):
    """The SPA fallback is bounded by the export's own top-level names.

    A blanket "anything unknown gets index.html" answers 200 to
    `/instance_secret` and `/devices.json` -- which is exactly what the brief's
    traversal inputs decay into once the client has normalised them. Bounding
    the fallback to names the bundle actually contains is what makes those
    refusals real instead of accidental, and it is also plain good behaviour:
    a name that was never a route should 404, not hand back an HTML shell.
    """
    client = _client(ui_bundle=UiBundle.open(zip_path=ui_zip, dir_path=None))
    for path in ("/instance_secret", "/devices.json", "/passwd", "/etc/passwd"):
        assert client.get(path).status_code == 404, path
    # ...while a client-side route under a real top-level name still falls back
    assert client.get("/app/deep/unknown").status_code == 200


def test_a_missing_asset_never_becomes_html(ui_zip):
    """An extension-bearing miss is a broken asset, not a route. Answering it
    with `index.html` at 200 is how a missing chunk turns into a syntax error
    in the console instead of a 404 anyone can read."""
    client = _client(ui_bundle=UiBundle.open(zip_path=ui_zip, dir_path=None))
    for path in ("/_next/static/chunks/nope.js", "/app/gone.css", "/favicon.png"):
        r = client.get(path)
        assert r.status_code == 404, path


def test_a_zip_entry_that_escapes_the_root_is_dropped(tmp_path):
    """Zip-slip, from the archive side. Nothing here extracts to disk, so the
    classic write primitive does not exist -- but an entry stored as
    `../devices.json` would still be *readable* by name if the loader trusted
    the names it was given."""
    path = _write_zip(tmp_path / "evil.zip",
                      {**EXPORT,
                       "../devices.json": b"stolen",
                       "..\\creds.json": b"stolen",
                       "/etc/passwd": b"stolen"})
    bundle = UiBundle.open(zip_path=path, dir_path=None)
    assert bundle is not None
    for name in ("../devices.json", "../creds.json", "/etc/passwd",
                 "devices.json", "creds.json"):
        assert bundle.read(name) is None, name


def test_a_dev_directory_cannot_be_escaped(tmp_path, ui_dir):
    (tmp_path / "devices.json").write_text("stolen", encoding="utf-8")
    bundle = UiBundle.open(zip_path=None, dir_path=ui_dir)
    assert bundle is not None
    assert bundle.read("index.html") is not None
    for name in ("../devices.json", "..\\devices.json", "/etc/passwd"):
        assert bundle.read(name) is None, name


def test_a_windows_device_name_is_refused_by_the_directory_reader(ui_dir):
    """The reserved-name rule lives here now, not in `normalise_member`.

    It is a property of the Win32 file namespace -- `out/AUX.woff2` resolves to
    a serial port, not a file -- so it belongs to the one path that actually
    opens a file by name. On the zip path nothing is opened by name, where the
    same rule can only produce false negatives; see the companion test below.
    """
    bundle = UiBundle.open(zip_path=None, dir_path=ui_dir)
    for name in ("CON", "NUL.html", "com1/x.html", "app/aux.a1b2c3.woff2"):
        assert bundle.read(name) is None, name


def test_a_source_basename_that_looks_like_a_device_still_serves_from_a_zip(tmp_path):
    """Next preserves source basenames in hashed asset names, so a font
    vendored as `aux.woff` exports as `_next/static/media/aux.a1b2c3.woff2`.
    Refusing that everywhere would make a real asset an unfixable 404 for a
    Windows rule that cannot apply to a zip member at all."""
    asset = "_next/static/media/aux.a1b2c3.woff2"
    path = _write_zip(tmp_path / "fonts.zip", {**EXPORT, asset: b"font-bytes"})
    client = _client(ui_bundle=UiBundle.open(zip_path=path, dir_path=None))
    r = client.get(f"/{asset}")
    assert r.status_code == 200
    assert r.content == b"font-bytes"
    assert r.headers["content-type"] == "font/woff2"


def test_a_symlink_out_of_the_dev_directory_is_refused(tmp_path, ui_dir):
    """Name-level normalisation cannot see a symlink; only resolving the real
    path can. Skipped where Windows refuses to create one without privileges."""
    root = ui_dir
    secret = tmp_path / "devices.json"
    secret.write_text("stolen", encoding="utf-8")
    try:
        (root / "leak.json").symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("this machine will not create symlinks")
    bundle = UiBundle.open(zip_path=None, dir_path=root)
    assert bundle.read("leak.json") is None


def test_a_directory_is_not_a_member(ui_zip):
    """`/_next` names a directory in the archive. It has no body to serve, and
    it must not become a listing or a fallback shell either."""
    client = _client(ui_bundle=UiBundle.open(zip_path=ui_zip, dir_path=None))
    assert client.get("/_next").status_code == 404
    assert client.get("/_next/static").status_code == 404


# ─── the dev directory, exercised over HTTP and not only through read() ──
def test_the_dev_directory_serves_through_the_route(ui_dir):
    """The gap that let a case-sensitive deny-list through review. Every dev
    assertion in round one went through `bundle.read()`, so the whole
    directory-backed half of this module had zero route coverage."""
    client = _client(ui_bundle=UiBundle.open(zip_path=None, dir_path=ui_dir))
    assert client.get("/").content == INDEX
    assert client.get("/app/chat").content == CHAT          # own prerendered doc
    assert client.get("/app/deep/unknown").content == INDEX  # bounded fallback
    assert client.get("/instance_secret").status_code == 404
    assert client.get("/v1/status").status_code == 401


@pytest.mark.parametrize("spelling", [
    ".tenka-ui.json",
    ".TENKA-UI.JSON",
    ".Tenka-Ui.json",
    ".tenka-ui.JSON",
    ".TENKA-ui.Json",
])
def test_the_build_marker_is_unreachable_in_any_casing(ui_dir, ui_zip, spelling):
    """Windows filesystems are case-insensitive, so an exact-string deny-list
    is defeated by pressing shift -- `/MANIFEST.JSON` served the marker while
    `/manifest.json` 404'd. The comparison is case-folded now, and because
    this is the module's only deny-list, whatever is added to it later
    inherits that rather than the hole."""
    for bundle in (UiBundle.open(zip_path=None, dir_path=ui_dir),
                   UiBundle.open(zip_path=ui_zip, dir_path=None)):
        client = _client(ui_bundle=bundle)
        response = client.get(f"/{spelling}")
        assert response.status_code == 404, spelling
        assert "contract" not in response.text


def test_a_second_marker_entry_under_another_case_is_still_unreachable(tmp_path):
    """A zip is case-*sensitive*, so an archive can carry both spellings. The
    deny-list has to refuse the copy as well as the original."""
    path = _write_zip(tmp_path / "twin.zip",
                      {**EXPORT, ".TENKA-UI.JSON": b'{"contract":"leaked"}'})
    client = _client(ui_bundle=UiBundle.open(zip_path=path, dir_path=None))
    assert client.get("/.TENKA-UI.JSON").status_code == 404
    assert client.get("/.tenka-ui.json").status_code == 404


# ─── the route sits where the other gates can still see it ───────────────
def test_the_host_gate_still_covers_the_ui_route(ui_zip):
    """DNS rebinding does not stop being a risk because a route is public.
    The UI route is mounted inside `HostGate`, not around it."""
    client = _client(ui_bundle=UiBundle.open(zip_path=ui_zip, dir_path=None))
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 421


def test_an_unrouted_v1_path_stays_api_shaped(ui_zip):
    """The catch-all is registered after the routers, so it also sees `/v1`
    paths no router claimed. Those are API misses and must answer like one --
    a JSON 404, never a 200 HTML shell that a fetch() would then try to parse."""
    client = _client(ui_bundle=UiBundle.open(zip_path=ui_zip, dir_path=None))
    r = client.get("/v1/no-such-route")
    assert r.status_code == 404
    assert r.json() == {"error": "not found"}


# ─── content types and caching ───────────────────────────────────────────
def test_content_types_do_not_depend_on_the_machine(ui_zip):
    """`mimetypes` reads the Windows registry, where `.js` has historically
    resolved to `text/plain` -- which a browser refuses to execute as a module.
    The daemon ships to other people's machines, so the web types it actually
    serves are pinned in code rather than looked up."""
    client = _client(ui_bundle=UiBundle.open(zip_path=ui_zip, dir_path=None))
    assert client.get("/").headers["content-type"] == "text/html; charset=utf-8"
    assert client.get("/_next/static/chunks/main-deadbeef.js").headers[
        "content-type"] == "text/javascript; charset=utf-8"
    assert client.get("/_next/static/css/site-deadbeef.css").headers[
        "content-type"] == "text/css; charset=utf-8"
    assert client.get("/favicon.ico").headers["content-type"] == "image/x-icon"


def test_a_document_is_never_cached_but_a_hashed_asset_is(ui_zip):
    """`_next/static` URLs carry a content hash, so they are immutable by
    construction; a phone on a tunnel re-downloading megabytes of chunks per
    reload is the whole reason to say so. A document must stay `no-store`: it
    is the one file whose URL does not change when the build does."""
    client = _client(ui_bundle=UiBundle.open(zip_path=ui_zip, dir_path=None))
    assert "no-store" in client.get("/app/chat").headers["cache-control"]
    chunk = client.get("/_next/static/chunks/main-deadbeef.js").headers["cache-control"]
    assert "immutable" in chunk and "max-age=31536000" in chunk


def test_every_ui_response_carries_the_headers(ui_zip):
    """Not only the document: every asset comes from the same public origin,
    and `nosniff` on a chunk is what stops a browser second-guessing its type."""
    client = _client(ui_bundle=UiBundle.open(zip_path=ui_zip, dir_path=None))
    for path in ("/app/chat", "/_next/static/chunks/main-deadbeef.js"):
        h = client.get(path).headers
        assert h["x-content-type-options"] == "nosniff"
        assert h["referrer-policy"] == "no-referrer"
        assert h["x-frame-options"] == "DENY"
        assert "frame-ancestors 'none'" in h["content-security-policy"]
        assert "object-src 'none'" in h["content-security-policy"]


def test_an_error_response_carries_the_headers_too(ui_zip):
    """The response an unauthenticated attacker can most reliably elicit is a
    404 or a 403, so it is the last one that should arrive without a CSP, a
    framing rule and `nosniff`."""
    client = _client(ui_bundle=UiBundle.open(zip_path=ui_zip, dir_path=None))
    for path, expected in (("/instance_secret", 404),
                           ("/..%2Finstance_secret", 403),
                           ("/v1/no-such-route", 404),
                           (f"/{MARKER_NAME}", 404)):
        r = client.get(path)
        assert r.status_code == expected, path
        assert r.headers["x-content-type-options"] == "nosniff", path
        assert r.headers["referrer-policy"] == "no-referrer", path
        assert r.headers["x-frame-options"] == "DENY", path
        assert "default-src 'self'" in r.headers["content-security-policy"], path
        assert "no-store" in r.headers["cache-control"], path


# ─── methods ─────────────────────────────────────────────────────────────
def test_head_is_answered_and_not_405(ui_zip):
    """FastAPI's `APIRoute` does not add HEAD to a GET the way Starlette's
    plain `Route` does. In 6b the pairing URL is pasted into chat apps, and a
    link unfurler, an uptime probe or a tunnel health check sends HEAD first --
    a 405 there reads as 'the daemon is down'."""
    client = _client(ui_bundle=UiBundle.open(zip_path=ui_zip, dir_path=None))
    for path in ("/", "/app/chat", "/_next/static/chunks/main-deadbeef.js"):
        r = client.head(path)
        assert r.status_code == 200, path
        assert r.headers["x-content-type-options"] == "nosniff", path
    assert client.head("/instance_secret").status_code == 404


def test_a_write_verb_is_left_to_the_framework(ui_zip):
    """The route claims `GET` and `HEAD` and nothing else, so the 405 comes
    from Starlette's router rather than from this handler.

    An earlier revision claimed every verb so that the 405 could carry the
    security headers. Wrong trade: the headers protect nothing on a fixed body
    that reflects no input and renders nothing, while a catch-all owning every
    verb silently swallows any route registered after `create_app()` returns.
    Least privilege applies to routing as much as to capabilities, so this
    test pins the *narrowness* -- the bare 405 is the intended outcome, not a
    regression in header coverage."""
    client = _client(ui_bundle=UiBundle.open(zip_path=ui_zip, dir_path=None))
    for verb in ("POST", "PUT", "PATCH", "DELETE"):
        r = client.request(verb, "/app/chat")
        assert r.status_code == 405, verb
        assert "content-security-policy" not in r.headers, verb


def test_a_route_registered_after_the_ui_is_still_reachable(ui_zip):
    """The concrete cost of the reverted decision, pinned so nobody re-widens
    it. A non-GET route registered after `create_app()` returns -- which is
    what a later task adding a router does -- must not be swallowed by the
    catch-all. While the catch-all owned every verb, this answered 405."""
    from fastapi import APIRouter

    client = _client(ui_bundle=UiBundle.open(zip_path=ui_zip, dir_path=None))
    later = APIRouter()

    @later.post("/hook")
    async def hook_handler():
        return {"ok": True}

    client.app.include_router(later)
    assert client.post("/hook").status_code == 200


# ─── the contract hash itself ────────────────────────────────────────────
def test_the_contract_hash_is_over_a_canonical_serialisation():
    """Task 16 computes this same hash on the packaging side. Hashing the file
    bytes of a schema dump instead would make the guard a whitespace
    detector -- two identical APIs serialised differently would refuse to
    serve each other."""
    import hashlib

    app = _app(ui_bundle=None)
    schema = app.openapi()
    expected = hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert contract_hash(app) == expected

    # Same schema, different serialisation: the hash must not move.
    reordered = json.loads(json.dumps(schema, indent=4, sort_keys=False))
    assert hashlib.sha256(
        json.dumps(reordered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest() == expected


def test_two_daemons_of_the_same_build_agree_on_the_contract():
    assert contract_hash(_app(ui_bundle=None)) == contract_hash(_app(ui_bundle=None))


def test_mounting_the_ui_does_not_freeze_the_schema(ui_zip):
    """`contract_hash` calls `app.openapi()`, which FastAPI caches on the app.
    Leaving that cache populated would mean a route added after mount is
    invisible to every later `app.openapi()` -- including the auth sweeps in
    test_api_auth.py, which walk exactly that schema."""
    from fastapi import APIRouter

    client = _client(ui_bundle=UiBundle.open(zip_path=ui_zip, dir_path=None))
    later = APIRouter()

    @later.get("/later")
    async def later_handler():
        return {"ok": True}

    client.app.include_router(later, prefix="/v1")
    assert "/v1/later" in client.app.openapi()["paths"]


# ─── opening a bundle ────────────────────────────────────────────────────
def test_nothing_configured_is_no_bundle():
    assert UiBundle.open(zip_path=None, dir_path=None) is None


def test_a_missing_file_is_no_bundle(tmp_path):
    assert UiBundle.open(zip_path=tmp_path / "absent.zip", dir_path=None) is None
    assert UiBundle.open(zip_path=None, dir_path=tmp_path / "absent") is None


def test_a_bundle_without_a_marker_is_no_bundle(tmp_path):
    path = tmp_path / "bare.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("index.html", INDEX)
    assert UiBundle.open(zip_path=path, dir_path=None) is None


def test_a_plain_pwa_manifest_is_not_mistaken_for_the_marker(tmp_path):
    """The rename's whole point. A bundle carrying a Next PWA `manifest.json`
    and no `.tenka-ui.json` is not a TENKA bundle at all -- and must not be
    read as one just because a file with the old name happens to parse."""
    path = tmp_path / "pwa.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("index.html", INDEX)
        archive.writestr("manifest.json",
                         json.dumps({"version": 1, "name": "TENKA Studio",
                                     "icons": [], "start_url": "/"}))
    assert UiBundle.open(zip_path=path, dir_path=None) is None


def test_a_corrupt_archive_is_no_bundle(tmp_path):
    path = tmp_path / "corrupt.zip"
    path.write_bytes(b"this is not a zip file at all")
    assert UiBundle.open(zip_path=path, dir_path=None) is None


def test_the_manifest_is_readable(ui_zip):
    bundle = UiBundle.open(zip_path=ui_zip, dir_path=None)
    manifest = bundle.manifest()
    assert manifest["version"] == UI_MANIFEST_VERSION
    assert manifest["contract"] == _reference_contract()
    assert manifest["builtAt"]


@pytest.mark.parametrize("version", [1.0, True, "1", 0, 2, None, [1], {"v": 1}])
def test_only_an_exact_integer_marker_version_is_accepted(tmp_path, version):
    """`1.0` and `True` both compare equal to `1` in Python, so a bare `!=`
    accepts a marker whose version is a float or a boolean. The house rule for
    an on-disk marker is to reject what is not exactly right rather than guess
    what was meant."""
    path = _write_zip(tmp_path / f"v-{type(version).__name__}.zip", EXPORT,
                      version=version)
    assert UiBundle.open(zip_path=path, dir_path=None) is None


def test_a_stale_bundle_names_both_contracts(ui_zip_wrong_contract):
    """'Stale' with no hashes is a dead end for whoever has to fix it."""
    client = _client(ui_bundle=UiBundle.open(zip_path=ui_zip_wrong_contract,
                                             dir_path=None))
    text = client.get("/").text
    assert "0" * 64 in text
    assert _reference_contract() in text


def test_a_stale_bundle_does_not_echo_an_unbounded_contract(tmp_path):
    """The declared contract comes out of an untrusted marker. Interpolated
    whole, a 500 K-char value makes *every request* answer with a 500 KB body
    -- an amplifier handed to anyone who can get a bundle onto the machine, on
    a route that needs no credential."""
    path = _write_zip(tmp_path / "loud.zip", EXPORT, contract="A" * 500_000)
    client = _client(ui_bundle=UiBundle.open(zip_path=path, dir_path=None))
    r = client.get("/")
    assert r.status_code == 503
    assert len(r.content) < 1024
    assert "500000 chars" in r.text


def test_a_stale_bundle_refuses_assets_too(ui_zip_wrong_contract):
    """The mismatch is not cosmetic: stale JS against a new API is the actual
    breakage, so the whole bundle goes dark, not only its documents."""
    client = _client(ui_bundle=UiBundle.open(zip_path=ui_zip_wrong_contract,
                                             dir_path=None))
    assert client.get("/_next/static/chunks/main-deadbeef.js").status_code == 503


def test_a_stale_bundle_does_not_break_the_api(ui_zip_wrong_contract):
    client = _client(ui_bundle=UiBundle.open(zip_path=ui_zip_wrong_contract,
                                             dir_path=None))
    assert client.get("/v1/status").status_code == 401


def test_reading_a_member_twice_is_the_same_bytes(ui_zip):
    """The cache is populated on first access; it must not consume the source."""
    bundle = UiBundle.open(zip_path=ui_zip, dir_path=None)
    first = bundle.read("app/chat.html")
    second = bundle.read("app/chat.html")
    assert first == second == (CHAT, "text/html; charset=utf-8")


def test_a_dev_directory_is_not_cached(ui_dir):
    """The dev path wins over the zip so a developer can see a change without
    re-packaging. Caching it would take that straight back -- the first read
    would pin the file for the life of the daemon."""
    bundle = UiBundle.open(zip_path=None, dir_path=ui_dir)
    (ui_dir / "index.html").write_text("<html>first</html>", encoding="utf-8")
    assert b"first" in bundle.read("index.html")[0]
    (ui_dir / "index.html").write_text("<html>second</html>", encoding="utf-8")
    assert b"second" in bundle.read("index.html")[0]


def test_an_absurdly_large_member_is_refused(tmp_path):
    """A zip bomb is a memory primitive against a cache that reads on demand.
    The declared size is checked before the read, not after."""
    from assistant.io.api.ui import MAX_MEMBER_BYTES

    path = _write_zip(tmp_path / "bomb.zip",
                      {"index.html": INDEX,
                       "bomb.bin": b"\0" * (MAX_MEMBER_BYTES + 1)},
                      contract="any")
    bundle = UiBundle.open(zip_path=path, dir_path=None)
    assert bundle.read("index.html") is not None
    assert bundle.read("bomb.bin") is None


def test_an_absurdly_large_dev_directory_member_is_refused(ui_dir):
    """The directory read is bounded at cap+1 exactly as both zip reads are: a
    size that was true when `stat()` ran is not a size that is still true when
    `read()` does, and a dev directory is writable by whatever else is running
    on the machine."""
    from assistant.io.api.ui import MAX_MEMBER_BYTES

    (ui_dir / "huge.bin").write_bytes(b"\0" * (MAX_MEMBER_BYTES + 1))
    bundle = UiBundle.open(zip_path=None, dir_path=ui_dir)
    assert bundle.read("huge.bin") is None
    assert bundle.read("index.html") is not None


def test_an_oversized_marker_is_no_bundle(tmp_path):
    path = tmp_path / "huge-marker.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MARKER_NAME, " " * (2 * 1024 * 1024) + "{}")
    assert UiBundle.open(zip_path=path, dir_path=None) is None


def test_the_configured_path_is_a_declared_setting():
    """`studio_ui_path` has to be a real runtime setting, not a constant, so
    `/set` and the settings UI can point the daemon at a dev export."""
    from assistant import config
    from assistant.core import runtime_config

    assert "studio_ui_path" in runtime_config.REGISTRY
    assert hasattr(config, "STUDIO_UI_PATH")


def test_an_io_helper_reads_the_zip_without_an_archive_handle_left_open(ui_zip):
    """Windows will not let a test's tmp_path be torn down while a handle on a
    file inside it is still open. A bundle that leaked its ZipFile would make
    every fixture using it flaky, not obviously wrong."""
    bundle = UiBundle.open(zip_path=ui_zip, dir_path=None)
    assert bundle.read("index.html") is not None
    with io.open(ui_zip, "r+b"):          # would raise if a handle were held
        pass
