# assistant/io/api/ui.py
"""Serve the Studio front-end from the daemon itself.

Why the daemon serves the UI at all: it puts the page and the API on one
origin. That removes mixed content, removes Private Network Access, and is the
only reason the `httpOnly` device cookie of Task 5 can exist -- a cookie is
host-scoped, so a UI on someone else's origin could never hold one. From
Milestone 6b the same origin is a public tunnel URL.

Which makes this module one of exactly two unauthenticated surfaces in the
whole API. It cannot demand a credential: the page it serves is what
*bootstraps* pairing, so requiring one would be a closed loop. Everything here
is written on the assumption that the caller is hostile and remote.

Three things carry that weight:

* `normalise_member()` -- a request path becomes a member name only if it
  survives every escape shape a path can take. It runs *before* the archive is
  touched, and it is also applied to the archive's own stored names, so an
  entry that escapes the root is dropped at load rather than trusted.
* the resolution order -- exact member, `<path>.html`, `<path>/index.html`,
  then a *bounded* index fallback. Task 1 established that the export writes
  `app/chat.html`, so an index-only server would silently discard every
  prerendered page while appearing to work.
* the contract hash -- a bundle built against a different API refuses to serve
  at all, rather than half-working in a way nobody can diagnose.

Layering: io/api -- core + config only.
"""
from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import threading
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# Bumped whenever the manifest's meaning changes. An older marker is rejected
# rather than guessed at -- the same rule every other on-disk marker in this
# codebase follows.
UI_MANIFEST_VERSION = 1

# The largest single file the daemon will hand out. The real export's biggest
# asset is a ~790 KB JS chunk, so this is generous by an order of magnitude
# while still bounding what a hand-crafted archive can make the cache hold.
MAX_MEMBER_BYTES = 8 * 1024 * 1024

# The marker is small by construction. Reading it is the first thing that
# happens to an untrusted archive, so it is also the first thing to bound.
_MAX_MARKER_BYTES = 1024 * 1024

# The daemon's own build marker, written by the packaging step and read from
# the root of the bundle. *Not* `manifest.json`: that is a name a Next.js app
# may legitimately ship as a PWA web app manifest, and two different files
# competing for one name is a collision waiting to be discovered in
# production. A dotted name also keeps it out of the way of anything the
# exporter emits.
MARKER_NAME = ".tenka-ui.json"

# Members no HTTP request may reach, whatever it spells them.
#
# Compared case-folded, and that is the mechanism rather than a fix for one
# name: this is the module's only deny-list, so anything added here later
# inherits whatever comparison it uses. An exact-string compare is defeated on
# Windows by pressing shift -- `/MANIFEST.JSON` and `/Manifest.json` name the
# same file on a case-insensitive filesystem -- and in a zip by storing a
# second entry under a different case.
#
# The marker is private because it carries the exact API contract hash. An
# unauthenticated fingerprint of the running build is a targeting aid on a
# public URL and buys a browser nothing.
_PRIVATE_MEMBERS = frozenset({MARKER_NAME.casefold()})


def _is_private(member: str) -> bool:
    return member.casefold() in _PRIVATE_MEMBERS


# ─── content types ───────────────────────────────────────────────────────
# Pinned in code rather than looked up. `mimetypes` consults the Windows
# registry, where `.js` has historically resolved to `text/plain` -- which a
# browser refuses to execute as a module script -- and where `.woff2`/`.webp`
# are simply absent. The daemon runs on other people's machines; what it
# serves must not depend on which ones.
_CONTENT_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".map": "application/json",
    ".webmanifest": "application/manifest+json",
    ".txt": "text/plain; charset=utf-8",
    ".xml": "application/xml",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".avif": "image/avif",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".wasm": "application/wasm",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}


def content_type_for(member: str) -> str:
    suffix = Path(member).suffix.lower()
    known = _CONTENT_TYPES.get(suffix)
    if known:
        return known
    guessed, _ = mimetypes.guess_type(member)
    # Never `text/html` by accident: a sniffed type is exactly how an uploaded
    # asset turns into a script in someone's origin. Anything unrecognised is
    # a download, not a document.
    if guessed and not guessed.startswith("text/html"):
        return guessed
    return "application/octet-stream"


# ─── path normalisation ──────────────────────────────────────────────────
# Windows resolves these to devices rather than files, matching on the name up
# to the first period no matter which directory it appears in -- so opening
# `out/AUX.woff2` talks to a serial port.
#
# Applied *only* on the directory read, and deliberately not in
# `normalise_member`. The rule is a property of the Win32 file namespace, not
# of what a legal bundle member is: nothing is opened by name on the zip path,
# so there it can produce false negatives and nothing else. That is not
# hypothetical -- Next preserves source basenames in hashed asset names, so a
# font vendored as `aux.woff` exports as `_next/static/media/aux.a1b2c3.woff2`,
# and refusing it everywhere would make that asset an unfixable 404.
_RESERVED_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{n}" for n in range(1, 10)}
    | {f"LPT{n}" for n in range(1, 10)}
)


def _names_a_windows_device(safe: str) -> bool:
    return any(part.split(".", 1)[0].upper() in _RESERVED_STEMS
               for part in safe.split("/"))


# Characters that are either separators on some platform, illegal on Windows,
# or a way of naming something other than the file you appear to be naming
# (`:` selects an NTFS alternate data stream). Backslash is folded to `/`
# before this check, so it is a separator by then and never reaches here.
_ILLEGAL_CHARS = frozenset('\x00:*?"<>|\r\n\t')

_MAX_MEMBER_DEPTH = 24
_MAX_MEMBER_LENGTH = 512


def normalise_member(raw: str) -> str | None:
    """Turn a request path into an archive member name, or `None` to refuse.

    Refuses rather than repairs. A normaliser that *fixes* a hostile path is a
    normaliser somebody has to prove is complete; one that refuses only has to
    be sure the input was not obviously legitimate. Every shape below has been
    a real traversal at some point: dot segments, both separators, absolute
    and network-path forms, UNC, drive letters and drive-relative names, NUL
    truncation, Windows' habit of stripping trailing dots and spaces, and NTFS
    alternate data streams. Reserved device names are *not* checked here -- see
    `_names_a_windows_device` for why that belongs to the directory read alone.

    Percent-encoding is already decoded by the ASGI server before a path
    reaches here, so `%2e%2e%2f` arrives as `../` and is caught below. A
    double-encoded `%252e` decodes exactly once, to the inert literal `%2e`,
    and is simply a name the bundle does not contain.
    """
    if raw is None:
        return None
    if len(raw) > _MAX_MEMBER_LENGTH:
        return None
    # Folded first: a backslash is a separator on the platform this daemon
    # runs on, so `..\x` must be seen as the dot segment it is, not as a
    # single exotic filename.
    candidate = raw.replace("\\", "/")
    # A trailing slash is a browser habit (`/app/`), not an escape. Stripping
    # it before the empty-segment check keeps `/app/` meaning `/app` while
    # still refusing an interior `//`.
    candidate = candidate.rstrip("/")
    if not candidate:
        return ""                       # the root document
    if candidate.startswith("/"):
        # Absolute, and the network-path form `//host/share` with it. Neither
        # is a name inside this archive.
        return None
    parts = candidate.split("/")
    if len(parts) > _MAX_MEMBER_DEPTH:
        return None
    for part in parts:
        if part in ("", ".", ".."):
            return None
        if _ILLEGAL_CHARS & set(part):
            return None
        # Windows silently strips these when opening a file, so `x.html.` and
        # `x.html ` both open `x.html` -- two names for one member is one name
        # too many for anything that has to be reasoned about.
        if part != part.strip() or part.endswith("."):
            return None
    return "/".join(parts)


# ─── the contract hash ───────────────────────────────────────────────────
def contract_hash(app: FastAPI) -> str:
    """A stable fingerprint of the API this daemon serves.

    Over a *canonical* serialisation, never over the bytes of a schema file:
    the packaging step computes the same hash on its side, and two identical
    APIs dumped with different indentation must agree. Hashing raw bytes would
    make the guard a whitespace detector.
    """
    canonical = json.dumps(app.openapi(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ─── the bundle ──────────────────────────────────────────────────────────
class UiBundle:
    """A Studio build, read from a zip or from a directory.

    Members are read on demand and cached in memory. The archive handle is
    *not* held open between reads: on Windows an open handle inside a
    directory blocks that directory from being removed, which turns a leak
    here into flaky teardown everywhere else rather than into an obvious bug.
    """

    def __init__(self, *, manifest: dict[str, Any], names: frozenset[str],
                 zip_path: Path | None, dir_path: Path | None) -> None:
        self._manifest = manifest
        self._names = names
        self._zip_path = zip_path
        self._dir_path = dir_path
        self._cache: dict[str, tuple[bytes, str]] = {}
        self._lock = threading.Lock()

    # ─── opening ─────────────────────────────────────────────────────────
    @classmethod
    def open(cls, *, zip_path: Path | None, dir_path: Path | None) -> "UiBundle | None":
        """Prefer a directory, fall back to the zip, else no UI at all.

        The directory is a developer's own `next build` output, named by the
        `studio_ui_path` setting and resolved by the caller. It wins over the
        zip so that iterating on Studio needs no re-vendoring.
        """
        if dir_path is not None:
            bundle = cls._from_dir(Path(dir_path))
            if bundle is not None:
                return bundle
        if zip_path is not None:
            return cls._from_zip(Path(zip_path))
        return None

    @classmethod
    def _from_zip(cls, path: Path) -> "UiBundle | None":
        try:
            with zipfile.ZipFile(path) as archive:
                names: set[str] = set()
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    safe = normalise_member(info.filename)
                    if safe is None or not safe:
                        # Zip-slip, from the archive side. Nothing here
                        # extracts to disk, so the classic write primitive
                        # does not exist -- but an entry stored as
                        # `../devices.json` would still be *readable by name*
                        # if the loader trusted the names it was handed.
                        logger.warning("[UI] dropped an unsafe archive entry")
                        continue
                    names.add(safe)
                manifest = cls._read_marker_from_zip(archive)
        except (OSError, zipfile.BadZipFile, ValueError) as exc:
            logger.warning(f"[UI] cannot open the Studio bundle at {path}: {exc}")
            return None
        if manifest is None:
            return None
        return cls(manifest=manifest, names=frozenset(names),
                   zip_path=path, dir_path=None)

    @classmethod
    def _from_dir(cls, root: Path) -> "UiBundle | None":
        marker = root / MARKER_NAME
        try:
            if not marker.is_file() or marker.stat().st_size > _MAX_MARKER_BYTES:
                return None
            manifest = cls._validated(json.loads(marker.read_text(encoding="utf-8")))
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            logger.warning(f"[UI] cannot open the Studio directory at {root}: {exc}")
            return None
        if manifest is None:
            return None
        # Names are not enumerated up front: a dev directory changes under the
        # daemon while it runs, so membership is decided per read instead.
        return cls(manifest=manifest, names=frozenset(),
                   zip_path=None, dir_path=root)

    @classmethod
    def _read_marker_from_zip(cls, archive: zipfile.ZipFile) -> dict[str, Any] | None:
        try:
            info = archive.getinfo(MARKER_NAME)
        except KeyError:
            logger.warning(f"[UI] the Studio bundle carries no {MARKER_NAME}")
            return None
        if info.file_size > _MAX_MARKER_BYTES:
            return None
        with archive.open(info) as handle:
            # One byte past the cap, so a lying `file_size` is caught by the
            # read rather than trusted by the check above it.
            raw = handle.read(_MAX_MARKER_BYTES + 1)
        if len(raw) > _MAX_MARKER_BYTES:
            return None
        try:
            return cls._validated(json.loads(raw.decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            return None

    @staticmethod
    def _validated(manifest: Any) -> dict[str, Any] | None:
        if not isinstance(manifest, dict):
            return None
        version = manifest.get("version")
        # Typed, not merely compared. Python says `1.0 == 1` and `True == 1`,
        # so a bare `!=` accepts a marker whose version is a float or a
        # boolean -- and the house rule for an on-disk marker is to reject
        # what is not exactly right rather than to guess what was meant.
        # `bool` is excluded explicitly because it is a subclass of `int`.
        if isinstance(version, bool) or not isinstance(version, int) \
                or version != UI_MANIFEST_VERSION:
            logger.warning(
                f"[UI] refusing a Studio bundle at marker version "
                f"{version!r}; this daemon speaks {UI_MANIFEST_VERSION}")
            return None
        return manifest

    # ─── reading ─────────────────────────────────────────────────────────
    def manifest(self) -> dict:
        return dict(self._manifest)

    def read(self, member: str) -> tuple[bytes, str] | None:
        """Return `(body, content_type)` for a member, or `None` if there
        isn't one. The name is normalised here too, so a caller that skips the
        route's own check still cannot escape the root."""
        safe = normalise_member(member)
        if not safe:
            return None
        # A dev directory changes under a running daemon -- that is the entire
        # reason it wins over the zip -- so nothing read from one is cached. A
        # zip cannot change without the process being restarted, so it is.
        cacheable = self._dir_path is None
        if cacheable:
            with self._lock:
                cached = self._cache.get(safe)
                if cached is not None:
                    return cached
        body = self._read_bytes(safe)
        if body is None:
            return None
        entry = (body, content_type_for(safe))
        if cacheable:
            with self._lock:
                self._cache[safe] = entry
        return entry

    def _read_bytes(self, safe: str) -> bytes | None:
        if self._dir_path is not None:
            return self._read_from_dir(safe)
        return self._read_from_zip(safe)

    def _read_from_dir(self, safe: str) -> bytes | None:
        root = self._dir_path
        assert root is not None
        # Before anything touches the filesystem: on Windows this name would
        # be resolved to a device, not to the file it appears to describe.
        if _names_a_windows_device(safe):
            return None
        try:
            resolved = (root / safe).resolve()
            # The name was already proven not to *say* `..`; this proves the
            # filesystem does not take it somewhere else anyway. A symlink is
            # invisible to name-level normalisation, and so is an 8.3 short
            # name -- only resolving the real path catches either.
            if not resolved.is_relative_to(root.resolve()):
                logger.warning("[UI] refused a dev-directory member that escaped the root")
                return None
            if not resolved.is_file():
                return None
            with resolved.open("rb") as handle:
                # Bounded at cap+1, exactly as both zip paths are. And bounded
                # *only* here: a `stat().st_size` pre-check would look like a
                # second guard while being neither necessary nor sufficient --
                # a size that was true when `stat()` ran is not a size that is
                # still true when `read()` does, and a dev directory is
                # writable by whatever else is running on the machine. One
                # guard that is always right beats two where the weaker one
                # silently does the work.
                body = handle.read(MAX_MEMBER_BYTES + 1)
            if len(body) > MAX_MEMBER_BYTES:
                logger.warning(f"[UI] refused an oversized dev-directory member: {safe}")
                return None
            return body
        except (OSError, ValueError):
            return None

    def _read_from_zip(self, safe: str) -> bytes | None:
        if safe not in self._names:
            return None
        try:
            with zipfile.ZipFile(self._zip_path) as archive:
                info = archive.getinfo(safe)
                if info.file_size > MAX_MEMBER_BYTES:
                    logger.warning(f"[UI] refused an oversized bundle member: {safe}")
                    return None
                with archive.open(info) as handle:
                    # Bounded regardless of what the header claimed: a zip
                    # bomb's declared size is attacker-controlled, the bytes
                    # actually produced are what costs memory.
                    body = handle.read(MAX_MEMBER_BYTES + 1)
            if len(body) > MAX_MEMBER_BYTES:
                logger.warning(f"[UI] refused a bundle member that lied about its size: {safe}")
                return None
            return body
        except (KeyError, OSError, zipfile.BadZipFile, ValueError):
            return None

    # ─── the bounded index fallback ──────────────────────────────────────
    def is_route_prefix(self, segment: str) -> bool:
        """Whether an unknown path under `segment` may fall back to the shell.

        The fallback exists so a client-side route still loads on a hard
        refresh. It is deliberately *not* blanket: a server that answered
        `index.html` to everything would answer 200 to `/instance_secret` and
        `/devices.json` -- names that were never routes -- and would turn a
        missing asset into an HTML body a browser then tries to execute.

        A segment qualifies when the export itself prerendered it as a route,
        which for a `trailingSlash: false` export means a sibling
        `<segment>.html` exists (`/app` -> `app.html`, alongside the `app/`
        directory holding its children). An asset root like `_next/` has no
        such sibling and therefore never falls back.
        """
        return (self.read(f"{segment}.html") is not None
                or self.read(segment) is not None)


# ─── mounting ────────────────────────────────────────────────────────────
# Same-origin by construction, so nothing here needs a remote host. The two
# `'unsafe-inline'` allowances are not laziness: a Next.js App Router export
# ships its hydration payload as inline `<script>self.__next_f.push(...)</script>`
# and its critical CSS as inline `<style>`, both written into prerendered HTML
# at build time. A nonce would have to be injected into every document on every
# request, which means rewriting bytes we otherwise hand through untouched --
# and a nonce that is not also applied to the `_next` chunk graph buys nothing.
# What the policy does buy is real: no plugins (`object-src`), no `<base>`
# rewrite, no form posts anywhere, no framing, and no origin but this one for
# scripts, styles, fonts or network calls.
_CSP = "; ".join((
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "font-src 'self' data:",
    # `'self'` covers the same-origin `ws:`/`wss:` the event socket uses:
    # CSP3 matches a WebSocket URL against `'self'` when host and port agree.
    "connect-src 'self'",
    "media-src 'self' blob: data:",
    "worker-src 'self' blob:",
    "object-src 'none'",
    "base-uri 'none'",
    "form-action 'none'",
    "frame-ancestors 'none'",
))

_SECURITY_HEADERS = {
    "Content-Security-Policy": _CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    # Redundant with `frame-ancestors` for a modern browser, and the only
    # thing an old one understands.
    "X-Frame-Options": "DENY",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}

# `_next/static/*` URLs carry a content hash, so they are immutable by
# construction and re-downloading megabytes of them per reload is pure cost on
# a phone over a tunnel. A document's URL does *not* change when the build
# does, so it can never be cached.
_IMMUTABLE_PREFIX = "_next/static/"
_IMMUTABLE_CACHE = "public, max-age=31536000, immutable"
_NO_CACHE = "no-store, no-cache, must-revalidate"


def _stamp(response: Response, *, cache_control: str = _NO_CACHE) -> Response:
    for name, value in _SECURITY_HEADERS.items():
        response.headers[name] = value
    response.headers["Cache-Control"] = cache_control
    return response


def _refuse(status: int, body: dict) -> Response:
    """An error answer that carries the same headers a page does.

    A 403 or a 404 is the response an unauthenticated attacker can most
    reliably elicit, so it is the last one that should arrive without
    `nosniff`, a CSP and a framing rule. Built here rather than raised as an
    `HTTPException`, because the app's exception handlers sit outside this
    route and cannot know these belong to the UI.

    Only for the refusals this route actually owns. A 405 is not one of them:
    claiming verbs it does not serve, purely so it could stamp their 405, was
    a bad trade -- see the route registration below.
    """
    return _stamp(JSONResponse(status_code=status, content=body))


def _short(value: object, limit: int = 80) -> str:
    """Bound a value that came out of an untrusted marker before it is put in
    a response body. A 500 K-char `contract` would otherwise make every single
    request answer with a 500 KB body."""
    text = str(value)
    return text if len(text) <= limit else f"{text[:limit]}...({len(text)} chars)"


def mount_ui(app: FastAPI, bundle: UiBundle | None) -> None:
    """Register the catch-all that serves `bundle`, if there is one.

    Registered last, after every router, so a real API path always wins the
    match. Registered as an ordinary route rather than a `Mount`, because a
    `Mount` is invisible to the OpenAPI-based auth sweeps in test_api_auth.py
    -- and a public route that the auth sweep cannot even see is the one kind
    this app should never grow.

    It is nonetheless `include_in_schema=False`: a catch-all belongs in
    nobody's generated TypeScript client, and it is deliberately
    unauthenticated, so appearing in the sweep would only make the sweep
    wrong. That is a *named* exemption in `test_api_auth.py`'s
    `test_no_route_hides_from_the_schema_sweep`, not an absence, and
    tests/test_api_ui_serving.py is the compensating coverage the exemption
    points at.
    """
    if bundle is None:
        return

    expected = contract_hash(app)
    # `app.openapi()` caches its result on the app, so taking the hash also
    # freezes the schema -- and every later `app.openapi()` call, including the
    # auth sweeps in test_api_auth.py that walk exactly that schema, would then
    # be blind to any route registered afterwards. Dropping the cache costs one
    # rebuild and keeps the sweeps honest.
    app.openapi_schema = None

    declared = bundle.manifest().get("contract")
    stale: str | None = None
    if declared != expected:
        # `declared` came out of an untrusted marker, so it is bounded before
        # it goes anywhere near a response body -- see `_short`.
        stale = (f"the Studio UI bundle is stale: it was built against API "
                 f"contract {_short(declared)}, and this daemon serves {expected}")
        logger.warning(f"[UI] {stale}")

    # `HEAD` as well as `GET`, because FastAPI's `APIRoute` does not add it for
    # free the way Starlette's plain `Route` does. In 6b the pairing URL gets
    # pasted into chat apps, and a link unfurler, an uptime probe or a tunnel
    # health check sends `HEAD` first -- a 405 there reads as "the daemon is
    # down".
    #
    # And *only* those two. An earlier revision claimed every verb so that the
    # 405 would be one this route could stamp its security headers onto. That
    # trade does not pay: the headers buy nothing on a fixed JSON body that
    # reflects no input and renders nothing, while a catch-all owning every
    # verb silently swallows any route registered after `create_app()` returns
    # -- for `/v1` the auth sweep would catch it, but a non-GET route outside
    # `/v1` would simply 405 with nothing watching. Least privilege applies to
    # routing as much as to capabilities: claim the verbs you serve and leave
    # the rest to the framework's own 405.
    @app.api_route("/{ui_path:path}", include_in_schema=False,
                   methods=["GET", "HEAD"])
    async def serve_studio_ui(ui_path: str) -> Response:
        # An unrouted `/v1` path is an API miss, not a page. Answering it with
        # a 200 HTML shell would hand a `fetch()` a body it then tries to
        # parse as JSON, and would turn every typo'd endpoint into a silent
        # success. Checked on the raw path, before anything else. The body is
        # the same one the app's own 404 handler produces, so an API miss looks
        # identical whether or not a UI bundle happens to be mounted.
        if ui_path == "v1" or ui_path.startswith("v1/"):
            return _refuse(404, {"error": "not found"})

        member = normalise_member(ui_path)
        if member is None:
            # Deliberately not echoing the path back: it is attacker-supplied
            # and this response is rendered in a browser.
            return _refuse(403, {"detail": "forbidden"})

        if stale is not None:
            # The whole bundle goes dark, documents and assets alike. Stale JS
            # against a new API is the actual breakage; serving the assets
            # while withholding the page would only make it harder to see.
            return _stamp(JSONResponse(status_code=503, content={"error": stale}))

        if _is_private(member):
            return _refuse(404, {"error": "not found"})

        if member:
            # Task 1's export shape, in order: the exact file (assets), then
            # `<path>.html` (every prerendered route), then
            # `<path>/index.html` (the root, and any future `trailingSlash`
            # export).
            candidates = [member, f"{member}.html", f"{member}/index.html"]
        else:
            candidates = ["index.html"]
        for candidate in candidates:
            if _is_private(candidate):
                continue
            found = bundle.read(candidate)
            if found is not None:
                body, content_type = found
                immutable = member.startswith(_IMMUTABLE_PREFIX)
                return _stamp(
                    Response(content=body, media_type=content_type),
                    cache_control=_IMMUTABLE_CACHE if immutable else _NO_CACHE)

        # The bounded fallback. A leaf with an extension is an asset, and a
        # missing asset must stay a 404 -- see `is_route_prefix` for why the
        # rest of it is bounded rather than blanket.
        leaf = member.rsplit("/", 1)[-1]
        if member and "." not in leaf:
            first = member.split("/", 1)[0]
            if bundle.is_route_prefix(first):
                shell = bundle.read("index.html")
                if shell is not None:
                    body, content_type = shell
                    return _stamp(Response(content=body, media_type=content_type))

        return _refuse(404, {"error": "not found"})

# Nothing in this module reads `config`. It is tempting -- `studio_ui_path`
# names exactly what `UiBundle.open()` wants -- but `import-linter` follows
# imports transitively, and `assistant.config` reaches `assistant.llm` and
# (via `core.runtime_config`) `assistant.storage`, so a single lazy
# `from ... import config` in here breaks the `io.api never reaches past core
# and config` contract twice over. Paths arrive the same way `runtime`,
# `vault` and `origins` already do: resolved by main.py, injected by the
# caller. See `config.STUDIO_UI_PATH` for the setting itself.
