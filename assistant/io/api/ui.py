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

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
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

# How many members the directory loader will publish. A real export is a few
# hundred files; this bounds what a `studio_ui_path` pointed at something
# enormous (a home directory, a node_modules tree) costs at mount time, and it
# is a refusal rather than a truncation because a half-enumerated bundle would
# serve some of the export and 404 the rest, which reads as a build problem.
_MAX_DIR_MEMBERS = 20_000

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
        # member -> the one task currently reading it. Two requests racing the
        # same cache miss await the same task instead of each paying for the
        # decompression; see `read_async`. Guarded by `_lock`, the same lock
        # the cache dict uses, because "is it cached?" and "is somebody
        # already fetching it?" have to be answered as one decision -- two
        # locks would leave a window in which both answers are no.
        self._inflight: dict[str, asyncio.Future] = {}

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
        names = cls._enumerate_dir(root)
        if names is None:
            return None
        return cls(manifest=manifest, names=names,
                   zip_path=None, dir_path=root)

    @classmethod
    def _enumerate_dir(cls, root: Path) -> frozenset[str] | None:
        """Every member the directory publishes, decided once at mount time.

        This is the half `_from_zip` always had and this loader did not. The
        zip path enumerates `infolist()` into `self._names` and refuses
        anything outside that set; this one passed `names=frozenset()`, so the
        only barrier between an unauthenticated request and any file under the
        root was `_PRIVATE_MEMBERS` -- a deny-list with one entry. `GET /.env`
        returned 200 with the file's contents. Anything a build tool, an
        editor or a developer leaves in the export directory (`.env*`,
        `.git/`, a swap file, a stray copy of `devices.json`) was public.
        An allow-list built from what is actually there is the only shape that
        does not need a new deny-list entry per tool anybody ever runs.

        The cost is that a file appearing after mount is not served until the
        daemon restarts, which is a real hit to the iterate-on-Studio workflow
        this loader exists for. Re-walking on a miss was the obvious repair
        and is the wrong one: it hands an unauthenticated caller a directory
        walk per request for a name that does not exist, on the one route in
        this API with no credential requirement. Editing an existing file
        still shows up immediately -- nothing from a directory is cached.

        Enumeration alone is not the whole of it, and it is worth being exact
        about why. The zip's names come from an archive a build step produced;
        a directory's come from whatever is on disk, so walking it faithfully
        would publish a `.env` sitting next to `index.html` just as readily as
        the old per-read check did -- lens 1 F7's observed `GET /.env` -> 200
        would still hold. So a *hidden* entry -- any path segment beginning
        with a dot -- is not a member here. That is a rule about a shape, not a
        list of names: `.env`, `.env.local`, `.git/`, `.DS_Store` and an
        editor's dotfile all fall out of it without any of them being written
        down, which is the point. A static export needs no dot-prefixed file
        served; the one this daemon does read, the marker, is opened by path
        and is `_PRIVATE_MEMBERS` besides.

        Not applied to the zip loader, deliberately. That archive is a build
        artifact the packaging step already scans for secrets before it ships,
        and its members were chosen by a script rather than by whatever a
        developer's working directory happens to contain. The loaders publish
        different sets, and the difference is in the safe direction.

        `followlinks=False`, so a symlinked directory is not descended into
        and cannot make this walk unbounded. A symlinked *file* is still
        enumerated by name and still refused at read time by the resolved-path
        guard in `_read_from_dir`, which is where that check belongs.
        """
        names: set[str] = set()
        try:
            for current, dirs, files in os.walk(root, followlinks=False):
                relative = Path(current).relative_to(root)
                # Pruned in place, which is what stops `os.walk` descending
                # into them at all -- a hidden directory's contents are not
                # members and are not worth the stat calls either.
                dirs[:] = [name for name in dirs if not name.startswith(".")]
                for filename in files:
                    if filename.startswith("."):
                        continue
                    if len(names) >= _MAX_DIR_MEMBERS:
                        logger.warning(
                            f"[UI] refusing a Studio directory at {root}: more "
                            f"than {_MAX_DIR_MEMBERS} files under its root")
                        return None
                    raw = filename if relative == Path(".") \
                        else f"{relative.as_posix()}/{filename}"
                    safe = normalise_member(raw)
                    if not safe:
                        # Same reasoning as the zip side: a name that does not
                        # survive normalisation is not a member, whatever the
                        # filesystem lets it be called.
                        logger.warning("[UI] dropped an unsafe directory entry")
                        continue
                    names.add(safe)
        except OSError as exc:
            logger.warning(f"[UI] cannot enumerate the Studio directory "
                           f"at {root}: {exc}")
            return None
        return frozenset(names)

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
        entry = self._read_entry(safe)
        if entry is not None and cacheable:
            with self._lock:
                self._cache[safe] = entry
        return entry

    async def read_async(self, member: str) -> tuple[bytes, str] | None:
        """`read()`, off the event loop and coalesced. What the route calls.

        Two things this fixes, both measured by lens 5 F4 against the largest
        member this daemon will hand out (8 MiB, `MAX_MEMBER_BYTES`):

        * a first read cost 47-97ms of *synchronous* work on the loop the
          assistant herself runs on, from a route with no credential
          requirement at all -- so `asyncio.to_thread`;
        * ten concurrent first-requests for the same member produced ten
          independent full decompressions, because the lock is held only
          around the cache dict's get and set -- so the in-flight map.

        A miss is still never cached (`_read_entry` returning `None` populates
        nothing), which is what keeps a flood of requests for paths that do
        not exist from being a cache-churn primitive. The in-flight entry is
        cleared either way, so it does not become a negative cache by accident.
        """
        safe = normalise_member(member)
        if not safe:
            return None
        cacheable = self._dir_path is None
        with self._lock:
            if cacheable:
                cached = self._cache.get(safe)
                if cached is not None:
                    return cached
            task = self._inflight.get(safe)
            if task is None:
                task = asyncio.ensure_future(
                    asyncio.to_thread(self._read_entry, safe))
                self._inflight[safe] = task
                task.add_done_callback(
                    lambda done, key=safe: self._settle(key, done))
        # Shielded: a client that disconnects mid-read cancels *its* await,
        # and cancelling the shared task would fail every other request racing
        # the same member for a reason none of them caused.
        return await asyncio.shield(task)

    def _settle(self, safe: str, task: asyncio.Future) -> None:
        """Retire an in-flight read and cache what it produced, if anything."""
        with self._lock:
            if self._inflight.get(safe) is task:
                del self._inflight[safe]
            if self._dir_path is not None or task.cancelled():
                return
            if task.exception() is not None:
                return
            entry = task.result()
            if entry is not None:
                self._cache[safe] = entry

    def _read_entry(self, safe: str) -> tuple[bytes, str] | None:
        """The whole read, cache aside: bytes plus the type they are served as.

        One function so the sync and async paths cannot drift -- and so the
        thing handed to `asyncio.to_thread` is the entire piece of blocking
        work, not most of it.
        """
        body = self._read_bytes(safe)
        if body is None:
            return None
        return body, content_type_for(safe)

    def _read_bytes(self, safe: str) -> bytes | None:
        if self._dir_path is not None:
            return self._read_from_dir(safe)
        return self._read_from_zip(safe)

    def _read_from_dir(self, safe: str) -> bytes | None:
        root = self._dir_path
        assert root is not None
        # The same membership check `_read_from_zip` has always had. The two
        # loaders serve the same route and had different answers to "is this a
        # member?", which is the asymmetry lens 1 F7 is about.
        if safe not in self._names:
            return None
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

    async def is_route_prefix_async(self, segment: str) -> bool:
        """`is_route_prefix`, on the route's own non-blocking path.

        The fallback runs on a cache miss, which is exactly the case where the
        synchronous version would decompress on the event loop -- the one
        thing this route may not do.
        """
        return (await self.read_async(f"{segment}.html") is not None
                or await self.read_async(segment) is not None)


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
            found = await bundle.read_async(candidate)
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
            if await bundle.is_route_prefix_async(first):
                shell = await bundle.read_async("index.html")
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
