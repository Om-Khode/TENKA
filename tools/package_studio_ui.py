"""Turn Studio's static export into the single zip the daemon serves.

    npm run build:bundled                       # in D:\\Code\\tenka-studio
    py -3.11 tools/package_studio_ui.py ../tenka-studio/out

The output is `assistant/io/api/studio_ui.zip`, tracked in git, and read by
`UiBundle.open()` when no `studio_ui_path` names a local export.

**Run this on a release, not on a UI tweak.** A zip is an opaque blob to git:
it gets no delta compression, so every regeneration adds its *full* size --
several MiB -- to the repository forever, whether one pixel moved or the whole
app was rewritten. Day-to-day Studio work does not need it at all: point the
`studio_ui_path` setting at a local `out/` directory and the daemon serves that
instead, live, with no packaging step in the loop. Do not wire this into a
build script.

    py -3.11 tools/package_studio_ui.py ../tenka-studio/out --marker

writes just the marker into that directory, which is what makes the override
work -- `UiBundle` recognises a directory as a bundle by the marker, and
`next build` has no reason to write one.

Three things the zip must carry, and why each is checked rather than assumed:

* **The marker** (`MARKER_NAME`, imported -- never spelled here). A bundle
  under any other name is not a broken bundle to the daemon, it is *no bundle
  at all*: `_from_zip` logs one WARNING, returns `None`, and the app comes up
  with no `/` route. The name is imported from the module that reads it so the
  two cannot drift, and the finished archive is reopened and checked.

* **The contract hash**, over the same canonical serialisation
  `assistant.io.api.ui.contract_hash` uses -- `sort_keys=True`, no whitespace.
  Both sides hash the schema *object*, never the bytes of a schema file;
  hashing bytes would turn a guard against a stale UI into a whitespace
  detector.

* **A same-origin API base.** Studio inlines `NEXT_PUBLIC_STUDIO_API_BASE` at
  build time and falls back to an absolute `http://127.0.0.1:8787`. A bundle
  carrying that default works on loopback and fails completely the moment the
  daemon serves it from a tunnel: every request becomes cross-origin, which the
  daemon's own `connect-src 'self'` refuses before mixed content or Private
  Network Access are even consulted. It presents as a dead WebSocket and is
  not one. `npm run build:bundled` sets the base to `/`; this asserts it,
  because "we remembered to use the right script" is not a property anyone can
  verify six months later.

The secret scan reads members **out of the archive**, not out of the export
directory, and never shells out. `out/` is gitignored, so `git grep` over it
searches nothing and reports clean for any content whatsoever -- a scan that
cannot fail is worse than no scan, because it is believed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.io.api.ui import (                             # noqa: E402
    MARKER_NAME,
    MAX_MEMBER_BYTES,
    UI_MANIFEST_VERSION,
    is_hidden_member,
    normalise_member,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = _REPO_ROOT / "assistant" / "io" / "api" / "studio_ui.zip"

# Every member gets this stamp instead of its mtime. A zip stores a per-entry
# timestamp, so packaging the same export twice would otherwise produce two
# different files -- and a tracked binary that changes on every run is one
# nobody can review. 1980-01-01 is the zip format's own epoch; anything earlier
# is unrepresentable.
_FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

# Fixed rather than default, for the same reason: the default has changed
# between zlib builds, and a compression level is part of the output bytes.
_COMPRESS_LEVEL = 9


# ─── the scan ────────────────────────────────────────────────────────────
# Byte patterns, not decoded text: a third of the export is fonts and images,
# and decoding them to search for an ASCII string is cost with no coverage.
#
# Each rule is a *shape*, chosen so that the legitimate code that mentions the
# same word does not trip it. Studio really does build an `Authorization:
# Bearer` header for the `next dev` path, so a bare `Bearer` match would refuse
# every export ever produced -- and a check that always fails gets deleted
# rather than fixed. Minifiers keep the literal as `` `Bearer ${t}` `` or
# `"Bearer ".concat(t)`, so requiring a token *character* right after the space
# separates "code that will attach a credential" from "a credential".
_SECRET_RULES: tuple[tuple[str, re.Pattern[bytes], str], ...] = (
    ("bearer-token", re.compile(rb"Bearer\s+[A-Za-z0-9._\-]+"),
     "a literal bearer token"),
    ("instance-secret", re.compile(rb"TENKA_SECRET"),
     "the instance secret's environment name"),
    ("access-token", re.compile(rb"access_token"),
     "the credential-in-a-URL parameter Task 5 removed from both sides"),
)

# An absolute origin naming this machine. Baked into the bundle it is the
# single most likely way the first live test fails -- see the module docstring.
# Written as a shape rather than as Studio's exact `DEFAULT_BASE` string so a
# port change, an IPv6 spelling or a stray `localhost` is caught too.
_ABSOLUTE_LOCAL_BASE = re.compile(
    rb"https?://(?:127(?:\.[0-9]{1,3}){3}|localhost|\[::1\])(?::[0-9]+)?")

_BASE_RULE = ("absolute-api-base", _ABSOLUTE_LOCAL_BASE,
              "an absolute loopback API base; build with "
              "`npm run build:bundled`, which sets NEXT_PUBLIC_STUDIO_API_BASE=/")


# ─── the API base, asserted rather than assumed ──────────────────────────
# The rule above only proves the *loopback* default is absent. It says nothing
# about a build made against, say, a Vercel origin -- which would be just as
# cross-origin and just as blocked. So the base is also read positively, out of
# the shipped JS, and required to be relative.
#
# The anchor is `apiBase()`'s own trailing-slash strip, which survives
# minification as `"<base>".replace(/\/+$/,"")` -- Studio deliberately writes
# that function so the whole thing folds to one string literal (see the comment
# on `apiBase` in `services/http.ts`). Reading the literal is the only way to
# know what a built bundle will actually talk to; the alternative is a build
# script writing down what it *intended*, which is provenance, not evidence.
#
# **It fails closed on no match.** If a Next upgrade reshapes the minified
# output, packaging stops and says so, rather than quietly becoming a check
# that can no longer fail -- which is the exact trap the `git grep` note in the
# docstring is about.
_INLINED_BASE = re.compile(rb"""return\s*(["'])(?P<base>[^"']*)\1\.replace\(/\\/\+\$/""")

# A scheme, or a protocol-relative `//host`. Either one leaves the daemon's
# origin, and `connect-src 'self'` refuses it.
_ABSOLUTE_URL = re.compile(rb"\A(?:[A-Za-z][A-Za-z0-9+.\-]*:|//)")


def _excerpt(body: bytes, match: re.Match[bytes]) -> str:
    """Enough context to find the line, bounded so a minified 800 KB chunk on
    one line does not become the error message."""
    start = max(match.start() - 40, 0)
    end = min(match.end() + 40, len(body))
    return body[start:end].decode("utf-8", "replace")


def _scan(member: str, body: bytes) -> None:
    for name, pattern, why in (*_SECRET_RULES, _BASE_RULE):
        match = pattern.search(body)
        if match is not None:
            raise ValueError(
                f"{member} contains {why} [{name}]: ...{_excerpt(body, match)}...")


# ─── reading the export ──────────────────────────────────────────────────
def _collect(source: Path) -> list[tuple[str, bytes]]:
    """Every file under `source`, as sorted `(member name, bytes)` pairs.

    Sorted because zip entry order is part of the output bytes and
    `Path.rglob` order is the filesystem's, not a promise.
    """
    if not source.is_dir():
        raise ValueError(f"{source} is not a directory")
    index = source / "index.html"
    if not index.is_file():
        # The one file the daemon's root route and its whole SPA fallback are
        # built on. An export missing it is a failed `next build` that left a
        # populated directory behind, which is exactly the shape that packages
        # cleanly and then serves 404 to every visitor.
        raise ValueError(f"{source} has no index.html; it is not a Studio export")

    members: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        member = path.relative_to(source).as_posix()
        # Hidden entries are skipped, not refused, and the distinction is the
        # whole of this gate's usefulness.
        #
        # `rglob("*")` is a faithful walk of a developer's working directory,
        # so it picks up `.env`, `.git/`, an editor's swap file, and -- the one
        # that actually shipped -- the `.tenka-ui.json` that `--marker` wrote
        # into that same directory earlier. The last of those is why this is a
        # skip: a second copy of the marker landed in the archive alongside the
        # one `package()` writes, `zipfile` resolves a repeated name to the
        # last entry, and the daemon has been reading the stale one ever since.
        # Refusing outright would turn the documented `--marker` workflow into
        # a packaging failure; skipping leaves it working and ships neither
        # copy but the computed one.
        #
        # It is also the honest reading of the daemon's own rule: since
        # `is_hidden_member` a dot member is not a member at all, so there is
        # nothing here to fail about. Names that are malformed for other
        # reasons still raise below -- those are build accidents, and a
        # 404-in-production is worth converting into a packaging error at a
        # point where somebody is looking.
        if is_hidden_member(member):
            continue
        if normalise_member(member) != member:
            raise ValueError(f"{member} is not a name the daemon can serve")
        # Two source paths cannot normally fold to one member name, but a
        # case-insensitive filesystem and a future normalisation step both
        # could -- and the daemon refuses a bundle with a repeated name
        # outright, so shipping one would take the whole UI dark.
        if member in seen:
            raise ValueError(
                f"{member} appears twice; the daemon refuses an archive that "
                f"stores one name more than once, because what it serves for "
                f"that name is whichever entry came last")
        seen.add(member)
        body = path.read_bytes()
        if len(body) > MAX_MEMBER_BYTES:
            raise ValueError(
                f"{member} is {len(body)} bytes; the daemon refuses anything "
                f"over {MAX_MEMBER_BYTES}, so packaging it would ship a 404")
        members.append((member, body))
    return members


# ─── the contract ────────────────────────────────────────────────────────
def live_contract_hash() -> str:
    """The hash of the API this checkout serves.

    Imported inside the function rather than at module scope: it pulls in the
    whole app, its fake runtime and FastAPI's schema generator, and every
    caller that passes an explicit `contract` wants none of that.
    """
    import tempfile

    from assistant.io.api.app import create_app
    from assistant.io.api.ui import contract_hash
    from assistant.io.api.vault import TokenVault
    from tests.fakes.studio_runtime import build_fake_runtime

    app = create_app(build_fake_runtime(),
                     TokenVault(Path(tempfile.mkdtemp()) / "vault"),
                     origins=["http://localhost:3000"])
    return contract_hash(app)


# ─── packaging ───────────────────────────────────────────────────────────
def package(source: Path | str, destination: Path | str, *,
            built_at: str | None = None, contract: str | None = None) -> Path:
    """Write `source` to `destination` as a bundle the daemon will accept.

    `built_at` and `contract` exist so a test can pin them; leave both unset
    and the marker records now, against this checkout's live API.
    """
    source = Path(source)
    destination = Path(destination)
    members = _collect(source)
    marker = json.dumps({
        "version": UI_MANIFEST_VERSION,
        "contract": live_contract_hash() if contract is None else contract,
        "builtAt": (datetime.now(timezone.utc).isoformat()
                    if built_at is None else built_at),
    }, sort_keys=True, separators=(",", ":"))

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=_COMPRESS_LEVEL) as archive:
        _write(archive, MARKER_NAME, marker.encode("utf-8"))
        for member, body in members:
            _write(archive, member, body)

    _verify(destination)
    return destination


def write_dev_marker(source: Path | str, *, contract: str | None = None) -> Path:
    """Write the marker into an export directory, and package nothing.

    The developer override (`studio_ui_path`) is worth having and does not work
    without this. `UiBundle._from_dir` requires the marker -- it is how a
    directory is recognised as a bundle at all -- and `next build` has no
    reason to write one, so a freshly built `out/` loses silently to the
    vendored zip and the developer sees a stale UI with no explanation. This
    costs nothing to offer: `out/` is gitignored in Studio, so the marker never
    leaves the machine that wrote it.

    Nothing here is scanned. A dev directory is not an artefact anyone ships,
    and the checks in `package()` are about what leaves this machine.
    """
    source = Path(source)
    if not (source / "index.html").is_file():
        raise ValueError(f"{source} has no index.html; it is not a Studio export")
    marker = source / MARKER_NAME
    marker.write_text(json.dumps({
        "version": UI_MANIFEST_VERSION,
        "contract": live_contract_hash() if contract is None else contract,
        "builtAt": datetime.now(timezone.utc).isoformat(),
    }, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return marker


def _write(archive: zipfile.ZipFile, member: str, body: bytes) -> None:
    info = zipfile.ZipInfo(member, date_time=_FIXED_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    # Fixed too. `writestr` derives these from the host platform otherwise, so
    # a bundle built on Windows and one built on Linux would differ in bytes
    # while being identical in content.
    info.create_system = 0
    info.external_attr = 0o644 << 16
    archive.writestr(info, body, compresslevel=_COMPRESS_LEVEL)


def _verify(destination: Path) -> None:
    """Read the finished archive back the way the daemon will.

    Not belt and braces: the scan is the only place the bundle's *contents* are
    ever checked, and doing it here means it runs against the bytes that
    shipped rather than against the directory they came from. The marker check
    is here for the same reason -- a bundle under the wrong name reads to the
    daemon as no bundle at all, and that failure is one WARNING deep.
    """
    found_base = False
    with zipfile.ZipFile(destination) as archive:
        stored = archive.namelist()
        names = set(stored)
        # Read off the finished archive rather than trusted from `_collect`,
        # because this is the check the daemon's `_from_zip` also makes and a
        # bundle that fails it mounts as no bundle at all. A duplicate is how
        # the shipped artifact came to carry two markers with different
        # `builtAt` stamps, and `zipfile` answers `read()` with the last one --
        # so the contract guard was measuring a document nobody wrote on
        # purpose.
        if len(stored) != len(names):
            duplicated = sorted({n for n in stored if stored.count(n) > 1})
            raise ValueError(
                f"the archive stores these names more than once: {duplicated}")
        # `MARKER_NAME` is the one dot member that belongs here: it is written
        # by `package()` and read by literal name, never by `normalise_member`.
        hidden = sorted(n for n in names
                        if n != MARKER_NAME and is_hidden_member(n))
        if hidden:
            raise ValueError(
                f"the archive carries hidden members the daemon will not "
                f"serve: {hidden}")
        if MARKER_NAME not in names:
            raise ValueError(f"the archive carries no {MARKER_NAME}")
        if "index.html" not in names:
            raise ValueError("the archive carries no index.html")
        for member in sorted(names):
            if member == MARKER_NAME:
                # The one member that legitimately contains a hash of the API
                # and is withheld from HTTP by name. Scanning it would only
                # ever produce a false positive.
                continue
            body = archive.read(member)
            _scan(member, body)
            found_base |= _check_api_base(member, body)

    if not found_base:
        raise ValueError(
            "no API base literal was found in the export. Either this is not a "
            "Studio build, or `apiBase()`/the minifier changed shape and "
            "`_INLINED_BASE` needs updating -- see the comment on it. This is "
            "deliberately fatal: a base check that finds nothing is a check "
            "that can no longer fail.")


def _check_api_base(member: str, body: bytes) -> bool:
    """Whether this member carried an API base, refusing any absolute one."""
    seen = False
    for match in _INLINED_BASE.finditer(body):
        seen = True
        base = match.group("base")
        if _ABSOLUTE_URL.match(base):
            raise ValueError(
                f"{member} was built against the absolute API base "
                f"{base.decode('utf-8', 'replace')!r}. A bundle the daemon "
                f"serves must use a relative base, or every request from a "
                f"tunnel origin becomes cross-origin and `connect-src 'self'` "
                f"refuses it. Rebuild with `npm run build:bundled`.")
    return seen


# ─── cli ─────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path,
                        help="Studio's export directory (`out/`)")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_DESTINATION,
                        help=f"where to write the zip (default: {DEFAULT_DESTINATION})")
    parser.add_argument("--marker", action="store_true",
                        help="write the marker into SOURCE and package nothing, "
                             "so `studio_ui_path` can serve that directory live")
    args = parser.parse_args(argv)

    if args.marker:
        try:
            marker = write_dev_marker(args.source)
        except ValueError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {marker} — point studio_ui_path at {args.source}")
        return 0

    try:
        written = package(args.source, args.output)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1

    with zipfile.ZipFile(written) as archive:
        count = len(archive.namelist())
        marker = json.loads(archive.read(MARKER_NAME))
    print(f"wrote {written} — {count} members, "
          f"{written.stat().st_size / 1024 / 1024:.2f} MiB, "
          f"contract {marker['contract'][:12]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
