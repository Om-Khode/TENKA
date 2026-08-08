# assistant/io/api/errors.py
"""One exception -> HTTPException mapping, shared by every route module and
the app-wide catch-all in app.py.

Before this module existed, only files.py mapped a runtime exception onto a
status code (its own private `_fail`); every other route module either
caught nothing (letting FastAPI's default handling turn it into a bare 500
with no audit trail -- see app.py's `audit_and_tag`) or duplicated the same
three `isinstance` checks locally. One mapping, used everywhere, means a new
route inherits sane behaviour by construction instead of by remembering to
copy it.
"""
from __future__ import annotations

from fastapi import HTTPException


def to_http_exception(exc: Exception) -> HTTPException:
    """Map a runtime/domain exception to the HTTPException it should become.

    Ordering matters: PermissionError and LookupError-family checks come
    before the bare ValueError/RuntimeError/OSError catch-alls so a more
    specific exception never falls through to a coarser bucket.
    """
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=403, detail="protected path")
    if isinstance(exc, (KeyError, FileNotFoundError, NotADirectoryError)):
        # detail is dead on a 404: app.py's status-code handler answers a
        # fixed body regardless of what is passed here.
        return HTTPException(status_code=404)
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail="invalid request")
    if isinstance(exc, RuntimeError):
        # A precondition the caller could plausibly resolve (e.g. "the
        # backup key is not unlocked yet") -- not "you're not allowed" (403),
        # not "malformed" (400), not "gone" (404). 409 is the closest honest
        # fit among status codes this app already uses elsewhere (chat's
        # busy-turn response).
        return HTTPException(status_code=409, detail="precondition failed")
    if isinstance(exc, OSError):
        # Matches files.py's pre-existing fallthrough for a filesystem
        # operation that raised for a reason that is neither a bad path
        # (ValueError) nor a protected one (PermissionError).
        return HTTPException(status_code=404)
    return HTTPException(status_code=500, detail="internal error")
