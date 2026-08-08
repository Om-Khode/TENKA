# assistant/io/api/routes/files.py
"""Three roots, not a file manager.

Path-keyed: a node's id is its path, so the client's breadcrumb is a split()
and a listing request is the same string it just rendered. Confinement is
enforced in the runtime; this layer turns its refusals into status codes.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..errors import to_http_exception
from ..payloads import DeletedPayload, FileContentPayload, FileEntryPayload, FilesListingPayload, RootsPayload
from ..schemas import DeleteRequest, Envelope, RenameRequest
from ..security import require
from ..vault import Capability

router = APIRouter()


def _entry(entry) -> FileEntryPayload:
    return FileEntryPayload(
        id=entry.path,
        name=entry.name,
        kind=entry.kind,
        size_bytes=entry.size_bytes,
        modified_at=entry.modified_at,
        content_kind=entry.content_kind,
    )


# `_fail` used to be this module's own private copy of the exception ->
# HTTPException mapping; it is now the shared one in errors.py, kept as a
# local alias so every call site below stays unchanged.
_fail = to_http_exception


@router.get("/files/roots")
async def list_roots(request: Request,
                     _=Depends(require(Capability.FILES))) -> Envelope[RootsPayload]:
    try:
        roots = await request.app.state.runtime.files.roots()
    except OSError as exc:
        raise _fail(exc)
    return Envelope(data=RootsPayload(roots=roots))


@router.get("/files")
async def list_files(request: Request, path: str = Query(min_length=1, max_length=1_024),
                     _=Depends(require(Capability.FILES))) -> Envelope[FilesListingPayload]:
    try:
        entries = await request.app.state.runtime.files.listing(path)
    except (KeyError, ValueError, FileNotFoundError, NotADirectoryError, OSError) as exc:
        raise _fail(exc)
    return Envelope(data=FilesListingPayload(path=path, entries=[_entry(e) for e in entries]))


@router.get("/files/content")
async def read_file(request: Request,
                    path: str = Query(min_length=1, max_length=1_024),
                    _=Depends(require(Capability.FILES))) -> Envelope[FileContentPayload]:
    try:
        content = await request.app.state.runtime.files.read(path)
    except (KeyError, ValueError, FileNotFoundError, OSError) as exc:
        raise _fail(exc)
    return Envelope(data=FileContentPayload(
        id=content.path,
        content_kind=content.content_kind,
        content=content.text,
        language=content.language,
        truncated=content.truncated,
    ))


@router.post("/files/rename")
async def rename_file(body: RenameRequest, request: Request,
                      _=Depends(require(Capability.FILES))) -> Envelope[FileEntryPayload]:
    try:
        entry = await request.app.state.runtime.files.rename(body.path, body.new_name)
    except (KeyError, ValueError, PermissionError, FileNotFoundError, OSError) as exc:
        raise _fail(exc)
    return Envelope(data=_entry(entry))


@router.delete("/files")
async def delete_file(body: DeleteRequest, request: Request,
                      _=Depends(require(Capability.FILES))) -> Envelope[DeletedPayload]:
    try:
        removed = await request.app.state.runtime.files.delete(body.path)
    except (KeyError, ValueError, PermissionError, OSError) as exc:
        raise _fail(exc)
    if not removed:
        raise HTTPException(status_code=404)  # detail is dead here; see _fail()
    return Envelope(data=DeletedPayload(deleted=body.path))
