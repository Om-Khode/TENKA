# assistant/io/api/routes/files.py
"""Three roots, not a file manager.

Path-keyed: a node's id is its path, so the client's breadcrumb is a split()
and a listing request is the same string it just rendered. Confinement is
enforced in the runtime; this layer turns its refusals into status codes.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..schemas import DeleteRequest, Envelope, RenameRequest
from ..security import require
from ..vault import Capability

router = APIRouter()


def _entry(entry) -> dict:
    return {
        "id": entry.path,
        "name": entry.name,
        "kind": entry.kind,
        "sizeBytes": entry.size_bytes,
        "modifiedAt": entry.modified_at,
        "contentKind": entry.content_kind,
    }


def _fail(exc: Exception) -> HTTPException:
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail="invalid path")
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=403, detail="protected path")
    return HTTPException(status_code=404, detail="not found")


@router.get("/files/roots")
async def list_roots(request: Request,
                     _=Depends(require(Capability.FILES))) -> Envelope:
    roots = await request.app.state.runtime.files.roots()
    return Envelope(data={"roots": roots})


@router.get("/files")
async def list_files(request: Request, path: str = Query(min_length=1, max_length=1_024),
                     _=Depends(require(Capability.FILES))) -> Envelope:
    try:
        entries = await request.app.state.runtime.files.listing(path)
    except (KeyError, ValueError, FileNotFoundError, NotADirectoryError, OSError) as exc:
        raise _fail(exc)
    return Envelope(data={"path": path, "entries": [_entry(e) for e in entries]})


@router.get("/files/content")
async def read_file(request: Request,
                    path: str = Query(min_length=1, max_length=1_024),
                    _=Depends(require(Capability.FILES))) -> Envelope:
    try:
        content = await request.app.state.runtime.files.read(path)
    except (KeyError, ValueError, FileNotFoundError, OSError) as exc:
        raise _fail(exc)
    return Envelope(data={
        "id": content.path,
        "contentKind": content.content_kind,
        "content": content.text,
        "language": content.language,
        "truncated": content.truncated,
    })


@router.post("/files/rename")
async def rename_file(body: RenameRequest, request: Request,
                      _=Depends(require(Capability.FILES))) -> Envelope:
    try:
        entry = await request.app.state.runtime.files.rename(body.path, body.new_name)
    except (KeyError, ValueError, PermissionError, FileNotFoundError, OSError) as exc:
        raise _fail(exc)
    return Envelope(data=_entry(entry))


@router.delete("/files")
async def delete_file(body: DeleteRequest, request: Request,
                      _=Depends(require(Capability.FILES))) -> Envelope:
    try:
        removed = await request.app.state.runtime.files.delete(body.path)
    except (KeyError, ValueError, PermissionError, OSError) as exc:
        raise _fail(exc)
    if not removed:
        raise HTTPException(status_code=404, detail="not found")
    return Envelope(data={"deleted": body.path})
