# assistant/io/api/routes/memory.py
"""What she knows, made readable, and forgettable.

Knowledge is served as a graph -- entities, facts, relationships -- because the
page renders supersession, provenance and an ego graph from exactly those three
tables. Keys are camelCase here and only here: this is the boundary between
Python's naming and the client's, and doing it in one place beats a mapping
layer on the far side.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request

from ..schemas import Envelope
from ..security import require
from ..vault import Capability

router = APIRouter()

Scope = Literal["knowledge", "preferences", "procedures"]


def _entity(entity) -> dict:
    return {
        "id": entity.id,
        "type": entity.type,
        "canonicalName": entity.canonical_name,
        "displayName": entity.display_name,
        "properties": entity.properties,
        "source": entity.source,
        "confidence": entity.confidence,
        "createdAt": entity.created_at,
        "updatedAt": entity.updated_at,
        "sourceTurnId": entity.source_turn_id,
    }


def _fact(fact) -> dict:
    return {
        "id": fact.id,
        "subjectId": fact.subject_id,
        "predicate": fact.predicate,
        "object": fact.object,
        "confidence": fact.confidence,
        "source": fact.source,
        "eventAt": fact.event_at,
        "invalidAt": fact.invalid_at,
        "expiresAt": fact.expires_at,
        "verifiedAt": fact.verified_at,
        "createdAt": fact.created_at,
        "sourceTurnId": fact.source_turn_id,
    }


def _relationship(rel) -> dict:
    return {
        "id": rel.id,
        "fromId": rel.from_id,
        "toId": rel.to_id,
        "type": rel.type,
        "properties": rel.properties,
        "confidence": rel.confidence,
        "source": rel.source,
        "sourceTurnId": rel.source_turn_id,
    }


@router.get("/memory/{scope}")
async def list_scope(scope: Scope, request: Request,
                     _=Depends(require(Capability.CHAT))) -> Envelope:
    memory = request.app.state.runtime.memory

    if scope == "knowledge":
        graph = await memory.knowledge()
        return Envelope(data={
            "entities": [_entity(e) for e in graph.entities],
            "facts": [_fact(f) for f in graph.facts],
            "relationships": [_relationship(r) for r in graph.relationships],
        })

    if scope == "preferences":
        records = await memory.preferences()
        return Envelope(data={"preferences": [
            {
                "key": record.key,
                "value": record.value,
                "updatedAt": record.updated_at,
                "history": [{"value": h.value, "changedAt": h.changed_at}
                            for h in record.history],
            }
            for record in records
        ]})

    records = await memory.procedures()
    return Envelope(data={"procedures": [
        {
            "id": record.id,
            "name": record.name,
            "steps": record.steps,
            "taughtAt": record.taught_at,
            "runCount": record.run_count,
        }
        for record in records
    ]})


@router.delete("/memory/{scope}/{item_id}")
async def forget_item(scope: Scope, item_id: str, request: Request,
                      _=Depends(require(Capability.CHAT))) -> Envelope:
    removed = await request.app.state.runtime.memory.forget(scope, item_id)
    if not removed:
        raise HTTPException(status_code=404, detail="not found")
    return Envelope(data={"forgotten": item_id})


@router.delete("/memory")
async def forget_everything(request: Request,
                            _=Depends(require(Capability.SYSTEM_CONTROL))) -> Envelope:
    removed = await request.app.state.runtime.memory.forget_all()
    return Envelope(data={"removed": removed})
