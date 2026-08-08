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

from ..payloads import (
    EntityPayload, FactPayload, ForgottenPayload, KnowledgeGraphPayload,
    PreferenceChangePayload, PreferenceRecordPayload, PreferencesPayload,
    ProcedureRecordPayload, ProceduresPayload, RelationshipPayload, RemovedPayload,
)
from ..schemas import Envelope
from ..security import require
from ..vault import Capability

router = APIRouter()

Scope = Literal["knowledge", "preferences", "procedures"]

MemoryPayload = KnowledgeGraphPayload | PreferencesPayload | ProceduresPayload


def _entity(entity) -> EntityPayload:
    return EntityPayload(
        id=entity.id,
        type=entity.type,
        canonical_name=entity.canonical_name,
        display_name=entity.display_name,
        properties=entity.properties,
        source=entity.source,
        confidence=entity.confidence,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
        source_turn_id=entity.source_turn_id,
    )


def _fact(fact) -> FactPayload:
    return FactPayload(
        id=fact.id,
        subject_id=fact.subject_id,
        predicate=fact.predicate,
        object=fact.object,
        confidence=fact.confidence,
        source=fact.source,
        event_at=fact.event_at,
        invalid_at=fact.invalid_at,
        expires_at=fact.expires_at,
        verified_at=fact.verified_at,
        created_at=fact.created_at,
        source_turn_id=fact.source_turn_id,
    )


def _relationship(rel) -> RelationshipPayload:
    return RelationshipPayload(
        id=rel.id,
        from_id=rel.from_id,
        to_id=rel.to_id,
        type=rel.type,
        properties=rel.properties,
        confidence=rel.confidence,
        source=rel.source,
        source_turn_id=rel.source_turn_id,
    )


@router.get("/memory/{scope}")
async def list_scope(scope: Scope, request: Request,
                     _=Depends(require(Capability.CHAT))) -> Envelope[MemoryPayload]:
    memory = request.app.state.runtime.memory

    if scope == "knowledge":
        graph = await memory.knowledge()
        return Envelope(data=KnowledgeGraphPayload(
            entities=[_entity(e) for e in graph.entities],
            facts=[_fact(f) for f in graph.facts],
            relationships=[_relationship(r) for r in graph.relationships],
        ))

    if scope == "preferences":
        records = await memory.preferences()
        return Envelope(data=PreferencesPayload(preferences=[
            PreferenceRecordPayload(
                key=record.key,
                value=record.value,
                updated_at=record.updated_at,
                history=[PreferenceChangePayload(value=h.value, changed_at=h.changed_at)
                         for h in record.history],
            )
            for record in records
        ]))

    records = await memory.procedures()
    return Envelope(data=ProceduresPayload(procedures=[
        ProcedureRecordPayload(
            id=record.id,
            name=record.name,
            steps=record.steps,
            taught_at=record.taught_at,
            run_count=record.run_count,
        )
        for record in records
    ]))


@router.delete("/memory/{scope}/{item_id}")
async def forget_item(scope: Scope, item_id: str, request: Request,
                      _=Depends(require(Capability.CHAT))) -> Envelope[ForgottenPayload]:
    removed = await request.app.state.runtime.memory.forget(scope, item_id)
    if not removed:
        raise HTTPException(status_code=404)  # detail is dead on a 404 -- see app.py's handler
    return Envelope(data=ForgottenPayload(forgotten=item_id))


@router.delete("/memory")
async def forget_everything(request: Request,
                            _=Depends(require(Capability.SYSTEM_CONTROL))) -> Envelope[RemovedPayload]:
    removed = await request.app.state.runtime.memory.forget_all()
    return Envelope(data=RemovedPayload(removed=removed))
