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


# Three explicit routes, not one `/memory/{scope}` dispatching on a path
# parameter (review finding, 2026-08-08): the response shape is fully
# determined by `scope`, which the caller already supplied in the URL, so a
# single parameterised route could only describe its response as a union of
# all three payloads -- `oneOf` in the schema, with no discriminator, and a
# generated client left to duck-type which one it got. Three routes give a
# clean 1:1 type per operation instead. The URLs a client calls are
# unchanged -- /v1/memory/knowledge, /v1/memory/preferences,
# /v1/memory/procedures were always the only three paths this ever served;
# only the routing (one dynamic segment vs three static ones) and the
# handler split, not what a caller sends or receives for any of them.
# DELETE /memory/{scope}/{item_id} stays parameterised below: it has one
# response shape regardless of scope, so a union was never the issue there.
@router.get("/memory/knowledge")
async def get_knowledge(request: Request,
                        _=Depends(require(Capability.CHAT))) -> Envelope[KnowledgeGraphPayload]:
    graph = await request.app.state.runtime.memory.knowledge()
    return Envelope(data=KnowledgeGraphPayload(
        entities=[_entity(e) for e in graph.entities],
        facts=[_fact(f) for f in graph.facts],
        relationships=[_relationship(r) for r in graph.relationships],
    ))


@router.get("/memory/preferences")
async def get_preferences(request: Request,
                          _=Depends(require(Capability.CHAT))) -> Envelope[PreferencesPayload]:
    records = await request.app.state.runtime.memory.preferences()
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


@router.get("/memory/procedures")
async def get_procedures(request: Request,
                         _=Depends(require(Capability.CHAT))) -> Envelope[ProceduresPayload]:
    records = await request.app.state.runtime.memory.procedures()
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
                      # CHAT_SEND, not CHAT. The earlier ruling here -- "a
                      # phone paired for conversation can delete one thing it
                      # was told about", while the wipe below demands
                      # SYSTEM_CONTROL -- predates the CHAT/CHAT_SEND split,
                      # and CHAT has since come to mean "may read" and
                      # nothing more. CHAT_SEND keeps that ruling's intent
                      # exactly: forgetting one item is the same class of act
                      # as saying "forget that" in a turn, which is precisely
                      # what CHAT_SEND authorises. It is deliberately *not*
                      # raised to SYSTEM_CONTROL -- that would collapse the
                      # distinction between correcting one memory and
                      # erasing all of them, which the wipe's separate grant
                      # exists to preserve.
                      _=Depends(require(Capability.CHAT_SEND))) -> Envelope[ForgottenPayload]:
    removed = await request.app.state.runtime.memory.forget(scope, item_id)
    if not removed:
        raise HTTPException(status_code=404)  # detail is dead on a 404 -- see app.py's handler
    return Envelope(data=ForgottenPayload(forgotten=item_id))


@router.delete("/memory")
async def forget_everything(request: Request,
                            _=Depends(require(Capability.SYSTEM_CONTROL))) -> Envelope[RemovedPayload]:
    removed = await request.app.state.runtime.memory.forget_all()
    return Envelope(data=RemovedPayload(removed=removed))
