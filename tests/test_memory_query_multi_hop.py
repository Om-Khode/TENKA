"""Tests for knowledge-graph Session 4 D wiring — memory_query escalates to
expand_multi_hop when 1-hop KG returned entity hits but hybrid-retrieval produced
nothing to ground a synthesis."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.storage.db import _reset_for_testing, init_db


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def db(tmp_path):
    _reset_for_testing()
    db_obj = init_db(tmp_path / "test.db")
    yield db_obj
    db_obj.close()
    _reset_for_testing()


def test_multi_hop_runs_when_hybrid_empty_but_kg_matched(db, monkeypatch):
    """1-hop KG returns an entity. hybrid-retrieval facts/convos/recordings empty.
    Multi-hop expansion fires; its context block is added to the synth
    prompt under the 'Knowledge graph (deep)' header."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()
    repo = kg_mod._get_repo()
    aid, _ = repo.upsert_entity("person", "Aanya", source="user_msg")

    from assistant.actions import memory_search

    monkeypatch.setattr(
        "assistant.memory.hybrid_search_facts", lambda q, limit=10: []
    )
    monkeypatch.setattr(
        "assistant.memory.hybrid_search_conversations", lambda q, limit=5: []
    )
    monkeypatch.setattr(
        "assistant.memory.search_recording_sessions", lambda q, limit=3: []
    )

    # 1-hop KG search returns Aanya — the seed for multi-hop.
    monkeypatch.setattr(
        kg_mod, "search_entities",
        lambda q: [{"id": aid, "type": "person", "display_name": "Aanya"}],
    )

    multi_hop_called: dict = {}

    async def fake_multi_hop(question, seed_ids, **kw):
        multi_hop_called["question"] = question
        multi_hop_called["seed_ids"] = seed_ids
        return {
            "context_block": "[KNOWLEDGE]\nAanya (person): deep-context-result.",
            "visited_ids": seed_ids,
            "iterations": 2,
            "stopped_reason": "sufficient",
        }

    monkeypatch.setattr(kg_mod, "expand_multi_hop", fake_multi_hop)

    captured: dict = {}

    async def fake_synth(prompt, **kw):
        captured["prompt"] = prompt
        return "answer"

    monkeypatch.setattr("assistant.llm.contracts.ask_for_synthesis", fake_synth)

    _run(memory_search.handle_memory_query({"query": "what about Aanya"}, ""))

    assert multi_hop_called["seed_ids"] == [aid]
    assert "Knowledge graph (deep)" in captured["prompt"]
    assert "deep-context-result" in captured["prompt"]


def test_multi_hop_skipped_when_hybrid_facts_present(db, monkeypatch):
    """When hybrid-retrieval returned facts, the cheap path is enough — multi-hop
    must NOT fire, to keep this branch zero-LLM."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()

    from assistant.actions import memory_search

    monkeypatch.setattr(
        "assistant.memory.hybrid_search_facts",
        lambda q, limit=10: [{"key": "fav_food", "value": "biryani"}],
    )
    monkeypatch.setattr(
        "assistant.memory.hybrid_search_conversations", lambda q, limit=5: []
    )
    monkeypatch.setattr(
        "assistant.memory.search_recording_sessions", lambda q, limit=3: []
    )
    monkeypatch.setattr(kg_mod, "search_entities", lambda q: [])

    called = {"n": 0}

    async def fake_multi_hop(q, seeds, **kw):
        called["n"] += 1
        return {"context_block": "", "visited_ids": [], "iterations": 0,
                "stopped_reason": "no_seeds"}

    monkeypatch.setattr(kg_mod, "expand_multi_hop", fake_multi_hop)

    async def fake_synth(prompt, **kw):
        return "answer"

    monkeypatch.setattr("assistant.llm.contracts.ask_for_synthesis", fake_synth)
    _run(memory_search.handle_memory_query({"query": "favourite food"}, ""))

    assert called["n"] == 0
