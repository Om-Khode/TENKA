"""Tests for knowledge-graph Session 3 D — multi-hop COT expansion.

Covers both:
- assistant.llm.contracts.ask_for_kg_followup (the validator contract)
- assistant.knowledge_graph.expand_multi_hop  (the loop helper)

The contract is exercised against mocked get_llm_response. The helper is
exercised against a real in-memory KG with ask_for_kg_followup itself
mocked, so test cases stay deterministic and never hit a real LLM.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

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


# ─── ask_for_kg_followup contract ──────────────────────────────────────────


class _FakeLLMResult:
    def __init__(self, text: str):
        self.text = text


def _mock_llm(reply: str):
    async def _fake(*args, **kwargs):
        return _FakeLLMResult(reply)
    return _fake


def test_followup_sufficient_true_parsed():
    from assistant.llm import contracts
    with patch.object(
        contracts, "get_llm_response",
        _mock_llm('{"sufficient": true, "follow_up": null}'),
    ):
        out = _run(contracts.ask_for_kg_followup("q", "ctx"))
    assert out == {"sufficient": True, "follow_up": None}


def test_followup_returns_named_entity():
    from assistant.llm import contracts
    with patch.object(
        contracts, "get_llm_response",
        _mock_llm('{"sufficient": false, "follow_up": "Razorpay"}'),
    ):
        out = _run(contracts.ask_for_kg_followup("q", "ctx"))
    assert out == {"sufficient": False, "follow_up": "Razorpay"}


def test_followup_empty_context_returns_stop():
    """Empty inputs short-circuit to graceful stop — no LLM call."""
    from assistant.llm import contracts
    out = _run(contracts.ask_for_kg_followup("q", ""))
    assert out == {"sufficient": True, "follow_up": None}


def test_followup_malformed_json_returns_stop():
    from assistant.llm import contracts
    with patch.object(contracts, "get_llm_response", _mock_llm("not json at all")):
        out = _run(contracts.ask_for_kg_followup("q", "ctx"))
    assert out == {"sufficient": True, "follow_up": None}


def test_followup_strips_code_fences():
    from assistant.llm import contracts
    fenced = '```json\n{"sufficient": false, "follow_up": "Aanya"}\n```'
    with patch.object(contracts, "get_llm_response", _mock_llm(fenced)):
        out = _run(contracts.ask_for_kg_followup("q", "ctx"))
    assert out == {"sufficient": False, "follow_up": "Aanya"}


def test_followup_incoherent_sets_sufficient():
    """sufficient=false but no follow_up name → treat as stop (no loop spin)."""
    from assistant.llm import contracts
    with patch.object(
        contracts, "get_llm_response",
        _mock_llm('{"sufficient": false, "follow_up": null}'),
    ):
        out = _run(contracts.ask_for_kg_followup("q", "ctx"))
    assert out == {"sufficient": True, "follow_up": None}


# ─── expand_multi_hop loop ──────────────────────────────────────────────────


def _noop_loader():
    """Returns a callable returning an embed model stub. Cosine match is
    not exercised by these tests — every seed is a fresh (type, name)."""
    class _ZeroModel:
        def encode(self, texts, normalize_embeddings=True):
            import numpy as np
            if isinstance(texts, str):
                return np.zeros(3, dtype="float32")
            return np.zeros((len(texts), 3), dtype="float32")
    return lambda: _ZeroModel()


def _seed_entities(db_obj):
    """Insert two linked entities and a fact, return their ids. Uses the
    same singleton-shaped wiring as ingest_turn — go through the repo on
    the KG facade so init_kg()'s singleton sees the seeded rows."""
    from assistant.storage.repos.knowledge_graph import KnowledgeGraphRepo
    repo = KnowledgeGraphRepo(db_obj, embed_model_loader=_noop_loader())
    aanya_id, _ = repo.upsert_entity("person", "Aanya", source="user_msg")
    razorpay_id, _ = repo.upsert_entity("project", "Razorpay", source="user_msg")
    repo.add_fact(aanya_id, "works_at", "Razorpay", source="user_msg")
    repo.add_relationship(aanya_id, razorpay_id, "related_to", source="user_msg")
    return aanya_id, razorpay_id


def test_multi_hop_no_seeds_returns_empty(db):
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()
    out = _run(kg_mod.expand_multi_hop("q", []))
    assert out["stopped_reason"] == "no_seeds"
    assert out["iterations"] == 0
    assert out["context_block"] == ""


def test_multi_hop_sufficient_first_hop(db):
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()
    aanya_id, _ = _seed_entities(db)

    async def fake_verdict(question, context):
        return {"sufficient": True, "follow_up": None}

    with patch.object(kg_mod, "ask_for_kg_followup", fake_verdict):
        out = _run(kg_mod.expand_multi_hop("Who is Aanya?", [aanya_id]))

    assert out["stopped_reason"] == "sufficient"
    assert out["iterations"] == 1
    assert out["visited_ids"] == [aanya_id]
    assert out["context_block"].startswith("[KNOWLEDGE]")
    assert "Aanya" in out["context_block"]


def test_multi_hop_follow_up_resolves_and_completes(db):
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()
    aanya_id, razorpay_id = _seed_entities(db)

    call_log: list[str] = []

    async def fake_verdict(question, context):
        call_log.append(context)
        if len(call_log) == 1:
            return {"sufficient": False, "follow_up": "Razorpay"}
        return {"sufficient": True, "follow_up": None}

    with patch.object(kg_mod, "ask_for_kg_followup", fake_verdict):
        out = _run(kg_mod.expand_multi_hop("What does Aanya do?", [aanya_id]))

    assert out["stopped_reason"] == "sufficient"
    assert out["iterations"] == 2
    assert aanya_id in out["visited_ids"]
    assert razorpay_id in out["visited_ids"]


def test_multi_hop_unresolvable_follow_up(db):
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()
    aanya_id, _ = _seed_entities(db)

    async def fake_verdict(question, context):
        return {"sufficient": False, "follow_up": "NeverHeardOfThisOne"}

    with patch.object(kg_mod, "ask_for_kg_followup", fake_verdict):
        out = _run(kg_mod.expand_multi_hop("q", [aanya_id]))

    assert out["stopped_reason"] == "unresolvable"
    assert out["iterations"] == 1
    assert out["visited_ids"] == [aanya_id]


def test_multi_hop_max_iter_cap(db):
    """LLM never says sufficient AND each round resolves to a fresh entity:
    loop must terminate at max_iter rather than 'unresolvable'.

    Needs at least max_iter+1 distinct named entities so the final
    iteration's follow-up still resolves to something unseen — otherwise
    the loop short-circuits as 'unresolvable' when the named entity is
    already visited."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()
    aanya_id, razorpay_id = _seed_entities(db)
    # Seed a third entity so iteration 2 has a fresh next_frontier.
    from assistant.storage.repos.knowledge_graph import KnowledgeGraphRepo
    repo = KnowledgeGraphRepo(db, embed_model_loader=_noop_loader())
    rohan_id, _ = repo.upsert_entity("person", "Rohan", source="user_msg")

    call_log: list[str] = []

    async def fake_verdict(question, context):
        # Round 1 names Razorpay; round 2 names Rohan; both resolve to
        # fresh entities so the loop never breaks early.
        call_log.append(context)
        if len(call_log) == 1:
            return {"sufficient": False, "follow_up": "Razorpay"}
        return {"sufficient": False, "follow_up": "Rohan"}

    with patch.object(kg_mod, "ask_for_kg_followup", fake_verdict):
        out = _run(kg_mod.expand_multi_hop("q", [aanya_id], max_iter=2))

    assert out["iterations"] == 2
    assert out["stopped_reason"] == "max_iter"
    assert aanya_id in out["visited_ids"]
    assert razorpay_id in out["visited_ids"]
    # Rohan was named but never processed — the loop exited before iter 3.
    assert rohan_id not in out["visited_ids"]


def test_multi_hop_context_block_respects_char_budget(db):
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()
    aanya_id, _ = _seed_entities(db)

    async def fake_verdict(q, ctx):
        return {"sufficient": True, "follow_up": None}

    with patch.object(kg_mod, "ask_for_kg_followup", fake_verdict):
        out = _run(kg_mod.expand_multi_hop(
            "q", [aanya_id], char_budget=20,
        ))

    assert len(out["context_block"]) <= 20
    assert out["context_block"].endswith("...")
