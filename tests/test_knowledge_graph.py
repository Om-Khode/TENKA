"""Tests for knowledge-graph — Knowledge Graph Layer."""

import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.storage.db import Database


@pytest.fixture
def db(tmp_path):
    # Import inside the fixture so any sibling test file (notably
    # test_runtime_config) that flushed `assistant.*` from sys.modules
    # doesn't leave us holding stale function references bound to a
    # detached storage module. With per-call imports we always touch the
    # current Database._instance.
    from assistant.storage.db import _reset_for_testing, init_db
    _reset_for_testing()
    db_path = tmp_path / "test.db"
    # Initialize through the singleton path so facade tests that call
    # assistant.knowledge_graph.init_kg() can find the active DB via get_db().
    db_obj = init_db(db_path)
    yield db_obj
    db_obj.close()
    _reset_for_testing()


# ─── Section 1: Schema v15 migration ───────────────────────────────────────


def test_fresh_db_has_kg_tables(db):
    """V15 migration creates the base KG triple (entities/facts/relationships).
    Later migrations may add more kg_ tables (kg_commitments in v18) — assert
    the base set as a subset rather than full equality so this test stays
    forward-compatible."""
    rows = db.fetchall(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'kg_%'"
    )
    names = {r["name"] for r in rows}
    assert {"kg_entities", "kg_facts", "kg_relationships"} <= names


def test_fresh_db_has_kg_indexes(db):
    """V15 migration creates required indexes."""
    rows = db.fetchall(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_kg_%'"
    )
    names = {r["name"] for r in rows}
    expected = {
        "idx_kg_entities_canon",
        "idx_kg_entities_type",
        "idx_kg_facts_subj",
        "idx_kg_facts_pred",
        "idx_kg_rel_from",
        "idx_kg_rel_to",
    }
    assert expected.issubset(names)


def test_v15_unique_constraints_entities(db):
    """UNIQUE(type, canonical_name) enforced on kg_entities."""
    db.execute(
        "INSERT INTO kg_entities (type, canonical_name, display_name, properties_json, "
        "source, confidence, created_at, updated_at) "
        "VALUES ('person', 'aanya', 'Aanya', '{}', 'user_msg', 1.0, '2026-05-31', '2026-05-31')"
    )
    db.commit()
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO kg_entities (type, canonical_name, display_name, properties_json, "
            "source, confidence, created_at, updated_at) "
            "VALUES ('person', 'aanya', 'Aanya', '{}', 'user_msg', 1.0, '2026-05-31', '2026-05-31')"
        )


def test_v15_unique_constraints_facts(db):
    """UNIQUE(subject_id, predicate, object) enforced on kg_facts."""
    db.execute(
        "INSERT INTO kg_entities (type, canonical_name, display_name, properties_json, "
        "source, confidence, created_at, updated_at) "
        "VALUES ('person', 'aanya', 'Aanya', '{}', 'user_msg', 1.0, '2026-05-31', '2026-05-31')"
    )
    db.commit()
    eid = db.fetchone("SELECT id FROM kg_entities WHERE canonical_name = 'aanya'")["id"]
    db.execute(
        "INSERT INTO kg_facts (subject_id, predicate, object, confidence, source, "
        "verified_at, expires_at, created_at) "
        "VALUES (?, 'lives_in', 'Berlin', 1.0, 'user_msg', NULL, NULL, '2026-05-31')",
        (eid,),
    )
    db.commit()
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO kg_facts (subject_id, predicate, object, confidence, source, "
            "verified_at, expires_at, created_at) "
            "VALUES (?, 'lives_in', 'Berlin', 1.0, 'user_msg', NULL, NULL, '2026-05-31')",
            (eid,),
        )


def test_v15_migration_idempotent(tmp_path):
    """Calling _migrate_v15 twice does not error (IF NOT EXISTS guards)."""
    from assistant.storage.db import _reset_for_testing
    _reset_for_testing()
    db_path = tmp_path / "idem.db"
    db_obj = Database(db_path)
    db_obj._migrate_v15()  # Should be a no-op now
    rows = db_obj.fetchall(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='kg_entities'"
    )
    assert len(rows) == 1
    db_obj.close()
    _reset_for_testing()


# (test_v15_latest_version_constant removed — superseded by
# tests/test_kg_v16_schema.py::test_latest_version_is_16 in knowledge-graph Session 1.)


# ─── Section 2: Repo entity ops (exact-match path) ─────────────────────────

from assistant.storage.repos.knowledge_graph import KnowledgeGraphRepo


def _stub_embed_loader():
    """Test stub — returns a callable that returns a fixed 384-d zero vector.

    Cosine-match path is exercised in a later test with a richer stub.
    """
    def _load():
        class _StubModel:
            def encode(self, text, normalize_embeddings=True):
                import numpy as np
                return np.zeros(384, dtype="float32")
        return _StubModel()
    return _load


@pytest.fixture
def kg_repo(db):
    return KnowledgeGraphRepo(db, embed_model_loader=_stub_embed_loader())


def test_upsert_entity_creates_new(kg_repo):
    eid, created = kg_repo.upsert_entity(
        entity_type="person", name="Aanya", source="user_msg",
    )
    assert created is True
    assert eid > 0


def test_upsert_entity_exact_match_returns_existing(kg_repo):
    eid1, c1 = kg_repo.upsert_entity(entity_type="person", name="Aanya", source="user_msg")
    eid2, c2 = kg_repo.upsert_entity(entity_type="person", name="Aanya", source="user_msg")
    assert eid1 == eid2
    assert c1 is True and c2 is False


def test_upsert_entity_different_type_creates_separate(kg_repo):
    eid1, _ = kg_repo.upsert_entity(entity_type="person", name="Berlin", source="user_msg")
    eid2, _ = kg_repo.upsert_entity(entity_type="place", name="Berlin", source="user_msg")
    assert eid1 != eid2


def test_canonical_name_is_lowercased(kg_repo):
    eid1, _ = kg_repo.upsert_entity(entity_type="person", name="Aanya K", source="user_msg")
    eid2, _ = kg_repo.upsert_entity(entity_type="person", name="aanya k", source="user_msg")
    assert eid1 == eid2  # canonical_name collision


def test_display_name_preserves_original_casing(kg_repo):
    eid, _ = kg_repo.upsert_entity(entity_type="person", name="Aanya Sharma", source="user_msg")
    row = kg_repo.get_entity(eid)
    assert row["display_name"] == "Aanya Sharma"
    assert row["canonical_name"] == "aanya sharma"


# ─── Section 3: Repo cosine-match dedup ────────────────────────────────────


def _cosine_embed_loader(name_to_vec: dict):
    """Stub that returns deterministic embeddings keyed by canonical_name."""
    import numpy as np

    def _load():
        class _StubModel:
            def encode(self, text, normalize_embeddings=True):
                # Match sentence-transformers shape contract: a single string
                # in → 1-D vec; a list (any length, including 1) → 2-D matrix.
                # The previous version collapsed length-1 lists to 1-D, which
                # broke _try_cosine_merge whenever there was exactly one
                # existing candidate to compare against.
                is_single = isinstance(text, str)
                texts = [text] if is_single else list(text)
                vecs = []
                for t in texts:
                    canon = _canonicalize_for_test(t)
                    vec = name_to_vec.get(canon)
                    if vec is None:
                        vec = np.zeros(384, dtype="float32")
                    else:
                        vec = np.asarray(vec, dtype="float32")
                        n = (vec ** 2).sum() ** 0.5
                        if n > 0:
                            vec = vec / n
                    vecs.append(vec)
                if is_single:
                    return vecs[0]
                return np.stack(vecs)
        return _StubModel()
    return _load


def _canonicalize_for_test(name: str) -> str:
    """Mirror of repo's _canonicalize for test stub keying."""
    return " ".join(name.lower().strip().split())


def test_cosine_merges_when_above_threshold(db):
    """'Aanya K' merges with existing 'Aanya' when cosine >= 0.85."""
    import numpy as np
    vecs = {
        "aanya":   np.array([1.0, 0.0, 0.0]),
        "aanya k": np.array([0.95, 0.31, 0.0]),
    }
    repo = KnowledgeGraphRepo(db, embed_model_loader=_cosine_embed_loader(vecs))
    eid1, c1 = repo.upsert_entity(entity_type="person", name="Aanya", source="user_msg")
    eid2, c2 = repo.upsert_entity(entity_type="person", name="Aanya K", source="user_msg")
    assert eid1 == eid2
    assert c1 is True and c2 is False
    row = repo.get_entity(eid1)
    import json as _json
    props = _json.loads(row["properties_json"])
    assert "Aanya K" in props.get("aliases", [])


def test_cosine_does_not_merge_below_threshold(db):
    """'Mumbai' does NOT merge with 'Berlin' (cosine < 0.85)."""
    import numpy as np
    vecs = {
        "berlin": np.array([1.0, 0.0, 0.0]),
        "mumbai": np.array([0.0, 1.0, 0.0]),
    }
    repo = KnowledgeGraphRepo(db, embed_model_loader=_cosine_embed_loader(vecs))
    eid1, _ = repo.upsert_entity(entity_type="place", name="Berlin", source="user_msg")
    eid2, _ = repo.upsert_entity(entity_type="place", name="Mumbai", source="user_msg")
    assert eid1 != eid2


def test_cosine_only_compares_same_type(db):
    """Cosine match is scoped to same type — 'Aanya' (person) never merges with 'Aanya' (concept)."""
    import numpy as np
    vecs = {"aanya": np.array([1.0, 0.0, 0.0])}
    repo = KnowledgeGraphRepo(db, embed_model_loader=_cosine_embed_loader(vecs))
    eid_p, _ = repo.upsert_entity(entity_type="person", name="Aanya", source="user_msg")
    eid_c, _ = repo.upsert_entity(entity_type="concept", name="Aanya", source="user_msg")
    assert eid_p != eid_c


def test_token_subset_merges_in_moderate_cosine_band(db):
    """'Aanya Sharma' merges with 'Aanya' when cosine is in 0.75-0.85 band
    AND one token-set is a subset of the other.

    Real-world signal: all-MiniLM-L6-v2 puts "aanya" vs "aanya sharma"
    at ~0.80 — below strong-merge threshold but token-subset confirms
    they're the same entity."""
    import numpy as np
    vecs = {
        "aanya":        np.array([1.0,  0.0,  0.0]),
        "aanya sharma": np.array([0.80, 0.60, 0.0]),  # cos ~= 0.80
    }
    repo = KnowledgeGraphRepo(db, embed_model_loader=_cosine_embed_loader(vecs))
    eid1, c1 = repo.upsert_entity(entity_type="person", name="Aanya", source="user_msg")
    eid2, c2 = repo.upsert_entity(entity_type="person", name="Aanya Sharma", source="user_msg")
    assert eid1 == eid2, "moderate-cosine + token-subset must merge"
    assert c1 is True and c2 is False
    row = repo.get_entity(eid1)
    import json as _json
    props = _json.loads(row["properties_json"])
    assert "Aanya Sharma" in props.get("aliases", [])


def test_token_subset_does_not_merge_when_no_overlap(db):
    """'John Doe' and 'John Smith' share a token but neither is a subset
    of the other — at moderate cosine (0.75-0.85) they must stay separate.
    Prevents over-merging of distinct people who share a first name."""
    import numpy as np
    # Pick vectors so cosine sits ~0.78 (subset band, not strong-merge band).
    vecs = {
        "john doe":   np.array([1.0,  0.0,   0.0]),
        "john smith": np.array([0.78, 0.626, 0.0]),  # already unit-norm; cos ~= 0.78
    }
    repo = KnowledgeGraphRepo(db, embed_model_loader=_cosine_embed_loader(vecs))
    eid1, _ = repo.upsert_entity(entity_type="person", name="John Doe", source="user_msg")
    eid2, _ = repo.upsert_entity(entity_type="person", name="John Smith", source="user_msg")
    assert eid1 != eid2, "no token-subset relation -> must NOT merge"


def test_token_subset_does_not_merge_below_subset_threshold(db):
    """Token-subset path requires cosine >= 0.75. Below that, distinct entities even with shared tokens."""
    import numpy as np
    # Force cosine ~= 0.50 (well below subset threshold)
    vecs = {
        "aanya":        np.array([1.0, 0.0, 0.0]),
        "aanya sharma": np.array([0.5, 0.87, 0.0]),  # cos ~= 0.50
    }
    repo = KnowledgeGraphRepo(db, embed_model_loader=_cosine_embed_loader(vecs))
    eid1, _ = repo.upsert_entity(entity_type="person", name="Aanya", source="user_msg")
    eid2, _ = repo.upsert_entity(entity_type="person", name="Aanya Sharma", source="user_msg")
    assert eid1 != eid2


# ─── Section 4: Repo fact ops ──────────────────────────────────────────────


def test_add_fact_inserts(kg_repo):
    eid, _ = kg_repo.upsert_entity(entity_type="person", name="Aanya", source="user_msg")
    fid = kg_repo.add_fact(
        subject_id=eid, predicate="lives_in", object="Berlin", source="user_msg"
    )
    assert fid > 0


def test_add_fact_unique_conflict_bumps_confidence(kg_repo):
    eid, _ = kg_repo.upsert_entity(entity_type="person", name="Aanya", source="user_msg")
    kg_repo.add_fact(eid, "lives_in", "Berlin", source="user_msg", confidence=0.6)
    kg_repo.add_fact(eid, "lives_in", "Berlin", source="user_msg", confidence=0.9)
    rows = kg_repo.get_facts_for_entity(eid)
    assert len(rows) == 1
    assert rows[0]["confidence"] == pytest.approx(0.9)
    assert rows[0]["verified_at"] is not None


def test_add_fact_never_lowers_confidence(kg_repo):
    eid, _ = kg_repo.upsert_entity(entity_type="person", name="Aanya", source="user_msg")
    kg_repo.add_fact(eid, "lives_in", "Berlin", source="user_msg", confidence=1.0)
    kg_repo.add_fact(eid, "lives_in", "Berlin", source="user_msg", confidence=0.4)
    rows = kg_repo.get_facts_for_entity(eid)
    assert rows[0]["confidence"] == pytest.approx(1.0)


def test_cleanup_expired_facts(kg_repo):
    from datetime import datetime, timedelta
    eid, _ = kg_repo.upsert_entity(entity_type="person", name="Aanya", source="user_msg")
    past = (datetime.now() - timedelta(days=1)).isoformat()
    kg_repo.add_fact(eid, "fad", "wears purple", source="user_msg", expires_at=past)
    kg_repo.add_fact(eid, "born_in", "1995", source="user_msg")
    removed = kg_repo.cleanup_expired_facts()
    assert removed == 1
    rows = kg_repo.get_facts_for_entity(eid)
    assert len(rows) == 1
    assert rows[0]["predicate"] == "born_in"


# ─── Section 5: Repo relationship ops ──────────────────────────────────────


def test_add_relationship_inserts(kg_repo):
    a, _ = kg_repo.upsert_entity(entity_type="person", name="Alex", source="user_msg")
    b, _ = kg_repo.upsert_entity(entity_type="person", name="Aanya", source="user_msg")
    rid = kg_repo.add_relationship(from_id=a, to_id=b, rel_type="knows", source="user_msg")
    assert rid > 0


def test_add_relationship_unique_conflict_is_noop(kg_repo):
    a, _ = kg_repo.upsert_entity(entity_type="person", name="Alex", source="user_msg")
    b, _ = kg_repo.upsert_entity(entity_type="person", name="Aanya", source="user_msg")
    rid1 = kg_repo.add_relationship(a, b, "knows", source="user_msg")
    rid2 = kg_repo.add_relationship(a, b, "knows", source="user_msg")
    assert rid1 == rid2
    rows = kg_repo._db.fetchall("SELECT * FROM kg_relationships")
    assert len(rows) == 1


def test_get_neighbors_depth_1(kg_repo):
    om, _    = kg_repo.upsert_entity(entity_type="person", name="Alex",    source="user_msg")
    aanya, _ = kg_repo.upsert_entity(entity_type="person", name="Aanya", source="user_msg")
    berlin, _ = kg_repo.upsert_entity(entity_type="place", name="Berlin", source="user_msg")
    kg_repo.add_relationship(om, aanya, "knows", source="user_msg")
    kg_repo.add_relationship(om, berlin, "lives_in", source="user_msg")
    nbrs = kg_repo.get_neighbors(om, depth=1)
    names = {n["entity"]["display_name"] for n in nbrs}
    assert names == {"Aanya", "Berlin"}


# ─── Section 6: Compound query ─────────────────────────────────────────────


def test_find_entities_by_name_exact(kg_repo):
    eid, _ = kg_repo.upsert_entity(entity_type="person", name="Aanya", source="user_msg")
    rows = kg_repo.find_entities_by_name("Aanya")
    assert len(rows) == 1
    assert rows[0]["id"] == eid


def test_find_entities_by_name_via_alias(db):
    """If entity has 'Aanya K' stored as alias, querying 'Aanya K' returns it."""
    import numpy as np
    vecs = {
        "aanya":   np.array([1.0, 0.0, 0.0]),
        "aanya k": np.array([0.95, 0.31, 0.0]),
    }
    repo = KnowledgeGraphRepo(db, embed_model_loader=_cosine_embed_loader(vecs))
    repo.upsert_entity(entity_type="person", name="Aanya", source="user_msg")
    repo.upsert_entity(entity_type="person", name="Aanya K", source="user_msg")  # merges as alias
    rows = repo.find_entities_by_name("Aanya K")
    assert len(rows) == 1


def test_expand_entity_context_returns_facts_and_neighbors(kg_repo):
    om, _    = kg_repo.upsert_entity(entity_type="person", name="Alex", source="user_msg")
    berlin, _ = kg_repo.upsert_entity(entity_type="place", name="Berlin", source="user_msg")
    kg_repo.add_fact(om, "works_on", "TENKA", source="user_msg")
    kg_repo.add_relationship(om, berlin, "lives_in", source="user_msg")
    ctx = kg_repo.expand_entity_context(om)
    assert ctx["entity"]["display_name"] == "Alex"
    assert any(f["predicate"] == "works_on" for f in ctx["facts"])
    assert any(n["entity"]["display_name"] == "Berlin" for n in ctx["neighbors"])


def test_expand_entity_context_respects_limits(kg_repo):
    om, _ = kg_repo.upsert_entity(entity_type="person", name="Alex", source="user_msg")
    for i in range(10):
        kg_repo.add_fact(om, f"prop_{i}", f"val_{i}", source="user_msg")
    ctx = kg_repo.expand_entity_context(om, fact_limit=3, neighbor_limit=2)
    assert len(ctx["facts"]) == 3
    assert len(ctx["neighbors"]) == 0


# ─── Section 7: LLM routing ────────────────────────────────────────────────


def test_kg_extraction_task_routes_to_flash_lite():
    from assistant.llm.router import TASK_MODEL_MAP
    chain = TASK_MODEL_MAP.get("kg_extraction")
    assert chain is not None, "kg_extraction missing from TASK_MODEL_MAP"
    primary = chain[0]
    assert primary == ("gemini", "gemini-2.5-flash-lite")
    # at least one free-tier fallback
    assert any(p[0] in {"cerebras", "groq"} for p in chain[1:])


# ─── Section 8: LLM contract — ask_for_entity_extraction ───────────────────

import asyncio


class _LLMResultStub:
    def __init__(self, text):
        self.text = text
        self.provider = "stub"
        self.model = "stub"
        self.latency_ms = 0
        self.fallback_depth = 0


def _run(coro):
    # `asyncio.get_event_loop()` raises in 3.10+ if no loop is set for the
    # current thread — which happens after sibling test files (notably
    # test_typed_memory) close their loops. Use asyncio.run, which creates
    # and tears down its own loop.
    return asyncio.run(coro)


def test_ask_for_entity_extraction_parses_valid_payload():
    from assistant.llm import contracts

    valid_json = (
        '{"entities":[{"type":"person","name":"Aanya","confidence":1.0}],'
        '"facts":[{"subject":"Aanya","predicate":"lives_in","object":"Berlin","confidence":1.0}],'
        '"relationships":[]}'
    )

    async def fake_get_llm_response(*args, **kwargs):
        return _LLMResultStub(valid_json)

    with patch.object(contracts, "get_llm_response", fake_get_llm_response):
        result = _run(contracts.ask_for_entity_extraction("My friend Aanya lives in Berlin", "user_msg"))
    assert result["entities"][0]["name"] == "Aanya"
    assert result["facts"][0]["predicate"] == "lives_in"
    assert result["relationships"] == []


def test_ask_for_entity_extraction_malformed_json_returns_empty():
    from assistant.llm import contracts

    async def fake_get_llm_response(*args, **kwargs):
        return _LLMResultStub("this is not json at all")

    with patch.object(contracts, "get_llm_response", fake_get_llm_response):
        result = _run(contracts.ask_for_entity_extraction("hi", "user_msg"))
    assert result == {"entities": [], "facts": [], "relationships": [], "commitments": []}


def test_ask_for_entity_extraction_network_failure_returns_empty():
    from assistant.llm import contracts

    async def fake_get_llm_response(*args, **kwargs):
        raise RuntimeError("network is down")

    with patch.object(contracts, "get_llm_response", fake_get_llm_response):
        result = _run(contracts.ask_for_entity_extraction("hi", "user_msg"))
    assert result == {"entities": [], "facts": [], "relationships": [], "commitments": []}


def test_ask_for_entity_extraction_uses_kg_extraction_task_type():
    from assistant.llm import contracts

    captured = {}

    async def fake_get_llm_response(prompt, **kwargs):
        captured["task_type"] = kwargs.get("task_type")
        return _LLMResultStub('{"entities":[],"facts":[],"relationships":[]}')

    with patch.object(contracts, "get_llm_response", fake_get_llm_response):
        _run(contracts.ask_for_entity_extraction("hi", "user_msg"))
    assert captured["task_type"] == "kg_extraction"


def test_ask_for_entity_extraction_prompt_warns_against_templated_phrases():
    """Live-test surfaced: TENKA's store_memory reply 'Got it, I'll remember
    that. New Mockups Sender: Aanya Sharma.' produced a junk 'New Mockups
    Sender' project entity. The prompt must explicitly tell the LLM to skip
    templated key:value confirmation phrases."""
    from assistant.llm import contracts

    captured = {}

    async def fake_get_llm_response(prompt, **kwargs):
        captured["system_prompt"] = kwargs.get("system_prompt", "")
        return _LLMResultStub('{"entities":[],"facts":[],"relationships":[]}')

    with patch.object(contracts, "get_llm_response", fake_get_llm_response):
        _run(contracts.ask_for_entity_extraction("hi", "tenka_resp"))

    sp = captured["system_prompt"].lower()
    assert "templated" in sp or "template" in sp, "prompt must mention templated phrases"
    assert "key" in sp and "value" in sp, "prompt must reference key:value structure"


def test_ask_for_entity_extraction_prompt_forbids_pronouns():
    """Live-test surfaced: '...she uses Figma' produced a junk concept entity
    'she' plus a (she, uses, Figma) relationship. The prompt must explicitly
    instruct the LLM to skip pronouns."""
    from assistant.llm import contracts

    captured = {}

    async def fake_get_llm_response(prompt, **kwargs):
        captured["system_prompt"] = kwargs.get("system_prompt", "")
        return _LLMResultStub('{"entities":[],"facts":[],"relationships":[]}')

    with patch.object(contracts, "get_llm_response", fake_get_llm_response):
        _run(contracts.ask_for_entity_extraction("hi", "user_msg"))

    sp = captured["system_prompt"].lower()
    assert "pronoun" in sp, "prompt must mention pronouns"
    for token in ("she", "he", "they", "them", "their"):
        assert token in sp, f"prompt must list pronoun {token!r}"


# ─── Section 9: Facade pre-filter ──────────────────────────────────────────


def test_pre_filter_blocks_pleasantries():
    from assistant.knowledge_graph import _has_entity_signal
    assert _has_entity_signal("ok thanks") is False
    assert _has_entity_signal("hmm") is False
    assert _has_entity_signal("") is False
    assert _has_entity_signal("👋") is False


def test_pre_filter_allows_personal_signals():
    from assistant.knowledge_graph import _has_entity_signal
    assert _has_entity_signal("I live in Berlin") is True
    assert _has_entity_signal("my friend Aanya called") is True


def test_pre_filter_allows_capitalized_nouns():
    from assistant.knowledge_graph import _has_entity_signal
    assert _has_entity_signal("Aanya is doing the UI work") is True
    assert _has_entity_signal("BookMyShow has the listings") is True


def test_pre_filter_allows_response_cues():
    from assistant.knowledge_graph import _has_entity_signal
    assert _has_entity_signal("you mentioned the deadline earlier") is True
    assert _has_entity_signal("your project sounds interesting") is True


# ─── Section 10: Facade ingest_turn ────────────────────────────────────────


def test_ingest_turn_skips_when_pre_filter_fails(db):
    """No LLM call when text lacks entity signal."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()

    called = {"count": 0}

    async def fake_extract(text, source, context_hint=None):
        called["count"] += 1
        return {"entities": [], "facts": [], "relationships": []}

    with patch("assistant.knowledge_graph.ask_for_entity_extraction", fake_extract, create=True):
        _run(kg_mod.ingest_turn("ok thanks", "user_msg"))
    assert called["count"] == 0


def test_ingest_turn_persists_entities_and_facts(db):
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()

    payload = {
        "entities": [
            {"type": "person", "name": "Aanya", "confidence": 1.0},
            {"type": "place",  "name": "Berlin",  "confidence": 1.0},
        ],
        "facts": [
            {"subject": "Aanya", "predicate": "lives_in", "object": "Berlin", "confidence": 1.0}
        ],
        "relationships": [],
    }

    async def fake_extract(text, source, context_hint=None):
        return payload

    with patch("assistant.knowledge_graph.ask_for_entity_extraction", fake_extract, create=True):
        _run(kg_mod.ingest_turn("My friend Aanya lives in Berlin", "user_msg"))

    repo = kg_mod._get_repo()
    aanya = repo.find_entities_by_name("Aanya")
    assert len(aanya) == 1
    facts = repo.get_facts_for_entity(aanya[0]["id"])
    assert any(f["predicate"] == "lives_in" and f["object"] == "Berlin" for f in facts)


def test_tenka_resp_confidence_is_down_weighted(db):
    """Assistant-source extractions store lower confidence than user-source.
    Defense against template-echo noise where the assistant's reply contains
    storage-label phrases that aren't real-world facts."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()

    payload = {
        "entities": [{"type": "person", "name": "Kiran", "confidence": 1.0}],
        "facts": [{"subject": "Kiran", "predicate": "is_a", "object": "engineer", "confidence": 1.0}],
        "relationships": [],
    }

    async def fake_extract(text, source, context_hint=None):
        return payload

    with patch("assistant.knowledge_graph.ask_for_entity_extraction", fake_extract, create=True):
        _run(kg_mod.ingest_turn("Got it. Kiran is an engineer.", "tenka_resp"))

    repo = kg_mod._get_repo()
    kiran = repo.find_entities_by_name("Kiran")
    assert len(kiran) == 1
    # 1.0 (LLM) * 0.7 (tenka_resp factor) = 0.7
    assert abs(kiran[0]["confidence"] - 0.7) < 0.001
    facts = repo.get_facts_for_entity(kiran[0]["id"])
    assert abs(facts[0]["confidence"] - 0.7) < 0.001


def test_user_msg_confirms_tenka_resp_fact_raises_confidence(db):
    """When the assistant first asserts a fact (low conf) and the user later
    confirms it, the fact's stored confidence rises via add_fact's max(old, new).
    Ensures down-weighting doesn't permanently suppress legit knowledge."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()

    payload = {
        "entities": [{"type": "person", "name": "Priya", "confidence": 1.0}],
        "facts": [{"subject": "Priya", "predicate": "lives_in", "object": "Delhi", "confidence": 1.0}],
        "relationships": [],
    }

    async def fake_extract(text, source, context_hint=None):
        return payload

    with patch("assistant.knowledge_graph.ask_for_entity_extraction", fake_extract, create=True):
        _run(kg_mod.ingest_turn("Priya lives in Delhi.", "tenka_resp"))
        _run(kg_mod.ingest_turn("Priya lives in Delhi.", "user_msg"))

    repo = kg_mod._get_repo()
    priya = repo.find_entities_by_name("Priya")
    facts = repo.get_facts_for_entity(priya[0]["id"])
    assert len(facts) == 1
    assert abs(facts[0]["confidence"] - 1.0) < 0.001, "user_msg must raise confidence"


def test_ingest_turn_swallows_extraction_errors(db):
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()

    async def fake_extract(text, source, context_hint=None):
        raise RuntimeError("kapow")

    with patch("assistant.knowledge_graph.ask_for_entity_extraction", fake_extract, create=True):
        # Must not raise
        _run(kg_mod.ingest_turn("My friend Aanya lives in Berlin", "user_msg"))


def test_ingest_turn_disabled_by_env(db, monkeypatch):
    """KG_INGEST_ENABLED=false → ingest is a no-op (no LLM call)."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()

    monkeypatch.setenv("KG_INGEST_ENABLED", "false")
    called = {"count": 0}

    async def fake_extract(text, source, context_hint=None):
        called["count"] += 1
        return {"entities": [], "facts": [], "relationships": []}

    with patch("assistant.knowledge_graph.ask_for_entity_extraction", fake_extract, create=True):
        _run(kg_mod.ingest_turn("My friend Aanya lives in Berlin", "user_msg"))
    assert called["count"] == 0


def test_persist_drops_pronoun_entities(db):
    """Even if the LLM leaks a pronoun-as-entity, the persist layer must drop
    it before it hits the DB. Defense-in-depth against prompt regressions."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()

    payload = {
        "entities": [
            {"type": "person", "name": "she", "confidence": 0.5},
            {"type": "person", "name": "Aanya", "confidence": 1.0},
        ],
        "facts": [],
        "relationships": [],
    }

    async def fake_extract(text, source, context_hint=None):
        return payload

    with patch("assistant.knowledge_graph.ask_for_entity_extraction", fake_extract, create=True):
        _run(kg_mod.ingest_turn("Aanya and she work together.", "user_msg"))

    repo = kg_mod._get_repo()
    assert repo.find_entities_by_name("she") == []
    assert len(repo.find_entities_by_name("Aanya")) == 1


def test_persist_drops_pronoun_relationship_endpoints(db):
    """Junk relationship (she, uses, Figma) must be dropped even if extractor
    returns it. Prevents pollution from pronoun-referent relationships."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()

    payload = {
        "entities": [
            {"type": "tool", "name": "Figma", "confidence": 1.0},
        ],
        "facts": [],
        "relationships": [
            {"from": "she", "to": "Figma", "type": "uses", "confidence": 0.5},
            {"from": "Figma", "to": "them", "type": "related_to", "confidence": 0.5},
        ],
    }

    async def fake_extract(text, source, context_hint=None):
        return payload

    with patch("assistant.knowledge_graph.ask_for_entity_extraction", fake_extract, create=True):
        _run(kg_mod.ingest_turn("She uses Figma a lot.", "user_msg"))

    repo = kg_mod._get_repo()
    figma = repo.find_entities_by_name("Figma")
    assert len(figma) == 1
    # No "she" or "them" entity should have been auto-upserted as concept
    assert repo.find_entities_by_name("she") == []
    assert repo.find_entities_by_name("them") == []
    # And no relationship row touching them
    neighbors = repo.get_neighbors(figma[0]["id"])
    assert neighbors == []


def test_persist_drops_pronoun_fact_subject(db):
    """A fact with a pronoun subject must be dropped — no entity auto-created
    for the pronoun, no fact stored."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()

    payload = {
        "entities": [],
        "facts": [
            {"subject": "she", "predicate": "lives_in", "object": "Berlin", "confidence": 0.5},
        ],
        "relationships": [],
    }

    async def fake_extract(text, source, context_hint=None):
        return payload

    with patch("assistant.knowledge_graph.ask_for_entity_extraction", fake_extract, create=True):
        _run(kg_mod.ingest_turn("She lives in Berlin.", "user_msg"))

    repo = kg_mod._get_repo()
    assert repo.find_entities_by_name("she") == []


def test_ingest_turn_skips_tenka_resp_for_non_conversational_intent(db):
    """tenka_resp from non-conversational intents (web_search, set_reminder,
    code_executor, store_memory, ...) must be skipped BEFORE the LLM call.
    Prevents weather data, error codes, storage labels from polluting the KG."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()

    called = {"count": 0}

    async def fake_extract(text, source, context_hint=None):
        called["count"] += 1
        return {"entities": [{"type": "place", "name": "Berlin", "confidence": 1.0}],
                "facts": [], "relationships": []}

    weather_reply = "The current temperature in Berlin is 28C with clear sky."
    with patch("assistant.knowledge_graph.ask_for_entity_extraction", fake_extract, create=True):
        for intent in ("web_search", "set_reminder", "code_executor",
                       "store_memory", "planner", "get_time"):
            _run(kg_mod.ingest_turn(weather_reply, "tenka_resp", reply_intent=intent))

    assert called["count"] == 0, "non-conversational tenka_resp must not call extractor"
    repo = kg_mod._get_repo()
    assert repo.find_entities_by_name("Berlin") == []


def test_ingest_turn_allows_tenka_resp_for_conversational_intents(db):
    """small_talk / unknown / memory_query replies ARE ingested — they
    contain real biographical content the KG should learn from."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()

    called = {"count": 0}
    payload = {
        "entities": [{"type": "person", "name": "Aanya", "confidence": 1.0}],
        "facts": [], "relationships": [],
    }

    async def fake_extract(text, source, context_hint=None):
        called["count"] += 1
        return payload

    with patch("assistant.knowledge_graph.ask_for_entity_extraction", fake_extract, create=True):
        for intent in ("small_talk", "unknown", "memory_query"):
            _run(kg_mod.ingest_turn("Aanya is a UI designer.", "tenka_resp",
                                    reply_intent=intent))

    assert called["count"] == 3
    repo = kg_mod._get_repo()
    assert len(repo.find_entities_by_name("Aanya")) == 1


def test_ingest_turn_always_ingests_user_msg_regardless_of_intent(db):
    """source='user_msg' is always ingested — the user's own words count
    even if the intent that handled the turn was non-conversational."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()

    called = {"count": 0}
    payload = {
        "entities": [{"type": "person", "name": "Aanya", "confidence": 1.0}],
        "facts": [], "relationships": [],
    }

    async def fake_extract(text, source, context_hint=None):
        called["count"] += 1
        return payload

    with patch("assistant.knowledge_graph.ask_for_entity_extraction", fake_extract, create=True):
        _run(kg_mod.ingest_turn("Remind me Aanya is the designer", "user_msg",
                                reply_intent="set_reminder"))

    assert called["count"] == 1
    repo = kg_mod._get_repo()
    assert len(repo.find_entities_by_name("Aanya")) == 1


def test_ingest_turn_tenka_resp_with_no_intent_still_ingests(db):
    """Backwards-compat: callers that don't pass reply_intent (None) get the
    old behavior — ingest proceeds. Keeps existing tests + call sites valid."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()

    called = {"count": 0}

    async def fake_extract(text, source, context_hint=None):
        called["count"] += 1
        return {"entities": [{"type": "person", "name": "Aanya", "confidence": 1.0}],
                "facts": [], "relationships": []}

    with patch("assistant.knowledge_graph.ask_for_entity_extraction", fake_extract, create=True):
        _run(kg_mod.ingest_turn("Aanya is a designer.", "tenka_resp"))

    assert called["count"] == 1


# ─── Section 11: Facade build_kg_context + search_entities ─────────────────


def test_build_kg_context_returns_none_when_no_entities(db):
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()
    out = kg_mod.build_kg_context("how is the weather today")
    assert out is None


def test_build_kg_context_formats_resolved_entity(db):
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()
    repo = kg_mod._get_repo()
    aid, _ = repo.upsert_entity(entity_type="person", name="Aanya", source="user_msg")
    pid, _ = repo.upsert_entity(entity_type="place", name="Berlin", source="user_msg")
    repo.add_fact(aid, "lives_in", "Berlin", source="user_msg")
    repo.add_relationship(aid, pid, "lives_in", source="user_msg")

    out = kg_mod.build_kg_context("did Aanya finish that?")
    assert out is not None
    assert "Aanya" in out
    assert "[KNOWLEDGE]" in out
    assert len(out) <= 600


def test_build_kg_context_skipped_when_query_injection_disabled(db, monkeypatch):
    monkeypatch.setenv("KG_QUERY_INJECTION_ENABLED", "false")
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()
    repo = kg_mod._get_repo()
    repo.upsert_entity(entity_type="person", name="Aanya", source="user_msg")
    out = kg_mod.build_kg_context("did Aanya finish that?")
    assert out is None


def test_search_entities_returns_matches(db):
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()
    repo = kg_mod._get_repo()
    repo.upsert_entity(entity_type="person", name="Aanya", source="user_msg")
    out = kg_mod.search_entities("Aanya")
    assert len(out) == 1


# ─── Section 14: memory_query KG fallback ──────────────────────────────────


def test_memory_query_uses_hybrid_when_facts_present(db, monkeypatch):
    """When hybrid_search_facts returns results, KG fallback is NOT consulted."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()

    from assistant.actions import memory_search

    monkeypatch.setattr(
        "assistant.memory.hybrid_search_facts",
        lambda q, limit=10: [{"key": "fav_food", "value": "biryani"}],
    )
    monkeypatch.setattr(
        "assistant.memory.hybrid_search_conversations",
        lambda q, limit=5: [],
    )
    monkeypatch.setattr(
        "assistant.memory.search_recording_sessions",
        lambda q, limit=3: [],
    )
    called = {"kg": 0}
    monkeypatch.setattr(
        kg_mod,
        "search_entities",
        lambda q: called.__setitem__("kg", called["kg"] + 1) or [],
    )

    async def fake_synth(prompt, **kw):
        return "biryani"

    monkeypatch.setattr("assistant.llm.contracts.ask_for_synthesis", fake_synth)
    # handle_memory_query signature: (params: dict, llm_response: str, bridge=None)
    _run(memory_search.handle_memory_query({"query": "favorite food"}, ""))
    assert called["kg"] == 0


def test_memory_query_falls_back_to_kg_when_hybrid_empty(db, monkeypatch):
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()
    repo = kg_mod._get_repo()
    aid, _ = repo.upsert_entity(entity_type="person", name="Aanya", source="user_msg")
    repo.add_fact(aid, "lives_in", "Berlin", source="user_msg")

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

    captured = {}

    async def fake_synth(prompt, **kw):
        captured["prompt"] = prompt
        return "Aanya lives in Berlin"

    monkeypatch.setattr("assistant.llm.contracts.ask_for_synthesis", fake_synth)
    _run(memory_search.handle_memory_query({"query": "Aanya"}, ""))
    assert "Aanya" in captured["prompt"]


def test_memory_query_empty_path_unchanged(db, monkeypatch):
    """When both hybrid-retrieval and KG return empty, the existing 'no memory' response stands."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()

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
    monkeypatch.setattr(kg_mod, "search_entities", lambda q: [])

    out = _run(memory_search.handle_memory_query({"query": "unknown thing"}, ""))
    # Existing empty-path message is preserved — just assert it's a non-empty string
    assert isinstance(out, str) and len(out) > 0
