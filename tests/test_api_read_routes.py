"""Read routes return the runtime's data, shaped for the wire."""
import pytest
from fastapi.testclient import TestClient

from assistant.io.api.app import create_app
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.studio_runtime import build_fake_runtime


@pytest.fixture()
def context(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("studio", frozenset(Capability))
    runtime = build_fake_runtime()
    client = TestClient(create_app(runtime, vault, origins=["http://localhost:3000"]))
    return client, runtime, {"Authorization": f"Bearer {token}"}


def test_knowledge_returns_the_graph_not_a_flat_list(context):
    client, _, headers = context
    body = client.get("/v1/memory/knowledge", headers=headers).json()["data"]
    assert set(body) == {"entities", "facts", "relationships"}
    assert [e["displayName"] for e in body["entities"]] == ["Sister", "Thesis defence"]


def test_knowledge_keys_are_camel_case_for_the_client(context):
    client, _, headers = context
    body = client.get("/v1/memory/knowledge", headers=headers).json()["data"]
    entity = body["entities"][0]
    assert {"canonicalName", "displayName", "createdAt", "sourceTurnId"} <= set(entity)
    assert "canonical_name" not in entity
    fact = body["facts"][0]
    assert {"subjectId", "invalidAt", "eventAt", "sourceTurnId"} <= set(fact)


def test_a_superseded_fact_survives_the_wire(context):
    client, _, headers = context
    facts = client.get("/v1/memory/knowledge", headers=headers).json()["data"]["facts"]
    superseded = [f for f in facts if f["invalidAt"]]
    assert superseded, "supersession was flattened away"
    assert superseded[0]["object"] == "Pune"


def test_provenance_survives_the_wire(context):
    client, _, headers = context
    facts = client.get("/v1/memory/knowledge", headers=headers).json()["data"]["facts"]
    assert all("sourceTurnId" in f for f in facts)
    assert any(f["sourceTurnId"] for f in facts), "every fact lost its turn id"


def test_relationship_properties_survive_the_wire(context):
    """Relationship.properties mirrors Entity.properties (see runtime.py); a
    serialiser that maps entities' properties but drops relationships' would
    silently lose the ego graph's edge labels."""
    client, _, headers = context
    relationships = client.get("/v1/memory/knowledge", headers=headers).json()["data"]["relationships"]
    with_role = [r for r in relationships if r["properties"]]
    assert with_role, "relationship properties were dropped on the wire"
    assert with_role[0]["properties"] == {"role": "plus one"}


def test_a_dangling_relationship_passes_through_untouched(context):
    """The fixture points a relationship at entity 99, which does not exist.
    The route must not filter it out -- the client's ego graph, not this
    layer, decides what to do with an edge to a missing node."""
    client, _, headers = context
    body = client.get("/v1/memory/knowledge", headers=headers).json()["data"]
    entity_ids = {e["id"] for e in body["entities"]}
    dangling = [r for r in body["relationships"] if r["toId"] not in entity_ids]
    assert dangling, "the dangling relationship was filtered out"
    assert dangling[0]["toId"] == 99


def test_preferences_carry_their_history(context):
    client, _, headers = context
    body = client.get("/v1/memory/preferences", headers=headers).json()["data"]
    first = body["preferences"][0]
    assert first["key"] == "reading_pace"
    assert first["history"][0]["value"] == "1.25x"
    assert first["history"][0]["changedAt"] == "2026-06-10T08:00:00Z"


def test_procedures_carry_their_steps(context):
    client, _, headers = context
    body = client.get("/v1/memory/procedures", headers=headers).json()["data"]
    assert body["procedures"][0]["steps"][0] == "dim the lights"
    assert body["procedures"][0]["runCount"] == 12


def test_memory_scope_is_validated(context):
    client, _, headers = context
    assert client.get("/v1/memory/not-a-scope", headers=headers).status_code == 422


def test_every_scope_is_reachable(context):
    client, _, headers = context
    for scope in ("knowledge", "preferences", "procedures"):
        assert client.get(f"/v1/memory/{scope}", headers=headers).status_code == 200


def test_settings_returns_rows_with_resolution_metadata(context):
    client, _, headers = context
    rows = client.get("/v1/settings", headers=headers).json()["data"]["rows"]
    by_key = {r["key"]: r for r in rows}
    assert by_key["camera_enabled"]["source"] == "env"
    assert by_key["active_personality"]["needsRestart"] is True
    assert by_key["active_personality"]["options"] == ["warm", "dry", "brisk"]


def test_source_distinguishes_stored_from_merely_default(context):
    """"db" and "default" must not collapse: one is a user choice, one is not."""
    client, _, headers = context
    rows = client.get("/v1/settings", headers=headers).json()["data"]["rows"]
    by_key = {r["key"]: r for r in rows}
    assert by_key["followup_timer"]["source"] == "db"
    assert by_key["tts_speed"]["source"] == "default"
    assert by_key["tts_speed"]["value"] == by_key["tts_speed"]["default"]


def test_settings_row_keeps_value_and_default_distinct(context):
    client, _, headers = context
    rows = client.get("/v1/settings", headers=headers).json()["data"]["rows"]
    row = [r for r in rows if r["key"] == "followup_timer"][0]
    assert row["value"] == 4.5
    assert row["default"] == 3.0


def test_personality_returns_base_traits_and_a_sample(context):
    client, _, headers = context
    body = client.get("/v1/personality", headers=headers).json()["data"]
    assert body["base"] == "warm"
    assert len(body["traits"]) == 6
    assert body["sample_line"]


def test_read_routes_are_enveloped(context):
    client, _, headers = context
    for path in ("/v1/memory/knowledge", "/v1/settings", "/v1/personality"):
        body = client.get(path, headers=headers).json()
        assert set(body) == {"data", "meta"}, f"{path} is not enveloped"
