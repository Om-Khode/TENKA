"""Read routes return the runtime's data, shaped for the wire."""
import pytest
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.api_client import build_api_client
from tests.fakes.studio_runtime import build_fake_runtime


@pytest.fixture()
def context(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("studio", frozenset(Capability))
    runtime = build_fake_runtime()
    client = build_api_client(runtime, vault)
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


def test_a_non_string_taught_property_round_trips_not_400s(context):
    """`Entity.properties` is arbitrary JSON a user taught -- a number, a
    bool, a null are all legal values, not just strings (Finding 1,
    2026-08-08 review). Before EntityPayload.properties was widened to
    `dict[str, JsonValue]`, a `dict[str, str]` annotation made this 400 the
    *entire* /v1/memory/knowledge response -- ResponseValidationError
    subclasses ValueError, so errors.py mapped it to a 400 that looked like
    a malformed client request for what was actually a server-side type
    mismatch. This must come back 200 with the values intact.
    """
    client, _, headers = context
    response = client.get("/v1/memory/knowledge", headers=headers)
    assert response.status_code == 200
    entities = {e["id"]: e for e in response.json()["data"]["entities"]}
    props = entities[1]["properties"]
    assert props["relation"] == "family"
    assert props["age"] == 34
    assert props["verified"] is True
    assert props["nickname"] is None


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


def test_an_unknown_memory_path_is_404_not_422(context):
    """GET /v1/memory/{scope} used to be one route with a `Literal` path
    parameter, so an unknown scope was a 422 FastAPI raised before the
    handler ran. It is now three static routes -- /v1/memory/knowledge,
    /v1/memory/preferences, /v1/memory/procedures (review finding,
    2026-08-08: the response shape is fully determined by scope, so three
    typed routes beat one route describing a union) -- and those are the
    only three URLs a client has ever called here. "not-a-scope" was never
    one of them; it now falls through to ordinary routing (no matching
    path) rather than being validated and rejected by a route that no
    longer exists, so it is a 404 like any other unmatched path.
    """
    client, _, headers = context
    assert client.get("/v1/memory/not-a-scope", headers=headers).status_code == 404


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
    assert body["sampleLine"]


def test_read_routes_are_enveloped(context):
    client, _, headers = context
    for path in ("/v1/memory/knowledge", "/v1/settings", "/v1/personality"):
        body = client.get(path, headers=headers).json()
        assert set(body) == {"data", "meta"}, f"{path} is not enveloped"


# ─── fix wave: meta is no longer two permanently empty strings ───────────
def test_meta_carries_a_request_id_matching_the_response_header(context):
    client, _, headers = context
    response = client.get("/v1/status", headers=headers)
    assert response.json()["meta"]["requestId"] == response.headers["X-Request-Id"]


def test_meta_request_id_differs_between_requests(context):
    client, _, headers = context
    first = client.get("/v1/status", headers=headers).json()["meta"]["requestId"]
    second = client.get("/v1/status", headers=headers).json()["meta"]["requestId"]
    assert first and second and first != second


def test_meta_generated_at_is_populated(context):
    client, _, headers = context
    body = client.get("/v1/status", headers=headers).json()
    assert body["meta"]["generatedAt"]
