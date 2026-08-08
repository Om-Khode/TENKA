"""Mutations mutate, report what they rejected, and never lie about it."""
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


def test_forgetting_an_entity_removes_it(context):
    client, _, headers = context
    assert client.delete("/v1/memory/knowledge/1", headers=headers).status_code == 200
    graph = client.get("/v1/memory/knowledge", headers=headers).json()["data"]
    assert 1 not in [e["id"] for e in graph["entities"]]


def test_forgetting_an_entity_takes_its_facts_and_edges(context):
    client, _, headers = context
    client.delete("/v1/memory/knowledge/1", headers=headers)
    graph = client.get("/v1/memory/knowledge", headers=headers).json()["data"]
    assert 1 not in [f["subjectId"] for f in graph["facts"]], "orphaned facts survived"
    assert all(r["fromId"] != 1 and r["toId"] != 1 for r in graph["relationships"])


def test_forgetting_a_preference_removes_it(context):
    client, _, headers = context
    assert client.delete("/v1/memory/preferences/reading_pace",
                         headers=headers).status_code == 200
    body = client.get("/v1/memory/preferences", headers=headers).json()["data"]
    assert body["preferences"] == []


def test_forgetting_an_unknown_item_is_404(context):
    client, _, headers = context
    assert client.delete("/v1/memory/knowledge/9999", headers=headers).status_code == 404


def test_forget_all_empties_every_scope(context):
    client, _, headers = context
    response = client.delete("/v1/memory", headers=headers)
    assert response.status_code == 200
    assert response.json()["data"]["removed"] == 4
    graph = client.get("/v1/memory/knowledge", headers=headers).json()["data"]
    assert graph["entities"] == [] and graph["facts"] == []
    assert client.get("/v1/memory/preferences",
                      headers=headers).json()["data"]["preferences"] == []
    assert client.get("/v1/memory/procedures",
                      headers=headers).json()["data"]["procedures"] == []


def test_saving_a_setting_changes_what_is_read_back(context):
    client, _, headers = context
    response = client.patch("/v1/settings", headers=headers,
                            json={"changes": {"followup_timer": 6.0}})
    assert response.json()["data"]["saved"] == ["followup_timer"]
    rows = client.get("/v1/settings", headers=headers).json()["data"]["rows"]
    assert [r for r in rows if r["key"] == "followup_timer"][0]["value"] == 6.0


def test_saving_reports_restart_required_separately_from_saved(context):
    client, _, headers = context
    body = client.patch("/v1/settings", headers=headers,
                        json={"changes": {"active_personality": "dry"}}).json()["data"]
    assert body["saved"] == ["active_personality"]
    assert body["restart_required"] == ["active_personality"]


def test_saving_a_row_sourced_from_env_flips_it_to_db(context):
    """The assistant resolves DB before env, so this is a legal override."""
    client, _, headers = context
    body = client.patch("/v1/settings", headers=headers,
                        json={"changes": {"camera_enabled": False}}).json()["data"]
    assert body["saved"] == ["camera_enabled"]
    rows = client.get("/v1/settings", headers=headers).json()["data"]["rows"]
    row = [r for r in rows if r["key"] == "camera_enabled"][0]
    assert row["source"] == "db"
    assert row["value"] is False


def test_a_mixed_patch_saves_the_good_and_rejects_the_bad(context):
    client, _, headers = context
    body = client.patch("/v1/settings", headers=headers, json={"changes": {
        "followup_timer": 5.0, "nonsense_key": 1,
    }}).json()["data"]
    assert body["saved"] == ["followup_timer"]
    assert set(body["rejected"]) == {"nonsense_key"}


def test_an_empty_patch_is_accepted_and_changes_nothing(context):
    client, _, headers = context
    body = client.patch("/v1/settings", headers=headers, json={"changes": {}}).json()["data"]
    assert body["saved"] == [] and body["rejected"] == {}


def test_changing_the_personality_base_reads_back(context):
    client, _, headers = context
    body = client.patch("/v1/personality", headers=headers, json={"base": "dry"}).json()["data"]
    assert body["base"] == "dry"
    assert client.get("/v1/personality", headers=headers).json()["data"]["base"] == "dry"


def test_an_oversized_body_is_refused(context):
    client, _, headers = context
    response = client.patch("/v1/personality", headers=headers, json={"base": "x" * 5_000})
    assert response.status_code == 422
