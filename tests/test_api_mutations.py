"""Mutations mutate, report what they rejected, and never lie about it."""
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


def test_forget_all_demands_system_control_not_just_a_sending_device(tmp_path):
    """Wiping every entity, fact, preference and procedure is not ordinary use.

    The two deletes stay on different grants, but both moved up a tier once
    the read capabilities came to mean "may read" and nothing more.
    Single-item forget is
    `chat_send`: deleting one thing she was told about is the same class of
    act as saying "forget that" in a turn, so a device trusted to drive her is
    trusted to do it. The wipe is different in kind, not degree -- erasing her
    memory outright still demands `system_control`, so even a sending device
    cannot reach it.
    """
    vault = TokenVault(tmp_path)
    sender = vault.issue("phone", frozenset({Capability.OBSERVE, Capability.CHAT_SEND}))
    runtime = build_fake_runtime()
    client = build_api_client(runtime, vault)
    headers = {"Authorization": f"Bearer {sender}"}

    wipe = client.delete("/v1/memory", headers=headers)
    assert wipe.status_code == 403

    single = client.delete("/v1/memory/preferences/reading_pace", headers=headers)
    assert single.status_code == 200


def test_a_read_only_device_cannot_forget_even_one_item(tmp_path):
    """The regression this closes: one read grant alone used to carry a memory
    delete, so a device deliberately issued read-only could erase what she
    knows one item at a time -- and on the removed `quick` listener (Milestone
    6b), whose entire ceiling used to be that one grant, it was the only grant
    any device could hold at all."""
    vault = TokenVault(tmp_path)
    reader = vault.issue("reader", frozenset({Capability.RECALL}))
    client = build_api_client(build_fake_runtime(), vault)
    headers = {"Authorization": f"Bearer {reader}"}

    assert client.delete("/v1/memory/preferences/reading_pace",
                         headers=headers).status_code == 403
    # ...and the read it *is* entitled to still answers, so this narrowed the
    # write side only.
    assert client.get("/v1/memory/preferences", headers=headers).status_code == 200


# ─── capability tier: writing settings needs system_control ──────────────
def test_saving_settings_demands_system_control_not_just_a_read(tmp_path):
    """The same ruling that moved forget-all off of a read grant: a phone
    paired only to watch must not be able to rewrite the daemon's own CORS
    allow-list, switch the camera on, or flip any other setting. Reading
    stays on OBSERVE -- proven here too, and that nothing changed.
    """
    vault = TokenVault(tmp_path)
    watcher = vault.issue("phone", frozenset({Capability.OBSERVE}))
    runtime = build_fake_runtime()
    client = build_api_client(runtime, vault)
    headers = {"Authorization": f"Bearer {watcher}"}

    before = client.get("/v1/settings", headers=headers).json()["data"]["rows"]

    response = client.patch("/v1/settings", headers=headers,
                            json={"changes": {"followup_timer": 9.0}})
    assert response.status_code == 403

    after = client.get("/v1/settings", headers=headers).json()["data"]["rows"]
    assert after == before, "settings changed despite the 403"


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
    assert body["restartRequired"] == ["active_personality"]


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


def test_an_unknown_personality_is_400_and_changes_nothing(context):
    """switch_personality() reports an unknown base as a return string, not
    an exception -- unchecked, the route used to answer 200 with the
    *previous*, unchanged state. Validated against state().available now, so
    an unknown base is refused outright rather than silently no-opping
    behind a success code.
    """
    client, _, headers = context
    before = client.get("/v1/personality", headers=headers).json()["data"]

    response = client.patch("/v1/personality", headers=headers,
                            json={"base": "does-not-exist"})
    assert response.status_code == 400

    after = client.get("/v1/personality", headers=headers).json()["data"]
    assert after == before


def test_resetting_the_personality_puts_traits_back(context):
    client, _, headers = context
    client.patch("/v1/personality", headers=headers, json={"base": "dry"})

    body = client.post("/v1/personality/reset", headers=headers).json()["data"]
    assert body["base"] == "warm"
    assert all(v == 0.5 for v in body["traits"].values())
    assert client.get("/v1/personality", headers=headers).json()["data"]["base"] == "warm"


def test_a_settings_patch_with_too_many_keys_is_refused(context):
    client, _, headers = context
    changes = {f"key_{i}": 1 for i in range(201)}
    response = client.patch("/v1/settings", headers=headers, json={"changes": changes})
    assert response.status_code == 422


def test_a_settings_patch_with_an_overlong_key_is_refused(context):
    client, _, headers = context
    changes = {"x" * 201: 1}
    response = client.patch("/v1/settings", headers=headers, json={"changes": changes})
    assert response.status_code == 422


def test_a_settings_patch_with_an_overlong_string_value_is_refused(context):
    client, _, headers = context
    changes = {"followup_timer": "x" * 4_097}
    response = client.patch("/v1/settings", headers=headers, json={"changes": changes})
    assert response.status_code == 422


def test_a_settings_patch_with_a_list_value_is_refused(context):
    client, _, headers = context
    changes = {"followup_timer": [1, 2, 3]}
    response = client.patch("/v1/settings", headers=headers, json={"changes": changes})
    assert response.status_code == 422


def test_a_settings_patch_with_a_dict_value_is_refused(context):
    client, _, headers = context
    changes = {"followup_timer": {"nested": True}}
    response = client.patch("/v1/settings", headers=headers, json={"changes": changes})
    assert response.status_code == 422


def test_a_settings_patch_with_a_null_value_is_refused(context):
    client, _, headers = context
    changes = {"followup_timer": None}
    response = client.patch("/v1/settings", headers=headers, json={"changes": changes})
    assert response.status_code == 422


def test_a_settings_patch_with_an_oversized_integer_is_refused(context):
    client, _, headers = context
    changes = {"followup_timer": 10 ** 13}
    response = client.patch("/v1/settings", headers=headers, json={"changes": changes})
    assert response.status_code == 422
