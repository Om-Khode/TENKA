"""A command's grant comes from the command, not from the route."""
import pytest
from fastapi.testclient import TestClient

from assistant.io.api.app import create_app
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.studio_runtime import build_fake_runtime


@pytest.fixture()
def context(tmp_path):
    vault = TokenVault(tmp_path)
    runtime = build_fake_runtime()
    client = TestClient(create_app(runtime, vault, origins=["http://localhost:3000"]))
    tokens = {
        "full": vault.issue("studio", frozenset(Capability)),
        "chat": vault.issue("phone", frozenset({Capability.CHAT})),
    }
    return client, runtime, tokens


def head(token):
    return {"Authorization": f"Bearer {token}"}


def test_the_catalogue_is_readable_by_a_chat_device(context):
    client, _, tokens = context
    response = client.get("/v1/commands", headers=head(tokens["chat"]))
    assert response.status_code == 200
    assert [c["command_id"] for c in response.json()["data"]["commands"]] == [
        "lock_workstation", "volume_up"]


def test_the_catalogue_marks_destructive_entries(context):
    client, _, tokens = context
    commands = client.get("/v1/commands", headers=head(tokens["chat"])).json()["data"]["commands"]
    by_id = {c["command_id"]: c for c in commands}
    assert by_id["lock_workstation"]["destructive"] is True
    assert by_id["volume_up"]["destructive"] is False


def test_running_needs_the_command_s_own_grant(context):
    client, _, tokens = context
    assert client.post("/v1/commands/volume_up/run",
                       headers=head(tokens["chat"])).status_code == 403


def test_a_granted_device_can_run_it(context):
    client, runtime, tokens = context
    response = client.post("/v1/commands/volume_up/run", headers=head(tokens["full"]))
    assert response.status_code == 200
    assert runtime.commands.ran == ["volume_up"]


def test_an_unknown_command_is_404_not_a_failed_run(context):
    client, runtime, tokens = context
    assert client.post("/v1/commands/nope/run", headers=head(tokens["full"])).status_code == 404
    assert runtime.commands.ran == []


def test_a_refused_run_does_not_execute_anything(context):
    client, runtime, tokens = context
    client.post("/v1/commands/lock_workstation/run", headers=head(tokens["chat"]))
    assert runtime.commands.ran == []
