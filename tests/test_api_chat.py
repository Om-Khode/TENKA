"""Chat enters the existing pipeline, one turn at a time."""
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


def test_sending_a_message_reaches_the_runtime(context):
    client, runtime, headers = context
    response = client.post("/v1/chat", headers=headers, json={"text": "what time is it"})
    assert response.status_code == 202
    assert runtime.chat.sent == ["what time is it"]
    assert response.json()["data"]["turnId"] == "t1"


def test_a_busy_assistant_answers_409_and_does_not_queue(context):
    client, runtime, headers = context
    runtime.chat.busy = True
    response = client.post("/v1/chat", headers=headers, json={"text": "hello"})
    assert response.status_code == 409
    assert runtime.chat.sent == []


def test_an_empty_message_is_422(context):
    client, runtime, headers = context
    assert client.post("/v1/chat", headers=headers, json={"text": ""}).status_code == 422
    assert runtime.chat.sent == []


def test_an_oversized_message_is_422(context):
    client, runtime, headers = context
    assert client.post("/v1/chat", headers=headers,
                       json={"text": "x" * 9_000}).status_code == 422
    assert runtime.chat.sent == []


def test_a_sealed_envelope_is_refused_for_now(context):
    """Milestone 6 implements it. Until then it must not be silently ignored."""
    client, _, headers = context
    response = client.post("/v1/chat", headers=headers,
                           json={"sealed": "abc", "nonce": "def"})
    assert response.status_code in (415, 422)


def test_conversations_list(context):
    client, _, headers = context
    body = client.get("/v1/chat/conversations", headers=headers).json()["data"]
    assert body["conversations"][0]["conversationId"] == "c1"


def test_a_conversation_returns_its_messages_in_order(context):
    client, _, headers = context
    body = client.get("/v1/chat/conversations/c1", headers=headers).json()["data"]
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]


def test_an_unknown_conversation_is_404(context):
    client, _, headers = context
    assert client.get("/v1/chat/conversations/nope", headers=headers).status_code == 404


def test_abort_reaches_the_runtime(context):
    client, runtime, headers = context
    assert client.post("/v1/abort", headers=headers).status_code == 200
    assert runtime.chat.aborted == 1
