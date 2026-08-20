"""Chat enters the existing pipeline, one turn at a time."""
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


# ─── Gap 2: a device flooding /v1/chat must not buy unbounded spoken TTS ──
#
# Each message that reaches the pipeline either arms a pending confirmation
# or is refused by `try_arm` -- and a refusal returns the SAME challenge text
# an ordinary first arm would (deliberate, for non-disclosure), which
# `main.py` then hands to TTS: 13-18 seconds of audio per message, observed
# three times in one live-test minute. The shared per-device budget
# `authenticate()` already enforces (120/60s) never comes close to catching
# a message a second. `routes/chat.py` stacks a tighter, route-specific
# `throttle()` budget (10/60s) on `POST /v1/chat` for exactly this.

def test_repeated_chat_sends_from_one_device_are_throttled(context):
    """The mechanism, proven both ways: budget exhausts into an explicit 429
    (never a silent drop -- the device is told, plainly, that it is over
    budget) and the runtime never sees the throttled sends at all -- they
    are refused before `_StudioDispatch.submit()`/TTS is ever reached."""
    client, runtime, headers = context

    for i in range(10):
        response = client.post("/v1/chat", headers=headers,
                               json={"text": f"message {i}"})
        assert response.status_code == 202, (
            f"message {i} was throttled inside the allowed budget: "
            f"{response.status_code} {response.text}")

    over_budget = client.post("/v1/chat", headers=headers,
                              json={"text": "one message too many"})
    assert over_budget.status_code == 429, (
        f"an over-budget chat send was not throttled: {over_budget.status_code}")
    assert over_budget.json()["detail"] == "too many requests"
    assert len(runtime.chat.sent) == 10, (
        "a throttled send still reached the runtime/pipeline -- the budget "
        "did not stop the turn before TTS could speak its challenge")


def test_a_different_device_is_not_throttled_by_anothers_budget(context):
    """Keyed by device (`throttle()`'s own contract), not by route alone or by
    source: one device's flood must not cost a second, unrelated device its
    own budget -- the shared per-source anonymous key exists for the
    *unauthenticated* case only (see `authenticate()`'s own docstring), and
    this budget must not accidentally recreate that collapse for
    authenticated callers."""
    from assistant.io.api.vault import Capability, TokenVault

    client, runtime, headers = context
    vault = client.app.state.auth.vault
    other_token = vault.issue("laptop", frozenset(Capability))
    other_headers = {"Authorization": f"Bearer {other_token}"}

    for i in range(10):
        assert client.post("/v1/chat", headers=headers,
                           json={"text": f"flood {i}"}).status_code == 202
    assert client.post("/v1/chat", headers=headers,
                       json={"text": "one too many"}).status_code == 429

    # The second device's own, untouched budget.
    response = client.post("/v1/chat", headers=other_headers,
                           json={"text": "hello from the other device"})
    assert response.status_code == 202, (
        "a second device was throttled by the first device's budget")
