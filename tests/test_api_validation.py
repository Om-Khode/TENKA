"""The 422 FastAPI builds for a rejected body must never echo the body back.

Pydantic v2's ValidationError.errors() carries an "input" key holding the raw
offending value, and FastAPI's default RequestValidationError handler forwards
exc.errors() verbatim into the response. Left alone, that turns a bounded field
-- a recovery phrase, a chat message, a settings value, a file path -- into
something the 422 prints in full. The fix lives in assistant/io/api/app.py, as
a RequestValidationError handler that strips the value out at the source, so
every route inherits it for free.
"""
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
    token = vault.issue("studio", frozenset(Capability))
    return client, {"Authorization": f"Bearer {token}"}


def test_an_oversized_recovery_phrase_is_422_and_never_echoed(context):
    client, headers = context
    phrase = "z" * 600
    response = client.post("/v1/backup/restore", headers=headers,
                           json={"recoveryPhrase": phrase})
    assert response.status_code == 422
    assert phrase not in response.text
    assert '"input"' not in response.text


def test_a_missing_recovery_phrase_is_422_with_no_input_key(context):
    client, headers = context
    response = client.post("/v1/backup/restore", headers=headers, json={})
    assert response.status_code == 422
    assert '"input"' not in response.text


def test_an_empty_recovery_phrase_is_422_with_no_input_key(context):
    client, headers = context
    response = client.post("/v1/backup/restore", headers=headers,
                           json={"recoveryPhrase": ""})
    assert response.status_code == 422
    assert '"input"' not in response.text


def test_an_oversized_chat_message_is_422_and_never_echoed(context):
    """The same handler, proven on a second route so it is not restore-shaped."""
    client, headers = context
    text = "q" * 9_000
    response = client.post("/v1/chat", headers=headers, json={"text": text})
    assert response.status_code == 422
    assert text not in response.text
    assert '"input"' not in response.text
