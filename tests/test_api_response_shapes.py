"""One sweep pinning every route's exact key set.

Per-route test files already assert individual keys and values against the
fake runtime's fixtures; this file's job is narrower and different: fail if
any route's response *gains, loses, or renames* a key relative to the shape
the typed-response rework (`payloads.py`) declares. A route that starts
returning an extra field, drops one, or renames one is exactly the kind of
drift the whole point of typed responses is to catch at review time instead
of in a generated TypeScript client months later -- this is that catch,
exercised end to end through the real route + `response_model` validation,
not just against the payload models in isolation.
"""
from __future__ import annotations

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


def _keys(d: dict) -> set:
    return set(d.keys())


# ─── envelope shape, every response ───────────────────────────────────────
def test_every_envelope_has_exactly_data_and_meta(context):
    client, _, headers = context
    body = client.get("/v1/status", headers=headers).json()
    assert _keys(body) == {"data", "meta"}
    assert _keys(body["meta"]) == {"requestId", "generatedAt"}


# ─── status ────────────────────────────────────────────────────────────────
def test_status_keys(context):
    client, _, headers = context
    data = client.get("/v1/status", headers=headers).json()["data"]
    assert _keys(data) == {"assistantName", "activeModel", "personality", "busy"}


# ─── memory ──────────────────────────────────────────────────────────────
def test_knowledge_keys(context):
    client, _, headers = context
    data = client.get("/v1/memory/knowledge", headers=headers).json()["data"]
    assert _keys(data) == {"entities", "facts", "relationships"}
    assert _keys(data["entities"][0]) == {
        "id", "type", "canonicalName", "displayName", "properties",
        "source", "confidence", "createdAt", "updatedAt", "sourceTurnId",
    }
    assert _keys(data["facts"][0]) == {
        "id", "subjectId", "predicate", "object", "confidence", "source",
        "eventAt", "invalidAt", "expiresAt", "verifiedAt", "createdAt", "sourceTurnId",
    }
    assert _keys(data["relationships"][0]) == {
        "id", "fromId", "toId", "type", "properties", "confidence", "source", "sourceTurnId",
    }


def test_preferences_keys(context):
    client, _, headers = context
    data = client.get("/v1/memory/preferences", headers=headers).json()["data"]
    assert _keys(data) == {"preferences"}
    pref = data["preferences"][0]
    assert _keys(pref) == {"key", "value", "updatedAt", "history"}
    assert _keys(pref["history"][0]) == {"value", "changedAt"}


def test_procedures_keys(context):
    client, _, headers = context
    data = client.get("/v1/memory/procedures", headers=headers).json()["data"]
    assert _keys(data) == {"procedures"}
    assert _keys(data["procedures"][0]) == {"id", "name", "steps", "taughtAt", "runCount"}


def test_forget_item_keys(context):
    client, _, headers = context
    data = client.delete("/v1/memory/preferences/reading_pace", headers=headers).json()["data"]
    assert _keys(data) == {"forgotten"}


def test_forget_all_keys(context):
    client, _, headers = context
    data = client.delete("/v1/memory", headers=headers).json()["data"]
    assert _keys(data) == {"removed"}


# ─── settings and personality ─────────────────────────────────────────────
def test_settings_keys(context):
    client, _, headers = context
    data = client.get("/v1/settings", headers=headers).json()["data"]
    assert _keys(data) == {"rows"}
    assert _keys(data["rows"][0]) == {
        "key", "group", "description", "kind", "value", "default",
        "needsRestart", "source", "options",
    }


def test_personality_keys(context):
    client, _, headers = context
    for response in (
        client.get("/v1/personality", headers=headers),
        client.patch("/v1/personality", headers=headers, json={"base": "dry"}),
        client.post("/v1/personality/reset", headers=headers),
    ):
        assert _keys(response.json()["data"]) == {"base", "available", "traits", "sampleLine"}


def test_save_settings_keys(context):
    client, _, headers = context
    data = client.patch("/v1/settings", headers=headers,
                        json={"changes": {"followup_timer": 6.0}}).json()["data"]
    assert _keys(data) == {"saved", "rejected", "restartRequired"}


# ─── files ───────────────────────────────────────────────────────────────
def test_files_roots_keys(context):
    client, _, headers = context
    data = client.get("/v1/files/roots", headers=headers).json()["data"]
    assert _keys(data) == {"roots"}


def test_files_listing_keys(context):
    client, _, headers = context
    data = client.get("/v1/files?path=desktop", headers=headers).json()["data"]
    assert _keys(data) == {"path", "entries"}
    assert _keys(data["entries"][0]) == {
        "id", "name", "kind", "sizeBytes", "modifiedAt", "contentKind",
    }


def test_files_content_keys(context):
    client, _, headers = context
    data = client.get("/v1/files/content?path=desktop/notes.md", headers=headers).json()["data"]
    assert _keys(data) == {"id", "contentKind", "content", "language", "truncated"}


def test_files_rename_keys(context):
    client, _, headers = context
    data = client.post("/v1/files/rename", headers=headers,
                       json={"path": "desktop/notes.md", "newName": "renamed.md"}).json()["data"]
    assert _keys(data) == {"id", "name", "kind", "sizeBytes", "modifiedAt", "contentKind"}


def test_files_delete_keys(context):
    client, _, headers = context
    data = client.request("DELETE", "/v1/files", headers=headers,
                          json={"path": "desktop/notes.md"}).json()["data"]
    assert _keys(data) == {"deleted"}


# ─── commands ────────────────────────────────────────────────────────────
def test_commands_catalogue_keys(context):
    client, _, headers = context
    data = client.get("/v1/commands", headers=headers).json()["data"]
    assert _keys(data) == {"commands"}
    assert _keys(data["commands"][0]) == {
        "commandId", "label", "description", "destructive", "requiredGrant",
    }


def test_commands_run_keys(context):
    client, _, headers = context
    data = client.post("/v1/commands/volume_up/run", headers=headers).json()["data"]
    assert _keys(data) == {"commandId", "message"}


# ─── chat ────────────────────────────────────────────────────────────────
def test_chat_send_keys(context):
    client, _, headers = context
    data = client.post("/v1/chat", headers=headers, json={"text": "hi"}).json()["data"]
    assert _keys(data) == {"turnId", "conversationId"}


def test_chat_conversations_keys(context):
    client, _, headers = context
    data = client.get("/v1/chat/conversations", headers=headers).json()["data"]
    assert _keys(data) == {"conversations"}
    assert _keys(data["conversations"][0]) == {
        "conversationId", "title", "updatedAt", "messageCount",
    }


def test_chat_conversation_detail_keys(context):
    client, _, headers = context
    data = client.get("/v1/chat/conversations/c1", headers=headers).json()["data"]
    assert _keys(data) == {"conversationId", "title", "messages"}
    assert _keys(data["messages"][0]) == {"messageId", "role", "text", "createdAt", "intent"}


def test_abort_keys(context):
    client, _, headers = context
    data = client.post("/v1/abort", headers=headers).json()["data"]
    assert _keys(data) == {"aborted"}


# ─── system ──────────────────────────────────────────────────────────────
def test_telemetry_keys(context):
    client, _, headers = context
    data = client.get("/v1/telemetry", headers=headers).json()["data"]
    assert _keys(data) == {"cpuPercent", "ramPercent", "batteryPercent", "activeModel", "uptimeSeconds"}


def test_backup_state_keys(context):
    client, _, headers = context
    for response in (
        client.get("/v1/backup", headers=headers),
        client.post("/v1/backup/run", headers=headers),
    ):
        assert _keys(response.json()["data"]) == {
            "enabled", "provider", "lastBackupAt", "lastResult", "sizeBytes",
        }


def test_backup_restore_keys(context):
    client, _, headers = context
    data = client.post("/v1/backup/restore", headers=headers,
                       json={"recoveryPhrase": "amber moss steel gold bone quiet signal drift"}
                       ).json()["data"]
    assert _keys(data) == {"restored"}


def test_enrollment_keys(context):
    client, _, headers = context
    data = client.get("/v1/enrollment", headers=headers).json()["data"]
    assert _keys(data) == {"voices", "faces"}
    assert _keys(data["voices"][0]) == {"itemId", "name", "enrolledAt", "count", "lastSeenAt"}
    assert _keys(data["faces"][0]) == {"itemId", "name", "enrolledAt", "count", "lastSeenAt"}


def test_forget_enrolled_keys(context):
    client, _, headers = context
    data = client.delete("/v1/enrollment/face/f1", headers=headers).json()["data"]
    assert _keys(data) == {"forgotten", "kind"}


def test_audit_keys(context):
    client, _, headers = context
    client.get("/v1/status", headers=headers)  # generate at least one entry
    data = client.get("/v1/audit", headers=headers).json()["data"]
    assert _keys(data) == {"entries"}
    assert _keys(data["entries"][0]) == {"at", "deviceId", "method", "path", "outcome"}
