"""File routes carry their own grant and refuse anything outside a root."""
import pytest
from fastapi.testclient import TestClient

from assistant.io.api.app import create_app
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.studio_runtime import build_fake_runtime


@pytest.fixture()
def context(tmp_path):
    vault = TokenVault(tmp_path)
    full = vault.issue("studio", frozenset(Capability))
    chat_only = vault.issue("phone", frozenset({Capability.CHAT}))
    client = TestClient(create_app(build_fake_runtime(), vault,
                                   origins=["http://localhost:3000"]))
    return client, {"Authorization": f"Bearer {full}"}, {"Authorization": f"Bearer {chat_only}"}


def test_listing_a_root_returns_its_entries(context):
    client, headers, _ = context
    entries = client.get("/v1/files?path=desktop", headers=headers).json()["data"]["entries"]
    assert [e["name"] for e in entries] == ["notes.md", "captures"]


def test_every_entry_is_keyed_by_its_path(context):
    client, headers, _ = context
    entries = client.get("/v1/files?path=desktop", headers=headers).json()["data"]["entries"]
    assert [e["id"] for e in entries] == ["desktop/notes.md", "desktop/captures"]


def test_a_nested_directory_is_listable(context):
    client, headers, _ = context
    entries = client.get("/v1/files?path=desktop/captures",
                         headers=headers).json()["data"]["entries"]
    assert [e["id"] for e in entries] == ["desktop/captures/shot.png"]


def test_a_directory_is_marked_as_one_and_reports_no_size(context):
    client, headers, _ = context
    entries = client.get("/v1/files?path=desktop", headers=headers).json()["data"]["entries"]
    directory = [e for e in entries if e["name"] == "captures"][0]
    assert directory["kind"] == "dir"
    assert directory["sizeBytes"] == 0


def test_an_unknown_path_is_404(context):
    client, headers, _ = context
    assert client.get("/v1/files?path=nope", headers=headers).status_code == 404


def test_a_missing_path_parameter_is_422(context):
    client, headers, _ = context
    assert client.get("/v1/files", headers=headers).status_code == 422


def test_reading_a_file_returns_its_text_and_kind(context):
    client, headers, _ = context
    body = client.get("/v1/files/content?path=desktop/notes.md",
                      headers=headers).json()["data"]
    assert body["contentKind"] == "text"
    assert body["content"].startswith("# notes")


def test_reading_something_with_no_preview_is_404(context):
    client, headers, _ = context
    assert client.get("/v1/files/content?path=downloads/statement.pdf",
                      headers=headers).status_code == 404


def test_renaming_reads_back_under_the_new_path(context):
    client, headers, _ = context
    response = client.post("/v1/files/rename", headers=headers, json={
        "path": "desktop/notes.md", "new_name": "renamed.md"})
    assert response.status_code == 200
    assert response.json()["data"]["id"] == "desktop/renamed.md"
    entries = client.get("/v1/files?path=desktop", headers=headers).json()["data"]["entries"]
    assert "desktop/renamed.md" in [e["id"] for e in entries]


def test_deleting_removes_the_entry(context):
    client, headers, _ = context
    assert client.request("DELETE", "/v1/files", headers=headers,
                          json={"path": "desktop/notes.md"}).status_code == 200
    entries = client.get("/v1/files?path=desktop", headers=headers).json()["data"]["entries"]
    assert "desktop/notes.md" not in [e["id"] for e in entries]


def test_deleting_something_absent_is_404(context):
    client, headers, _ = context
    assert client.request("DELETE", "/v1/files", headers=headers,
                          json={"path": "desktop/ghost.md"}).status_code == 404


def test_a_chat_only_device_cannot_touch_files(context):
    client, _, chat_only = context
    assert client.get("/v1/files?path=desktop", headers=chat_only).status_code == 403
    assert client.request("DELETE", "/v1/files", headers=chat_only,
                          json={"path": "desktop/notes.md"}).status_code == 403


def test_a_traversing_path_is_refused_by_the_route(context):
    client, headers, _ = context
    for attack in ("desktop/../../outside.txt", "..\\outside.txt", "/etc/passwd"):
        response = client.get("/v1/files/content", headers=headers, params={"path": attack})
        assert response.status_code in (400, 404), f"{attack} answered {response.status_code}"


def test_a_rename_to_a_path_is_refused(context):
    client, headers, _ = context
    response = client.post("/v1/files/rename", headers=headers, json={
        "path": "desktop/notes.md", "new_name": "../escaped.md"})
    assert response.status_code == 400


# ─── roots ───────────────────────────────────────────────────────────────
# Not in the brief's route list. A Studio client rendering a root picker needs
# the live set of roots from the runtime -- a hardcoded ["desktop", "downloads",
# "documents"] on the client side is exactly the app-specific hardcoding THE
# rule forbids, so the route exists even though no task text asked for it by
# name; FileRuntime.roots() was already there for it to call.
def test_roots_lists_the_configured_roots(context):
    client, headers, _ = context
    response = client.get("/v1/files/roots", headers=headers)
    assert response.status_code == 200
    assert response.json()["data"]["roots"] == ["desktop", "documents", "downloads"]


def test_a_chat_only_device_cannot_list_roots(context):
    client, _, chat_only = context
    assert client.get("/v1/files/roots", headers=chat_only).status_code == 403
