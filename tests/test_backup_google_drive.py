"""Tests for io/backup/google_drive.py — Google Drive backup provider.

Network calls are mocked; this tests the provider's request-building and
response-parsing logic, not live Drive access (that's a manual live test).
"""
from unittest.mock import patch, MagicMock

import pytest
import requests

from assistant.io.backup import backup_provider_registry
from assistant.io.backup.google_drive import GoogleDriveBackupProvider, SERVICE_NAME
from assistant.io.backup.provider import BackupProviderError


@pytest.fixture
def provider():
    return GoogleDriveBackupProvider()


def test_registers_under_google_drive_key():
    assert backup_provider_registry.require("google_drive") is not None


def test_is_connected_false_without_credentials(provider, monkeypatch):
    monkeypatch.setattr("assistant.credentials.has_credential", lambda svc: False)
    assert provider.is_connected() is False


def test_is_connected_true_with_token(provider, monkeypatch):
    monkeypatch.setattr("assistant.credentials.has_credential", lambda svc: True)
    monkeypatch.setattr("assistant.oauth_helper.has_token", lambda svc: True)
    assert provider.is_connected() is True


@patch("assistant.io.backup.google_drive.requests.post")
def test_upload_posts_multipart_to_appdata(mock_post, provider, monkeypatch):
    monkeypatch.setattr(provider, "_access_token", lambda: "fake-token")
    mock_post.return_value = MagicMock(status_code=200, json=lambda: {"id": "file123"})
    provider.upload(b"encrypted-bytes", "20260802T120000Z")

    assert mock_post.called
    args, kwargs = mock_post.call_args
    assert "uploadType=multipart" in args[0]
    assert kwargs["headers"]["Authorization"] == "Bearer fake-token"


@patch("assistant.io.backup.google_drive.requests.get")
def test_list_versions_returns_names_newest_first(mock_get, provider, monkeypatch):
    monkeypatch.setattr(provider, "_access_token", lambda: "fake-token")
    mock_get.return_value = MagicMock(status_code=200, json=lambda: {
        "files": [
            {"id": "f2", "name": "20260802T130000Z"},
            {"id": "f1", "name": "20260802T120000Z"},
        ]
    })
    versions = provider.list_versions()
    assert versions == ["20260802T130000Z", "20260802T120000Z"]


@patch("assistant.io.backup.google_drive.requests.get")
def test_download_raises_when_label_not_found(mock_get, provider, monkeypatch):
    monkeypatch.setattr(provider, "_access_token", lambda: "fake-token")
    mock_get.return_value = MagicMock(status_code=200, json=lambda: {"files": []})
    with pytest.raises(BackupProviderError):
        provider.download("nonexistent")


# ─── Network-level failures ────────────────────────────────────────────
# requests.exceptions.ConnectionError / Timeout never produce a response
# object, so they can't hit the status_code >= 300 branch — each call site
# needs its own try/except to convert them into BackupProviderError.


@patch("assistant.io.backup.google_drive.requests.post")
def test_upload_wraps_connection_error(mock_post, provider, monkeypatch):
    monkeypatch.setattr(provider, "_access_token", lambda: "fake-token")
    mock_post.side_effect = requests.exceptions.ConnectionError("no route to host")
    with pytest.raises(BackupProviderError):
        provider.upload(b"encrypted-bytes", "20260802T120000Z")


@patch("assistant.io.backup.google_drive.requests.get")
def test_list_files_wraps_timeout(mock_get, provider, monkeypatch):
    monkeypatch.setattr(provider, "_access_token", lambda: "fake-token")
    mock_get.side_effect = requests.exceptions.Timeout("timed out")
    with pytest.raises(BackupProviderError):
        provider.list_versions()


@patch("assistant.io.backup.google_drive.requests.get")
def test_download_wraps_connection_error_on_media_fetch(mock_get, provider, monkeypatch):
    monkeypatch.setattr(provider, "_access_token", lambda: "fake-token")
    # First get() (inside _find_file_id -> _list_files) succeeds; second
    # get() (the actual media download) fails at the network level.
    mock_get.side_effect = [
        MagicMock(status_code=200, json=lambda: {"files": [{"id": "f1", "name": "v1"}]}),
        requests.exceptions.ConnectionError("connection reset"),
    ]
    with pytest.raises(BackupProviderError):
        provider.download("v1")


@patch("assistant.io.backup.google_drive.requests.delete")
@patch("assistant.io.backup.google_drive.requests.get")
def test_delete_wraps_connection_error(mock_get, mock_delete, provider, monkeypatch):
    monkeypatch.setattr(provider, "_access_token", lambda: "fake-token")
    mock_get.return_value = MagicMock(status_code=200, json=lambda: {"files": [{"id": "f1", "name": "v1"}]})
    mock_delete.side_effect = requests.exceptions.ConnectionError("no route to host")
    with pytest.raises(BackupProviderError):
        provider.delete("v1")
