"""
io/backup/google_drive.py — Google Drive backup provider.

Stores encrypted backup blobs in the app's hidden "appDataFolder" space —
invisible in the user's normal Drive UI, removed automatically if the
user revokes TENKA's access. Talks to the Drive v3 REST API directly via
requests; does not use google-api-python-client (that's for code_executor
Tier-2 scripts, a different concern — see services.json).

OAuth token storage reuses credentials.py + oauth_helper.py under the
service name "google_drive_backup" — kept distinct from any other Google
service TENKA might connect to, since scopes and tokens are independently
managed per capability (same convention gmail/spotify already use).
"""
import logging

import requests

from ... import credentials, oauth_helper
from . import backup_provider_registry
from .provider import BackupProvider, BackupProviderError

logger = logging.getLogger("backup.google_drive")

SERVICE_NAME = "google_drive_backup"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPES = "https://www.googleapis.com/auth/drive.appdata"
REDIRECT_URI = "http://127.0.0.1:8888/callback"

_UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
_FILES_URL = "https://www.googleapis.com/drive/v3/files"
_TIMEOUT = 30


class GoogleDriveBackupProvider(BackupProvider):
    name = "google_drive"

    def is_connected(self) -> bool:
        return credentials.has_credential(SERVICE_NAME) and oauth_helper.has_token(SERVICE_NAME)

    def _access_token(self) -> str:
        """Return a valid bearer token, refreshing first if needed.

        Raises BackupProviderError if not connected or refresh fails.
        """
        if not oauth_helper.has_token(SERVICE_NAME):
            raise BackupProviderError("Google Drive is not connected.")
        if oauth_helper.is_token_expired(SERVICE_NAME):
            client_id = credentials.get_credential(SERVICE_NAME, "client_id")
            client_secret = credentials.get_credential(SERVICE_NAME, "client_secret")
            if not client_id or not client_secret:
                raise BackupProviderError("Missing Google Drive client credentials.")
            if not oauth_helper.refresh_token(SERVICE_NAME, TOKEN_URL, client_id, client_secret):
                raise BackupProviderError("Failed to refresh Google Drive token.")
        token = credentials.get_credential(SERVICE_NAME, "access_token")
        if not token:
            raise BackupProviderError("No Google Drive access token available.")
        return token

    def upload(self, blob: bytes, label: str) -> None:
        token = self._access_token()
        boundary = "tenka-backup-boundary"
        body = (
            f"--{boundary}\r\n"
            f"Content-Type: application/json; charset=UTF-8\r\n\r\n"
            f'{{"name": "{label}", "parents": ["appDataFolder"]}}\r\n'
            f"--{boundary}\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode("utf-8") + blob + f"\r\n--{boundary}--".encode("utf-8")

        try:
            resp = requests.post(
                _UPLOAD_URL,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": f"multipart/related; boundary={boundary}",
                },
                data=body,
                timeout=_TIMEOUT,
            )
        except requests.exceptions.RequestException as e:
            raise BackupProviderError(f"Drive upload failed: {e}") from e
        if resp.status_code >= 300:
            raise BackupProviderError(f"Drive upload failed: {resp.status_code} {resp.text[:200]}")
        logger.info(f"[BACKUP][DRIVE] Uploaded version '{label}'")

    def _list_files(self) -> list[dict]:
        token = self._access_token()
        try:
            resp = requests.get(
                _FILES_URL,
                headers={"Authorization": f"Bearer {token}"},
                params={
                    "spaces": "appDataFolder",
                    "fields": "files(id,name,createdTime)",
                    "orderBy": "createdTime desc",
                    "pageSize": 100,
                },
                timeout=_TIMEOUT,
            )
        except requests.exceptions.RequestException as e:
            raise BackupProviderError(f"Drive list failed: {e}") from e
        if resp.status_code >= 300:
            raise BackupProviderError(f"Drive list failed: {resp.status_code} {resp.text[:200]}")
        return resp.json().get("files", [])

    def list_versions(self) -> list[str]:
        return [f["name"] for f in self._list_files()]

    def _find_file_id(self, label: str) -> str:
        for f in self._list_files():
            if f["name"] == label:
                return f["id"]
        raise BackupProviderError(f"No backup version named '{label}' found.")

    def download(self, label: str) -> bytes:
        token = self._access_token()
        file_id = self._find_file_id(label)
        try:
            resp = requests.get(
                f"{_FILES_URL}/{file_id}",
                headers={"Authorization": f"Bearer {token}"},
                params={"alt": "media"},
                timeout=_TIMEOUT,
            )
        except requests.exceptions.RequestException as e:
            raise BackupProviderError(f"Drive download failed: {e}") from e
        if resp.status_code >= 300:
            raise BackupProviderError(f"Drive download failed: {resp.status_code} {resp.text[:200]}")
        return resp.content

    def delete(self, label: str) -> None:
        token = self._access_token()
        file_id = self._find_file_id(label)
        try:
            resp = requests.delete(
                f"{_FILES_URL}/{file_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=_TIMEOUT,
            )
        except requests.exceptions.RequestException as e:
            raise BackupProviderError(f"Drive delete failed: {e}") from e
        if resp.status_code >= 300 and resp.status_code != 404:
            raise BackupProviderError(f"Drive delete failed: {resp.status_code} {resp.text[:200]}")
        logger.info(f"[BACKUP][DRIVE] Deleted version '{label}'")


backup_provider_registry.register("google_drive", GoogleDriveBackupProvider())
