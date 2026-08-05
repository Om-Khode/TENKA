"""io/backup/provider.py — BackupProvider interface.

Every cloud backup destination implements this. New provider = one file
+ one self-registration call into backup_provider_registry. No other
code changes — same shape as llm/providers and io/channels.
"""
from abc import ABC, abstractmethod


class BackupProviderError(Exception):
    """Raised for any provider-level failure (auth, network, quota)."""


class BackupProvider(ABC):
    name: str

    @abstractmethod
    def is_connected(self) -> bool:
        """Whether this provider currently has valid stored credentials."""

    @abstractmethod
    def upload(self, blob: bytes, label: str) -> None:
        """Upload an encrypted backup blob under a version label."""

    @abstractmethod
    def list_versions(self) -> list[str]:
        """Return version labels, newest first."""

    @abstractmethod
    def download(self, label: str) -> bytes:
        """Download and return the raw encrypted blob for a version label."""

    @abstractmethod
    def delete(self, label: str) -> None:
        """Delete a version by label."""
