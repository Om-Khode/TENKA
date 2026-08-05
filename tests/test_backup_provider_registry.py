"""Tests for io/backup/provider.py and the backup_provider_registry."""
import pytest

from assistant.io.backup import BackupProvider, BackupProviderError, backup_provider_registry


class _FakeProvider(BackupProvider):
    name = "fake"

    def __init__(self):
        self._store: dict[str, bytes] = {}

    def is_connected(self) -> bool:
        return True

    def upload(self, blob: bytes, label: str) -> None:
        self._store[label] = blob

    def list_versions(self) -> list[str]:
        return sorted(self._store.keys(), reverse=True)

    def download(self, label: str) -> bytes:
        return self._store[label]

    def delete(self, label: str) -> None:
        del self._store[label]


def test_provider_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BackupProvider()


def test_fake_provider_round_trips():
    p = _FakeProvider()
    p.upload(b"data1", "v1")
    p.upload(b"data2", "v2")
    assert p.list_versions() == ["v2", "v1"]
    assert p.download("v1") == b"data1"
    p.delete("v1")
    assert p.list_versions() == ["v2"]


def test_registry_register_and_require():
    p = _FakeProvider()
    backup_provider_registry.register("fake_register_and_require_test", p)
    assert backup_provider_registry.require("fake_register_and_require_test") is p


def test_registry_require_missing_raises_keyerror():
    with pytest.raises(KeyError):
        backup_provider_registry.require("definitely_nonexistent_provider_xyz")
