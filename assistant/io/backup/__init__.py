"""io/backup/ — Encrypted, recovery-phrase-protected cloud backup for TENKA.

Sits parallel to domain, alongside io/adapters and io/channels: pure file
I/O, network, and crypto, no storage/ repo queries needed. May import
core/ and config only, per io/'s layering rule.

Importing this package creates the backup_provider_registry singleton.
Provider modules self-register when imported — see Task 3 for the first
one (Google Drive).
"""
from ...core.registry import RegistryBase
from .provider import BackupProvider, BackupProviderError

backup_provider_registry: RegistryBase[BackupProvider] = RegistryBase("backup_provider")
