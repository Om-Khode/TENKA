"""io/backup/ — Encrypted, recovery-phrase-protected cloud backup for TENKA.

Sits parallel to domain, alongside io/adapters and io/channels: pure file
I/O, network, and crypto, no storage/ repo queries needed. May import
core/ and config only, per io/'s layering rule.
"""
