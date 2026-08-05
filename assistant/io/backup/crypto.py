"""
io/backup/crypto.py — Recovery-phrase-based encryption for TENKA backups.

The recovery phrase is a BIP39 mnemonic, generated once and shown to the
user exactly once. It is never persisted anywhere. Key derivation is
deterministic from the phrase text, so the same phrase always yields the
same AES key — across machines, across reinstalls.
"""
import hashlib
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from mnemonic import Mnemonic

_WORDLIST_LANGUAGE = "english"
_MNEMONIC_STRENGTH_BITS = 128  # -> 12 words
_KDF_SALT = hashlib.sha256(b"tenka.backup.kdf.salt.v1").digest()
_KEY_LENGTH = 32  # AES-256


def generate_recovery_phrase() -> str:
    """Generate a fresh 12-word BIP39 recovery phrase."""
    return Mnemonic(_WORDLIST_LANGUAGE).generate(strength=_MNEMONIC_STRENGTH_BITS)


def is_valid_recovery_phrase(phrase: str) -> bool:
    """Check a phrase is a well-formed BIP39 mnemonic (checksum included)."""
    try:
        return Mnemonic(_WORDLIST_LANGUAGE).check(phrase.strip())
    except Exception:
        return False


def derive_key(recovery_phrase: str) -> bytes:
    """Derive a 32-byte AES-256 key from a recovery phrase via scrypt.

    The salt is a fixed, non-secret constant — its job is domain
    separation, not secrecy, so nothing per-install needs to be stored.
    """
    kdf = Scrypt(salt=_KDF_SALT, length=_KEY_LENGTH, n=2**14, r=8, p=1)
    return kdf.derive(recovery_phrase.strip().encode("utf-8"))


def encrypt(data: bytes, key: bytes) -> bytes:
    """AES-256-GCM encrypt. Returns nonce (12 bytes) + ciphertext + tag."""
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    return nonce + aesgcm.encrypt(nonce, data, None)


def decrypt(blob: bytes, key: bytes) -> bytes:
    """AES-256-GCM decrypt.

    Raises cryptography.exceptions.InvalidTag on a wrong key or
    tampered/corrupted data — never returns partial or garbage plaintext.
    """
    nonce, ciphertext = blob[:12], blob[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)
