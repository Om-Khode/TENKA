"""Tests for io/backup/crypto.py — recovery phrase and encryption."""
import pytest
from cryptography.exceptions import InvalidTag

from assistant.io.backup import crypto


def test_generate_recovery_phrase_is_valid_bip39():
    phrase = crypto.generate_recovery_phrase()
    assert len(phrase.split()) == 12
    assert crypto.is_valid_recovery_phrase(phrase)


def test_generate_recovery_phrase_is_random():
    a = crypto.generate_recovery_phrase()
    b = crypto.generate_recovery_phrase()
    assert a != b


def test_is_valid_recovery_phrase_rejects_garbage():
    assert crypto.is_valid_recovery_phrase("not a real bip39 phrase at all") is False


def test_derive_key_is_deterministic():
    phrase = crypto.generate_recovery_phrase()
    key1 = crypto.derive_key(phrase)
    key2 = crypto.derive_key(phrase)
    assert key1 == key2
    assert len(key1) == 32


def test_derive_key_differs_for_different_phrases():
    key1 = crypto.derive_key(crypto.generate_recovery_phrase())
    key2 = crypto.derive_key(crypto.generate_recovery_phrase())
    assert key1 != key2


def test_encrypt_decrypt_round_trip():
    key = crypto.derive_key(crypto.generate_recovery_phrase())
    plaintext = b"tenka backup archive bytes go here"
    blob = crypto.encrypt(plaintext, key)
    assert blob != plaintext
    assert crypto.decrypt(blob, key) == plaintext


def test_decrypt_with_wrong_key_raises():
    key1 = crypto.derive_key(crypto.generate_recovery_phrase())
    key2 = crypto.derive_key(crypto.generate_recovery_phrase())
    blob = crypto.encrypt(b"secret data", key1)
    with pytest.raises(InvalidTag):
        crypto.decrypt(blob, key2)


def test_decrypt_tampered_blob_raises():
    key = crypto.derive_key(crypto.generate_recovery_phrase())
    blob = bytearray(crypto.encrypt(b"secret data", key))
    blob[-1] ^= 0xFF  # flip a bit in the auth tag
    with pytest.raises(InvalidTag):
        crypto.decrypt(bytes(blob), key)
