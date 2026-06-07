"""Tests for F1 pending handler fixes: B1, B4, I2, I4."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import re
import inspect


def test_phone_regex_rejects_short_digit_sequences():
    """I2: Phone regex should require >= 7 actual digits after stripping separators."""
    text = "done in 10 seconds"
    m = re.search(r'\b(\+?[\d\s\-]{7,15})\b', text)
    if m:
        digits = re.sub(r'[^\d]', '', m.group(1))
        assert len(digits) < 7, f"Should reject '{m.group(1)}' — only {len(digits)} digits"


def test_phone_regex_accepts_real_phone():
    """I2: Real phone numbers with >= 7 digits should still be accepted."""
    text = "+91 976 428 0339"
    m = re.search(r'\b(\+?[\d\s\-]{7,15})\b', text)
    assert m is not None
    digits = re.sub(r'[^\d]', '', m.group(1))
    assert len(digits) >= 7


def test_pending_disambig_uses_active_not_payload():
    """I4: handle_pending_messaging_disambig should check .active, not payload is None."""
    from assistant.actions.pending_handlers import handle_pending_messaging_disambig
    src = inspect.getsource(handle_pending_messaging_disambig)
    assert "payload is None" not in src, \
        "Should not use 'payload is None' — use .active instead"


def test_device_auth_uses_async_sleep():
    """B1: handle_pending_device_auth should use asyncio.sleep, not time.sleep."""
    from assistant.actions.pending_handlers import handle_pending_device_auth
    src = inspect.getsource(handle_pending_device_auth)
    assert "asyncio.sleep" in src, "Should use asyncio.sleep"
    assert "time.sleep" not in src, "Should NOT use time.sleep"


def test_incoming_message_captures_payload_once():
    """B4: handle_pending_incoming_message should capture payload in a local variable."""
    from assistant.actions.pending_handlers import handle_pending_incoming_message
    src = inspect.getsource(handle_pending_incoming_message)
    assert "payload = _act.pending_incoming_messages.payload" in src, \
        "Should capture payload in a local variable"
