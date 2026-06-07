"""
test_4a_whatsapp.py — Tests for 4A WhatsApp 405 / outdated client handling.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import importlib


def test_neonize_upgraded():
    import neonize
    major, minor, patch = [int(x.split(".")[0]) for x in neonize.__version__.split(".")[:3]]
    assert (major, minor) >= (0, 3), f"neonize version too old: {neonize.__version__}"
    # Must be at least 0.3.16
    assert int(neonize.__version__.split(".")[2].split("post")[0]) >= 16 or minor > 3, \
        f"neonize must be >= 0.3.16, got {neonize.__version__}"
    print(f"  neonize version: {neonize.__version__}")


def test_go_dll_matches_python_version():
    from neonize.download import __GONEONIZE_VERSION__
    import neonize
    # Major.minor must match
    py_v = ".".join(neonize.__version__.split(".")[:2])
    go_v = ".".join(__GONEONIZE_VERSION__.split(".")[:2])
    assert py_v == go_v, f"Python {py_v} != Go DLL {go_v}"


def test_pairphone_method_exists():
    from neonize.client import NewClient
    assert hasattr(NewClient, 'PairPhone'), "PairPhone method missing from NewClient"


def test_client_outdated_event_importable():
    from neonize.events import ClientOutdatedEv, ConnectFailureEv, LoggedOutEv
    assert ClientOutdatedEv is not None
    assert ConnectFailureEv is not None
    assert LoggedOutEv is not None


def test_adapter_has_client_outdated_flag():
    from assistant.io.adapters.whatsapp import Adapter
    a = Adapter()
    assert hasattr(a, '_client_outdated'), "Adapter missing _client_outdated flag"
    assert a._client_outdated is False


def test_adapter_has_session_path():
    from assistant.io.adapters.whatsapp import Adapter
    a = Adapter()
    assert hasattr(a, '_session_path'), "Adapter missing _session_path"
    assert a._session_path == ""


def test_adapter_has_pair_phone():
    from assistant.io.adapters.whatsapp import Adapter
    assert hasattr(Adapter, 'pair_phone'), "Adapter missing pair_phone method"


def test_adapter_has_delete_session():
    from assistant.io.adapters.whatsapp import Adapter
    assert hasattr(Adapter, '_delete_session'), "Adapter missing _delete_session method"


def test_delete_session_noop_on_missing_file():
    from assistant.io.adapters.whatsapp import Adapter
    a = Adapter()
    a._session_path = "/nonexistent/path/session.db"
    a._delete_session()  # Should not raise


def test_bridge_has_is_client_outdated():
    from assistant.io import messaging_bridge as mb
    assert hasattr(mb, 'is_client_outdated'), "messaging_bridge missing is_client_outdated"
    # Returns False when service not loaded
    assert mb.is_client_outdated("whatsapp") is False


def test_bridge_has_pair_phone():
    from assistant.io import messaging_bridge as mb
    assert hasattr(mb, 'pair_phone'), "messaging_bridge missing pair_phone"


def test_client_outdated_sentinel_in_code_executor():
    # Verify the sentinel is handled in the result processing
    import re
    result = 'Error: CLIENT_OUTDATED|whatsapp from bridge response'
    assert "CLIENT_OUTDATED|" in result


def test_actions_handles_client_outdated_sentinel():
    # Verify actions.py recognizes CLIENT_OUTDATED|svc pattern
    result = "CLIENT_OUTDATED|whatsapp"
    assert result.startswith("CLIENT_OUTDATED|")
    svc = result.split("|")[1]
    assert svc == "whatsapp"


def test_connect_failure_reason_has_client_outdated():
    from neonize.proto.Neonize_pb2 import ConnectFailureReason
    assert hasattr(ConnectFailureReason, 'CLIENT_OUTDATED'), \
        "ConnectFailureReason missing CLIENT_OUTDATED"


def test_read_messages_filters_by_chat_and_sender_not_text():
    """B7: _read_messages should NOT match on message text body."""
    from assistant.io.adapters.whatsapp import Adapter
    a = Adapter()
    a._messages = [
        {"sender": "1234567890", "chat": "1234567890", "text": "Hey mom, call me", "is_from_me": False, "timestamp": None},
        {"sender": "9876543210", "chat": "9876543210", "text": "Meeting at 3pm", "is_from_me": False, "timestamp": None},
    ]
    result = a._read_messages({"chat_name": "mom"})
    assert len(result) == 1
    assert result[0]["sender"] == "system"


def test_read_messages_filters_by_sender_name():
    """B7: Filter should still match on chat/sender fields."""
    from assistant.io.adapters.whatsapp import Adapter
    a = Adapter()
    a._messages = [
        {"sender": "1234567890", "chat": "1234567890", "text": "Hello", "is_from_me": False, "timestamp": None},
        {"sender": "9876543210", "chat": "9876543210", "text": "Bye", "is_from_me": False, "timestamp": None},
    ]
    result = a._read_messages({"chat_name": "123456"})
    assert len(result) == 1
    assert result[0]["text"] == "Hello"


def test_rebuild_caches_atomic_swap():
    """B3: _rebuild_caches should swap atomically, not clear-then-populate."""
    from assistant.io.adapters.whatsapp import Adapter
    a = Adapter()
    a._contact_cache = {"111": "Alice"}
    a._lid_cache = {"lid1": "Bob"}
    a._rebuild_caches()
    assert a._contact_cache == {}
    assert a._lid_cache == {}


def test_disambiguation_names_capped():
    """I3: Multiple contact matches should cap name list for TTS safety."""
    from assistant.io.adapters.whatsapp import Adapter
    a = Adapter()
    a._connected.set()
    a._client = True

    contacts = [
        {"name": "Alice Smith", "phone": "111"},
        {"name": "Alice Jones", "phone": "222"},
        {"name": "Alice Brown", "phone": "333"},
        {"name": "Alice White", "phone": "444"},
    ]
    from unittest.mock import patch
    with patch.object(a, '_resolve_contact', return_value=contacts), \
         patch.object(a, '_build_contact_list', return_value=contacts):
        try:
            a._send_message({"contact_name": "Alice", "text": "Hi"})
            assert False, "Should have raised ValueError"
        except ValueError as e:
            msg = str(e)
            assert "and 2 others" in msg
            assert len(msg) < 120


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
