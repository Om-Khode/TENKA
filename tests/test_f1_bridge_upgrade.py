"""Tests for F1 messaging bridge: auto-upgrade + CONNECT_TIMEOUT."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock


def _make_mock_adapter(connected=False, outdated=False):
    adapter = MagicMock()
    adapter.is_connected.return_value = connected
    adapter._client_outdated = outdated
    adapter._session_path = ""
    return adapter


def test_connect_timeout_returns_sentinel():
    """B6: _connect_service returns CONNECT_TIMEOUT sentinel on timeout, not None."""
    from assistant.io import messaging_bridge as mb

    mock_adapter = _make_mock_adapter(connected=False, outdated=False)
    with patch.object(mb, '_adapters', {"test_svc": mock_adapter}), \
         patch.object(mb, '_threads', {}), \
         patch.object(mb, '_load_adapter', return_value=mock_adapter), \
         patch("os.path.exists", return_value=True), \
         patch("os.path.getsize", return_value=100), \
         patch("time.sleep"), \
         patch("threading.Thread") as mock_thread:
        mock_thread.return_value.start = MagicMock()
        result = mb._connect_service("test_svc")

    assert result is not None
    assert "CONNECT_TIMEOUT|test_svc" == result


def test_auto_upgrade_on_outdated():
    """Auto-upgrade triggers when adapter._client_outdated is True."""
    from assistant.io import messaging_bridge as mb
    from assistant.core.package_upgrade import UpgradeResult

    mock_adapter = _make_mock_adapter(connected=False, outdated=True)
    mock_new_adapter = _make_mock_adapter(connected=True, outdated=False)

    upgrade_result = UpgradeResult(success=True, old_version="0.3.15", new_version="0.3.16", error_msg=None)

    with patch.object(mb, '_adapters', {"whatsapp": mock_adapter}), \
         patch.object(mb, '_threads', {}), \
         patch.object(mb, '_upgrade_attempted', {}), \
         patch.object(mb, '_load_adapter', return_value=mock_new_adapter), \
         patch("assistant.core.package_upgrade.upgrade_package", return_value=upgrade_result), \
         patch("os.path.exists", return_value=True), \
         patch("os.path.getsize", return_value=100), \
         patch("time.sleep"), \
         patch("threading.Thread") as mock_thread:
        mock_thread.return_value.start = MagicMock()
        result = mb._connect_service("whatsapp", packages=["neonize"])

    assert result is None
    mock_adapter.disconnect.assert_called_once()


def test_auto_upgrade_skipped_if_already_attempted():
    """Second 405 after upgrade attempt returns CLIENT_OUTDATED sentinel."""
    from assistant.io import messaging_bridge as mb

    mock_adapter = _make_mock_adapter(connected=False, outdated=True)

    with patch.object(mb, '_adapters', {"whatsapp": mock_adapter}), \
         patch.object(mb, '_threads', {}), \
         patch.object(mb, '_upgrade_attempted', {"whatsapp": True}), \
         patch("os.path.exists", return_value=True), \
         patch("os.path.getsize", return_value=100), \
         patch("time.sleep"), \
         patch("threading.Thread") as mock_thread:
        mock_thread.return_value.start = MagicMock()
        result = mb._connect_service("whatsapp")

    assert result == "CLIENT_OUTDATED|whatsapp"


def test_auto_upgrade_same_version_falls_back():
    """If pip upgrade doesn't change version, fall back to CLIENT_OUTDATED."""
    from assistant.io import messaging_bridge as mb
    from assistant.core.package_upgrade import UpgradeResult

    mock_adapter = _make_mock_adapter(connected=False, outdated=True)
    upgrade_result = UpgradeResult(success=True, old_version="0.3.16", new_version="0.3.16", error_msg=None)

    with patch.object(mb, '_adapters', {"whatsapp": mock_adapter}), \
         patch.object(mb, '_threads', {}), \
         patch.object(mb, '_upgrade_attempted', {}), \
         patch("assistant.core.package_upgrade.upgrade_package", return_value=upgrade_result), \
         patch("os.path.exists", return_value=True), \
         patch("os.path.getsize", return_value=100), \
         patch("time.sleep"), \
         patch("threading.Thread") as mock_thread:
        mock_thread.return_value.start = MagicMock()
        result = mb._connect_service("whatsapp", packages=["neonize"])

    assert result == "CLIENT_OUTDATED|whatsapp"


def test_execute_handles_connect_timeout():
    """execute() returns user-friendly error on CONNECT_TIMEOUT sentinel."""
    from assistant.io import messaging_bridge as mb

    with patch.object(mb, '_adapters', {}), \
         patch.object(mb, '_connect_service', return_value="CONNECT_TIMEOUT|test_svc"):
        result = mb.execute("test_svc", "read_messages")

    assert result["ok"] is False
    assert "timed out" in result["error"].lower()
