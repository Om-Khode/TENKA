"""Tests for core.package_upgrade — generic pip upgrade utility."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock
import subprocess


def test_upgrade_result_fields():
    from assistant.core.package_upgrade import UpgradeResult
    r = UpgradeResult(success=True, old_version="1.0", new_version="1.1", error_msg=None)
    assert r.success is True
    assert r.old_version == "1.0"
    assert r.new_version == "1.1"
    assert r.error_msg is None


def test_upgrade_success():
    from assistant.core.package_upgrade import upgrade_package

    with patch("assistant.core.package_upgrade._get_version", side_effect=["1.0.0", "1.1.0"]), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = upgrade_package("fakepkg")

    assert result.success is True
    assert result.old_version == "1.0.0"
    assert result.new_version == "1.1.0"
    assert result.error_msg is None
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == sys.executable
    assert "--upgrade" in cmd
    assert "fakepkg" in cmd


def test_upgrade_failure_nonzero_exit():
    from assistant.core.package_upgrade import upgrade_package

    with patch("assistant.core.package_upgrade._get_version", return_value="1.0.0"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="ERROR: no matching distribution")
        result = upgrade_package("fakepkg")

    assert result.success is False
    assert "no matching distribution" in result.error_msg


def test_upgrade_timeout():
    from assistant.core.package_upgrade import upgrade_package

    with patch("assistant.core.package_upgrade._get_version", return_value="1.0.0"), \
         patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="pip", timeout=60)):
        result = upgrade_package("fakepkg", timeout=60)

    assert result.success is False
    assert "timed out" in result.error_msg.lower()


def test_upgrade_same_version():
    from assistant.core.package_upgrade import upgrade_package

    with patch("assistant.core.package_upgrade._get_version", return_value="1.0.0"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = upgrade_package("fakepkg")

    assert result.success is True
    assert result.old_version == "1.0.0"
    assert result.new_version == "1.0.0"


def test_upgrade_unknown_package():
    from assistant.core.package_upgrade import upgrade_package

    with patch("assistant.core.package_upgrade._get_version", return_value=None), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = upgrade_package("fakepkg")

    assert result.success is True
    assert result.old_version is None


def test_get_service_packages():
    from assistant.service_registry import get_service_packages
    pkgs = get_service_packages("whatsapp")
    assert pkgs == ["neonize"]


def test_get_service_packages_unknown():
    from assistant.service_registry import get_service_packages
    pkgs = get_service_packages("nonexistent_service")
    assert pkgs == []
