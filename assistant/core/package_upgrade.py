"""Generic pip package upgrade utility."""

import importlib
import logging
import subprocess
import sys
from dataclasses import dataclass

logger = logging.getLogger("package_upgrade")


@dataclass(frozen=True)
class UpgradeResult:
    success: bool
    old_version: str | None
    new_version: str | None
    error_msg: str | None


def _get_version(package_name: str) -> str | None:
    importlib.invalidate_caches()
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def upgrade_package(package_name: str, timeout: int = 60) -> UpgradeResult:
    old_version = _get_version(package_name)

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", package_name, "--quiet"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return UpgradeResult(
            success=False,
            old_version=old_version,
            new_version=old_version,
            error_msg=f"pip upgrade timed out after {timeout}s",
        )

    if proc.returncode != 0:
        err = proc.stderr.strip() or f"pip exited with code {proc.returncode}"
        logger.error(f"[UPGRADE] pip install --upgrade {package_name} failed: {err}")
        return UpgradeResult(
            success=False,
            old_version=old_version,
            new_version=old_version,
            error_msg=err,
        )

    new_version = _get_version(package_name)
    if old_version != new_version:
        logger.info(f"[UPGRADE] {package_name}: {old_version} -> {new_version}")
    else:
        logger.info(f"[UPGRADE] {package_name}: already at {old_version}")

    return UpgradeResult(
        success=True,
        old_version=old_version,
        new_version=new_version,
        error_msg=None,
    )
