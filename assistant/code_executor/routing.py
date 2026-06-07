"""routing.py — Goal routing, service detection, and package management."""

import importlib.util
import logging
import subprocess
import sys

import json as _json

logger = logging.getLogger("code_executor")

from .. import service_registry as _sr
from ..core.json_utils import extract_json_object as _extract_json
from .prompts import get_router_system_prompt
from .packages import (
    TIER2_ALLOWED_PACKAGES,
    _PACKAGE_IMPORT_NAMES,
    _IMPORT_TO_PACKAGE,
)


def detect_service_from_packages(requires: list[str]) -> str | None:
    """Detect which OAuth service a task needs from its required packages."""
    for pkg in requires:
        if pkg in _sr.OAUTH_PACKAGE_MAP:
            return _sr.OAUTH_PACKAGE_MAP[pkg]
    return None


def get_oauth_env_map(service: str) -> dict[str, str]:
    """
    Build the env-var mapping for a service's OAuth credentials.
    Pure convention — no service-specific config needed.
    """
    u = service.upper()
    return {
        "access_token":  f"{u}_ACCESS_TOKEN",
        "refresh_token": f"{u}_REFRESH_TOKEN",
        "client_id":     f"{u}_CLIENT_ID",
        "client_secret": f"{u}_CLIENT_SECRET",
    }


def detect_messaging_service(requires: list[str]) -> str | None:
    """Detect if a task needs a messaging bridge service."""
    for pkg in requires:
        if pkg in _sr.DEVICE_AUTH_PACKAGE_MAP:
            return _sr.DEVICE_AUTH_PACKAGE_MAP[pkg]
    return None


def _check_service_blocklist(code: str, service: str | None) -> str | None:
    """
    Check if generated code contains blocked actions for the detected service.
    Returns a BLOCKED message if found, None if clean.
    """
    if not service or service not in _sr.SERVICE_BLOCKED_ACTIONS:
        return None

    for pattern, reason in _sr.SERVICE_BLOCKED_ACTIONS[service]:
        if pattern in code:
            return f"BLOCKED: {reason}"
    return None


def _ensure_packages(packages: list[str]) -> tuple[bool, str]:
    """Install required packages if missing. Only allows whitelisted packages."""
    for pkg in packages:
        if pkg not in TIER2_ALLOWED_PACKAGES:
            return False, f"Package '{pkg}' is not on the approved list."
        import_name = _PACKAGE_IMPORT_NAMES.get(pkg, pkg).replace("-", "_")
        if importlib.util.find_spec(import_name) is not None:
            continue
        logger.info(f"[CODE] Installing '{pkg}'...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60,
            )
            if result.returncode != 0:
                return False, f"Failed to install '{pkg}': {result.stderr.strip()[:200]}"
        except subprocess.TimeoutExpired:
            return False, f"Installation of '{pkg}' timed out."
        except Exception as e:
            return False, f"Installation error for '{pkg}': {e}"
    return True, ""


async def _route_goal(goal: str, llm_func, preference_hints: str = "") -> dict:
    """
    Route a user goal to tier/slug/requires/params via LLM.
    """
    # Preferences ride in the system prompt (native Gemini behavior).
    system_prompt = get_router_system_prompt()
    if preference_hints:
        system_prompt = (
            system_prompt
            + "\n\nUser preferences (apply these when picking a service):\n"
            + preference_hints
        )

    from ..core.datetime_utils import date_context_line
    raw = await llm_func(f"{date_context_line()}\nGoal: {goal}", system_prompt=system_prompt,
                         task_type="intent", json_mode=True, max_tokens=150, temperature=0)
    if raw == "__LLM_UNAVAILABLE__":
        return {"tier": 1, "template_slug": None, "requires": [], "params": {}, "verification_needed": False}
    try:
        clean = _extract_json(raw, sanitize=True, repair=True) or "{}"
        data = _json.loads(clean)
        tier = data.get("tier", 1)
        if tier not in (1, 2, "gui"):
            tier = 1
        return {"tier": tier, "template_slug": data.get("template_slug") or None,
                "requires": data.get("requires", []), "params": data.get("params", {}),
                "verification_needed": bool(data.get("verification_needed", False))}
    except Exception as e:
        logger.warning(f"[CODE] Router parse failed: {e}")
        return {"tier": 1, "template_slug": None, "requires": [], "params": {}, "verification_needed": False}
