"""
service_registry.py — Service configuration loaded from services.json.

All service dicts, auth class lists, and router prompt fragments live here.
code_executor and actions/pending_handlers import this module at startup.

To add or update a service, edit assistant/services.json.
Schema is versioned — bump schema_version when the shape changes.
"""

import json
from pathlib import Path

_SERVICES_PATH = Path(__file__).parent / "services.json"
_EXPECTED_SCHEMA_VERSION = 1


def _load_services(path: Path = _SERVICES_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"services.json not found at {path}. "
            "Restore it with: git checkout assistant/services.json"
        )
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"services.json is not valid JSON: {exc}") from exc
    version = data.get("schema_version", 0)
    if version != _EXPECTED_SCHEMA_VERSION:
        raise ValueError(
            f"services.json schema_version={version}, "
            f"expected {_EXPECTED_SCHEMA_VERSION}"
        )
    return data


_DATA = _load_services()
_SERVICES: dict = _DATA["services"]

# ─── Package → service maps ────────────────────────────────────────────────

OAUTH_PACKAGE_MAP: dict[str, str] = {
    pkg: name
    for name, svc in _SERVICES.items()
    if svc["auth_type"] == "oauth"
    for pkg in svc["packages"]
}

DEVICE_AUTH_PACKAGE_MAP: dict[str, str] = {
    pkg: name
    for name, svc in _SERVICES.items()
    if svc["auth_type"] == "device"
    for pkg in svc["packages"]
}

# ─── Per-service config (only entries with non-empty values) ───────────────

OAUTH_AUTH_EXTRAS: dict[str, dict[str, str]] = {
    name: svc["auth_extras"]
    for name, svc in _SERVICES.items()
    if svc.get("auth_extras")
}

OAUTH_MIN_SCOPES: dict[str, list[str]] = {
    name: svc["min_scopes"]
    for name, svc in _SERVICES.items()
    if svc.get("min_scopes")
}

SERVICE_BLOCKED_ACTIONS: dict[str, list[tuple[str, str]]] = {
    name: [(p, r) for p, r in svc["blocked_actions"]]
    for name, svc in _SERVICES.items()
    if svc.get("blocked_actions")
}

PACKAGE_ENV_ALIASES: dict[str, dict[str, str]] = {
    pkg: aliases
    for svc in _SERVICES.values()
    for pkg, aliases in svc.get("env_aliases", {}).items()
    if aliases
}

# ─── Derived aggregates ────────────────────────────────────────────────────

ALL_SERVICE_NAMES: frozenset[str] = frozenset(_SERVICES.keys())

ALL_BANNED_AUTH_CLASSES: list[str] = [
    cls
    for svc in _SERVICES.values()
    for cls in svc.get("banned_auth_classes", [])
]

ALL_CONSTRUCTOR_NAMES: list[str] = list(
    _DATA.get("generic_constructor_names", [])
) + [
    name
    for svc in _SERVICES.values()
    for name in svc.get("constructor_names", [])
]

CONSTRUCTOR_SKIP_PATTERN: str = "|".join(
    f"(?:{name})" for name in ALL_CONSTRUCTOR_NAMES
)

DEVELOPER_URLS: dict[str, str] = {
    name: svc["developer_url"]
    for name, svc in _SERVICES.items()
    if svc.get("developer_url")
}

# ─── Router prompt fragments ──────────────────────────────────────────────

_service_hints = [
    svc["router_hint"]
    for svc in _SERVICES.values()
    if svc.get("router_hint")
]
_generic_hints = _DATA.get("generic_router_hints", [])

ROUTER_HINTS: str = (
    "Package rules:\n"
    + "\n".join(f"  - {h}" for h in _service_hints + _generic_hints)
)

def get_service_packages(service: str) -> list[str]:
    """Return the package list for a service, or [] if unknown."""
    svc = _SERVICES.get(service)
    return list(svc["packages"]) if svc else []
