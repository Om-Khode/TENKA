"""manifest_schema.py — Pydantic models + YAML serde for manifest-based app manifests.

Per-app YAML lives under SANDBOX_DIR/manifests/<app_id>.yaml. This module
is the only place that knows the file shape. Migration table at the
bottom dispatches old schema_version values forward.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ConfigDict


# ─── Schema constants ──────────────────────────────────────────────────────

CURRENT_SCHEMA_VERSION = 1


# ─── Models ────────────────────────────────────────────────────────────────

class Match(BaseModel):
    model_config = ConfigDict(extra="forbid")
    process_names: list[str] = Field(default_factory=list)
    window_title_patterns: list[str] = Field(default_factory=list)
    url_patterns: list[str] = Field(default_factory=list)


class Selector(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["hotkey", "uia", "vision_reground"]
    # hotkey
    keys: str | None = None
    # uia
    control_type: str | None = None
    automation_id: str | None = None
    parent_chain: list[str] = Field(default_factory=list)
    name_hint: str | None = None
    # vision_reground
    query: str | None = None
    # health/ranking
    weight: float = 1.0
    successes: int = 0
    failures: int = 0


class Captured(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: str = "tenka_promotion"
    timestamp: str
    app_version_seen: str | None = None
    uia_provider: str = "UIA"
    promoted_from_trace_ids: list[str] = Field(default_factory=list)


class Health(BaseModel):
    model_config = ConfigDict(extra="forbid")
    last_success: str | None = None
    # Selector-walk failures from dispatch — bumped by _on_selector_failure on
    # primary (idx=0), reset by any successful dispatch.
    consecutive_failures: int = 0
    # T1-driven correction signals from the user — bumped by record_correction,
    # only reset on demotion. Tracked separately so a successful dispatch in
    # the middle of a correction sequence does not erase the user's intent
    # signal (the spec's correction-driven Voyager demotion only works if
    # corrections accumulate independent of intervening successes).
    consecutive_corrections: int = 0
    total_dispatches: int = 0


class Intent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    display_name: str
    phrases: list[str] = Field(default_factory=list)
    version: int = 1
    handler_selectors: list[Selector] = Field(default_factory=list)
    timeout_ms: int = 3000
    captured: Captured | None = None
    health: Health = Field(default_factory=Health)


class AppManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int
    app_id: str
    display_name: str
    match: Match
    intents: list[Intent] = Field(default_factory=list)


# ─── YAML serde ────────────────────────────────────────────────────────────

def load_manifest_from_yaml(path: Path) -> AppManifest:
    """Read + validate. Raises pydantic.ValidationError on schema drift."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Manifest {path} is not a YAML mapping")
    raw = migrate(raw, raw.get("schema_version", 0), CURRENT_SCHEMA_VERSION)
    return AppManifest.model_validate(raw)


def dump_manifest_to_yaml(manifest: AppManifest, path: Path) -> None:
    """Atomic write via .tmp + os.replace. Diff-friendly (no flow style)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = manifest.model_dump(mode="json", by_alias=False, exclude_none=True)
    text = yaml.safe_dump(payload, default_flow_style=False, sort_keys=False)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ─── Migration ─────────────────────────────────────────────────────────────

def migrate(d: dict, from_version: int, to_version: int) -> dict:
    """Walk d through migrators from from_version → to_version. Identity at v1."""
    if from_version > to_version:
        raise RuntimeError(
            f"Manifest schema_version {from_version} is newer than "
            f"this build (v{to_version}). Refusing to load."
        )
    cur = from_version
    while cur < to_version:
        migrator = _MIGRATORS.get(cur)
        if migrator is None:
            raise RuntimeError(f"No migrator from schema v{cur} to v{cur + 1}")
        d = migrator(d)
        cur += 1
    return d


# Migrator dispatch table. Add new entries as schema evolves.
_MIGRATORS: dict[int, Callable[[dict], dict]] = {
    # No migrators yet — v1 is the first version.
}
