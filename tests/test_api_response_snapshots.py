"""Golden-snapshot tests for a few representative routes.

`test_api_response_shapes.py` proves no route gains, loses, or renames a
key -- strong evidence against structural drift, weak evidence against a
*value* or *formatting* drift (a float rendered with different precision, a
bool serialised as `1`/`0`, a null silently coerced to `""`). This file is
the other half: three routes, chosen for exercising nesting and
nullability, each diffed whole against a fixture committed to the repo. Not
every route needs one -- the point is one mechanism that would catch a
formatting change, not a snapshot of the entire API surface.

Chosen routes:
  - `GET /v1/memory/knowledge` -- three-level nesting (entities/facts/
    relationships), five distinct nullable fields (`sourceTurnId` on all
    three, plus `eventAt`/`invalidAt`/`expiresAt`/`verifiedAt` on facts), and
    a taught property with a string, a number, a bool and a null in one dict
    (Finding 1, 2026-08-08 review).
  - `GET /v1/settings` -- every `SettingRow.kind`/`source` Literal value the
    fake runtime can produce, plus the float/int/str/bool union in `value`/
    `default` that a naive re-serialisation could round or coerce.
  - `GET /v1/enrollment` -- `count: int | None` nullable in exactly the way
    the brief called out as hard-won (a missing accessor, not a zero).

Fixtures live at tests/fixtures/api_snapshots/*.json, keyed by the route's
own `data`, not the full envelope -- `meta.requestId`/`generatedAt` are
non-deterministic by design (a fresh UUID and a real timestamp per request)
and are already pinned by their own tests in test_api_read_routes.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.api_client import build_api_client
from tests.fakes.studio_runtime import build_fake_runtime

FIXTURES = Path(__file__).parent / "fixtures" / "api_snapshots"


@pytest.fixture()
def context(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("studio", frozenset(Capability))
    client = build_api_client(build_fake_runtime(), vault)
    return client, {"Authorization": f"Bearer {token}"}


def _golden(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def test_memory_knowledge_matches_its_golden_snapshot(context):
    client, headers = context
    data = client.get("/v1/memory/knowledge", headers=headers).json()["data"]
    assert data == _golden("memory_knowledge")


def test_settings_matches_its_golden_snapshot(context):
    client, headers = context
    data = client.get("/v1/settings", headers=headers).json()["data"]
    assert data == _golden("settings")


def test_enrollment_matches_its_golden_snapshot(context):
    client, headers = context
    data = client.get("/v1/enrollment", headers=headers).json()["data"]
    assert data == _golden("enrollment")
