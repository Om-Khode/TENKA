"""The exported schema must describe payloads, not just the envelope.

Before the typed-response rework, every route's response schema resolved to
`Envelope` with `data: Any` -- `openapi.json` described every one of the 27
operations as an untyped object, so a client generating TypeScript from it got
`data: unknown` everywhere. `Envelope[SomePayload]` on each route's return
annotation is supposed to fix that; these tests are the proof, not just the
claim.
"""
from __future__ import annotations

import pytest

from assistant.io.api.app import create_app
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.studio_runtime import build_fake_runtime


@pytest.fixture()
def schema(tmp_path):
    vault = TokenVault(tmp_path)
    app = create_app(build_fake_runtime(), vault, origins=["http://localhost:3000"])
    app.openapi_url = "/openapi.json"
    return app.openapi()


def _resolve(schema: dict, node: dict) -> dict:
    """Follow one `$ref` hop into `components/schemas`, if present."""
    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        return schema["components"]["schemas"][name]
    return node


def _response_schema(schema: dict, path: str, method: str) -> dict:
    op = schema["paths"][path][method]
    body = op["responses"]["200"]["content"]["application/json"]["schema"]
    return _resolve(schema, body)


def test_status_response_resolves_to_a_schema_with_properties(schema):
    envelope = _response_schema(schema, "/v1/status", "get")
    data_schema = _resolve(schema, envelope["properties"]["data"])
    assert "properties" in data_schema, "GET /v1/status still describes data as an untyped object"
    assert set(data_schema["properties"]) == {
        "assistantName", "activeModel", "personality", "busy",
    }


def test_no_route_s_response_data_is_a_bare_untyped_object(schema):
    """Sweep every operation FastAPI's own schema builder resolved paths for.
    A bare `{"type": "object"}` (or no `type`/`properties` at all) for `data`
    is exactly what `Envelope.data: Any` used to produce -- this fails if any
    operation regresses to that.
    """
    bare = []
    for path, methods in schema["paths"].items():
        for method, op in methods.items():
            if method not in ("get", "post", "patch", "delete"):
                continue
            responses = op.get("responses", {})
            success = responses.get("200") or responses.get("202")
            if success is None:
                continue
            body = success.get("content", {}).get("application/json", {}).get("schema")
            if body is None:
                continue
            envelope = _resolve(schema, body)
            data_node = envelope.get("properties", {}).get("data")
            if data_node is None:
                bare.append((path, method))
                continue
            data_schema = _resolve(schema, data_node)
            # A typed payload has either its own `properties` (an object
            # shape) or, for the one union response (GET /v1/memory/{scope}),
            # a `oneOf`/`anyOf` of typed alternatives -- either is a real
            # schema. Only a schema with neither, or the untyped `{}` /
            # `{"type": "object"}` pydantic used to emit for `Any`, counts as
            # bare.
            has_shape = (
                "properties" in data_schema
                or "oneOf" in data_schema
                or "anyOf" in data_schema
            )
            if not has_shape:
                bare.append((path, method))
    assert bare == [], f"these operations still describe data as untyped: {bare}"


def test_envelope_meta_is_still_a_typed_object(schema):
    envelope = _response_schema(schema, "/v1/status", "get")
    meta_schema = _resolve(schema, envelope["properties"]["meta"])
    assert set(meta_schema["properties"]) == {"requestId", "generatedAt"}


def test_enrolled_item_count_is_nullable_in_the_schema_not_just_at_runtime(schema):
    count_schema = schema["components"]["schemas"]["EnrolledItemPayload"]["properties"]["count"]
    variants = {v.get("type") for v in count_schema.get("anyOf", [])}
    assert variants == {"integer", "null"}, (
        "EnrolledItem.count must be described as nullable in the schema -- "
        "a generated client that trusts this schema over a runtime null "
        "check needs the type itself to say `null` is possible"
    )


def test_file_entry_content_kind_is_nullable_but_file_content_s_is_not(schema):
    entry_kind = schema["components"]["schemas"]["FileEntryPayload"]["properties"]["contentKind"]
    content_kind = schema["components"]["schemas"]["FileContentPayload"]["properties"]["contentKind"]
    assert "anyOf" in entry_kind, "a directory's contentKind is null -- the schema must allow it"
    assert "anyOf" not in content_kind, (
        "a file's own content always has a real contentKind -- the schema "
        "must not widen it to nullable"
    )
