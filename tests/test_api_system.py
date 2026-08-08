"""Telemetry, backup and enrollment — including the restore that must be real."""
import pytest
from fastapi.testclient import TestClient

from assistant.io.api.app import create_app
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.studio_runtime import build_fake_runtime

PHRASE = "amber moss steel gold bone quiet signal drift"


@pytest.fixture()
def context(tmp_path):
    vault = TokenVault(tmp_path)
    runtime = build_fake_runtime()
    client = TestClient(create_app(runtime, vault, origins=["http://localhost:3000"]))
    tokens = {
        "full": vault.issue("studio", frozenset(Capability)),
        "chat": vault.issue("phone", frozenset({Capability.CHAT})),
    }
    return client, runtime, tokens


def head(token):
    return {"Authorization": f"Bearer {token}"}


def test_telemetry_reports_the_meters(context):
    client, _, tokens = context
    body = client.get("/v1/telemetry", headers=head(tokens["full"])).json()["data"]
    assert body["cpu_percent"] == 21.5
    assert body["ram_percent"] == 63.0
    assert body["battery_percent"] == 88.0


def test_telemetry_tolerates_a_machine_with_no_battery(context):
    client, runtime, tokens = context

    async def no_battery():
        from assistant.io.api.runtime import TelemetrySnapshot
        return TelemetrySnapshot(10.0, 20.0, None, "m", 1)

    runtime.system.telemetry = no_battery
    body = client.get("/v1/telemetry", headers=head(tokens["full"])).json()["data"]
    assert body["battery_percent"] is None


def test_backup_state_is_readable(context):
    client, _, tokens = context
    body = client.get("/v1/backup", headers=head(tokens["full"])).json()["data"]
    assert body["provider"] == "google_drive"
    assert body["last_result"] == "ok"


def test_running_a_backup_updates_the_state(context):
    client, runtime, tokens = context
    body = client.post("/v1/backup/run", headers=head(tokens["full"])).json()["data"]
    assert runtime.system.backups_run == 1
    assert body["last_backup_at"] == "2026-08-08T09:15:00Z"


def test_restore_needs_system_control(context):
    client, runtime, tokens = context
    response = client.post("/v1/backup/restore", headers=head(tokens["chat"]),
                           json={"recovery_phrase": PHRASE})
    assert response.status_code == 403
    assert runtime.system.restored_with == []


def test_restore_verifies_the_phrase_server_side(context):
    client, runtime, tokens = context
    bad = client.post("/v1/backup/restore", headers=head(tokens["full"]),
                      json={"recovery_phrase": "too short"})
    assert bad.status_code == 400
    assert runtime.system.restored_with == ["too short"]


def test_a_valid_phrase_restores(context):
    client, _, tokens = context
    response = client.post("/v1/backup/restore", headers=head(tokens["full"]),
                           json={"recovery_phrase": PHRASE})
    assert response.status_code == 200


def test_the_phrase_is_never_echoed_back(context):
    client, _, tokens = context
    for phrase in (PHRASE, "wrong wrong wrong"):
        response = client.post("/v1/backup/restore", headers=head(tokens["full"]),
                               json={"recovery_phrase": phrase})
        assert phrase not in response.text


def test_enrollment_lists_voices_and_faces(context):
    client, _, tokens = context
    body = client.get("/v1/enrollment", headers=head(tokens["full"])).json()["data"]
    assert [v["name"] for v in body["voices"]] == ["primary"]
    assert [f["name"] for f in body["faces"]] == ["Om"]


def test_forgetting_a_face_removes_it(context):
    client, _, tokens = context
    assert client.delete("/v1/enrollment/face/f1",
                         headers=head(tokens["full"])).status_code == 200
    body = client.get("/v1/enrollment", headers=head(tokens["full"])).json()["data"]
    assert body["faces"] == []


def test_forgetting_an_unknown_kind_is_422(context):
    client, _, tokens = context
    assert client.delete("/v1/enrollment/fingerprint/f1",
                         headers=head(tokens["full"])).status_code == 422


def test_enrollment_carries_count_and_last_seen_at_camel_cased(context):
    """count and lastSeenAt must reach the wire as themselves, camelCased,
    with their fixture values -- not dropped, not renamed, not zeroed.
    """
    client, _, tokens = context
    body = client.get("/v1/enrollment", headers=head(tokens["full"])).json()["data"]

    face = body["faces"][0]
    assert face["count"] == 14
    assert face["lastSeenAt"] == "2026-08-06T21:24:00Z"

    voice = body["voices"][0]
    assert voice["count"] == 6
    assert voice["lastSeenAt"] == "2026-08-07T22:10:00Z"


def test_an_unknown_count_survives_as_null_not_zero(context):
    """A real 0 would be a wrong fact stated confidently about an enrollment
    that plainly exists -- speaker_verify exposes no accessor for it today, so
    the fixture's default fake (count=6/14) can never exercise this branch.
    Swapping in an item with count=None proves the field is None-preserving,
    not `item.count or 0`, and that it is present on the wire rather than
    dropped by a response model that excludes None.
    """
    client, runtime, tokens = context
    from assistant.io.api.runtime import EnrolledItem, EnrollmentState

    async def unknown_count():
        return EnrollmentState(
            voices=[EnrolledItem("v1", "primary", "2026-07-01T08:00:00Z",
                                 count=None, last_seen_at="")],
            faces=[],
        )

    runtime.system.enrollment = unknown_count
    body = client.get("/v1/enrollment", headers=head(tokens["full"])).json()["data"]
    voice = body["voices"][0]
    assert "count" in voice
    assert voice["count"] is None
