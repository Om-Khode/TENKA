"""Rate limiting, lockout, audit, and a kill switch that actually kills."""
import pytest
from fastapi.testclient import TestClient

from assistant.io.api.security import RateLimiter
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.api_client import build_api_client
from tests.fakes.studio_runtime import build_fake_runtime


@pytest.fixture()
def context(tmp_path):
    vault = TokenVault(tmp_path)
    client = build_api_client(build_fake_runtime(), vault)
    tokens = {
        "full": vault.issue("studio", frozenset(Capability)),
        "chat": vault.issue("phone", frozenset({Capability.CHAT})),
    }
    return client, vault, tokens


def head(token):
    return {"Authorization": f"Bearer {token}"}


# ─── limiter unit tests ──────────────────────────────────────────────────
def test_the_window_allows_traffic_below_the_cap():
    limiter = RateLimiter()
    assert all(limiter.check("a", now=float(i) * 0.001) for i in range(120))


def test_the_window_refuses_traffic_above_the_cap():
    limiter = RateLimiter()
    for i in range(120):
        limiter.check("a", now=float(i) * 0.001)
    assert limiter.check("a", now=0.2) is False


def test_the_window_slides():
    limiter = RateLimiter()
    for i in range(120):
        limiter.check("a", now=float(i) * 0.001)
    assert limiter.check("a", now=120.0) is True


def test_keys_are_independent():
    limiter = RateLimiter()
    for i in range(120):
        limiter.check("a", now=float(i) * 0.001)
    assert limiter.check("b", now=0.2) is True


def test_repeated_failures_lock_the_key_out():
    limiter = RateLimiter()
    for _ in range(10):
        limiter.record_failure("a", now=0.0)
    assert limiter.check("a", now=0.5) is False


def test_lockout_expires():
    limiter = RateLimiter()
    for _ in range(10):
        limiter.record_failure("a", now=0.0)
    assert limiter.check("a", now=600.0) is True


def test_a_success_clears_the_failure_count():
    limiter = RateLimiter()
    for _ in range(9):
        limiter.record_failure("a", now=0.0)
    limiter.record_success("a")
    for _ in range(9):
        limiter.record_failure("a", now=1.0)
    assert limiter.check("a", now=1.5) is True


# ─── end-to-end ──────────────────────────────────────────────────────────
def test_a_flood_of_bad_tokens_stops_getting_401s(context):
    client, _, _ = context
    codes = {client.get("/v1/status", headers=head("bad")).status_code
             for _ in range(140)}
    assert 429 in codes, "the limiter never engaged over 140 anonymous attempts"


def test_the_audit_log_records_an_authenticated_call(context):
    client, _, tokens = context
    client.get("/v1/status", headers=head(tokens["full"]))
    entries = client.get("/v1/audit", headers=head(tokens["full"])).json()["data"]["entries"]
    assert any(e["path"] == "/v1/status" and e["outcome"] == "200" for e in entries)


def test_the_audit_log_records_a_rejected_call(context):
    client, _, tokens = context
    client.get("/v1/status", headers=head("bad"))
    entries = client.get("/v1/audit", headers=head(tokens["full"])).json()["data"]["entries"]
    assert any(e["outcome"] == "401" for e in entries)


def test_a_rejected_call_is_recorded_without_a_device(context):
    client, _, tokens = context
    client.get("/v1/status", headers=head("bad"))
    entries = client.get("/v1/audit", headers=head(tokens["full"])).json()["data"]["entries"]
    assert any(e["deviceId"] == "-" for e in entries)


def test_the_audit_log_needs_system_control(context):
    client, _, tokens = context
    assert client.get("/v1/audit", headers=head(tokens["chat"])).status_code == 403


def test_the_audit_log_records_the_path_without_its_query_string(context):
    """Corrected from the brief's `test_the_audit_log_never_holds_a_token`,
    which asserted the token wasn't in the *response body* -- a property
    that holds whether `redact_secrets` works, is a no-op, or is deleted
    entirely. Starlette parses a query string into `request.url.query`,
    never `.path`, and the audit middleware only ever records `.path` --
    there was never anything here for `redact_secrets` to strip, so the
    original assertion passed for a reason that had nothing to do with
    redaction. This pins the real, structural mechanism instead: a token
    riding the query string never reaches the recorded path at all.
    """
    client, _, tokens = context
    client.get(f"/v1/status?access_token={tokens['full']}", headers=head(tokens["full"]))
    entries = client.get("/v1/audit", headers=head(tokens["full"])).json()["data"]["entries"]
    match = next(e for e in entries
                 if e["method"] == "GET" and e["path"] == "/v1/status" and e["outcome"] == "200")
    assert "access_token" not in match["path"]
    assert tokens["full"] not in match["path"]


def test_a_secret_shaped_path_segment_is_redacted_in_the_audit_log(context):
    """Where `redact_secrets` genuinely earns its place in the audit
    middleware: a path *parameter* -- unlike a query string -- really does
    reach `request.url.path`. A high-entropy item_id (mixed case + digits,
    24+ chars, shaped like `_BARE` in core/redact.py) must not survive into
    the logged path.
    """
    client, _, tokens = context
    secret_shaped = "AbCdEfGh12345678ZzYyXxWw"
    client.delete(f"/v1/memory/knowledge/{secret_shaped}", headers=head(tokens["full"]))
    entries = client.get("/v1/audit", headers=head(tokens["full"])).json()["data"]["entries"]
    match = next(e for e in entries
                 if e["method"] == "DELETE" and e["path"].startswith("/v1/memory/knowledge/"))
    assert secret_shaped not in match["path"]
    assert "[REDACTED]" in match["path"]


# ─── kill switch ─────────────────────────────────────────────────────────
def test_shutdown_revokes_every_device(tmp_path):
    from assistant.io.api import server
    vault = TokenVault(tmp_path)
    vault.issue("studio", frozenset(Capability))
    vault.issue("phone", frozenset({Capability.CHAT}))
    server.shutdown(None, vault)
    assert vault.devices() == []


# ─── deferred item 1: a guess earns lockout, silence does not ────────────
def test_a_wrong_token_still_locks_its_source_out(context):
    """A credential guess spends the lockout budget exactly as before."""
    client, _, _ = context
    codes = [client.get("/v1/status", headers=head("wrong")).status_code
             for _ in range(12)]
    assert 429 in codes, "a run of wrong tokens never escalated into a lockout"


def test_an_absent_token_never_locks_its_source_out(context):
    """No credential was presented, so there is nothing to have guessed.

    A source that only ever sends bare, tokenless requests must keep
    getting plain 401s (bounded by the sliding window, never by the
    exponential lockout) -- proven here by a valid token from the same
    source succeeding immediately afterward.
    """
    client, _, tokens = context
    codes = [client.get("/v1/status").status_code for _ in range(12)]
    assert set(codes) == {401}
    assert client.get("/v1/status", headers=head(tokens["full"])).status_code == 200


# ─── deferred item 2: authenticated traffic gets its own budget ──────────
def test_a_valid_device_survives_a_flood_of_wrong_tokens_from_its_own_source(context):
    """One caller behind shared NAT must not be able to exhaust another's
    budget. A flood of wrong tokens from this TestClient's one source
    locks *that source* out, but a distinct, valid device token -- checked
    against its own device-keyed budget, never the source's -- must still
    be served.
    """
    client, _, tokens = context
    for _ in range(150):
        client.get("/v1/status", headers=head("bad"))
    assert client.get("/v1/status", headers=head(tokens["full"])).status_code == 200


def test_a_valid_token_is_never_refused_by_an_exhausted_anonymous_window(context):
    """verify() runs before any budget is checked, so a valid token is
    checked against its own device key and never inherits a 429 earned by
    anonymous traffic sharing its source address.
    """
    client, _, tokens = context
    for _ in range(130):
        client.get("/v1/status")  # bare, tokenless -- exhausts the source's window
    assert client.get("/v1/status", headers=head(tokens["full"])).status_code == 200


# ─── deferred item 9: a heavy route gets a tighter budget than CHAT's ────
def test_run_backup_is_throttled_tighter_than_the_shared_budget(context):
    client, _, tokens = context
    codes = [client.post("/v1/backup/run", headers=head(tokens["full"])).status_code
             for _ in range(20)]
    assert 429 in codes, "run_backup never engaged its own, tighter budget"


def test_reading_backup_state_is_unaffected_by_runs_tighter_budget(context):
    client, _, tokens = context
    for _ in range(20):
        client.post("/v1/backup/run", headers=head(tokens["full"]))
    assert client.get("/v1/backup", headers=head(tokens["full"])).status_code == 200


# ─── deferred item 3: inf/nan settings ────────────────────────────────────
def test_settings_patch_rejects_non_finite_floats():
    import math

    from pydantic import ValidationError

    from assistant.io.api.schemas import SettingsPatch

    for bad in (math.inf, -math.inf, math.nan):
        with pytest.raises(ValidationError):
            SettingsPatch(changes={"x": bad})


def test_settings_patch_accepts_an_ordinary_float():
    from assistant.io.api.schemas import SettingsPatch

    patch = SettingsPatch(changes={"volume": 0.75})
    assert patch.changes["volume"] == 0.75


def test_patching_settings_with_a_non_finite_float_is_422(context):
    """httpx's own encoder refuses to serialize inf/nan (`allow_nan=False`),
    so the attack has to be built as raw bytes -- Python's stdlib `json`
    module (what Starlette parses the request body with) accepts `Infinity`
    on the way in even though it will not emit it on the way out, which is
    exactly the asymmetry a hand-crafted client could exploit.
    """
    client, _, tokens = context
    headers = dict(head(tokens["full"]))
    headers["Content-Type"] = "application/json"
    response = client.patch("/v1/settings", headers=headers,
                            content=b'{"changes": {"x": Infinity}}')
    assert response.status_code == 422
