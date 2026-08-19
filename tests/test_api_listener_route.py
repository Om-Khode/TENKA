"""`GET /v1/listener` -- the one unauthenticated read in this API.

Studio's `/connect` screen has no other way to learn which listener served it
before any credential exists, and it needs that fact to decide whether to
offer the bearer-token exchange (`local` only) at all. The properties pinned
here:

- it answers on every listener, with no credential;
- `allow_bearer` is true only for `local` -- the one policy that accepts an
  `Authorization` header at all;
- `can_pair` is false for `quick` -- the one policy `pair_device` refuses
  outright (spec §5.5) -- and true everywhere else;
- it carries nothing beyond the three documented fields;
- a port nobody declared answers 401, exactly like every other route;
- it is metered on the shared anonymous budget, like other unauthenticated
  traffic.
"""
from __future__ import annotations

from assistant.io.api.app import create_app
from assistant.io.api.listeners import port_for
from assistant.io.api.policy import POLICIES
from tests.fakes.api_client import LOCAL_PORT, ApiTestClient
from tests.fakes.studio_runtime import build_fake_runtime
from assistant.io.api.vault import TokenVault

PORTS: dict[str, int] = {name: port_for(name, LOCAL_PORT)
                         for name in ("local", "tailnet", "funnel", "quick")}
POLICY_REGISTRY: dict[int, str] = {port: name for name, port in PORTS.items()}


def build(tmp_path):
    vault = TokenVault(tmp_path)
    app = create_app(build_fake_runtime(), vault,
                     origins=["http://localhost:3000"],
                     listener_policies=dict(POLICY_REGISTRY))
    return app


def client_on(app, listener: str) -> ApiTestClient:
    return ApiTestClient(app, base_url=f"http://127.0.0.1:{PORTS[listener]}")


def test_it_answers_on_every_listener_with_no_credential(tmp_path):
    app = build(tmp_path)
    for listener in ("local", "tailnet", "funnel", "quick"):
        r = client_on(app, listener).get("/v1/listener")
        assert r.status_code == 200, listener
        assert r.json()["data"]["policy"] == listener


def test_allow_bearer_is_true_only_on_local(tmp_path):
    app = build(tmp_path)
    for listener in ("local", "tailnet", "funnel", "quick"):
        data = client_on(app, listener).get("/v1/listener").json()["data"]
        expected = POLICIES[listener].allow_bearer
        assert data["allowBearer"] == expected, listener
    assert client_on(app, "local").get("/v1/listener").json()["data"]["allowBearer"] is True
    for listener in ("tailnet", "funnel", "quick"):
        assert client_on(app, listener).get(
            "/v1/listener").json()["data"]["allowBearer"] is False, listener


def test_can_pair_is_false_only_on_quick(tmp_path):
    app = build(tmp_path)
    for listener in ("local", "tailnet", "funnel"):
        data = client_on(app, listener).get("/v1/listener").json()["data"]
        assert data["canPair"] is True, listener
    quick_data = client_on(app, "quick").get("/v1/listener").json()["data"]
    assert quick_data["canPair"] is False


def test_can_pair_agrees_with_pair_device_actually_refusing_quick(tmp_path):
    """Not a re-spelling: redeem a real, correct code over `quick` and confirm
    it really is refused, on the same app this route answered `canPair: false`
    for -- so the field is proved to describe what `pair_device` does, not
    merely what a second copy of the condition claims it does."""
    from assistant.io.api.pairing import PairCodeStore
    from assistant.io.api.vault import Capability

    vault = TokenVault(tmp_path)
    store = PairCodeStore()
    app = create_app(build_fake_runtime(), vault,
                     origins=["http://localhost:3000"],
                     listener_policies=dict(POLICY_REGISTRY),
                     pair_store=store)
    code = store.mint("phone", frozenset({Capability.OBSERVE})).code

    assert client_on(app, "quick").get(
        "/v1/listener").json()["data"]["canPair"] is False
    r = client_on(app, "quick").post("/v1/pair", json={"code": code})
    assert r.status_code != 204


def test_the_payload_carries_nothing_beyond_the_three_fields(tmp_path):
    app = build(tmp_path)
    data = client_on(app, "local").get("/v1/listener").json()["data"]
    assert set(data) == {"policy", "allowBearer", "canPair"}


def test_a_port_nobody_declared_is_refused(tmp_path):
    vault = TokenVault(tmp_path)
    app = create_app(build_fake_runtime(), vault,
                     origins=["http://localhost:3000"],
                     listener_policies={})
    client = ApiTestClient(app, base_url=f"http://127.0.0.1:{LOCAL_PORT}")
    assert client.get("/v1/listener").status_code == 401


def test_it_is_metered_on_the_shared_anonymous_budget(tmp_path):
    """Not unmetered: a flood of anonymous reads on one listener eventually
    earns the same 429 an anonymous flood against any other route would."""
    app = build(tmp_path)
    client = client_on(app, "local")
    statuses = [client.get("/v1/listener").status_code for _ in range(130)]
    assert 429 in statuses
