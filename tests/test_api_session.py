"""GET /v1/session: who is this device, and what can it do *here*.

Task 5 moved the device credential into an httpOnly cookie, so JavaScript can
no longer read a token out of `localStorage` to decide whether Studio's `/app`
tree should render. This route answers "who am I, and what can I do on this
connection" instead -- 401 when there is no valid credential at all, 200 with
two capability lists otherwise.

The two lists matter for different reasons: `grants` is what the device was
*issued* at pairing, `effective` is what survives the listener's ceiling on
*this* connection. Studio needs both to explain a disabled control honestly --
"your device may, this connection may not" is a different message than "your
device may never" -- so both round-trip here rather than collapsing to one.

`POST /v1/session/cookie` is the second route here and it fixes a real, live
failure: a session that authenticated with `Authorization: Bearer` had working
HTTP and an event socket that could never authenticate at all, because the
socket reads the cookie and only the cookie. Studio showed LIVE -
RECONNECTING forever and no chat reply ever rendered. The tests below pin
what that route may and may not do -- it moves the credential onto the cookie
channel, and it does not mint, re-grant, or enrol anything.
"""
import pytest
from starlette.websockets import WebSocketDisconnect

from assistant.io.api.app import create_app
from assistant.io.api.security import COOKIE_NAME, CSRF_HEADER
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.api_client import BASE_URL, ApiTestClient
from tests.fakes.studio_runtime import build_fake_runtime


def _client(vault: TokenVault, *, policies: dict[int, str]) -> ApiTestClient:
    app = create_app(build_fake_runtime(), vault,
                     origins=["http://localhost:3000"],
                     listener_policies=policies)
    return ApiTestClient(app, base_url=BASE_URL)


def test_session_reports_the_calling_device(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("Pixel 8", frozenset({Capability.CHAT_SEND, Capability.FILES}))
    client = _client(vault, policies={8787: "local"})
    client.cookies.set(COOKIE_NAME, token)
    data = client.get("/v1/session").json()["data"]
    assert data["label"] == "Pixel 8"
    assert set(data["grants"]) == {"chat_send", "files"}
    assert data["policy"] == "local"


def test_session_shows_the_ceiling_separately_from_the_grants(tmp_path):
    """Studio needs both to explain a disabled control: 'your device may, this
    connection may not'."""
    vault = TokenVault(tmp_path)
    token = vault.issue("Pixel 8", frozenset(Capability))
    client = _client(vault, policies={8787: "quick"})
    client.cookies.set(COOKIE_NAME, token)
    data = client.get("/v1/session").json()["data"]
    assert set(data["effective"]) == {"observe"}
    assert len(data["grants"]) > 1


def test_session_is_401_when_unpaired(tmp_path):
    client = _client(TokenVault(tmp_path), policies={8787: "local"})
    assert client.get("/v1/session").status_code == 401


def test_session_needs_no_capability(tmp_path):
    """A device with any single grant can ask who it is -- otherwise the gate
    cannot render for a narrowly-scoped device."""
    vault = TokenVault(tmp_path)
    token = vault.issue("reader", frozenset({Capability.SCREEN}))
    client = _client(vault, policies={8787: "local"})
    client.cookies.set(COOKIE_NAME, token)
    assert client.get("/v1/session").status_code == 200


# ─── POST /v1/session/cookie ─────────────────────────────────────────────
def test_the_exchange_lets_the_socket_authenticate(tmp_path):
    """The whole bug, end to end.

    A session that authenticated with `Authorization: Bearer` had working HTTP
    and a socket that closed 1008 on every attempt, because the socket reads
    the cookie and only the cookie. That is what the user saw as LIVE -
    RECONNECTING with no chat reply ever arriving.

    Asserted by *connecting and reading a frame*, not by finding a
    `Set-Cookie` in the response. A cookie header proves the daemon said
    something; it does not prove the browser would send it back on a handshake
    or that the handshake would accept it. The status frame is the first thing
    a real client waits for, so receiving one is the failure actually being
    fixed.
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    client = _client(vault, policies={8787: "local"})

    # Before: a bearer session has no cookie, and a browser cannot put one on
    # a handshake, so the socket is refused. This is the reported failure.
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/v1/events"):
            pass

    r = client.post("/v1/session/cookie",
                    headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 204

    with client.websocket_connect("/v1/events") as ws:
        assert ws.receive_json()["type"] == "status"


def test_the_exchange_hands_back_the_same_credential(tmp_path):
    """Not a reissue. The cookie carries the string that arrived, verbatim --
    which is why nothing downstream can have been widened: there is nothing
    new to widen."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    client = _client(vault, policies={8787: "local"})
    client.post("/v1/session/cookie", headers={"Authorization": f"Bearer {token}"})
    assert client.cookies.get(COOKIE_NAME) == token


def test_the_exchange_is_refused_on_a_remote_listener(tmp_path):
    """`allow_bearer` is loopback alone, and so is this route.

    The cookie is set deliberately, and the `GET` beside it is the point: the
    credential is *good* on this listener -- 200 -- and the exchange is still
    refused. Without that pair the test would pass just as happily against a
    route that was never reached because the credential itself was rejected,
    proving nothing about the gate it claims to pin.
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    for name in ("tailnet", "funnel", "quick"):
        client = _client(vault, policies={8787: name})
        client.cookies.set(COOKIE_NAME, token)
        assert client.get("/v1/session").status_code == 200, name
        r = client.post("/v1/session/cookie", headers={CSRF_HEADER: "1"})
        assert r.status_code == 401, name
        assert r.json()["detail"] == "unauthorized", name


def test_the_exchange_changes_nothing_about_the_device(tmp_path):
    """Same device id, same grants, same number of rows in the vault.

    The failure this guards against is an implementation that "exchanges" by
    calling `vault.issue()` again -- which would work, would set a usable
    cookie, and would quietly leave a second device record behind that nobody
    paired, that the revoke list cannot explain, and that outlives any
    revocation of the first.
    """
    vault = TokenVault(tmp_path)
    grants = frozenset({Capability.CHAT_SEND, Capability.OBSERVE})
    token = vault.issue("laptop", grants)
    before = vault.devices()
    client = _client(vault, policies={8787: "local"})

    bearer = client.get("/v1/session",
                        headers={"Authorization": f"Bearer {token}"}).json()["data"]
    assert client.post("/v1/session/cookie",
                       headers={"Authorization": f"Bearer {token}"}).status_code == 204
    cookied = client.get("/v1/session").json()["data"]

    assert cookied["deviceId"] == bearer["deviceId"]
    assert cookied["grants"] == bearer["grants"]
    assert cookied["effective"] == bearer["effective"]

    after = vault.devices()
    assert len(after) == len(before) == 1
    assert [d.device_id for d in after] == [d.device_id for d in before]


def test_the_exchange_needs_a_credential(tmp_path):
    """No body, no arguments, no way in without already being authenticated --
    the same 401 shape as every other auth failure."""
    client = _client(TokenVault(tmp_path), policies={8787: "local"})
    r = client.post("/v1/session/cookie")
    assert r.status_code == 401
    assert r.json()["detail"] == "unauthorized"


def test_a_cookie_call_is_an_idempotent_no_op_and_still_needs_csrf(tmp_path):
    """The one case a cross-site page could actually reach.

    It cannot set `Authorization`, so the only way it reaches this route is
    with the ambient cookie -- and there the ordinary CSRF gate applies,
    unchanged, because this is a POST. Refused without the header; with it,
    all it achieves is re-setting the cookie the browser already had, byte for
    byte. That is why the cookie case is a harmless no-op rather than a
    special refusal.
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={8787: "local"})
    client.cookies.set(COOKIE_NAME, token)
    assert client.post("/v1/session/cookie").status_code == 403
    r = client.post("/v1/session/cookie", headers={CSRF_HEADER: "1"})
    assert r.status_code == 204
    # Read off the header rather than the jar: the manually seeded cookie and
    # the one the route set land under different domains in httpx's jar, and
    # `Cookies.get` refuses to pick between two entries of the same name. The
    # header is the claim being made anyway -- that what came back is the same
    # string that went out.
    assert f"{COOKIE_NAME}={token}" in r.headers.get("set-cookie", "")


def test_the_exchange_sets_the_flags_the_helper_chose(tmp_path):
    """`cookie_kwargs(policy)`, not a hand-written header. The host-only /
    SameSite / Secure / Max-Age decisions stay in one place, and this route
    inherits whatever that place decides next."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    client = _client(vault, policies={8787: "local"})
    r = client.post("/v1/session/cookie",
                    headers={"Authorization": f"Bearer {token}"})
    set_cookie = r.headers.get("set-cookie", "")
    assert "HttpOnly" in set_cookie, set_cookie
    assert "SameSite=strict" in set_cookie, set_cookie
    assert "Domain" not in set_cookie, set_cookie
