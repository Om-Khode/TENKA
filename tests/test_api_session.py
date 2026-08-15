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
"""
from assistant.io.api.app import create_app
from assistant.io.api.security import COOKIE_NAME
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
