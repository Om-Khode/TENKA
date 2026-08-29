"""One place that knows how a test reaches the daemon.

Every API test used to build its own `TestClient(create_app(...))`. That was
fine while a request's permissions came only from its token. It stopped being
fine when the listener started deciding things: policy is keyed on the local
port a connection was accepted on, and the `Host` allow-list rejects anything
that is not a loopback name -- so a client built against `TestClient`'s default
`http://testserver` base URL now arrives on port 80 with `Host: testserver`,
which is neither a registered listener nor an allowed host.

Rather than repeat the base URL and the policy registry in fourteen files,
they say `build_api_client(runtime, vault)` and get a client that looks like a
browser talking to the real loopback daemon. Tests that care about a *different*
listener (the cookie-auth suite) pass `policies=` explicitly.
"""
from __future__ import annotations

from typing import Any, Sequence

from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from assistant.io.api.app import create_app
from assistant.io.api.events import EventHub
from assistant.io.api.runtime import StudioRuntime
from assistant.io.api.vault import TokenVault

# The port `server.serve()` defaults to. Nothing depends on this exact number
# beyond the two loopback origins built from it; it is here so that a test
# asserting on an origin string and the client that sends it cannot drift.
LOCAL_PORT = 8787
BASE_URL = f"http://127.0.0.1:{LOCAL_PORT}"
LOCAL_POLICIES: dict[int, str] = {LOCAL_PORT: "local"}
DEV_ORIGINS = ["http://localhost:3000"]


class ApiTestClient(TestClient):
    """`TestClient`, with its WebSocket base URL actually honouring `base_url`.

    Starlette 1.4.1's `websocket_connect` does `urljoin("ws://testserver", url)`
    -- a hard-coded literal, ignoring the client's own `base_url` entirely. So
    a socket opened through the stock client always arrives on host
    `testserver`, port 80, no matter which listener the same client's HTTP
    requests go to.

    That is unusable now that the listener decides things: the socket would
    land on an unregistered port with a `Host` nothing allows, while every
    HTTP request from the same client landed on the real one. Overriding it
    with an absolute `ws://` URL (which `urljoin` passes through untouched)
    makes the socket arrive exactly where a browser would send it -- the same
    origin the page was served from.
    """

    def websocket_connect(self, url: str,
                          subprotocols: Sequence[str] | None = None,
                          **kwargs: Any) -> WebSocketTestSession:
        # Derived from this client's own `base_url`, not from `LOCAL_PORT`.
        #
        # The original fix replaced Starlette's hard-coded `ws://testserver`
        # with a hard-coded `ws://127.0.0.1:8787`, which is right for every
        # caller that talks to the local listener and silently wrong for any
        # that does not: the socket lands on 8787 while the same client's HTTP
        # requests land on the port it asked for. A test opening a socket on a
        # *different* listener would then be exercising `local` and asserting
        # about something else -- green, and measuring nothing.
        #
        # Every existing caller passes `base_url=BASE_URL`, so this changes no
        # behaviour for any of them.
        if url.startswith("/"):
            base = str(self.base_url).rstrip("/")
            scheme = "wss" if base.startswith("https://") else "ws"
            authority = base.split("://", 1)[1]
            url = f"{scheme}://{authority}{url}"
        return super().websocket_connect(url, subprotocols, **kwargs)


def build_api_client(runtime: StudioRuntime, vault: TokenVault, *,
                     origins: list[str] | None = None,
                     policies: dict[int, str] | None = None,
                     hub: EventHub | None = None,
                     ui_bundle: Any = None,
                     pair_store: Any = None) -> TestClient:
    """`ui_bundle` defaults to None, so every existing caller keeps an app with
    no UI route at all -- the daemon Milestone 5a shipped. Only the UI-serving
    suite passes one.

    `pair_store` is the same shape of escape hatch: left unset, the app builds
    its own private `PairCodeStore`. The pairing suite passes one in when it
    needs to mint a code without going through the loopback-only route, or to
    assert afterwards that the route burned it.
    """
    app = create_app(
        runtime, vault,
        origins=list(DEV_ORIGINS if origins is None else origins),
        hub=hub,
        listener_policies=LOCAL_POLICIES if policies is None else policies,
        ui_bundle=ui_bundle,
        pair_store=pair_store,
    )
    return ApiTestClient(app, base_url=BASE_URL)
