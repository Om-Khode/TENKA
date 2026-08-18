# tests/test_6b_transport_adapters.py
"""Milestone 6b Task 6 -- the transport registry and adapter contract.

This file's registry half is self-contained: no provider module
(`tailscale.py`, `cloudflare.py`) exists yet, so every test here drives the
registry directly against a test-double adapter rather than a real one.
Tasks 7 and 8 extend this file with the provider-specific tests named in
their own briefs.

Task 7 adds the Tailscale-specific tests below (`tailnet` and `funnel`):
command construction, the KI-17 layer-2 preflight, and hostname recognition.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from assistant.io.api.transports import TransportRegistry, transport_registry
from assistant.io.api.transports.base import TransportAdapter, TransportSession
from assistant.io.api.transports.tailscale import (
    FunnelAdapter,
    TailnetAdapter,
    parse_serve_status,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "tailscale"


class _FakeAdapter:
    """A minimal `TransportAdapter` double: just enough to register."""

    def __init__(self, name: str) -> None:
        self.name = name

    def command(self, port: int) -> list[str]:
        return ["true", str(port)]

    def hostname_from(self, line: str) -> str | None:
        return None

    def preflight(self, port: int) -> str | None:
        return None

    def stop_command(self, port: int) -> list[str] | None:
        return None

    def status_command(self, port: int) -> list[str] | None:
        return None


@pytest.fixture(autouse=True)
def _snapshot_registry():
    """Every test gets a clean registry and leaves none of its own fakes
    behind -- the same isolation `test_provider_registry.py` uses for
    `provider_registry`."""
    snapshot = transport_registry.list_all()
    transport_registry.reset()
    yield
    transport_registry.reset()
    for name, adapter in snapshot.items():
        transport_registry.register(name, adapter)


def test_an_adapter_registers_under_its_own_policy_name():
    assert isinstance(transport_registry, TransportRegistry)
    adapter = _FakeAdapter("tailnet")
    transport_registry.register("tailnet", adapter)
    assert transport_registry.get("tailnet") is adapter
    assert isinstance(adapter, TransportAdapter)


def test_registering_a_name_that_is_not_a_policy_is_refused():
    with pytest.raises(ValueError):
        transport_registry.register("carrier_pigeon", _FakeAdapter("carrier_pigeon"))
    assert transport_registry.get("carrier_pigeon") is None


def test_registering_the_same_name_twice_is_refused():
    transport_registry.register("funnel", _FakeAdapter("funnel"))
    with pytest.raises(ValueError):
        transport_registry.register("funnel", _FakeAdapter("funnel"))


def test_local_can_never_be_registered_as_a_transport():
    """KI-17 sibling: `local` is the loopback listener and has no tunnel. An
    adapter claiming that name would be handed local's port -- and local's
    full policy, admin and bearer and every capability included."""
    with pytest.raises(ValueError):
        transport_registry.register("local", _FakeAdapter("local"))
    assert transport_registry.get("local") is None


def test_an_unknown_name_resolves_to_nothing():
    assert transport_registry.get("nope") is None


def test_names_are_stable_and_sorted():
    transport_registry.register("quick", _FakeAdapter("quick"))
    transport_registry.register("funnel", _FakeAdapter("funnel"))
    transport_registry.register("tailnet", _FakeAdapter("tailnet"))
    assert transport_registry.names() == ["funnel", "quick", "tailnet"]
    # Stable: calling again in a different registration order still sorts.
    assert transport_registry.names() == sorted(transport_registry.names())


def _make_session(*, hostname: str | None) -> TransportSession:
    """A bare `TransportSession` for exercising `.url` -- `process`, `sock`
    and `serve_task` are never touched by the property, so plain sentinels
    stand in for the real runtime objects a `TransportManager` (Task 9)
    would supply."""
    return TransportSession(
        policy_name="funnel",
        port=8789,
        owner="session-1",
        process=object(),
        sock=object(),
        serve_task=object(),
        hostname=hostname,
    )


def test_url_is_none_before_a_hostname_is_announced():
    assert _make_session(hostname=None).url is None


def test_url_is_the_https_hostname_once_announced():
    session = _make_session(hostname="laptop.tail1234.ts.net")
    assert session.url == "https://laptop.tail1234.ts.net"


# ─── Task 7: tailnet / funnel command construction ───────────────────────────

def test_the_tailnet_command_targets_the_port_it_is_given():
    argv = TailnetAdapter().command(8788)
    assert argv[0] == "tailscale"
    assert "serve" in argv
    assert "http://127.0.0.1:8788" in argv
    # A different port produces a different argv -- the port is not baked in
    # as a constant, it comes from the caller's integer.
    assert "http://127.0.0.1:9999" in TailnetAdapter().command(9999)


def test_the_funnel_command_is_the_top_level_funnel_verb_not_a_serve_flag():
    """Spec and the 6a.5 `policy.py` comment both record it: there is no
    `tailscale serve --funnel`. `funnel` is its own top-level command."""
    argv = FunnelAdapter().command(8789)
    assert argv[0] == "tailscale"
    assert argv[1] == "funnel"
    assert "serve" not in argv
    assert "--funnel" not in argv
    assert "http://127.0.0.1:8789" in argv


def test_the_two_transports_never_share_a_public_port():
    """Fix round 1, F2: Tailscale keys a serve/funnel mapping on the public
    `--https` port, not on the local target it forwards to. Both adapters
    defaulting to 443 would let starting one silently overwrite the other's
    mapping -- and this milestone requires both to run at once, each with
    its own capability ceiling. Pins the exact assignment, not only that
    they differ -- a swap (tailnet 443 / funnel 8443) would pass a mere
    inequality check but still be wrong (fix round 2 minor)."""
    tailnet_argv = TailnetAdapter().command(8788)
    funnel_argv = FunnelAdapter().command(8789)

    def _https_port(argv: list[str]) -> str:
        return argv[argv.index("--https") + 1]

    tailnet_port = _https_port(tailnet_argv)
    funnel_port = _https_port(funnel_argv)
    assert tailnet_port != funnel_port
    assert tailnet_port == "8443"
    assert funnel_port == "443"
    # Funnel is restricted by Tailscale itself to 443, 8443 or 10000
    # (https://tailscale.com/kb/1223/funnel).
    assert funnel_port in {"443", "8443", "10000"}


def test_a_stop_names_the_same_public_port_its_start_claimed():
    """Fix round 2, Important 3: the argv with the highest blast radius in
    this module had zero coverage. Derives both values from `command()` and
    `stop_command()`'s own output -- never from the module constants -- so
    a future edit that lets a `command()`/`stop_command()` pair drift apart
    fails this test rather than silently leaving `off` pointed at the wrong
    mapping."""
    def _https_port(argv: list[str]) -> str:
        return argv[argv.index("--https") + 1]

    for adapter_cls in (TailnetAdapter, FunnelAdapter):
        adapter = adapter_cls()
        start_port = _https_port(adapter.command(8788))
        stop_argv = adapter.stop_command(8788)
        assert stop_argv is not None
        assert _https_port(stop_argv) == start_port

    assert _https_port(TailnetAdapter().stop_command(8788)) != _https_port(
        FunnelAdapter().stop_command(8789)
    )


def test_a_non_numeric_string_is_rejected_before_it_can_reach_the_command_line():
    """Renamed from `test_no_caller_supplied_string_reaches_the_command_line`
    (fix round 2 minor): that name overclaimed -- `int("8788")` succeeds, so
    a numeric *string* does reach `command()`, normalised through `int()`.
    What actually must hold, and what this proves, is narrower: a
    non-numeric string is rejected outright rather than formatted into the
    argv (spec §8's subprocess-injection row)."""
    malicious = "8788; rm -rf /"
    with pytest.raises((TypeError, ValueError)):
        TailnetAdapter().command(malicious)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        FunnelAdapter().command(malicious)  # type: ignore[arg-type]


# ─── Task 7: preflight -- KI-17 layer 2 ──────────────────────────────────────
#
# Ports used consistently across these tests, matching `listeners.py`'s real
# offsets from a base port: local=8787, tailnet's own local target=8788,
# funnel's own local target=8789; tailnet's public port=8443, funnel's=443.

_LOCAL_PORT = 8787
_TAILNET_TARGET = 8788
_FUNNEL_TARGET = 8789
_TAILNET_PUBLIC = 8443
_FUNNEL_PUBLIC = 443


def _load_fixture(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def test_preflight_refuses_a_stale_mapping_pointed_at_another_port():
    """This is the actual KI-17 scenario (fix round 2, Important 1): a
    mapping -- here, under tailnet's own public port -- forwards straight
    into the port `local` holds, regardless of which public port carries
    it. This is unconditional and checked across every `Web` entry, not
    only the caller's own. Pinned from *both* adapters' perspective against
    the same fixture (fix round 3, fold-in): check 1 does not key on the
    caller's own public port at all, so it must catch the danger whether
    `tailnet` or `funnel` is asking -- that property held in code before
    this round but was never asserted."""
    payload = _load_fixture("serve_status_local_port_danger.json")
    refusal = parse_serve_status(
        payload, verb="serve", public_port=_TAILNET_PUBLIC,
        target_port=_TAILNET_TARGET, local_port=_LOCAL_PORT,
        forbid_funnel=True,
    )
    assert refusal is not None
    assert str(_LOCAL_PORT) in refusal
    # Names the offending mapping's public port (fix round 3 fold-in) --
    # the fixture's danger entry is keyed under tailnet's own 8443.
    assert "8443" in refusal
    # Names the misconfiguration, never a hostname, token or path.
    assert "example-host" not in refusal
    assert "ts.net" not in refusal
    # Never recommends the config-wide reset (fix round 2 minor).
    assert "reset" not in refusal

    refusal_from_funnel = parse_serve_status(
        payload, verb="funnel", public_port=_FUNNEL_PUBLIC,
        target_port=_FUNNEL_TARGET, local_port=_LOCAL_PORT,
    )
    assert refusal_from_funnel is not None
    assert str(_LOCAL_PORT) in refusal_from_funnel


def test_preflight_still_catches_the_danger_entry_despite_a_malformed_sibling():
    """Fix round 3, Must fix 1: the original single `try` wrapped the whole
    sweep, so an `AttributeError`/`TypeError`/`ValueError` raised while
    inspecting one `Web` entry returned `None` -- "clear" -- without ever
    inspecting the next entry, which could be the one actually proxying
    into `local`'s port. Adding `ValueError` for Important 2 widened this
    rather than narrowing it. Neither entry here is under this test's own
    `public_port`, isolating check 1's cross-entry tolerance from check 2/3
    entirely."""
    payload = {
        "Web": {
            "bad-entry.example-tailnet.ts.net:9999": {
                "Handlers": {"/": {"Proxy": "http://127.0.0.1:99999"}}
            },
            "danger-entry.example-tailnet.ts.net:8443": {
                "Handlers": {"/": {"Proxy": "http://127.0.0.1:8787"}}
            },
        }
    }
    refusal = parse_serve_status(
        payload, verb="serve", public_port=_FUNNEL_PUBLIC,
        target_port=_FUNNEL_TARGET, local_port=_LOCAL_PORT,
    )
    assert refusal is not None
    assert str(_LOCAL_PORT) in refusal
    assert "8443" in refusal
    assert "example-tailnet" not in refusal


@pytest.mark.parametrize(
    "bad_proxy", [8787, 1.5, True, ["http://127.0.0.1:8787"], {"port": 8787}],
)
def test_preflight_still_catches_the_danger_entry_despite_a_non_string_proxy(
    bad_proxy, caplog,
):
    """Fix round 4, Must fix -- a regression relative to fix round 2.

    Round 3 replaced a single `try` around the whole sweep with per-entry
    handling, which was right, but narrowed the caught exception from
    `(AttributeError, TypeError, ValueError)` to `ValueError` alone. A
    status document is JSON, so `Proxy` can hold any scalar -- and
    `urlparse` raises **`AttributeError`**, not `ValueError`, on every
    non-string it is handed (`'int' object has no attribute 'decode'`,
    from `urlparse`'s own `_decode_args`). The narrowed `except` therefore
    let the exception propagate out of `parse_serve_status` entirely: not
    merely a false "clear", but a hard failure in the exact direction spec
    §2.3 forbids, and with the second entry -- a mapping proxying straight
    into `local`'s port, the KI-17 danger itself -- never inspected.

    No test covered a non-string `Proxy`, which is how the narrowing
    survived a round. This one covers every JSON scalar type that reaches
    it: `int`, `float`, `bool`, `list`, `dict`."""
    payload = {
        "Web": {
            "bad-entry.example-tailnet.ts.net:9999": {
                "Handlers": {"/": {"Proxy": bad_proxy}}
            },
            "danger-entry.example-tailnet.ts.net:8443": {
                "Handlers": {"/": {"Proxy": "http://127.0.0.1:8787"}}
            },
        }
    }
    with caplog.at_level(logging.WARNING):
        refusal = parse_serve_status(
            payload, verb="serve", public_port=_FUNNEL_PUBLIC,
            target_port=_FUNNEL_TARGET, local_port=_LOCAL_PORT,
        )
    assert refusal is not None
    assert str(_LOCAL_PORT) in refusal
    assert "8443" in refusal
    assert "example-tailnet" not in refusal
    # The unrecognised entry is not silently swallowed either -- it is
    # visible as a warning, so an operator can see which entry preflight
    # could not read.
    assert any("9999" in record.message for record in caplog.records)


def test_a_refusal_never_leaks_a_hostname_from_a_key_of_unexpected_shape():
    """Fix round 4 fold-in: `_public_port_from_key` returned everything
    after the last colon without checking it was numeric, so a `Web` key
    of an unexpected shape put the *hostname* into the refusal -- against
    the helper's own docstring and spec §2.3 L2, which requires a refusal
    to name the misconfiguration and never a hostname, token or path.
    Before the `.isdigit()` guard, the key below produced "... already
    forwards a public mapping on port host.example-tailnet.ts.net straight
    to port 8787 ..."."""
    payload = {
        "Web": {
            "weird:host.example-tailnet.ts.net": {
                "Handlers": {"/": {"Proxy": "http://127.0.0.1:8787"}}
            }
        }
    }
    refusal = parse_serve_status(
        payload, verb="serve", public_port=_TAILNET_PUBLIC,
        target_port=_TAILNET_TARGET, local_port=_LOCAL_PORT,
    )
    # The danger is still refused -- the key's shape does not excuse it.
    assert refusal is not None
    assert str(_LOCAL_PORT) in refusal
    # But nothing of the hostname reaches the sentence.
    assert "host.example-tailnet" not in refusal
    assert "ts.net" not in refusal
    assert "weird" not in refusal


def test_preflight_refuses_our_own_mapping_pointed_elsewhere():
    """A mapping under *this adapter's own* public port, proxying to some
    local port that is neither `local`'s nor this adapter's own target, is
    still a stale/wrong mapping and still refused -- but the refusal must
    name *this* adapter's own verb, never hardcode 'serve' when `funnel`
    asks (fix round 2 minor)."""
    payload = {
        "Web": {
            "example-host.example-tailnet.ts.net:443": {
                "Handlers": {"/": {"Proxy": "http://127.0.0.1:9999"}}
            }
        }
    }
    refusal = parse_serve_status(
        payload, verb="funnel", public_port=_FUNNEL_PUBLIC,
        target_port=_FUNNEL_TARGET, local_port=_LOCAL_PORT,
    )
    assert refusal is not None
    assert "9999" in refusal
    assert "funnel" in refusal
    assert "serve" not in refusal
    # Warns what `reset` would destroy rather than recommending it as *the*
    # fix -- the targeted `off` form is the one actually recommended.
    assert "off" in refusal
    assert "would also remove" in refusal


def test_preflight_accepts_a_mapping_that_already_points_at_our_own_port():
    payload = _load_fixture("serve_status_own_port.json")
    assert parse_serve_status(
        payload, verb="serve", public_port=_TAILNET_PUBLIC,
        target_port=_TAILNET_TARGET, local_port=_LOCAL_PORT,
        forbid_funnel=True,
    ) is None


def test_preflight_accepts_the_real_empty_status_this_machine_returns():
    """`{}` is the only payload shape actually observed by running
    `tailscale serve status --json` on this machine (fix round 2 minor) --
    nothing configured, which must read as clear, not as unparseable."""
    assert parse_serve_status(
        {}, verb="serve", public_port=_TAILNET_PUBLIC,
        target_port=_TAILNET_TARGET, local_port=_LOCAL_PORT,
        forbid_funnel=True,
    ) is None


def test_preflight_never_refuses_one_transport_for_the_others_own_mapping():
    """Fix round 2, Important 1 -- the core regression: `tailnet` and
    `funnel`'s mappings live in the same `Web` document (a funnel mapping is
    a serve mapping with `AllowFunnel` set), so the original
    `parse_serve_status` treated the sibling transport's own legitimate
    mapping as offending. Both must read as clear when the other is already
    configured correctly."""
    payload = _load_fixture("serve_status_both_transports_configured.json")
    assert parse_serve_status(
        payload, verb="serve", public_port=_TAILNET_PUBLIC,
        target_port=_TAILNET_TARGET, local_port=_LOCAL_PORT,
        forbid_funnel=True,
    ) is None
    assert parse_serve_status(
        payload, verb="funnel", public_port=_FUNNEL_PUBLIC,
        target_port=_FUNNEL_TARGET, local_port=_LOCAL_PORT,
    ) is None


def test_preflight_refuses_tailnets_own_mapping_marked_allowfunnel():
    """Fix round 3, fold-in: `tailnet`'s public port (8443) is one of
    Funnel's three permitted ports, so a pre-existing or leftover
    `AllowFunnel` entry there would publish `tailnet` -- the one raisable
    listener, reachable up to `EXECUTE` -- to the open internet while TENKA
    believes it is tailnet-only. Only `forbid_funnel=True` (which
    `TailnetAdapter` alone sets) triggers this; `FunnelAdapter` is expected
    to carry `AllowFunnel` on its own mapping and must not be refused for
    it (covered by the mutual-noninterference test above, which already
    calls the funnel side with `forbid_funnel` left at its `False` default
    against a fixture where funnel's own entry is `AllowFunnel: true`)."""
    payload = _load_fixture("serve_status_tailnet_allowfunnel.json")
    refusal = parse_serve_status(
        payload, verb="serve", public_port=_TAILNET_PUBLIC,
        target_port=_TAILNET_TARGET, local_port=_LOCAL_PORT,
        forbid_funnel=True,
    )
    assert refusal is not None
    assert "AllowFunnel" in refusal
    assert "8443" in refusal
    assert "example-host" not in refusal
    assert "ts.net" not in refusal

    # The identical mapping is accepted when forbid_funnel is left False --
    # proving the refusal is `tailnet`-specific, not a blanket rule.
    assert parse_serve_status(
        payload, verb="serve", public_port=_TAILNET_PUBLIC,
        target_port=_TAILNET_TARGET, local_port=_LOCAL_PORT,
    ) is None


def test_preflight_refuses_a_leftover_allowfunnel_with_no_web_entry():
    """Fix round 4 fold-in -- the other half of round 3's own threat.

    Round 3's `AllowFunnel` check sat inside the `Web` loop, so it could
    only fire when a `Web` entry still existed under `tailnet`'s public
    port. But `AllowFunnel` is a top-level document keyed the same way, and
    the *leftover* case the check was written for is precisely a surviving
    flag whose mapping has already been removed -- TENKA's own subsequent
    `tailscale serve --https 8443 ...` would inherit it and publish the one
    raisable listener to the open internet. Nothing in the document here is
    keyed under 8443 except the flag itself."""
    payload = _load_fixture("serve_status_leftover_allowfunnel_no_web_entry.json")
    assert "example-host.example-tailnet.ts.net:8443" not in payload["Web"]

    refusal = parse_serve_status(
        payload, verb="serve", public_port=_TAILNET_PUBLIC,
        target_port=_TAILNET_TARGET, local_port=_LOCAL_PORT,
        forbid_funnel=True,
    )
    assert refusal is not None
    assert "AllowFunnel" in refusal
    assert "8443" in refusal
    assert "example-host" not in refusal
    assert "ts.net" not in refusal

    # `funnel`'s own surviving entry, correctly targeted and legitimately
    # marked AllowFunnel, still reads as clear from funnel's side -- the
    # new independent check must not have become a blanket rule.
    assert parse_serve_status(
        payload, verb="funnel", public_port=_FUNNEL_PUBLIC,
        target_port=_FUNNEL_TARGET, local_port=_LOCAL_PORT,
    ) is None


def test_preflight_degrades_to_a_warning_on_output_it_cannot_parse(caplog):
    """Layer 3 contains the failure either way -- a preflight that hard-fails
    on an unrecognised shape would take the whole transport down for a
    formatting change."""
    unrecognised = ["this", "is", "a", "list", "not", "the", "documented", "shape"]
    with caplog.at_level(logging.WARNING):
        result = parse_serve_status(
            unrecognised, verb="serve", public_port=_TAILNET_PUBLIC,
            target_port=_TAILNET_TARGET, local_port=_LOCAL_PORT,
        )
    assert result is None
    assert any("tailscale" in record.message.lower() for record in caplog.records)


def test_preflight_degrades_to_a_warning_on_a_malformed_proxy_port(caplog):
    """Fix round 2, Important 2: `urlparse(...).port` raises `ValueError`
    on an out-of-range or non-numeric port, and the original `except` tuple
    (`AttributeError`, `TypeError`) let it escape `parse_serve_status`
    uncaught -- a hard failure in exactly the direction the spec forbids."""
    payload = {
        "Web": {
            "example-host.example-tailnet.ts.net:8443": {
                "Handlers": {"/": {"Proxy": "http://127.0.0.1:99999"}}
            }
        }
    }
    with caplog.at_level(logging.WARNING):
        result = parse_serve_status(
            payload, verb="serve", public_port=_TAILNET_PUBLIC,
            target_port=_TAILNET_TARGET, local_port=_LOCAL_PORT,
        )
    assert result is None
    assert any("tailscale" in record.message.lower() for record in caplog.records)


# ─── Task 7: hostname recognition ────────────────────────────────────────────

def test_the_hostname_is_recognised_from_real_tailscale_output():
    """Real `tailscale serve --bg` / `tailscale funnel --bg` stdout shape:
    'Available within your tailnet: https://<host>/' and
    'Available on the internet: https://<host>/'."""
    tailnet_line = "Available within your tailnet: https://laptop.tail1234.ts.net/"
    funnel_line = "Available on the internet: https://laptop.tail1234.ts.net/"
    assert TailnetAdapter().hostname_from(tailnet_line) == "laptop.tail1234.ts.net"
    assert FunnelAdapter().hostname_from(funnel_line) == "laptop.tail1234.ts.net"


def test_a_hostname_outside_the_providers_own_domain_is_rejected():
    """A name announced by the subprocess becomes a trusted Host and Origin
    (spec §8), so a shape outside the provider's own `*.ts.net` domain must
    be rejected rather than accepted as anything that merely looks like a
    hostname."""
    adapter = TailnetAdapter()
    assert adapter.hostname_from("Available within your tailnet: https://evil.example.com/") is None
    # Suffix confusion: contains "ts.net" but does not end with it.
    assert adapter.hostname_from("https://laptop.ts.net.evil.com/") is None
    # Userinfo confusion: `urlparse(...).hostname` resolves to "evil.com",
    # not "laptop.tail1234.ts.net", so this must fail too.
    assert adapter.hostname_from("https://laptop.tail1234.ts.net@evil.com/") is None
    # No URL on the line at all.
    assert adapter.hostname_from("Success.") is None
