# tests/test_6b_transport_cloudflare.py
"""Milestone 6b Task 8 -- the Cloudflare `quick` transport adapter.

Its own file rather than a section of `tests/test_6b_transport_adapters.py`:
Task 7 owns that file and it now carries 31 tests, so appending here keeps
the two providers' suites independently runnable (the standing constraint in
this repo is that tests are run per file, never as a whole suite).

`cloudflared` is **not installed on the machine these tests were written on**,
so nothing here shells out. Every assertion is against argv this module builds
itself, or against output lines quoted from Cloudflare's own documentation --
see `assistant/io/api/transports/cloudflare.py`'s module docstring for the
citations. The live-test task installs the binary and confirms the shape.
"""
from __future__ import annotations

import pytest

from assistant.io.api.transports import transport_registry
from assistant.io.api.transports.base import TransportAdapter
from assistant.io.api.transports.cloudflare import QuickAdapter

# `quick`'s own local target port -- `listeners.py`'s offset 3 from a base
# port of 8787, matching the numbers Task 7's file uses for the other two.
_LOCAL_PORT = 8787
_QUICK_TARGET = 8790


# ─── Task 8: command construction ────────────────────────────────────────────

def test_the_quick_command_targets_the_port_it_is_given():
    argv = QuickAdapter().command(_QUICK_TARGET)
    assert argv[0] == "cloudflared"
    assert argv[1] == "tunnel"
    assert "--url" in argv
    assert f"http://127.0.0.1:{_QUICK_TARGET}" in argv
    # The port comes from the caller's integer, not from a baked-in constant.
    assert "http://127.0.0.1:9999" in QuickAdapter().command(9999)
    # `cloudflared tunnel --url <target>` and nothing else (spec §4): four
    # tokens, so a flag added later has to be argued for here first.
    assert argv == [
        "cloudflared", "tunnel", "--url", f"http://127.0.0.1:{_QUICK_TARGET}",
    ]


def test_the_quick_command_never_sets_http_host_header():
    """Spec §2.3's stated gap, turned into a guard on our own argv.

    KI-17's layer 3 -- the load-bearing one -- is that the `local` listener
    accepts loopback `Host` names only, so a tunnelled request arriving on
    the local port carries the public authority and is answered 421 before
    authentication. `cloudflared --http-host-header 127.0.0.1:8787` rewrites
    the `Host` cloudflared forwards to a loopback name and defeats exactly
    that. TENKA must never emit the flag, and this test is what stops a
    future "fix" for a `Host`-related bug from reaching for it.

    Checked on every argv this adapter can produce, and by substring as well
    as by token -- `--http-host-header=127.0.0.1:8787` is one argv element,
    not two, so an exact-match check alone would miss it.

    `_LOCAL_PORT` is in the loop only because it is the port the flag would
    be pointed at, so it is the one an accidental reintroduction would most
    likely use; it is **not** an assertion that the adapter should accept it.
    Refusing a target port that is `local`'s is spec §2.3 L1's spawn-time
    check, which lives in the manager -- an adapter is handed a port and
    knows nothing of the base port it was derived from.
    """
    adapter = QuickAdapter()
    for port in (_LOCAL_PORT, _QUICK_TARGET, 9999):
        argv = adapter.command(port)
        assert "--http-host-header" not in argv
        assert not any("http-host-header" in token for token in argv)
        # The stop path builds no argv at all, but assert it stays that way
        # rather than assume it: a future `stop_command` returning an argv
        # must be held to the same rule.
        stop = adapter.stop_command(port)
        assert stop is None or not any(
            "http-host-header" in token for token in stop
        )


def test_a_non_numeric_string_is_rejected_before_it_can_reach_the_command_line():
    """Spec §8's subprocess-injection row: no caller-supplied string is
    formatted into an argv. A numeric string is normalised through `int()`;
    a non-numeric one raises rather than reaching the command line."""
    with pytest.raises((TypeError, ValueError)):
        QuickAdapter().command("8790; rm -rf /")  # type: ignore[arg-type]


# ─── Task 8: hostname recognition ────────────────────────────────────────────

def test_the_trycloudflare_hostname_is_recognised_from_real_output():
    """Both documented banner wordings, quoted from Cloudflare's own docs
    (see the adapter's module docstring). The adapter must key on the URL
    alone, never on the surrounding prose, since the prose has already
    changed once between cloudflared releases."""
    adapter = QuickAdapter()
    old_banner = (
        "2021-07-15T20:11:32Z INF |    "
        "https://seasonal-deck-organisms-sf.trycloudflare.com     |"
    )
    new_banner = (
        "2026-08-16T09:00:00Z INF |  "
        "https://corporate-mention-tiles-coordinates.trycloudflare.com    |"
    )
    assert (
        adapter.hostname_from(old_banner)
        == "seasonal-deck-organisms-sf.trycloudflare.com"
    )
    assert (
        adapter.hostname_from(new_banner)
        == "corporate-mention-tiles-coordinates.trycloudflare.com"
    )
    # Ordinary log lines carry no hostname.
    assert adapter.hostname_from(
        "2026-08-16T09:00:00Z INF Registered tunnel connection connIndex=0"
    ) is None


def test_a_hostname_outside_trycloudflare_is_rejected():
    """A name announced by the subprocess becomes a trusted `Host` and
    `Origin` (spec §8), so a shape outside Cloudflare's own quick-tunnel
    domain must be rejected rather than accepted as anything that merely
    looks like a hostname. Same bypass family Task 7's adapter was hardened
    against, re-run against this one."""
    adapter = QuickAdapter()
    # Wrong domain entirely.
    assert adapter.hostname_from("INF |  https://evil.example.com  |") is None
    # Suffix confusion: contains the domain but does not end with it.
    assert adapter.hostname_from("https://a.trycloudflare.com.evil.com/") is None
    # Userinfo confusion: `urlparse(...).hostname` is "evil.com" here.
    assert adapter.hostname_from("https://a.trycloudflare.com@evil.com/") is None
    # Prefix confusion: a registrable domain that merely ends in the string.
    assert adapter.hostname_from("https://a.nottrycloudflare.com/") is None
    # The bare apex is not a quick tunnel -- a quick tunnel is always a
    # subdomain, and the apex is Cloudflare's own site.
    assert adapter.hostname_from("https://trycloudflare.com/") is None
    # A hostile line appending a legitimate name after a bad one: only the
    # first URL on the line is considered, so this fails on `evil.com`.
    assert adapter.hostname_from(
        "https://evil.com/ https://a.trycloudflare.com/"
    ) is None
    # Not https.
    assert adapter.hostname_from("http://a.trycloudflare.com/") is None
    # No URL on the line at all.
    assert adapter.hostname_from("Registered tunnel connection") is None


# ─── Task 8: preflight and stop ──────────────────────────────────────────────

def test_quick_has_no_preflight_and_says_so():
    """An unnamed tunnel has no persisted configuration to reconcile with --
    Cloudflare mints the hostname per run and forgets it -- so there is no
    §2.3 L2 check to make. `preflight` returns `None` for every port, and
    the docstring says why rather than leaving a reader to wonder whether
    the check was forgotten."""
    adapter = QuickAdapter()
    assert adapter.preflight(_QUICK_TARGET) is None
    assert adapter.preflight(_LOCAL_PORT) is None
    # `base.py`'s Protocol asks for the reason in the docstring, so the
    # docstring is pinned rather than left to a convention. One precise
    # substring, not a bag of common words: `"no" in doc` would be true of
    # any English text containing "not".
    doc = (QuickAdapter.preflight.__doc__ or "").lower()
    assert "persisted configuration" in doc, (
        "preflight must say it has nothing to reconcile with, so a reader "
        "can tell the check was considered rather than forgotten"
    )


def test_quick_stops_by_terminating_its_own_process():
    """`cloudflared tunnel --url` runs in the foreground: the spawned
    process's lifetime *is* the tunnel's, so terminating it is the stop.
    That is the opposite of the Tailscale adapters, which daemonise via
    `--bg` and need an explicit `off` argv -- see `base.py`'s
    `stop_command` docstring for the distinction and for the verification
    obligation a non-`None` argv would carry."""
    adapter = QuickAdapter()
    assert adapter.stop_command(_QUICK_TARGET) is None
    assert adapter.stop_command(_LOCAL_PORT) is None


# ─── Task 8: registration ────────────────────────────────────────────────────

def test_quick_registers_itself_under_its_own_policy_name():
    adapter = transport_registry.get("quick")
    assert isinstance(adapter, QuickAdapter)
    assert adapter.name == "quick"
    assert isinstance(adapter, TransportAdapter)
