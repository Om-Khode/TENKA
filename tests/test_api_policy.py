"""Listener policy: what a transport permits, independent of device grants."""
from assistant.io.api.policy import POLICIES, effective, policy_for_port
from assistant.io.api.vault import Capability


def test_the_four_policies_exist_and_only_local_is_admin():
    assert set(POLICIES) == {"local", "tailnet", "funnel", "quick"}
    assert POLICIES["local"].admin is True
    for name in ("tailnet", "funnel", "quick"):
        assert POLICIES[name].admin is False, name


def test_bearer_is_local_only():
    assert POLICIES["local"].allow_bearer is True
    for name in ("tailnet", "funnel", "quick"):
        assert POLICIES[name].allow_bearer is False, name


def test_secure_cookie_everywhere_but_local():
    assert POLICIES["local"].secure_cookie is False   # plain http on loopback
    for name in ("tailnet", "funnel", "quick"):
        assert POLICIES[name].secure_cookie is True, name


def test_quick_ceiling_is_read_only_and_excludes_screen():
    """CHAT alone. No CHAT_SEND (arbitrary code execution), no FILES, no
    SYSTEM_CONTROL -- and no SCREEN, because Cloudflare reads this transport's
    plaintext and screen capture is the largest disclosure in the API."""
    assert POLICIES["quick"].ceiling == frozenset({Capability.CHAT})


def test_full_ceilings_are_every_capability():
    for name in ("local", "tailnet", "funnel"):
        assert POLICIES[name].ceiling == frozenset(Capability), name


def test_effective_is_an_intersection_never_a_widening():
    device = frozenset({Capability.CHAT, Capability.CHAT_SEND, Capability.FILES})
    assert effective(device, POLICIES["funnel"]) == device
    assert effective(device, POLICIES["quick"]) == frozenset({Capability.CHAT})
    # a ceiling cannot grant what the device lacks
    assert Capability.SYSTEM_CONTROL not in effective(device, POLICIES["local"])


def test_a_registered_port_resolves_to_its_named_policy():
    registry = {8787: "local", 8788: "quick"}
    assert policy_for_port(8787, registry) is POLICIES["local"]
    assert policy_for_port(8788, registry) is POLICIES["quick"]


def test_an_unregistered_port_has_no_policy():
    """Fail closed: a socket nobody registered grants nothing at all."""
    assert policy_for_port(9999, {8787: "local"}) is None


def test_a_registry_naming_an_unknown_policy_fails_closed():
    assert policy_for_port(8787, {8787: "made_up"}) is None
