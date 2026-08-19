"""Listener policy: what a transport permits, independent of device grants."""
import pytest

from assistant.io.api.policy import POLICIES, ListenerPolicy, effective, policy_for_port
from assistant.io.api.vault import Capability


def test_the_three_policies_exist_and_only_local_is_admin():
    assert set(POLICIES) == {"local", "tailnet", "funnel"}
    assert POLICIES["local"].admin is True
    for name in ("tailnet", "funnel"):
        assert POLICIES[name].admin is False, name


def test_bearer_is_local_only():
    assert POLICIES["local"].allow_bearer is True
    for name in ("tailnet", "funnel"):
        assert POLICIES[name].allow_bearer is False, name


def test_secure_cookie_everywhere_but_local():
    assert POLICIES["local"].secure_cookie is False   # plain http on loopback
    for name in ("tailnet", "funnel"):
        assert POLICIES[name].secure_cookie is True, name


def test_observe_and_recall_are_distinct_capabilities():
    assert Capability.OBSERVE.value == "observe"
    assert Capability.RECALL.value == "recall"
    assert not hasattr(Capability, "CHAT")      # the ambiguous name is gone


def test_every_policy_is_pairable_today():
    """Milestone 6b's `quick` transport was the one policy `pairable=False`
    ever described -- Cloudflare terminated TLS and would have read a minted
    device credential in the clear. It was removed outright rather than kept
    around as a live example, so this is the property's whole remaining
    footprint in production data: every transport left standing may mint a
    credential. `test_a_policy_without_pairable_fails_loudly` below is what
    still proves the field cannot be silently defaulted to this same answer.
    """
    for name, policy in POLICIES.items():
        assert policy.pairable is True, name


def test_a_policy_without_pairable_fails_loudly():
    """`pairable` has no default, exactly like `ceiling` and `raisable` --
    the dataclass raises rather than silently admitting a policy nobody
    declared a pairing stance for. Constructing one omitting the field is
    what a future transport's author would do by mistake; this pins that the
    mistake cannot pass quietly."""
    with pytest.raises(TypeError):
        ListenerPolicy(  # type: ignore[call-arg]
            name="future", admin=False, allow_bearer=False, secure_cookie=True,
            ceiling=frozenset(), raisable=frozenset(),
        )


def test_pairable_is_ast_pinned_to_a_bare_boolean_literal():
    """The `ceiling`/`raisable` AST pin (`test_6b_raise_invariants.py`'s
    `test_i4b_...`) only walks those two keywords -- `pairable` needs its own,
    for the identical reason: a future policy could write `pairable=name !=
    "quick"` or some other derivation that happens to evaluate `True` today
    and silently changes the moment a new transport is added, rather than
    being a decision someone made by writing `True` or `False` in the open.
    """
    import ast
    import inspect

    from assistant.io.api import policy as policy_module

    tree = ast.parse(inspect.getsource(policy_module))
    checked = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "ListenerPolicy"):
            continue
        for kw in node.keywords:
            if kw.arg != "pairable":
                continue
            checked += 1
            assert isinstance(kw.value, ast.Constant) and isinstance(
                kw.value.value, bool), (
                f"pairable must be a bare True/False literal, got "
                f"{ast.dump(kw.value)}")
    assert checked == len(POLICIES), (
        f"expected a pairable= keyword on each of {len(POLICIES)} policies; "
        f"found {checked}"
    )


def test_the_local_ceiling_is_every_capability():
    """The operator at the keyboard keeps full power. Milestone 6a.5 narrowed
    the two tunnel ceilings; it did not downgrade the local path."""
    assert POLICIES["local"].ceiling == frozenset(Capability)


def test_no_tunnel_ceiling_carries_execute_or_system_control():
    """Was `ceiling == frozenset(Capability)` for all three until 6a.5. That
    spelling is what would have handed `EXECUTE` to the funnel listener the
    moment the capability joined the enum -- a leaked pair code becoming a
    remote shell, with nobody having decided to allow it."""
    for name in ("tailnet", "funnel"):
        assert Capability.EXECUTE not in POLICIES[name].ceiling, name
        assert Capability.SYSTEM_CONTROL not in POLICIES[name].ceiling, name
        assert Capability.CHAT_SEND in POLICIES[name].ceiling, name


def test_effective_is_an_intersection_never_a_widening():
    device = frozenset({Capability.OBSERVE, Capability.RECALL,
                        Capability.CHAT_SEND, Capability.FILES})
    assert effective(device, POLICIES["funnel"]) == device
    # a ceiling narrows: EXECUTE and SYSTEM_CONTROL are never carried by a
    # non-local transport, no matter what the device itself holds.
    wide_device = device | {Capability.EXECUTE, Capability.SYSTEM_CONTROL}
    assert effective(wide_device, POLICIES["funnel"]) == device
    # a ceiling cannot grant what the device lacks
    assert Capability.SYSTEM_CONTROL not in effective(device, POLICIES["local"])


def test_a_registered_port_resolves_to_its_named_policy():
    registry = {8787: "local", 8788: "funnel"}
    assert policy_for_port(8787, registry) is POLICIES["local"]
    assert policy_for_port(8788, registry) is POLICIES["funnel"]


def test_an_unregistered_port_has_no_policy():
    """Fail closed: a socket nobody registered grants nothing at all."""
    assert policy_for_port(9999, {8787: "local"}) is None


def test_a_registry_naming_an_unknown_policy_fails_closed():
    assert policy_for_port(8787, {8787: "made_up"}) is None
