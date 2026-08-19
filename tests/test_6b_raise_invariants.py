"""Milestone 6b, spec §3.5 — the six invariants that replace 6a.5's
'a ceiling can only ever narrow'.

I1, I2 and I3 are exhaustive rather than sampled: 128 grant sets x 128 raise
sets x 4 policies. A property this load-bearing is cheap enough to prove
totally, and a sampled version would have let 6a.5's two derived ceilings
through.
"""
from itertools import combinations

import pytest

from assistant.io.api.policy import POLICIES, effective
from assistant.io.api.vault import Capability

ALL = list(Capability)


def _powerset():
    for size in range(len(ALL) + 1):
        for combo in combinations(ALL, size):
            yield frozenset(combo)


POWERSET = list(_powerset())
POLICY_LIST = list(POLICIES.values())


def test_i1_the_outer_bound_never_moves():
    """No raise state can ever hand a device a capability it was not issued."""
    for policy in POLICY_LIST:
        for grants in POWERSET:
            for raised in POWERSET:
                assert effective(grants, policy, raised) <= grants


def test_i2_a_transport_never_carries_what_it_was_not_vetted_for():
    for policy in POLICY_LIST:
        bound = policy.ceiling | policy.raisable
        for grants in POWERSET:
            for raised in POWERSET:
                assert effective(grants, policy, raised) <= bound


def test_i3_no_raise_means_no_change_from_6a5():
    for policy in POLICY_LIST:
        for grants in POWERSET:
            assert effective(grants, policy) == grants & policy.ceiling
            assert effective(grants, policy, frozenset()) == grants & policy.ceiling


def test_i4_raisable_is_an_explicit_literal_not_an_enum_derivation():
    """`frozenset(Capability)` and 'everything minus X' are both banned.

    6a.5 deleted two sites with the derived spelling; both silently widened
    when EXECUTE joined the enum. A `raisable` written that way would widen
    the same way, on the one field whose whole job is to be narrow.
    """
    everything = frozenset(Capability)
    for policy in POLICY_LIST:
        assert policy.raisable != everything, policy.name
        assert policy.raisable < everything, policy.name


def test_i4b_ceiling_and_raisable_are_ast_pinned_to_a_pure_frozenset_literal():
    """Ruling 11 / fix round 5, Minor 6, confirmed by reviewer B: the runtime
    check above catches the exact full-enum spelling (`frozenset(Capability)`
    is neither `!=` nor `<` itself) but not a *subtractive* one --
    `frozenset(Capability) - {SCREEN}` is a proper subset of `everything`, so
    it passes both assertions above while still being derived from the enum,
    exactly the shape the standing rule exists to forbid: it would silently
    widen the moment a new capability joins, the same way the two sites
    Milestone 6a.5 deleted did.

    By the time such an expression has been evaluated into a frozenset there
    is nothing left in the *value* to tell it apart from one written out by
    hand, so this walks `policy.py`'s own source instead: every `ceiling=`
    and `raisable=` keyword on a `ListenerPolicy(...)` call must be exactly
    `frozenset({...})` -- a call to `frozenset` with at most one set/dict
    literal argument, no `Capability` name, no `|`/`-`/other `BinOp` at any
    depth inside it.
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
            if kw.arg not in ("ceiling", "raisable"):
                continue
            checked += 1
            value = kw.value
            assert (isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "frozenset"), (
                f"{kw.arg} must be a bare frozenset({{...}}) literal, "
                f"got {ast.dump(value)}")
            assert len(value.args) <= 1, (kw.arg, ast.dump(value))
            if value.args:
                assert isinstance(value.args[0], (ast.Set, ast.Dict)), (
                    f"{kw.arg}'s frozenset(...) argument must be a set "
                    f"literal, not {ast.dump(value.args[0])} -- "
                    f"frozenset(Capability) is exactly the derivation this "
                    f"pins against")
            for sub in ast.walk(value):
                assert not isinstance(sub, ast.BinOp), (
                    f"{kw.arg} is derived via a BinOp ({ast.dump(sub)}), "
                    f"e.g. a union or subtraction, rather than written out "
                    f"as an explicit literal")
    assert checked == 2 * len(POLICY_LIST), (
        f"expected a ceiling= and a raisable= keyword on each of "
        f"{len(POLICY_LIST)} policies ({2 * len(POLICY_LIST)} total); found "
        f"{checked} -- a policy or keyword was not parsed as expected")


def test_i5_a_raise_touches_capabilities_only():
    """admin, allow_bearer and secure_cookie are not raisable by anything."""
    import inspect
    source = inspect.getsource(effective)
    for flag in ("admin", "allow_bearer", "secure_cookie"):
        assert flag not in source


def test_i6_only_tailnet_is_raisable():
    assert POLICIES["tailnet"].raisable == frozenset(
        {Capability.EXECUTE, Capability.SYSTEM_CONTROL})
    for name in ("local", "funnel", "quick"):
        assert POLICIES[name].raisable == frozenset(), name


@pytest.mark.parametrize("name", ["local", "funnel", "quick"])
def test_an_unraisable_transport_ignores_every_raise(name):
    policy = POLICIES[name]
    grants = frozenset(Capability)
    assert effective(grants, policy, frozenset(Capability)) == grants & policy.ceiling


def test_tailnet_raised_to_execute_carries_execute_only_when_the_device_holds_it():
    tailnet = POLICIES["tailnet"]
    raised = frozenset({Capability.EXECUTE})
    holder = frozenset({Capability.CHAT_SEND, Capability.EXECUTE})
    assert Capability.EXECUTE in effective(holder, tailnet, raised)
    non_holder = frozenset({Capability.CHAT_SEND})
    assert Capability.EXECUTE not in effective(non_holder, tailnet, raised)
