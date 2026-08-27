"""One merge rule, reachable from both layers.

P5 asked for `Executor.run(step) -> Verdict` in `brain/` and named
`actions/planner/executor.py` as a file that would use it. Both cannot be true:
`brain` sits above `actions`, the tree holds zero `actions -> brain` imports,
and adding one would be the fifth layer inversion -- the shape P4b had just
finished removing from `automation/event_bus.py`.

So the pure part lives in `core/`, below both, and this file pins that it is
genuinely shared rather than shared in a comment. The rule it implements is the
one `CLAUDE.md` calls out by name: a user who pins "mobile as 99999" gets
99999, not a value some adapter found more plausible.

Run with:  py -3.11 -m pytest tests/test_step_params.py -v
"""
import ast
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.core.step_params import build_step_params  # noqa: E402


# ─── structure decides ───────────────────────────────────────────────────────

def test_structured_parameters_pass_through():
    assert build_step_params({"target": "notepad", "mode": "max"}, goal_key="goal") == {
        "target": "notepad", "mode": "max"}


def test_the_goal_lands_under_the_key_the_adapter_expects():
    """The planner's manifest already chooses this -- `url` for `open_browser`,
    `query` for `web_search` -- and that choice is the adapter's contract. A
    merge that hardcoded "goal" would have broken every tool whose row names
    something else, which is the concrete way this migration could have gone
    wrong quietly."""
    assert build_step_params(goal="bbc.co.uk", goal_key="url") == {
        "url": "bbc.co.uk"}


def test_a_structured_field_is_not_overwritten_by_the_sentence():
    """"The string is data, not the meaning" has to mean something in code, and
    this is it: a step that stated the field said something more precise than
    the sentence did."""
    params = build_step_params({"url": "https://example.com"},
                               goal="go to example dot com", goal_key="url")
    assert params["url"] == "https://example.com"


def test_an_empty_goal_adds_no_key():
    """A single-step goal should not carry an empty labelled field into a
    prompt or an adapter."""
    assert build_step_params({"a": 1}, goal="", goal_key="goal") == {"a": 1}


# ─── constraints are hard ────────────────────────────────────────────────────

def test_a_pinned_value_wins_over_a_parameter():
    params = build_step_params({"mobile": "0000000000"},
                               {"mobile": "99999"}, goal_key="goal")
    assert params["mobile"] == "99999"


def test_a_pinned_value_is_byte_identical():
    """Not normalised, not coerced, not trimmed. The pin is what the user
    said."""
    pinned = "  99999  "
    assert build_step_params({}, {"mobile": pinned}, goal_key="goal")["mobile"] == pinned


def test_a_task_constraint_applies_to_a_step_that_never_mentions_it():
    params = build_step_params({"mobile": "0"}, None,
                               extra_constraints={"mobile": "99999"},
                               goal_key="goal")
    assert params["mobile"] == "99999"


def test_the_step_s_own_pin_is_overridden_by_the_task_s():
    """Stated so the ordering is a decision rather than an accident: the
    enclosing Task's constraints are applied last, so a value pinned for the
    whole task holds across every step in it."""
    params = build_step_params({}, {"seat": "14A"},
                               extra_constraints={"seat": "1A"},
                               goal_key="goal")
    assert params["seat"] == "1A"


def test_unconstrained_parameters_survive():
    """A merge that replaced the dict instead of updating it would satisfy
    every constraint test above and lose everything else."""
    params = build_step_params({"seat": "any", "date": "tomorrow"},
                               {"seat": "14A"}, goal_key="goal")
    assert params["date"] == "tomorrow"


@pytest.mark.parametrize("mutate", [
    lambda d: d.__setitem__("mobile", "0"),
    lambda d: d.pop("mobile"),
])
def test_the_caller_s_dicts_are_never_mutated(mutate):
    """**The one that bites across steps.** An adapter that normalises its
    params in place would rewrite the pin for every later step in the plan if
    this returned a reference."""
    parameters = {"other": 1}
    constraints = {"mobile": "99999"}

    params = build_step_params(parameters, constraints, goal_key="goal")
    mutate(params)

    assert constraints == {"mobile": "99999"}, "the pin itself was mutated"
    assert parameters == {"other": 1}, "the step's parameters were mutated"


def test_none_arguments_are_accepted():
    """Every caller has at least one of these empty today."""
    assert build_step_params(None, None, "", goal_key="goal") == {}


# ─── genuinely shared, not shared in a comment ───────────────────────────────

def _calls_build_step_params(rel: str) -> bool:
    tree = ast.parse((_ROOT / rel).read_text(encoding="utf-8"))
    return any(
        isinstance(n, ast.Call)
        and (getattr(n.func, "id", None) == "build_step_params"
             or getattr(n.func, "attr", None) == "build_step_params")
        for n in ast.walk(tree)
    )


@pytest.mark.parametrize("rel", [
    "assistant/brain/executor.py",
    "assistant/actions/planner/executor.py",
])
def test_both_layers_call_the_shared_merge(rel):
    assert _calls_build_step_params(rel), (
        f"{rel} builds its parameters itself. Two merge rules is how the "
        "pinned-value gotcha gets broken in one of them without the other "
        "noticing.")


def test_the_planner_no_longer_builds_the_dict_inline():
    """The exact line this replaced. `params = {param_key: resolved_goal}` is
    where the sentence *was* the meaning."""
    src = (_ROOT / "assistant" / "actions" / "planner"
           / "executor.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        if getattr(node.targets[0], "id", None) != "params":
            continue
        rendered = ast.unparse(node.value)
        assert "resolved_goal" not in rendered or "build_step_params" in rendered, (
            f"the planner assembles params from the sentence again: {rendered}")


def test_neither_layer_imports_the_other():
    """The inversion this arrangement exists to avoid. Asserted rather than
    trusted to the contracts file, which forbids `automation -> brain` and
    `io -> brain` but says nothing about `actions -> brain`."""
    actions = (_ROOT / "assistant" / "actions").rglob("*.py")
    offenders = []
    for path in actions:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if "brain" in node.module.split("."):
                    offenders.append(
                        f"{path.relative_to(_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "brain" in alias.name.split("."):
                        offenders.append(
                            f"{path.relative_to(_ROOT)}:{node.lineno}")
    assert not offenders, (
        f"actions/ reaches up into brain/: {offenders}. The Brain is above "
        "the handler package; share downward through core/ instead.")


def test_core_reaches_nothing_above_it():
    """Anti-vacuity for the arrangement: `core/step_params.py` is only a valid
    place to share from if it imports nothing from either layer."""
    tree = ast.parse(
        (_ROOT / "assistant" / "core" / "step_params.py").read_text(
            encoding="utf-8"))
    modules = {n.module for n in ast.walk(tree)
               if isinstance(n, ast.ImportFrom) and n.module}
    modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                for a in n.names}
    assert modules <= {"__future__", "typing"}, (
        f"the shared merge reaches for more than the stdlib: {modules}")


def test_the_goal_key_must_be_stated():
    """**From a green mutant.** `goal_key` had a `"goal"` default, and changing
    it to nonsense turned nothing red -- both callers state it, so the default
    was reachable only by a caller who had not thought about which parameter
    carries the instruction. That caller is the whole reason the rule exists,
    so the default is gone and forgetting it is now a TypeError rather than a
    silent fallback to the sentence-as-meaning behaviour being removed."""
    with pytest.raises(TypeError):
        build_step_params({"a": 1}, goal="x")
