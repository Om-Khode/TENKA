"""A command's grant comes from the command, not from the route."""
import dataclasses

import pytest
from assistant.io.api.runtime import CommandDef, CommandOutcome
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.api_client import build_api_client
from tests.fakes.studio_runtime import build_fake_runtime


@pytest.fixture()
def context(tmp_path):
    vault = TokenVault(tmp_path)
    runtime = build_fake_runtime()
    client = build_api_client(runtime, vault)
    tokens = {
        "full": vault.issue("studio", frozenset(Capability)),
        "chat": vault.issue("phone", frozenset({Capability.CHAT})),
        # Holds exactly one of the fixture's two distinct grants -- neither
        # "full" nor "chat" can exercise the route's per-command grant
        # lookup, because both clear or fail every entry in FakeCommandRuntime
        # identically regardless of which grant each entry actually declares.
        "screen_only": vault.issue("glasses", frozenset({Capability.SCREEN})),
    }
    return client, runtime, tokens


def head(token):
    return {"Authorization": f"Bearer {token}"}


def test_the_catalogue_is_readable_by_a_chat_device(context):
    client, _, tokens = context
    response = client.get("/v1/commands", headers=head(tokens["chat"]))
    assert response.status_code == 200
    assert [c["commandId"] for c in response.json()["data"]["commands"]] == [
        "lock_workstation", "volume_up", "screenshot"]


def test_the_catalogue_marks_destructive_entries(context):
    client, _, tokens = context
    commands = client.get("/v1/commands", headers=head(tokens["chat"])).json()["data"]["commands"]
    by_id = {c["commandId"]: c for c in commands}
    assert by_id["lock_workstation"]["destructive"] is True
    assert by_id["volume_up"]["destructive"] is False


def test_running_needs_the_command_s_own_grant(context):
    client, _, tokens = context
    assert client.post("/v1/commands/volume_up/run",
                       headers=head(tokens["chat"])).status_code == 403


def test_a_granted_device_can_run_it(context):
    client, runtime, tokens = context
    response = client.post("/v1/commands/volume_up/run", headers=head(tokens["full"]))
    assert response.status_code == 200
    assert runtime.commands.ran == ["volume_up"]


def test_an_unknown_command_is_404_not_a_failed_run(context):
    client, runtime, tokens = context
    assert client.post("/v1/commands/nope/run", headers=head(tokens["full"])).status_code == 404
    assert runtime.commands.ran == []


def test_a_refused_run_does_not_execute_anything(context):
    client, runtime, tokens = context
    client.post("/v1/commands/lock_workstation/run", headers=head(tokens["chat"]))
    assert runtime.commands.ran == []


def test_a_screen_grant_runs_screenshot_but_not_a_system_control_command(context):
    """The one test a differentiated fixture makes possible.

    A route that checked a grant hardcoded onto itself -- the exact
    anti-pattern this task exists to prevent -- would answer identically for
    "full" (holds every grant, including whichever one got hardcoded) and
    "chat" (holds none of the grants any command needs, so is refused no
    matter which one is checked). Neither can tell "reads required_grant off
    the command" apart from "always checks system_control". A caller holding
    screen but not system_control can: it must be let through for screenshot
    (screen) and refused for lock_workstation (system_control), which only
    happens if the route looks at the command's own declared grant.
    """
    client, runtime, tokens = context
    allowed = client.post("/v1/commands/screenshot/run", headers=head(tokens["screen_only"]))
    assert allowed.status_code == 200
    assert runtime.commands.ran == ["screenshot"]

    refused = client.post("/v1/commands/lock_workstation/run",
                          headers=head(tokens["screen_only"]))
    assert refused.status_code == 403
    assert runtime.commands.ran == ["screenshot"]  # unchanged -- nothing extra ran


# ─── fix wave: running a command is throttled tighter than the shared budget ─
def test_running_commands_is_throttled_tighter_than_the_shared_budget(context):
    """A SCREEN-granted device left calling POST /v1/commands/screenshot/run
    in a loop used to be bounded only by the shared 120/60s budget every
    other route shares -- ~120 captures a minute into SANDBOX_DIR/captures
    with no retention. A tighter, route-scoped budget bounds it the same
    way backup/run's does.
    """
    client, _, tokens = context
    codes = [client.post("/v1/commands/volume_up/run", headers=head(tokens["full"])).status_code
             for _ in range(25)]
    assert 429 in codes, "commands/run never engaged its own, tighter budget"


def test_a_command_declaring_an_unknown_grant_fails_closed(tmp_path):
    """`Capability(match.required_grant)` raises for any string outside the
    enum. Fail-closed means the run never happens, not merely a 5xx status --
    a 500 that still executed the command would be exactly the kind of
    refusal-with-a-side-effect the other tests in this file exist to catch."""

    class _BogusGrantCommandRuntime:
        def __init__(self) -> None:
            self.ran: list[str] = []

        async def catalogue(self) -> list[CommandDef]:
            return [CommandDef("mystery", "Mystery", "Declares a grant nobody has.",
                               False, "reticulate_splines")]

        async def run(self, command_id: str) -> CommandOutcome:
            self.ran.append(command_id)
            return CommandOutcome(True, "done")

    vault = TokenVault(tmp_path)
    bogus = _BogusGrantCommandRuntime()
    runtime = dataclasses.replace(build_fake_runtime(), commands=bogus)
    client = build_api_client(runtime, vault)
    token = vault.issue("studio", frozenset(Capability))

    response = client.post("/v1/commands/mystery/run", headers=head(token))

    assert response.status_code == 500
    assert bogus.ran == []
