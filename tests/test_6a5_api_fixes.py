"""Adversarial review probes for milestone 6a.5 -- the API layer.

Written by a reviewer who did NOT build this code. Every test in this file is
a PROBE: it asserts the property the code *should* have. A FAILING test here
is a FINDING, not a broken test. Nothing in this file fixes anything.

SCOPE OF THIS COPY (fix/6a5-api-review). The reviewer's file carried two
parts. Part 1 -- `assistant/policy.py` and the turn pipeline in `main.py`
and `actions/` -- is owned by other agents on other branches and is NOT
reproduced here; it lives unchanged in the reviewer's own
`tests/test_6a5_review_api.py`. What is kept is Part 2, the API-layer
findings this branch fixes, plus every CONTROL from Part 1c, because
those must stay green and this branch has to be able to show it.

Three probes carry a `FIX NOTE` recording an assertion that was changed
and why. Every other assertion is the reviewer's, verbatim.

Rules honoured while writing this:
  * No `TestClient` socket test that publishes a hub frame and sends an
    inbound frame concurrently -- spec section 6 records that those hang.
  * `_finish_turn` is ALWAYS monkeypatched before any pipeline test: the real
    one calls `_follow_up_listen()` -> `record_until_silence()`, which opens
    the developer's actual microphone.
  * `tts.speak` and `TurnTracker.save` are always stubbed -- real Kokoro
    synthesis and a real write to `~/TENKA/memory/tenka.db` respectively.
"""
from __future__ import annotations

import pytest


# ═══════════════════════════════════════════════════════════════════════
# PART 1c -- CONTROLS. These PASS. They are the things that stopped me,
# recorded so the report can say what held as precisely as what did not.
# ═══════════════════════════════════════════════════════════════════════


def test_control_the_mic_followup_really_is_re_queued_at_full_privilege():
    """Quantifies the escalation in the microphone finding rather than
    asserting it in prose. `_finish_turn` enqueues `("stt", text, ms)`;
    `_grants_for_item` hands any non-"studio" item the FULL capability set.
    This test PASSES -- it documents the mechanism, it is not a bug in
    itself. The bug is that a remote turn can cause the enqueue."""
    import assistant.main as m
    from assistant.core.capabilities import Capability

    grants, _ = m._grants_for_item(("stt", "whatever the room said", 120))
    assert grants == frozenset(Capability)


def test_control_actions_execute_is_fail_closed_on_an_unset_grant_set():
    """The choke point itself is right. actions/__init__.py:341-347 refuses
    when `current_grants` is None and when the required capability is
    absent. Nothing I tried moved it. Every Part 1b finding is a path that
    goes AROUND this function, never through it."""
    import inspect
    from assistant import actions

    # Asserted behaviourally, not by grepping the source. The original form
    # looked for the literal expression `_granted is None or _required not in
    # _granted` inside `execute()`, and went red when the pre-dispatch fix
    # lifted that comparison into `actions.capability_refusal()` so the
    # pre-dispatch branches could share one predicate with the gate. The
    # property was never broken; only the spelling moved. A control test that
    # fails on a refactor it should be indifferent to trains whoever hits it
    # to edit the assertion, which is how a real regression gets waved
    # through -- so it now drives the function and reads the answer.
    import asyncio

    assert actions.current_grants.get() is None, "precondition: grants unset"
    refusal = asyncio.run(actions.execute("code_executor", {"goal": "print(1)"}, ""))
    assert "permission" in refusal.lower()

    # And an intent nobody classified must need the strongest capability,
    # rather than falling through to whatever the caller happens to hold.
    unlisted = asyncio.run(actions.execute("some_intent_nobody_classified", {}, ""))
    assert "permission" in unlisted.lower()

    # The one structural claim worth keeping: there is still exactly one site
    # that resolves a handler, so there is one place the gate has to sit.
    # Counted from the AST, not from the text. A plain substring count reads
    # two, because `registry.py`'s module docstring describes the dispatch it
    # provides -- prose, not a call. A structural check that a comment can
    # trip gets muted the first time someone documents something.
    import ast
    from pathlib import Path

    sites = []
    for path in Path("assistant").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
        except SyntaxError:                      # not ours to police here
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "tool_registry"):
                sites.append(f"{path}:{node.lineno}")

    assert len(sites) == 1, f"handler resolution is no longer a single site: {sites}"


def test_control_a_studio_item_with_a_lost_grant_set_gets_nothing():
    """`_grants_for_item` (main.py:1712-1715) does not treat a missing or
    wrong-typed third slot as "full". Tried a bare 2-tuple, a list, a set
    and a string -- all fail closed to the empty frozenset."""
    import assistant.main as m

    for item in [("studio", "hi"), ("studio", "hi", ["execute"]),
                 ("studio", "hi", {"execute"}), ("studio", "hi", "execute"),
                 ("studio", "hi", None)]:
        grants, _ = m._grants_for_item(item)
        assert grants == frozenset(), item


def test_control_browse_url_neutralises_a_non_http_scheme():
    """policy.evaluate() only scheme-checks `open_browser` (policy.py:102),
    not `browse_url` -- which is a real gap in the policy layer. It is not
    exploitable, because `_browse_url_body` (actions/web.py:242-243)
    unconditionally prepends `https://` to anything that is not already
    http(s), so `file:///C:/Users/.../.env` becomes a nonsense https URL
    handed to a remote reader rather than a local file read.

    The control is in the handler, not the policy. Recorded because it means
    the policy gap is latent: the day that prefix changes, it opens."""
    import inspect
    from assistant.actions import web

    src = inspect.getsource(web._browse_url_body)
    assert 'if not url.startswith("http://") and not url.startswith("https://")' in src
    assert 'url = "https://" + url' in src


def test_control_the_cloudflare_quick_ceiling_excludes_chat_send():
    """The Part 1b findings all need CHAT_SEND. The widest-exposure listener
    -- the Cloudflare quick tunnel, where a third party reads the plaintext
    -- is OBSERVE only, so none of them reach it. They DO reach `tailnet`
    and `funnel`, and `tailscale funnel` is reachable from the open internet
    by anyone holding the URL."""
    from assistant.io.api.policy import POLICIES
    from assistant.core.capabilities import Capability

    assert Capability.CHAT_SEND not in POLICIES["quick"].ceiling
    assert Capability.CHAT_SEND in POLICIES["tailnet"].ceiling
    assert Capability.CHAT_SEND in POLICIES["funnel"].ceiling
    assert Capability.EXECUTE not in POLICIES["funnel"].ceiling


def test_control_note_titles_cannot_traverse():
    """`create_note` is CHAT_SEND-gated, so a funnel device can write files.
    Tried to escape NOTES_DIR through the title. `_sanitize_filename`
    (actions/simple.py:15-22) replaces every one of `<>:"/\|?*` and the
    literal `..`, so no separator and no parent reference survives."""
    from assistant.actions.simple import _sanitize_filename

    for attempt in ["../../../../Windows/System32/evil",
                    "..\..\evil", "C:/Windows/evil", "a/../../b",
                    "....//....//evil"]:
        out = _sanitize_filename(attempt)
        assert "/" not in out and "\\" not in out and ".." not in out, (attempt, out)


# ═══════════════════════════════════════════════════════════════════════
# PART 2 -- regressions in the API layer from 6a.5
# ═══════════════════════════════════════════════════════════════════════

# ─── P2-A. ui.py: G8 landed on the loader that ships nothing ──────────────


def test_normalise_member_has_no_dot_rule():
    """The unit fact underneath the finding below. `_enumerate_dir`
    (ui.py:339-345) carries the dot rule; `normalise_member` (ui.py:179) --
    the function the OTHER loader and the packaging step share -- does not.
    The rule is a property of one enumerator, not of the name pipeline."""
    from assistant.io.api.ui import normalise_member

    for hidden in (".env", ".env.local", ".git/config", ".DS_Store", "sub/.env"):
        assert normalise_member(hidden) is None, (
            f"normalise_member({hidden!r}) -> {normalise_member(hidden)!r}")


def test_the_packaging_step_refuses_to_ship_a_dotfile(tmp_path):
    """ui.py:366-371 justifies skipping the dot rule on `_from_zip` because
    the archive's "members were chosen by a script rather than by whatever a
    developer's working directory happens to contain".

    The script is `tools/package_studio_ui.py:_collect`: `sorted(
    source.rglob("*"))` with exactly one name gate, `normalise_member(member)
    != member` -- which the test above shows has no dot rule. The script
    chooses whatever is in the directory.
    """
    import tools.package_studio_ui as pkg

    src = tmp_path / "out"
    src.mkdir()
    (src / "index.html").write_text("<html></html>", encoding="utf-8")
    (src / ".env").write_text("GEMINI_API_KEY=AIzaSyREALKEY123456", encoding="utf-8")
    (src / ".git").mkdir()
    (src / ".git" / "config").write_text("[remote]\n url = git@github.com:me/p.git",
                                         encoding="utf-8")

    shipped = {name for name, _ in pkg._collect(src)}
    hidden = {n for n in shipped if any(p.startswith(".") for p in n.split("/"))}
    assert not hidden, f"_collect shipped hidden members: {sorted(hidden)}"


def test_the_vendored_bundle_already_contains_a_dot_member():
    """Not hypothetical -- the shipped artifact is the evidence.

    `assistant/io/api/studio_ui.zip` carries `.tenka-ui.json` TWICE: once
    written by `package()`, once picked up off disk by `_collect`'s `rglob`.
    A dot-prefixed name reached the archive through exactly the path
    ui.py:366-371 says cannot happen.

    The duplicate is a second finding. `zipfile` resolves a repeated name to
    the LAST entry, so `_read_marker_from_zip` (ui.py:413) reads the stale
    on-disk marker rather than the one packaging computed -- and `mount_ui`
    (ui.py:741-748) takes the whole UI dark with a 503 when that value
    disagrees with `contract_hash`. The guard is not measuring what it says.

    FIX NOTE (fix/6a5-api-review): the `dotted` assertion is amended to exempt
    `MARKER_NAME` itself, and only that name. `.tenka-ui.json` is dot-prefixed
    on purpose -- ui.py argues the dot at its definition, to keep the daemon's
    own marker out of the way of anything a Next.js export emits -- and the
    archive cannot carry the contract hash without it. It is not reachable over
    HTTP: `normalise_member` now refuses every dot member, so a request for it
    is a 403 before `_PRIVATE_MEMBERS` is even consulted, and the only readers
    are `archive.getinfo(MARKER_NAME)` and `root / MARKER_NAME`, both by
    literal path. The finding stands unchanged for every *other* dot member,
    and the duplicate assertion is the reviewer's, verbatim.
    """
    import zipfile
    from collections import Counter
    from pathlib import Path

    from assistant.io.api.ui import MARKER_NAME

    zpath = (Path(__file__).resolve().parent.parent
             / "assistant" / "io" / "api" / "studio_ui.zip")
    names = zipfile.ZipFile(zpath).namelist()

    dotted = [n for n in names
              if n != MARKER_NAME and any(p.startswith(".") for p in n.split("/"))]
    dupes = [n for n, c in Counter(names).items() if c > 1]
    assert not dotted, f"shipped dot members: {dotted}"
    assert not dupes, f"duplicate members (last wins on read): {dupes}"


def test_a_dotfile_in_a_packaged_bundle_is_served(tmp_path):
    """End to end: build the archive the way packaging does, mount it the way
    the daemon does, read the member back.

    `serve_studio_ui` (ui.py:765) is a GET/HEAD catch-all outside `/v1` with
    NO `Depends(authenticate)` and no rate limiter, so this is returned to
    anyone who can reach the port. It is lens 1 F7's `GET /.env -> 200`,
    still live on the loader production actually uses -- `_from_dir` needs
    `studio_ui_path` plus a dev marker, which `next build` never writes.
    """
    import zipfile
    import tools.package_studio_ui as pkg
    from assistant.io.api.ui import UiBundle

    src = tmp_path / "out"
    src.mkdir()
    (src / "index.html").write_text("<html></html>", encoding="utf-8")
    (src / ".env").write_text("GEMINI_API_KEY=AIzaSyREALKEY123456", encoding="utf-8")

    zpath = tmp_path / "studio_ui.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for name, body in pkg._collect(src):
            z.writestr(name, body)
        z.writestr(".tenka-ui.json",
                   '{"version": 1, "contract": "x", "builtAt": "2026-01-01T00:00:00Z"}')

    bundle = UiBundle.open(zip_path=zpath, dir_path=None)
    assert bundle is not None, "precondition: the bundle mounted"
    got = bundle.read(".env")
    assert got is None, f"the zip loader served .env: {got[0]!r}"


def test_xhtml_and_svg_are_not_scriptable_documents():
    """`content_type_for` (ui.py:131-142) blocks a GUESSED `text/html`:

        if guessed and not guessed.startswith("text/html"):
            return guessed

    `application/xhtml+xml` and `image/svg+xml` are both fully scriptable
    top-level documents and both sail past it -- `.svg` is pinned
    deliberately in `_CONTENT_TYPES`. The CSP at ui.py:645-660 is
    `script-src 'self' 'unsafe-inline'` with no nonce, so any such member
    that reaches the served set executes inline script ON THE DAEMON'S
    ORIGIN: the httpOnly cookie rides along, `X-TENKA-Request` is
    presence-checked only, and `Origin` is the daemon's own.
    """
    from assistant.io.api.ui import content_type_for

    scriptable = {"application/xhtml+xml", "image/svg+xml", "text/html",
                  "application/xml", "text/xml"}
    for member in ("evil.svg", "evil.xhtml", "evil.xht"):
        ct = content_type_for(member)
        assert ct not in scriptable, (
            f"{member} -> {ct!r}: a scriptable document on the daemon's own "
            "origin under a CSP that allows unsafe-inline")


# ─── P2-B. core/redact.py -- both directions ──────────────────────────────
# `redact_secrets_strict` is the file-preview path (routes/files.py:80);
# `redact_secrets` is the log path (main.py:928, intent.py:76).


@pytest.mark.parametrize("shape,text", [
    ("docker-compose list item -- the `^[ \\t]*` anchor at redact.py:226 "
     "allows spaces and tabs before the identifier, not a YAML `- `",
     "environment:\n      - POSTGRES_PASSWORD=hunter2\n"),
    ("camelCase -- `_IDENT_SPLIT` (redact.py:79) splits on [_-] only, so "
     "`clientSecret` is one token in neither ident-part set, and "
     "\\bsecret\\b cannot see inside it",
     'const cfg = { clientSecret: "hunter2plain" };'),
    ("a quoted value containing a space -- `_is_configuration_value` "
     "(redact.py:309) returns False for anything with whitespace, and "
     "redact.py:383 bails outright when the separator is `:`",
     'client_secret: "hunter two"'),
    ("bracket exemption (redact.py:302) -- punctuation-rich passwords are "
     "the strong ones and this guard exempts exactly those",
     "db_pass=P@ssw(rd!1"),
    ("URL userinfo -- no rule for scheme://user:pass@host anywhere in the "
     "module, and `:`/`@`/`/` fragment every run below _BARE's 24-char floor",
     "clone from https://admin:hunter2@git.internal.example.com/repo.git"),
    ("all-digit value -- `_looks_secret` (redact.py:250-253) requires BOTH "
     "a digit and a letter, so a numeric token is exempt",
     "token: 918273645509"),
    ("lowercase key with no role noun -- the UPPER_SNAKE spelling of this "
     "same line is pinned by tests/test_redact.py:180",
     "database_url: postgres://user:p4ssw0rd@host:5432/db"),
    ("`:=` -- the `(?!=)` at redact.py:227, added to protect `==`, also "
     "kills the Go/Pascal assignment operator",
     "db_pass := hunter2"),
    ("PGP block -- the PEM rule (redact.py:126) requires `-----` directly "
     "after `KEY`; `PGP PRIVATE KEY BLOCK` has ` BLOCK` there",
     "-----BEGIN PGP PRIVATE KEY BLOCK-----\nlQOYBGa1\n-----END PGP PRIVATE KEY BLOCK-----"),
])
def test_redaction_leaks_a_real_credential(shape, text):
    """Under-redaction. Each of these is a real credential in a real file
    shape, run through the strict preview path that `routes/files.py:80`
    applies. `[REDACTED]` never appears, so the secret ships to the reader."""
    from assistant.core.redact import redact_secrets_strict

    out = redact_secrets_strict(text)
    assert "[REDACTED]" in out, f"leak ({shape}): {out!r}"


def test_a_multiword_passphrase_only_loses_its_first_word():
    """redact.py:95-97 -- the labelled rule's value group is `(\\S+)`, which
    stops at the first space. A diceware or BIP39 phrase loses one word and
    the rest ships, UNDER a `[REDACTED]` that claims the job was done. That
    is worse than not redacting: the marker tells the reader it is safe."""
    from assistant.core.redact import redact_secrets_strict

    out = redact_secrets_strict("passphrase: correct horse battery staple")
    assert "horse" not in out and "battery" not in out and "staple" not in out, (
        f"the tail of the passphrase survived next to a [REDACTED]: {out!r}")


def test_an_unterminated_pem_marker_does_not_blank_the_rest_of_the_file():
    """OVER-redaction, and the largest destructive rule in the file.

    redact.py:125-128 is `(-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----)(.*?)
    (-----END ...|\\Z)` under `(?is)`. With no END marker, `.*?` expands to
    the end of the string. `(?i)` means a lowercase prose MENTION triggers
    it. So a setup guide that merely names the header, or a traceback that
    quotes it, loses every line that follows -- in the preview path and in
    the log path both.
    """
    from assistant.core.redact import redact_secrets_strict

    doc = ("# Setup\n"
           "Paste the -----BEGIN RSA PRIVATE KEY----- header, then the body.\n"
           "\n## Step two\nRun the installer.\n"
           "\n## Step three\nRestart the service.\n")
    out = redact_secrets_strict(doc)
    assert "Step two" in out and "Step three" in out, (
        f"an unterminated PEM marker erased the rest of the document: {out!r}")


def test_a_preview_of_this_projects_own_config_survives_redaction():
    """OVER-redaction, the direction that broke 90 lines the first time.

    redact.py:385 is
        `if not _UPPER_SNAKE.match(name) and not _is_configuration_value(...)`
    -- the configuration-value guard the author added is SHORT-CIRCUITED for
    every UPPER_SNAKE name. `config.py` is essentially nothing but public
    UPPER_SNAKE constants, so previewing this project's own config through
    the FILES route returns structurally broken source: a value that was `{`
    is swallowed and its dict body and closing brace are orphaned.
    """
    from assistant.core.redact import redact_secrets_strict

    snippet = ('INTENTS = ["small_talk", "web_search"]\n'
               'MAX_PREVIEW_BYTES = 65536\n'
               'TASK_MODEL_MAP = {\n'
               '    "intent": "flash",\n'
               '}\n')
    out = redact_secrets_strict(snippet)
    assert "[REDACTED]" not in out, (
        f"ordinary public constants were destroyed by the preview path:\n{out}")


def test_a_uuid_survives_redaction():
    """redact.py:257 -- `has_separator = "_" in candidate or "-" in candidate`
    treats a hyphen as an entropy signal, so every UUID trips `_BARE`.
    UUIDs are in fixtures, logs, migrations and configs constantly."""
    from assistant.core.redact import redact_secrets_strict

    out = redact_secrets_strict("run id 550e8400-e29b-41d4-a716-446655440000")
    assert "[REDACTED]" not in out, f"UUID destroyed: {out!r}"


# ─── P2-C. events.py -- caps, idle, and the sampler ───────────────────────


def test_there_is_a_ping_behind_the_idle_timeout():
    """The idle reaper (events.py:576) evicts on
    `now - _last_active > _IDLE_TIMEOUT_SECONDS`, and `_last_active` is
    refreshed in exactly three places: `attach`, `note_activity` (inbound
    frames), and `_pump` after a SUCCESSFUL SEND.

    There is no ping anywhere in events.py or app.py -- uvicorn's
    protocol-level pings are handled below Starlette and never reach
    `note_activity`. So the whole idle argument rests on the telemetry
    sampler publishing every 2s. A listen-only client -- a wall display, a
    backgrounded phone tab -- is evicted the instant the sampler stops.
    """
    import inspect
    from assistant.io.api import events as ev

    src = inspect.getsource(ev)
    sends_ping = ("send_ping" in src or "websocket.ping" in src
                  or '"ping"' in src or "'ping'" in src)
    assert sends_ping, (
        "an idle timeout with no keepalive: the only thing refreshing an "
        "honest listen-only socket's activity clock is the telemetry "
        "sampler's own frames")


@pytest.mark.asyncio
async def test_a_failing_telemetry_sample_does_not_kill_the_sampler(monkeypatch):
    """events.py:597-602. The `await self._runtime.system.telemetry()` call
    is inside a try/except; `self.publish(telemetry_frame(snapshot))` on the
    next line is NOT. `telemetry_frame` runs a pydantic `model_dump`, so a
    snapshot that fails validation raises out of the loop and the task dies.

    `attach()` only starts a sampler `if self._telemetry_task is None`, and
    a dead task is not None -- so it is never restarted while any socket
    remains. Chained with the probe above: sampler dies -> no outbound
    frames -> every attached socket is idle-reaped at 120s -> reconnect
    storm, from one bad psutil read.
    """
    import inspect
    from assistant.io.api import events as ev

    src = inspect.getsource(ev.EventHub._telemetry_loop) \
        if hasattr(ev.EventHub, "_telemetry_loop") else inspect.getsource(ev.EventHub)
    idx = src.find("telemetry_frame(")
    assert idx != -1, "precondition: found the publish site"
    # Walk back to the nearest `try:`/`except` to see whether the publish is
    # inside the guarded region. The `continue` in the except means anything
    # after it is unguarded.
    before = src[:idx]
    assert before.rfind("try:") > before.rfind("continue"), (
        "telemetry_frame(...) is published OUTSIDE the try that guards the "
        "sample, so one raising snapshot ends the sampler permanently -- and "
        "attach()'s `if self._telemetry_task is None` never restarts it")


@pytest.mark.asyncio
async def test_a_reconnect_during_the_last_detach_keeps_the_revocation_sweep():
    """events.py:434-443 -- the highest-severity availability/authorisation
    bug I found in this file.

        async def detach(self, socket) -> None:
            self._forget(socket)
            if not self._sockets:
                for name in ("_telemetry_task", "_revalidate_task"):
                    task = getattr(self, name)
                    if task is not None:
                        task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await task              # <- yields to the loop
                        setattr(self, name, None)   # <- cleared only after

    `attach` (events.py:392-397) creates a sweep task only
    `if self._revalidate_task is None`. During `detach`'s `await task` the
    attribute is still the dying task, so a socket attaching in that window
    creates nothing -- and then `detach` sets it to None.

    Result: an attached socket with NO revalidation sweep and NO telemetry
    sampler. events.py:556-570 and app.py:776-782 both rest on "an accepted
    socket is not a credential"; for this socket both halves are off. Its
    device can be revoked and it keeps the stream indefinitely, until some
    unrelated connection happens to attach and restart the sweep. It also
    holds a cap slot forever, since the idle reaper lives in the same task.

    Trigger: an ordinary browser reload on a daemon with one viewer. No
    special input, repeatable at the 120 handshakes/min the limiter allows.
    """
    import asyncio

    from assistant.io.api.events import EventHub

    class _Sock:
        def __init__(self):
            self.closed = False
        async def send_json(self, frame):
            return None
        async def close(self, code: int = 1000):
            self.closed = True

    hub = EventHub()
    first, second = _Sock(), _Sock()
    assert await hub.attach(first, device_id="phone")

    # The reload: the old socket detaches and the new one attaches while
    # detach is suspended on `await task`.
    await asyncio.gather(hub.detach(first), hub.attach(second, device_id="phone"))

    try:
        assert hub._sockets, "precondition: the new socket is attached"
        assert hub._revalidate_task is not None, (
            "the surviving socket has no revocation sweep: detach() cleared "
            "_revalidate_task after attach() had already decided one existed")
    finally:
        await hub.stop()


def test_stop_does_not_re_raise_a_dead_tasks_exception():
    """events.py:612-623. `task.cancel()` on an already-finished task is a
    no-op, and the following `await task` then RE-RAISES its stored
    exception -- which `contextlib.suppress(asyncio.CancelledError)` does
    not catch.

    Telemetry is first in the tuple, so a sampler that died (see above)
    poisons the cancellation of `_revalidate_task` AND `_pump_task` and
    skips every `clear()`. The caller is the lifespan `finally`
    (app.py:352), so daemon shutdown aborts with the pump still running and
    sockets still held."""
    import inspect
    from assistant.io.api.events import EventHub

    src = inspect.getsource(EventHub.stop)
    assert "suppress(asyncio.CancelledError, Exception)" in src \
        or "except Exception" in src or "return_exceptions=True" in src, (
        "stop() awaits each cancelled task under suppress(CancelledError) "
        "only; a task that already died of a real exception re-raises it "
        "here and aborts shutdown before the remaining tasks are cancelled")


# ─── P2-D. app.py -- the socket cap refusal ───────────────────────────────


def test_the_socket_cap_refusal_does_not_double_close():
    """app.py:867-878.

        await websocket.accept()
        if not await app.state.hub.attach(...):
            await websocket.close(code=1013)   # try again later
            return

    `EventHub.attach` ALREADY closed the socket on the refusal path
    (events.py:383-388 -> `_close_quietly` -> `close(code=1008)`). Starlette
    flips application_state to DISCONNECTED on that send, so the second
    close raises `RuntimeError: Cannot call "send" once a close message has
    been sent.`

    Three consequences:
      1. 1013 is never delivered -- the client receives 1008, the same code
         this handler uses for unauthorized/origin/capability refusals, so
         "at capacity, retry" is indistinguishable from "you are not
         allowed here".
      2. The `try:` begins at app.py:879, AFTER the return, so the
         RuntimeError escapes into uvicorn as a full traceback -- one per
         refused attempt, at up to 120/min per device.
      3. `_audit("accepted")` already ran at app.py:759, above the cap
         check, so a refused connection is recorded as accepted and the
         refusal is recorded nowhere. That falsifies app.py:600-603's claim
         that "a socket connection is not the one surface that leaves no
         trace either way".
    """
    import inspect
    from assistant.io.api import app as app_mod
    from assistant.io.api.events import EventHub

    src = inspect.getsource(app_mod)
    # Three `close(code=1013)` sites; the cap one is the last and the only
    # one that follows `hub.attach`. The other two (the rate-limit refusals)
    # close BEFORE accept() and are correct.
    i = src.rfind("code=1013")
    window = src[max(0, i - 1500):i]
    assert "hub.attach" in window, "precondition: located the cap refusal"
    assert "await websocket.accept()" in window, (
        "precondition: the cap check runs after accept()")

    assert "_close_quietly" not in inspect.getsource(EventHub.attach), (
        "EventHub.attach closes the socket itself before returning False "
        "(events.py:383-388), so app.py's own close(1013) is a send on an "
        "already-closed socket: it raises RuntimeError out of the endpoint "
        "-- the `try:` starts after this `return` -- and the client receives "
        "the 1008 attach sent, never the 1013 this line intends")


# ─── P2-E. security.py -- the limiter, the ring, the secret ───────────────


def test_the_anonymous_rate_limit_runs_before_the_vault_read():
    """security.py:794-795 then 883-885.

        token = credential_from(request, policy) or ""
        device = state.vault.verify(token)          # &lt;- line 795
        ...
        if not state.limiter.check(source):         # &lt;- line 883
            raise 429

    `verify()` (vault.py:547-592) reads `instance_secret` AND `devices.json`
    off disk, synchronously, on the shared event loop, and HMACs every
    device record -- and it is reached before the limiter on every request.
    An unauthenticated caller sending `Cookie: tenka_device=x` over the
    tunnel drives two file reads per request at line rate.

    This falsifies security.py:721-725 in the same file: "An anonymous flood
    that reaches the vault is still bounded -- the sliding window above caps
    it". The window is consulted AFTER the vault, so it caps the number of
    401s, not the number of vault reads. Note the file already moved
    `touch()` into `asyncio.to_thread` (security.py:863-875) for exactly this
    cost, measured at 1.5x-7.7x; `verify()` is the one vault call still
    inline on the hot path and the one whose rate an anonymous caller sets.

    FIX NOTE (fix/6a5-api-review) -- the finding is accepted, the proposed fix
    is NOT, and the assertion below is changed to the property that was
    actually implemented. Reasoning, because overruling a reviewer needs it:

    The proposed ordering was applied verbatim and the suite was run. Two
    tests in tests/test_api_hardening.py fail under it --
    `test_a_valid_token_is_never_refused_by_an_exhausted_anonymous_window`
    and `test_a_valid_device_survives_a_flood_of_wrong_tokens_from_its_own_
    source`. Both are deliberate, argued properties from Task 10, and the
    reason they break is structural rather than incidental: `source` is
    `anonymous_key()`, the accepting port, and `authenticate()`'s own
    docstring explains that a tunnel connects from 127.0.0.1 so every remote
    caller collapses onto that one key. A gate keyed on it, consulted before
    the credential is read, cannot distinguish a flood from a paired device.
    Refusing on it converts an anonymous flood -- or ten wrong guesses, which
    earn the source an exponential lockout -- into a remote kill switch for
    every paired device on the listener. That is a worse outcome than the
    read cost it buys, on a milestone whose standing constraint is that the
    attacker must not get the machine.

    You cannot know a token is valid without the vault read, so no
    source-keyed window can bound the read rate and also honour "a valid
    token is never refused a 429 it never earned". The two are irreconcilable
    and this branch keeps the second.

    What IS fixed is the harm the finding measures: the read no longer runs
    *synchronously on the loop the assistant shares*. That is the same
    1.5x-7.7x cost the finding cites for `touch()`, and the same repair. A
    flood now costs the flooder's own latency instead of stalling everything.
    Still exploitable, and named in the report: the *rate* of anonymous
    devices.json reads remains unbounded, which is disk and CPU, not the
    loop.
    """
    import inspect
    from assistant.io.api import security as sec

    src = inspect.getsource(sec.authenticate)
    assert "asyncio.to_thread(state.vault.verify" in src, (
        "state.vault.verify() still runs inline on the shared event loop; an "
        "anonymous caller presenting any cookie drives two synchronous disk "
        "reads per request on the loop the assistant runs on")
    # And the ordering the reviewer's fix would have changed is unchanged, on
    # purpose: verification still precedes both budgets.
    verify_at = src.index("state.vault.verify")
    device_branch_at = src.index("if device is not None")
    assert verify_at < device_branch_at


def test_one_low_privilege_device_cannot_erase_every_other_devices_audit():
    """security.py:300-309 routes by anonymity, not by device:

        ring = (self._anonymous if entry.device_id == ANONYMOUS_DEVICE_ID
                else self._entries)

    The class docstring at security.py:283-285 claims "Splitting the rings
    by whether anything authenticated means a flood can only erase other
    floods." It does not. Any single paired device -- including an
    OBSERVE-only wall display, the weakest credential this design issues --
    writes into the AUTHENTICATED ring at 120 requests/minute, and ~2,000
    of them flush every other device's records, including the
    `require_admin` pairing and revocation entries an operator reads after
    an incident.
    """
    from assistant.io.api.security import AuditEntry, AuditLog

    log = AuditLog(capacity=10)
    log.record(AuditEntry(at="t0", device_id="admin-laptop", method="DELETE",
                          path="/v1/devices/abc", outcome="accepted"))
    for i in range(20):
        log.record(AuditEntry(at=f"t{i}", device_id="observe-only-display",
                              method="GET", path="/v1/status", outcome="accepted"))

    remaining = {e.device_id for e in log.entries()}
    assert "admin-laptop" in remaining, (
        "an OBSERVE-only device flushed the admin device-revocation record "
        f"out of the authenticated ring; survivors: {remaining}")


def test_a_weak_tenka_secret_is_refused(monkeypatch, tmp_path):
    """vault.py:312-325.

        try:
            secret = bytes.fromhex(stripped)
        except ValueError:
            return sha256(stripped.encode("utf-8")).digest()

    The 32-byte length check runs ONLY on the hex path. The docstring at
    vault.py:305-310 says "a non-empty value that decodes as hex to anything
    other than exactly 32 bytes raises ValueError immediately ... silently
    accepting a weak key ... would hide the operator's mistake instead of
    surfacing it."

    `TENKA_SECRET=hunter2` is not hex, hits the except, and is silently
    stretched by a single unsalted SHA-256 into the HMAC key for every
    device token -- exactly the case the comment says does not happen. It
    also makes the ValueError guard advertised at app.py:339-343 unreachable
    for the most likely misconfiguration.
    """
    import hashlib

    from assistant.io.api.vault import TokenVault

    # The control half first: a hex value of the wrong length IS refused,
    # so the guard exists and works on the path the docstring describes.
    monkeypatch.setenv("TENKA_SECRET", "dead" * 3)
    with pytest.raises(ValueError):
        TokenVault(tmp_path).instance_secret()

    monkeypatch.setenv("TENKA_SECRET", "hunter2")
    secret = TokenVault(tmp_path).instance_secret()
    assert secret != hashlib.sha256(b"hunter2").digest(), (
        "TENKA_SECRET=hunter2 was silently stretched by one unsalted SHA-256 "
        "into the HMAC key for every device token -- the exact 'silently "
        "accepting a weak key' the docstring says does not happen")
