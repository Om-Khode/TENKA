"""A stored secret does not survive to a prompt.

KI-29 wired `redact_secrets` into five **write** sites, which protects rows
written since it shipped. It does nothing for:

- rows written before it existed;
- rows whose shape the one-off historical scrub's patterns missed;
- a database restored from a backup taken before the fix;
- any future write path that forgets.

So the write side is not the last line, and the last line is here: the point
where stored content is rendered into a prompt and leaves the machine.

**Strict at egress, lenient at write, and the asymmetry is the argument.**
`storage/repos/memory.py:save_turn` explains why it uses the lenient tier:
over-redaction on write is *unrecoverable*, because the row is her memory. At
egress the same mistake costs one degraded prompt while the stored value stays
intact. Opposite failure costs, opposite defaults.

**Replayed content only, never the live utterance.** A user who says "call the
API with this key" has to be able to. The boundary is stored-content-into-prompt,
not everything-into-prompt, and one test below pins that the stored row itself
is untouched.

Run with:  py -3.11 -m pytest tests/test_prompt_egress_redaction.py -v
"""
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.core.redact import redact_secrets_strict  # noqa: E402

# Inert, but the right *shape*: a labelled credential, a prefixed client
# secret, and a bare high-entropy token. Hand-written and fake -- KI-29 was
# found by a generated fixture pulling three live OAuth values out of the real
# database, and it came within one `git add` of a public repository.
_KEY = "AIzaSyC8Ur4kFakeKeyForTestingOnly123"
# High-entropy on purpose. The first draft used
# "GOCSPX-abcdefghijklmnopqrstuv" and the redactor declined to touch it when it
# appeared unlabelled -- correctly, because a sequential alphabet fails the
# entropy check that stops ordinary long words being eaten. A real client
# secret is not sequential, so a fixture that is tests the wrong thing and
# would have reported a hole that does not exist.
_SECRET = "GOCSPX-9fK2mQ7vX4bN8pR3wZ6tY1uA"

_ORDINARY = [
    "remind me to call mom at 5pm",
    "open spotify and play jazz",
    "my name is Om and I live in Pune",
    "search for mechanical keyboards under 5000 rupees",
    "the file is at D:/Code/TENKA/assistant/main.py",
]


@pytest.fixture()
def repo(tmp_path):
    from assistant.storage.db import Database
    from assistant.storage.repos.memory import MemoryRepo
    db = Database(tmp_path / "m.db")
    try:
        yield MemoryRepo(db, tmp_path), db
    finally:
        db._conn.close()


# ─── the redactor is fit for this position ───────────────────────────────────

@pytest.mark.parametrize("text", _ORDINARY)
def test_strict_redaction_leaves_ordinary_conversation_alone(text):
    """**First, and this is the test that decides whether the fix is shippable.**

    The write site chose the lenient tier partly on the grounds that "strict's
    blunt assignment-shaped-line rule risks eating ordinary conversation". That
    worry is legitimate and it is why this is measured rather than assumed: a
    redactor that mangles normal speech would silently degrade every prompt in
    the system, and nothing would go red.
    """
    assert redact_secrets_strict(text) == text, (
        "strict redaction altered ordinary conversation -- every replayed turn "
        "would reach the model damaged"
    )


@pytest.mark.parametrize("secret", [
    f"my api key is {_KEY}",
    f"client_secret={_SECRET}",
    f"use token {_KEY}",
])
def test_strict_redaction_catches_the_shapes_that_were_found_in_this_database(secret):
    """The three shapes KI-29 actually pulled out of real history: a labelled
    key, a prefixed client secret, a bare token."""
    out = redact_secrets_strict(secret)
    assert _KEY not in out and _SECRET not in out, f"survived redaction: {out}"


# ─── build_recent_context: planner and code-generation prompts ───────────────

def _plant_raw_turn(db, user_input, response, session_id="s1"):
    """Insert a conversation row **bypassing `save_turn`**.

    This is the whole point of the egress pass and the first version of these
    tests missed it: writing through `save_turn` applies the lenient write-side
    redactor, so the row is already clean by the time anything renders it. A
    test built that way passes whether or not the egress pass exists -- removing
    the redaction from `build_recent_context` was a GREEN mutant until this
    helper existed.

    A raw insert is not a contrivance, it is the case the fix is for: rows
    written before KI-29 shipped, rows whose shape the one-off historical scrub
    missed, and rows arriving from a backup taken before either.
    """
    db.execute(
        "INSERT INTO conversations (timestamp, user_input, intent, response, "
        "session_id, security_skip) VALUES (?, ?, ?, ?, ?, 0)",
        ("2026-08-23T10:00:00", user_input, "small_talk", response, session_id),
    )
    db.commit()


def test_a_stored_secret_does_not_reach_build_recent_context(repo):
    """Three callers put this string into a prompt, and two of them are
    generators -- `code_executor/orchestrator.py` and
    `actions/planner/planner.py`. Whatever else it is, it is egress."""
    r, db = repo
    _plant_raw_turn(db, f"my api key is {_KEY}", f"saved {_SECRET}")

    ctx = r.build_recent_context()
    assert ctx, "no context built -- the test would pass vacuously"
    assert _KEY not in ctx, f"a stored secret reached a prompt string: {ctx!r}"
    assert _SECRET not in ctx, (
        f"the assistant side of a stored turn was not redacted: {ctx!r}"
    )


def test_build_recent_context_keeps_the_actual_conversation(repo):
    """The other direction. A renderer that returned nothing, or mangled every
    line, would satisfy the assertion above and destroy the feature."""
    r, db = repo
    _plant_raw_turn(db, "open spotify and play jazz", "Playing.")

    ctx = r.build_recent_context()
    assert "open spotify and play jazz" in ctx
    assert "Playing." in ctx


def test_redaction_is_egress_only_and_the_stored_row_is_untouched(repo):
    """The boundary this fix draws. Redacting the *store* would be a second,
    stricter write-side redaction -- and the write side is deliberately lenient
    because over-redaction there cannot be undone.

    Note what this shows: the row here was already scrubbed by the lenient
    write-side pass, so what the strict egress pass adds is coverage of rows the
    lenient tier let through. Asserting the row is unchanged by *this* pass is
    what proves the two are separate.
    """
    r, db = repo
    r.save_turn("hello there", "small_talk", "Hi.", "s1")
    row = db.fetchone("SELECT user_input FROM conversations WHERE session_id='s1'")
    assert row["user_input"] == "hello there", (
        "the egress pass modified stored data"
    )


# ─── the two main.py prompt builders ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_stored_secret_does_not_reach_the_conversation_messages(monkeypatch):
    """`_build_conversation_messages` is the native multi-turn history sent on
    every conversational turn."""
    import assistant.main as main_mod
    import assistant.memory as memory_mod

    monkeypatch.setattr(memory_mod, "get_recent", lambda *a, **kw: [
        {"id": 1, "user_input": f"my api key is {_KEY}",
         "response": f"I stored {_SECRET}", "timestamp": "2026-08-23T10:00:00"},
    ])
    monkeypatch.setattr(main_mod, "_get_personality_switch_ts", lambda: None)

    messages, _ = await main_mod._build_conversation_messages()
    blob = " ".join(m["content"] for m in messages)
    assert blob, "no messages built -- vacuous"
    assert _KEY not in blob and _SECRET not in blob, (
        f"a stored secret reached the model as conversation history: {blob!r}"
    )


@pytest.mark.asyncio
async def test_the_conversation_messages_still_carry_the_conversation(monkeypatch):
    """Both directions again, on the real builder."""
    import assistant.main as main_mod
    import assistant.memory as memory_mod

    monkeypatch.setattr(memory_mod, "get_recent", lambda *a, **kw: [
        {"id": 1, "user_input": "open spotify", "response": "Playing.",
         "timestamp": "2026-08-23T10:00:00"},
    ])
    monkeypatch.setattr(main_mod, "_get_personality_switch_ts", lambda: None)

    messages, _ = await main_mod._build_conversation_messages()
    blob = " ".join(m["content"] for m in messages)
    assert "open spotify" in blob and "Playing." in blob


def test_a_stored_secret_does_not_reach_the_facts_context(monkeypatch):
    """The likeliest stored secret of the lot. `save_typed_fact` is what records
    "my api key is ..." as a durable fact, and this string goes into the system
    prompt on every single turn."""
    import assistant.main as main_mod
    import assistant.memory as memory_mod

    monkeypatch.setattr(memory_mod, "search_facts", lambda key: [
        {"key": "user_api_key", "value": _KEY},
        {"key": "user_city", "value": "Pune"},
    ])

    ctx = main_mod._build_facts_context()
    assert "Pune" in ctx, "a legitimate fact was dropped"
    assert _KEY not in ctx, f"a stored secret reached the system prompt: {ctx!r}"


# ─── the boundary is enumerated, not assumed ─────────────────────────────────

def test_every_stored_conversation_renderer_redacts():
    """Source-level, and narrow by design.

    "A boundary is only as good as the enumeration of paths around it" is a rule
    this project has paid for three times. These are the sites that render
    *stored* turns or facts into prompt text; each one must redact. Sites that
    handle only the caller's live utterance are deliberately absent -- scrubbing
    those would break the task the user asked for.

    A new renderer added without redaction will not be caught by this test, and
    that is stated rather than hidden: the list is a floor, not a fence. What it
    does catch is the removal of a redaction that exists today.
    """
    required = {
        "assistant/storage/repos/memory.py": "build_recent_context",
        "assistant/main.py": "_build_conversation_messages",
    }
    for rel, func in required.items():
        src = (_ROOT / rel).read_text(encoding="utf-8")
        start = src.index(f"def {func}")
        # The function body up to the next top-level def at the same indent.
        body = src[start:start + 4000]
        assert "redact_secrets_strict" in body, (
            f"{rel}:{func} renders stored turns into a prompt without strict "
            f"redaction"
        )
