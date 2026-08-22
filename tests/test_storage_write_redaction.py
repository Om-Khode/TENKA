"""Secrets are redacted at the storage-write boundary, not only at log sites.

`redact_secrets` was wired into eight log/preview sites and zero write sites.
A credential pasted into the chat was therefore scrubbed on its way to
`debug.log` and stored verbatim in the same turn -- the file an operator greps
after an incident was the one clean copy of it.

Storage is the worse place for it than the log, for two reasons this file
exists to keep true:

  * `conversations` is replayed into prompts, so a stored secret is re-sent to
    a cloud model on later turns;
  * `io/backup/orchestrator.py` snapshots the whole database, so a stored
    secret leaves the machine.

Found on 2026-08-22 while extracting a routing corpus from real history: three
live Google OAuth values (a client id, a `GOCSPX-` client secret, and a `4/0...`
authorization code) came straight out of `interaction_events.transcript` and
`conversations.user_input`.

Real SQLite in a tmp dir, never a mock -- mocked DBs have masked migration and
trigger failures in this tree before, and `conversations_fts` is populated by an
INSERT trigger, so the redaction has to be verified through the trigger too.

Run with:  py -3.11 -m pytest tests/test_storage_write_redaction.py -v
"""
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

REDACTED = "[REDACTED]"

# Shapes taken from what was actually found in the database, with the secret
# bodies replaced by same-shaped inert values. Each is a distinct mechanism in
# `core/redact.py`, so a regression in any one of them shows up as its own
# failure rather than hiding behind the others.
SECRETS = [
    pytest.param(
        "123456789012-abc1def2ghi3jkl4.apps.googleusercontent.com",
        id="oauth-client-id",
    ),
    pytest.param("GOCSPX-uknILdZyX5wmmBxTkqjHIzI2rex", id="oauth-client-secret"),
    pytest.param(
        "4/0AXEQxIATPXQ67qRmpXIbA2nYj2nwQzU3lEjZDeFzMwl2blxtePHSB8AwacAAGLKcaVw",
        id="oauth-auth-code",
    ),
    pytest.param("my api key is sk_live_abcd1234efgh5678ijkl", id="labelled-api-key"),
]

# Ordinary conversation that must survive untouched. Over-redaction here is
# unrecoverable -- this is her memory, and a false positive silently deletes
# something the user said. A test that only checked secrets would pass while
# blanking every turn.
INNOCENT = [
    pytest.param("My favorite color is black, what is yours?", id="preference"),
    pytest.param('delete "C:/Users/user/Desktop/Temp/notes.txt"', id="file-path"),
    pytest.param("the key thing to remember is that I hate mornings", id="weak-label"),
    pytest.param("check commit 307253d1f4a9c8b2e5d7f0a1b3c4d5e6f7a8b9c0", id="hash"),
    pytest.param("Open Wikipedia and find Alan Turing's birth year.", id="plain-goal"),
]


@pytest.fixture()
def repos(tmp_path):
    """A real database at schema head, with the repos that write to it."""
    from assistant.storage.db import Database
    from assistant.storage.repos.memory import MemoryRepo
    from assistant.storage.repos.telemetry import TelemetryRepo

    db = Database(tmp_path / "tenka.db")
    data_dir = tmp_path / "memory"
    data_dir.mkdir(parents=True, exist_ok=True)
    return db, MemoryRepo(db, data_dir), TelemetryRepo(db)


# ─── conversations ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("secret", SECRETS)
def test_a_secret_in_a_turn_is_not_stored_verbatim(repos, secret):
    db, memory, _ = repos
    memory.save_turn(f"here it is: {secret}", "small_talk", "noted", "s1")

    row = db.fetchone("SELECT user_input FROM conversations WHERE session_id = 's1'")
    stored = row["user_input"]

    assert secret not in stored, (
        f"the secret was stored verbatim in conversations.user_input: {stored!r}. "
        "This column is replayed into prompts and snapshotted to cloud backup."
    )
    assert REDACTED in stored, (
        f"nothing was redacted from {stored!r} -- the value vanished rather than "
        "being masked, which loses the turn instead of protecting it"
    )


@pytest.mark.parametrize("secret", SECRETS)
def test_a_secret_in_her_reply_is_not_stored_verbatim(repos, secret):
    """The response column too. She echoes what she was given more often than
    not -- a setup flow reads a value back to confirm it."""
    db, memory, _ = repos
    memory.save_turn("save this", "small_talk", f"stored {secret}", "s2")

    stored = db.fetchone(
        "SELECT response FROM conversations WHERE session_id = 's2'"
    )["response"]
    assert secret not in stored, f"secret stored verbatim in response: {stored!r}"


@pytest.mark.parametrize("secret", SECRETS)
def test_the_search_index_does_not_keep_a_copy(repos, secret):
    """`conversations_fts` is filled by an INSERT trigger on `conversations`,
    so redacting the column is what keeps the index clean. If the redaction
    ever moves to a read path, this is the assertion that catches the copy
    left behind in the index."""
    db, memory, _ = repos
    memory.save_turn(f"token: {secret}", "small_talk", "ok", "s3")

    hits = db.fetchall("SELECT user_input FROM conversations_fts")
    assert hits, "the FTS trigger stored nothing -- this test would pass vacuously"
    for h in hits:
        assert secret not in (h["user_input"] or ""), (
            "the secret survives in conversations_fts even though the base "
            "table was redacted"
        )


# ─── facts, telemetry, recordings ────────────────────────────────────────────

@pytest.mark.parametrize("secret", SECRETS)
def test_a_secret_is_not_stored_as_a_fact(repos, secret):
    db, memory, _ = repos
    memory.save_typed_fact("user_note", secret, "conversation", "fact")

    stored = db.fetchone("SELECT value FROM facts WHERE key = 'user_note'")["value"]
    assert secret not in stored, f"secret stored verbatim in facts.value: {stored!r}"


@pytest.mark.parametrize("secret", SECRETS)
def test_a_secret_is_not_stored_in_telemetry(repos, secret):
    """The column the routing corpus was extracted from, and the one that
    actually held the three real credentials found on 2026-08-22."""
    db, _, telemetry = repos
    telemetry.create(
        session_id="s4", timestamp="2026-08-22T00:00:00", input_modality="text",
        transcript=f"paste: {secret}", intent_detected="unknown",
        intent_source="llm", action_dispatched=None, action_outcome="skipped",
        error_class=None, latency_total_ms=1, latency_stt_ms=None,
        latency_intent_ms=1, latency_action_ms=None, latency_tts_ms=None,
        llm_calls_count=0, llm_tokens_in=0, llm_tokens_out=0,
        fallback_chain_depth=0, vision_calls_count=0,
    )

    stored = db.fetchone(
        "SELECT transcript FROM interaction_events WHERE session_id = 's4'"
    )["transcript"]
    assert secret not in stored, (
        f"secret stored verbatim in interaction_events.transcript: {stored!r}"
    )


@pytest.mark.parametrize("secret", SECRETS)
def test_a_secret_spoken_into_a_recording_is_not_stored(repos, secret):
    db, memory, _ = repos
    memory.save_chunk("rec1", 0, f"and the key is {secret}")

    stored = db.fetchone(
        "SELECT transcript FROM recording_sessions WHERE session_id = 'rec1'"
    )["transcript"]
    assert secret not in stored, f"secret stored verbatim in a recording: {stored!r}"


def test_a_null_transcript_does_not_crash_the_write(repos):
    """`transcript` is nullable and the redactor takes a str. A turn with no
    transcript must still record its telemetry rather than raising."""
    db, _, telemetry = repos
    telemetry.create(
        session_id="s5", timestamp="2026-08-22T00:00:00", input_modality="text",
        transcript=None, intent_detected=None, intent_source=None,
        action_dispatched=None, action_outcome="skipped", error_class=None,
        latency_total_ms=None, latency_stt_ms=None, latency_intent_ms=None,
        latency_action_ms=None, latency_tts_ms=None, llm_calls_count=0,
        llm_tokens_in=0, llm_tokens_out=0, fallback_chain_depth=0,
        vision_calls_count=0,
    )
    assert db.fetchone(
        "SELECT id FROM interaction_events WHERE session_id = 's5'"
    ) is not None


# ─── the other half: nothing innocent is destroyed ───────────────────────────

@pytest.mark.parametrize("text", INNOCENT)
def test_ordinary_conversation_is_stored_unchanged(repos, text):
    """Half of this file's value. A redactor that blanked every turn would pass
    every test above and quietly delete her memory, so the permitted case is
    pinned as tightly as the refused one -- live-test the answer, not the
    refusal."""
    db, memory, _ = repos
    memory.save_turn(text, "small_talk", "sure", "clean")

    stored = db.fetchone(
        "SELECT user_input FROM conversations WHERE session_id = 'clean'"
    )["user_input"]
    assert stored == text, (
        f"ordinary text was altered on the way into storage:\n"
        f"  in:  {text!r}\n  out: {stored!r}\n"
        "Over-redaction here is unrecoverable -- it deletes what the user said."
    )


@pytest.mark.parametrize("text", INNOCENT)
def test_ordinary_text_survives_as_a_fact(repos, text):
    db, memory, _ = repos
    memory.save_typed_fact("k", text, "conversation", "fact")
    stored = db.fetchone("SELECT value FROM facts WHERE key = 'k'")["value"]
    assert stored == text, f"fact value altered: {text!r} -> {stored!r}"
