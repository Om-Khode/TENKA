"""A stated cadence is not the model's to reinterpret.

Two schedules created four minutes apart in the same session, both phrased
"every minute":

    'schedule a web search for today's tech news every minute'
      -> cron_expr '* * * * *'   fired at 20:19, 20:20, 20:21   correct
    'schedule the scratchpad procedure every minute'
      -> cron_expr '0 * * * *'   next_fire_at 21:00              hourly

Both answered "Got it, I scheduled ...". Nothing in the parse prompt pinned the
mapping, so the same words landed on two different cadences and the operator was
told the same thing either way. Ask for a five-minute check, get an hourly one,
and find out when it matters.

Not a model failure to route around -- an unspecified mapping. So both halves:
the prompt now states the minute rules, and an explicitly stated interval is
resolved deterministically and **overrides** whatever came back.
`.claude/rules/llm-and-intents.md` asks for exactly that -- skip the LLM
wherever a deterministic path exists -- and an interval written in the sentence
is not a judgement call.

Run with:  py -3.11 -m pytest tests/test_schedule_cadence.py -v
"""
import pathlib
import sys

import pytest
from croniter import croniter

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.actions.schedule import _explicit_minute_cron  # noqa: E402


# ─── what it claims ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,cron", [
    ("schedule the scratchpad procedure every minute", "* * * * *"),
    ("every 1 minute", "* * * * *"),
    ("every minute or so", "* * * * *"),
    ("every 5 minutes", "*/5 * * * *"),
    ("check the site every 15 mins", "*/15 * * * *"),
    ("remind me every 2 minutes", "*/2 * * * *"),
    ("every 59 minutes", "*/59 * * * *"),
])
def test_a_stated_minute_cadence_is_resolved_without_the_model(text, cron):
    assert _explicit_minute_cron(text) == cron


@pytest.mark.parametrize("text", [
    "every hour",
    "every morning",
    "every day at 9am",
    "every Monday",
    "every few minutes",
    "every couple of minutes",
    "search for news",
    "",
])
def test_anything_not_plainly_stated_is_left_to_the_model(text):
    """**The half that keeps this honest.** A matcher that claimed more would
    quietly become a second, worse schedule parser -- and "every few minutes" is
    a judgement, which is what the model is for. The prompt covers those."""
    assert _explicit_minute_cron(text) is None


@pytest.mark.parametrize("text", ["every 0 minutes", "every 60 minutes",
                                  "every 90 minutes", "every 1440 minutes"])
def test_out_of_range_intervals_are_declined_not_clamped(text):
    """`*/60` is not a valid minute field, and clamping a stated number to
    something else is the exact failure this file is about. Declining hands it
    back to the model, which can express it in hours."""
    assert _explicit_minute_cron(text) is None


# ─── the crons are real ──────────────────────────────────────────────────────

@pytest.mark.parametrize("text", ["every minute", "every 5 minutes",
                                  "every 15 mins", "every 59 minutes"])
def test_every_cron_it_produces_is_valid(text):
    """`_create_schedule` calls `croniter.is_valid` and refuses on failure, so
    an invalid expression here would not be a wrong cadence -- it would be
    "Sorry, I couldn't figure out the timing", which is worse than hourly."""
    cron = _explicit_minute_cron(text)
    assert cron is not None
    assert croniter.is_valid(cron), cron


def test_a_minute_cron_actually_fires_within_a_minute():
    """The claim the whole fix rests on. The scheduler polls every 30s and fires
    when `next_fire_at` has passed, so a one-minute cron is reachable -- it was
    never a question of whether the *scheduler* could do it."""
    from datetime import datetime, timedelta

    start = datetime(2026, 8, 23, 20, 47, 20)
    nxt = croniter("* * * * *", start).get_next(datetime)
    assert nxt - start <= timedelta(minutes=1), nxt


def test_the_reported_hourly_cron_is_not_within_a_minute():
    """The other side of the same arithmetic, so the diagnosis is pinned rather
    than remembered: `0 * * * *` from a 20:47 creation is next due at 21:00."""
    from datetime import datetime, timedelta

    start = datetime(2026, 8, 23, 20, 47, 20)
    nxt = croniter("0 * * * *", start).get_next(datetime)
    assert nxt == datetime(2026, 8, 23, 21, 0)
    assert nxt - start > timedelta(minutes=1)


# ─── the override is wired, and it overrides ─────────────────────────────────

@pytest.mark.asyncio
async def test_the_stated_cadence_beats_the_parsed_one(monkeypatch):
    """The observed failure, reproduced: the model answers hourly for a request
    that says "every minute". An *override*, not a fallback -- a fallback only
    helps when the parse fails, and this parse succeeded and was wrong."""
    created = {}

    async def _fake_parse(goal):
        return {"name": "run scratchpad procedure", "cron_expr": "0 * * * *",
                "task_type": "procedure", "goal": "scratchpad",
                "notify_mode": "on_match_only", "condition_text": None}

    class _Repo:
        def create(self, **kw):
            created.update(kw)

    import assistant.llm.contracts as contracts
    import assistant.actions.schedule as sched
    monkeypatch.setattr(contracts, "ask_for_schedule_parse", _fake_parse)
    monkeypatch.setattr(sched, "_get_repo", lambda: _Repo())

    out = await sched._create_schedule("schedule the scratchpad procedure every minute")

    assert "scheduled" in out.lower()
    assert created["cron_expr"] == "* * * * *", (
        f"the parsed hourly cron survived a request that said every minute: "
        f"{created['cron_expr']}"
    )
    assert created["task_type"] == "procedure", "the rest of the parse was lost"


@pytest.mark.asyncio
async def test_a_parse_with_no_stated_cadence_is_left_alone(monkeypatch):
    """**The direction an override gets wrong.** Everything the model decides
    that the sentence does not state must pass through untouched, or this
    becomes a parser that ignores its own parser."""
    created = {}

    async def _fake_parse(goal):
        return {"name": "morning news", "cron_expr": "0 9 * * *",
                "task_type": "web_search", "goal": "tech news",
                "notify_mode": "always", "condition_text": None}

    class _Repo:
        def create(self, **kw):
            created.update(kw)

    import assistant.llm.contracts as contracts
    import assistant.actions.schedule as sched
    monkeypatch.setattr(contracts, "ask_for_schedule_parse", _fake_parse)
    monkeypatch.setattr(sched, "_get_repo", lambda: _Repo())

    await sched._create_schedule("schedule a news search every morning")
    assert created["cron_expr"] == "0 9 * * *", "the model's answer was overridden"


@pytest.mark.asyncio
async def test_a_failed_parse_still_fails(monkeypatch):
    """The override must not paper over a parse that returned nothing -- the
    name and task type come from it, and inventing them would create a schedule
    nobody asked for."""
    async def _fake_parse(goal):
        return None

    import assistant.llm.contracts as contracts
    import assistant.actions.schedule as sched
    monkeypatch.setattr(contracts, "ask_for_schedule_parse", _fake_parse)

    out = await sched._create_schedule("schedule something every minute")
    assert "couldn't understand" in out.lower()


# ─── the prompt states the rules too ─────────────────────────────────────────

def test_the_parse_prompt_states_the_minute_rules():
    """Both halves, because the override is deliberately narrow. "every couple
    of minutes" is a judgement the model should make well, and it can only do
    that if the mapping is written down -- an unwritten mapping is what produced
    two different answers to one phrasing."""
    from assistant.llm.contracts import _SCHEDULE_PARSE_PROMPT

    assert '"every minute" = "* * * * *"' in _SCHEDULE_PARSE_PROMPT
    assert '*/N * * * *' in _SCHEDULE_PARSE_PROMPT
    assert 'every couple of minutes' in _SCHEDULE_PARSE_PROMPT
