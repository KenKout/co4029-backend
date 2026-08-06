"""The room intro must not become a third introduction.

A hybrid session runs the whole REST onboarding ceremony before the agent ever
joins: the candidate is greeted by name ("I'm Ha, an engineering manager..."),
briefed, and finally handed off with the ready_transition line ("...the
introduction is complete. Let's begin. Here is your first question.").

The agent then joined and said "Ha here. Let's get started." on top of that --
a third introduction the candidate had no reason to expect. Worse, `on_enter`
speaks it BEFORE question one and awaits its playout, so it also delayed the
question's own audio while the on-screen text had already been released.

A pure voice session has none of that ceremony, so its room intro must survive
untouched -- that intro is the whole reason the feature exists (without it a
voice candidate's first audio is a raw bank question with nobody introducing
it).
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from abridgeai.features.interviews.realtime.orchestration_bridge import (
    _ceremony_already_introduced,
)


class _Result:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def scalars(self) -> _Result:
        return self

    def all(self) -> list[object]:
        return self._rows


class _FakeDb:
    """Minimal async DB double, matching the shape the bridge actually uses."""

    def __init__(self, rows: list[object]) -> None:
        self._rows = rows
        self.calls = 0

    async def execute(self, _statement: object) -> _Result:
        self.calls += 1
        return _Result(self._rows)


class _ExplodingDb:
    async def execute(self, _statement: object) -> _Result:
        raise RuntimeError("connection lost")


def _msg(ceremony_key: object) -> SimpleNamespace:
    return SimpleNamespace(metadata_json={"ceremony_key": ceremony_key})


@pytest.mark.asyncio
async def test_hybrid_session_that_finished_onboarding_is_already_introduced() -> None:
    """ready_transition present => the candidate was greeted and handed off."""
    db = _FakeDb([_msg("candidate_confirmation"), _msg("briefing"), _msg("ready_transition")])
    assert await _ceremony_already_introduced(db, uuid4()) is True


@pytest.mark.asyncio
async def test_pure_voice_session_still_gets_its_room_intro() -> None:
    """No REST ceremony ran, so the intro is the only introduction there is."""
    db = _FakeDb([])
    assert await _ceremony_already_introduced(db, uuid4()) is False


@pytest.mark.asyncio
async def test_partial_onboarding_does_not_count_as_introduced() -> None:
    """Greeted but never handed off: the transition line is the marker.

    Only ready_transition is written once onboarding actually completed, so an
    interrupted setup must not silence the intro.
    """
    db = _FakeDb([_msg("candidate_confirmation"), _msg("briefing")])
    assert await _ceremony_already_introduced(db, uuid4()) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [None, {}, "not-a-dict", 42, {"ceremony_key": None}])
async def test_malformed_metadata_never_raises(metadata: object) -> None:
    """A junk metadata blob must not blow up the agent's room join."""
    db = _FakeDb([SimpleNamespace(metadata_json=metadata)])
    assert await _ceremony_already_introduced(db, uuid4()) is False


@pytest.mark.asyncio
async def test_lookup_failure_keeps_the_intro() -> None:
    """Degrade toward the PREVIOUS behaviour, not toward silence.

    A DB hiccup must not be the reason a voice candidate hears nobody.
    """
    assert await _ceremony_already_introduced(_ExplodingDb(), uuid4()) is False
