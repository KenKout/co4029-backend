"""Narration audio cache behaviour.

The onboarding transition line is a fixed string, identical for every candidate
in a language, yet each session paid a full ~3.0-3.6s Deepgram round trip for
it — and the browser holds text AND voice back for that whole window, which is
the delay reported at the head of that line.

These tests pin the properties that make caching it safe: identity is the full
(text, voice, persona, language) tuple, so no session can ever be served another
config's voice, and the cache is bounded so a long-lived worker cannot grow
without limit.
"""

from __future__ import annotations

import pytest

from abridgeai.features.interviews.services import narration_cache

TRANSITION = "Great—the introduction is complete. Let's begin. Here is your first question."


@pytest.fixture(autouse=True)
def _clear_cache():
    narration_cache.clear()
    yield
    narration_cache.clear()


def test_miss_returns_none():
    assert (
        narration_cache.get(
            text=TRANSITION, voice="aura-2-thalia-en", persona="neutral", language="en"
        )
        is None
    )


def test_round_trips_the_same_request():
    """The whole point: the fixed ceremony line is synthesized once."""
    narration_cache.put(
        text=TRANSITION,
        voice="aura-2-thalia-en",
        persona="neutral",
        language="en",
        audio=b"mp3-bytes",
    )
    assert (
        narration_cache.get(
            text=TRANSITION, voice="aura-2-thalia-en", persona="neutral", language="en"
        )
        == b"mp3-bytes"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("text", "A different utterance entirely."),
        ("voice", "aura-2-orpheus-en"),
        ("persona", "strict"),
        ("language", "vi"),
    ],
)
def test_every_field_is_part_of_the_identity(field: str, value: str):
    """A different voice/persona/language/text must never hit another's audio.

    This is the property that keeps the cache from becoming a cross-config
    audio leak: two interviews configured with different Aura voices must
    never hear each other's synthesis.
    """
    base = {
        "text": TRANSITION,
        "voice": "aura-2-thalia-en",
        "persona": "neutral",
        "language": "en",
    }
    narration_cache.put(**base, audio=b"original")

    lookup = {**base, field: value}
    assert narration_cache.get(**lookup) is None


def test_absent_voice_and_persona_are_distinct_from_present_ones():
    """A config with no explicit voice must not collide with one that has it."""
    narration_cache.put(
        text=TRANSITION, voice=None, persona=None, language="en", audio=b"default-voice"
    )
    assert (
        narration_cache.get(
            text=TRANSITION, voice="aura-2-thalia-en", persona="neutral", language="en"
        )
        is None
    )
    assert (
        narration_cache.get(text=TRANSITION, voice=None, persona=None, language="en")
        == b"default-voice"
    )


def test_language_matching_is_case_insensitive():
    narration_cache.put(text=TRANSITION, voice=None, persona=None, language="EN", audio=b"audio")
    assert narration_cache.get(text=TRANSITION, voice=None, persona=None, language="en") == b"audio"


def test_eviction_keeps_the_cache_bounded():
    """A long-lived worker must not grow without limit."""
    for index in range(narration_cache.MAX_ENTRIES + 25):
        narration_cache.put(
            text=f"utterance {index}",
            voice=None,
            persona=None,
            language="en",
            audio=b"a",
        )
    assert narration_cache.size() == narration_cache.MAX_ENTRIES


def test_eviction_is_least_recently_used():
    """A repeatedly-read entry (the ceremony line) must survive the churn."""
    narration_cache.put(text=TRANSITION, voice=None, persona=None, language="en", audio=b"ceremony")
    for index in range(narration_cache.MAX_ENTRIES - 1):
        narration_cache.put(
            text=f"question {index}", voice=None, persona=None, language="en", audio=b"q"
        )
        # Keep reading the ceremony line, as real traffic would.
        narration_cache.get(text=TRANSITION, voice=None, persona=None, language="en")

    # Now overflow: the cold question entries must go before the hot one.
    for index in range(50):
        narration_cache.put(
            text=f"later {index}", voice=None, persona=None, language="en", audio=b"l"
        )
    assert (
        narration_cache.get(text=TRANSITION, voice=None, persona=None, language="en") == b"ceremony"
    )


def test_oversized_entries_are_not_cached():
    """One runaway payload must not evict a cache full of useful small ones."""
    narration_cache.put(
        text=TRANSITION,
        voice=None,
        persona=None,
        language="en",
        audio=b"x" * (narration_cache.MAX_ENTRY_BYTES + 1),
    )
    assert narration_cache.get(text=TRANSITION, voice=None, persona=None, language="en") is None


def test_empty_audio_is_not_cached():
    """A failed synthesis must never be remembered as the answer."""
    narration_cache.put(text=TRANSITION, voice=None, persona=None, language="en", audio=b"")
    assert narration_cache.get(text=TRANSITION, voice=None, persona=None, language="en") is None
