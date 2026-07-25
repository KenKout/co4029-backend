"""Unit tests for the voice-harness synthetic candidate personas (Phase 5).

These are pure/offline: they exercise the profile catalogue, the per-profile
static fallback (used when no LLM gateway is available), and the prompt
builders. The live LLM generation path is not called here — it is best-effort
and falls back to these same static answers, which is what we assert stays
distinct and in-character per profile.
"""

from __future__ import annotations

import asyncio

import pytest

from scripts.voice_harness import candidate


_PROFILE_KEYS = ("confident_strong", "nervous_mid", "rambling_mid", "stuck_weak")


def test_exactly_the_four_plan_profiles_exist() -> None:
    assert set(candidate.PROFILES.keys()) == set(_PROFILE_KEYS)
    assert candidate.profile_names() == list(_PROFILE_KEYS)


@pytest.mark.parametrize("key", _PROFILE_KEYS)
def test_each_profile_has_prose_ability_and_state(key: str) -> None:
    p = candidate.PROFILES[key]
    # Ability/state are prose (the whole point — a bare label produces bland
    # text that never trips affect detection), so assert they are non-trivial.
    assert len(p.ability) > 10
    assert len(p.state) > 10
    assert len(p.style_hint) > 10
    assert p.key == key


@pytest.mark.parametrize("key", _PROFILE_KEYS)
@pytest.mark.parametrize("lang", ["en", "vi"])
def test_fallback_answer_is_nonempty_and_localized(key: str, lang: str) -> None:
    p = candidate.PROFILES[key]
    answer = candidate._fallback_answer(p, lang)
    assert isinstance(answer, str)
    assert answer.strip() != ""


def test_fallback_answers_are_distinct_across_profiles() -> None:
    """The four profiles must produce four DIFFERENT static answers — otherwise
    the harness isn't exercising different candidate behaviours at all."""
    en_answers = {
        candidate._fallback_answer(candidate.PROFILES[k], "en") for k in _PROFILE_KEYS
    }
    assert len(en_answers) == len(_PROFILE_KEYS)


def test_stuck_profile_asks_for_help() -> None:
    """The stuck_weak fallback should read as someone stuck (asks for a hint),
    which is what exercises the hint ladder downstream."""
    answer = candidate._fallback_answer(candidate.PROFILES["stuck_weak"], "en").lower()
    assert "hint" in answer or "not" in answer  # "not sure" / asks for a hint


def test_nervous_profile_hedges() -> None:
    answer = candidate._fallback_answer(candidate.PROFILES["nervous_mid"], "en").lower()
    assert "i think" in answer or "not" in answer or "sure" in answer


def test_user_prompt_carries_profile_and_question_no_scoring_fields() -> None:
    import json

    p = candidate.PROFILES["confident_strong"]
    raw = candidate._user_prompt(p, "What is a fact table?", "en")
    payload = json.loads(raw)
    # Question + character reach the model.
    assert payload["interviewer_just_said"] == "What is a fact table?"
    assert payload["your_ability"] == p.ability
    assert payload["your_state"] == p.state
    # Nothing scoring-related is ever handed to the candidate generator.
    for banned in ("rubric", "score", "expected_answer", "model_answer", "outcome"):
        assert banned not in payload


def test_vietnamese_language_switches_prompt_language_name() -> None:
    import json

    p = candidate.PROFILES["confident_strong"]
    en = json.loads(candidate._user_prompt(p, "q", "en"))
    vi = json.loads(candidate._user_prompt(p, "q", "vi"))
    assert en["language"] == "English"
    assert vi["language"] == "Vietnamese"


def test_generate_answer_falls_back_when_gateway_unavailable(monkeypatch) -> None:
    """With no usable gateway, generate_answer must still return the profile's
    static answer rather than raising — the run is never blocked."""

    class _BoomGateway:
        def __init__(self, *a, **k):
            raise RuntimeError("no creds")

    import abridgeai.ai.llm as llm_mod

    monkeypatch.setattr(llm_mod, "LLMGateway", _BoomGateway)
    p = candidate.PROFILES["nervous_mid"]
    answer = asyncio.run(
        candidate.generate_answer(p, question="What is a fact table?", language="en", settings=object())
    )
    assert answer == candidate._fallback_answer(p, "en")
