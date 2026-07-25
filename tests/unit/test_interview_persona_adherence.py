"""Unit tests for the persona-adherence diagnostic stage.

This stage audits whether the AI interviewer HELD its configured persona, over a
stored transcript, offline. It is a tone-only diagnostic — it must never gate
pass/fail — so the tests focus on: the parser degrades safely on bad LLM output,
only interviewer turns are judged, the closed violation vocabulary is enforced,
and the whole thing never raises (a broken judge cannot break evaluation).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from abridgeai.features.interviews.ai.stages.persona_adherence.logic import (
    audit_persona_adherence,
)
from abridgeai.features.interviews.ai.stages.persona_adherence.parsers import (
    parse_persona_adherence,
    unavailable,
)
from abridgeai.features.interviews.orchestrator.persona import PRESETS


def _msg(role: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(role=role, content_text=text)


def _gateway_returning(payload: object) -> SimpleNamespace:
    generate = AsyncMock(return_value=SimpleNamespace(content_json=payload))
    return SimpleNamespace(generate_json=generate)


# ── Parser: happy path ───────────────────────────────────────────────────────


def test_parser_reads_full_payload() -> None:
    payload = {
        "tone_consistency": {"reasoning": "Said 'Good.' curtly throughout.", "score": 8},
        "warmth_observed": 1,
        "directness_observed": 4,
        "verbosity_observed": 1,
        "formality_observed": 4,
        "drift_turns": [3, 7],
        "violations": ["over_polite_for_strict"],
    }
    result = parse_persona_adherence(payload)
    assert result.available is True
    assert result.tone_consistency == 8
    assert "curtly" in result.reasoning
    assert result.warmth_observed == 1
    assert result.drift_turns == [3, 7]
    assert result.violations == ["over_polite_for_strict"]


def test_parser_tolerates_flattened_tone_shape() -> None:
    result = parse_persona_adherence({"tone_consistency": 6, "reasoning": "mixed tone"})
    assert result.tone_consistency == 6
    assert result.reasoning == "mixed tone"


# ── Parser: safe degradation ─────────────────────────────────────────────────


def test_non_mapping_payload_is_unavailable() -> None:
    assert parse_persona_adherence(None).available is False
    assert parse_persona_adherence("garbage").available is False  # type: ignore[arg-type]
    assert parse_persona_adherence([1, 2]).available is False  # type: ignore[arg-type]


def test_scores_are_clamped_to_range() -> None:
    result = parse_persona_adherence(
        {
            "tone_consistency": {"score": 99, "reasoning": "x"},
            "warmth_observed": -4,
            "directness_observed": 40,
        }
    )
    assert result.tone_consistency == 10  # clamped 0-10
    assert result.warmth_observed == 0  # clamped 0-4
    assert result.directness_observed == 4


def test_unknown_violation_tags_are_dropped() -> None:
    result = parse_persona_adherence(
        {
            "tone_consistency": {"score": 5, "reasoning": "x"},
            "violations": ["declared_answer", "made_up_tag", "cold_for_supportive"],
        }
    )
    assert result.violations == ["declared_answer", "cold_for_supportive"]


def test_drift_turns_dedupe_and_reject_junk() -> None:
    result = parse_persona_adherence(
        {
            "tone_consistency": {"score": 5, "reasoning": "x"},
            "drift_turns": [3, 3, "nope", -1, 7, True],
        }
    )
    assert result.drift_turns == [3, 7]


def test_missing_fields_default_safely() -> None:
    result = parse_persona_adherence({"tone_consistency": {"score": 7}})
    assert result.available is True
    assert result.reasoning == ""
    assert result.warmth_observed == 0
    assert result.drift_turns == []
    assert result.violations == []


# ── Logic: only interviewer turns are judged ─────────────────────────────────


@pytest.mark.asyncio
async def test_only_ai_turns_are_sent_to_the_judge() -> None:
    gateway = _gateway_returning({"tone_consistency": {"score": 7, "reasoning": "ok"}})
    messages = [
        _msg("ai", "Welcome. First question: explain idempotency."),
        _msg("user", "It means repeated calls have the same effect."),
        _msg("ai", "Good. Now, how would you make a POST idempotent?"),
        _msg("system", "session started"),
    ]

    result = await audit_persona_adherence(
        AsyncMock(),
        persona=PRESETS["strict"],
        messages=messages,
        gateway=gateway,
    )

    assert result.available is True
    # Inspect what actually got sent to the model.
    import json

    sent = json.loads(gateway.generate_json.await_args.kwargs["user_prompt"])
    turns = sent["interviewer_turns"]
    assert len(turns) == 2  # only the two ai turns
    assert all("idempotenc" in t["text"].lower() or "POST" in t["text"] for t in turns)
    assert [t["turn"] for t in turns] == [1, 2]  # renumbered 1-based
    # The declared persona traits ride along, tone-only.
    assert sent["declared_persona"]["key"] == "strict"
    assert "difficulty" not in sent["declared_persona"]


@pytest.mark.asyncio
async def test_no_interviewer_turns_is_unavailable_without_calling_llm() -> None:
    gateway = _gateway_returning({"tone_consistency": {"score": 7, "reasoning": "x"}})
    messages = [_msg("user", "hello"), _msg("system", "note")]

    result = await audit_persona_adherence(
        AsyncMock(),
        persona=PRESETS["neutral"],
        messages=messages,
        gateway=gateway,
    )

    assert result.available is False
    gateway.generate_json.assert_not_awaited()


@pytest.mark.asyncio
async def test_llm_failure_degrades_to_unavailable() -> None:
    gateway = SimpleNamespace(generate_json=AsyncMock(side_effect=RuntimeError("gateway down")))
    messages = [_msg("ai", "First question.")]

    result = await audit_persona_adherence(
        AsyncMock(),
        persona=PRESETS["supportive"],
        messages=messages,
        gateway=gateway,
    )

    assert result.available is False  # never raises


def test_unavailable_sentinel_shape() -> None:
    result = unavailable()
    assert result.available is False
    assert result.tone_consistency == 0
    assert result.to_json()["available"] is False
