"""Tests for identity reaching the LLM layers (the wiring, not the presets).

Before this, ``as_prompt_identity`` had no production caller: identity only ever
changed the deterministic ceremony sentence, so ``register_en`` / ``register_vi``
— the only field that can make a tech lead word things differently from a staff
engineer — was dead. These tests pin the two wires that fix it and, just as
importantly, pin what must NOT change:

* the phrasing prompt receives ``interviewer`` only for a NAMED identity, so an
  opted-out config's prompt payload stays byte-identical to the pre-wiring one;
* the tone judge receives the same declared identity, so a role's register is
  not mistaken for tone drift;
* neither payload carries anything decision-bearing.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from abridgeai.features.interviews.ai.stages.persona_adherence.logic import (
    audit_persona_adherence,
)
from abridgeai.features.interviews.orchestrator import utterance_logic
from abridgeai.features.interviews.orchestrator.decision import (
    InterviewerActionType,
    InterviewerDecision,
    ReasonCode,
)
from abridgeai.features.interviews.orchestrator.interviewer_identity import (
    InterviewerRole,
    identity_from,
)
from abridgeai.features.interviews.orchestrator.persona import PRESETS
from abridgeai.features.interviews.orchestrator.utterance import Persona

_FORBIDDEN_SUBSTRINGS = ("difficulty", "score", "weight", "outcome", "pass", "rubric")


class _DB:
    """A DB whose ``begin_nested()`` is an async-context no-op."""

    def begin_nested(self) -> Any:
        class _Ctx:
            async def __aenter__(self) -> None:
                return None

            async def __aexit__(self, *a: Any) -> None:
                return None

        return _Ctx()


def _gateway(question_or_probe: str) -> SimpleNamespace:
    """Returns a rewrite that preserves the question so the LLM path is accepted."""
    return SimpleNamespace(
        generate_json=AsyncMock(
            return_value=SimpleNamespace(
                content_json={
                    "acknowledgement": "Thanks.",
                    "transition": "",
                    "ai_turn_text": f"Thanks. {question_or_probe}",
                }
            )
        )
    )


async def _sent_payload(identity: object, language: str = "en") -> dict[str, Any]:
    """Run generate_utterance and return the JSON actually handed to the model."""
    decision = InterviewerDecision(
        action=InterviewerActionType.ASK_FOR_EXAMPLE,
        reason_code=ReasonCode.MISSING_EXAMPLE,
    )
    fallback = utterance_logic.build_fallback_utterance(
        decision, persona=Persona.NEUTRAL, language=language, question_text="Explain indexes."
    )
    gateway = _gateway(fallback.question_or_probe)

    _, status = await utterance_logic.generate_utterance(
        _DB(),  # type: ignore[arg-type]
        decision,
        persona=Persona.NEUTRAL,
        language=language,
        question_text="Explain indexes.",
        identity=identity,
        use_llm=True,
        gateway=gateway,  # type: ignore[arg-type]
    )
    assert status == "llm", "prompt failed to render / rewrite rejected"
    return json.loads(gateway.generate_json.await_args.kwargs["user_prompt"])


# ── Phrasing layer ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_named_identity_reaches_the_phrasing_prompt() -> None:
    """The register — the whole point of the role — must arrive at the model."""
    sent = await _sent_payload(identity_from(InterviewerRole.BACKEND_TECH_LEAD.value))
    assert sent["interviewer"] == {
        "role": "backend_tech_lead",
        "name": "Minh",
        "title": "backend tech lead",
        "register": "practical and implementation-minded",
    }


@pytest.mark.asyncio
async def test_identity_is_language_resolved_for_the_prompt() -> None:
    """The model is handed ONE language, never a translation choice."""
    sent = await _sent_payload(identity_from("staff_engineer"), language="vi")
    assert sent["interviewer"]["title"] == "staff engineer"
    assert sent["interviewer"]["register"] == "chính xác, quan tâm tới lập luận và đánh đổi"


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", [None, identity_from(None), "not-an-identity", 42])
async def test_unnamed_or_bogus_identity_omits_the_key_entirely(identity: object) -> None:
    """An opted-out config's prompt must be identical to the pre-wiring payload.

    Omitting the key (rather than sending an empty dict) is what makes that
    true — a present-but-blank field would still change the model's input.
    """
    sent = await _sent_payload(identity)
    assert "interviewer" not in sent


@pytest.mark.asyncio
async def test_two_roles_differ_only_in_the_interviewer_block() -> None:
    """Identity must not perturb the tone traits or the approved parts."""
    lead = await _sent_payload(identity_from("backend_tech_lead"))
    staff = await _sent_payload(identity_from("staff_engineer"))
    assert lead["interviewer"] != staff["interviewer"]
    for key in ("persona", "persona_traits", "language", "action", "approved_parts"):
        assert lead[key] == staff[key]


@pytest.mark.asyncio
async def test_phrasing_payload_carries_nothing_decision_bearing() -> None:
    """Mirrors the persona invariant: identity shapes language only."""
    sent = await _sent_payload(identity_from("eng_manager"))
    blob = json.dumps(sent["interviewer"]).lower()
    for bad in _FORBIDDEN_SUBSTRINGS:
        assert bad not in blob


# ── Tone judge ───────────────────────────────────────────────────────────────


def _judge_gateway() -> SimpleNamespace:
    return SimpleNamespace(
        generate_json=AsyncMock(
            return_value=SimpleNamespace(
                content_json={"tone_consistency": {"score": 8, "reasoning": "ok"}}
            )
        )
    )


def _ai_turn(text: str) -> SimpleNamespace:
    return SimpleNamespace(role="ai", content_text=text)


@pytest.mark.asyncio
async def test_judge_is_told_which_interviewer_was_declared() -> None:
    """Without this the judge scores a role's register as tone drift."""
    gateway = _judge_gateway()
    result = await audit_persona_adherence(
        AsyncMock(),
        persona=PRESETS["neutral"],
        messages=[_ai_turn("Walk me through how you'd index that table.")],  # type: ignore[list-item]
        identity=identity_from("backend_tech_lead"),
        language="en",
        gateway=gateway,  # type: ignore[arg-type]
    )
    assert result.available is True
    sent = json.loads(gateway.generate_json.await_args.kwargs["user_prompt"])
    assert sent["declared_interviewer"]["register"] == "practical and implementation-minded"
    # The traits still ride along unchanged — identity is additive.
    assert sent["declared_persona"]["key"] == "neutral"


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", [None, identity_from(None)])
async def test_judge_payload_unchanged_when_no_identity_declared(identity: object) -> None:
    gateway = _judge_gateway()
    await audit_persona_adherence(
        AsyncMock(),
        persona=PRESETS["strict"],
        messages=[_ai_turn("Next question.")],  # type: ignore[list-item]
        identity=identity,  # type: ignore[arg-type]
        gateway=gateway,  # type: ignore[arg-type]
    )
    sent = json.loads(gateway.generate_json.await_args.kwargs["user_prompt"])
    assert set(sent) == {"declared_persona", "interviewer_turns"}


@pytest.mark.asyncio
async def test_judge_still_never_raises_with_identity_wired() -> None:
    """The diagnostic must stay unable to break a student's evaluation."""
    gateway = SimpleNamespace(generate_json=AsyncMock(side_effect=RuntimeError("down")))
    result = await audit_persona_adherence(
        AsyncMock(),
        persona=PRESETS["supportive"],
        messages=[_ai_turn("First question.")],  # type: ignore[list-item]
        identity=identity_from("hr_screener"),
        gateway=gateway,  # type: ignore[arg-type]
    )
    assert result.available is False


# ── The failure mode that matters: a bad identity must not lose the turn ─────


@pytest.mark.asyncio
async def test_phrasing_falls_back_rather_than_raising_on_identity_trouble() -> None:
    """If the interviewer block somehow breaks the call, the candidate still gets a turn."""
    decision = InterviewerDecision(
        action=InterviewerActionType.ASK_FOR_EXAMPLE,
        reason_code=ReasonCode.MISSING_EXAMPLE,
    )
    gateway = SimpleNamespace(generate_json=AsyncMock(side_effect=RuntimeError("boom")))
    utterance, status = await utterance_logic.generate_utterance(
        _DB(),  # type: ignore[arg-type]
        decision,
        persona=Persona.NEUTRAL,
        language="en",
        question_text="Explain indexes.",
        identity=identity_from("backend_tech_lead"),
        use_llm=True,
        gateway=gateway,  # type: ignore[arg-type]
    )
    assert status == "fallback"
    assert "Explain indexes." in utterance.ai_turn_text
