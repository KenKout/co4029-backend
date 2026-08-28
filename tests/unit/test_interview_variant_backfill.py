"""Unit tests for the variant-mode backfill contract (Slice 21 regression).

Regression: in ``all_angles`` mode the backfill loop passed the raw
*shortfall* as ``override_question_count``, but the generation stage treats
``override_question_count`` as a TOTAL row budget and re-divides it by the
angle count — so a shortfall of 2 rows became ceil(2/4)=1 logical question
whose variant set may not include the missing angle (observed: bank ended
3/3/3/1 instead of 2/2/2/2). The fix: request in whole angle groups and trim
the overshoot back to the target.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.features.interviews.ai.pipelines.backfill import (
    _trim_to_shortfall,
    generate_with_backfill,
)
from abridgeai.features.interviews.ai.stages.generation import (
    InterviewQuestionDraft,
)
from abridgeai.features.interviews.ai.stages.validation.verdicts import Verdict


def _draft(
    idx: int,
    question_type: str = "technical",
) -> InterviewQuestionDraft:
    return InterviewQuestionDraft(
        question_type=question_type,  # type: ignore[arg-type]
        prompt_text=f"Variant {question_type} question #{idx} for testing purposes.",
        difficulty="easy",
        expected_depth=2,
        linked_outcome_id=None,
        source_refs=[uuid4()],
        rationale=f"probe {idx}",
    )


def _verdict(idx: int, *, accepted: bool) -> Verdict:
    return Verdict(question_index=idx, accepted=accepted, failed_criteria=[], rationale="")


def _fake_stubs() -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    state = SimpleNamespace(id=uuid4(), config_json={})
    config = SimpleNamespace(supplementary_instructions=None, persona=None)
    context = SimpleNamespace(chunks=[])
    return state, config, context


@pytest.mark.asyncio
async def test_all_angles_backfill_requests_whole_angle_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backfill must round the shortfall UP to whole angle groups."""
    state, config, context = _fake_stubs()

    # Round 1: 8 requested (2 logical x 4), 6 accepted → shortfall of 2 rows.
    round1_drafts = [
        _draft(i, ["technical", "system_design", "situational", "behavioral"][i % 4])
        for i in range(8)
    ]
    round1_verdicts = [_verdict(i, accepted=i not in (3, 7)) for i in range(8)]
    round2_drafts = [
        _draft(20 + i, qtype)
        for i, qtype in enumerate(["technical", "system_design", "situational", "behavioral"])
    ]
    round2_verdicts = [_verdict(i, accepted=True) for i in range(4)]

    generate = AsyncMock(side_effect=[round1_drafts, round2_drafts])
    validate = AsyncMock(side_effect=[round1_verdicts, round2_verdicts])
    monkeypatch.setattr(
        "abridgeai.features.interviews.ai.pipelines.backfill.generate_interview_questions",
        generate,
    )
    monkeypatch.setattr(
        "abridgeai.features.interviews.ai.pipelines.backfill.validate_interview_questions",
        validate,
    )

    all_drafts, all_verdicts, accepted, rounds = await generate_with_backfill(
        AsyncMock(),
        state=state,
        config=config,
        context=context,
        outcomes=[],
        target_count=8,
        variant_strategy="all_angles",
        role_type=None,
    )

    assert len(accepted) == 8
    assert len(all_drafts) == 12
    assert len(all_verdicts) == 12
    assert rounds == 1
    # Round 2 override is the shortfall (2) rounded UP to a whole angle group
    # (4 rows) so every angle is re-requested; surplus rows are trimmed below.
    assert generate.await_args_list[1].kwargs["override_question_count"] == 4
    # Behavioral was rejected in round 1. It must outrank response-leading
    # technical/system-design candidates during the two-row trim.
    tail_types = [d.question_type for d in accepted[6:]]
    assert tail_types == ["behavioral", "technical"]
    assert [d.prompt_text for d in accepted[6:]] == [
        "Variant behavioral question #23 for testing purposes.",
        "Variant technical question #20 for testing purposes.",
    ]
    assert {
        question_type: sum(d.question_type == question_type for d in accepted)
        for question_type in ("technical", "system_design", "situational", "behavioral")
    } == {
        "technical": 3,
        "system_design": 2,
        "situational": 2,
        "behavioral": 1,
    }


def test_trim_to_shortfall_prioritizes_absent_angle_over_response_order() -> None:
    accepted = [
        _draft(0, "technical"),
        _draft(1, "system_design"),
        _draft(2, "situational"),
        _draft(3, "technical"),
        _draft(4, "system_design"),
        _draft(5, "situational"),
    ]
    candidates = [
        _draft(10, "technical"),
        _draft(11, "system_design"),
        _draft(12, "situational"),
        _draft(13, "behavioral"),
    ]

    kept = _trim_to_shortfall(candidates, accepted, missing=2)

    assert [draft.question_type for draft in kept] == ["behavioral", "technical"]


def test_trim_to_shortfall_never_exceeds_shortfall() -> None:
    accepted: list[InterviewQuestionDraft] = []
    candidates = [_draft(index, "technical") for index in range(3)]

    kept = _trim_to_shortfall(candidates, accepted, missing=2)

    assert len(kept) == 2


@pytest.mark.asyncio
async def test_legacy_mode_backfill_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-variant mode keeps the exact-shortfall (+buffer) request shape."""
    state, config, context = _fake_stubs()

    drafts1 = [_draft(i) for i in range(5)]
    verdicts1 = [_verdict(i, accepted=i != 4) for i in range(5)]
    drafts2 = [_draft(30)]
    verdicts2 = [_verdict(0, accepted=True)]

    generate = AsyncMock(side_effect=[drafts1, drafts2])
    validate = AsyncMock(side_effect=[verdicts1, verdicts2])
    monkeypatch.setattr(
        "abridgeai.features.interviews.ai.pipelines.backfill.generate_interview_questions",
        generate,
    )
    monkeypatch.setattr(
        "abridgeai.features.interviews.ai.pipelines.backfill.validate_interview_questions",
        validate,
    )

    _d, _v, accepted, _r = await generate_with_backfill(
        AsyncMock(),
        state=state,
        config=config,
        context=context,
        outcomes=[],
        target_count=5,
        variant_strategy=None,
        role_type=None,
    )
    assert len(accepted) == 5
    assert generate.await_args_list[1].kwargs["override_question_count"] == 2  # 1+buffer
