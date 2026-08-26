"""Unit tests for the variant-mode backfill contract (Slice 21 regression).

Regression: in ``all_angles`` mode the backfill loop passed the *logical*
shortfall as ``override_question_count``, but the generation stage treats
``override_question_count`` as a TOTAL row budget and re-divides it by the
angle count — so a backfill round asked for ceil(missing/4) logical
questions instead of exactly ``missing`` rows.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.features.interviews.ai.pipelines.backfill import (
    generate_with_backfill,
)
from abridgeai.features.interviews.ai.stages.generation import (
    InterviewQuestionDraft,
)
from abridgeai.features.interviews.ai.stages.validation.verdicts import Verdict


def _draft(idx: int) -> InterviewQuestionDraft:
    return InterviewQuestionDraft(
        question_type="technical",
        prompt_text=f"Variant question #{idx} for testing purposes.",
        difficulty="easy",
        expected_depth=2,
        linked_outcome_id=None,
        source_refs=[uuid4()],
        rationale=f"probe {idx}",
    )


def _verdict(idx: int, *, accepted: bool) -> Verdict:
    return Verdict(
        question_index=idx, accepted=accepted, failed_criteria=[], rationale=""
    )


def _fake_stubs() -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    state = SimpleNamespace(id=uuid4(), config_json={})
    config = SimpleNamespace(supplementary_instructions=None, persona=None)
    context = SimpleNamespace(chunks=[])
    return state, config, context


@pytest.mark.asyncio
async def test_all_angles_backfill_requests_total_rows_not_logical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backfill must request the missing ROW count; the stage owns the /4."""
    state, config, context = _fake_stubs()

    # Round 1: 8 requested, 6 accepted → shortfall of 2 rows.
    round1_drafts = [_draft(i) for i in range(8)]
    round1_verdicts = [_verdict(i, accepted=i not in (2, 5)) for i in range(8)]
    round2_drafts = [_draft(20), _draft(21)]
    round2_verdicts = [_verdict(i, accepted=True) for i in range(2)]

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

    # Round 1 accepted 6 of 8 (indices 2 & 5 rejected); round 2 tops up with
    # the 2 backfill drafts → 8 accepted total.
    assert len(accepted) == 8
    assert len(all_drafts) == 10
    # Round 2 override is the shortfall (2) + backfill buffer (+1) = 3 TOTAL
    # rows; the generation stage re-divides that by the angle count itself.
    assert generate.await_args_list[1].kwargs["override_question_count"] == 3


@pytest.mark.asyncio
async def test_legacy_mode_backfill_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
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
