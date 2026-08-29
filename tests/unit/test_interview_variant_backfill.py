"""Unit tests for atomic all-angle interview-generation backfill."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from abridgeai.features.interviews.ai.pipelines.backfill import generate_with_backfill
from abridgeai.features.interviews.ai.stages.generation import InterviewQuestionDraft
from abridgeai.features.interviews.ai.stages.generation.resolve import VARIANT_ANGLES
from abridgeai.features.interviews.ai.stages.validation.verdicts import Verdict


def _draft(
    idx: int,
    question_type: str = "technical",
    *,
    group_id: UUID | None = None,
) -> InterviewQuestionDraft:
    return InterviewQuestionDraft(
        question_type=question_type,  # type: ignore[arg-type]
        prompt_text=f"Variant {question_type} question #{idx} for testing purposes.",
        difficulty="easy",
        expected_depth=2,
        linked_outcome_id=None,
        variant_group_id=group_id,
        source_refs=[uuid4()],
        rationale=f"probe {idx}",
    )


def _group(start: int, group_id: UUID | None = None) -> list[InterviewQuestionDraft]:
    return [
        _draft(start + index, question_type, group_id=group_id or uuid4())
        for index, question_type in enumerate(VARIANT_ANGLES)
    ]


def _coherent_group(start: int) -> list[InterviewQuestionDraft]:
    group_id, outcome_id = uuid4(), uuid4()
    drafts = _group(start, group_id)
    for draft in drafts:
        draft.linked_outcome_id = outcome_id
    return drafts


def _verdict(idx: int, *, accepted: bool) -> Verdict:
    return Verdict(question_index=idx, accepted=accepted, failed_criteria=[], rationale="")


def _fake_stubs() -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    state = SimpleNamespace(id=uuid4(), config_json={})
    config = SimpleNamespace(supplementary_instructions=None, persona=None)
    context = SimpleNamespace(chunks=[])
    return state, config, context


@pytest.mark.asyncio
async def test_all_angles_backfill_retains_complete_groups_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, config, context = _fake_stubs()
    group_one = _coherent_group(0)
    group_two = _coherent_group(20)
    group_three = _coherent_group(40)
    generate = AsyncMock(side_effect=[group_one, [*group_two, *group_three]])
    validate = AsyncMock(
        side_effect=[
            [_verdict(index, accepted=True) for index in range(4)],
            [_verdict(index, accepted=True) for index in range(8)],
        ]
    )
    monkeypatch.setattr(
        "abridgeai.features.interviews.ai.pipelines.backfill.generate_interview_questions",
        generate,
    )
    monkeypatch.setattr(
        "abridgeai.features.interviews.ai.pipelines.backfill.validate_interview_questions",
        validate,
    )

    _drafts, _verdicts, accepted, rounds = await generate_with_backfill(
        AsyncMock(),
        state=state,
        config=config,
        context=context,
        outcomes=[],
        target_count=8,
        variant_strategy="all_angles",
        role_type=None,
    )

    assert rounds == 1
    assert len(accepted) == 8
    assert generate.await_args_list[1].kwargs["override_question_count"] == 4
    group_sizes: dict[UUID, int] = {}
    group_types: dict[UUID, set[str]] = {}
    for draft in accepted:
        assert draft.variant_group_id is not None
        group_sizes[draft.variant_group_id] = group_sizes.get(draft.variant_group_id, 0) + 1
        group_types.setdefault(draft.variant_group_id, set()).add(draft.question_type)
    assert set(group_sizes.values()) == {4}
    assert all(types == set(VARIANT_ANGLES) for types in group_types.values())


@pytest.mark.asyncio
async def test_all_angles_discards_a_group_when_one_angle_fails_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One rejected angle discards the whole group — and ends the loop.

    A partial (3/4) group is never persisted (atomicity), and a round that
    produced no complete group stops the backfill immediately instead of
    burning up to three more LLM rounds re-attempting the same group.
    """
    state, config, context = _fake_stubs()
    invalid_group = _coherent_group(0)
    generate = AsyncMock(return_value=invalid_group)
    validate = AsyncMock(
        return_value=[_verdict(index, accepted=index != 3) for index in range(4)]
    )
    monkeypatch.setattr(
        "abridgeai.features.interviews.ai.pipelines.backfill.generate_interview_questions",
        generate,
    )
    monkeypatch.setattr(
        "abridgeai.features.interviews.ai.pipelines.backfill.validate_interview_questions",
        validate,
    )

    _drafts, _verdicts, accepted, rounds = await generate_with_backfill(
        AsyncMock(),
        state=state,
        config=config,
        context=context,
        outcomes=[],
        target_count=4,
        variant_strategy="all_angles",
        role_type=None,
    )

    assert accepted == []
    assert rounds == 0
    assert generate.await_count == 1


@pytest.mark.asyncio
async def test_all_angles_requires_divisible_target() -> None:
    state, config, context = _fake_stubs()

    with pytest.raises(ValueError, match="divisible"):
        await generate_with_backfill(
            AsyncMock(),
            state=state,
            config=config,
            context=context,
            outcomes=[],
            target_count=6,
            variant_strategy="all_angles",
            role_type=None,
        )


@pytest.mark.asyncio
async def test_legacy_mode_backfill_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    state, config, context = _fake_stubs()
    drafts_one = [_draft(index) for index in range(5)]
    verdicts_one = [_verdict(index, accepted=index != 4) for index in range(5)]
    generate = AsyncMock(side_effect=[drafts_one, [_draft(30)]])
    validate = AsyncMock(
        side_effect=[
            verdicts_one,
            [_verdict(0, accepted=True)],
        ]
    )
    monkeypatch.setattr(
        "abridgeai.features.interviews.ai.pipelines.backfill.generate_interview_questions",
        generate,
    )
    monkeypatch.setattr(
        "abridgeai.features.interviews.ai.pipelines.backfill.validate_interview_questions",
        validate,
    )

    _drafts, _verdicts, accepted, _rounds = await generate_with_backfill(
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
    assert generate.await_args_list[1].kwargs["override_question_count"] == 2
