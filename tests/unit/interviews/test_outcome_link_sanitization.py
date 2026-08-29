"""LLM-supplied outcome links are sanitized to the config's own outcomes.

Covers the pre-pass added to :func:`generate_interview_questions`: a
``linked_outcome_id`` that is not one of the config's outcomes (fake UUID
or another config's outcome) is treated as a missing link and replaced by
the round-robin assignment, so neither an FK error nor a foreign-config
rubric link can reach ``_persist_questions``.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from abridgeai.features.interviews.ai.stages.generation.logic import (
    generate_interview_questions,
)


def _outcome(outcome_id: uuid.UUID) -> SimpleNamespace:
    return SimpleNamespace(
        id=outcome_id,
        outcome_text="Linked Outcome",
        outcome_type="knowledge",
        importance_weight=3,
    )


def _entry(linked_outcome_id: uuid.UUID | None) -> dict[str, object]:
    return {
        "question_type": "technical",
        "prompt_text": "What does this outcome measure?",
        "difficulty": "medium",
        "linked_outcome_id": str(linked_outcome_id) if linked_outcome_id else None,
    }


class _FakeGateway:
    def __init__(self, entries: list[dict[str, object]]) -> None:
        self._entries = entries

    async def generate_json(self, **_: object) -> SimpleNamespace:  # noqa: ANN003
        return SimpleNamespace(content_json={"questions": self._entries})


def _run() -> SimpleNamespace:
    return SimpleNamespace(id=uuid.uuid4(), config_json={"question_count": 3})


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        title="Linked Outcome Interview",
        persona="neutral",
        supplementary_instructions=None,
        module_id=uuid.uuid4(),
    )


@pytest.mark.asyncio
async def test_foreign_and_fake_outcome_ids_are_replaced_by_round_robin() -> None:
    own_id, second_id = uuid.uuid4(), uuid.uuid4()
    outcomes = [_outcome(own_id), _outcome(second_id)]
    entries = [_entry(uuid.uuid4()), _entry(uuid.uuid4()), _entry(None)]
    drafts = await generate_interview_questions(
        None,  # db unused by the fake gateway
        run=_run(),
        config=_config(),
        context=SimpleNamespace(chunks=[]),
        outcomes=outcomes,
        gateway=_FakeGateway(entries),
    )
    assert len(drafts) == 3
    assert {draft.linked_outcome_id for draft in drafts} <= {own_id, second_id}
    assert all(draft.linked_outcome_id is not None for draft in drafts)


@pytest.mark.asyncio
async def test_valid_own_outcome_id_is_preserved() -> None:
    own_id = uuid.uuid4()
    outcomes = [_outcome(own_id), _outcome(uuid.uuid4())]
    drafts = await generate_interview_questions(
        None,
        run=_run(),
        config=_config(),
        context=SimpleNamespace(chunks=[]),
        outcomes=outcomes,
        gateway=_FakeGateway([_entry(own_id)]),
    )
    assert drafts[0].linked_outcome_id == own_id


@pytest.mark.asyncio
async def test_no_outcomes_leaves_links_null() -> None:
    drafts = await generate_interview_questions(
        None,
        run=_run(),
        config=_config(),
        context=SimpleNamespace(chunks=[]),
        outcomes=[],
        gateway=_FakeGateway([_entry(uuid.uuid4()), _entry(uuid.uuid4())]),
    )
    assert len(drafts) == 2
    assert all(draft.linked_outcome_id is None for draft in drafts)
