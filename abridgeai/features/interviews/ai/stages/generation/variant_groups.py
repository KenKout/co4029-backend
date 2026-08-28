"""All-angle logical-group selection for parsed interview drafts."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from abridgeai.features.interviews.ai.stages.generation.resolve import (
    VARIANT_ANGLES,
)

if TYPE_CHECKING:
    from abridgeai.features.interviews.ai.stages.generation.parsers import (
        InterviewQuestionDraft,
    )


_EXPECTED_ANGLE_SET = frozenset(VARIANT_ANGLES)


def select_all_angle_groups(
    drafts: list[InterviewQuestionDraft],
    max_questions: int | None,
) -> list[InterviewQuestionDraft]:
    """Keep valid logical groups whole, assigning each a server-owned UUID.

    The model ordinal is a correlation hint only. A malformed index group cannot
    leak into later validation, and the physical row cap never slices a group.
    """
    groups: dict[int, list[InterviewQuestionDraft]] = {}
    for draft in drafts:
        if draft.logical_question_index is not None:
            groups.setdefault(draft.logical_question_index, []).append(draft)

    kept: list[InterviewQuestionDraft] = []
    for logical_index in sorted(groups):
        members = groups[logical_index]
        types = {draft.question_type for draft in members}
        outcomes = {draft.linked_outcome_id for draft in members}
        difficulties = {draft.difficulty for draft in members}
        if (
            len(members) != len(VARIANT_ANGLES)
            or types != _EXPECTED_ANGLE_SET
            or len(outcomes) != 1
            or len(difficulties) != 1
        ):
            continue
        if max_questions is not None and len(kept) + len(members) > max_questions:
            continue
        group_id = uuid4()
        by_type = {draft.question_type: draft for draft in members}
        for angle in VARIANT_ANGLES:
            by_type[angle].variant_group_id = group_id
            kept.append(by_type[angle])
    return kept


__all__ = ["select_all_angle_groups"]
