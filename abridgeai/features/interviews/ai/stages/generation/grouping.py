"""Variant-group assignment for multi-angle (``all_angles``) generation.

Peers probing the same logical problem share one server-assigned UUID so
the teacher can review or reject an entire group at once. Parsers own raw
LLM JSON handling; this module owns the grouping bookkeeping.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4


def coerce_logical_question_index(value: Any) -> int | None:  # noqa: ANN401 -- raw LLM JSON
    """Accept non-negative integer group ordinals, never booleans/floats."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def assign_variant_group_ids(
    drafts: list[Any],  # noqa: ANN401 -- duck-typed InterviewQuestionDraft
    *,
    require_index: bool,
) -> list[Any]:
    """Assign one durable UUID per model-supplied logical ordinal.

    Drafts sharing a ``logical_question_index`` get the same
    ``variant_group_id``; the ordinal itself is never persisted. When
    ``require_index`` is set, drafts without the ordinal are dropped
    (all-angle mode must yield complete groups).
    """
    group_ids: dict[int, UUID] = {}
    out: list[Any] = []
    for draft in drafts:
        if require_index:
            if draft.logical_question_index is None:
                continue
            draft.variant_group_id = group_ids.setdefault(
                draft.logical_question_index,
                uuid4(),
            )
        out.append(draft)
    return out
