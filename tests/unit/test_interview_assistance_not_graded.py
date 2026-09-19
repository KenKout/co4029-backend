"""P1 #11 regression: an assistance request must never enter grading evidence.

A typed ``hint`` / ``repeat`` / ``clarify`` turn is ACK'd and replied to
without a receipt — but the model's reply flow feeds the user text back
through the SDK chat context, and the conversation recorder persisted every
user item as ``kind="answer"`` linked to the current question. The evaluator
then read "give me a hint" as the candidate's answer to Q1.

Contract: a user item produced by an assistance action carries
``metadata_json.kind = "assistance"`` (plus the action name), and the shared
evaluation predicate rejects that kind. Ordinary answers are untouched.
"""

from __future__ import annotations

import pytest
from uuid import uuid4

from abridgeai.features.interviews.ai.stages.evaluation.logic import (
    _is_candidate_answer,
)

pytestmark = pytest.mark.asyncio


def _message(metadata: dict[str, object] | None, *, role: str = "user") -> object:
    return type(
        "_M",
        (),
        {
            "role": role,
            "metadata_json": metadata,
            "session_question_id": uuid4(),
            "content_text": "give me a hint",
        },
    )()


async def test_assistance_kind_user_row_is_not_evidence() -> None:
    message = _message(
        {"kind": "assistance", "turn_action": "hint", "source": "native_agent"}
    )
    assert _is_candidate_answer(message) is False


async def test_recorder_assistance_marker_is_not_evidence() -> None:
    """A recorder-written assistance row (voice "give me a hint", or the SDK
    echo of a typed action) carries kind=answer today and IS graded — the
    marker must exclude it."""
    message = _message({"kind": "assistance", "turn_action": "repeat"})
    assert _is_candidate_answer(message) is False


async def test_plain_typed_answer_receipt_is_still_evidence() -> None:
    message = _message(
        {
            "kind": "answer",
            "source": "native_agent",
            "turn_key": "k1",
            "turn_state": "applied",
        }
    )
    assert _is_candidate_answer(message) is True


async def test_rest_answer_without_metadata_is_still_evidence() -> None:
    message = _message(None)
    assert _is_candidate_answer(message) is True
