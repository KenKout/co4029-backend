"""P1 #6: the answer payload must carry a size cap (422 before any work).

``InterviewSubmitAnswerRequest.answer_text`` had no ``max_length`` — an
oversized payload walked the security classifier, the adaptive pipeline and
the transcript storage before anything rejected it. The schema now enforces
``MAX_ANSWER_CHARS`` (Pydantic → FastAPI 422) at the door; the typed
realtime door shares the constant.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from abridgeai.features.interviews.schemas.session import (
    MAX_ANSWER_CHARS,
    InterviewSubmitAnswerRequest,
)

pytestmark = pytest.mark.asyncio


def _body(answer_text: str | None) -> dict[str, object]:
    return {
        "session_id": "00000000-0000-0000-0000-000000000001",
        "session_question_id": "00000000-0000-0000-0000-000000000002",
        "answer_text": answer_text,
    }


async def test_answer_at_the_cap_is_accepted() -> None:
    payload = InterviewSubmitAnswerRequest.model_validate(_body("a" * MAX_ANSWER_CHARS))
    assert payload.answer_text is not None
    assert len(payload.answer_text) == MAX_ANSWER_CHARS


async def test_answer_over_the_cap_is_rejected_with_422_shape() -> None:
    with pytest.raises(ValidationError) as exc_info:
        InterviewSubmitAnswerRequest.model_validate(_body("a" * (MAX_ANSWER_CHARS + 1)))
    errors = exc_info.value.errors()
    assert any(e["type"] in ("too_long", "string_too_long", "string_type_max_length") for e in errors), errors


async def test_null_answer_still_validates() -> None:
    payload = InterviewSubmitAnswerRequest.model_validate(_body(None))
    assert payload.answer_text is None
