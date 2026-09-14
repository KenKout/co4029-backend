"""Contract tests for the dean path-change rejection payload."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from abridgeai.features.learning_programs.schemas import ChangeRequestRejection


def test_predefined_reason_allows_an_optional_note_without_custom_reason() -> None:
    payload = ChangeRequestRejection(
        reason_code="documentation_missing",
        note="Please attach the transcript before requesting again.",
    )

    assert payload.reason is None
    assert payload.note == "Please attach the transcript before requesting again."


def test_other_requires_a_separate_custom_reason_but_note_stays_optional() -> None:
    payload = ChangeRequestRejection(
        reason_code="other",
        reason="The destination path is closing this semester.",
    )

    assert payload.reason == "The destination path is closing this semester."
    assert payload.note is None


def test_other_rejects_a_blank_custom_reason_even_when_note_is_present() -> None:
    with pytest.raises(ValidationError, match="reason_is_required_when_reason_code_is_other"):
        ChangeRequestRejection(
            reason_code="other",
            reason="   ",
            note="This note must not substitute for the reason.",
        )
