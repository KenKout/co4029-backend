"""Phase 8 feedback schema + band-selection unit tests."""

from __future__ import annotations

from decimal import Decimal

from abridgeai.features.quizzes.schemas import FeedbackBandIn, FeedbackBandRead
from abridgeai.features.quizzes.schemas.feedback import OverallFeedbackRead


def test_public_option_schema_excludes_feedback():
    from abridgeai.features.quizzes.schemas import QuizQuestionOptionPublic

    # Security invariant: feedback + is_correct never leak on the public schema.
    assert "feedback_text" not in QuizQuestionOptionPublic.model_fields
    assert "is_correct" not in QuizQuestionOptionPublic.model_fields


def test_band_roundtrip():
    b = FeedbackBandRead.model_validate(
        {
            "id": "00000000-0000-0000-0000-000000000001",
            "min_grade": "80.00",
            "max_grade": "100.00",
            "feedback_text": "Great work!",
            "feedback_format": "markdown",
        }
    )
    assert b.max_grade == Decimal("100.00")


def test_band_in_defaults_markdown():
    b = FeedbackBandIn(min_grade=Decimal("0"), max_grade=Decimal("50"), feedback_text="Try again")
    assert b.feedback_format == "markdown"


def test_overall_feedback_read_shape():
    o = OverallFeedbackRead(feedback_text="x", feedback_format="markdown")
    assert o.feedback_text == "x"
