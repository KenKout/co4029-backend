from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

from sqlalchemy import Boolean

from abridgeai.features.quizzes.models import Quiz
from abridgeai.features.quizzes.schemas.public import QuizPublic
from abridgeai.features.quizzes.services.integrity import integrity_policy_snapshot_from_quiz


def test_require_camera_is_a_false_by_default_quiz_setting() -> None:
    column = Quiz.__table__.c.require_camera
    assert isinstance(column.type, Boolean)
    assert column.nullable is False
    assert str(column.server_default.arg).upper() == "FALSE"


def test_require_camera_is_visible_to_learners_without_policy_weights() -> None:
    quiz = QuizPublic(
        id=uuid4(),
        title="Camera quiz",
        status="published",
        passing_score_percent=Decimal("70.00"),
    )
    assert quiz.require_camera is False


def test_attempt_policy_snapshot_freezes_required_camera_setting() -> None:
    snapshot = integrity_policy_snapshot_from_quiz(
        SimpleNamespace(
            integrity_weight_tab_switch=3,
            integrity_weight_focus_lost=1,
            integrity_weight_fullscreen_exit=2,
            integrity_score_threshold=3,
            require_camera=True,
        )
    )
    assert snapshot["require_camera"] is True
