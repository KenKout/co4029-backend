"""Phase 13 unit tests: the audit event registry guard."""

from __future__ import annotations

import pytest

from abridgeai.features.quizzes.services.audit import QUIZ_AUDIT_EVENTS


def test_registry_contains_correctness_bearing_events():
    for name in (
        "attempt_submitted",
        "attempt_regraded",
        "attempt_manually_graded",
        "override_created",
        "override_updated",
        "override_deleted",
        "question_edited",
        "quiz_published",
    ):
        assert name in QUIZ_AUDIT_EVENTS


def test_registry_is_frozen():
    assert isinstance(QUIZ_AUDIT_EVENTS, frozenset)
