"""The scheduling window that decides whether a quiz may be started.

``available_from`` and ``available_until`` are hard bounds: before the first
a student cannot begin, after the second they cannot begin. ``due_at`` is
deliberately not enforced here — it is a soft deadline that marks an attempt
late without preventing it.

The value-based form exists so a per-student override can shift the window,
which means the override path and the plain path must agree about what "open"
means. Both bounds accept a naive datetime and read it as UTC; a naive value
misread as local time would open or close a quiz hours off, and every bound
in this system is stored UTC.

Pure functions over datetimes — the gate needs no database.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from abridgeai.features.quizzes.queries.published import (
    QuizClosed,
    QuizNotYetOpen,
    _assert_within_schedule_window_values,
    _looks_like_uuid,
)


@pytest.fixture
def quiz_id() -> UUID:
    return uuid4()


def _now() -> datetime:
    return datetime.now(UTC)


class TestTheWindowIsOpen:
    def test_no_bounds_means_always_open(self, quiz_id: UUID) -> None:
        """Pre-scheduling quizzes carry NULL on both columns."""
        _assert_within_schedule_window_values(quiz_id, None, None)

    def test_after_the_opening_bound(self, quiz_id: UUID) -> None:
        _assert_within_schedule_window_values(quiz_id, _now() - timedelta(hours=1), None)

    def test_before_the_closing_bound(self, quiz_id: UUID) -> None:
        _assert_within_schedule_window_values(quiz_id, None, _now() + timedelta(hours=1))

    def test_inside_both_bounds(self, quiz_id: UUID) -> None:
        _assert_within_schedule_window_values(
            quiz_id, _now() - timedelta(hours=1), _now() + timedelta(hours=1)
        )


class TestTheWindowIsShut:
    def test_before_the_opening_bound_refuses(self, quiz_id: UUID) -> None:
        opens = _now() + timedelta(hours=1)
        with pytest.raises(QuizNotYetOpen) as caught:
            _assert_within_schedule_window_values(quiz_id, opens, None)
        assert caught.value.quiz_id == quiz_id

    def test_after_the_closing_bound_refuses(self, quiz_id: UUID) -> None:
        closed = _now() - timedelta(hours=1)
        with pytest.raises(QuizClosed) as caught:
            _assert_within_schedule_window_values(quiz_id, None, closed)
        assert caught.value.quiz_id == quiz_id

    def test_the_opening_bound_is_checked_first(self, quiz_id: UUID) -> None:
        """A window whose bounds are inverted must name the opening failure.

        Both bounds fail here. Reporting the close would tell a student the
        quiz has finished when in truth it has not started, which is the more
        misleading of the two.
        """
        with pytest.raises(QuizNotYetOpen):
            _assert_within_schedule_window_values(
                quiz_id, _now() + timedelta(hours=1), _now() - timedelta(hours=1)
            )


class TestNaiveDatetimesAreReadAsUTC:
    """A bound arriving without a timezone must not shift the window.

    Every timestamp in this system is stored UTC, so a naive value is a UTC
    value that lost its marker in transit. Treating it as local time would
    move a quiz's opening by the host's offset — silently, and differently on
    a developer's machine than in production.
    """

    def test_a_naive_future_opening_still_refuses(self, quiz_id: UUID) -> None:
        opens = (_now() + timedelta(hours=1)).replace(tzinfo=None)
        with pytest.raises(QuizNotYetOpen):
            _assert_within_schedule_window_values(quiz_id, opens, None)

    def test_a_naive_past_opening_still_allows(self, quiz_id: UUID) -> None:
        opens = (_now() - timedelta(hours=1)).replace(tzinfo=None)
        _assert_within_schedule_window_values(quiz_id, opens, None)

    def test_a_naive_past_closing_still_refuses(self, quiz_id: UUID) -> None:
        closed = (_now() - timedelta(hours=1)).replace(tzinfo=None)
        with pytest.raises(QuizClosed):
            _assert_within_schedule_window_values(quiz_id, None, closed)

    def test_a_naive_future_closing_still_allows(self, quiz_id: UUID) -> None:
        closed = (_now() + timedelta(hours=1)).replace(tzinfo=None)
        _assert_within_schedule_window_values(quiz_id, None, closed)

    def test_the_refusal_carries_an_aware_timestamp(self, quiz_id: UUID) -> None:
        """The router shows this to the student, so it must not be ambiguous."""
        opens = (_now() + timedelta(hours=1)).replace(tzinfo=None)
        with pytest.raises(QuizNotYetOpen) as caught:
            _assert_within_schedule_window_values(quiz_id, opens, None)
        assert caught.value.available_from.tzinfo is not None


class TestSlugOrIdentifier:
    """A quiz is fetched by id or by slug, so the lookup has to tell them apart."""

    def test_a_uuid_is_recognised(self) -> None:
        assert _looks_like_uuid(str(uuid4())) is True

    @pytest.mark.parametrize(
        "value", ["not-a-uuid", "", "123", "intro-to-etl", "0000", "  "]
    )
    def test_a_slug_is_not_mistaken_for_one(self, value: str) -> None:
        assert _looks_like_uuid(value) is False

    def test_a_non_string_does_not_raise(self) -> None:
        """The value arrives from a URL path, so it must fail closed."""
        assert _looks_like_uuid(None) is False  # type: ignore[arg-type]
