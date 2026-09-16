from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from abridgeai.core.exceptions import AppError, NotFoundError
from abridgeai.features.quizzes.services import audit, gradebook, regrade
from abridgeai.features.quizzes.services.grader import GradeResult


class _Result:
    def __init__(self, rows: object = None, *, scalar: object = ...):
        self.rows = [] if rows is None else rows
        self.scalar = scalar

    def scalar_one_or_none(self):
        if self.scalar is ...:
            raise AssertionError("scalar result was not configured")
        return self.scalar

    def scalars(self) -> _Result:
        return self

    def all(self):
        return self.rows

    def __iter__(self):
        return iter(self.rows)


def test_selected_option_ids_normalizes_all_storage_shapes():
    first = uuid4()
    second = uuid4()
    answer = SimpleNamespace(
        selected_option_id=first,
        selected_option_ids=[str(second), "bad", None],
        answer_text=None,
    )
    assert regrade._selected_option_ids(answer) == [first, second]

    encoded = SimpleNamespace(
        selected_option_id=None,
        selected_option_ids=None,
        answer_text=f'["{first}", "invalid"]',
    )
    assert regrade._selected_option_ids(encoded) == [first]

    malformed = SimpleNamespace(
        selected_option_id=None, selected_option_ids=None, answer_text="not-json"
    )
    assert regrade._selected_option_ids(malformed) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "attrs", "expected_key", "expected_value"),
    [
        ("short_answer", {"original_generated_payload": {"correct_answer": "x"}}, "correct_answer", "x"),
        ("fill_blank", {"original_generated_payload": {}}, "correct_answer", None),
        ("numerical", {"numeric_answer": Decimal("2"), "numeric_tolerance": Decimal("0.1")}, "numeric_answer", Decimal("2")),
        ("matching", {"match_pairs": [{"left": "a", "right": "b"}]}, "match_pairs", [{"left": "a", "right": "b"}]),
        ("ordering", {"ordering_sequence": ["a", "b"]}, "ordering_sequence", ["a", "b"]),
        ("code", {}, "question_type", "code"),
    ],
)
async def test_snapshot_live_question_for_non_option_types(
    kind: str, attrs: dict[str, object], expected_key: str, expected_value: object
):
    question = SimpleNamespace(
        id=uuid4(), question_type=kind, single_answer=True, **attrs
    )
    payload = await regrade._snapshot_live_question(object(), question)
    assert payload[expected_key] == expected_value


@pytest.mark.asyncio
async def test_snapshot_live_question_loads_option_keys():
    question = SimpleNamespace(id=uuid4(), question_type="multiple_choice", single_answer=False)
    options = [
        SimpleNamespace(option_key="A", is_correct=True),
        SimpleNamespace(option_key="B", is_correct=False),
    ]
    db = SimpleNamespace(execute=AsyncMock(return_value=_Result(options)))

    payload = await regrade._snapshot_live_question(db, question)

    assert payload == {
        "question_type": "multiple_choice",
        "single_answer": False,
        "options": [
            {"option_key": "A", "is_correct": True},
            {"option_key": "B", "is_correct": False},
        ],
    }


@pytest.mark.asyncio
async def test_require_quiz_and_current_revision_handle_missing_values():
    quiz_id = uuid4()
    db = SimpleNamespace(
        execute=AsyncMock(side_effect=[_Result(scalar=None), _Result(scalar=None)])
    )
    with pytest.raises(NotFoundError, match=str(quiz_id)):
        await regrade._require_quiz(db, quiz_id)
    assert await regrade._current_revision_id(db, uuid4()) is None


@pytest.mark.asyncio
async def test_compute_regrade_empty_scope_persists_zero_counts(monkeypatch):
    quiz_id = uuid4()
    quiz = SimpleNamespace(id=quiz_id)
    added: list[object] = []

    async def refresh(row: object) -> None:
        if getattr(row, "id", None) is None:
            row.id = uuid4()

    db = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[_Result(scalar=quiz), _Result([]), _Result([])]
        ),
        add=lambda row: added.append(row),
        refresh=AsyncMock(side_effect=refresh),
    )
    flush = AsyncMock()
    monkeypatch.setattr(regrade, "flush_or_conflict", flush)

    run = await regrade.compute_regrade(
        db,
        quiz_id=quiz_id,
        attempt_ids=None,
        question_ids=None,
        requested_by=uuid4(),
    )

    assert run.attempts_affected == 0
    assert run.answers_changed == 0
    assert run.status == "dry_run"
    assert added == [run]
    assert flush.await_count == 2


@pytest.mark.asyncio
async def test_compute_regrade_creates_only_changed_auto_graded_items(monkeypatch):
    quiz_id = uuid4()
    question = SimpleNamespace(
        id=uuid4(),
        question_type="short_answer",
        single_answer=True,
        original_generated_payload={"correct_answer": "new"},
    )
    attempt = SimpleNamespace(id=uuid4())
    changed = SimpleNamespace(
        id=uuid4(),
        attempt_id=attempt.id,
        question_id=question.id,
        manual_score=None,
        needs_manual_grade=False,
        selected_option_id=None,
        selected_option_ids=None,
        answer_text="new",
        is_correct=False,
        points_awarded=Decimal("0"),
    )
    manual = SimpleNamespace(**{**changed.__dict__, "id": uuid4(), "manual_score": Decimal("1")})
    unchanged = SimpleNamespace(**{**changed.__dict__, "id": uuid4(), "is_correct": True, "points_awarded": Decimal("1")})
    added: list[object] = []

    async def refresh(row: object) -> None:
        if getattr(row, "id", None) is None:
            row.id = uuid4()

    db = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                _Result(scalar=SimpleNamespace(id=quiz_id)),
                _Result([question]),
                _Result([attempt]),
                _Result([changed, manual, unchanged]),
            ]
        ),
        add=lambda row: added.append(row),
        refresh=AsyncMock(side_effect=refresh),
    )
    monkeypatch.setattr(regrade, "flush_or_conflict", AsyncMock())
    monkeypatch.setattr(
        regrade,
        "grade_answer_against_revision",
        Mock(return_value=GradeResult(is_correct=True, points_awarded=Decimal("1"))),
    )

    run = await regrade.compute_regrade(
        db,
        quiz_id=quiz_id,
        attempt_ids=[attempt.id],
        question_ids=[question.id],
        requested_by=None,
    )

    assert run.answers_changed == 1
    assert run.attempts_affected == 1
    assert len(added) == 2
    item = added[1]
    assert item.answer_id == changed.id
    assert item.old_is_correct is False
    assert item.new_is_correct is True


@pytest.mark.asyncio
async def test_get_regrade_run_returns_scoped_result():
    run = SimpleNamespace(id=uuid4())
    db = SimpleNamespace(execute=AsyncMock(return_value=_Result(scalar=run)))
    assert await regrade.get_regrade_run(db, quiz_id=uuid4(), run_id=run.id) is run


@pytest.mark.asyncio
async def test_commit_regrade_guards_missing_and_already_committed(monkeypatch):
    quiz = SimpleNamespace(id=uuid4())
    monkeypatch.setattr(regrade, "_require_quiz", AsyncMock(return_value=quiz))
    monkeypatch.setattr(regrade, "get_regrade_run", AsyncMock(return_value=None))
    with pytest.raises(NotFoundError, match="not found"):
        await regrade.commit_regrade(object(), quiz_id=quiz.id, run_id=uuid4())

    run = SimpleNamespace(id=uuid4(), status="committed")
    regrade.get_regrade_run.return_value = run
    with pytest.raises(AppError, match="not in dry_run"):
        await regrade.commit_regrade(object(), quiz_id=quiz.id, run_id=run.id)


@pytest.mark.asyncio
async def test_commit_empty_regrade_marks_run_committed(monkeypatch):
    now = datetime(2026, 9, 16, tzinfo=UTC)
    quiz = SimpleNamespace(id=uuid4())
    run = SimpleNamespace(id=uuid4(), status="dry_run", committed_at=None)
    db = SimpleNamespace(execute=AsyncMock(return_value=_Result([])), refresh=AsyncMock())
    monkeypatch.setattr(regrade, "_require_quiz", AsyncMock(return_value=quiz))
    monkeypatch.setattr(regrade, "get_regrade_run", AsyncMock(return_value=run))
    monkeypatch.setattr(regrade, "flush_or_conflict", AsyncMock())
    monkeypatch.setattr(regrade, "utcnow", lambda: now)

    result = await regrade.commit_regrade(db, quiz_id=quiz.id, run_id=run.id)

    assert result is run
    assert run.status == "committed"
    assert run.committed_at == now
    db.refresh.assert_awaited_once_with(run)


@pytest.mark.asyncio
async def test_commit_regrade_applies_items_recomputes_and_audits(monkeypatch):
    now = datetime(2026, 9, 16, tzinfo=UTC)
    quiz = SimpleNamespace(id=uuid4(), passing_score_percent=Decimal("70"))
    run = SimpleNamespace(
        id=uuid4(),
        status="dry_run",
        answers_changed=0,
        attempts_affected=0,
        committed_at=None,
    )
    attempt_id = uuid4()
    question_id = uuid4()
    answer = SimpleNamespace(
        id=uuid4(),
        manual_score=None,
        needs_manual_grade=False,
        is_correct=False,
        points_awarded=Decimal("0"),
        graded_revision_id=None,
    )
    skipped = SimpleNamespace(
        id=uuid4(), manual_score=Decimal("1"), needs_manual_grade=False
    )
    items = [
        SimpleNamespace(
            answer_id=answer.id,
            attempt_id=attempt_id,
            question_id=question_id,
            new_is_correct=True,
            new_points=Decimal("1"),
        ),
        SimpleNamespace(
            answer_id=skipped.id,
            attempt_id=attempt_id,
            question_id=question_id,
            new_is_correct=False,
            new_points=Decimal("0"),
        ),
        SimpleNamespace(
            answer_id=uuid4(),
            attempt_id=attempt_id,
            question_id=question_id,
            new_is_correct=False,
            new_points=Decimal("0"),
        ),
    ]
    revision_id = uuid4()
    attempt = SimpleNamespace(
        id=attempt_id,
        student_id=uuid4(),
        score_points=None,
        score_percent=None,
        passed=None,
    )
    db = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                _Result(items),
                _Result([answer, skipped]),
                _Result([(question_id, "short_answer")]),
                _Result(scalar=revision_id),
                _Result([attempt]),
                _Result([]),
            ]
        ),
        refresh=AsyncMock(),
    )
    monkeypatch.setattr(regrade, "_require_quiz", AsyncMock(return_value=quiz))
    monkeypatch.setattr(regrade, "get_regrade_run", AsyncMock(return_value=run))
    monkeypatch.setattr(regrade, "flush_or_conflict", AsyncMock())
    monkeypatch.setattr(regrade, "utcnow", lambda: now)
    monkeypatch.setattr(regrade, "needs_manual_grade", Mock(return_value=False))
    monkeypatch.setattr(
        regrade,
        "_recompute_attempt_score",
        AsyncMock(return_value=(Decimal("4"), Decimal("80"), 4, 5)),
    )
    recompute_grade = AsyncMock()
    record = AsyncMock()
    monkeypatch.setattr(gradebook, "recompute_final_grade", recompute_grade)
    monkeypatch.setattr(audit, "record_event", record)
    logger = Mock()
    monkeypatch.setattr(regrade, "_logger", logger)

    result = await regrade.commit_regrade(
        db, quiz_id=quiz.id, run_id=run.id, reconcile_sr=True
    )

    assert result is run
    assert answer.is_correct is True
    assert answer.points_awarded == Decimal("1")
    assert answer.graded_revision_id == revision_id
    assert run.answers_changed == 1
    assert run.attempts_affected == 1
    assert attempt.score_points == Decimal("4")
    assert attempt.score_percent == Decimal("80")
    assert attempt.passed is True
    recompute_grade.assert_awaited_once_with(db, quiz, attempt.student_id)
    logger.warning.assert_called_once()
    assert run.status == "committed"
    assert run.committed_at == now
    record.assert_awaited_once()
    db.refresh.assert_awaited_once_with(run)
