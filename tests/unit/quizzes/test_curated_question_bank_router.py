"""Unit coverage for curated Quiz Question Bank HTTP orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.core.pagination import CursorPage
from abridgeai.core.security import CurrentUser
from abridgeai.features.quizzes.routers import curated_question_bank as bank_router
from abridgeai.features.quizzes.schemas import (
    QuizQuestionBankCopyRequest,
    QuizQuestionBankImportRequest,
    QuizQuestionBankItemCreate,
    QuizQuestionBankItemUpdate,
)
from abridgeai.features.quizzes.services.curated_question_bank import (
    CuratedBankCopyResult,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def ids() -> SimpleNamespace:
    return SimpleNamespace(course=uuid4(), item=uuid4(), quiz=uuid4(), question=uuid4())


@pytest.fixture
def actor() -> CurrentUser:
    return CurrentUser(user_id=uuid4(), session_id=uuid4())


@pytest.fixture
def db() -> AsyncMock:
    return AsyncMock(spec=AsyncSession)


def _bank_item(ids: SimpleNamespace) -> dict:
    now = datetime.now(UTC)
    return {
        "id": ids.item,
        "course_id": ids.course,
        "status": "draft",
        "content_hash": "a" * 64,
        "question_type": "short_answer",
        "prompt_text": "What is dependency inversion?",
        "created_at": now,
        "updated_at": now,
    }


def _quiz_question(ids: SimpleNamespace) -> dict:
    now = datetime.now(UTC)
    return {
        "id": ids.question,
        "quiz_id": ids.quiz,
        "position": 1,
        "question_type": "short_answer",
        "prompt_text": "Explain dependency inversion.",
        "review_status": "pending",
        "created_at": now,
        "updated_at": now,
    }


async def test_list_forwards_filters_and_serializes_page(
    monkeypatch: pytest.MonkeyPatch,
    ids: SimpleNamespace,
    actor: CurrentUser,
    db: AsyncMock,
) -> None:
    service = AsyncMock(
        return_value=CursorPage(items=[_bank_item(ids)], next_cursor="next-page")
    )
    monkeypatch.setattr(bank_router.curated_bank_service, "list_curated_bank_items", service)

    result = await bank_router.list_curated_quiz_question_bank(
        ids.course,
        actor,
        db,
        bank_status="draft",
        question_type="short_answer",
        bloom_level="understand",
        difficulty="medium",
        search="dependency",
        limit=25,
        cursor="cursor",
    )

    assert result.next_cursor == "next-page"
    assert result.items[0].id == ids.item
    service.assert_awaited_once_with(
        db,
        course_id=ids.course,
        status="draft",
        question_type="short_answer",
        bloom_level="understand",
        difficulty="medium",
        search="dependency",
        limit=25,
        cursor="cursor",
    )
    db.commit.assert_not_awaited()


async def test_write_endpoints_commit_and_serialize_results(
    monkeypatch: pytest.MonkeyPatch,
    ids: SimpleNamespace,
    actor: CurrentUser,
    db: AsyncMock,
) -> None:
    item = _bank_item(ids)
    create = AsyncMock(return_value=item)
    copy = AsyncMock(
        return_value=CuratedBankCopyResult(
            created=[item], skipped_question_ids=[ids.question]
        )
    )
    update = AsyncMock(return_value={**item, "prompt_text": "Updated"})
    set_status = AsyncMock(return_value={**item, "status": "approved"})
    delete = AsyncMock(return_value=None)
    import_items = AsyncMock(return_value=[_quiz_question(ids)])
    monkeypatch.setattr(bank_router.curated_bank_service, "create_curated_bank_item", create)
    monkeypatch.setattr(
        bank_router.curated_bank_service, "copy_questions_to_curated_bank", copy
    )
    monkeypatch.setattr(bank_router.curated_bank_service, "update_curated_bank_item", update)
    monkeypatch.setattr(
        bank_router.curated_bank_service, "set_curated_bank_item_status", set_status
    )
    monkeypatch.setattr(bank_router.curated_bank_service, "delete_curated_bank_item", delete)
    monkeypatch.setattr(
        bank_router.curated_bank_service, "import_curated_bank_items", import_items
    )

    created = await bank_router.create_curated_quiz_question_bank_item(
        ids.course,
        QuizQuestionBankItemCreate(
            question_type="short_answer", prompt_text="A question"
        ),
        actor,
        db,
    )
    copied = await bank_router.copy_quiz_questions_to_curated_bank(
        ids.course,
        QuizQuestionBankCopyRequest(question_ids=[ids.question]),
        actor,
        db,
    )
    updated = await bank_router.update_curated_quiz_question_bank_item(
        ids.course,
        ids.item,
        QuizQuestionBankItemUpdate(prompt_text="Updated"),
        actor,
        db,
    )
    approved = await bank_router.set_curated_quiz_question_bank_item_status(
        ids.course,
        ids.item,
        bank_router._QuizBankStatusBody(status="approved"),
        actor,
        db,
    )
    deleted = await bank_router.delete_curated_quiz_question_bank_item(
        ids.course, ids.item, actor, db
    )
    imported = await bank_router.import_curated_quiz_question_bank_items(
        ids.quiz,
        QuizQuestionBankImportRequest(item_ids=[ids.item]),
        actor,
        db,
    )

    assert created.id == ids.item
    assert copied.created[0].id == ids.item
    assert copied.skipped == [ids.question]
    assert updated.prompt_text == "Updated"
    assert approved.status == "approved"
    assert deleted.status_code == 204
    assert imported[0].imported_from_bank_item_id is None
    assert db.commit.await_count == 6


_ERROR_CASES = [
    ("list", NotFoundError("missing"), 404),
    ("list", AppError("invalid cursor"), 400),
    ("create", NotFoundError("missing"), 404),
    ("create", ConflictError("quiz_question_bank_duplicate_content"), 409),
    ("create", AppError("invalid"), 400),
    ("copy", NotFoundError("missing"), 404),
    ("copy", ConflictError("busy"), 409),
    ("copy", AppError("invalid"), 400),
    ("update", NotFoundError("missing"), 404),
    ("update", ConflictError("busy"), 409),
    ("update", AppError("invalid"), 400),
    ("status", NotFoundError("missing"), 404),
    ("status", ConflictError("busy"), 409),
    ("status", AppError("invalid"), 400),
    ("delete", NotFoundError("missing"), 404),
    ("import", NotFoundError("missing"), 404),
    ("import", ConflictError("busy"), 409),
    ("import", AppError("invalid"), 400),
]


async def _invoke_error_case(
    operation: str,
    ids: SimpleNamespace,
    actor: CurrentUser,
    db: AsyncMock,
) -> None:
    if operation == "list":
        await bank_router.list_curated_quiz_question_bank(ids.course, actor, db)
    elif operation == "create":
        await bank_router.create_curated_quiz_question_bank_item(
            ids.course,
            QuizQuestionBankItemCreate(
                question_type="short_answer", prompt_text="Question"
            ),
            actor,
            db,
        )
    elif operation == "copy":
        await bank_router.copy_quiz_questions_to_curated_bank(
            ids.course,
            QuizQuestionBankCopyRequest(question_ids=[ids.question]),
            actor,
            db,
        )
    elif operation == "update":
        await bank_router.update_curated_quiz_question_bank_item(
            ids.course,
            ids.item,
            QuizQuestionBankItemUpdate(prompt_text="Updated"),
            actor,
            db,
        )
    elif operation == "status":
        await bank_router.set_curated_quiz_question_bank_item_status(
            ids.course,
            ids.item,
            bank_router._QuizBankStatusBody(status="approved"),
            actor,
            db,
        )
    elif operation == "delete":
        await bank_router.delete_curated_quiz_question_bank_item(
            ids.course, ids.item, actor, db
        )
    else:
        await bank_router.import_curated_quiz_question_bank_items(
            ids.quiz,
            QuizQuestionBankImportRequest(item_ids=[ids.item]),
            actor,
            db,
        )


@pytest.mark.parametrize(("operation", "service_error", "expected_status"), _ERROR_CASES)
async def test_service_errors_are_mapped_to_http_responses(
    monkeypatch: pytest.MonkeyPatch,
    ids: SimpleNamespace,
    actor: CurrentUser,
    db: AsyncMock,
    operation: str,
    service_error: Exception,
    expected_status: int,
) -> None:
    service_names = {
        "list": "list_curated_bank_items",
        "create": "create_curated_bank_item",
        "copy": "copy_questions_to_curated_bank",
        "update": "update_curated_bank_item",
        "status": "set_curated_bank_item_status",
        "delete": "delete_curated_bank_item",
        "import": "import_curated_bank_items",
    }
    monkeypatch.setattr(
        bank_router.curated_bank_service,
        service_names[operation],
        AsyncMock(side_effect=service_error),
    )

    with pytest.raises(HTTPException) as caught:
        await _invoke_error_case(operation, ids, actor, db)

    assert caught.value.status_code == expected_status
    assert db.commit.await_count == 0
    if isinstance(service_error, ConflictError):
        expected_message = (
            "This question already exists in the curated question bank."
            if str(service_error) == "quiz_question_bank_duplicate_content"
            else str(service_error)
        )
        assert caught.value.detail["message"] == expected_message


async def test_error_helpers_keep_machine_readable_detail(ids: SimpleNamespace) -> None:
    not_found = bank_router._not_found("bank_item", UUID(str(ids.item)))
    bad_request = bank_router._bad_request("bad filter")

    assert not_found.detail == {
        "error": "not_found",
        "resource": "bank_item",
        "id": str(ids.item),
    }
    assert bad_request.detail == {"error": "bad_request", "message": "bad filter"}
