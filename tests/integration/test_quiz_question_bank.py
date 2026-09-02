"""Service-level integration tests for the question-bank feature.

Exercises the bank list query + import operation against the real
integration DB (same pattern as ``test_quiz_grader.py``). The HTTP
surface is covered by FastAPI's response_model serialisation; here we
focus on filtering correctness, course-scope guards, and the cloning
contract (new ids, fresh review state, lineage pointer).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from abridgeai.core.security import CurrentUser
from abridgeai.features.identity import (
    models as _identity_models,  # noqa: F401  -- side-effect: register users table
)
from abridgeai.features.interviews import (
    models as _interview_models,  # noqa: F401  -- register InterviewConfig for ModuleItem
)
from abridgeai.features.quizzes.models import (
    QuizQuestion,
    QuizQuestionBankItem,
    QuizQuestionOption,
)
from abridgeai.features.quizzes.schemas.bank import (
    QuizQuestionBankItemCreate,
    QuizQuestionBankItemRead,
    QuizQuestionBankItemUpdate,
    QuizQuestionBankOptionCreate,
)
from abridgeai.features.quizzes.services import (
    curated_question_bank as curated_bank_service,
)
from abridgeai.features.quizzes.services import question_bank as bank_service

pytestmark = pytest.mark.asyncio


def _async_url(database_url: str) -> str:
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


def _ensure_head() -> None:
    cfg_path = Path(__file__).resolve().parents[2] / "alembic.ini"
    cfg = Config(str(cfg_path))
    cfg.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parents[2] / "migrations"),
    )
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    from abridgeai.core.config import get_settings

    _ensure_head()
    eng = create_async_engine(
        _async_url(get_settings().database_url), pool_pre_ping=True
    )
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@dataclass
class BankFixture:
    """One course with two quizzes — bank source + import target."""

    actor: CurrentUser
    course_id: uuid.UUID
    other_course_id: uuid.UUID
    module_id: uuid.UUID
    other_module_id: uuid.UUID
    source_quiz_id: uuid.UUID
    target_quiz_id: uuid.UUID
    other_course_quiz_id: uuid.UUID
    approved_question_id: uuid.UUID
    pending_question_id: uuid.UUID
    other_course_question_id: uuid.UUID


@pytest_asyncio.fixture
async def bank_fixture(engine: AsyncEngine) -> BankFixture:
    """Seed two courses, two quizzes in one, plus a foreign quiz.

    Bank list defaults filter by ``review_status='approved'`` and scope
    to the calling course; the foreign-course quiz must NEVER appear in
    a bank query for the primary course, even when filters widen.
    """
    org_id = uuid.uuid4()
    user_id = uuid.uuid4()
    course_id = uuid.uuid4()
    other_course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    other_module_id = uuid.uuid4()
    source_quiz_id = uuid.uuid4()
    target_quiz_id = uuid.uuid4()
    other_course_quiz_id = uuid.uuid4()
    approved_question_id = uuid.uuid4()
    pending_question_id = uuid.uuid4()
    other_course_question_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"bank-{org_id.hex[:8]}", "name": "Bank Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": user_id, "email": f"bank-{user_id.hex[:8]}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title) "
                "VALUES (:c1, :org, :owner, :s1, :t1), (:c2, :org, :owner, :s2, :t2)"
            ),
            {
                "c1": course_id,
                "c2": other_course_id,
                "org": org_id,
                "owner": user_id,
                "s1": f"bank-course-{course_id.hex[:8]}",
                "t1": "Bank Course",
                "s2": f"other-course-{other_course_id.hex[:8]}",
                "t2": "Other Course",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position) VALUES "
                "(:m1, :c1, 'Module 1', 1), (:m2, :c2, 'Other Module', 1)"
            ),
            {
                "m1": module_id,
                "m2": other_module_id,
                "c1": course_id,
                "c2": other_course_id,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, status, slug) VALUES (:q1, :c1, :m1, 'Source quiz', 'draft', 'slug-' || uuid_generate_v4()::text), (:q2, :c1, :m1, 'Target quiz', 'draft', 'slug-' || uuid_generate_v4()::text), (:q3, :c2, :m2, 'Foreign quiz', 'draft', 'slug-' || uuid_generate_v4()::text);"
            ),
            {
                "q1": source_quiz_id,
                "q2": target_quiz_id,
                "q3": other_course_quiz_id,
                "c1": course_id,
                "c2": other_course_id,
                "m1": module_id,
                "m2": other_module_id,
            },
        )
        # Approved MCQ in source quiz (must surface in default bank list).
        await conn.execute(
            text(
                "INSERT INTO quiz_questions ("
                "id, quiz_id, position, question_type, prompt_text, "
                "review_status, difficulty, bloom_level, "
                "expected_response_time_ms, expected_ef_ceiling, "
                "source_refs, original_generated_payload) "
                "VALUES (:id, :quiz, 1, 'multiple_choice', 'Approved stem?', "
                "'approved', 'medium', 'understand', 30000, 2.50, "
                "'[]'::jsonb, '{\"correct_answer\": \"A\"}'::jsonb)"
            ),
            {"id": approved_question_id, "quiz": source_quiz_id},
        )
        # Pending T/F in source quiz (filtered out by default review_status='approved').
        await conn.execute(
            text(
                "INSERT INTO quiz_questions ("
                "id, quiz_id, position, question_type, prompt_text, "
                "review_status, difficulty, bloom_level, source_refs) "
                "VALUES (:id, :quiz, 2, 'true_false', 'Pending stem?', "
                "'pending', 'easy', 'remember', '[]'::jsonb)"
            ),
            {"id": pending_question_id, "quiz": source_quiz_id},
        )
        # Approved MCQ in OTHER course (must never appear in bank for the primary course).
        await conn.execute(
            text(
                "INSERT INTO quiz_questions ("
                "id, quiz_id, position, question_type, prompt_text, "
                "review_status, difficulty, bloom_level, source_refs) "
                "VALUES (:id, :quiz, 1, 'multiple_choice', 'Foreign stem?', "
                "'approved', 'hard', 'apply', '[]'::jsonb)"
            ),
            {"id": other_course_question_id, "quiz": other_course_quiz_id},
        )
        # Two options for the approved MCQ so cloning has something to copy.
        await conn.execute(
            text(
                "INSERT INTO quiz_question_options ("
                "id, question_id, option_key, option_text, is_correct, position) "
                "VALUES "
                "(:o1, :q, 'A', 'Right answer', TRUE, 1), "
                "(:o2, :q, 'B', 'Wrong answer', FALSE, 2)"
            ),
            {
                "o1": uuid.uuid4(),
                "o2": uuid.uuid4(),
                "q": approved_question_id,
            },
        )

    return BankFixture(
        actor=CurrentUser(user_id=user_id, session_id=uuid.uuid4()),
        course_id=course_id,
        other_course_id=other_course_id,
        module_id=module_id,
        other_module_id=other_module_id,
        source_quiz_id=source_quiz_id,
        target_quiz_id=target_quiz_id,
        other_course_quiz_id=other_course_quiz_id,
        approved_question_id=approved_question_id,
        pending_question_id=pending_question_id,
        other_course_question_id=other_course_question_id,
    )


async def test_list_bank_returns_only_approved_within_course(
    session_factory: async_sessionmaker[AsyncSession],
    bank_fixture: BankFixture,
) -> None:
    """Default review_status='approved' + course scope hides drafts and foreign quizzes."""
    async with session_factory() as session:
        page = await bank_service.list_bank_entries(
            session, course_id=bank_fixture.course_id
        )
    ids = {row["question"].id for row in page.items}
    assert bank_fixture.approved_question_id in ids
    assert bank_fixture.pending_question_id not in ids  # filtered by review_status
    assert bank_fixture.other_course_question_id not in ids  # foreign course


async def test_list_bank_can_widen_review_status(
    session_factory: async_sessionmaker[AsyncSession],
    bank_fixture: BankFixture,
) -> None:
    """``review_status=None`` returns drafts + edits + approvals."""
    async with session_factory() as session:
        page = await bank_service.list_bank_entries(
            session, course_id=bank_fixture.course_id, review_status=None
        )
    ids = {row["question"].id for row in page.items}
    assert bank_fixture.approved_question_id in ids
    assert bank_fixture.pending_question_id in ids


async def test_list_bank_filter_by_question_type(
    session_factory: async_sessionmaker[AsyncSession],
    bank_fixture: BankFixture,
) -> None:
    async with session_factory() as session:
        page = await bank_service.list_bank_entries(
            session,
            course_id=bank_fixture.course_id,
            review_status=None,
            question_type="true_false",
        )
    ids = {row["question"].id for row in page.items}
    assert ids == {bank_fixture.pending_question_id}


async def test_list_bank_search_matches_prompt(
    session_factory: async_sessionmaker[AsyncSession],
    bank_fixture: BankFixture,
) -> None:
    async with session_factory() as session:
        page = await bank_service.list_bank_entries(
            session,
            course_id=bank_fixture.course_id,
            search="Approved",
        )
    ids = {row["question"].id for row in page.items}
    assert ids == {bank_fixture.approved_question_id}


async def test_list_bank_exclude_quiz_filters_out_target(
    session_factory: async_sessionmaker[AsyncSession],
    bank_fixture: BankFixture,
) -> None:
    """``exclude_quiz_id`` removes the target quiz from the listing."""
    async with session_factory() as session:
        page = await bank_service.list_bank_entries(
            session,
            course_id=bank_fixture.course_id,
            exclude_quiz_id=bank_fixture.source_quiz_id,
        )
    assert page.items == []
    assert page.next_cursor is None


async def test_import_clones_with_new_ids_and_lineage(
    session_factory: async_sessionmaker[AsyncSession],
    bank_fixture: BankFixture,
) -> None:
    """Import creates new question + option rows and stamps imported_from."""
    async with session_factory() as session:
        cloned = await bank_service.import_questions(
            session,
            target_quiz_id=bank_fixture.target_quiz_id,
            source_question_ids=[bank_fixture.approved_question_id],
            actor=bank_fixture.actor,
        )
        await session.commit()
        clone_id = cloned[0].id
        clone_options_count = (
            await session.execute(
                select(QuizQuestionOption).where(
                    QuizQuestionOption.question_id == clone_id
                )
            )
        ).scalars().all()
        clone_row = (
            await session.execute(
                select(QuizQuestion).where(QuizQuestion.id == clone_id)
            )
        ).scalar_one()

    assert clone_id != bank_fixture.approved_question_id
    assert clone_row.quiz_id == bank_fixture.target_quiz_id
    assert clone_row.imported_from_question_id == bank_fixture.approved_question_id
    assert clone_row.review_status == "pending"
    assert clone_row.reviewed_by is None
    assert clone_row.position == 1
    # Options copied across with fresh ids
    assert len(clone_options_count) == 2
    assert {opt.option_key for opt in clone_options_count} == {"A", "B"}
    correct = [opt for opt in clone_options_count if opt.is_correct]
    assert len(correct) == 1
    assert correct[0].option_key == "A"


async def test_import_rejects_cross_course_sources(
    session_factory: async_sessionmaker[AsyncSession],
    bank_fixture: BankFixture,
) -> None:
    """Importing a question from a foreign course raises AppError."""
    from abridgeai.core.exceptions import AppError

    async with session_factory() as session:
        with pytest.raises(AppError, match="course boundaries"):
            await bank_service.import_questions(
                session,
                target_quiz_id=bank_fixture.target_quiz_id,
                source_question_ids=[bank_fixture.other_course_question_id],
                actor=bank_fixture.actor,
            )


async def test_duplicate_question_clones_in_place_as_pending(
    session_factory: async_sessionmaker[AsyncSession],
    bank_fixture: BankFixture,
) -> None:
    """duplicate_question copies a question into its OWN quiz, forced pending.

    Distinct from import_questions (which targets a different quiz): the clone
    stays in the source quiz, appends after the last position, resets review
    state to ``pending``, and copies the option set with fresh ids.
    """
    async with session_factory() as session:
        clone = await bank_service.duplicate_question(
            session,
            question_id=bank_fixture.approved_question_id,
            actor=bank_fixture.actor,
        )
        await session.commit()
        clone_id = clone.id
        clone_row = (
            await session.execute(select(QuizQuestion).where(QuizQuestion.id == clone_id))
        ).scalar_one()
        clone_opts = (
            (
                await session.execute(
                    select(QuizQuestionOption).where(QuizQuestionOption.question_id == clone_id)
                )
            )
            .scalars()
            .all()
        )

    assert clone_id != bank_fixture.approved_question_id
    # Same quiz — this is an in-place duplicate, not a cross-quiz import.
    assert clone_row.quiz_id == bank_fixture.source_quiz_id
    assert clone_row.imported_from_question_id == bank_fixture.approved_question_id
    # Always unpublished/unvetted regardless of the source's approved state.
    assert clone_row.review_status == "pending"
    assert clone_row.reviewed_by is None
    assert clone_row.reviewed_at is None
    assert clone_row.published_at is None
    # Options copied with fresh ids.
    assert {opt.option_key for opt in clone_opts} == {"A", "B"}
    assert all(opt.question_id == clone_id for opt in clone_opts)


async def test_duplicate_question_missing_raises_not_found(
    session_factory: async_sessionmaker[AsyncSession],
    bank_fixture: BankFixture,
) -> None:
    """Duplicating a non-existent question raises NotFoundError."""
    from abridgeai.core.exceptions import NotFoundError

    async with session_factory() as session:
        with pytest.raises(NotFoundError):
            await bank_service.duplicate_question(
                session,
                question_id=uuid.uuid4(),
                actor=bank_fixture.actor,
            )


async def test_legacy_import_rejects_published_target(
    session_factory: async_sessionmaker[AsyncSession],
    bank_fixture: BankFixture,
) -> None:
    """The API guard is backed by a service invariant, not only a hidden FE button."""
    from abridgeai.core.exceptions import ConflictError

    async with session_factory() as session:
        await session.execute(
            text("UPDATE quizzes SET status = 'published' WHERE id = :quiz_id"),
            {"quiz_id": bank_fixture.target_quiz_id},
        )
        with pytest.raises(ConflictError, match="quiz_published_readonly"):
            await bank_service.import_questions(
                session,
                target_quiz_id=bank_fixture.target_quiz_id,
                source_question_ids=[bank_fixture.approved_question_id],
                actor=bank_fixture.actor,
            )


async def test_legacy_import_preserves_matching_answer_content(
    session_factory: async_sessionmaker[AsyncSession],
    bank_fixture: BankFixture,
) -> None:
    """Expanded question types keep their hidden answer key during a deep copy."""
    question_id = uuid.uuid4()
    pairs = '[{"left":"A","right":"1"},{"left":"B","right":"2"}]'
    async with session_factory() as session:
        await session.execute(
            text(
                "INSERT INTO quiz_questions ("
                "id, quiz_id, position, question_type, prompt_text, review_status, "
                "difficulty, bloom_level, source_refs, match_pairs, match_distractors, "
                "prompt_format, hint_format, explanation_format) VALUES ("
                ":id, :quiz_id, 3, 'matching', 'Match these', 'approved', "
                "'hard', 'analyze', '[]'::jsonb, CAST(:pairs AS jsonb), "
                "'[\"3\"]'::jsonb, 'markdown', 'html', 'markdown')"
            ),
            {
                "id": question_id,
                "quiz_id": bank_fixture.source_quiz_id,
                "pairs": pairs,
            },
        )
        created = await bank_service.import_questions(
            session,
            target_quiz_id=bank_fixture.target_quiz_id,
            source_question_ids=[question_id],
            actor=bank_fixture.actor,
        )
        clone = created[0]
        assert clone.match_pairs == [
            {"left": "A", "right": "1"},
            {"left": "B", "right": "2"},
        ]
        assert clone.match_distractors == ["3"]
        assert clone.prompt_format == "markdown"
        assert clone.hint_format == "html"
        assert clone.explanation_format == "markdown"


async def test_curated_bank_snapshot_diverges_from_source_and_imports_with_bank_lineage(
    session_factory: async_sessionmaker[AsyncSession],
    bank_fixture: BankFixture,
) -> None:
    """Bank content is a durable snapshot; source edits do not rewrite it."""
    async with session_factory() as session:
        bank_items = await curated_bank_service.copy_questions_to_curated_bank(
            session,
            course_id=bank_fixture.course_id,
            question_ids=[bank_fixture.approved_question_id],
            actor=bank_fixture.actor,
        )
        bank_item_id = bank_items[0].id
        assert bank_items[0].status == "approved"

        await session.execute(
            text("UPDATE quiz_questions SET prompt_text = 'Source changed' WHERE id = :id"),
            {"id": bank_fixture.approved_question_id},
        )
        imported = await curated_bank_service.import_curated_bank_items(
            session,
            target_quiz_id=bank_fixture.target_quiz_id,
            item_ids=[bank_item_id],
            actor=bank_fixture.actor,
        )
        await session.commit()

        clone = imported[0]
        assert clone.prompt_text == "Approved stem?"
        assert clone.imported_from_bank_item_id == bank_item_id
        assert clone.imported_from_question_id is None
        stored_bank = await session.get(QuizQuestionBankItem, bank_item_id)
        assert stored_bank is not None
        assert stored_bank.prompt_text == "Approved stem?"


async def test_curated_bank_manual_lifecycle_and_option_replacement(
    session_factory: async_sessionmaker[AsyncSession],
    bank_fixture: BankFixture,
) -> None:
    """Manual draft → edit → approve is validated and serializable."""
    async with session_factory() as session:
        item = await curated_bank_service.create_curated_bank_item(
            session,
            course_id=bank_fixture.course_id,
            actor=bank_fixture.actor,
            payload=QuizQuestionBankItemCreate(
                question_type="multiple_choice",
                prompt_text="Which value is correct?",
                options=[
                    QuizQuestionBankOptionCreate(
                        option_key="A",
                        option_text="One",
                        is_correct=True,
                        position=1,
                    ),
                    QuizQuestionBankOptionCreate(
                        option_key="B",
                        option_text="Two",
                        is_correct=False,
                        position=2,
                    ),
                ],
            ),
        )
        assert item.status == "draft"
        item = await curated_bank_service.update_curated_bank_item(
            session,
            course_id=bank_fixture.course_id,
            item_id=item.id,
            actor=bank_fixture.actor,
            payload=QuizQuestionBankItemUpdate(
                prompt_text="Which value is definitely correct?",
                options=[
                    QuizQuestionBankOptionCreate(
                        option_key="A",
                        option_text="First",
                        is_correct=False,
                        position=1,
                    ),
                    QuizQuestionBankOptionCreate(
                        option_key="B",
                        option_text="Second",
                        is_correct=True,
                        position=2,
                    ),
                ],
            ),
        )
        item = await curated_bank_service.set_curated_bank_item_status(
            session,
            course_id=bank_fixture.course_id,
            item_id=item.id,
            status="approved",
            actor=bank_fixture.actor,
        )
        view = QuizQuestionBankItemRead.model_validate(item)
        assert view.status == "approved"
        assert view.prompt_text == "Which value is definitely correct?"
        assert [option.option_text for option in view.options] == ["First", "Second"]
        assert view.options[1].is_correct is True


async def test_curated_bank_import_rejects_draft_item(
    session_factory: async_sessionmaker[AsyncSession],
    bank_fixture: BankFixture,
) -> None:
    from abridgeai.core.exceptions import NotFoundError

    async with session_factory() as session:
        item = await curated_bank_service.create_curated_bank_item(
            session,
            course_id=bank_fixture.course_id,
            actor=bank_fixture.actor,
            payload=QuizQuestionBankItemCreate(
                question_type="short_answer",
                prompt_text="Draft-only question",
            ),
        )
        with pytest.raises(NotFoundError, match="approved bank items"):
            await curated_bank_service.import_curated_bank_items(
                session,
                target_quiz_id=bank_fixture.target_quiz_id,
                item_ids=[item.id],
                actor=bank_fixture.actor,
            )
