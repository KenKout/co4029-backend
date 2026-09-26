"""Integration tests for ``features.quizzes.services.authoring`` (T5.13).

Covers the create-quiz happy path, the start_generation_run enqueue
contract, and the authoring write paths that decide a quiz's answer key
and when it may still be changed: question creation and its revision
trail, option synchronisation, and the freeze a published quiz imposes.

The payloads are built with the router's own ``_AttrShim`` rather than a
local stand-in. That shim is what production hands these services, and
the service carries code (``_as_plain_json``) whose only reason to exist
is unwrapping it -- a plainer test payload would exercise a shape no
caller sends.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import (
    Column,
    Table,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.access_control.models  # noqa: F401  -- register users/orgs FK targets
import abridgeai.features.courses.models  # noqa: F401  -- register courses/modules/lessons FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
import abridgeai.features.interviews.models  # noqa: F401  -- T6.1 registers interview_* tables
from abridgeai.ai.models import GenerationRun
from abridgeai.core.audit import audit_maintenance
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base
from abridgeai.core.exceptions import AppError, ConflictError
from abridgeai.core.security import CurrentUser
from abridgeai.features.quizzes.models import Quiz, QuizQuestion
from abridgeai.features.quizzes.routers.authoring import _AttrShim
from abridgeai.features.quizzes.schemas import (
    CoverageOptions,
    QuizGenerationRequest,
)
from abridgeai.features.quizzes.services import authoring as authoring_service

for _stub_name in ("interview_configs", "learning_materials", "learning_material_versions"):
    if _stub_name not in Base.metadata.tables:
        Table(
            _stub_name,
            Base.metadata,
            Column("id", PGUUID(as_uuid=True), primary_key=True),
        )


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
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
    _ensure_head()
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def scenario(engine: AsyncEngine) -> AsyncIterator[dict]:
    org_id = uuid.uuid4()
    owner_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    suffix = org_id.hex[:8]

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"qa-{suffix}", "name": "Auth Test Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": owner_id, "email": f"qa-{suffix}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'Auth Course', 'draft')"
            ),
            {
                "id": course_id,
                "org": org_id,
                "owner": owner_id,
                "slug": f"course-{suffix}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :course, 'Module', 1, 'draft')"
            ),
            {"id": module_id, "course": course_id},
        )

    yield {
        "owner_id": owner_id,
        "org_id": org_id,
        "course_id": course_id,
        "module_id": module_id,
    }

    async with engine.begin() as conn:
        await audit_maintenance(conn)
        await conn.execute(
            text(
                "DELETE FROM quiz_audit_events WHERE quiz_id IN "
                "(SELECT id FROM quizzes WHERE module_id = :m)"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text("DELETE FROM generation_runs WHERE module_id = :m"),
            {"m": module_id},
        )
        await conn.execute(
            text("DELETE FROM module_items WHERE module_id = :m"),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_source_lessons WHERE quiz_id IN "
                "(SELECT id FROM quizzes WHERE module_id = :m)"
            ),
            {"m": module_id},
        )
        # Question rows FK back to quizzes, so they have to go first or the
        # quiz delete below fails for any test that authored a question.
        await conn.execute(
            text(
                "DELETE FROM quiz_question_revisions WHERE question_id IN "
                "(SELECT id FROM quiz_questions WHERE quiz_id IN "
                "(SELECT id FROM quizzes WHERE module_id = :m))"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_question_options WHERE question_id IN "
                "(SELECT id FROM quiz_questions WHERE quiz_id IN "
                "(SELECT id FROM quizzes WHERE module_id = :m))"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_questions WHERE quiz_id IN "
                "(SELECT id FROM quizzes WHERE module_id = :m)"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text("DELETE FROM quizzes WHERE module_id = :m"),
            {"m": module_id},
        )
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": owner_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


def _actor(user_id: uuid.UUID) -> CurrentUser:
    return CurrentUser(user_id=user_id, session_id=uuid.uuid4())


class _CreatePayload:
    def __init__(self, **fields: object) -> None:
        self._fields = fields

    def model_dump(self, exclude_unset: bool = False) -> dict:
        del exclude_unset
        return dict(self._fields)


@pytest.mark.asyncio
async def test_create_quiz_inserts_row_and_links_module_item(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    payload = _CreatePayload(
        title="Photosynthesis Quiz",
        description="Auto-graded MCQs",
        time_limit_seconds=600,
    )

    async with session_factory() as session, session.begin():
        quiz = await authoring_service.create_quiz(
            session, scenario["module_id"], payload, _actor(scenario["owner_id"])
        )

    assert isinstance(quiz, Quiz)
    assert quiz.title == "Photosynthesis Quiz"
    assert quiz.module_id == scenario["module_id"]
    assert quiz.course_id == scenario["course_id"]

    async with session_factory() as session:
        rows = (
            (
                await session.execute(
                    text("SELECT item_type, quiz_id FROM module_items WHERE module_id = :m"),
                    {"m": scenario["module_id"]},
                )
            )
            .mappings()
            .all()
        )
    assert len(rows) == 1
    assert rows[0]["item_type"] == "quiz"
    assert rows[0]["quiz_id"] == quiz.id


@pytest.mark.asyncio
async def test_replace_enqueue_failure_preserves_existing_questions(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """Replace mode must not wipe the draft before ARQ accepts the run."""
    async with session_factory() as session:
        quiz = await _make_quiz(session, scenario, title="Replace Safety Quiz")
        question = await _approved_question(session, quiz.id, _actor(scenario["owner_id"]))
        await session.commit()
        question_id = question.id
        quiz_id = quiz.id

    payload = QuizGenerationRequest(
        title="Replace Safety Quiz",
        quiz_id=quiz_id,
        append=False,
        generation_mode="topic",
    )
    arq_pool = SimpleNamespace(
        enqueue_job=AsyncMock(side_effect=ConnectionError("redis unavailable"))
    )

    with pytest.raises(ConnectionError):
        async with session_factory() as session:
            await authoring_service.start_generation_run(
                session,
                scenario["module_id"],
                payload,
                _actor(scenario["owner_id"]),
                arq_pool=arq_pool,
            )

    async with session_factory() as session:
        remaining = await session.get(QuizQuestion, question_id)
        assert remaining is not None
        assert remaining.quiz_id == quiz_id


@pytest.mark.asyncio
async def test_replace_generation_failure_preserves_existing_questions(
    monkeypatch: pytest.MonkeyPatch,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """A worker/LLM failure must roll back the staged replacement."""
    from abridgeai.features.quizzes.services import generation as generation_service

    async with session_factory() as session:
        quiz = await _make_quiz(session, scenario, title="AI Failure Safety Quiz")
        question = await _approved_question(session, quiz.id, _actor(scenario["owner_id"]))
        await session.commit()
        question_id = question.id
        quiz_id = quiz.id

    payload = QuizGenerationRequest(
        title="AI Failure Safety Quiz",
        quiz_id=quiz_id,
        append=False,
        generation_mode="topic",
    )
    async with session_factory() as session:
        run = await authoring_service.start_generation_run(
            session,
            scenario["module_id"],
            payload,
            _actor(scenario["owner_id"]),
            arq_pool=None,
        )
        run_id = run.id

    async def _fail_pipeline(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("LLM generation failed")

    monkeypatch.setattr(generation_service.full_pipeline, "run_full_pipeline", _fail_pipeline)
    with pytest.raises(RuntimeError, match="LLM generation failed"):
        async with session_factory() as session:
            await generation_service.run_quiz_generation(session, run_id)

    async with session_factory() as session:
        remaining = await session.get(QuizQuestion, question_id)
        assert remaining is not None
        assert remaining.quiz_id == quiz_id

@pytest.mark.asyncio
async def test_replace_success_swaps_questions_after_generation(
    monkeypatch: pytest.MonkeyPatch,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """Successful Replace commits only the new question graph."""
    from abridgeai.features.quizzes.ai.stages.persistence import persist_questions
    from abridgeai.features.quizzes.services import generation as generation_service

    async with session_factory() as session:
        quiz = await _make_quiz(session, scenario, title="Replace Success Quiz")
        old_question = await _approved_question(session, quiz.id, _actor(scenario["owner_id"]))
        await session.commit()
        old_question_id = old_question.id
        quiz_id = quiz.id

    payload = QuizGenerationRequest(
        title="Replace Success Quiz",
        quiz_id=quiz_id,
        append=False,
        generation_mode="topic",
    )
    async with session_factory() as session:
        run = await authoring_service.start_generation_run(
            session,
            scenario["module_id"],
            payload,
            _actor(scenario["owner_id"]),
            arq_pool=None,
        )
        run_id = run.id

    async def _success_pipeline(
        db: AsyncSession,
        run: GenerationRun,
        quiz: Quiz,
        **kwargs: object,
    ) -> list[QuizQuestion]:
        del kwargs
        return await persist_questions(
            db,
            run,
            quiz,
            [],
            [
                {
                    "question_type": "multiple_choice",
                    "prompt_text": "New generated question",
                    "expected_response_time_ms": 30_000,
                    "source_refs": [],
                    "options": [],
                    "original_generated_payload": {},
                }
            ],
        )

    monkeypatch.setattr(generation_service.full_pipeline, "run_full_pipeline", _success_pipeline)
    async with session_factory() as session:
        await generation_service.run_quiz_generation(session, run_id)

    async with session_factory() as session:
        old = await session.get(QuizQuestion, old_question_id)
        rows = list(
            (
                await session.scalars(
                    select(QuizQuestion)
                    .where(QuizQuestion.quiz_id == quiz_id)
                    .order_by(QuizQuestion.position)
                )
            ).all()
        )
        assert old is None
        assert len(rows) == 1
        assert rows[0].prompt_text == "New generated question"
        assert rows[0].position == 1

@pytest.mark.asyncio
async def test_start_generation_run_creates_run_quiz_and_enqueues_job(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    arq_pool = SimpleNamespace(enqueue_job=AsyncMock(return_value=None))
    payload = QuizGenerationRequest(
        title="AI-Generated Quiz",
        description="Created by start_generation_run",
        question_count=5,
        question_types=["multiple_choice"],
        difficulty="medium",
        bloom_distribution={"understand": 1},
        include_prerequisites=True,
        generation_mode="topic",
    )

    async with session_factory() as session:
        run = await authoring_service.start_generation_run(
            session,
            scenario["module_id"],
            payload,
            _actor(scenario["owner_id"]),
            arq_pool=arq_pool,
        )

    assert isinstance(run, GenerationRun)
    assert run.status == "pending"
    assert run.config_json["quiz_id"]
    assert run.module_id == scenario["module_id"]
    # Phase 2 of FR-5: every structured field lands in
    # ``GenerationRun.config_json`` verbatim. Verifies the service
    # layer reads them via direct attribute access (no getattr).
    cfg = run.config_json
    assert cfg["question_count"] == 5
    assert cfg["question_types"] == ["multiple_choice"]
    assert cfg["difficulty"] == "medium"
    assert cfg["bloom_distribution"] == {"understand": 1}
    assert cfg["include_prerequisites"] is True
    assert cfg["generation_mode"] == "topic"
    assert cfg["focus_topics"] == []
    assert cfg["avoid_topics"] == []
    assert cfg["coverage_options"] is None
    assert cfg["append"] is False
    arq_pool.enqueue_job.assert_awaited_once()
    invocation = arq_pool.enqueue_job.await_args
    assert invocation.args[0] == "run_quiz_generation_task"
    assert invocation.args[1] == scenario["owner_id"]
    assert invocation.args[2] == run.id


@pytest.mark.asyncio
async def test_start_generation_run_persists_full_fr5_payload(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """Full FR-5 coverage payload survives the service layer with all
    nested structured fields preserved in ``config_json``."""
    arq_pool = SimpleNamespace(enqueue_job=AsyncMock(return_value=None))
    payload = QuizGenerationRequest(
        title="Coverage Run",
        question_count=8,
        question_types=["multiple_choice", "short_answer"],
        difficulty="mixed",
        bloom_distribution={"remember": 2, "understand": 3, "apply": 3},
        generation_mode="coverage",
        focus_topics=["matrices", "vectors"],
        avoid_topics=["geometry"],
        extra_instructions="Avoid trick questions.",
        coverage_options=CoverageOptions(
            min_per_section=1,
            max_per_section=3,
            skip_summaries=True,
            slides_per_section=4,
            parallelism=8,
        ),
    )

    async with session_factory() as session:
        run = await authoring_service.start_generation_run(
            session,
            scenario["module_id"],
            payload,
            _actor(scenario["owner_id"]),
            arq_pool=arq_pool,
        )

    cfg = run.config_json
    assert cfg["generation_mode"] == "coverage"
    assert cfg["focus_topics"] == ["matrices", "vectors"]
    assert cfg["avoid_topics"] == ["geometry"]
    assert cfg["extra_instructions"] == "Avoid trick questions."
    # Coverage options serialise to dict (not the Pydantic model) so
    # the ARQ worker can JSON-roundtrip via ``GenerationRun.config_json``.
    assert isinstance(cfg["coverage_options"], dict)
    assert cfg["coverage_options"]["min_per_section"] == 1
    assert cfg["coverage_options"]["max_per_section"] == 3
    assert cfg["coverage_options"]["parallelism"] == 8


@pytest.mark.asyncio
async def test_start_generation_run_no_arq_skips_enqueue(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """When ``arq_pool=None`` the run is persisted but no job is
    enqueued — used by tests and by the dev-mode synchronous path."""
    payload = QuizGenerationRequest(title="No-ARQ Run")

    async with session_factory() as session:
        run = await authoring_service.start_generation_run(
            session,
            scenario["module_id"],
            payload,
            _actor(scenario["owner_id"]),
            arq_pool=None,
        )

    assert run.status == "pending"
    assert run.config_json["quiz_id"]


def _question_payload(**fields: object) -> _AttrShim:
    """A question body as the router would deliver it."""
    body: dict = {
        "prompt_text": "Which pigment absorbs light?",
        "question_type": "multiple_choice",
    }
    body.update(fields)
    return _AttrShim(body)


def _options(*triples: tuple[str, str, bool]) -> list[dict]:
    return [
        {"option_key": key, "option_text": body, "is_correct": correct}
        for key, body, correct in triples
    ]


_MCQ_OPTIONS = _options(("A", "Chlorophyll", True), ("B", "Keratin", False))


async def _make_quiz(session: AsyncSession, scenario: dict, title: str = "Draft Quiz") -> Quiz:
    return await authoring_service.create_quiz(
        session,
        scenario["module_id"],
        _AttrShim({"title": title}),
        _actor(scenario["owner_id"]),
    )


async def _approved_question(
    session: AsyncSession, quiz_id: uuid.UUID, actor: CurrentUser
) -> object:
    """A question that satisfies both publish gates: approved and timed."""
    return await authoring_service.create_question(
        session,
        quiz_id,
        _question_payload(
            options=_MCQ_OPTIONS,
            review_status="approved",
            expected_response_time_ms=30000,
        ),
        actor,
    )


async def _option_rows(session: AsyncSession, question_id: uuid.UUID) -> list[dict]:
    rows = (
        (
            await session.execute(
                text(
                    "SELECT option_key, option_text, is_correct, position "
                    "FROM quiz_question_options WHERE question_id = :q ORDER BY position"
                ),
                {"q": question_id},
            )
        )
        .mappings()
        .all()
    )
    return [dict(row) for row in rows]


@pytest.mark.asyncio
async def test_question_creation_numbers_positions_and_opens_a_revision_trail(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """Every question is born at the end of the quiz with revision 1.

    The revision row is the audit trail for AI-drafted content: it records
    what the teacher accepted, which is the only evidence of what a question
    looked like before a later edit.
    """
    async with session_factory() as session, session.begin():
        quiz = await _make_quiz(session, scenario)
        first = await authoring_service.create_question(
            session,
            quiz.id,
            _question_payload(options=_MCQ_OPTIONS),
            _actor(scenario["owner_id"]),
        )
        second = await authoring_service.create_question(
            session,
            quiz.id,
            _question_payload(prompt_text="And which does not?", options=_MCQ_OPTIONS),
            _actor(scenario["owner_id"]),
        )

    assert [first.position, second.position] == [1, 2]
    assert first.review_status == "pending", "a manually authored question still awaits sign-off"
    assert first.reviewed_by is None

    async with session_factory() as session:
        options = await _option_rows(session, first.id)
        revisions = (
            (
                await session.execute(
                    text(
                        "SELECT revision_no, source_kind FROM quiz_question_revisions "
                        "WHERE question_id = :q ORDER BY revision_no"
                    ),
                    {"q": first.id},
                )
            )
            .mappings()
            .all()
        )

    assert [(o["option_key"], o["position"]) for o in options] == [("A", 1), ("B", 2)]
    assert [o["is_correct"] for o in options] == [True, False]
    assert [(r["revision_no"], r["source_kind"]) for r in revisions] == [(1, "teacher")]


@pytest.mark.asyncio
async def test_approving_on_create_stamps_the_reviewer(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """A teacher writing a question themselves may sign it off in one step.

    The sign-off still has to be attributed: ``approved`` with nobody named
    would leave the publish gate unable to say who vouched for it.
    """
    async with session_factory() as session, session.begin():
        quiz = await _make_quiz(session, scenario)
        question = await authoring_service.create_question(
            session,
            quiz.id,
            _question_payload(options=_MCQ_OPTIONS, review_status="approved"),
            _actor(scenario["owner_id"]),
        )

    assert question.review_status == "approved"
    assert question.reviewed_by == scenario["owner_id"]
    assert question.reviewed_at is not None


@pytest.mark.asyncio
async def test_question_create_and_update_reject_zero_expected_time(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """Loose authoring dictionaries cannot bypass the positive T_exp invariant."""
    async with session_factory() as session, session.begin():
        quiz = await _make_quiz(session, scenario)
        with pytest.raises(AppError, match="positive integer"):
            await authoring_service.create_question(
                session,
                quiz.id,
                _question_payload(
                    options=_MCQ_OPTIONS,
                    expected_response_time_ms=0,
                ),
                _actor(scenario["owner_id"]),
            )
        question = await authoring_service.create_question(
            session,
            quiz.id,
            _question_payload(
                options=_MCQ_OPTIONS,
                expected_response_time_ms=30_000,
            ),
            _actor(scenario["owner_id"]),
        )

    async with session_factory() as session:
        with pytest.raises(AppError, match="positive integer"):
            await authoring_service.update_question(
                session,
                question.id,
                _AttrShim({"expected_response_time_ms": 0}),
                _actor(scenario["owner_id"]),
            )

    async with session_factory() as session:
        stored = await session.get(QuizQuestion, question.id)
        assert stored is not None
        assert stored.expected_response_time_ms == 30_000


@pytest.mark.asyncio
async def test_an_option_edit_cannot_leave_two_correct_answers(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """The single-answer invariant is re-checked on every option edit.

    A single-answer question with two keys marked correct is not a harder
    question -- it is one the grader scores inconsistently, since only one of
    the two can be the key a student is credited for.
    """
    async with session_factory() as session, session.begin():
        quiz = await _make_quiz(session, scenario)
        question = await authoring_service.create_question(
            session,
            quiz.id,
            _question_payload(options=_MCQ_OPTIONS),
            _actor(scenario["owner_id"]),
        )

    async with session_factory() as session:
        with pytest.raises(AppError, match="exactly one correct option"):
            await authoring_service.update_question(
                session,
                question.id,
                _AttrShim(
                    {"options": _options(("A", "Chlorophyll", True), ("B", "Keratin", True))}
                ),
                _actor(scenario["owner_id"]),
            )

    async with session_factory() as session:
        stored = await _option_rows(session, question.id)
    assert [o["is_correct"] for o in stored] == [True, False], "the refusal left nothing behind"


@pytest.mark.asyncio
async def test_an_option_key_the_question_does_not_have_is_refused(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """Option edits address existing rows by id or key; they do not insert.

    Silently ignoring an unknown key would accept the teacher's edit and
    leave the old answer key in place -- the question would read as corrected
    in the editor and grade as it did before.
    """
    async with session_factory() as session, session.begin():
        quiz = await _make_quiz(session, scenario)
        question = await authoring_service.create_question(
            session,
            quiz.id,
            _question_payload(options=_MCQ_OPTIONS),
            _actor(scenario["owner_id"]),
        )

    async with session_factory() as session:
        with pytest.raises(AppError, match="option not found"):
            await authoring_service.update_question(
                session,
                question.id,
                _AttrShim({"options": _options(("A", "Chlorophyll", False), ("Z", "New", True))}),
                _actor(scenario["owner_id"]),
            )


@pytest.mark.asyncio
async def test_an_option_edit_moves_the_answer_key_and_appends_a_revision(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    async with session_factory() as session, session.begin():
        quiz = await _make_quiz(session, scenario)
        question = await authoring_service.create_question(
            session,
            quiz.id,
            _question_payload(options=_MCQ_OPTIONS),
            _actor(scenario["owner_id"]),
        )

    async with session_factory() as session, session.begin():
        await authoring_service.update_question(
            session,
            question.id,
            _AttrShim({"options": _options(("A", "Chlorophyll", False), ("B", "Keratin", True))}),
            _actor(scenario["owner_id"]),
        )

    async with session_factory() as session:
        options = await _option_rows(session, question.id)
        revisions = (
            await session.execute(
                text("SELECT count(*) FROM quiz_question_revisions WHERE question_id = :q"),
                {"q": question.id},
            )
        ).scalar_one()

    assert [(o["option_key"], o["is_correct"]) for o in options] == [("A", False), ("B", True)]
    assert revisions == 2, "the edit is recorded beside the original"


@pytest.mark.asyncio
async def test_the_fill_blank_word_bank_is_replaced_wholesale(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """Unlike multiple choice, a word bank is swapped rather than patched.

    Bank entries are addressed by their text, never by id -- the grader reads
    the stored answer and the learner drags the words -- so a teacher may add
    and remove distractors freely, and the positions must come out dense.
    """
    bank = _options(
        ("O01", "photosynthesis", True),
        ("O02", "respiration", False),
        ("O03", "osmosis", False),
    )
    async with session_factory() as session, session.begin():
        quiz = await _make_quiz(session, scenario)
        question = await authoring_service.create_question(
            session,
            quiz.id,
            _question_payload(question_type="fill_blank", options=bank),
            _actor(scenario["owner_id"]),
        )

    async with session_factory() as session, session.begin():
        await authoring_service.update_question(
            session,
            question.id,
            _AttrShim(
                {
                    "options": [
                        {"option_text": "photosynthesis", "is_correct": True},
                        {"option_text": "diffusion", "is_correct": False},
                    ]
                }
            ),
            _actor(scenario["owner_id"]),
        )

    async with session_factory() as session:
        options = await _option_rows(session, question.id)

    assert [o["option_text"] for o in options] == ["photosynthesis", "diffusion"]
    assert [o["position"] for o in options] == [1, 2], "no gap where the dropped entry sat"
    assert [o["option_key"] for o in options] == ["O01", "O02"], (
        "the replacement path derives keys from position when the payload omits them"
    )


@pytest.mark.asyncio
async def test_a_word_bank_without_a_distractor_is_refused(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """A bank holding only the answers is not an exercise."""
    async with session_factory() as session:
        quiz = await _make_quiz(session, scenario)
        with pytest.raises(AppError, match="distractor"):
            await authoring_service.create_question(
                session,
                quiz.id,
                _question_payload(
                    question_type="fill_blank",
                    options=_options(("O01", "photosynthesis", True)),
                ),
                _actor(scenario["owner_id"]),
            )


@pytest.mark.asyncio
async def test_publishing_freezes_every_content_path(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """Students can see and attempt a published quiz, so its questions are
    fully frozen -- adding, editing, approving or removing one would change
    what an attempt in flight is being graded against.
    """
    actor = _actor(scenario["owner_id"])
    async with session_factory() as session, session.begin():
        quiz = await _make_quiz(session, scenario)
        question = await _approved_question(session, quiz.id, actor)
        await authoring_service.publish_quiz(session, quiz.id, actor)

    async with session_factory() as session:
        with pytest.raises(ConflictError, match="quiz_published_readonly"):
            await authoring_service.create_question(
                session, quiz.id, _question_payload(options=_MCQ_OPTIONS), actor
            )
    async with session_factory() as session:
        with pytest.raises(ConflictError, match="quiz_published_readonly"):
            await authoring_service.update_question(
                session, question.id, _AttrShim({"prompt_text": "Reworded"}), actor
            )
    async with session_factory() as session:
        with pytest.raises(ConflictError, match="quiz_published_readonly"):
            await authoring_service.delete_question(session, question.id, actor)
    async with session_factory() as session:
        with pytest.raises(ConflictError, match="quiz_published_readonly"):
            await authoring_service.bulk_approve_questions(session, quiz.id, [question.id], actor)


@pytest.mark.asyncio
async def test_archiving_reopens_a_published_quiz_for_editing(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """Archived quizzes are immutable and cannot be edited or republished."""
    actor = _actor(scenario["owner_id"])
    async with session_factory() as session, session.begin():
        quiz = await _make_quiz(session, scenario)
        await _approved_question(session, quiz.id, actor)
        await authoring_service.publish_quiz(session, quiz.id, actor)

    async with session_factory() as session, session.begin():
        archived = await authoring_service.archive_quiz(session, quiz.id, actor)
        assert archived.status == "archived"
        with pytest.raises(ConflictError, match="quiz.*cannot be edited"):
            await authoring_service.create_question(
                session, quiz.id, _question_payload(options=_MCQ_OPTIONS), actor
            )


@pytest.mark.asyncio
async def test_an_archived_quiz_cannot_be_published_again(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """Archive ends the quiz's life; re-publishing would resurrect it under
    students who were told it had closed.
    """
    actor = _actor(scenario["owner_id"])
    async with session_factory() as session, session.begin():
        quiz = await _make_quiz(session, scenario)
        await authoring_service.archive_quiz(session, quiz.id, actor)

    async with session_factory() as session:
        with pytest.raises(AppError, match="Cannot publish archived quiz"):
            await authoring_service.publish_quiz(session, quiz.id, actor)


@pytest.mark.asyncio
async def test_the_settings_freeze_is_field_aware_on_a_published_quiz(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """Renaming a live quiz or extending its deadline is safe; changing what
    a mark is worth is not. The whitelist is what separates them, and it is
    applied per PATCH: one frozen field in an otherwise safe body refuses the
    whole request rather than saving the rest.
    """
    actor = _actor(scenario["owner_id"])
    async with session_factory() as session, session.begin():
        quiz = await _make_quiz(session, scenario)
        await _approved_question(session, quiz.id, actor)
        await authoring_service.publish_quiz(session, quiz.id, actor)

    async with session_factory() as session, session.begin():
        renamed = await authoring_service.update_quiz(
            session,
            quiz.id,
            _AttrShim({"title": "Photosynthesis (Week 2)", "due_at": "2026-10-01T09:00:00Z"}),
            actor,
        )
    assert renamed.title == "Photosynthesis (Week 2)"
    assert renamed.due_at is not None
    assert renamed.due_at.tzinfo is not None, "the ISO string carried a zone and must keep it"

    async with session_factory() as session:
        with pytest.raises(ConflictError, match="passing_score_percent"):
            await authoring_service.update_quiz(
                session, quiz.id, _AttrShim({"passing_score_percent": 40}), actor
            )

    async with session_factory() as session:
        with pytest.raises(ConflictError, match="quiz_published_setting_locked"):
            await authoring_service.update_quiz(
                session,
                quiz.id,
                _AttrShim({"title": "Also renamed", "shuffle_questions": True}),
                actor,
            )


@pytest.mark.asyncio
async def test_quiz_slugs_are_numbered_within_their_module(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """Two quizzes may share a title; their URLs may not.

    Uniqueness is scoped to the module and backed by a unique index, so a
    collision that reached the flush would surface to the teacher as a 500
    rather than as a second quiz.
    """
    async with session_factory() as session, session.begin():
        first = await _make_quiz(session, scenario, title="Week 1 Check")
        second = await _make_quiz(session, scenario, title="Week 1 Check")
        third = await _make_quiz(session, scenario, title="Week 1 Check")

    assert first.slug == "week-1-check"
    assert second.slug == "week-1-check-1"
    assert third.slug == "week-1-check-2", "the suffix counts up rather than restarting"


@pytest.mark.asyncio
async def test_publishing_does_not_add_a_second_module_item(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """Both create and publish ensure the curriculum entry exists.

    The second call has to be a no-op: a duplicate item would show the same
    quiz twice in the course content tree, and a student finishing one copy
    would leave the other outstanding.
    """
    actor = _actor(scenario["owner_id"])
    async with session_factory() as session, session.begin():
        quiz = await _make_quiz(session, scenario)
        await _approved_question(session, quiz.id, actor)
        await authoring_service.publish_quiz(session, quiz.id, actor)

    async with session_factory() as session:
        count = (
            await session.execute(
                text(
                    "SELECT count(*) FROM module_items "
                    "WHERE quiz_id = :q AND deleted_at IS NULL"
                ),
                {"q": quiz.id},
            )
        ).scalar_one()
    assert count == 1


@pytest.mark.asyncio
async def test_bulk_approve_only_touches_questions_of_the_named_quiz(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """The ids arrive in the request body while the quiz comes from the URL.

    Approving by id alone would let a request authorised against one quiz
    approve a question belonging to another -- past a reviewer who never saw
    it, and into a publish gate that only counts approvals.
    """
    actor = _actor(scenario["owner_id"])
    async with session_factory() as session, session.begin():
        quiz = await _make_quiz(session, scenario, title="Reviewed Quiz")
        other = await _make_quiz(session, scenario, title="Another Quiz")
        mine = await authoring_service.create_question(
            session, quiz.id, _question_payload(options=_MCQ_OPTIONS), actor
        )
        theirs = await authoring_service.create_question(
            session, other.id, _question_payload(options=_MCQ_OPTIONS), actor
        )

    async with session_factory() as session, session.begin():
        updated = await authoring_service.bulk_approve_questions(
            session, quiz.id, [mine.id, theirs.id, uuid.uuid4()], actor
        )

    assert updated == 1, "the foreign question and the unknown id are both skipped"

    async with session_factory() as session:
        rows = dict(
            (
                await session.execute(
                    text("SELECT id, review_status FROM quiz_questions WHERE id = ANY(:ids)"),
                    {"ids": [mine.id, theirs.id]},
                )
            ).all()
        )
    assert rows[mine.id] == "approved"
    assert rows[theirs.id] == "pending"


@pytest.mark.asyncio
async def test_bulk_approve_with_no_ids_is_a_no_op(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """The client sends an empty selection whenever a teacher clicks through
    the review screen without ticking anything.
    """
    async with session_factory() as session, session.begin():
        quiz = await _make_quiz(session, scenario)
        updated = await authoring_service.bulk_approve_questions(
            session, quiz.id, [], _actor(scenario["owner_id"])
        )
    assert updated == 0
