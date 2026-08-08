from __future__ import annotations

from importlib import resources
from typing import Any, NamedTuple
from uuid import UUID

from sqlalchemy import delete, func, or_, select, text, true
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute, selectinload
from sqlalchemy.sql.elements import ColumnElement

from abridgeai.features.access_control.models import Role, UserRoleAssignment
from abridgeai.features.courses.models import (
    Course,
    CourseLearningOutcome,
    Lesson,
    LessonResource,
    Module,
    ModuleItem,
    ModulePrerequisite,
)
from abridgeai.features.enrollments.models import Enrollment
from abridgeai.features.identity.models import StorageObject, User, UserProfile

_ROSTER_WITH_PROGRESS_SQL = text(
    resources.files("abridgeai.features.courses.queries.sql")
    .joinpath("roster_with_progress.sql")
    .read_text(encoding="utf-8")
)


def _archived_filter(
    include_archived: bool, status_col: InstrumentedAttribute[str]
) -> ColumnElement[bool]:
    if include_archived:
        return true()
    return status_col != "archived"


async def count_students_and_modules_for_courses(
    db: AsyncSession, course_ids: list[UUID]
) -> dict[UUID, tuple[int, int]]:
    """Batch-count active students + published/draft modules per course.

    Returns ``{course_id: (student_count, module_count)}`` for every id in
    ``course_ids``. Two grouped aggregate queries (one per table) keep this
    O(1) round-trips regardless of course count — no N+1. Courses with no
    enrollments / modules are backfilled to ``(0, 0)`` by the caller.

    * student_count = enrollments with ``status='active'`` (matches the
      roster's "active students" definition; dropped/completed excluded).
    * module_count = modules not soft-deleted (any status).
    """
    result: dict[UUID, tuple[int, int]] = {cid: (0, 0) for cid in course_ids}
    if not course_ids:
        return result

    student_stmt = (
        select(Enrollment.course_id, func.count().label("n"))
        .where(Enrollment.course_id.in_(course_ids), Enrollment.status == "active")
        .group_by(Enrollment.course_id)
    )
    for course_id, n in (await db.execute(student_stmt)).all():
        _students, modules = result[course_id]
        result[course_id] = (int(n), modules)

    module_stmt = (
        select(Module.course_id, func.count().label("n"))
        .where(Module.course_id.in_(course_ids), Module.deleted_at.is_(None))
        .group_by(Module.course_id)
    )
    for course_id, n in (await db.execute(module_stmt)).all():
        students, _modules = result[course_id]
        result[course_id] = (students, int(n))

    return result


async def list_instructors_for_courses(
    db: AsyncSession, course_ids: list[UUID]
) -> dict[UUID, dict[str, Any]]:
    """Batch instructor blocks for authoring list endpoints (drafts included).

    Mirrors the public :func:`~abridgeai.features.courses.queries.published.
    get_course_instructor` composition but (a) runs over MANY course ids in
    one join (no N+1) and (b) drops the ``published_course_clause()`` filter
    — the manager/dept worklist needs the owner on draft rows too, since
    "no owner profile" is exactly the signal that turns a row into an
    "Unassigned" work item.

    Returns ``{course_id: {user_id, display_name, primary_email, headline,
    avatar_bucket, avatar_object_key}}`` for courses whose owner has a
    ``user_profiles`` row; courses with no owner profile are absent (caller
    leaves ``instructor=None``). The service layer mints the presigned
    ``avatar_url`` from the bucket/key — this query stays DB-only.
    """
    if not course_ids:
        return {}

    stmt = (
        select(
            Course.id.label("course_id"),
            User.id.label("user_id"),
            User.primary_email,
            UserProfile.display_name,
            UserProfile.bio,
            StorageObject.bucket.label("avatar_bucket"),
            StorageObject.object_key.label("avatar_object_key"),
        )
        .join(User, User.id == Course.owner_user_id)
        .outerjoin(UserProfile, UserProfile.user_id == User.id)
        .outerjoin(StorageObject, StorageObject.id == UserProfile.avatar_object_id)
        .where(Course.id.in_(course_ids))
    )
    result: dict[UUID, dict[str, Any]] = {}
    for row in (await db.execute(stmt)).all():
        if row.display_name is None:
            continue
        result[row.course_id] = {
            "user_id": row.user_id,
            "display_name": row.display_name,
            "avatar_bucket": row.avatar_bucket,
            "avatar_object_key": row.avatar_object_key,
            "headline": row.bio,
            "primary_email": row.primary_email,
        }
    return result


async def count_pending_grading_for_courses(
    db: AsyncSession, course_ids: list[UUID]
) -> tuple[int, int]:
    """Count ungraded quiz attempts + pending interview evaluations.

    Returns ``(ungraded_quizzes, pending_interviews)`` summed across all
    ``course_ids``. Two aggregate queries, no N+1:

    * ungraded_quizzes = quiz attempts with ``status='submitted'`` and
      ``passed IS NULL`` (submitted but not yet graded) whose quiz belongs to
      one of the courses.
    * pending_interviews = interview sessions in a terminal-but-ungraded state
      (``status IN ('completed','timed_out')`` and ``pass_verdict IS NULL``)
      whose config belongs to one of the courses.

    Both power the teacher dashboard's actionable "needs grading" widgets.
    """
    # Local imports keep the cross-feature model dependency out of module load
    # order (quizzes / interviews import courses, not the other way around).
    from abridgeai.features.interviews.models import (
        InterviewConfig,
        InterviewSession,
    )
    from abridgeai.features.quizzes.models import Quiz, QuizAttempt

    if not course_ids:
        return (0, 0)

    quiz_stmt = (
        select(func.count())
        .select_from(QuizAttempt)
        .join(Quiz, Quiz.id == QuizAttempt.quiz_id)
        .where(
            Quiz.course_id.in_(course_ids),
            Quiz.deleted_at.is_(None),
            QuizAttempt.status == "submitted",
            QuizAttempt.passed.is_(None),
        )
    )
    ungraded_quizzes = int((await db.execute(quiz_stmt)).scalar_one())

    interview_stmt = (
        select(func.count())
        .select_from(InterviewSession)
        .join(InterviewConfig, InterviewConfig.id == InterviewSession.interview_config_id)
        .where(
            InterviewConfig.course_id.in_(course_ids),
            InterviewConfig.deleted_at.is_(None),
            InterviewSession.status.in_(("completed", "timed_out")),
            InterviewSession.pass_verdict.is_(None),
            # Practice runs are ungraded by design, so their NULL verdict is not
            # work waiting for the teacher. Without this they would pile up on
            # the dashboard as permanently pending marking.
            InterviewSession.session_mode != "practice",
        )
    )
    pending_interviews = int((await db.execute(interview_stmt)).scalar_one())

    return (ungraded_quizzes, pending_interviews)


async def count_pending_review_by_course(
    db: AsyncSession, course_ids: list[UUID]
) -> dict[UUID, int]:
    """Per-course count of AI-generated items awaiting teacher review.

    Returns ``{course_id: pending_count}`` covering quiz questions AND interview
    questions still in ``review_status='pending'``. Courses with nothing pending
    are omitted, so the caller can treat a missing key as zero.

    Powers the pending-review dot on the dashboard's course cards: the aggregate
    total tells a teacher that work exists, this tells them *where*, without
    opening each course. Two GROUP BY queries — no per-course round trip.
    """
    from abridgeai.features.interviews.models import (
        InterviewConfig,
        InterviewQuestion,
    )
    from abridgeai.features.quizzes.models import Quiz, QuizQuestion

    if not course_ids:
        return {}

    counts: dict[UUID, int] = {}

    quiz_stmt = (
        select(Quiz.course_id, func.count())
        .select_from(QuizQuestion)
        .join(Quiz, Quiz.id == QuizQuestion.quiz_id)
        .where(
            Quiz.course_id.in_(course_ids),
            Quiz.deleted_at.is_(None),
            QuizQuestion.deleted_at.is_(None),
            QuizQuestion.review_status == "pending",
        )
        .group_by(Quiz.course_id)
    )
    for course_id, count in (await db.execute(quiz_stmt)).all():
        counts[course_id] = counts.get(course_id, 0) + int(count)

    interview_stmt = (
        select(InterviewConfig.course_id, func.count())
        .select_from(InterviewQuestion)
        .join(
            InterviewConfig,
            InterviewConfig.id == InterviewQuestion.interview_config_id,
        )
        .where(
            InterviewConfig.course_id.in_(course_ids),
            InterviewConfig.deleted_at.is_(None),
            InterviewQuestion.deleted_at.is_(None),
            InterviewQuestion.review_status == "pending",
        )
        .group_by(InterviewConfig.course_id)
    )
    for course_id, count in (await db.execute(interview_stmt)).all():
        counts[course_id] = counts.get(course_id, 0) + int(count)

    return counts


class TeacherReviewQueueCounts(NamedTuple):
    """Batched dashboard aggregates for the review queue + retention block.

    Returned by :func:`count_review_queue_and_retention_for_courses` so the
    service layer stays free of positional-tuple guesswork.
    """

    quiz_cards_pending_review: int
    interview_questions_pending_review: int
    published_quizzes_missing_texp: int
    materials_ready_for_quiz_gen: int
    students_below_ef_threshold: int
    avg_retention_ef: float
    cards_overdue: int


_EF_STRUGGLING_THRESHOLD = 2.0
"""Average easiness factor below which a student counts as struggling.

Mirrors ``lessons.ef_min_unlock``'s default (2.0) — the same SM-2 easiness
level the unlock rule treats as "not yet retained".
"""


async def count_review_queue_and_retention_for_courses(
    db: AsyncSession, course_ids: list[UUID]
) -> TeacherReviewQueueCounts:
    """Human-in-the-loop review backlog + spaced-repetition retention signal.

    All seven aggregates are summed across ``course_ids`` in five grouped
    queries — O(1) round-trips regardless of course count, same no-N+1
    property as :func:`count_pending_grading_for_courses`.

    Review queue:

    * quiz_cards_pending_review = ``quiz_questions.review_status='pending'``
      whose quiz belongs to an in-scope course (AI-generated cards awaiting
      teacher approval).
    * interview_questions_pending_review = the interview-side equivalent,
      scoped through ``interview_configs.course_id``.
    * published_quizzes_missing_texp = DISTINCT published quizzes holding at
      least one ``approved`` question with no usable
      ``expected_response_time_ms`` (NULL or ``<= 0``). SM-2 grading needs
      ``t_exp``, so these are live-but-uncalibrated quizzes.
    * materials_ready_for_quiz_gen = material versions whose ingestion
      finished (a ``full_pipeline`` ``processing_jobs`` row with
      ``status='completed'`` on ``entity_type='material_version'`` — the
      status does NOT live on ``learning_materials``) whose lesson has not
      yet been used as a quiz source (no ``quiz_source_lessons`` row).

    Retention (spaced repetition), scoped through
    ``student_card_state.question_id -> quiz_questions -> quizzes ->
    modules -> courses``:

    * students_below_ef_threshold = DISTINCT students whose AVERAGE ``ef``
      over in-scope cards is below 2.0.
    * avg_retention_ef = mean ``ef`` over in-scope cards, 2dp, 0.0 when the
      caller has no cards.
    * cards_overdue = in-scope cards with ``due_at < now()``.

    Soft-deleted questions / quizzes / modules / materials are excluded
    throughout, matching :func:`count_pending_grading_for_courses`.
    """
    # Local imports keep the cross-feature model dependency out of module
    # load order (quizzes / interviews / materials / spaced_repetition import
    # courses, not the other way around). Read-only JOINs only.
    from abridgeai.ai.models import ProcessingJob
    from abridgeai.features.interviews.models import InterviewConfig, InterviewQuestion
    from abridgeai.features.materials.models import LearningMaterial, LearningMaterialVersion
    from abridgeai.features.quizzes.models import Quiz, QuizQuestion, QuizSourceLesson
    from abridgeai.features.spaced_repetition.models import StudentCardState

    if not course_ids:
        return TeacherReviewQueueCounts(0, 0, 0, 0, 0, 0.0, 0)

    pending_cards_stmt = (
        select(func.count())
        .select_from(QuizQuestion)
        .join(Quiz, Quiz.id == QuizQuestion.quiz_id)
        .where(
            Quiz.course_id.in_(course_ids),
            Quiz.deleted_at.is_(None),
            QuizQuestion.deleted_at.is_(None),
            QuizQuestion.review_status == "pending",
        )
    )
    quiz_cards_pending_review = int((await db.execute(pending_cards_stmt)).scalar_one())

    pending_interview_q_stmt = (
        select(func.count())
        .select_from(InterviewQuestion)
        .join(InterviewConfig, InterviewConfig.id == InterviewQuestion.interview_config_id)
        .where(
            InterviewConfig.course_id.in_(course_ids),
            InterviewConfig.deleted_at.is_(None),
            InterviewQuestion.deleted_at.is_(None),
            InterviewQuestion.review_status == "pending",
        )
    )
    interview_questions_pending_review = int(
        (await db.execute(pending_interview_q_stmt)).scalar_one()
    )

    missing_texp_stmt = (
        select(func.count(func.distinct(Quiz.id)))
        .select_from(Quiz)
        .join(QuizQuestion, QuizQuestion.quiz_id == Quiz.id)
        .where(
            Quiz.course_id.in_(course_ids),
            Quiz.deleted_at.is_(None),
            Quiz.status == "published",
            QuizQuestion.deleted_at.is_(None),
            QuizQuestion.review_status == "approved",
            or_(
                QuizQuestion.expected_response_time_ms.is_(None),
                QuizQuestion.expected_response_time_ms <= 0,
            ),
        )
    )
    published_quizzes_missing_texp = int((await db.execute(missing_texp_stmt)).scalar_one())

    # A lesson is "already covered" once any live quiz names it as a source.
    # quizzes.module_id alone is too coarse (a module-wide quiz would mask
    # every other lesson in that module), so the anti-join runs on the
    # lesson-level quiz_source_lessons link table.
    lesson_has_quiz = (
        select(QuizSourceLesson.lesson_id)
        .join(Quiz, Quiz.id == QuizSourceLesson.quiz_id)
        .where(
            QuizSourceLesson.lesson_id == Lesson.id,
            Quiz.deleted_at.is_(None),
        )
        .exists()
    )
    ready_materials_stmt = (
        select(func.count(func.distinct(LearningMaterialVersion.id)))
        .select_from(LearningMaterialVersion)
        .join(LearningMaterial, LearningMaterial.id == LearningMaterialVersion.material_id)
        .join(Lesson, Lesson.id == LearningMaterial.lesson_id)
        .join(Module, Module.id == Lesson.module_id)
        .join(
            ProcessingJob,
            (ProcessingJob.entity_id == LearningMaterialVersion.id)
            & (ProcessingJob.entity_type == "material_version")
            & (ProcessingJob.job_type == "full_pipeline")
            & (ProcessingJob.status == "completed"),
        )
        .where(
            Module.course_id.in_(course_ids),
            Module.deleted_at.is_(None),
            Lesson.deleted_at.is_(None),
            LearningMaterial.deleted_at.is_(None),
            LearningMaterialVersion.deleted_at.is_(None),
            ~lesson_has_quiz,
        )
    )
    materials_ready_for_quiz_gen = int((await db.execute(ready_materials_stmt)).scalar_one())

    # student_card_state.question_id -> quiz_questions -> quizzes ->
    # modules -> courses. quizzes carries a denormalized course_id too, but
    # the module hop is the authoritative containment path, so both are
    # asserted (they agree in the schema's data).
    card_scope = (
        select(StudentCardState)
        .join(QuizQuestion, QuizQuestion.id == StudentCardState.question_id)
        .join(Quiz, Quiz.id == QuizQuestion.quiz_id)
        .join(Module, Module.id == Quiz.module_id)
        .where(
            Module.course_id.in_(course_ids),
            Module.deleted_at.is_(None),
            Quiz.deleted_at.is_(None),
            QuizQuestion.deleted_at.is_(None),
        )
    )

    card_totals_stmt = card_scope.with_only_columns(
        func.count(),
        func.round(func.avg(StudentCardState.ef), 2),
        func.count().filter(StudentCardState.due_at < func.now()),
    )
    card_count, avg_ef, cards_overdue = (await db.execute(card_totals_stmt)).one()

    below_threshold_stmt = (
        select(func.count())
        .select_from(
            card_scope.with_only_columns(StudentCardState.student_id)
            .group_by(StudentCardState.student_id)
            .having(func.avg(StudentCardState.ef) < _EF_STRUGGLING_THRESHOLD)
            .subquery()
        )
    )
    students_below_ef_threshold = int((await db.execute(below_threshold_stmt)).scalar_one())

    return TeacherReviewQueueCounts(
        quiz_cards_pending_review=quiz_cards_pending_review,
        interview_questions_pending_review=interview_questions_pending_review,
        published_quizzes_missing_texp=published_quizzes_missing_texp,
        materials_ready_for_quiz_gen=materials_ready_for_quiz_gen,
        students_below_ef_threshold=students_below_ef_threshold,
        # avg() is NULL when the caller's courses hold no cards at all.
        avg_retention_ef=float(avg_ef) if card_count and avg_ef is not None else 0.0,
        cards_overdue=int(cards_overdue),
    )


async def list_courses_for_owner(
    db: AsyncSession,
    user_id: UUID,
    *,
    include_archived: bool = False,
) -> list[Course]:
    """All courses (any status) owned by ``user_id``.

    No visibility filter — drafts surface for the author. ``include_archived``
    defaults FALSE per plan §4119; pass TRUE for the "all my courses" admin
    view.
    """
    stmt = (
        select(Course)
        .where(
            Course.owner_user_id == user_id,
            _archived_filter(include_archived, Course.status),
        )
        .order_by(Course.created_at.desc())
    )
    return list((await db.execute(stmt)).scalars().all())


async def list_courses_assigned_to_teacher(
    db: AsyncSession,
    user_id: UUID,
    *,
    include_archived: bool = False,
) -> list[Course]:
    """All courses where ``user_id`` holds an active ``role=teacher`` course-scoped assignment.

    Mirrors :func:`list_courses_for_owner` for the "co-author" path:
    ``user_role_assignments`` rows with ``scope_kind='course'`` and
    ``role.code='teacher'`` give a teacher edit access without owning
    the course. Active = not soft-deleted AND ``active_until`` IS NULL
    or in the future.
    """
    id_stmt = (
        select(UserRoleAssignment.course_id)
        .join(Role, Role.id == UserRoleAssignment.role_id)
        .where(
            UserRoleAssignment.user_id == user_id,
            UserRoleAssignment.scope_kind == "course",
            Role.code == "teacher",
            UserRoleAssignment.deleted_at.is_(None),
            (UserRoleAssignment.active_until.is_(None))
            | (UserRoleAssignment.active_until > func.now()),
        )
    )
    ids = [row[0] for row in (await db.execute(id_stmt)).all() if row[0] is not None]
    if not ids:
        return []
    stmt = (
        select(Course)
        .where(
            Course.id.in_(ids),
            _archived_filter(include_archived, Course.status),
        )
        .order_by(Course.created_at.desc())
    )
    return list((await db.execute(stmt)).scalars().all())


async def list_courses_in_org_unit(db: AsyncSession, org_unit_id: UUID) -> list[Course]:
    stmt = (
        select(Course).where(Course.org_unit_id == org_unit_id).order_by(Course.created_at.desc())
    )
    return list((await db.execute(stmt)).scalars().all())


async def get_course_for_authoring(db: AsyncSession, course_id: UUID) -> Course | None:
    """Course by id without status filter (returns drafts and archived)."""
    return await db.get(Course, course_id)


async def get_course_with_content_tree(
    db: AsyncSession,
    course_id: UUID,
    *,
    include_archived: bool = False,
) -> Course | None:
    """Course + modules + items + polymorphic targets (lesson/quiz/interview).

    Eager-loads modules → items → lesson/quiz/interview_config in 4 queries
    (one selectinload per branch). Soft-deleted rows are excluded by callers.
    Returns the ORM ``Course`` instance with relationships hydrated, or
    ``None`` if the course is missing or soft-deleted.
    """
    stmt = (
        select(Course)
        .where(Course.id == course_id, Course.deleted_at.is_(None))
        .options(
            selectinload(Course.modules).selectinload(Module.items).selectinload(ModuleItem.lesson),
            selectinload(Course.modules).selectinload(Module.items).selectinload(ModuleItem.quiz),
            selectinload(Course.modules)
            .selectinload(Module.items)
            .selectinload(ModuleItem.interview_config),
        )
    )
    if not include_archived:
        stmt = stmt.where(Course.status != "archived")
    return (await db.execute(stmt)).scalar_one_or_none()


async def list_modules_for_authoring(db: AsyncSession, course_id: UUID) -> list[Module]:
    stmt = select(Module).where(Module.course_id == course_id).order_by(Module.position)
    return list((await db.execute(stmt)).scalars().all())


async def list_lessons_for_authoring(db: AsyncSession, module_id: UUID) -> list[Lesson]:
    """Lessons that are LIVE members of ``module_id``.

    Membership is defined by a non-soft-deleted ``ModuleItem`` link, NOT by
    ``Lesson.module_id`` alone. Deleting a module item soft-deletes only the
    link (the lesson row survives, keeping its ``module_id``); joining through
    ``ModuleItem`` lets the T0.7 ``with_loader_criteria`` soft-delete filter
    drop lessons whose only link was removed — otherwise a deleted item keeps
    reappearing in the quiz-generation source picker.
    """
    stmt = (
        select(Lesson)
        .join(ModuleItem, ModuleItem.lesson_id == Lesson.id)
        .where(ModuleItem.module_id == module_id)
        .order_by(Lesson.title)
    )
    return list((await db.execute(stmt)).scalars().all())


async def list_all_lesson_resources(db: AsyncSession, lesson_id: UUID) -> list[LessonResource]:
    """All resources on the lesson (no visible_to_students filter)."""
    stmt = (
        select(LessonResource)
        .where(LessonResource.lesson_id == lesson_id)
        .order_by(LessonResource.position)
    )
    return list((await db.execute(stmt)).scalars().all())


async def get_module(db: AsyncSession, module_id: UUID) -> Module | None:
    return await db.get(Module, module_id)


async def course_slug_exists(db: AsyncSession, *, organization_id: UUID, slug: str) -> bool:
    """Whether an active (non-soft-deleted) course already uses ``slug`` in ``organization_id``.

    Mirrors the partial UNIQUE INDEX behind ``uq_courses_org_slug`` (see
    migration 0002): the constraint is scoped to rows where
    ``deleted_at IS NULL``, so the check must apply the same predicate.
    """
    stmt = (
        select(func.count())
        .select_from(Course)
        .where(
            Course.organization_id == organization_id,
            Course.slug == slug,
            Course.deleted_at.is_(None),
        )
    )
    return bool(int((await db.execute(stmt)).scalar_one()))


async def get_lesson(db: AsyncSession, lesson_id: UUID) -> Lesson | None:
    return await db.get(Lesson, lesson_id)


async def get_lesson_resource(db: AsyncSession, resource_id: UUID) -> LessonResource | None:
    return await db.get(LessonResource, resource_id)


async def get_authoring_resource_storage_target(
    db: AsyncSession, resource_id: UUID
) -> tuple[str, str] | None:
    """Bucket + object_key for a lesson resource the teacher can edit.

    Authoring sibling of :func:`published.get_visible_resource_storage_target`:
    no learner publish-chain gates and no ``visible_to_students``
    filter — the teacher must reach hidden / draft resources during
    course assembly. Soft-deleted rows are still excluded. Returns
    ``None`` for missing resources or resources without an attached
    storage object.
    """
    stmt = (
        select(StorageObject.bucket, StorageObject.object_key)
        .join(LessonResource, LessonResource.storage_object_id == StorageObject.id)
        .where(
            LessonResource.id == resource_id,
            LessonResource.deleted_at.is_(None),
        )
    )
    row = (await db.execute(stmt)).one_or_none()
    if row is None:
        return None
    return row.bucket, row.object_key


async def get_course_thumbnail_storage_target(
    db: AsyncSession, course_id: UUID
) -> tuple[str, str] | None:
    """Bucket + object_key for a course's thumbnail image, or ``None``.

    Joins ``courses.thumbnail_object_id → storage_objects.id``. Returns
    ``None`` when the course has no thumbnail set.
    """
    stmt = (
        select(StorageObject.bucket, StorageObject.object_key)
        .join(Course, Course.thumbnail_object_id == StorageObject.id)
        .where(Course.id == course_id)
    )
    row = (await db.execute(stmt)).one_or_none()
    if row is None:
        return None
    return row.bucket, row.object_key


async def get_module_item(db: AsyncSession, item_id: UUID) -> ModuleItem | None:
    return await db.get(ModuleItem, item_id)


async def next_module_item_position(db: AsyncSession, module_id: UUID) -> int:
    """Return ``MAX(position) + 1`` for ``module_items`` under ``module_id``.

    Mirrors the legacy ``backend/app/routes/courses/service.py:create_lesson``
    helper used to auto-place a new ``ModuleItem`` at the end of its module.
    Returns 1 when the module has no items yet.
    """
    stmt = select(func.coalesce(func.max(ModuleItem.position), 0)).where(
        ModuleItem.module_id == module_id
    )
    return int((await db.execute(stmt)).scalar_one()) + 1


async def list_module_items(db: AsyncSession, module_id: UUID) -> list[ModuleItem]:
    """All ``ModuleItem`` rows under ``module_id`` ordered by ``position``."""
    stmt = select(ModuleItem).where(ModuleItem.module_id == module_id).order_by(ModuleItem.position)
    return list((await db.execute(stmt)).scalars().all())


async def next_module_position(db: AsyncSession, course_id: UUID) -> int:
    """Return ``MAX(position) + 1`` for ``modules`` under ``course_id``.

    Mirrors :func:`next_module_item_position`. Returns 1 for an empty course.
    Ignores soft-deleted modules so a fresh clone lands after the live tail.
    """
    stmt = select(func.coalesce(func.max(Module.position), 0)).where(
        Module.course_id == course_id,
        Module.deleted_at.is_(None),
    )
    return int((await db.execute(stmt)).scalar_one()) + 1


async def list_module_prerequisites(db: AsyncSession, module_id: UUID) -> list[UUID]:
    """Return the list of prerequisite module ids for ``module_id``."""
    stmt = select(ModulePrerequisite.prerequisite_module_id).where(
        ModulePrerequisite.module_id == module_id
    )
    return [row[0] for row in (await db.execute(stmt)).all()]


async def replace_module_prerequisites(
    db: AsyncSession, module_id: UUID, prereq_module_ids: list[UUID]
) -> None:
    """Idempotent: clear the existing prereq set and insert the new one."""
    await db.execute(delete(ModulePrerequisite).where(ModulePrerequisite.module_id == module_id))
    for prereq_id in prereq_module_ids:
        db.add(ModulePrerequisite(module_id=module_id, prerequisite_module_id=prereq_id))
    await db.flush()


async def list_course_roster(db: AsyncSession, course_id: UUID) -> list[dict[str, Any]]:
    """Enrolled students for a course with user profile info."""
    stmt = (
        select(
            Enrollment.id.label("enrollment_id"),
            Enrollment.student_id,
            User.primary_email,
            UserProfile.display_name,
            Enrollment.status,
            Enrollment.enrolled_at,
            Enrollment.completed_at,
            Enrollment.dropped_at,
        )
        .join(User, User.id == Enrollment.student_id)
        .outerjoin(UserProfile, UserProfile.user_id == User.id)
        .where(Enrollment.course_id == course_id)
        .order_by(Enrollment.enrolled_at.desc())
    )
    rows = (await db.execute(stmt)).mappings().all()
    return [dict(row) for row in rows]


async def list_course_roster_with_progress(
    db: AsyncSession, course_id: UUID
) -> list[dict[str, Any]]:
    """Enrolled students for the teacher roster view, with progress + risk.

    Composes ``sql/roster_with_progress.sql`` — a richer projection than
    :func:`list_course_roster` (used by the ``/dept`` HOD-scope roster,
    which stays on the thin shape). Adds ``progress_percent``,
    ``last_activity_at`` and a derived ``at_risk_level``, matching the
    frontend's ``RosterStudent`` DTO 1:1. The same lesson-progress
    aggregation as ``progress/queries/sql/roster_progress.sql`` so the
    Students and Progress pages never disagree on a student's percent.
    """
    rows = (await db.execute(_ROSTER_WITH_PROGRESS_SQL, {"course_id": course_id})).mappings().all()
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Course learning outcomes (§LO-1/2) — teacher-side CRUD reads.
# ---------------------------------------------------------------------------
async def list_course_outcomes(db: AsyncSession, course_id: UUID) -> list[CourseLearningOutcome]:
    """All (non-deleted) learning outcomes for a course.

    Ordered by ``(parent_id, position)`` so siblings are contiguous. The
    dotted ``L.O.x.y`` display code + depth are derived from the parent
    chain at render time (see :func:`build_outcome_code_map`); the stored
    ``position`` is the sibling order within each parent.
    """
    stmt = (
        select(CourseLearningOutcome)
        .where(
            CourseLearningOutcome.course_id == course_id,
            CourseLearningOutcome.deleted_at.is_(None),
        )
        .order_by(CourseLearningOutcome.parent_id, CourseLearningOutcome.position)
    )
    return list((await db.execute(stmt)).scalars().all())


async def count_course_outcomes(db: AsyncSession, course_id: UUID) -> int:
    """Number of live learning outcomes on ``course_id``, at any depth.

    A COUNT rather than ``len(await list_course_outcomes(...))``: the publish
    gate and the readiness checklist only ever ask "any?", and the list query
    orders and materialises every row to answer it.
    """
    stmt = select(func.count()).where(
        CourseLearningOutcome.course_id == course_id,
        CourseLearningOutcome.deleted_at.is_(None),
    )
    return (await db.execute(stmt)).scalar_one()


async def count_questions_mapped_to_outcomes(
    db: AsyncSession, course_id: UUID, outcome_ids: set[UUID]
) -> dict[UUID, int]:
    """Live quiz questions mapping to each of ``outcome_ids``.

    Lazy import keeps the courses -> quizzes model edge out of module import
    time (same pattern as the readiness helper above). Questions are counted
    per-outcome so the delete confirmation can name exactly what loses its
    mapping; questions are NOT deleted — the FK is ``ON DELETE SET NULL`` and
    we soft-delete outcomes anyway, so a question's ``learning_outcome_id``
    simply stops resolving.
    """
    if not outcome_ids:
        return {}
    from abridgeai.features.quizzes.models import Quiz, QuizQuestion  # noqa: PLC0415

    stmt = (
        select(QuizQuestion.learning_outcome_id, func.count())
        .join(Quiz, Quiz.id == QuizQuestion.quiz_id)
        .where(
            Quiz.course_id == course_id,
            QuizQuestion.learning_outcome_id.in_(outcome_ids),
            QuizQuestion.deleted_at.is_(None),
            Quiz.deleted_at.is_(None),
        )
        .group_by(QuizQuestion.learning_outcome_id)
    )
    rows = (await db.execute(stmt)).all()
    return {outcome_id: count for outcome_id, count in rows}


async def list_course_outcome_siblings(
    db: AsyncSession, course_id: UUID, parent_id: UUID | None
) -> list[CourseLearningOutcome]:
    """All live outcomes sharing one parent, in position order.

    ``parent_id`` NULL = top-level. ``IS NOT DISTINCT FROM`` handles the
    NULL parent in a single predicate, mirroring
    :func:`next_course_outcome_position`.
    """
    stmt = (
        select(CourseLearningOutcome)
        .where(
            CourseLearningOutcome.course_id == course_id,
            CourseLearningOutcome.parent_id.is_not_distinct_from(parent_id),
            CourseLearningOutcome.deleted_at.is_(None),
        )
        .order_by(CourseLearningOutcome.position)
    )
    return list((await db.execute(stmt)).scalars().all())


async def get_course_outcome(db: AsyncSession, outcome_id: UUID) -> CourseLearningOutcome | None:
    return await db.get(CourseLearningOutcome, outcome_id)


async def next_course_outcome_position(
    db: AsyncSession, course_id: UUID, parent_id: UUID | None = None
) -> int:
    """Return ``MAX(position) + 1`` among a parent's children (1 when empty).

    Positions are per-parent: top-level rows (``parent_id IS NULL``) share
    one sequence scoped to the course; each parent's children share their
    own sequence. ``IS NOT DISTINCT FROM`` handles the NULL parent case in
    a single predicate.
    """
    stmt = select(func.coalesce(func.max(CourseLearningOutcome.position), 0)).where(
        CourseLearningOutcome.course_id == course_id,
        CourseLearningOutcome.parent_id.is_not_distinct_from(parent_id),
        CourseLearningOutcome.deleted_at.is_(None),
    )
    return int((await db.execute(stmt)).scalar_one()) + 1


async def reindex_course_outcome_siblings(
    db: AsyncSession, course_id: UUID, parent_id: UUID | None
) -> None:
    """Compact one parent's children to a contiguous 1..N chain (§LO-2).

    Called after a delete or move so sibling positions (and therefore the
    dotted codes) never gap. Two-phase offset shift dodges the per-parent
    ``uq_course_learning_outcomes_sibling_position`` UNIQUE collision
    mid-update: bump every sibling far out of the way, flush, then
    renumber 1..N in position order.
    """
    stmt = (
        select(CourseLearningOutcome)
        .where(
            CourseLearningOutcome.course_id == course_id,
            CourseLearningOutcome.parent_id.is_not_distinct_from(parent_id),
            CourseLearningOutcome.deleted_at.is_(None),
        )
        .order_by(CourseLearningOutcome.position)
    )
    siblings = list((await db.execute(stmt)).scalars().all())
    if not siblings:
        return
    for offset_idx, outcome in enumerate(siblings, start=1):
        outcome.position = 100_000 + offset_idx
    await db.flush()
    for new_pos, outcome in enumerate(siblings, start=1):
        outcome.position = new_pos
    await db.flush()


def build_outcome_code_map(
    outcomes: list[CourseLearningOutcome],
) -> dict[UUID, tuple[str, int]]:
    """Derive ``{outcome_id: (dotted_code, depth)}`` from a course's outcomes.

    ``dotted_code`` is the position path root→leaf (e.g. ``"1.2.1"``,
    rendered ``L.O.1.2.1``); ``depth`` is 0 for top-level. Pure function
    over the full (non-deleted) outcome list — no DB access — so callers
    resolve the whole tree in one pass. Orphaned rows (parent missing or
    soft-deleted) are treated as top-level so they still get a code.
    """
    by_parent: dict[UUID | None, list[CourseLearningOutcome]] = {}
    ids = {o.id for o in outcomes}
    for o in outcomes:
        parent = o.parent_id if o.parent_id in ids else None
        by_parent.setdefault(parent, []).append(o)
    for children in by_parent.values():
        children.sort(key=lambda o: o.position)

    code_map: dict[UUID, tuple[str, int]] = {}

    def walk(parent: UUID | None, prefix: str, depth: int) -> None:
        for idx, node in enumerate(by_parent.get(parent, []), start=1):
            code = f"{prefix}{idx}" if not prefix else f"{prefix}.{idx}"
            code_map[node.id] = (code, depth)
            walk(node.id, code, depth + 1)

    walk(None, "", 0)
    return code_map


def build_descendant_map(
    outcomes: list[CourseLearningOutcome],
) -> dict[UUID, set[UUID]]:
    """Derive ``{outcome_id: {all descendant ids}}`` (excludes self).

    Used for coverage rollup (a question on a child counts toward every
    ancestor) and for cycle checks on re-parent. Pure function over the
    full outcome list.
    """
    children: dict[UUID, list[UUID]] = {}
    for o in outcomes:
        if o.parent_id is not None:
            children.setdefault(o.parent_id, []).append(o.id)

    def collect(node: UUID) -> set[UUID]:
        out: set[UUID] = set()
        for child in children.get(node, []):
            out.add(child)
            out |= collect(child)
        return out

    return {o.id: collect(o.id) for o in outcomes}


__all__ = [
    "get_authoring_resource_storage_target",
    "get_course_for_authoring",
    "get_course_with_content_tree",
    "get_course_outcome",
    "get_lesson",
    "get_lesson_resource",
    "get_module",
    "get_module_item",
    "list_all_lesson_resources",
    "list_course_outcomes",
    "list_course_roster",
    "list_course_roster_with_progress",
    "list_courses_assigned_to_teacher",
    "list_courses_for_owner",
    "list_courses_in_org_unit",
    "list_lessons_for_authoring",
    "list_module_items",
    "list_module_prerequisites",
    "list_modules_for_authoring",
    "next_module_item_position",
    "replace_module_prerequisites",
]
