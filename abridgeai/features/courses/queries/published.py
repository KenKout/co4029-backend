from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.pagination.cursor import (
    CursorPage,
    decode_cursor,
    encode_cursor,
)
from abridgeai.features.access_control.api import public as access_control_api
from abridgeai.features.access_control.models import Role, UserRoleAssignment
from abridgeai.features.courses.models import (
    Course,
    CourseLearningOutcome,
    CourseSyllabusImport,
    CourseTag,
    Lesson,
    LessonResource,
    Module,
    ModuleItem,
    Tag,
)
from abridgeai.features.courses.visibility import (
    module_item_visible_clause,
    published_course_clause,
    published_lesson_clause,
    published_module_clause,
    student_visible_resource_clause,
)
from abridgeai.features.enrollments.models import Enrollment
from abridgeai.features.identity.models import StorageObject, User, UserProfile
from abridgeai.features.progress.models import LessonProgress
from abridgeai.features.quizzes.models import Quiz, QuizAttempt

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 100


def _clamp(limit: int) -> int:
    return min(max(limit, 1), _MAX_LIMIT)


async def list_published_courses(
    db: AsyncSession,
    *,
    organization_id: UUID,
    limit: int = _DEFAULT_LIMIT,
    cursor: str | None = None,
) -> CursorPage[Course]:
    """Cursor-paginated published courses scoped to an organization.

    Reconciliation §A10/§D2: cursor pagination, not offset.
    Visibility composes :func:`published_course_clause` from T3.3.
    """
    capped = _clamp(limit)
    after = decode_cursor(cursor) if cursor else None
    stmt = (
        select(Course)
        .where(
            published_course_clause(),
            Course.organization_id == organization_id,
        )
        .order_by(Course.id)
        .limit(capped)
    )
    if after is not None:
        stmt = stmt.where(Course.id > after)
    rows = list((await db.execute(stmt)).scalars().all())
    next_cursor = encode_cursor(rows[-1].id) if len(rows) == capped else None
    return CursorPage(items=rows, next_cursor=next_cursor)


async def list_enrolled_courses(
    db: AsyncSession,
    user_id: UUID,
    *,
    limit: int = _DEFAULT_LIMIT,
    cursor: str | None = None,
) -> CursorPage[Course]:
    """Active enrollments -> published courses, most recently active first.

    "Recent activity" = the latest lesson-progress heartbeat or quiz-attempt
    start on the course, so the dashboard's my-courses rows lead with the
    course the student was last studying. Enrollments with no activity yet
    fall to the tail in stable id order (pagination order stays total).
    """
    capped = _clamp(limit)
    after = decode_cursor(cursor) if cursor else None
    lesson_activity = (
        select(func.max(LessonProgress.last_activity_at))
        .select_from(LessonProgress)
        .join(Lesson, Lesson.id == LessonProgress.lesson_id)
        .join(Module, Module.id == Lesson.module_id)
        .where(
            LessonProgress.user_id == user_id,
            Module.course_id == Course.id,
        )
        .correlate(Course)
        .scalar_subquery()
    )
    quiz_activity = (
        select(func.max(QuizAttempt.started_at))
        .select_from(QuizAttempt)
        .join(Quiz, Quiz.id == QuizAttempt.quiz_id)
        .where(
            QuizAttempt.student_id == user_id,
            Quiz.course_id == Course.id,
        )
        .correlate(Course)
        .scalar_subquery()
    )
    # GREATEST is NULL when either side is NULL, so fall back to whichever
    # side exists for courses with only one kind of activity.
    last_active = func.coalesce(
        func.greatest(lesson_activity, quiz_activity),
        lesson_activity,
        quiz_activity,
    )
    stmt = (
        select(Course)
        .join(Enrollment, Enrollment.course_id == Course.id)
        .where(
            Enrollment.student_id == user_id,
            Enrollment.status == "active",
            published_course_clause(),
        )
        .order_by(last_active.desc().nulls_last(), Course.id)
        .limit(capped)
    )
    if after is not None:
        stmt = stmt.where(Course.id > after)
    courses = list((await db.execute(stmt)).scalars().all())
    next_cursor = encode_cursor(courses[-1].id) if len(courses) == capped else None
    return CursorPage(items=courses, next_cursor=next_cursor)


async def get_published_course_by_slug(
    db: AsyncSession, slug: str, organization_id: UUID
) -> Course | None:
    """Org-scoped slug lookup (Reconciliation §A11).

    Slug uniqueness is per-organization (partial unique index
    ``uq_courses_org_slug`` from migration 0002). Two orgs may both own
    a course slug ``"intro-to-python"``; this query MUST be passed an
    ``organization_id`` so the wrong org's course never leaks.
    """
    stmt = select(Course).where(
        Course.slug == slug,
        Course.organization_id == organization_id,
        published_course_clause(),
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def get_published_course_by_id(db: AsyncSession, course_id: UUID) -> Course | None:
    stmt = select(Course).where(Course.id == course_id, published_course_clause())
    return (await db.execute(stmt)).scalar_one_or_none()


async def get_course_instructor(db: AsyncSession, course_id: UUID) -> dict[str, Any] | None:
    """Compose the primary instructor block for a course's public detail page.

    With multiple Course Instructors legal (user decision 2026-08-30) the
    "instructor" of record is the LONGEST-SERVING active Course Instructor
    (earliest ``active_from``) — identical to the old single-CI semantics
    when there is exactly one, and a stable pick when there are many. Joins
    that teacher's profile and returns ``{user_id, display_name,
    avatar_bucket, avatar_object_key, headline}`` shaped for the service
    layer, which mints a presigned ``avatar_url``.

    Returns ``None`` when the course is unpublished or has no active Course
    Instructor. ``headline`` maps to ``user_profiles.bio`` (no dedicated
    headline column). The avatar bucket/key are ``None`` when the teacher
    has not uploaded an avatar; the service leaves ``avatar_url`` as
    ``None`` and the SPA falls back to initials.
    """
    stmt = (
        select(
            User.id.label("user_id"),
            UserProfile.display_name,
            UserProfile.bio,
            StorageObject.bucket.label("avatar_bucket"),
            StorageObject.object_key.label("avatar_object_key"),
        )
        # Explicit left side: none of the selected columns comes from Course, so
        # the join chain below has several candidate FROMs and SQLAlchemy raises
        # "Can't determine which FROM clause to join from" without this.
        .select_from(Course)
        .join(UserRoleAssignment, UserRoleAssignment.course_id == Course.id)
        .join(Role, Role.id == UserRoleAssignment.role_id)
        .join(User, User.id == UserRoleAssignment.user_id)
        .outerjoin(UserProfile, UserProfile.user_id == User.id)
        .outerjoin(StorageObject, StorageObject.id == UserProfile.avatar_object_id)
        .where(
            Course.id == course_id,
            published_course_clause(),
            UserRoleAssignment.is_instructor.is_(True),
            UserRoleAssignment.scope_kind == "course",
            Role.code == "teacher",
            UserRoleAssignment.deleted_at.is_(None),
            (UserRoleAssignment.active_until.is_(None))
            | (UserRoleAssignment.active_until > func.now()),
        )
        .order_by(UserRoleAssignment.active_from, UserRoleAssignment.id)
        .limit(1)
    )
    row = (await db.execute(stmt)).first()
    if row is None or row.display_name is None:
        return None
    return {
        "user_id": row.user_id,
        "display_name": row.display_name,
        "avatar_bucket": row.avatar_bucket,
        "avatar_object_key": row.avatar_object_key,
        "headline": row.bio,
    }


async def get_published_course_content(db: AsyncSession, course_id: UUID) -> dict[str, Any] | None:
    """Fetch the published course tree (course + modules + visible items).

    Returns ``{"course": Course, "modules": list[Module], "items": list[dict]}``
    or ``None`` when the course is missing / unpublished / soft-deleted.
    DRAFT_VISIBILITY rule (plan §4153): items pointing to draft / soft-
    deleted lessons / quizzes / interview configs are excluded entirely
    (not nullified).
    """
    # Lazy import to avoid breaking import-linter's
    # ``Features are independent`` contract at module load.
    from abridgeai.features.interviews.models import InterviewConfig
    from abridgeai.features.quizzes.models import Quiz

    course = await get_published_course_by_id(db, course_id)
    if course is None:
        return None

    modules = await list_published_modules(db, course_id)

    if not modules:
        return {"course": course, "modules": modules, "items": []}

    module_ids = [m.id for m in modules]
    items_stmt = (
        select(ModuleItem)
        .join(Module, Module.id == ModuleItem.module_id)
        .outerjoin(Lesson, Lesson.id == ModuleItem.lesson_id)
        .outerjoin(Quiz, Quiz.id == ModuleItem.quiz_id)
        .outerjoin(InterviewConfig, InterviewConfig.id == ModuleItem.interview_config_id)
        .where(
            ModuleItem.module_id.in_(module_ids),
            ModuleItem.deleted_at.is_(None),
            module_item_visible_clause(),
        )
        .order_by(ModuleItem.module_id, ModuleItem.position)
    )
    items = list((await db.execute(items_stmt)).scalars().all())

    lesson_ids = [item.lesson_id for item in items if item.lesson_id is not None]
    lessons_by_id: dict[UUID, Lesson] = {}
    if lesson_ids:
        lessons_stmt = select(Lesson).where(
            Lesson.id.in_(lesson_ids),
            published_lesson_clause(),
        )
        lessons = (await db.execute(lessons_stmt)).scalars().all()
        lessons_by_id = {lesson.id: lesson for lesson in lessons}

    quiz_ids = [item.quiz_id for item in items if item.quiz_id is not None]
    quizzes_by_id: dict[UUID, Any] = {}
    if quiz_ids:
        quizzes_stmt = select(Quiz).where(
            Quiz.id.in_(quiz_ids),
            Quiz.status == "published",
        )
        quizzes = (await db.execute(quizzes_stmt)).scalars().all()
        quizzes_by_id = {quiz.id: quiz for quiz in quizzes}

    interview_ids = [
        item.interview_config_id for item in items if item.interview_config_id is not None
    ]
    interviews_by_id: dict[UUID, Any] = {}
    if interview_ids:
        interviews_stmt = select(InterviewConfig).where(
            InterviewConfig.id.in_(interview_ids),
            InterviewConfig.status == "published",
        )
        interviews = (await db.execute(interviews_stmt)).scalars().all()
        interviews_by_id = {cfg.id: cfg for cfg in interviews}

    items_data = []
    for item in items:
        lesson = lessons_by_id.get(item.lesson_id) if item.lesson_id else None
        quiz = quizzes_by_id.get(item.quiz_id) if item.quiz_id else None
        interview = (
            interviews_by_id.get(item.interview_config_id) if item.interview_config_id else None
        )
        items_data.append(
            {
                "id": item.id,
                "module_id": item.module_id,
                "item_type": item.item_type,
                "lesson_id": item.lesson_id,
                "quiz_id": item.quiz_id,
                "interview_config_id": item.interview_config_id,
                "position": item.position,
                "unlock_rule_json": item.unlock_rule_json,
                "lesson": lesson,
                "quiz": quiz,
                "interview": interview,
            }
        )

    return {"course": course, "modules": modules, "items": items_data}


async def list_published_modules(db: AsyncSession, course_id: UUID) -> list[Module]:
    stmt = (
        select(Module)
        .join(Course, Course.id == Module.course_id)
        .where(
            Course.id == course_id,
            published_course_clause(),
            published_module_clause(),
        )
        .order_by(Module.position)
    )
    return list((await db.execute(stmt)).scalars().all())


async def list_published_lessons(db: AsyncSession, module_id: UUID) -> list[Lesson]:
    stmt = (
        select(Lesson)
        .join(Module, Module.id == Lesson.module_id)
        .where(
            Module.id == module_id,
            published_module_clause(),
            published_lesson_clause(),
        )
    )
    return list((await db.execute(stmt)).scalars().all())


async def list_visible_lesson_resources(db: AsyncSession, lesson_id: UUID) -> list[LessonResource]:
    """Resources that are both attached to a published lesson AND have
    ``visible_to_students = TRUE`` (T3.3 :func:`student_visible_resource_clause`).
    """
    stmt = (
        select(LessonResource)
        .join(Lesson, Lesson.id == LessonResource.lesson_id)
        .where(
            LessonResource.lesson_id == lesson_id,
            published_lesson_clause(),
            student_visible_resource_clause(),
        )
        .order_by(LessonResource.position)
    )
    return list((await db.execute(stmt)).scalars().all())


async def get_published_module_by_id(db: AsyncSession, module_id: UUID) -> Module | None:
    """Module by id, only when the module AND its parent course are published."""
    stmt = (
        select(Module)
        .join(Course, Course.id == Module.course_id)
        .where(
            Module.id == module_id,
            published_course_clause(),
            published_module_clause(),
        )
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def get_published_lesson_by_id(db: AsyncSession, lesson_id: UUID) -> Lesson | None:
    """Lesson by id, only when the lesson + parent module + parent course are published."""
    stmt = (
        select(Lesson)
        .join(Module, Module.id == Lesson.module_id)
        .join(Course, Course.id == Module.course_id)
        .where(
            Lesson.id == lesson_id,
            published_course_clause(),
            published_module_clause(),
            published_lesson_clause(),
        )
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def get_visible_lesson_resource(db: AsyncSession, resource_id: UUID) -> LessonResource | None:
    """Resource by id, only when ``visible_to_students=TRUE`` AND the
    parent lesson / module / course are all published.

    Returns ``None`` (which the router maps to 404) for invisible resources
    so existence is not leaked.
    """
    stmt = (
        select(LessonResource)
        .join(Lesson, Lesson.id == LessonResource.lesson_id)
        .join(Module, Module.id == Lesson.module_id)
        .join(Course, Course.id == Module.course_id)
        .where(
            LessonResource.id == resource_id,
            published_course_clause(),
            published_module_clause(),
            published_lesson_clause(),
            student_visible_resource_clause(),
        )
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def list_visible_module_items(db: AsyncSession, module_id: UUID) -> list[dict[str, Any]]:
    """``ModuleItem`` rows under ``module_id`` that point to non-draft targets.

    Per the DRAFT_VISIBILITY rule (plan §4153) items pointing to draft /
    soft-deleted lessons / quizzes / interview configs are EXCLUDED,
    not nullified. Returns plain dicts shaped for
    :class:`ModuleItemPublic.model_validate`, with ``target`` populated
    from the joined :class:`Lesson` / :class:`Quiz` /
    :class:`InterviewConfig` row so the polymorphic field is hydrated
    without relying on lazy ORM relationship access (would raise
    ``MissingGreenlet`` outside the awaited session).
    """
    # Lazy import to avoid breaking import-linter's
    # ``Features are independent`` contract at module load.
    from abridgeai.features.interviews.models import InterviewConfig
    from abridgeai.features.quizzes.models import Quiz

    stmt = (
        select(ModuleItem)
        .join(Module, Module.id == ModuleItem.module_id)
        .outerjoin(Lesson, Lesson.id == ModuleItem.lesson_id)
        .outerjoin(Quiz, Quiz.id == ModuleItem.quiz_id)
        .outerjoin(InterviewConfig, InterviewConfig.id == ModuleItem.interview_config_id)
        .where(
            ModuleItem.module_id == module_id,
            published_module_clause(),
            module_item_visible_clause(),
        )
        .order_by(ModuleItem.position)
    )
    items = list((await db.execute(stmt)).scalars().all())

    lesson_ids = [item.lesson_id for item in items if item.lesson_id is not None]
    lessons_by_id: dict[UUID, Lesson] = {}
    if lesson_ids:
        lessons_stmt = select(Lesson).where(
            Lesson.id.in_(lesson_ids),
            published_lesson_clause(),
        )
        lessons_by_id = {
            lesson.id: lesson for lesson in (await db.execute(lessons_stmt)).scalars().all()
        }

    quiz_ids = [item.quiz_id for item in items if item.quiz_id is not None]
    quizzes_by_id: dict[UUID, Any] = {}
    if quiz_ids:
        quizzes_stmt = select(Quiz).where(
            Quiz.id.in_(quiz_ids),
            Quiz.status == "published",
        )
        quizzes_by_id = {quiz.id: quiz for quiz in (await db.execute(quizzes_stmt)).scalars().all()}

    interview_ids = [
        item.interview_config_id for item in items if item.interview_config_id is not None
    ]
    interviews_by_id: dict[UUID, Any] = {}
    if interview_ids:
        interviews_stmt = select(InterviewConfig).where(
            InterviewConfig.id.in_(interview_ids),
            InterviewConfig.status == "published",
        )
        interviews_by_id = {
            cfg.id: cfg for cfg in (await db.execute(interviews_stmt)).scalars().all()
        }

    payload: list[dict[str, Any]] = []
    for item in items:
        if item.item_type == "lesson":
            target = lessons_by_id.get(item.lesson_id) if item.lesson_id else None
        elif item.item_type == "quiz":
            target = quizzes_by_id.get(item.quiz_id) if item.quiz_id else None
        elif item.item_type == "interview":
            target = (
                interviews_by_id.get(item.interview_config_id) if item.interview_config_id else None
            )
        else:
            target = None
        payload.append(
            {
                "id": item.id,
                "module_id": item.module_id,
                "item_type": item.item_type,
                "position": item.position,
                "target": target,
            }
        )
    return payload


async def list_published_course_tags(db: AsyncSession, course_id: UUID) -> list[Tag]:
    """Tags attached to a published course (404 path enforced by caller)."""
    stmt = (
        select(Tag)
        .join(CourseTag, CourseTag.tag_id == Tag.id)
        .join(Course, Course.id == CourseTag.course_id)
        .where(
            Course.id == course_id,
            published_course_clause(),
        )
        .order_by(Tag.name)
    )
    return list((await db.execute(stmt)).scalars().all())


async def list_published_course_outcomes(
    db: AsyncSession, course_id: UUID
) -> list[CourseLearningOutcome]:
    """Course learning outcomes for a published course (§A12).

    Ordered by ``(parent_id, position)`` so siblings are contiguous; the
    dotted ``L.O.x.y`` code and depth are derived by the service layer from
    the parent chain. Soft-deleted rows are excluded — leaving them in would
    both leak removed outcomes to learners and shift the derived codes.
    """
    stmt = (
        select(CourseLearningOutcome)
        .join(Course, Course.id == CourseLearningOutcome.course_id)
        .where(
            Course.id == course_id,
            CourseLearningOutcome.deleted_at.is_(None),
            published_course_clause(),
        )
        .order_by(CourseLearningOutcome.parent_id, CourseLearningOutcome.position)
    )
    return list((await db.execute(stmt)).scalars().all())


async def get_user_primary_organization_id(db: AsyncSession, user_id: UUID) -> UUID | None:
    """Resolve the requesting user's primary organization for catalog scoping.

    Delegates to :func:`access_control.api.public.get_user_primary_org`,
    which resolves via ``organization_memberships`` exclusively. Role
    assignments are not consulted -- belonging-to-org and
    permissions-in-org are independent concepts. Platform admins
    (``scope_kind='global'``) are NOT implicitly members of any one org
    and must hit endpoints that accept an explicit org param. Returns
    ``None`` for users without an active membership; the router then
    treats the catalog as empty.
    """
    org = await access_control_api.get_user_primary_org(db, user_id)
    return org.id if org is not None else None


async def get_visible_resource_storage_target(
    db: AsyncSession, resource_id: UUID
) -> tuple[str, str] | None:
    """Bucket + object_key for a student-visible resource (or ``None``).

    The composite visibility predicate inlines the same publish gates as
    :func:`get_visible_lesson_resource` so a 404 surfaces uniformly when
    any link in the lesson -> module -> course chain is unpublished /
    soft-deleted, OR when ``visible_to_students`` is FALSE. Caller (the
    catalog service) maps ``None`` to HTTP 404 — existence MUST NOT leak.
    """
    stmt = (
        select(StorageObject.bucket, StorageObject.object_key)
        .join(LessonResource, LessonResource.storage_object_id == StorageObject.id)
        .join(Lesson, Lesson.id == LessonResource.lesson_id)
        .join(Module, Module.id == Lesson.module_id)
        .join(Course, Course.id == Module.course_id)
        .where(
            LessonResource.id == resource_id,
            published_course_clause(),
            published_module_clause(),
            published_lesson_clause(),
            student_visible_resource_clause(),
        )
    )
    row = (await db.execute(stmt)).one_or_none()
    if row is None:
        return None
    return row.bucket, row.object_key


async def get_published_course_syllabus_storage_target(
    db: AsyncSession, course_id: UUID
) -> tuple[str, str] | None:
    """Bucket + object_key for a PUBLISHED course's syllabus PDF, or ``None``.

    The student-facing twin of the authoring query: same document, but
    gated on ``published_course_clause()`` so a draft course's syllabus is
    never reachable from the learner side. Returns the newest successful
    import, since re-importing a revised syllabus should hand out the
    current edition. The service maps ``None`` to HTTP 404 — a learner must
    not be able to tell "no syllabus" from "course not published".
    """
    stmt = (
        select(StorageObject.bucket, StorageObject.object_key)
        .join(
            CourseSyllabusImport,
            CourseSyllabusImport.storage_object_id == StorageObject.id,
        )
        .join(Course, Course.id == CourseSyllabusImport.course_id)
        .where(
            CourseSyllabusImport.course_id == course_id,
            CourseSyllabusImport.status == "succeeded",
            published_course_clause(),
        )
        .order_by(CourseSyllabusImport.created_at.desc())
        .limit(1)
    )
    row = (await db.execute(stmt)).one_or_none()
    if row is None:
        return None
    return row.bucket, row.object_key


async def published_course_has_syllabus(db: AsyncSession, course_id: UUID) -> bool:
    """Whether a PUBLISHED course has a downloadable syllabus document.

    Exists so the learner course payload can carry ``has_syllabus`` instead of
    the SPA probing the download endpoint and swallowing its 404. Reuses the
    same gate as the download itself, so the flag and the endpoint can never
    disagree — a true flag with a 404 behind it would be a broken button.
    """
    return await get_published_course_syllabus_storage_target(db, course_id) is not None


async def get_published_course_thumbnail_storage_target(
    db: AsyncSession, course_id: UUID
) -> tuple[str, str] | None:
    """Bucket + object_key for a published course's thumbnail image, or ``None``.

    Joins ``courses.thumbnail_object_id -> storage_objects.id`` gated on the
    published-course clause. Returns ``None`` when the course has no thumbnail
    or is unpublished. The service maps ``None`` to no thumbnail_url (the SPA
    falls back to the gradient banner).
    """
    stmt = (
        select(StorageObject.bucket, StorageObject.object_key)
        .join(Course, Course.thumbnail_object_id == StorageObject.id)
        .where(
            Course.id == course_id,
            published_course_clause(),
        )
    )
    row = (await db.execute(stmt)).one_or_none()
    if row is None:
        return None
    return row.bucket, row.object_key


__all__ = [
    "get_published_course_by_id",
    "get_published_course_by_slug",
    "get_published_course_content",
    "list_enrolled_courses",
    "list_published_courses",
    "list_published_lessons",
    "list_published_modules",
    "list_visible_lesson_resources",
    "published_course_has_syllabus",
]
