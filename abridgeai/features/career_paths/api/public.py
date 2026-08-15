"""Public, typed cross-feature read API for the career_paths feature.

Sibling features (identity user-detail for managers/HODs) MUST import from
this module rather than reaching into ``queries``/``services`` directly.
Reads return plain dicts projected from the feature's own queries; no ORM
model escapes the feature boundary.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.career_paths.queries import student as student_queries


async def list_user_career_enrollments(
    db: AsyncSession, *, student_id: UUID
) -> list[dict[str, object]]:
    """Career paths ``student_id`` is enrolled in, newest first.

    Rows carry ``career_path_id``, ``status``, ``started_at``,
    ``completed_at``, ``slug`` and ``name`` (see the SQL in
    ``career_paths/queries/student.py``). Backs the manager/HOD
    user-detail "career path + progress" section.
    """
    return await student_queries.list_my_career_enrollments(db, student_id)


async def get_path_course_progress_for_user(
    db: AsyncSession,
    *,
    career_path_id: UUID,
    student_id: UUID,
) -> list[dict[str, object]]:
    """Per-course completion rows inside ``career_path_id`` for ``student_id``.

    Each row reports the course (id/slug/title/status/position), the stage
    it sits in, required-or-optional, and the student's unit tally
    (``unit_total``/``unit_done``/``completion_percent`` plus the
    enrollment-derived ``satisfied``/``is_enrolled`` flags). Backs the
    manager/HOD user-detail progress breakdown. Gap 3: reads the student's
    PINNED version (the route they actually walk).
    """
    from abridgeai.features.career_paths.queries import authoring as authoring_queries

    enrollment = await student_queries.get_my_career_enrollment(
        db, student_id=student_id, career_path_id=career_path_id
    )
    if enrollment is not None:
        version_id = enrollment.version_id
    else:
        published = await authoring_queries.get_published_version(db, career_path_id)
        if published is None:
            return []
        version_id = published.id
    return await student_queries.get_path_course_progress(
        db, version_id=version_id, student_id=student_id
    )


__all__ = [
    "get_path_course_progress_for_user",
    "list_user_career_enrollments",
]
