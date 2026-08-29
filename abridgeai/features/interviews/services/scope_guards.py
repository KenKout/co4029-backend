"""Cross-course scope guards for interview authoring.

Interview authoring surfaces accept teacher-supplied course / module /
lesson scoping (config creation pins a module; generation takes
``source_module_ids`` / ``source_lesson_ids``). Those ids must never
escape the course the teacher is actually authorized for — a request
authorized for one interview must not be able to retrieve material from
another course or attach an interview to a foreign module.

These helpers resolve the supplied id *to its owning row* and verify the
owning course matches the authorized one. They live in their own module
because ``services/authoring.py`` sits at the 800-LOC metric gate and
every scope check added there pushes it back over.
"""

from __future__ import annotations

from uuid import UUID

from abridgeai.core.exceptions import AppError
from abridgeai.features.courses.api import public as courses_public


async def require_module_in_course(
    db: object,
    module_id: UUID,
    course_id: UUID,
) -> None:
    """Raise unless ``module_id`` resolves to a module of ``course_id``."""
    module = await courses_public.get_module_by_id(db, module_id)
    if module is None or module.course_id != course_id:
        raise AppError(f"Module {module_id} is not part of course {course_id}")


async def require_lessons_in_course(
    db: object,
    lesson_ids: list[UUID],
    course_id: UUID,
) -> None:
    """Raise unless every lesson exists and belongs to a module of ``course_id``."""
    for lesson_id in lesson_ids:
        lesson = await courses_public.get_lesson_by_id(db, lesson_id)
        if lesson is None:
            raise AppError(f"Lesson {lesson_id} not found")
        module = await courses_public.get_module_by_id(db, lesson.module_id)
        if module is None or module.course_id != course_id:
            raise AppError(f"Lesson {lesson_id} is not part of course {course_id}")
