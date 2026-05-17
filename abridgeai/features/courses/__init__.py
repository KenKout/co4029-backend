"""Courses feature public re-exports.

Per Reconciliation §A1, the courses aggregate has 8 ORM models +
2 self-edge association tables. ``CareerCourseItem`` is intentionally
absent — it ports to ``features/career_paths/`` in Phase 7. Same for
``Enrollment`` / ``InvitationCode`` (port to ``features/enrollments/``).
"""

from abridgeai.features.courses import services
from abridgeai.features.courses.models import (
    Course,
    CourseLearningOutcome,
    CourseTag,
    Lesson,
    LessonPrerequisite,
    LessonResource,
    LessonUnlockConfig,
    Module,
    ModuleItem,
    ModulePrerequisite,
    Tag,
)

__all__ = [
    "Course",
    "CourseLearningOutcome",
    "CourseTag",
    "Lesson",
    "LessonPrerequisite",
    "LessonResource",
    "LessonUnlockConfig",
    "Module",
    "ModuleItem",
    "ModulePrerequisite",
    "Tag",
    "services",
]
