"""Public re-exports for the courses-feature schema module (T3.2).

Three concerns split across sibling files:

* :mod:`.public`     — student-facing DTOs (no audit / draft / internal
  notes); status narrows to ``Literal["published"]``.
* :mod:`.authoring`  — teacher-facing DTOs that inherit from Public and
  widen with audit + soft-delete + status="draft|archived" + unlock
  config.
* :mod:`.request`    — request bodies for create / update / reorder /
  publish flows.
"""

from __future__ import annotations

from abridgeai.features.courses.schemas.authoring import (
    CourseAuthoring,
    CourseContentAuthoring,
    CourseLearningOutcomeAuthoring,
    InstructorAuthoring,
    LessonAuthoring,
    LessonResourceAuthoring,
    ModuleAuthoring,
    ModuleItemAuthoring,
    TagAuthoring,
)
from abridgeai.features.courses.schemas.public import (
    CourseContentPublic,
    CourseLearningOutcomePublic,
    CoursePublic,
    InstructorRead,
    LessonPublic,
    LessonResourcePublic,
    ModuleItemPublic,
    ModulePublic,
    TagPublic,
)
from abridgeai.features.courses.schemas.request import (
    CourseArchiveRequest,
    CourseCreate,
    CoursePublishRequest,
    CourseUpdate,
    LessonCreate,
    LessonResourceCreate,
    LessonResourceUpdate,
    LessonUpdate,
    ModuleCreate,
    ModuleItemReorder,
    ModuleItemUpdate,
    ModulePrerequisiteSet,
    ModuleUpdate,
)

__all__ = [
    "CourseArchiveRequest",
    "CourseAuthoring",
    "CourseContentAuthoring",
    "CourseContentPublic",
    "CourseCreate",
    "CourseLearningOutcomeAuthoring",
    "CourseLearningOutcomePublic",
    "CoursePublic",
    "CoursePublishRequest",
    "CourseUpdate",
    "InstructorAuthoring",
    "InstructorRead",
    "LessonAuthoring",
    "LessonCreate",
    "LessonPublic",
    "LessonResourceAuthoring",
    "LessonResourceCreate",
    "LessonResourcePublic",
    "LessonResourceUpdate",
    "LessonUpdate",
    "ModuleAuthoring",
    "ModuleCreate",
    "ModuleItemAuthoring",
    "ModuleItemPublic",
    "ModuleItemReorder",
    "ModuleItemUpdate",
    "ModulePrerequisiteSet",
    "ModulePublic",
    "ModuleUpdate",
    "TagAuthoring",
    "TagPublic",
]
