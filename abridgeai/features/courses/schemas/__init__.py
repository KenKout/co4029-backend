"""Public re-exports for the courses-feature schema module (T3.2).

Four concerns split across sibling files:

* :mod:`.public`          — student-facing DTOs (no audit / draft / internal
  notes); status narrows to ``Literal["published"]``.
* :mod:`.authoring`       — teacher-facing DTOs that inherit from Public and
  widen with audit + soft-delete + status="draft|archived" + unlock config.
* :mod:`.request`         — request bodies for create / update / reorder /
  publish flows.
* :mod:`.administration`  — IT Admin response DTOs (processing audit, stats).
* :mod:`.assignment`      — HOD/Manager teacher-assignment DTOs.
"""

from __future__ import annotations

from abridgeai.features.courses.schemas.administration import (
    AdminCoursePage,
    CourseProcessingAudit,
    CourseStats,
    CourseStatusCount,
    ProcessingJobRow,
    TopOwnerRow,
)
from abridgeai.features.courses.schemas.assignment import (
    AssignableTeacher,
    AssignTeacherRequest,
    CoursePathPlacement,
    CourseReadiness,
    CourseRosterRead,
    RosterEntry,
    RosterStudentRead,
    TeacherAssignmentCreated,
    TeacherAssignmentRead,
)
from abridgeai.features.courses.schemas.authoring import (
    CourseAuthoring,
    CourseContentAuthoring,
    CourseLearningOutcomeAuthoring,
    CourseLearningOutcomeCreate,
    CourseLearningOutcomeUpdate,
    InstructorAuthoring,
    LessonAuthoring,
    LessonOutline,
    LessonResourceAuthoring,
    ModuleAuthoring,
    ModuleItemAuthoring,
    OutlineSection,
    ReviewQueueItem,
    SlugAvailability,
    StreamUrlResponse,
    TagAuthoring,
    TeacherDashboardStats,
)
from abridgeai.features.courses.schemas.public import (
    CourseContentPublic,
    CourseLearningOutcomePublic,
    CoursePage,
    CoursePublic,
    InstructorRead,
    LessonPublic,
    LessonResourcePublic,
    ModuleItemPublic,
    ModulePublic,
    ResourceDownloadUrlResponse,
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
    LessonUnlockConfig,
    LessonUpdate,
    ModuleCreate,
    ModuleItemReorder,
    ModuleItemUpdate,
    ModulePrerequisiteSet,
    ModuleReorder,
    ModuleUpdate,
)

__all__ = [
    "AdminCoursePage",
    "AssignTeacherRequest",
    "AssignableTeacher",
    "CoursePathPlacement",
    "CourseReadiness",
    "CourseArchiveRequest",
    "CourseAuthoring",
    "ReviewQueueItem",
    "TeacherDashboardStats",
    "CourseContentAuthoring",
    "CourseContentPublic",
    "CourseCreate",
    "CourseRosterRead",
    "CourseLearningOutcomeAuthoring",
    "CourseLearningOutcomeCreate",
    "CourseLearningOutcomeUpdate",
    "CourseLearningOutcomePublic",
    "CoursePage",
    "CourseProcessingAudit",
    "CoursePublic",
    "CoursePublishRequest",
    "CourseStats",
    "CourseStatusCount",
    "CourseUpdate",
    "InstructorAuthoring",
    "InstructorRead",
    "LessonAuthoring",
    "LessonCreate",
    "LessonOutline",
    "LessonPublic",
    "LessonResourceAuthoring",
    "LessonResourceCreate",
    "LessonResourcePublic",
    "LessonResourceUpdate",
    "LessonUnlockConfig",
    "LessonUpdate",
    "ModuleAuthoring",
    "ModuleCreate",
    "ModuleItemAuthoring",
    "ModuleItemPublic",
    "ModuleItemReorder",
    "ModuleReorder",
    "ModuleItemUpdate",
    "ModulePrerequisiteSet",
    "ModulePublic",
    "ModuleUpdate",
    "OutlineSection",
    "ProcessingJobRow",
    "ResourceDownloadUrlResponse",
    "RosterEntry",
    "RosterStudentRead",
    "SlugAvailability",
    "StreamUrlResponse",
    "TagAuthoring",
    "TagPublic",
    "TeacherAssignmentCreated",
    "TeacherAssignmentRead",
    "TopOwnerRow",
]
