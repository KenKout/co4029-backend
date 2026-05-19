from __future__ import annotations

from abridgeai.features.enrollments.routers.assignment import (
    dept_router as assignment_dept_router,
)
from abridgeai.features.enrollments.routers.assignment import (
    management_router as assignment_management_router,
)
from abridgeai.features.enrollments.routers.assignment import (
    teacher_router as assignment_teacher_router,
)
from abridgeai.features.enrollments.routers.learner import me_enrollments_router

__all__ = [
    "assignment_dept_router",
    "assignment_management_router",
    "assignment_teacher_router",
    "me_enrollments_router",
]
