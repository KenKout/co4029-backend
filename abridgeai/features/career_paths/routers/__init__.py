from __future__ import annotations

from abridgeai.features.career_paths.routers.authoring import (
    management_router as authoring_management_router,
)
from abridgeai.features.career_paths.routers.authoring import (
    teacher_router as authoring_teacher_router,
)
from abridgeai.features.career_paths.routers.learner import (
    me_router as me_career_enrollments_router,
)
from abridgeai.features.career_paths.routers.learner import (
    router as career_paths_learner_router,
)

__all__ = [
    "authoring_management_router",
    "authoring_teacher_router",
    "career_paths_learner_router",
    "me_career_enrollments_router",
]
