"""Courses feature routers.

Mounted in Phase 3 integration (T3.10) via::

    app.include_router(learner_router, prefix="/api/v1")
    app.include_router(me_courses_router, prefix="/api/v1")
    # T3.7+ will add: app.include_router(authoring_router, prefix="/api/v1")
    # T3.8 will add:  app.include_router(assignment_router, prefix="/api/v1")
    # T3.9 will add:  app.include_router(administration_router, prefix="/api/v1")

T3.7 / T3.8 / T3.9 SHOULD APPEND their re-exports below the learner imports
(preserve docstring + alphabetic ordering inside ``__all__``).
"""

from __future__ import annotations

from .administration import router as administration_router
from .assignment import router as assignment_router
from .authoring import router as authoring_router
from .learner import me_courses_router
from .learner import router as learner_router
from .management_dashboard import router as management_dashboard_router

__all__ = [
    "administration_router",
    "assignment_router",
    "authoring_router",
    "management_dashboard_router",
    "learner_router",
    "me_courses_router",
]
