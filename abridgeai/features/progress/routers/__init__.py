from __future__ import annotations

from .authoring import router as authoring_router
from .learner import router as learner_router

__all__ = ["authoring_router", "learner_router"]
