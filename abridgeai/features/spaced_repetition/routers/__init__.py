"""Spaced-repetition feature routers (T7.5.12)."""

from __future__ import annotations

from .learner import router as learner_router
from .teacher import router as teacher_router

__all__ = ["learner_router", "teacher_router"]
