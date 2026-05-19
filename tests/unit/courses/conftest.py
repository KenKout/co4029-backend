"""Conftest for courses unit tests.

Ensures cross-feature ORM models referenced by string relationships
in courses models (Quiz, InterviewConfig) are registered with
SQLAlchemy's mapper before any test triggers mapper initialization.
"""

import abridgeai.features.interviews.models  # noqa: F401 - register InterviewConfig
import abridgeai.features.quizzes.models  # noqa: F401 - register Quiz
