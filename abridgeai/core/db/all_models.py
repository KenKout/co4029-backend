"""Import every ORM models module so the mapper registry is complete.

Why this exists: the content-tree refactor gave ``courses.ModuleItem``
string-referenced relationships to other features' classes
(``relationship("Quiz")``, ``relationship("InterviewConfig")``). SQLAlchemy
resolves those names at mapper-configure time from the registry — so any
process that imports ``courses.models`` without ALSO having imported
``quizzes.models`` and ``interviews.models`` dies on first flush with
``expression 'Quiz' failed to locate a name``.

The API and workers import every router and never hit this; the contexts
that do are standalone ones — a single test file run in isolation, a script,
a shell session. Importing this module makes the registry whole regardless
of entry point. ``tests/conftest.py`` imports it for exactly that reason.

Imports are for side effect only (class registration on ``Base.registry``);
nothing is re-exported.
"""

from __future__ import annotations

import abridgeai.ai.models  # noqa: F401
import abridgeai.features.access_control.models  # noqa: F401
import abridgeai.features.career_paths.models  # noqa: F401
import abridgeai.features.courses.models  # noqa: F401
import abridgeai.features.discussions.models  # noqa: F401
import abridgeai.features.enrollments.models  # noqa: F401
import abridgeai.features.identity.models  # noqa: F401
import abridgeai.features.interviews.models  # noqa: F401
import abridgeai.features.learning_programs.models  # noqa: F401
import abridgeai.features.materials.models  # noqa: F401
import abridgeai.features.notifications.models  # noqa: F401
import abridgeai.features.progress.models  # noqa: F401
import abridgeai.features.quizzes.models  # noqa: F401
import abridgeai.features.spaced_repetition.models  # noqa: F401

__all__: list[str] = []
