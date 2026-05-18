"""Public API package for the quizzes feature.

Re-exports the cross-feature read surface from
:mod:`abridgeai.features.quizzes.api.public`. Other features must
import from ``abridgeai.features.quizzes.api.public`` (not from
``abridgeai.features.quizzes.models`` or ``.services``).
"""

from __future__ import annotations

from abridgeai.features.quizzes.api import public

__all__ = ["public"]
