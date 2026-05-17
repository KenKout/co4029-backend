"""Quiz AI generation stages (T5.4-T5.9).

Each stage lives in its own sub-package: ``retrieval``, ``ideation``,
``drafting``, ``critique``, ``refinement``, ``persistence``. Stages are
composed by :mod:`abridgeai.features.quizzes.ai.pipelines` (T5.10) and
each one is independently testable with mocked dependencies.
"""

from __future__ import annotations
