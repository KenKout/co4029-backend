"""Quiz generation runner. Wires `quiz_generation` capability to the
`abridgeai.features.quizzes.ai` pipeline. T8.2 ships the stub.
"""

from __future__ import annotations

from eval.runners.base import CapabilityRunner


class QuizRunner(CapabilityRunner):
    capability = "quiz_generation"
