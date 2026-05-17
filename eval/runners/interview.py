"""Interview generation runner. Wires `interview_generation` capability to
the `abridgeai.features.interviews.ai` pipeline. T8.2 ships the stub.
"""

from __future__ import annotations

from eval.runners.base import CapabilityRunner


class InterviewRunner(CapabilityRunner):
    capability = "interview_generation"
