"""Interview pipeline orchestrators (T6.10+).

Each pipeline composes stages from ``..stages/``. Future pipelines (post-submit
evaluation, gap-report) will APPEND their re-exports below; do not reorder.
"""

from __future__ import annotations

from abridgeai.features.interviews.ai.pipelines.generation import (
    run_interview_generation,
)

__all__ = ["run_interview_generation"]
