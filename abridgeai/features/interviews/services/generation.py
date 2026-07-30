"""Interview generation pipeline entrypoint (T6.11).

Top-level dispatcher invoked by the ARQ worker for the
``run_interview_generation_task`` job. Delegates to the T6.10
pipeline orchestrator which owns retrieval → generation → validation
→ persistence + the run-status state machine.

Mirrors :mod:`features.quizzes.services.generation` — the worker is
unaware of the pipeline internals; everything routes through this
single entrypoint so future changes (e.g., per-question regeneration
fan-out) localize here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.features.interviews.ai.pipelines.generation import (
    run_interview_generation as _run_interview_generation_pipeline,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def run_interview_generation(
    db: AsyncSession,
    generation_run_id: UUID,
    *,
    arq_pool: object | None = None,
) -> None:
    """ARQ entrypoint: delegate to the T6.10 pipeline.

    ``arq_pool`` (the worker's ``ctx['redis']``) is threaded through solely so
    the completion/failure notification can enqueue an email; ``None`` in
    sync/test paths simply skips the email fan-out.
    """
    await _run_interview_generation_pipeline(db, generation_run_id, arq_pool=arq_pool)


__all__ = ["run_interview_generation"]
