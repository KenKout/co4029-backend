"""Single writer for ``ai_model_calls`` rows.

Only ``LLMGateway`` and ``EmbeddingClient`` are allowed to call
``write_ai_model_call``. Pipeline call sites must not construct
``AIModelCall`` directly — see chunk-3 task 3.12 onward in the spec.

The function ``flush()``es into the caller's transaction; it does not
``commit()``. Successful calls roll back together with the pipeline's other
work on a downstream error. Failed-call rows are committed by the
pipeline's outer exception handler in a fresh transaction (the existing
pattern in the worker).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.ai.llm.roles import LLMRole
from abridgeai.ai.models import AIModelCall

_ERROR_MESSAGE_LIMIT = 2000


async def write_ai_model_call(
    db: AsyncSession,
    *,
    role: LLMRole,
    tier: str | None,
    operation: Literal["chat_completion", "embedding"],
    model_name: str,
    base_url: str,
    stage_name: str | None,
    pipeline_run_id: UUID | None,
    parent_run_id: UUID | None,
    parent_job_id: UUID | None,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any] | None,
    input_tokens: int | None,
    output_tokens: int | None,
    cached_input_tokens: int | None,
    latency_ms: int,
    status: Literal["success", "failed"],
    error_message: str | None,
    estimated_cost_usd: Decimal | None,
) -> AIModelCall:
    """Insert one ``ai_model_calls`` row attributing the call to ``role``.

    Returns the created row so callers may inspect ``id`` if needed.
    """
    if input_tokens is not None:
        total_tokens: int | None = (input_tokens or 0) + (output_tokens or 0)
    else:
        total_tokens = None

    truncated_error: str | None
    if error_message is not None and len(error_message) > _ERROR_MESSAGE_LIMIT:
        truncated_error = error_message[:_ERROR_MESSAGE_LIMIT]
    else:
        truncated_error = error_message

    row = AIModelCall(
        generation_run_id=parent_run_id,
        processing_job_id=parent_job_id,
        pipeline_run_id=pipeline_run_id,
        role=role.value,
        tier=tier,
        operation=operation,
        model_name=model_name,
        base_url=base_url,
        stage_name=stage_name,
        request_payload=request_payload,
        response_payload=response_payload,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
        estimated_cost_usd=estimated_cost_usd,
        latency_ms=latency_ms,
        status=status,
        error_message=truncated_error,
    )
    db.add(row)
    await db.flush()
    return row
