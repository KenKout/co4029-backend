"""LLM chat-completion gateway.

One ``POST /chat/completions`` round-trip per call, role-aware via
``binding_for``. Writes exactly one ``ai_model_calls`` row per call (success
or failed). Re-raises ``AppError`` on any failure so the worker can mark the
job ``failed``.

This module does not know about embeddings — see ``embeddings.py``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.ai.llm.audit import write_ai_model_call
from abridgeai.ai.llm.client import OpenAICompatibleClient
from abridgeai.ai.llm.errors import ConfigError, ProviderError, ResponseFormatError
from abridgeai.ai.llm.pricing import compute_cost
from abridgeai.ai.llm.roles import LLMRole, binding_for
from abridgeai.core.config import Settings, get_settings
from abridgeai.core.exceptions import AppError

# Robust JSON parser for LLM responses. OpenAI's strict ``response_format=
# {"type": "json_object"}`` produces clean JSON; many open-source / OpenAI-
# compatible providers (gpt-oss, Ollama, vLLM with some templates) ignore
# that flag and return either:
#   - a single JSON object wrapped in a ```json fence;
#   - the JSON object followed by a trailing newline + extra commentary;
#   - two or more JSON objects concatenated ("Extra data" decode error).
# We always prefer the first valid JSON object the parser can extract.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def _parse_llm_json(content: str) -> dict[str, Any] | list[Any]:
    """Parse an LLM response into a JSON object/array.

    Order:
      1. Try strict ``json.loads``. Most calls succeed here.
      2. If the content has a fenced ``json`` code block, parse the first.
      3. Use ``raw_decode`` to take the first valid JSON object and ignore
         trailing junk (handles the "Extra data" case).

    Raises ``json.JSONDecodeError`` if all three strategies fail so the
    caller can write a ``failed`` audit row.
    """
    parsed: dict[str, Any] | list[Any]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        pass
    else:
        return parsed

    fence_match = _JSON_FENCE_RE.search(content)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass
        else:
            return parsed

    # Last resort: raw_decode peels off the first valid JSON value and
    # returns the index where parsing stopped. We discard everything after.
    stripped = content.lstrip()
    if stripped:
        try:
            value, _ = json.JSONDecoder().raw_decode(stripped)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(value, (dict, list)):
                return value

    # Re-raise the original strict failure so the caller sees the real
    # parse position.
    parsed = json.loads(content)
    return parsed


@dataclass(frozen=True)
class LLMResult:
    """Outcome of a single chat-completion call.

    Mirrors what callers used to read off the old ``LLMResult`` plus the new
    attribution fields (role/tier/stage_name/pipeline_run_id/base_url/
    total_tokens/...). The legacy ``provider`` field is gone — see the
    chunk-2 schema migration.
    """

    role: LLMRole
    tier: str | None
    model_name: str
    base_url: str
    stage_name: str | None
    pipeline_run_id: UUID | None
    request_payload: dict[str, Any]
    response_payload: dict[str, Any]
    content_json: dict[str, Any] | list[Any]
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cached_input_tokens: int | None
    latency_ms: int
    estimated_cost_usd: Decimal | None


class LLMGateway:
    """One method, one HTTP path, one audit row per call."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def generate_json(
        self,
        *,
        role: LLMRole,
        system_prompt: str,
        user_prompt: str,
        db: AsyncSession,
        stage_name: str | None = None,
        pipeline_run_id: UUID | None = None,
        parent_job_id: UUID | None = None,
        parent_run_id: UUID | None = None,
    ) -> LLMResult:
        if role is LLMRole.EMBEDDING:
            raise ConfigError(
                "LLMGateway.generate_json was called with role=EMBEDDING; "
                "use EmbeddingClient.embed instead"
            )

        binding = binding_for(role, self._settings)
        client = OpenAICompatibleClient(binding)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        request_payload: dict[str, Any] = {
            "model": binding.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }

        try:
            response_body, latency_ms = await client.chat_completions_json(
                messages, response_format={"type": "json_object"}
            )
        except (ProviderError, ResponseFormatError) as exc:
            await write_ai_model_call(
                db,
                role=role,
                tier=binding.tier,
                operation="chat_completion",
                model_name=binding.model,
                base_url=binding.base_url,
                stage_name=stage_name,
                pipeline_run_id=pipeline_run_id,
                parent_run_id=parent_run_id,
                parent_job_id=parent_job_id,
                request_payload=request_payload,
                response_payload=None,
                input_tokens=None,
                output_tokens=None,
                cached_input_tokens=None,
                latency_ms=0,
                status="failed",
                error_message=str(exc),
                estimated_cost_usd=None,
            )
            raise AppError(str(exc)) from exc

        usage = response_body.get("usage") or {}
        input_tokens: int | None = usage.get("prompt_tokens")
        output_tokens: int | None = usage.get("completion_tokens")
        cached_input_tokens: int | None = (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        )
        cost = compute_cost(binding.model, input_tokens, output_tokens)

        try:
            content_str = response_body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            await write_ai_model_call(
                db,
                role=role,
                tier=binding.tier,
                operation="chat_completion",
                model_name=binding.model,
                base_url=binding.base_url,
                stage_name=stage_name,
                pipeline_run_id=pipeline_run_id,
                parent_run_id=parent_run_id,
                parent_job_id=parent_job_id,
                request_payload=request_payload,
                response_payload=response_body,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                latency_ms=latency_ms,
                status="failed",
                error_message=f"unexpected response shape: {exc}",
                estimated_cost_usd=cost,
            )
            raise AppError(f"LLM response had unexpected shape: {exc}") from exc

        try:
            content_json = _parse_llm_json(content_str)
        except json.JSONDecodeError as exc:
            await write_ai_model_call(
                db,
                role=role,
                tier=binding.tier,
                operation="chat_completion",
                model_name=binding.model,
                base_url=binding.base_url,
                stage_name=stage_name,
                pipeline_run_id=pipeline_run_id,
                parent_run_id=parent_run_id,
                parent_job_id=parent_job_id,
                request_payload=request_payload,
                response_payload=response_body,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                latency_ms=latency_ms,
                status="failed",
                error_message=f"LLM response was not valid JSON: {exc}",
                estimated_cost_usd=cost,
            )
            raise AppError("LLM response was not valid JSON") from exc

        await write_ai_model_call(
            db,
            role=role,
            tier=binding.tier,
            operation="chat_completion",
            model_name=binding.model,
            base_url=binding.base_url,
            stage_name=stage_name,
            pipeline_run_id=pipeline_run_id,
            parent_run_id=parent_run_id,
            parent_job_id=parent_job_id,
            request_payload=request_payload,
            response_payload=response_body,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            latency_ms=latency_ms,
            status="success",
            error_message=None,
            estimated_cost_usd=cost,
        )

        total_tokens: int | None
        if input_tokens is not None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
        else:
            total_tokens = None

        return LLMResult(
            role=role,
            tier=binding.tier,
            model_name=binding.model,
            base_url=binding.base_url,
            stage_name=stage_name,
            pipeline_run_id=pipeline_run_id,
            request_payload=request_payload,
            response_payload=response_body,
            content_json=content_json,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_input_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
        )
