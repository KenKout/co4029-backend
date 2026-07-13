"""Audio extractor.

Two providers selected by ``settings.audio_extraction_local``:

* ``False`` (default) — calls the OpenAI-compatible Whisper transcription
  endpoint directly. The call writes one ``ai_model_calls`` row with
  ``role=stt`` / ``operation=chat_completion`` / ``stage_name="extraction"``
  via the same audit helper the gateway uses.
* ``True`` — runs ``faster-whisper`` locally. Optional dependency installed
  through the ``audio-local`` extra; absence raises a clear error.

Both paths return per-segment ``SourceLocation`` records carrying
``timestamp_start_ms`` / ``timestamp_end_ms`` so chunkers can attribute
text spans back to wall-clock positions in the source media.
"""

from __future__ import annotations

import asyncio
import io
import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any, BinaryIO

import httpx

from abridgeai.ai.extraction.base import ExtractedContent, SourceLocation
from abridgeai.ai.extraction.registry import register_extractor
from abridgeai.ai.llm.audit import write_ai_model_call
from abridgeai.ai.llm.errors import ConfigError, ProviderError
from abridgeai.ai.llm.pricing import compute_cost
from abridgeai.ai.llm.roles import LLMRole, ModelBinding, binding_for
from abridgeai.core.config import Settings, get_settings
from abridgeai.core.exceptions import AppError

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def _read_source(source: BinaryIO | bytes | str) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, str):
        with open(source, "rb") as fh:
            return fh.read()
    return source.read()


def _segments_to_locations(
    segments: list[dict[str, Any]],
) -> tuple[str, list[SourceLocation]]:
    parts: list[str] = []
    locations: list[SourceLocation] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = seg.get("start")
        end = seg.get("end")
        parts.append(text)
        locations.append(
            SourceLocation(
                timestamp_start_ms=int(round(float(start) * 1000)) if start is not None else None,
                timestamp_end_ms=int(round(float(end) * 1000)) if end is not None else None,
            )
        )
    return "\n".join(parts).strip(), locations


def _faster_whisper_transcribe(raw: bytes, *, model_name: str) -> dict[str, Any]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise AppError(
            "audio_extraction_local=True but faster-whisper is not installed; "
            "install the 'audio-local' extra (uv sync --extra audio-local)"
        ) from exc

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments_iter, info = model.transcribe(io.BytesIO(raw))
    segments = [
        {"start": float(s.start), "end": float(s.end), "text": s.text} for s in segments_iter
    ]
    return {
        "segments": segments,
        "language": getattr(info, "language", None),
        "duration": getattr(info, "duration", None),
    }


@register_extractor("audio/wav")
@register_extractor("audio/x-wav")
@register_extractor("audio/mpeg")
@register_extractor("audio/mp3")
@register_extractor("audio/m4a")
@register_extractor("audio/x-m4a")
@register_extractor("audio/mp4")
@register_extractor("audio/flac")
@register_extractor("audio/ogg")
@register_extractor("audio/webm")
class AudioExtractor:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        db: AsyncSession | None = None,
        stage_name: str = "extraction",
    ) -> None:
        self._settings = settings or get_settings()
        self._db = db
        self._stage_name = stage_name

    async def extract(self, source: BinaryIO | bytes | str) -> ExtractedContent:
        raw = await asyncio.to_thread(_read_source, source)
        if self._settings.audio_extraction_local:
            return await self._extract_local(raw)
        return await self._extract_whisper_api(raw)

    async def _extract_local(self, raw: bytes) -> ExtractedContent:
        result = await asyncio.to_thread(
            _faster_whisper_transcribe, raw, model_name=self._settings.whisper_model
        )
        text, locations = _segments_to_locations(result["segments"])
        return ExtractedContent(
            text=text,
            metadata={
                "stt_provider": "faster_whisper",
                "model_name": self._settings.whisper_model,
                "language": result.get("language"),
                "duration_seconds": result.get("duration"),
                "segment_count": len(locations),
            },
            source_type="audio",
            source_locations=locations,
        )

    async def _extract_whisper_api(self, raw: bytes) -> ExtractedContent:
        if self._db is None:
            raise RuntimeError(
                "AudioExtractor whisper-api path requires an AsyncSession for "
                "audit-row writes; pass db= when instantiating outside the "
                "registry dispatch path."
            )
        binding = binding_for(LLMRole.STT, self._settings)
        whisper_model = self._settings.whisper_model or binding.model
        url = f"{binding.base_url.rstrip('/')}/audio/transcriptions"
        request_payload: dict[str, Any] = {
            "model": whisper_model,
            "response_format": "verbose_json",
            "audio_bytes": len(raw),
        }
        headers = {"Authorization": f"Bearer {binding.api_key}", **binding.extra_headers}

        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=binding.timeout_s) as client:
                response = await client.post(
                    url,
                    headers=headers,
                    files={"file": ("audio", raw, "application/octet-stream")},
                    data={
                        "model": whisper_model,
                        "response_format": "verbose_json",
                    },
                )
        except httpx.HTTPError as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            await self._audit_failure(
                binding=binding,
                model_name=whisper_model,
                request_payload=request_payload,
                response_payload=None,
                latency_ms=latency_ms,
                error=str(exc),
            )
            raise AppError(f"Whisper API call failed: {exc}") from exc

        latency_ms = int((time.perf_counter() - start) * 1000)
        if response.status_code >= 400:
            await self._audit_failure(
                binding=binding,
                model_name=whisper_model,
                request_payload=request_payload,
                response_payload={"status": response.status_code, "body": response.text[:1000]},
                latency_ms=latency_ms,
                error=f"HTTP {response.status_code}",
            )
            raise AppError(f"Whisper API returned {response.status_code}: {response.text[:200]}")

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            await self._audit_failure(
                binding=binding,
                model_name=whisper_model,
                request_payload=request_payload,
                response_payload={"body": response.text[:1000]},
                latency_ms=latency_ms,
                error=f"non-JSON response: {exc}",
            )
            raise AppError("Whisper API returned non-JSON body") from exc

        segments = payload.get("segments") or []
        text, locations = _segments_to_locations(segments)
        if not text and isinstance(payload.get("text"), str):
            text = payload["text"].strip()

        cost = await compute_cost(self._db, whisper_model, None, None)
        await write_ai_model_call(
            self._db,
            role=LLMRole.STT,
            tier=binding.tier,
            operation="chat_completion",
            model_name=whisper_model,
            base_url=binding.base_url,
            stage_name=self._stage_name,
            pipeline_run_id=None,
            parent_run_id=None,
            parent_job_id=None,
            request_payload=request_payload,
            response_payload=payload,
            input_tokens=None,
            output_tokens=None,
            cached_input_tokens=None,
            latency_ms=latency_ms,
            status="success",
            error_message=None,
            estimated_cost_usd=cost,
        )

        return ExtractedContent(
            text=text,
            metadata={
                "stt_provider": "whisper_api",
                "model_name": whisper_model,
                "language": payload.get("language"),
                "duration_seconds": payload.get("duration"),
                "segment_count": len(locations),
            },
            source_type="audio",
            source_locations=locations,
        )

    async def _audit_failure(
        self,
        *,
        binding: ModelBinding,
        model_name: str,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any] | None,
        latency_ms: int,
        error: str,
    ) -> None:
        if self._db is None:
            return
        try:
            await write_ai_model_call(
                self._db,
                role=LLMRole.STT,
                tier=binding.tier,
                operation="chat_completion",
                model_name=model_name,
                base_url=binding.base_url,
                stage_name=self._stage_name,
                pipeline_run_id=None,
                parent_run_id=None,
                parent_job_id=None,
                request_payload=request_payload,
                response_payload=response_payload,
                input_tokens=None,
                output_tokens=None,
                cached_input_tokens=None,
                latency_ms=latency_ms,
                status="failed",
                error_message=error,
                estimated_cost_usd=Decimal("0"),
            )
        except (ConfigError, ProviderError):
            return


__all__ = ["AudioExtractor"]
