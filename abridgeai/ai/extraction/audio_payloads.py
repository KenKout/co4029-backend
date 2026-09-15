"""Provider-payload normalization for audio extraction."""

from __future__ import annotations

from typing import Any

from abridgeai.ai.extraction.base import SourceLocation


def parse_deepgram_payload(
    payload: dict[str, Any],
) -> tuple[str, list[SourceLocation], dict[str, Any]]:
    """Map a Deepgram response to transcript text, locations, and metadata."""
    results = payload.get("results") or {}
    utterances = results.get("utterances") or []

    parts: list[str] = []
    locations: list[SourceLocation] = []
    for utterance in utterances:
        text = (utterance.get("transcript") or "").strip()
        if not text:
            continue
        start = utterance.get("start")
        end = utterance.get("end")
        parts.append(text)
        locations.append(
            SourceLocation(
                timestamp_start_ms=(
                    int(round(float(start) * 1000)) if start is not None else None
                ),
                timestamp_end_ms=int(round(float(end) * 1000)) if end is not None else None,
            )
        )

    channels = results.get("channels") or []
    first_alt: dict[str, Any] = {}
    if channels:
        alternatives = channels[0].get("alternatives") or []
        if alternatives:
            first_alt = alternatives[0]

    if not parts:
        flat = (first_alt.get("transcript") or "").strip()
        if flat:
            parts.append(flat)
            locations.append(SourceLocation(timestamp_start_ms=0, timestamp_end_ms=None))

    language = first_alt.get("language")
    if not language and channels:
        language = channels[0].get("detected_language")
    metadata: dict[str, Any] = {
        "duration": (payload.get("metadata") or {}).get("duration"),
        "language": language,
    }
    return "\n".join(parts).strip(), locations, metadata


__all__ = ["parse_deepgram_payload"]
