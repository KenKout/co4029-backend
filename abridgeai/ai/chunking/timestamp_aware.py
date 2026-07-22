"""Timestamp-aware chunker for audio/video transcripts.

Reads ``ExtractedContent.source_locations`` and treats any gap longer than
``silence_gap_ms`` between consecutive locations' ``timestamp_end_ms`` and
the next location's ``timestamp_start_ms`` as a hard chunk boundary. Each
chunk's metadata carries ``timestamp_start_ms`` + ``timestamp_end_ms`` so
the frontend can deep-link to the moment in the source recording.

The text body is segmented in lockstep with ``source_locations`` — one
``SourceLocation`` per logical transcript segment (typically one utterance
or one diarization turn). When the body cannot be aligned (no segments
have timestamps), the chunker falls back to returning a single chunk
covering the whole content.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from abridgeai.ai.chunking.base import RawChunk

if TYPE_CHECKING:
    from abridgeai.ai.extraction import ExtractedContent, SourceLocation


class TimestampAwareChunker:
    """Split transcripts on silence gaps."""

    def __init__(
        self,
        *,
        silence_gap_ms: int = 2000,
        max_chunk_ms: int | None = None,
    ) -> None:
        if silence_gap_ms <= 0:
            raise ValueError("silence_gap_ms must be positive")
        self._silence_gap_ms = silence_gap_ms
        self._max_chunk_ms = max_chunk_ms

    def chunk(self, content: ExtractedContent, **opts: Any) -> list[RawChunk]:  # noqa: ANN401 -- forwarded chunker kwargs
        silence_gap_ms = int(opts.get("silence_gap_ms", self._silence_gap_ms))
        max_chunk_ms_opt = opts.get("max_chunk_ms", self._max_chunk_ms)
        max_chunk_ms = int(max_chunk_ms_opt) if max_chunk_ms_opt is not None else None

        segments = _segments_with_timestamps(content)
        if not segments:
            text = (content.text or "").strip()
            if not text:
                return []
            return [
                RawChunk(
                    content=text,
                    chunk_index=0,
                    metadata={
                        "source_type": content.source_type,
                        "timestamp_start_ms": None,
                        "timestamp_end_ms": None,
                    },
                )
            ]

        groups: list[list[tuple[str, SourceLocation]]] = []
        current: list[tuple[str, SourceLocation]] = []
        for seg_text, loc in segments:
            if not current:
                current.append((seg_text, loc))
                continue
            prev_end = current[-1][1].timestamp_end_ms
            curr_start = loc.timestamp_start_ms
            gap = (curr_start or 0) - (prev_end or 0)
            window_ms = (loc.timestamp_end_ms or 0) - (current[0][1].timestamp_start_ms or 0)
            should_break = gap >= silence_gap_ms or (
                max_chunk_ms is not None and window_ms > max_chunk_ms
            )
            if should_break:
                groups.append(current)
                current = [(seg_text, loc)]
            else:
                current.append((seg_text, loc))
        if current:
            groups.append(current)

        return [
            _build_chunk(group, index=i, source_type=content.source_type)
            for i, group in enumerate(groups)
        ]


def _segments_with_timestamps(
    content: ExtractedContent,
) -> list[tuple[str, SourceLocation]]:
    locations = list(content.source_locations or [])
    if not locations:
        return []

    # Align the text body to source_locations one segment per location.
    # Audio transcripts join utterances with "\n\n"; video frame-OCR joins
    # per-frame lines with a single "\n". Try the paragraph split first, then
    # fall back to a single-newline split, so both shapes map cleanly onto
    # their locations instead of collapsing to empty segments (which would
    # discard the text entirely and emit a blank chunk).
    aligned: list[tuple[str, SourceLocation]] = []
    for delimiter in ("\n\n", "\n"):
        text_segments = (content.text or "").split(delimiter)
        if len(text_segments) != len(locations):
            continue
        candidate: list[tuple[str, SourceLocation]] = []
        for seg_text, loc in zip(text_segments, locations, strict=True):
            if loc.timestamp_start_ms is not None or loc.timestamp_end_ms is not None:
                candidate.append((seg_text.strip(), loc))
        if candidate:
            return candidate

    for loc in locations:
        if loc.timestamp_start_ms is None and loc.timestamp_end_ms is None:
            continue
        aligned.append(("", loc))
    return aligned


def _build_chunk(
    group: list[tuple[str, SourceLocation]],
    *,
    index: int,
    source_type: str,
) -> RawChunk:
    text = "\n".join(seg for seg, _ in group if seg).strip()
    starts = [loc.timestamp_start_ms for _, loc in group if loc.timestamp_start_ms is not None]
    ends = [loc.timestamp_end_ms for _, loc in group if loc.timestamp_end_ms is not None]
    return RawChunk(
        content=text,
        chunk_index=index,
        metadata={
            "source_type": source_type,
            "timestamp_start_ms": min(starts) if starts else None,
            "timestamp_end_ms": max(ends) if ends else None,
            "segment_count": len(group),
        },
    )


__all__ = ["TimestampAwareChunker"]
