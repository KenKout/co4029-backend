"""Timestamp-aware chunker tests."""

from __future__ import annotations

from abridgeai.ai.chunking import RawChunk, TimestampAwareChunker
from abridgeai.ai.extraction import ExtractedContent, SourceLocation


def _transcript(
    segments: list[tuple[str, int, int]], source_type: str = "audio"
) -> ExtractedContent:
    text = "\n\n".join(seg for seg, _, _ in segments)
    locations = [SourceLocation(timestamp_start_ms=s, timestamp_end_ms=e) for _, s, e in segments]
    return ExtractedContent(
        text=text,
        metadata={"duration_ms": locations[-1].timestamp_end_ms if locations else 0},
        source_type=source_type,
        source_locations=locations,
    )


def test_timestamp_chunker_silence_boundary() -> None:
    content = _transcript(
        [
            ("Hello and welcome to the lecture.", 0, 5_000),
            ("We continue our introduction.", 5_500, 14_000),
            ("After a long silence we move on.", 17_500, 25_000),
            ("Final segment of the recording.", 25_500, 30_000),
        ]
    )

    chunks = TimestampAwareChunker(silence_gap_ms=2_000).chunk(content)

    assert len(chunks) >= 2, "silence > 2000ms should force a new chunk"

    for chunk in chunks:
        start = chunk.metadata["timestamp_start_ms"]
        end = chunk.metadata["timestamp_end_ms"]
        assert end - start <= 30_000
        assert not (start <= 15_000 <= 17_000 <= end), (
            "no chunk should span across the 15s-17.5s silence"
        )


def test_timestamp_chunker_metadata() -> None:
    content = _transcript(
        [
            ("First utterance.", 0, 1_000),
            ("Second utterance.", 1_500, 3_000),
        ]
    )

    chunks = TimestampAwareChunker(silence_gap_ms=2_000).chunk(content)

    assert chunks
    for chunk in chunks:
        assert isinstance(chunk, RawChunk)
        assert "timestamp_start_ms" in chunk.metadata
        assert "timestamp_end_ms" in chunk.metadata
        assert chunk.metadata["timestamp_start_ms"] is not None
        assert chunk.metadata["timestamp_end_ms"] is not None
        assert chunk.metadata["timestamp_end_ms"] >= chunk.metadata["timestamp_start_ms"]


def test_timestamp_chunker_no_locations_falls_back() -> None:
    content = ExtractedContent(
        text="Some plain text without any timestamps.",
        metadata={},
        source_type="audio",
        source_locations=[],
    )
    chunks = TimestampAwareChunker().chunk(content)
    assert len(chunks) == 1
    assert chunks[0].metadata["timestamp_start_ms"] is None


def test_timestamp_chunker_max_chunk_ms() -> None:
    content = _transcript(
        [
            ("Continuous speech segment one.", 0, 5_000),
            ("Continuous speech segment two.", 5_100, 10_000),
            ("Continuous speech segment three.", 10_100, 15_000),
            ("Continuous speech segment four.", 15_100, 20_000),
        ]
    )

    chunks = TimestampAwareChunker(silence_gap_ms=10_000, max_chunk_ms=8_000).chunk(content)
    assert len(chunks) >= 2
    for c in chunks:
        duration = c.metadata["timestamp_end_ms"] - c.metadata["timestamp_start_ms"]
        assert duration <= 12_000
