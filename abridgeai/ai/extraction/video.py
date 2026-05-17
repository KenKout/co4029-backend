"""Video extractor — ffmpeg pipeline that fans out to audio + image extractors.

Pipeline:

1. Extract audio track to a temp WAV (16 kHz mono PCM) — feed to AudioExtractor.
2. Sample frames at ``settings.video_frame_sample_fps`` (default 1 fps) — feed
   each frame to ImageExtractor.
3. Merge results into a single chronological transcript: audio segments
   interleaved with frame-OCR markers tagged ``[Frame at t=Ns]``.

Temp directory is created via ``tempfile.TemporaryDirectory`` so all
intermediates are cleaned up automatically on exit, even on error paths.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO

from abridgeai.ai.extraction.audio import AudioExtractor
from abridgeai.ai.extraction.base import ExtractedContent, SourceLocation
from abridgeai.ai.extraction.image import ImageExtractor
from abridgeai.ai.extraction.registry import register_extractor
from abridgeai.core.config import Settings, get_settings
from abridgeai.core.exceptions import AppError

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.ai.llm.gateway import LLMGateway


@dataclass(frozen=True)
class _FfmpegOutputs:
    audio_path: str
    frame_paths: list[str]


def _write_input_blob(raw: bytes, dest: str) -> None:
    with open(dest, "wb") as fh:
        fh.write(raw)


def _read_source(source: BinaryIO | bytes | str) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, str):
        with open(source, "rb") as fh:
            return fh.read()
    return source.read()


async def _run_ffmpeg(args: list[str]) -> tuple[int, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _stdout, stderr = await proc.communicate()
    return proc.returncode or 0, stderr or b""


async def _split_audio_and_frames(
    *,
    ffmpeg_path: str,
    input_path: str,
    workdir: str,
    fps: float,
) -> _FfmpegOutputs:
    audio_path = os.path.join(workdir, "audio.wav")
    audio_args = [
        ffmpeg_path,
        "-y",
        "-i",
        input_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_path,
    ]
    rc, stderr = await _run_ffmpeg(audio_args)
    if rc != 0:
        raise AppError(
            f"ffmpeg audio extraction failed (rc={rc}): {stderr.decode('utf-8', 'replace')[:500]}"
        )

    frames_pattern = os.path.join(workdir, "frame_%04d.jpg")
    frames_args = [
        ffmpeg_path,
        "-y",
        "-i",
        input_path,
        "-vf",
        f"fps={fps}",
        "-q:v",
        "4",
        frames_pattern,
    ]
    rc, stderr = await _run_ffmpeg(frames_args)
    if rc != 0:
        raise AppError(
            f"ffmpeg frame sampling failed (rc={rc}): {stderr.decode('utf-8', 'replace')[:500]}"
        )

    frame_paths = sorted(
        os.path.join(workdir, name)
        for name in os.listdir(workdir)
        if name.startswith("frame_") and name.endswith(".jpg")
    )
    return _FfmpegOutputs(audio_path=audio_path, frame_paths=frame_paths)


@register_extractor("video/mp4")
@register_extractor("video/quicktime")
@register_extractor("video/x-matroska")
@register_extractor("video/webm")
@register_extractor("video/x-msvideo")
class VideoExtractor:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        audio_extractor: AudioExtractor | None = None,
        image_extractor: ImageExtractor | None = None,
        gateway: LLMGateway | None = None,
        db: AsyncSession | None = None,
        stage_name: str = "extraction",
    ) -> None:
        self._settings = settings or get_settings()
        self._audio_extractor = audio_extractor or AudioExtractor(
            settings=self._settings, db=db, stage_name=stage_name
        )
        self._image_extractor = image_extractor or ImageExtractor(
            settings=self._settings, gateway=gateway, db=db, stage_name=stage_name
        )

    async def extract(self, source: BinaryIO | bytes | str) -> ExtractedContent:
        raw = await asyncio.to_thread(_read_source, source)
        with tempfile.TemporaryDirectory(prefix="abridgeai-video-") as workdir:
            input_path = os.path.join(workdir, "input.bin")
            await asyncio.to_thread(_write_input_blob, raw, input_path)
            outputs = await _split_audio_and_frames(
                ffmpeg_path=self._settings.ffmpeg_path,
                input_path=input_path,
                workdir=workdir,
                fps=self._settings.video_frame_sample_fps,
            )
            audio_result = await self._audio_extractor.extract(outputs.audio_path)
            frame_results: list[tuple[float, ExtractedContent]] = []
            for index, frame_path in enumerate(outputs.frame_paths):
                timestamp_seconds = index / self._settings.video_frame_sample_fps
                frame_content = await self._image_extractor.extract(frame_path)
                frame_results.append((timestamp_seconds, frame_content))

        return _merge(audio_result, frame_results)


def _merge(
    audio: ExtractedContent,
    frames: list[tuple[float, ExtractedContent]],
) -> ExtractedContent:
    parts: list[tuple[int, str, SourceLocation]] = []
    for loc in audio.source_locations:
        start_ms = loc.timestamp_start_ms or 0
        snippet = ""
        if loc in audio.source_locations:
            index = audio.source_locations.index(loc)
            audio_lines = audio.text.splitlines()
            if 0 <= index < len(audio_lines):
                snippet = audio_lines[index]
        parts.append((start_ms, f"[Audio @ {start_ms}ms] {snippet}".rstrip(), loc))

    for ts_seconds, frame in frames:
        ts_ms = int(round(ts_seconds * 1000))
        ocr_text = frame.text.strip()
        if not ocr_text:
            continue
        parts.append(
            (
                ts_ms,
                f"[Frame OCR @ {ts_ms}ms] {ocr_text}",
                SourceLocation(timestamp_start_ms=ts_ms, timestamp_end_ms=ts_ms),
            )
        )

    parts.sort(key=lambda item: item[0])
    text = "\n".join(line for _, line, _ in parts).strip()
    locations = [loc for _, _, loc in parts]

    return ExtractedContent(
        text=text,
        metadata={
            "audio_segment_count": len(audio.source_locations),
            "frame_count": len(frames),
            "audio_metadata": audio.metadata,
        },
        source_type="video",
        source_locations=locations,
    )


__all__ = ["VideoExtractor"]
