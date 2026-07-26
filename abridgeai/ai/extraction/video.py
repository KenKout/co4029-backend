"""Video extractor — ffmpeg pipeline that fans out to audio + image extractors.

Pipeline:

1. Extract audio track to a temp WAV (16 kHz mono PCM) — feed to AudioExtractor.
2. Extract frames at SCENE CHANGES (ffmpeg ``select=gt(scene,t)``) — feed each
   to ImageExtractor. Uniform low-rate sampling is only the fallback.
3. Merge results into a single chronological transcript: audio segments
   interleaved with frame-OCR markers tagged ``[Frame OCR @ Nms]``.

Why scene changes and not the old uniform ``fps=1`` sampling: with
``IMAGE_OCR_PROVIDER=llm_vision`` every sampled frame is one vision-model
call, so 1 fps turned a 60-minute lecture into ~3,600 sequential API calls —
far past the worker's ``job_timeout`` and almost all of it spent re-OCR-ing a
slide that had not changed since the previous second. A lecture's visual
content changes when the slide changes; detecting those transitions yields
tens of frames instead of thousands, each one carrying its REAL timestamp
(parsed from ffmpeg ``showinfo``) rather than an index-derived guess.

Frame extraction is best-effort: an audio-only container or a broken video
stream degrades to a transcript-only ingest instead of failing the job.

Temp directory is created via ``tempfile.TemporaryDirectory`` so all
intermediates are cleaned up automatically on exit, even on error paths.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO

from abridgeai.ai.extraction.audio import AudioExtractor
from abridgeai.ai.extraction.base import ExtractedContent, SourceLocation
from abridgeai.ai.extraction.image import ImageExtractor
from abridgeai.ai.extraction.registry import register_extractor
from abridgeai.core.config import Settings, get_settings
from abridgeai.core.exceptions import AppError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.ai.llm.gateway import LLMGateway


@dataclass(frozen=True)
class _FfmpegOutputs:
    # ``None`` when the container has no usable audio stream (a silent screen
    # recording) — the ingest continues frames-only instead of failing.
    audio_path: str | None
    # ``(timestamp_seconds, path)`` pairs. Timestamps come from ffmpeg's
    # ``showinfo`` pts, not from frame index arithmetic — scene-detected
    # frames are irregularly spaced, so index/fps would attribute a slide
    # shown at minute 40 to second 12.
    frames: list[tuple[float, str]]


# Frames whose ``showinfo`` line could not be paired keep this sentinel and
# are dropped rather than mis-attributed.
_NO_TIMESTAMP = -1.0

_SHOWINFO_PTS_RE = re.compile(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)")

# Fallback cadence when scene detection finds nothing (a truly static frame —
# camera on a lectern, one slide for the whole hour). One frame per minute
# bounds a 2-hour recording at 120 vision calls instead of 7,200.
_FALLBACK_FRAME_INTERVAL_S = 60.0


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
    scene_threshold: float,
    max_frames: int,
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
    audio_ok = rc == 0 and os.path.exists(audio_path)  # noqa: ASYNC240 -- one stat() call; not worth an anyio dependency
    if not audio_ok:
        # A silent screen recording has no audio stream; frames may still
        # carry the whole lecture. Whether the ingest can proceed at all is
        # decided by the caller once frame extraction has also run.
        logger.warning(
            "ffmpeg audio extraction failed (rc=%d); attempting frames-only: %s",
            rc,
            stderr.decode("utf-8", "replace")[:300],
        )

    # Frame extraction is best-effort from here down: the transcript is the
    # bulk of a lecture's value, so a container with no/broken video stream
    # must degrade to audio-only, not fail the whole ingest.
    try:
        frames = await _scene_change_frames(
            ffmpeg_path=ffmpeg_path,
            input_path=input_path,
            workdir=workdir,
            threshold=scene_threshold,
            max_frames=max_frames,
        )
        if len(frames) < 2:
            # Static visuals: nothing crossed the threshold (or only the
            # first frame did). Sample sparsely so a slide that never
            # changes is still captured once.
            frames = await _uniform_frames(
                ffmpeg_path=ffmpeg_path,
                input_path=input_path,
                workdir=workdir,
                interval_s=_FALLBACK_FRAME_INTERVAL_S,
                max_frames=max_frames,
            )
    except Exception:  # noqa: BLE001 -- degrade to transcript-only
        logger.warning("video frame extraction failed; continuing audio-only", exc_info=True)
        frames = []

    if not audio_ok and not frames:
        raise AppError(
            f"video has neither a usable audio stream nor extractable frames "
            f"(audio rc={rc}): {stderr.decode('utf-8', 'replace')[:300]}"
        )

    return _FfmpegOutputs(audio_path=audio_path if audio_ok else None, frames=frames)


async def _scene_change_frames(
    *,
    ffmpeg_path: str,
    input_path: str,
    workdir: str,
    threshold: float,
    max_frames: int,
) -> list[tuple[float, str]]:
    """One frame per detected scene change, with its real pts timestamp.

    ``select`` keeps frame 0 plus every frame whose scene score exceeds the
    threshold; ``showinfo`` logs each kept frame's ``pts_time`` to stderr,
    which is parsed positionally against the numbered outputs. ``-frames:v``
    bounds the write for pathological inputs (a camera pan trips the detector
    constantly); the evenly-spaced subsample below then enforces
    ``max_frames`` for the vision-call budget.
    """
    return await _select_frames(
        ffmpeg_path=ffmpeg_path,
        input_path=input_path,
        workdir=workdir,
        select_expr=f"eq(n,0)+gt(scene,{threshold})",
        prefix="scene",
        max_frames=max_frames,
    )


async def _uniform_frames(
    *,
    ffmpeg_path: str,
    input_path: str,
    workdir: str,
    interval_s: float,
    max_frames: int,
) -> list[tuple[float, str]]:
    """Sparse time-based sampling — the static-visuals fallback.

    Uses ``select`` on inter-frame time rather than the ``fps`` filter: the
    latter emits NOTHING for an input shorter than one interval (verified on
    ffmpeg 8: ``fps=1/60`` on a 5-second clip fails with "Nothing was written
    into output file"), whereas ``isnan(prev_selected_t)`` always admits the
    first frame regardless of duration.
    """
    return await _select_frames(
        ffmpeg_path=ffmpeg_path,
        input_path=input_path,
        workdir=workdir,
        select_expr=f"isnan(prev_selected_t)+gte(t-prev_selected_t,{interval_s})",
        prefix="uni",
        max_frames=max_frames,
    )


async def _select_frames(
    *,
    ffmpeg_path: str,
    input_path: str,
    workdir: str,
    select_expr: str,
    prefix: str,
    max_frames: int,
) -> list[tuple[float, str]]:
    """Run one ``select`` filter and return ``(pts_seconds, path)`` pairs."""
    pattern = os.path.join(workdir, f"{prefix}_%05d.jpg")
    args = [
        ffmpeg_path,
        "-y",
        "-i",
        input_path,
        "-vf",
        f"select='{select_expr}',showinfo",
        "-fps_mode",
        "vfr",
        "-frames:v",
        str(max(max_frames * 4, 240)),
        "-q:v",
        "4",
        pattern,
    ]
    rc, stderr = await _run_ffmpeg(args)
    if rc != 0:
        raise AppError(
            f"ffmpeg frame selection failed (rc={rc}, expr={select_expr!r}): "
            f"{stderr.decode('utf-8', 'replace')[:500]}"
        )

    timestamps = [
        float(match.group(1))
        for match in _SHOWINFO_PTS_RE.finditer(stderr.decode("utf-8", "replace"))
    ]
    paths = sorted(
        os.path.join(workdir, name)
        for name in os.listdir(workdir)
        if name.startswith(f"{prefix}_") and name.endswith(".jpg")
    )
    frames = [
        (timestamps[i] if i < len(timestamps) else _NO_TIMESTAMP, path)
        for i, path in enumerate(paths)
    ]
    frames = [f for f in frames if f[0] >= 0]

    if len(frames) > max_frames:
        step = len(frames) / max_frames
        frames = [frames[int(i * step)] for i in range(max_frames)]
    return frames


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
                scene_threshold=self._settings.video_scene_threshold,
                max_frames=self._settings.video_max_frames,
            )
            if outputs.audio_path is not None:
                audio_result = await self._audio_extractor.extract(outputs.audio_path)
            else:
                audio_result = ExtractedContent(
                    text="", metadata={"no_audio_stream": True}, source_type="audio"
                )
            frame_results: list[tuple[float, ExtractedContent]] = []
            for timestamp_seconds, frame_path in outputs.frames:
                try:
                    frame_content = await self._image_extractor.extract(frame_path)
                except Exception:  # noqa: BLE001 -- one bad frame must not sink the ingest
                    logger.warning(
                        "frame OCR failed at t=%.1fs; skipping frame",
                        timestamp_seconds,
                        exc_info=True,
                    )
                    continue
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
