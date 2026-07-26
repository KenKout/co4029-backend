"""Live integration tests for multimodal extractors.

Gated by ``audio_live`` and ``video_live`` markers — skipped in default
CI runs. Enable with ``-m audio_live`` (requires real Whisper API
credentials or faster-whisper installed) or ``-m video_live`` (requires
the ``ffmpeg`` binary on PATH).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import struct
from pathlib import Path

import pytest

from abridgeai.core.exceptions import AppError


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _whisper_creds_present() -> bool:
    return bool(os.getenv("LLM_API_KEY")) and bool(os.getenv("WHISPER_MODEL"))


def _generate_silent_wav(path: Path, duration_seconds: float = 0.5) -> None:
    sample_rate = 16000
    n_samples = int(duration_seconds * sample_rate)
    with open(path, "wb") as fh:
        fh.write(b"RIFF")
        fh.write(struct.pack("<I", 36 + n_samples * 2))
        fh.write(b"WAVE")
        fh.write(b"fmt ")
        fh.write(struct.pack("<I", 16))
        fh.write(struct.pack("<H", 1))
        fh.write(struct.pack("<H", 1))
        fh.write(struct.pack("<I", sample_rate))
        fh.write(struct.pack("<I", sample_rate * 2))
        fh.write(struct.pack("<H", 2))
        fh.write(struct.pack("<H", 16))
        fh.write(b"data")
        fh.write(struct.pack("<I", n_samples * 2))
        fh.write(b"\x00\x00" * n_samples)


@pytest.mark.audio_live
@pytest.mark.skipif(
    not _whisper_creds_present(), reason="LLM_API_KEY / WHISPER_MODEL not configured"
)
@pytest.mark.asyncio
async def test_audio_live_whisper_api(tmp_path: Path) -> None:
    from abridgeai.ai.extraction.audio import AudioExtractor

    wav_path = tmp_path / "silent.wav"
    _generate_silent_wav(wav_path)

    extractor = AudioExtractor()
    with pytest.raises(AppError):
        await extractor.extract(str(wav_path))


@pytest.mark.video_live
@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg binary not on PATH")
@pytest.mark.asyncio
async def test_video_live_ffmpeg_pipeline(tmp_path: Path) -> None:
    from abridgeai.ai.extraction.video import _split_audio_and_frames

    video_path = tmp_path / "test.mp4"
    ffmpeg_bin = shutil.which("ffmpeg")
    assert ffmpeg_bin is not None
    proc = await asyncio.create_subprocess_exec(
        ffmpeg_bin,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=2:size=160x120:rate=1",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=2",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-c:a",
        "aac",
        "-shortest",
        str(video_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    assert proc.returncode == 0
    workdir = tmp_path / "work"
    workdir.mkdir()
    outputs = await _split_audio_and_frames(
        ffmpeg_path=ffmpeg_bin,
        input_path=str(video_path),
        workdir=str(workdir),
        scene_threshold=0.2,
        max_frames=120,
    )
    audio_exists = await asyncio.to_thread(os.path.exists, outputs.audio_path)
    assert audio_exists
    # Scene-change extraction on the static 5s fixture falls back to sparse
    # uniform sampling — at least the first frame is always captured.
    assert len(outputs.frames) >= 1
