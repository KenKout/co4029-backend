"""Audio codec helpers for the voice harness.

Decodes gateway TTS MP3 into 48 kHz mono PCM (the sample rate LiveKit's
``rtc.AudioSource`` expects), chops it into 10 ms frames for publishing, and
writes captured agent PCM to WAV. Uses ``av`` (PyAV) for decode/resample and
stdlib ``wave`` for output — no ``soundfile`` dependency.
"""

from __future__ import annotations

import io
import wave

import av
import numpy as np

# LiveKit AudioSource works natively at 48 kHz; publishing at this rate avoids
# an extra resample hop inside the SDK.
SAMPLE_RATE = 48_000
NUM_CHANNELS = 1
# 10 ms frames — the granularity rtc.AudioSource.capture_frame paces at.
FRAME_MS = 10
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000  # 480


def decode_to_pcm48k_mono(audio_bytes: bytes) -> np.ndarray:
    """Decode arbitrary compressed audio (MP3/WAV/…) to int16 48 kHz mono PCM.

    Returns a 1-D ``np.int16`` array of interleaved mono samples.
    """
    container = av.open(io.BytesIO(audio_bytes))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    for frame in container.decode(audio=0):
        for resampled in resampler.resample(frame):
            # (channels, samples) -> flat mono int16
            arr = resampled.to_ndarray()
            chunks.append(arr.reshape(-1).astype(np.int16))
    container.close()
    if not chunks:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(chunks)


def iter_frames(pcm: np.ndarray) -> list[np.ndarray]:
    """Split a PCM buffer into fixed-size 10 ms frames (zero-padded tail)."""
    frames: list[np.ndarray] = []
    for start in range(0, len(pcm), SAMPLES_PER_FRAME):
        chunk = pcm[start : start + SAMPLES_PER_FRAME]
        if len(chunk) < SAMPLES_PER_FRAME:
            chunk = np.pad(chunk, (0, SAMPLES_PER_FRAME - len(chunk)))
        frames.append(chunk.astype(np.int16))
    return frames


def write_wav(path: str, pcm: np.ndarray, *, sample_rate: int = SAMPLE_RATE) -> None:
    """Write int16 mono PCM to a WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(NUM_CHANNELS)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.astype(np.int16).tobytes())


def pcm_duration_seconds(pcm: np.ndarray, *, sample_rate: int = SAMPLE_RATE) -> float:
    return len(pcm) / float(sample_rate) if sample_rate else 0.0
