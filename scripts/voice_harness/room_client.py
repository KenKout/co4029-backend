"""LiveKit room client for the voice harness.

A synthetic *student* participant that:
  * joins the interview room with a minted participant token (the token's
    room-config dispatches the real interview agent, exactly as the browser
    does);
  * subscribes to the agent's published audio track and records every received
    frame to an in-memory PCM buffer (per agent utterance), so the agent's TTS
    can be written to WAV and inspected;
  * publishes student "answers" as an audio track by streaming pre-decoded
    48 kHz mono PCM frames through an ``rtc.AudioSource`` (the agent's Whisper
    STT transcribes them, driving the interview forward).

This exercises the ENTIRE voice path end-to-end — token mint → dispatch →
agent STT → orchestration bridge → adaptive brain → agent TTS — without a human
at a microphone. It does NOT verify subjective audio quality; it verifies the
pipeline runs and produces non-empty, correctly-timed audio in both directions.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import numpy as np
from livekit import rtc

from . import audio_io

logger = logging.getLogger("voice_harness.room")

# rtc.TrackKind.KIND_AUDIO
_KIND_AUDIO = 1


def _utcnow() -> datetime:
    """Wall-clock UTC. Used for every observed timeline event so harness-side
    timestamps are directly comparable with DB row created_at values."""
    return datetime.now(UTC)


class AgentAudioCapture:
    """Accumulates received agent audio frames into per-utterance PCM buffers.

    A new utterance boundary is declared after ``silence_gap_s`` of no frames,
    so consecutive agent ``session.say`` calls land in separate buffers.
    """

    def __init__(self, *, silence_gap_s: float = 0.8) -> None:
        self._silence_gap_s = silence_gap_s
        self._current: list[np.ndarray] = []
        self._utterances: list[np.ndarray] = []
        self._sample_rate: int | None = None
        self._last_frame_at: float | None = None
        self._lock = asyncio.Lock()
        # Wall-clock UTC of the very first agent frame ever received and the
        # last frame received. These bound the agent's audio output for the
        # whole session (observable, not agent-internal).
        self.first_frame_at: datetime | None = None
        self.last_frame_at: datetime | None = None
        # Set by the driver right before it publishes a student answer, so we
        # can capture the FIRST agent frame that arrives AFTER that moment
        # (i.e. the response to that specific turn).
        self._await_frame_after: datetime | None = None
        self.first_frame_after_prompt_at: datetime | None = None

    def arm_response_timer(self) -> None:
        """Mark 'a student turn just ended'; the next agent frame is its reply."""
        self._await_frame_after = _utcnow()
        self.first_frame_after_prompt_at = None

    async def add_frame(self, frame: rtc.AudioFrame) -> None:
        now = asyncio.get_event_loop().time()
        wall = _utcnow()
        async with self._lock:
            if self.first_frame_at is None:
                self.first_frame_at = wall
            self.last_frame_at = wall
            if (
                self._await_frame_after is not None
                and self.first_frame_after_prompt_at is None
                and wall >= self._await_frame_after
            ):
                self.first_frame_after_prompt_at = wall
            if (
                self._last_frame_at is not None
                and now - self._last_frame_at > self._silence_gap_s
                and self._current
            ):
                self._flush_current()
            self._sample_rate = frame.sample_rate
            pcm = np.frombuffer(bytes(frame.data), dtype=np.int16)
            self._current.append(pcm)
            self._last_frame_at = now

    def _flush_current(self) -> None:
        if self._current:
            self._utterances.append(np.concatenate(self._current))
            self._current = []

    async def finalize(self) -> list[np.ndarray]:
        async with self._lock:
            self._flush_current()
            return list(self._utterances)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate or audio_io.SAMPLE_RATE


class HarnessRoom:
    """Wraps an ``rtc.Room`` for the synthetic student participant."""

    def __init__(self, *, url: str, token: str) -> None:
        self._url = url
        self._token = token
        self._room = rtc.Room()
        self._source = rtc.AudioSource(audio_io.SAMPLE_RATE, audio_io.NUM_CHANNELS)
        self._capture = AgentAudioCapture()
        self._agent_present = asyncio.Event()
        self._agent_gone = asyncio.Event()
        self._stream_tasks: list[asyncio.Task[None]] = []
        # Observed wall-clock UTC lifecycle timestamps + per-turn student-audio
        # spans. All client-side observable (never agent-internal).
        self.room_joined_at: datetime | None = None
        self.agent_joined_at: datetime | None = None
        self.disconnected_at: datetime | None = None
        # Each entry: {"started_at": dt, "ended_at": dt} for one student answer.
        self.student_turns: list[dict[str, datetime]] = []

    @property
    def capture(self) -> AgentAudioCapture:
        return self._capture

    @property
    def agent_present(self) -> asyncio.Event:
        return self._agent_present

    @property
    def agent_gone(self) -> asyncio.Event:
        return self._agent_gone

    async def connect(self) -> None:
        self._wire_events()
        await self._room.connect(self._url, self._token)
        self.room_joined_at = _utcnow()
        logger.info("connected to room=%s", self._room.name)

        # Publish our (initially silent) student mic track. The publish source
        # MUST be SOURCE_MICROPHONE (2): the agent's RoomIO only routes
        # microphone-sourced tracks into its STT pipeline. Publishing with the
        # default SOURCE_UNKNOWN (0) means the agent subscribes but never feeds
        # our audio to STT, so "input speech hasn't started yet" repeats forever
        # and no turn ever completes.
        track = rtc.LocalAudioTrack.create_audio_track("student-mic", self._source)
        opts = rtc.TrackPublishOptions()
        opts.source = rtc.TrackSource.SOURCE_MICROPHONE
        await self._room.local_participant.publish_track(track, opts)
        logger.info("published student mic track (source=microphone)")

    def _wire_events(self) -> None:
        @self._room.on("track_subscribed")
        def _on_sub(
            track: rtc.Track,
            _pub: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            if track.kind == _KIND_AUDIO:
                logger.info("subscribed to agent audio (participant=%s)", participant.identity)
                if self.agent_joined_at is None:
                    self.agent_joined_at = _utcnow()
                self._agent_present.set()
                task = asyncio.create_task(self._drain_track(track))
                self._stream_tasks.append(task)

        @self._room.on("participant_disconnected")
        def _on_left(participant: rtc.RemoteParticipant) -> None:
            logger.info("participant left: %s", participant.identity)
            self._agent_gone.set()

    async def _drain_track(self, track: rtc.Track) -> None:
        stream = rtc.AudioStream.from_track(track=track)
        try:
            async for event in stream:
                await self._capture.add_frame(event.frame)
        except Exception:  # noqa: BLE001 — best-effort capture
            logger.exception("audio stream drain error")
        finally:
            await stream.aclose()

    async def wait_for_agent(self, timeout: float = 30.0) -> bool:
        try:
            await asyncio.wait_for(self._agent_present.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    async def speak_pcm(self, pcm: np.ndarray, *, trailing_silence_s: float = 1.5) -> None:
        """Publish a PCM buffer as real-time audio frames (paced at 10 ms).

        Appends ``trailing_silence_s`` of real silence frames AFTER the speech.
        The agent's silero VAD needs to *receive* trailing silence to endpoint
        the turn (declare the student finished speaking); merely stopping frame
        capture leaves a gap the VAD may never resolve, so the turn never
        completes and ``on_user_turn_completed`` never fires.
        """
        silence = np.zeros(
            int(audio_io.SAMPLE_RATE * trailing_silence_s), dtype=np.int16
        )
        full = np.concatenate([pcm.astype(np.int16), silence])
        started_at = _utcnow()
        for chunk in audio_io.iter_frames(full):
            audio_frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=audio_io.SAMPLE_RATE,
                num_channels=audio_io.NUM_CHANNELS,
                samples_per_channel=len(chunk),
            )
            await self._source.capture_frame(audio_frame)
        # Let the source drain so the SDK forwards the tail before we continue.
        await asyncio.sleep(0.2)
        # end-of-speech = after the real speech + trailing silence is published.
        ended_at = _utcnow()
        self.student_turns.append({"started_at": started_at, "ended_at": ended_at})
        # Arm the response timer so the next agent frame is attributed to this
        # turn (end-of-speech → first-agent-audio latency).
        self._capture.arm_response_timer()

    async def disconnect(self) -> None:
        for task in self._stream_tasks:
            task.cancel()
        await self._room.disconnect()
        self.disconnected_at = _utcnow()
        logger.info("disconnected from room")
