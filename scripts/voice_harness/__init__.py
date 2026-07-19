"""Semi-automated LiveKit voice smoke harness for the adaptive interviewer.

A scripted *student* participant that joins the real LiveKit room, publishes
pre-synthesized answer clips as its microphone, and captures the agent's TTS
audio to WAV files on disk. Lets us exercise the full voice path
(STT -> adaptive brain -> TTS) without a human at a microphone.

See ``python -m scripts.voice_harness --help`` (module ``run``) for usage, and
``docs/voice-adaptive-smoke-tests.md`` for what it does and does not cover.
"""
