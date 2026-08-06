"""In-process cache for synthesized narration audio.

Narration latency is dominated by the TTS round trip: measured against the
deployment's Deepgram Aura-2 voices, the onboarding transition line
("Great—the introduction is complete. Let's begin. Here is your first
question.") costs ~3.0-3.6s *every single time*, for every session. The browser
holds its "preparing" indicator for that whole window before text and voice
start together, which is the reported delay at the head of that line.

That cost is pure waste for one specific class of utterance: the ceremony lines
are FIXED strings (see ``services/ceremony.py`` — ``ready_transition_text`` and
friends are the same bytes for every candidate in a given language). Same text,
same voice, same provider means byte-identical audio, so it only ever needs to
be synthesized once per process.

Scope, deliberately narrow:

* The key is the exact ``(text, voice, persona, language)`` tuple, so two
  configs with different voices never share audio, and a change to any of them
  is a different entry rather than a stale hit.
* The cache sits BEHIND the router's output guard. Only text that already
  passed the "is this an approved interview utterance" boundary can reach it,
  so caching cannot widen what the endpoint is willing to speak.
* Bounded and in-process. No new infrastructure, no cross-process coherence to
  reason about, and a bounded dict cannot grow without limit on a long-lived
  worker. A restart simply re-warms.
* Question text is cached too — a question re-read (replay, or the same bank
  question served to the next candidate) hits just as well.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from threading import Lock

# Ceremony lines are short; questions are capped at 1200 chars upstream. A few
# hundred entries covers every ceremony line in both languages plus a healthy
# working set of question audio, at a worst case of a few tens of MB.
MAX_ENTRIES = 256

# Do not cache anything unreasonably large: a runaway entry would evict many
# useful small ones for little benefit. ~1.5 MB is far above a normal
# question-length MP3 at Aura-2 bitrates.
MAX_ENTRY_BYTES = 1_500_000

_cache: OrderedDict[str, bytes] = OrderedDict()
_lock = Lock()


def _key(*, text: str, voice: str | None, persona: str | None, language: str | None) -> str:
    """Hash the full identity of a synthesis request.

    Every input that can change the produced audio is part of the key. Hashing
    (rather than storing the raw text) keeps keys uniformly small and avoids
    holding a second copy of every utterance in memory.
    """
    material = "\x00".join(
        [
            text,
            voice or "",
            persona or "",
            (language or "").lower(),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def get(*, text: str, voice: str | None, persona: str | None, language: str | None) -> bytes | None:
    """Return cached audio for this exact request, or ``None``."""
    key = _key(text=text, voice=voice, persona=persona, language=language)
    with _lock:
        audio = _cache.get(key)
        if audio is not None:
            # LRU: a hit is the strongest signal an entry is worth keeping.
            _cache.move_to_end(key)
        return audio


def put(
    *,
    text: str,
    voice: str | None,
    persona: str | None,
    language: str | None,
    audio: bytes,
) -> None:
    """Store synthesized audio, evicting the least recently used entry if full."""
    if not audio or len(audio) > MAX_ENTRY_BYTES:
        return
    key = _key(text=text, voice=voice, persona=persona, language=language)
    with _lock:
        _cache[key] = audio
        _cache.move_to_end(key)
        while len(_cache) > MAX_ENTRIES:
            _cache.popitem(last=False)


def clear() -> None:
    """Drop every entry. Test-support only."""
    with _lock:
        _cache.clear()


def size() -> int:
    """Current entry count. Test-support only."""
    with _lock:
        return len(_cache)


__all__ = ["MAX_ENTRIES", "MAX_ENTRY_BYTES", "clear", "get", "put", "size"]
