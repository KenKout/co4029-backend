"""Generic in-process TTL cache for expensive read-only aggregates.

Same infrastructure philosophy as :mod:`abridgeai.features.interviews.
services.narration_cache` — a bounded dict behind a lock, no Redis, no new
deployment surface. The difference is expiry: these entries serve *freshness
tolerant* aggregates (admin dashboards, permission resolution), so each entry
carries a wall-clock TTL and stale values are simply recomputed on the next
read after expiry.

Deliberately process-local. With a single API worker this is exact; with
several workers each keeps its own copy and invalidation lags by at most one
TTL window, which every consumer here has already accepted.

Usage::

    from abridgeai.core.ttl_cache import TTLCache

    _PERMS = TTLCache(max_entries=1024, ttl_seconds=30)
    perms = _PERMS.get(key) or await compute_and_store(...)
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass(slots=True)
class _Entry:
    value: Any
    expires_at: float


@dataclass
class TTLCache:
    """Bounded LRU cache whose entries expire after ``ttl_seconds``.

    ``max_entries`` bounds memory; ``ttl_seconds`` bounds staleness. Both are
    hard requirements for anything cached on a long-lived worker: without the
    bound the dict grows forever, without the TTL a role/grant change would be
    invisible until restart.
    """

    max_entries: int
    ttl_seconds: float
    _store: OrderedDict[Hashable, _Entry] = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def get(self, key: Hashable) -> Any | None:  # noqa: ANN401 -- heterogeneous payloads
        """Return the live value for ``key``, or ``None`` if absent/expired.

        An expired entry is dropped on read (lazy eviction) — no background
        sweeper thread to run or reason about.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                del self._store[key]
                return None
            # LRU: a hit is the strongest signal an entry is worth keeping.
            self._store.move_to_end(key)
            return entry.value

    def put(self, key: Hashable, value: Any) -> None:  # noqa: ANN401 -- heterogeneous payloads
        """Store ``value`` under ``key``, evicting LRU entries past the bound."""
        now = time.monotonic()
        with self._lock:
            self._store[key] = _Entry(value=value, expires_at=now + self.ttl_seconds)
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    def invalidate(self, key: Hashable) -> None:
        """Drop one entry (e.g. after a write that makes it stale)."""
        with self._lock:
            self._store.pop(key, None)

    def invalidate_where(self, predicate: Callable[[Hashable], bool]) -> int:
        """Drop every entry whose key satisfies ``predicate(key)``.

        Used for composite-keyed caches where a write invalidates all entries
        sharing one key component (e.g. every ``(user_id, course_id)`` entry
        for that ``user_id``). Returns the number of entries dropped.
        """
        dropped = 0
        with self._lock:
            for k in [k for k in self._store if predicate(k)]:
                del self._store[k]
                dropped += 1
        return dropped

    def clear(self) -> None:
        """Drop every entry (write path that cannot name its keys; tests)."""
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        """Entry count including not-yet-noticed expired items. Test-support."""
        with self._lock:
            return len(self._store)


__all__ = ["TTLCache"]
