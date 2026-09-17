"""Live ingest progress, and its promise never to break an ingest.

The whole multi-stage ingest runs inside one database transaction that
commits at the very end, so under MVCC nothing it writes to
``processing_jobs`` is visible to the polling endpoint until it finishes.
These three helpers exist to sidestep that: they write the percentage to
Redis, outside the transaction, where the endpoint can read it immediately.

That makes progress a *transient UX signal sitting next to* the ingest
rather than part of it, and the rule that follows is the one these tests
are mostly about: a Redis failure must cost the teacher a progress bar and
nothing else. Every helper swallows its errors, and if one stopped doing
so, a Redis hiccup would abort a document ingest that was otherwise fine --
minutes of extraction, chunking and embedding thrown away for a cosmetic
write.

The reader has the mirror-image duty: whatever it finds under the key, it
must hand back either a dict or ``None``, because the caller falls back to
the authoritative database row on ``None`` and would otherwise have to
defend against a half-written blob itself.

No Redis, no database - the cache client is a stub.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError

from abridgeai.core.cache.client import RedisFallbackError
from abridgeai.features.materials.ingestion import progress as progress_module
from abridgeai.features.materials.ingestion.progress import (
    clear_progress,
    publish_progress,
    read_progress,
)


class _FakeCache:
    """Records calls; optionally fails the way a dead Redis would."""

    def __init__(self, raise_on: Exception | None = None, value: Any = None) -> None:
        self._raise_on = raise_on
        self._value = value
        self.sets: list[tuple[str, str, int | None]] = []
        self.gets: list[str] = []
        self.deletes: list[str] = []

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        if self._raise_on:
            raise self._raise_on
        self.sets.append((key, value, ex))

    async def get(self, key: str) -> Any:
        if self._raise_on:
            raise self._raise_on
        self.gets.append(key)
        return self._value

    async def delete(self, key: str) -> None:
        if self._raise_on:
            raise self._raise_on
        self.deletes.append(key)


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch) -> _FakeCache:
    client = _FakeCache()
    monkeypatch.setattr(progress_module, "get_cache", lambda: client)
    return client


def _use(monkeypatch: pytest.MonkeyPatch, client: _FakeCache) -> _FakeCache:
    monkeypatch.setattr(progress_module, "get_cache", lambda: client)
    return client


class TestPublishing:
    async def test_the_snapshot_carries_the_stage_and_a_ttl(
        self, cache: _FakeCache
    ) -> None:
        version_id = uuid4()
        await publish_progress(
            version_id,
            status="processing",
            percent=60,
            stage_label="Building knowledge graph",
            detail="42/85",
        )

        assert len(cache.sets) == 1
        key, raw, ttl = cache.sets[0]
        assert str(version_id) in key
        assert ttl == 3600, (
            "the TTL must outlive the worker's job_timeout so a live run's key "
            "never expires under it, while a crashed run's key still self-cleans"
        )
        assert json.loads(raw) == {
            "status": "processing",
            "percent": 60,
            "stage_label": "Building knowledge graph",
            "detail": "42/85",
        }

    async def test_the_key_is_scoped_to_one_version(self, cache: _FakeCache) -> None:
        """Two versions of the same material can be ingesting at once."""
        first, second = uuid4(), uuid4()
        await publish_progress(first, status="processing", percent=10)
        await publish_progress(second, status="processing", percent=10)
        assert cache.sets[0][0] != cache.sets[1][0]

    @pytest.mark.parametrize(
        ("given", "stored"),
        [(-5, 0), (0, 0), (50, 50), (100, 100), (120, 100)],
    )
    async def test_the_percentage_is_clamped_to_a_sane_range(
        self, cache: _FakeCache, given: int, stored: int
    ) -> None:
        """A stage that miscounts its own sub-steps must not render a bar at
        140% or at minus something; the clamp is the last line of defence
        before the number reaches the screen.
        """
        await publish_progress(uuid4(), status="processing", percent=given)
        assert json.loads(cache.sets[0][1])["percent"] == stored

    async def test_the_optional_fields_are_present_as_null(
        self, cache: _FakeCache
    ) -> None:
        """The reader indexes the blob, so the keys are always written."""
        await publish_progress(uuid4(), status="queued", percent=0)
        payload = json.loads(cache.sets[0][1])
        assert payload["stage_label"] is None
        assert payload["detail"] is None

    @pytest.mark.parametrize(
        "failure",
        [
            RedisError("down"),
            RedisConnectionError("refused"),
            RedisFallbackError("no cache configured"),
            OSError("socket gone"),
        ],
    )
    async def test_a_dead_cache_does_not_reach_the_ingest(
        self, monkeypatch: pytest.MonkeyPatch, failure: Exception
    ) -> None:
        """This is the guarantee the module exists to make.

        The call site is a stage transition inside the ingest pipeline. If
        this raised, a Redis outage would abort a document ingest that had
        already done all of its real work.
        """
        _use(monkeypatch, _FakeCache(raise_on=failure))
        assert await publish_progress(uuid4(), status="processing", percent=30) is None


class TestReading:
    async def test_a_written_snapshot_comes_back_as_a_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = {"status": "processing", "percent": 60, "stage_label": None, "detail": None}
        _use(monkeypatch, _FakeCache(value=json.dumps(payload)))
        assert await read_progress(uuid4()) == payload

    async def test_bytes_from_redis_are_decoded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A client configured without ``decode_responses`` returns bytes."""
        _use(monkeypatch, _FakeCache(value=b'{"status": "processing", "percent": 10}'))
        assert await read_progress(uuid4()) == {"status": "processing", "percent": 10}

    async def test_no_key_means_no_live_progress(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Absent is the normal state: the run has finished or never started,
        and the caller falls back to the database row."""
        _use(monkeypatch, _FakeCache(value=None))
        assert await read_progress(uuid4()) is None

    @pytest.mark.parametrize(
        "failure",
        [RedisError("down"), RedisFallbackError("no cache"), OSError("socket gone")],
    )
    async def test_a_dead_cache_reads_as_absent(
        self, monkeypatch: pytest.MonkeyPatch, failure: Exception
    ) -> None:
        """Indistinguishable from "no key", and deliberately so -- both mean
        the endpoint should use the database row."""
        _use(monkeypatch, _FakeCache(raise_on=failure))
        assert await read_progress(uuid4()) is None

    @pytest.mark.parametrize("raw", ["not json at all", "", "{oops"])
    async def test_an_unparseable_blob_reads_as_absent(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        """A truncated write must not raise into the polling endpoint."""
        _use(monkeypatch, _FakeCache(value=raw))
        assert await read_progress(uuid4()) is None

    @pytest.mark.parametrize("raw", ["[1, 2, 3]", '"a string"', "42", "null"])
    async def test_valid_json_of_the_wrong_shape_reads_as_absent(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        """The caller indexes the result, so a list or a bare scalar would
        fail at the caller rather than here."""
        _use(monkeypatch, _FakeCache(value=raw))
        assert await read_progress(uuid4()) is None


class TestClearing:
    async def test_the_key_is_deleted_on_a_clean_finish(
        self, cache: _FakeCache
    ) -> None:
        """Left behind, the stale blob would outrank the database row for an
        hour -- the endpoint prefers live progress, so a finished ingest
        would keep reporting 95%."""
        version_id = uuid4()
        await clear_progress(version_id)
        assert cache.deletes == [f"material:ingest:progress:{version_id}"]

    @pytest.mark.parametrize(
        "failure",
        [RedisError("down"), RedisFallbackError("no cache"), OSError("socket gone")],
    )
    async def test_a_failed_delete_does_not_fail_the_ingest(
        self, monkeypatch: pytest.MonkeyPatch, failure: Exception
    ) -> None:
        """Clearing happens after the work is done and committed; raising
        here would report a completed ingest as failed. The key expires on
        its own regardless."""
        _use(monkeypatch, _FakeCache(raise_on=failure))
        assert await clear_progress(uuid4()) is None


class TestTheKeysAgree:
    async def test_all_three_helpers_address_the_same_key(
        self, cache: _FakeCache
    ) -> None:
        """Write, read and delete are three call sites for one key.

        If they disagreed the symptom would not be an error: progress would
        simply never appear, and the stale key would linger its full hour.
        """
        version_id = uuid4()
        await publish_progress(version_id, status="processing", percent=10)
        await read_progress(version_id)
        await clear_progress(version_id)

        assert {cache.sets[0][0], cache.gets[0], cache.deletes[0]} == {
            f"material:ingest:progress:{version_id}"
        }
