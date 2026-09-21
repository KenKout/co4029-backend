"""The published course tree, and its promise to survive a dead Redis.

`get_published_course_content_for_learner` rebuilds the learner curriculum
from half a dozen batch queries, and the result is identical for everyone on
the course — so it is read through one shared Redis key. What these tests
mostly guard is the boundary around that shortcut:

* a hit must render exactly what the source read would have rendered, because
  the endpoint's response model does not care where the DTO came from;
* every Redis failure — dead socket on the read, dead socket on the write,
  a payload left behind by an older DTO shape — must end in a normal response
  built from PostgreSQL, never in an error;
* ``None`` must never be cached: a course published a second ago would
  otherwise stay invisible for the rest of the TTL;
* the entry must expire before the presigned thumbnail and avatar URLs baked
  into it do, or a hit serves broken images.

No Redis, no database - the cache client is a stub and the source read is a
counter.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from abridgeai.core.cache import COURSE_CONTENT_PUBLISHED
from abridgeai.core.cache import json_store as json_store_module
from abridgeai.features.courses.schemas import CourseContentPublic
from abridgeai.features.courses.services import catalog as catalog_module


class _FakeCache:
    """Records calls; optionally fails the way a dead Redis would."""

    def __init__(self, raise_on: Exception | None = None) -> None:
        self._raise_on = raise_on
        self.store: dict[str, str] = {}
        self.sets: list[tuple[str, str, int | None]] = []
        self.gets: list[str] = []

    async def get(self, key: str) -> Any:
        if self._raise_on:
            raise self._raise_on
        self.gets.append(key)
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        if self._raise_on:
            raise self._raise_on
        self.sets.append((key, value, ex))
        self.store[key] = value


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch) -> _FakeCache:
    client = _FakeCache()
    monkeypatch.setattr(json_store_module, "get_cache", lambda: client)
    return client


COURSE_ID = uuid4()
MODULE_ID = uuid4()
LESSON_ID = uuid4()


def _tree() -> CourseContentPublic:
    """A minimal but realistic published tree (course + module + lesson item)."""
    return CourseContentPublic.model_validate(
        {
            "course": {
                "id": COURSE_ID,
                "slug": "intro-to-python",
                "title": "Intro to Python",
                "organization_id": uuid4(),
                "status": "published",
                "thumbnail_url": "https://s3.local/thumb.png?X-Amz-Signature=abc",
            },
            "modules": [
                {
                    "id": MODULE_ID,
                    "course_id": COURSE_ID,
                    "title": "Getting started",
                    "position": 1,
                    "items": [
                        {
                            "id": uuid4(),
                            "module_id": MODULE_ID,
                            "item_type": "lesson",
                            "position": 1,
                            "target": {
                                "id": LESSON_ID,
                                "title": "Hello, world",
                                "lesson_type": "reading",
                            },
                        }
                    ],
                }
            ],
        }
    )


class _SourceRead:
    """Stands in for the PostgreSQL rebuild; counts how often it ran."""

    def __init__(self, result: CourseContentPublic | None) -> None:
        self.result = result
        self.calls = 0

    async def __call__(self, db: Any, course_id: Any) -> CourseContentPublic | None:
        self.calls += 1
        return self.result


@pytest.fixture
def source(monkeypatch: pytest.MonkeyPatch) -> _SourceRead:
    read = _SourceRead(_tree())
    monkeypatch.setattr(catalog_module, "_build_published_course_content", read)
    return read


def _key() -> str:
    return COURSE_CONTENT_PUBLISHED.format(course_id=COURSE_ID)


class TestPopulatingTheCache:
    async def test_a_miss_reads_the_source_and_stores_the_tree(
        self, cache: _FakeCache, source: _SourceRead
    ) -> None:
        tree = await catalog_module.get_published_course_content_for_learner(None, COURSE_ID)

        assert tree == source.result
        assert source.calls == 1
        assert [call[0] for call in cache.sets] == [_key()]

    async def test_the_stored_payload_is_the_wire_shape_not_orm_rows(
        self, cache: _FakeCache, source: _SourceRead
    ) -> None:
        await catalog_module.get_published_course_content_for_learner(None, COURSE_ID)

        stored = json.loads(cache.store[_key()])
        assert stored["course"]["id"] == str(COURSE_ID)
        assert stored["modules"][0]["items"][0]["item_type"] == "lesson"

    async def test_a_missing_course_is_not_cached(
        self, cache: _FakeCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Caching the 404 would hide a course for the rest of the TTL the
        # moment someone published it.
        read = _SourceRead(None)
        monkeypatch.setattr(catalog_module, "_build_published_course_content", read)

        assert (
            await catalog_module.get_published_course_content_for_learner(None, COURSE_ID) is None
        )
        assert cache.sets == []


class TestServingFromTheCache:
    async def test_a_hit_skips_the_source_read(
        self, cache: _FakeCache, source: _SourceRead
    ) -> None:
        first = await catalog_module.get_published_course_content_for_learner(None, COURSE_ID)
        second = await catalog_module.get_published_course_content_for_learner(None, COURSE_ID)

        assert source.calls == 1
        assert second == first

    async def test_a_hit_renders_the_same_response_as_the_source(
        self, cache: _FakeCache, source: _SourceRead
    ) -> None:
        # Object equality is not the contract — what the endpoint serialises is.
        fresh = await catalog_module.get_published_course_content_for_learner(None, COURSE_ID)
        cached = await catalog_module.get_published_course_content_for_learner(None, COURSE_ID)

        assert fresh is not None
        assert cached is not None
        assert cached.model_dump(mode="json") == fresh.model_dump(mode="json")

    async def test_a_payload_from_an_older_dto_shape_is_treated_as_a_miss(
        self, cache: _FakeCache, source: _SourceRead
    ) -> None:
        cache.store[_key()] = json.dumps({"course": {"title": "shape from last release"}})

        tree = await catalog_module.get_published_course_content_for_learner(None, COURSE_ID)

        assert tree == source.result
        assert source.calls == 1

    async def test_an_unparseable_payload_is_treated_as_a_miss(
        self, cache: _FakeCache, source: _SourceRead
    ) -> None:
        cache.store[_key()] = "{not json"

        tree = await catalog_module.get_published_course_content_for_learner(None, COURSE_ID)

        assert tree == source.result
        assert source.calls == 1


class TestRedisOutage:
    async def test_a_failing_read_falls_back_to_postgresql(
        self, monkeypatch: pytest.MonkeyPatch, source: _SourceRead
    ) -> None:
        dead = _FakeCache(raise_on=RedisConnectionError("no route to host"))
        monkeypatch.setattr(json_store_module, "get_cache", lambda: dead)

        tree = await catalog_module.get_published_course_content_for_learner(None, COURSE_ID)

        assert tree == source.result
        assert source.calls == 1

    async def test_a_failing_write_still_returns_the_tree(
        self, monkeypatch: pytest.MonkeyPatch, source: _SourceRead
    ) -> None:
        dead = _FakeCache(raise_on=RedisConnectionError("no route to host"))
        monkeypatch.setattr(json_store_module, "get_cache", lambda: dead)

        assert (
            await catalog_module.get_published_course_content_for_learner(None, COURSE_ID)
            is not None
        )


class TestExpiry:
    async def test_the_entry_outlives_neither_the_thumbnail_nor_the_avatar_url(
        self, cache: _FakeCache, source: _SourceRead, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Presigned URLs are baked into the payload; a 120s URL life must pull
        # the cache TTL down with it rather than serve dead image links.
        monkeypatch.setattr(
            catalog_module,
            "get_settings",
            lambda: type("S", (), {"s3_url_ttl_seconds": 120})(),
        )

        await catalog_module.get_published_course_content_for_learner(None, COURSE_ID)

        assert cache.sets[0][2] == 60

    async def test_a_long_url_life_leaves_the_namespace_ttl_in_charge(
        self, cache: _FakeCache, source: _SourceRead, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            catalog_module,
            "get_settings",
            lambda: type("S", (), {"s3_url_ttl_seconds": 3600})(),
        )

        await catalog_module.get_published_course_content_for_learner(None, COURSE_ID)

        assert cache.sets[0][2] == COURSE_CONTENT_PUBLISHED.ttl_seconds
