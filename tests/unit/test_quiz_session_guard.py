from __future__ import annotations

from uuid import uuid4

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from abridgeai.features.quizzes.services import session_guard


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    async def eval(self, script: str, numkeys: int, key: str, owner: str, ttl: int) -> int:
        assert numkeys == 1
        if "TAKEOVER" in script:
            self.values[key] = owner
            self.ttls[key] = int(ttl)
            return 1
        if "RELEASE" in script:
            if self.values.get(key) == owner:
                del self.values[key]
                self.ttls.pop(key, None)
                return 1
            return 0
        existing = self.values.get(key)
        if existing is None or existing == owner:
            self.values[key] = owner
            self.ttls[key] = int(ttl)
            return 1
        return 0


@pytest.mark.asyncio
async def test_same_session_claim_renews_and_other_session_conflicts(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = FakeRedis()
    monkeypatch.setattr(session_guard, "get_cache", lambda: redis)
    attempt_id = uuid4()
    owner = uuid4()
    other = uuid4()

    assert await session_guard.claim(attempt_id, owner) is True
    assert await session_guard.validate_and_renew(attempt_id, owner) is True
    assert await session_guard.claim(attempt_id, other) is False
    assert redis.ttls[session_guard.session_key(attempt_id)] == session_guard.SESSION_GUARD_TTL_SECONDS


@pytest.mark.asyncio
async def test_missing_key_is_claimed_by_first_heartbeat(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = FakeRedis()
    monkeypatch.setattr(session_guard, "get_cache", lambda: redis)
    attempt_id = uuid4()
    owner = uuid4()

    assert await session_guard.validate_and_renew(attempt_id, owner) is True
    assert redis.values[session_guard.session_key(attempt_id)] == str(owner)


@pytest.mark.asyncio
async def test_takeover_and_release_are_owner_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = FakeRedis()
    monkeypatch.setattr(session_guard, "get_cache", lambda: redis)
    attempt_id = uuid4()
    owner = uuid4()
    replacement = uuid4()

    assert await session_guard.claim(attempt_id, owner) is True
    assert await session_guard.takeover(attempt_id, replacement) is True
    assert await session_guard.release(attempt_id, owner) is False
    assert await session_guard.release(attempt_id, replacement) is True
    assert session_guard.session_key(attempt_id) not in redis.values


@pytest.mark.asyncio
async def test_redis_errors_become_guard_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenRedis:
        async def eval(self, *args: object) -> int:
            raise RedisConnectionError("down")

    monkeypatch.setattr(session_guard, "get_cache", lambda: BrokenRedis())

    with pytest.raises(session_guard.QuizSessionGuardUnavailable):
        await session_guard.claim(uuid4(), uuid4())
