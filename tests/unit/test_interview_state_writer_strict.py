"""StateWriter strict save: CAS semantics for the typed path (plan §2).

The typed fold must not overwrite a concurrent writer's state, and a strict
caller must SEE the loss instead of silently continuing. The spoken path keeps
its best-effort behaviour (log + reload + continue); the typed path raises
:class:`StaleStateError` after re-reading, with the shared in-room state object
re-synced to the persisted winner so a retry replays the fold onto CURRENT
state rather than a stale copy.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from abridgeai.features.interviews.orchestrator import repository as state_repo_module
from abridgeai.features.interviews.realtime import native_bridge as nb


class _Loaded:
    def __init__(self, version: int, data: Any) -> None:
        self.version = version
        self.data = data


class _FakeRepo:
    """In-memory stand-in for state_repo (load_or_init / save).

    Exposes the real StaleStateError because the writer raises/catches it via
    the module reference that is monkeypatched to this fake.
    """

    StaleStateError = state_repo_module.StaleStateError

    def __init__(self) -> None:
        self.version = 1
        self.saved: list[int] = []
        self.row: dict[str, Any] = {}
        self.fail_next_save = False

    async def load_or_init(self, db: Any, session_id: Any) -> _Loaded:
        return _Loaded(self.version, None)

    async def save(
        self,
        db: Any,
        session_id: Any,
        data: Any,
        *,
        expected_version: int,
        turn_idempotency_key: str | None = None,
    ) -> int:
        if self.fail_next_save:
            self.fail_next_save = False
            self.version += 1  # a foreign writer committed first
            raise state_repo_module.StaleStateError(session_id, expected_version, self.version)
        if expected_version != self.version:
            raise state_repo_module.StaleStateError(session_id, expected_version, self.version)
        self.version = expected_version + 1
        self.saved.append(self.version)
        self.row["turn_key"] = turn_idempotency_key
        return self.version


class _RecordingLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []


@pytest.fixture
def fake_repo(monkeypatch: pytest.MonkeyPatch) -> _FakeRepo:
    repo = _FakeRepo()
    monkeypatch.setattr(nb, "state_repo", repo)
    return repo


@pytest.fixture
def committed(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    calls: list[bool] = []

    class _FakeDb:
        async def commit(self) -> None:
            calls.append(True)

    class _FakeMakerFactory:
        def __call__(self) -> Any:
            return self

        async def __aenter__(self) -> Any:
            return _FakeDb()

        async def __aexit__(self, *exc: Any) -> None:
            pass

    monkeypatch.setattr(nb, "get_sessionmaker", lambda: _FakeMakerFactory())
    return calls


def _writer(state: Any = None) -> nb.StateWriter:
    w = nb.StateWriter(uuid4(), state or object())
    w.adopt_version(1)
    return w


class TestBestEffortSpokenPath:
    async def test_stale_save_logs_and_continues_without_raising(
        self, fake_repo: _FakeRepo, committed: list[bool]
    ) -> None:
        fake_repo.fail_next_save = True
        w = _writer()

        await w.save()  # must not raise

        assert fake_repo.saved == [], "a lost best-effort save must not be recorded"
        # The writer drops its held version so the NEXT save re-syncs fresh.
        assert w._version is None


class TestStrictTypedPath:
    async def test_stale_save_raises_stale_state_error(
        self, fake_repo: _FakeRepo, committed: list[bool]
    ) -> None:
        fake_repo.fail_next_save = True
        w = _writer()

        with pytest.raises(state_repo_module.StaleStateError):
            await w.save(strict=True, turn_key="tk-1")

    async def test_strict_save_stamps_the_turn_key_on_success(
        self, fake_repo: _FakeRepo, committed: list[bool]
    ) -> None:
        w = _writer()

        await w.save(strict=True, turn_key="tk-2")

        assert fake_repo.row["turn_key"] == "tk-2"

    async def test_load_time_difference_is_resynced_then_saved_against_current(
        self, fake_repo: _FakeRepo, committed: list[bool]
    ) -> None:
        """A version move detected AT LOAD is re-synced, not blindly overwritten.

        The save targets the row's CURRENT version after copying the winner's
        data onto the shared state object — the honest replay surface for a
        retry. (The UPDATE-time race itself is covered by the CAS tests above.)
        """
        w = _writer()
        fake_repo.version = 5  # a foreign writer moved the row after setup

        await w.save(strict=True, turn_key="tk-3")

        assert fake_repo.saved == [6], (
            "the strict save must land on the winner's version + 1, not fail or "
            "silently overwrite"
        )
        assert fake_repo.row["turn_key"] == "tk-3"
