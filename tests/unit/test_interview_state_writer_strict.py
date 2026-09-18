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


class _SyncTarget:
    """Minimal shared-state stand-in: remembers synced fields.

    ``__getattr__`` answers every name so ``_sync_from_winner``'s
    ``hasattr`` gate passes for arbitrary winner keys — a real
    ``InterviewRuntimeStateData`` carries those fields; this fake is generic.
    """

    merged: dict[str, Any]

    def __init__(self) -> None:
        object.__setattr__(self, "merged", {})

    def __getattr__(self, name: str) -> Any:
        # only called for MISSING attributes -> report None so hasattr passes
        return None

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "merged":
            object.__setattr__(self, name, value)
        else:
            self.merged[name] = value


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
        # Optional FIFO of (version, data) returned by successive load_or_init
        # calls; when empty, falls back to (self.version, None).
        self.load_results: list[_Loaded] = []

    async def load_or_init(self, db: Any, session_id: Any) -> _Loaded:
        if self.load_results:
            return self.load_results.pop(0)
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

    async def test_strict_stale_sync_uses_reloaded_winner_not_stale_snapshot(
        self, fake_repo: _FakeRepo, committed: list[bool]
    ) -> None:
        """The V0-snapshot bug: after a CAS loss, the winner must be RELOADED.

        Interleaving: the writer loads V0; a reconcile worker persists V1; the
        writer's CAS fails; the strict path re-syncs the shared state. If it
        syncs from the PRE-CAS ``loaded`` (V0) instead of re-reading, the
        foreign writer's fields are clobbered with stale ones on the retry.
        """
        fake_repo.load_results = [
            _Loaded(0, {"coverage": "v0", "foreign_field": "stale"}),  # pre-save load
            _Loaded(1, {"coverage": "v1", "foreign_field": "winner"}),  # post-CAS reload
        ]
        fake_repo.fail_next_save = True
        state = _SyncTarget()
        w = nb.StateWriter(uuid4(), state)
        w.adopt_version(0)

        with pytest.raises(state_repo_module.StaleStateError):
            await w.save(strict=True, turn_key="tk-stale")

        assert state.merged.get("foreign_field") == "winner", (
            "the strict stale path must re-sync from the RELOADED winner, "
            "not the pre-CAS V0 snapshot"
        )

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
