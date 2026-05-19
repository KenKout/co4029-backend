"""Unit tests for :mod:`abridgeai.core.db.conflict_mapper`.

Validates the dispatch contract that every feature relies on:

* a registered constraint name in the ``IntegrityError.orig`` text yields
  :class:`ConflictError` with the registered message;
* an unregistered constraint name re-raises the original
  ``IntegrityError`` so genuine bugs stay loud;
* the session is rolled back before raising so the caller can keep using
  it for a fresh attempt;
* registration is idempotent and last-write-wins per constraint name.
"""

from __future__ import annotations

import pytest

from abridgeai.core.db.conflict_mapper import (
    flush_or_conflict,
    register_conflict_mappings,
)
from abridgeai.core.exceptions import ConflictError


class _StubOrig:
    def __init__(self, text: str) -> None:
        self._text = text

    def __str__(self) -> str:
        return self._text


class _StubIntegrityError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.orig = _StubOrig(message)


class _FakeSession:
    """Minimal AsyncSession stand-in honouring the helper's contract.

    Real ``IntegrityError`` requires a SQLAlchemy ``StatementError`` shape;
    the helper only needs ``isinstance(exc, IntegrityError)`` to be True
    and ``str(exc.orig)`` to match. We monkeypatch the helper's import to
    treat ``_StubIntegrityError`` as the IntegrityError class for the test
    instead of constructing a real one.
    """

    def __init__(self, raise_on_flush: Exception | None = None) -> None:
        self._raise = raise_on_flush
        self.flushed = 0
        self.rolled_back = 0

    async def flush(self) -> None:
        self.flushed += 1
        if self._raise is not None:
            raise self._raise

    async def rollback(self) -> None:
        self.rolled_back += 1


@pytest.fixture(autouse=True)
def _patch_integrity_error_class(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("abridgeai.core.db.conflict_mapper.IntegrityError", _StubIntegrityError)


async def test_known_constraint_maps_to_conflict_error() -> None:
    register_conflict_mappings({"my_test_unique_index": "test_taken: something"})
    err = _StubIntegrityError(
        'duplicate key value violates unique constraint "my_test_unique_index"'
    )
    session = _FakeSession(raise_on_flush=err)

    with pytest.raises(ConflictError) as captured:
        await flush_or_conflict(session)  # type: ignore[arg-type]

    assert "test_taken: something" in str(captured.value)
    assert session.rolled_back == 1


async def test_unknown_constraint_reraises_original_integrity_error() -> None:
    err = _StubIntegrityError(
        'duplicate key value violates unique constraint "totally_unregistered_constraint"'
    )
    session = _FakeSession(raise_on_flush=err)

    with pytest.raises(_StubIntegrityError):
        await flush_or_conflict(session)  # type: ignore[arg-type]

    assert session.rolled_back == 1


async def test_clean_flush_does_not_roll_back() -> None:
    session = _FakeSession(raise_on_flush=None)
    await flush_or_conflict(session)  # type: ignore[arg-type]
    assert session.flushed == 1
    assert session.rolled_back == 0


async def test_registration_is_last_write_wins() -> None:
    register_conflict_mappings({"shared_constraint": "first message"})
    register_conflict_mappings({"shared_constraint": "second message"})
    err = _StubIntegrityError('duplicate key value violates unique constraint "shared_constraint"')
    session = _FakeSession(raise_on_flush=err)

    with pytest.raises(ConflictError) as captured:
        await flush_or_conflict(session)  # type: ignore[arg-type]

    assert "second message" in str(captured.value)
