"""Map ``IntegrityError`` -> :class:`ConflictError` so writers return HTTP 409.

Every service that calls ``db.add(...)`` followed by ``await db.flush()`` on
an ORM model with a UNIQUE constraint must wrap the flush with
:func:`flush_or_conflict`. Without it, a duplicate (slug, position, attempt
number, ...) bubbles up as ``sqlalchemy.exc.IntegrityError`` and the request
ends with HTTP 500 even though the input was simply a fixable user mistake.

Each feature registers the constraint names it cares about with
:func:`register_conflict_mappings`. Unknown constraints re-raise the original
``IntegrityError`` so they remain visible bugs (loud-fail on real corruption,
quiet-fail on user input).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy.exc import IntegrityError

from abridgeai.core.exceptions import ConflictError

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_CONSTRAINT_MESSAGES: dict[str, str] = {}


def register_conflict_mappings(mappings: dict[str, str]) -> None:
    """Register constraint-name -> human-readable conflict message.

    Idempotent; later registrations overwrite earlier ones for the same
    constraint name. Both the SQLAlchemy declarative names
    (``uq_<table>_<cols>``) and the post-migration partial-index names
    (``<table>_<col>_<col>_key``) should be registered for the same logical
    conflict so the mapper hits regardless of which name PostgreSQL surfaces.
    """
    _CONSTRAINT_MESSAGES.update(mappings)


async def flush_or_conflict(db: AsyncSession) -> None:
    """Flush; map a known UNIQUE-constraint violation to :class:`ConflictError`.

    On match the session is rolled back so the caller can keep using it for
    a fresh attempt; on no match the original ``IntegrityError`` is re-raised
    untouched.
    """
    try:
        await db.flush()
    except IntegrityError as exc:
        await db.rollback()
        orig = str(exc.orig)
        for constraint, message in _CONSTRAINT_MESSAGES.items():
            if constraint in orig:
                raise ConflictError(message) from exc
        raise


__all__ = ["flush_or_conflict", "register_conflict_mappings"]
