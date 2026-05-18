from __future__ import annotations

import base64
from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class CursorPage[T]:
    """Generic cursor-paginated page (Reconciliation §A10 / §D2).

    ``next_cursor`` is set when the page was filled to ``limit`` (more rows
    may exist); ``None`` otherwise. Cursors are opaque base64-encoded
    UUIDs of the last row's id — see :func:`encode_cursor` /
    :func:`decode_cursor`. T1.9's :class:`UserListPage` follows the same
    convention; future tasks may widen the cursor format.
    """

    items: list[T]
    next_cursor: str | None = None


def encode_cursor(last_id: UUID) -> str:
    return base64.urlsafe_b64encode(str(last_id).encode()).decode().rstrip("=")


def decode_cursor(cursor: str) -> UUID:
    padding = "=" * (-len(cursor) % 4)
    raw = base64.urlsafe_b64decode((cursor + padding).encode()).decode()
    return UUID(raw)


__all__ = ["CursorPage", "decode_cursor", "encode_cursor"]
