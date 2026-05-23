from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any
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


def _b64encode(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _b64decode(cursor: str) -> bytes:
    padding = "=" * (-len(cursor) % 4)
    return base64.urlsafe_b64decode((cursor + padding).encode())


def encode_composite_cursor(sort_value: Any, last_id: UUID) -> str:
    """Encode a (sort_value, id) tuple cursor.

    ``sort_value`` may be a ``datetime``, ``str``, ``int``, or ``None``.
    Datetimes are serialised as ISO-8601 strings. The cursor is opaque to
    callers — it round-trips via :func:`decode_composite_cursor`.
    """
    serialisable: Any
    if isinstance(sort_value, datetime):
        serialisable = {"k": "dt", "v": sort_value.isoformat()}
    elif isinstance(sort_value, UUID):
        serialisable = {"k": "uuid", "v": str(sort_value)}
    elif isinstance(sort_value, str | int | float) or sort_value is None:
        serialisable = {"k": "raw", "v": sort_value}
    else:
        raise TypeError(f"Unsupported sort_value type: {type(sort_value).__name__}")
    payload = json.dumps([serialisable, str(last_id)], separators=(",", ":")).encode()
    return _b64encode(payload)


def decode_composite_cursor(cursor: str) -> tuple[Any, UUID]:
    """Decode a cursor produced by :func:`encode_composite_cursor`."""
    try:
        raw = _b64decode(cursor).decode()
        wrapper, last_id_str = json.loads(raw)
        kind = wrapper["k"]
        value = wrapper["v"]
    except (KeyError, ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid cursor") from exc
    sort_value: Any
    if kind == "dt":
        sort_value = datetime.fromisoformat(value)
    elif kind == "uuid":
        sort_value = UUID(value)
    elif kind == "raw":
        sort_value = value
    else:
        raise ValueError(f"Unknown cursor kind: {kind!r}")
    return sort_value, UUID(last_id_str)


__all__ = [
    "CursorPage",
    "decode_composite_cursor",
    "decode_cursor",
    "encode_composite_cursor",
    "encode_cursor",
]
