from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from abridgeai.core.pagination.cursor import (
    CursorPage,
    decode_cursor,
    encode_cursor,
)


class TestEncodeDecodeCursor:
    def test_round_trip_random_uuid(self) -> None:
        original = uuid4()
        assert decode_cursor(encode_cursor(original)) == original

    def test_round_trip_nil_uuid(self) -> None:
        nil = UUID("00000000-0000-0000-0000-000000000000")
        assert decode_cursor(encode_cursor(nil)) == nil

    def test_round_trip_max_uuid(self) -> None:
        maxv = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
        assert decode_cursor(encode_cursor(maxv)) == maxv

    @pytest.mark.parametrize(
        "uid",
        [
            UUID("01234567-89ab-cdef-0123-456789abcdef"),
            UUID("a1b2c3d4-e5f6-7890-1234-567890abcdef"),
            UUID("deadbeef-cafe-babe-f00d-feedfacecafe"),
        ],
    )
    def test_round_trip_parametrized(self, uid: UUID) -> None:
        assert decode_cursor(encode_cursor(uid)) == uid

    def test_encoded_cursor_is_url_safe_no_padding(self) -> None:
        encoded = encode_cursor(uuid4())
        assert "=" not in encoded
        assert "+" not in encoded
        assert "/" not in encoded

    def test_encoded_cursor_is_str(self) -> None:
        assert isinstance(encode_cursor(uuid4()), str)

    def test_decode_invalid_cursor_raises(self) -> None:
        with pytest.raises(ValueError):
            decode_cursor("not-a-valid-cursor!!!")


class TestCursorPage:
    def test_default_next_cursor_is_none(self) -> None:
        page: CursorPage[int] = CursorPage(items=[1, 2, 3])
        assert page.next_cursor is None
        assert page.items == [1, 2, 3]

    def test_with_next_cursor(self) -> None:
        cursor = encode_cursor(uuid4())
        page: CursorPage[str] = CursorPage(items=["a", "b"], next_cursor=cursor)
        assert page.next_cursor == cursor
        assert page.items == ["a", "b"]

    def test_empty_items(self) -> None:
        page: CursorPage[int] = CursorPage(items=[])
        assert page.items == []
        assert page.next_cursor is None

    def test_frozen_dataclass(self) -> None:
        page: CursorPage[int] = CursorPage(items=[1])
        with pytest.raises(AttributeError):
            page.next_cursor = "x"  # type: ignore[misc]
