"""Unit tests for the role-aware retrieval filter (FR-11)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from abridgeai.ai.retrieval.role_filter import split_by_role


@dataclass
class _Chunk:
    name: str
    metadata: dict[str, Any] | None = field(default=None)


def _make(role: str, count: int, prefix: str) -> list[_Chunk]:
    return [_Chunk(name=f"{prefix}{i}", metadata={"content_role": role}) for i in range(count)]


def test_caps_summary_at_quarter_when_body_abundant() -> None:
    """When body chunks fill the quota, summary/review/front_matter
    chunks are capped at floor(limit / 4) so cover slides + ToC + recap
    can't soak up the rerank pool."""
    body = _make("body", 10, "b")
    front = _make("front_matter", 5, "f")
    summary = _make("summary", 5, "s")
    out = split_by_role(body + front + summary, 12)

    assert len(out) == 12
    # 12 // 4 = 3 deprioritized; 12 - 3 = 9 body
    body_count = sum(1 for c in out if c.name.startswith("b"))
    deprio_count = sum(1 for c in out if not c.name.startswith("b"))
    assert body_count == 9
    assert deprio_count == 3
    # Body chunks come first; deprioritized after.
    assert all(c.name.startswith("b") for c in out[:9])


def test_backfills_with_summary_when_body_short() -> None:
    """If body alone can't fill the body quota, the gap is back-filled
    with extra deprioritized chunks so the caller still gets `limit`
    total when the pool is large enough."""
    body = _make("body", 2, "b")
    front = _make("front_matter", 10, "f")
    out = split_by_role(body + front, 12)

    assert len(out) == 12
    # 2 body + 10 front_matter (3 from cap + 7 backfill)
    assert sum(1 for c in out if c.name.startswith("b")) == 2
    assert sum(1 for c in out if c.name.startswith("f")) == 10


def test_returns_only_what_is_available_when_pool_small() -> None:
    """Pool smaller than limit just returns the pool — no padding."""
    pool = _make("front_matter", 4, "f")
    out = split_by_role(pool, 12)
    assert len(out) == 4


def test_treats_missing_metadata_as_body() -> None:
    """Items whose metadata is None or missing content_role are treated
    as body — the safe default since we can't classify them."""
    pool = [_Chunk(name=f"x{i}", metadata=None) for i in range(5)]
    out = split_by_role(pool, 12)
    assert len(out) == 5
    # All survive (default role = body, no cap applied to them).
    assert {c.name for c in out} == {"x0", "x1", "x2", "x3", "x4"}


def test_treats_unknown_role_as_body() -> None:
    """Roles outside the deprioritized set (body, code, ...) are not
    capped. An unknown role string is also treated as body."""
    pool = [
        _Chunk(name="a", metadata={"content_role": "code"}),
        _Chunk(name="b", metadata={"content_role": "weird"}),
        _Chunk(name="c", metadata={"content_role": "BODY"}),  # case-insensitive
    ]
    out = split_by_role(pool, 5)
    assert len(out) == 3


def test_review_role_is_deprioritized() -> None:
    """review chunks (legacy "Review questions" recap slides) belong to
    the deprioritized bucket alongside summary and front_matter."""
    body = _make("body", 1, "b")
    review = _make("review", 5, "r")
    out = split_by_role(body + review, 8)
    # 8 // 4 = 2 deprioritized cap; body=1 fills 1 of 6 body quota,
    # backfill takes 5 more review.
    assert len(out) == 6 if len(body) + len(review) < 8 else 8
    # Body comes first.
    assert out[0].name == "b0"


def test_empty_pool_returns_empty() -> None:
    assert split_by_role([], 12) == []


def test_zero_or_negative_limit_returns_empty() -> None:
    pool = _make("body", 5, "b")
    assert split_by_role(pool, 0) == []
    assert split_by_role(pool, -3) == []


def test_invalid_ratio_disables_cap() -> None:
    """A ratio < 1 is defensive — fall back to no cap rather than blow
    up. Caller bug, but we don't want to abort retrieval."""
    body = _make("body", 2, "b")
    front = _make("front_matter", 8, "f")
    out = split_by_role(body + front, 5, deprioritized_ratio=0)
    assert len(out) == 5  # No cap — first 5 in input order.


def test_semantic_role_outranks_the_rule_based_top_level() -> None:
    """The LLM's label wins over the rule classifier's.

    On a real lecture deck the closing "Summary" slide and the "Review
    questions" slide both come out of the rule classifier as ``body`` while the
    enrichment LLM labels them ``summary`` / ``review``. Reading only the top
    level left exactly the two slides this filter exists to cap sitting in the
    pool uncapped.
    """
    body = [
        _Chunk(name=f"b{i}", metadata={"content_role": "body"}) for i in range(8)
    ]
    recap = [
        _Chunk(
            name="recap",
            metadata={"content_role": "body", "semantic": {"content_role": "summary"}},
        ),
        _Chunk(
            name="review",
            metadata={"content_role": "body", "semantic": {"content_role": "review"}},
        ),
    ]
    out = split_by_role(recap + body, 4)

    # limit 4 -> cap of 1 deprioritized; the other recap slide is squeezed out
    # by body content instead of displacing it.
    assert sum(1 for c in out if c.name in {"recap", "review"}) == 1
    assert sum(1 for c in out if c.name.startswith("b")) == 3


def test_falls_back_to_top_level_when_semantic_is_absent_or_blank() -> None:
    deprioritized = [
        _Chunk(name="no-semantic", metadata={"content_role": "summary"}),
        _Chunk(
            name="blank-semantic",
            metadata={"content_role": "review", "semantic": {"content_role": "  "}},
        ),
        _Chunk(
            name="malformed-semantic",
            metadata={"content_role": "front_matter", "semantic": "not-a-dict"},
        ),
    ]
    # Body chunks present so the cap binds rather than the backfill path.
    body = [_Chunk(name=f"b{i}", metadata={"content_role": "body"}) for i in range(8)]
    out = split_by_role(deprioritized + body, 4)

    # All three still resolve to a deprioritized role via the top-level
    # fallback, so the cap of floor(4/4)=1 admits exactly one of them.
    assert sum(1 for c in out if not c.name.startswith("b")) == 1
