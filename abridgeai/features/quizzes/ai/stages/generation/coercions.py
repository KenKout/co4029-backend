"""Type vocabulary + raw-LLM-JSON coercion helpers for the generation stage.

Split out of ``parsers.py`` (which grew past the 250-LOC god-file budget) so
each concern lives in its own module:

* This module owns the **type/format vocabulary** (the ``Literal`` aliases and
  the valid-value frozensets that mirror the DB CHECK constraints) plus the
  defensive **coercion helpers** that turn drifted LLM JSON into the canonical
  shapes the schemas expect. Every coercion fails safe — an unusable value maps
  to ``None`` or the safe default rather than raising — so a single bad field
  never aborts a whole generation batch.

``parsers.py`` re-exports the public names below, so existing imports of
``...generation.parsers`` keep working unchanged.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

QuizQuestionType = Literal[
    "multiple_choice",
    "true_false",
    "short_answer",
    "fill_blank",
    # Phase 7 expanded types. Each has its own answer-shape contract in the
    # system prompt and its own validator in ``shape_validators``.
    "numerical",
    "matching",
    "ordering",
]
RichFormat = Literal["plain", "markdown", "html"]
"""Phase 3 render discriminator. Mirrors the ``ck_quiz_questions_*_format``
CHECK constraints. Defaults to ``plain`` everywhere so AI output is treated
as escaped text unless a prompt explicitly opts into markdown."""
BloomLevel = Literal["remember", "understand", "apply", "analyze", "evaluate", "create"]
Difficulty = Literal["easy", "medium", "hard"]

# Legacy alias map. The pipeline used "mcq" historically; the DB CHECK
# always wanted "multiple_choice". Normalise at the parser boundary so
# every downstream consumer sees the DB vocabulary.
_LEGACY_TYPE_ALIASES: dict[str, str] = {
    "mcq": "multiple_choice",
    "fill_in_the_blank": "fill_blank",
    "true/false": "true_false",
    "tf": "true_false",
}

_VALID_TYPES = frozenset(
    {
        "multiple_choice",
        "true_false",
        "short_answer",
        "fill_blank",
        "numerical",
        "matching",
        "ordering",
    }
)


_VALID_FORMATS = frozenset({"plain", "markdown", "html"})


def _normalize_question_type(raw: Any) -> str:  # noqa: ANN401 -- raw LLM JSON
    """Map legacy or LLM-drifted aliases onto DB vocabulary."""
    if not isinstance(raw, str):
        return "multiple_choice"
    cleaned = raw.strip().lower()
    return _LEGACY_TYPE_ALIASES.get(cleaned, cleaned)


def _coerce_decimal(raw: Any) -> Decimal | None:  # noqa: ANN401 -- raw LLM JSON
    """Coerce a numeric answer/tolerance to Decimal, or None when unusable.

    Accepts int/float/str (models emit all three). Rejects bool explicitly —
    ``Decimal(True)`` would silently become 1.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, Decimal):
        return raw
    if isinstance(raw, (int, float)):
        return Decimal(str(raw))
    if isinstance(raw, str):
        token = raw.strip().replace(",", "")
        if not token:
            return None
        try:
            return Decimal(token)
        except ArithmeticError:
            return None
    return None


def _coerce_match_pairs(raw: Any) -> list[dict[str, str]] | None:  # noqa: ANN401 -- raw LLM JSON
    """Coerce a matching answer key into ``[{"left":..,"right":..}]``.

    Accepts the canonical list-of-objects, tolerating ``prompt``/``answer`` and
    ``term``/``definition`` key aliases (models drift), plus a plain
    ``{left: right}`` mapping. Entries missing either side are dropped;
    ``validate_matching`` then enforces count/uniqueness.
    """
    if isinstance(raw, dict):
        pairs = [
            {"left": str(key).strip(), "right": str(value).strip()}
            for key, value in raw.items()
            if str(key).strip() and str(value).strip()
        ]
        return pairs or None
    if not isinstance(raw, list):
        return None
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        left = item.get("left") or item.get("prompt") or item.get("term")
        right = item.get("right") or item.get("answer") or item.get("definition")
        if left is None or right is None:
            continue
        left_s, right_s = str(left).strip(), str(right).strip()
        if not left_s or not right_s:
            continue
        out.append({"left": left_s, "right": right_s})
    return out or None


def _coerce_match_distractors(raw: Any) -> list[str] | None:  # noqa: ANN401 -- raw LLM JSON
    """Coerce matching distractors into a list of non-empty strings.

    Distractors are extra right-side values with no left partner. Accepts a
    plain list of strings, tolerating a list of objects carrying
    ``right``/``value``/``text`` (models drift). Empty entries are dropped;
    ``validate_matching`` then enforces uniqueness against the answer values.
    """
    if not isinstance(raw, list) or not raw:
        return None
    out: list[str] = []
    for item in raw:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            value = item.get("right") or item.get("value") or item.get("text")
            text = str(value).strip() if value is not None else ""
        else:
            continue
        if text:
            out.append(text)
    return out or None


def _coerce_ordering_sequence(raw: Any) -> list[str] | None:  # noqa: ANN401 -- raw LLM JSON
    """Coerce an ordering answer key into a list of item strings in correct order.

    Accepts a plain list of strings, or a list of objects carrying
    ``item``/``text``/``value`` (optionally with a ``position`` to sort by,
    since some models emit unordered objects with explicit positions).
    """
    if not isinstance(raw, list) or not raw:
        return None
    if all(isinstance(item, str) for item in raw):
        cleaned = [item.strip() for item in raw if item.strip()]
        return cleaned or None
    entries: list[tuple[int, str]] = []
    for index, item in enumerate(raw, start=1):
        if isinstance(item, str):
            text = item.strip()
            position = index
        elif isinstance(item, dict):
            value = item.get("item") or item.get("text") or item.get("value")
            text = str(value).strip() if value is not None else ""
            try:
                position = int(item.get("position") or index)
            except (TypeError, ValueError):
                position = index
        else:
            continue
        if text:
            entries.append((position, text))
    if not entries:
        return None
    entries.sort(key=lambda pair: pair[0])
    return [text for _position, text in entries]


def _normalize_format(raw: Any) -> str:  # noqa: ANN401 -- raw LLM JSON
    """Coerce a rich-format discriminator, defaulting to ``plain``.

    Fails safe: an unknown/absent value becomes ``plain`` so the content is
    rendered as escaped text rather than trusted as HTML.
    """
    if not isinstance(raw, str):
        return "plain"
    cleaned = raw.strip().lower()
    return cleaned if cleaned in _VALID_FORMATS else "plain"


__all__ = [
    "BloomLevel",
    "Difficulty",
    "QuizQuestionType",
    "RichFormat",
    "_LEGACY_TYPE_ALIASES",
    "_VALID_FORMATS",
    "_VALID_TYPES",
    "_coerce_decimal",
    "_coerce_match_pairs",
    "_coerce_match_distractors",
    "_coerce_ordering_sequence",
    "_normalize_format",
    "_normalize_question_type",
]
