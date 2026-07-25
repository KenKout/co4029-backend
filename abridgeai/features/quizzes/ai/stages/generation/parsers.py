"""Parsers and Pydantic schemas for the quiz GENERATION stage (T5.6).

Ports ``normalize_quiz_questions`` (mappers/quiz.py) and
``_question_for_review`` (pipelines/quiz_generation.py:1259-1277).

**Type vocabulary** matches the DB CHECK constraint exactly:
``multiple_choice``, ``true_false``, ``short_answer``, ``fill_blank``.
The legacy ``"mcq"`` alias is normalised to ``"multiple_choice"`` at the
parser boundary so old prompts/payloads still parse, but every consumer
downstream of this module sees the DB vocabulary.

**Type shape rules** (enforced by ``GeneratedQuestion._check_shape``):

* ``multiple_choice`` — exactly 4 options A-D, exactly 1 marked correct.
* ``true_false`` — exactly 2 options keyed ``T``/``F``, exactly 1
  marked correct. The parser auto-generates these options from the
  LLM's ``correct_answer`` ("True"/"False") when the LLM omits them
  (the prompt requests them but small models drift).
* ``short_answer`` — no options. ``correct_answer`` is a free-text
  string carried in ``original_generated_payload`` for grading.
* ``fill_blank`` — no options. ``correct_answer`` is a list of strings
  (one per blank, in order) carried in ``original_generated_payload``.
  The grader matches the student's drag-drop slots positionally.

Option-list shaping for ``multiple_choice`` and ``true_false`` lives in
the sibling ``option_normalizers`` module.
"""

from __future__ import annotations

import string
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from abridgeai.features.quizzes.ai.stages.generation.option_normalizers import (
    coerce_fill_blank_answer,
    normalize_options,
)
from abridgeai.features.quizzes.ai.stages.generation.shape_validators import (
    validate_fill_blank,
    validate_matching,
    validate_multiple_choice,
    validate_numerical,
    validate_ordering,
    validate_short_answer,
    validate_true_false,
)

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


class GeneratedQuestionOption(BaseModel):
    option_key: str
    option_text: str
    is_correct: bool
    position: int


class GeneratedQuestion(BaseModel):
    """Normalised LLM-generated question consumed by T5.7 / T5.8."""

    position: int = Field(ge=1)
    question_type: QuizQuestionType = "multiple_choice"
    prompt_text: str = Field(min_length=1)
    hint_text: str | None = None
    explanation: str = Field(min_length=1)
    difficulty: Difficulty = "medium"
    bloom_level: BloomLevel = "understand"
    # Phase 3 rich-content discriminators. Default ``plain`` so existing
    # prompts/behaviour are unchanged; the persistence stage sanitizes each
    # field according to its own format before writing.
    prompt_format: RichFormat = "plain"
    hint_format: RichFormat = "plain"
    explanation_format: RichFormat = "plain"
    expected_response_ms: int = Field(default=60000, ge=0)
    source_refs_json: list[str] = Field(default_factory=list)
    original_generated_payload: dict[str, Any] = Field(default_factory=dict)
    options: list[GeneratedQuestionOption] = Field(default_factory=list)

    # --- Phase 7 type-specific answer fields ------------------------------
    # Only meaningful for their own type; every other type leaves them unset.
    # These map 1:1 onto the ``quiz_questions`` columns the grader reads.
    single_answer: bool = True
    """``multiple_choice`` only. False → multi-select (>=1 correct option)."""

    numeric_answer: Decimal | None = None
    """``numerical`` only. The expected value."""

    numeric_tolerance: Decimal | None = None
    """``numerical`` only. Accepted absolute deviation (``>= 0``)."""

    match_pairs: list[dict[str, str]] | None = None
    """``matching`` only. ``[{"left": .., "right": ..}]`` — the answer key."""

    ordering_sequence: list[str] | None = None
    """``ordering`` only. Items in their CORRECT order (shuffled for students)."""

    @model_validator(mode="after")
    def _check_shape(self) -> GeneratedQuestion:
        if self.question_type == "multiple_choice":
            validate_multiple_choice(self)
        elif self.question_type == "true_false":
            validate_true_false(self)
        elif self.question_type == "fill_blank":
            validate_fill_blank(self)
        elif self.question_type == "short_answer":
            validate_short_answer(self)
        elif self.question_type == "numerical":
            validate_numerical(self)
        elif self.question_type == "matching":
            validate_matching(self)
        elif self.question_type == "ordering":
            validate_ordering(self)
        return self


def parse_generation_response(payload: Any) -> list[GeneratedQuestion]:  # noqa: ANN401 -- raw LLM JSON
    """Validate raw LLM JSON into normalised questions; drop bad entries."""
    raw = _extract_question_list(payload)
    out: list[GeneratedQuestion] = []
    for index, entry in enumerate(raw, start=1):
        prepared = _prepare_question(entry, default_position=index)
        if prepared is None:
            continue
        try:
            out.append(GeneratedQuestion.model_validate(prepared))
        except (ValidationError, ValueError, TypeError):
            continue
    return out


def _render_answer_for_review(qtype: str, data: dict[str, Any]) -> str:
    """Render a Phase 7 answer key as readable text for the validation stage.

    The validator judges groundedness from a compact text view of the question,
    so these answer shapes have to be flattened into prose rather than passed
    as raw JSON:

    * ``numerical`` → ``"3 (tolerance 0)"``
    * ``matching``  → ``"Extract -> Reads data; Load -> Writes data"``
    * ``ordering``  → ``"1. Extract; 2. Transform; 3. Load"``
    """
    if qtype == "numerical":
        answer = data.get("numeric_answer")
        if answer is None:
            return ""
        tolerance = data.get("numeric_tolerance")
        if tolerance is None:
            return str(answer)
        return f"{answer} (tolerance {tolerance})"

    if qtype == "matching":
        pairs = data.get("match_pairs")
        if not isinstance(pairs, list):
            return ""
        return "; ".join(
            f"{pair.get('left')} -> {pair.get('right')}"
            for pair in pairs
            if isinstance(pair, dict)
        )

    if qtype == "ordering":
        items = data.get("ordering_sequence")
        if not isinstance(items, list):
            return ""
        return "; ".join(
            f"{index}. {item}" for index, item in enumerate(items, start=1)
        )

    return ""


def question_for_review(question: GeneratedQuestion | dict[str, Any]) -> dict[str, Any]:
    """Reshape one question into the validation-stage (T5.7) input dict.

    The validator needs to see the question_type so it can apply the
    right shape rules per type. For non-MCQ questions we surface the
    expected answer text/list so the validator can judge groundedness.
    """
    data: dict[str, Any]
    if isinstance(question, dict):
        data = question
    elif hasattr(question, "model_dump"):
        data = question.model_dump()
    else:
        # Fallback for duck-typed candidate objects used in tests — read
        # their attributes directly.
        data = {
            "prompt_text": getattr(question, "prompt_text", None),
            "question_type": getattr(question, "question_type", None),
            "options": getattr(question, "options", None),
            "explanation": getattr(question, "explanation", None),
            "bloom_level": getattr(question, "bloom_level", None),
            "difficulty": getattr(question, "difficulty", None),
            "source_refs": getattr(question, "source_refs", None),
            "original_generated_payload": getattr(
                question, "original_generated_payload", None
            ),
        }
    options = data.get("options") or []
    options_dict: dict[str, str] = {}
    correct: str | None = None
    if isinstance(options, list):
        for opt in options:
            key = opt.get("option_key") if isinstance(opt, dict) else None
            if isinstance(key, str):
                options_dict[key] = opt.get("option_text", "")
                if opt.get("is_correct"):
                    correct = key
    payload = data.get("original_generated_payload") or {}
    qtype = data.get("question_type")
    if qtype in {"short_answer", "fill_blank"}:
        correct_text = payload.get("correct_answer")
    elif qtype == "true_false":
        # Map the canonical option keys ("T"/"F") to the literal strings
        # the validator's system prompt expects ("True"/"False").
        if correct == "T":
            correct_text = "True"
        elif correct == "F":
            correct_text = "False"
        else:
            correct_text = correct
    elif qtype in {"numerical", "matching", "ordering"}:
        # Phase 7: these carry their answer on dedicated fields, not option
        # rows. Render it as readable text so the validator can judge
        # groundedness — without this it would see an EMPTY correct_answer and
        # reject every question of these types as unsupported by the source.
        correct_text = _render_answer_for_review(qtype, data)
    else:
        correct_text = correct
    if qtype == "multiple_choice" and not data.get("single_answer", True):
        # Multi-select: surface every correct letter, not just the last one
        # seen, so the validator judges the whole answer set.
        correct_keys = sorted(
            str(opt.get("option_key"))
            for opt in (options if isinstance(options, list) else [])
            if isinstance(opt, dict) and opt.get("is_correct")
        )
        if correct_keys:
            correct_text = ", ".join(correct_keys)
    return {
        "prompt_text": data.get("prompt_text"),
        "question_type": data.get("question_type"),
        "options": options_dict,
        "correct_answer": correct_text,
        "explanation": data.get("explanation"),
        "bloom_level": data.get("bloom_level"),
        "difficulty": data.get("difficulty"),
    }


def normalize_question_text(text: str) -> str:
    """Lowercase + strip punctuation/whitespace for dedup keys."""
    translator = str.maketrans("", "", string.punctuation)
    return " ".join(text.lower().translate(translator).split())


def _extract_question_list(payload: Any) -> list[Any]:  # noqa: ANN401 -- raw LLM JSON
    if isinstance(payload, dict):
        questions = payload.get("questions")
        return questions if isinstance(questions, list) else []
    return payload if isinstance(payload, list) else []


def _prepare_question(entry: Any, *, default_position: int) -> dict[str, Any] | None:  # noqa: ANN401 -- raw LLM JSON
    if not isinstance(entry, dict):
        return None
    raw_question = entry.get("question") or entry.get("prompt_text")
    if not isinstance(raw_question, str) or not raw_question.strip():
        return None

    question_type = _normalize_question_type(entry.get("question_type"))
    if question_type not in _VALID_TYPES:
        return None
    correct_raw = entry.get("correct_answer") or entry.get("correct")
    options = normalize_options(entry.get("options"), correct_raw, question_type)

    canonical_payload = dict(entry)
    canonical_payload["question_type"] = question_type
    if question_type == "fill_blank":
        canonical_payload["correct_answer"] = coerce_fill_blank_answer(correct_raw)
    elif question_type == "short_answer":
        canonical_payload["correct_answer"] = (
            correct_raw.strip() if isinstance(correct_raw, str) else ""
        )

    source_refs = entry.get("source_refs") or entry.get("source_chunk_ids") or []
    if not isinstance(source_refs, list):
        source_refs = []

    return {
        "position": int(entry.get("position") or default_position),
        "question_type": question_type,
        "prompt_text": raw_question.strip(),
        "hint_text": entry.get("hint") or entry.get("hint_text"),
        "explanation": (entry.get("explanation") or "").strip() or "(no explanation)",
        "difficulty": entry.get("difficulty") or "medium",
        "bloom_level": entry.get("bloom_level") or "understand",
        # Phase 3: only accept known format tokens; anything else (including a
        # drifted LLM value) falls back to ``plain`` so the persistence stage
        # never writes an unsanitized field under a bogus discriminator.
        "prompt_format": _normalize_format(entry.get("prompt_format")),
        "hint_format": _normalize_format(entry.get("hint_format")),
        "explanation_format": _normalize_format(entry.get("explanation_format")),
        "expected_response_ms": int(entry.get("expected_response_ms") or 60000),
        "source_refs_json": [str(ref) for ref in source_refs],
        "original_generated_payload": canonical_payload,
        "options": options,
        # Phase 7 answer keys. Each is only read by its own type's validator,
        # so passing them unconditionally is safe — a stray ``match_pairs`` on
        # an MCQ is ignored (and MCQ's validator rejects nothing on its
        # account). ``single_answer`` defaults True unless the model explicitly
        # asks for multi-select.
        "single_answer": _coerce_single_answer(entry, question_type),
        "numeric_answer": _coerce_decimal(
            entry.get("numeric_answer")
            if entry.get("numeric_answer") is not None
            else correct_raw
        )
        if question_type == "numerical"
        else None,
        "numeric_tolerance": _coerce_decimal(entry.get("numeric_tolerance"))
        if question_type == "numerical"
        else None,
        "match_pairs": _coerce_match_pairs(
            entry.get("match_pairs")
            if entry.get("match_pairs") is not None
            else entry.get("pairs")
        )
        if question_type == "matching"
        else None,
        "ordering_sequence": _coerce_ordering_sequence(
            entry.get("ordering_sequence")
            if entry.get("ordering_sequence") is not None
            else (entry.get("items") if entry.get("items") is not None else correct_raw)
        )
        if question_type == "ordering"
        else None,
    }


def _coerce_single_answer(entry: dict[str, Any], question_type: str) -> bool:
    """Decide the ``single_answer`` flag for a generated question.

    Only ``multiple_choice`` supports multi-select. The model may signal it via
    an explicit ``single_answer``/``multiple_correct`` flag, or implicitly by
    marking more than one option correct — honour either so a model that just
    flags two correct answers doesn't get rejected by the single-answer rule.
    """
    if question_type != "multiple_choice":
        return True
    raw = entry.get("single_answer")
    if isinstance(raw, bool):
        return raw
    multi = entry.get("multiple_correct") or entry.get("multi_select")
    if isinstance(multi, bool):
        return not multi
    correct = entry.get("correct_answer") or entry.get("correct")
    if isinstance(correct, list) and len(correct) > 1:
        return False
    options_raw = entry.get("options")
    if isinstance(options_raw, list):
        flagged = sum(
            1
            for item in options_raw
            if isinstance(item, dict) and item.get("is_correct")
        )
        if flagged > 1:
            return False
    return True


__all__ = [
    "GeneratedQuestion",
    "GeneratedQuestionOption",
    "QuizQuestionType",
    "normalize_question_text",
    "parse_generation_response",
    "question_for_review",
]
