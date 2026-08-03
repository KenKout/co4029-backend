"""Raw-entry preparation for the quiz generation stage.

Split out of ``parsers.py`` (which grew past the 250-LOC god-file budget). This
module owns the step that turns ONE raw LLM question entry into the canonical
kwargs dict that ``GeneratedQuestion.model_validate`` consumes:

* ``_extract_question_list`` — pull the question array out of the LLM payload
  (tolerating both ``{"questions": [...]}`` and a bare list).
* ``_prepare_question`` — normalise a single entry: resolve the type, shape the
  options, coerce the Phase-7 answer keys, and fail-safe every field.
* ``_coerce_single_answer`` — decide the multi-select flag for MCQs.

``parsers.py`` re-exports these so existing imports keep working.
"""

from __future__ import annotations

from typing import Any

from abridgeai.features.quizzes.ai.stages.generation.coercions import (
    _VALID_TYPES,
    _coerce_decimal,
    _coerce_match_distractors,
    _coerce_match_pairs,
    _coerce_ordering_sequence,
    _normalize_format,
    _normalize_question_type,
)
from abridgeai.features.quizzes.ai.stages.generation.option_normalizers import (
    coerce_fill_blank_answer,
    normalize_options,
)


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
            entry.get("numeric_answer") if entry.get("numeric_answer") is not None else correct_raw
        )
        if question_type == "numerical"
        else None,
        "numeric_tolerance": _coerce_decimal(entry.get("numeric_tolerance"))
        if question_type == "numerical"
        else None,
        "match_pairs": _coerce_match_pairs(
            entry.get("match_pairs") if entry.get("match_pairs") is not None else entry.get("pairs")
        )
        if question_type == "matching"
        else None,
        "match_distractors": _coerce_match_distractors(
            entry.get("match_distractors")
            if entry.get("match_distractors") is not None
            else entry.get("distractors")
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
            1 for item in options_raw if isinstance(item, dict) and item.get("is_correct")
        )
        if flagged > 1:
            return False
    return True


__all__ = [
    "_coerce_single_answer",
    "_extract_question_list",
    "_prepare_question",
]
