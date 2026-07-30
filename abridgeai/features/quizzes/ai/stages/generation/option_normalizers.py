"""Option-list normalisers for the quiz GENERATION stage parser.

Per-type helpers that the main parser delegates to. ``multiple_choice``
keeps the LLM's option dict / list, ``true_false`` synthesizes a
canonical T/F pair from the correct flag, ``fill_blank`` projects the
LLM's word-bank array into canonical option rows (correct entries +
distractors, ``is_correct`` flagged), and ``short_answer`` returns
``[]`` (its answer lives on ``original_generated_payload`` instead).
"""

from __future__ import annotations

from typing import Any


def normalize_options(
    options_raw: Any,  # noqa: ANN401 -- raw LLM JSON
    correct: Any,  # noqa: ANN401 -- raw LLM JSON
    question_type: str,
) -> list[dict[str, Any]]:
    if question_type == "multiple_choice":
        return _normalize_mcq_options(options_raw, correct)
    if question_type == "true_false":
        return _normalize_true_false_options(options_raw, correct)
    if question_type == "fill_blank":
        return _normalize_fill_blank_options(options_raw, correct)
    # short_answer / numerical / matching / ordering carry their answer on the
    # question's own columns, not option rows. Returning [] here also means an
    # LLM that wrongly emits ``options`` for these types has them discarded
    # (their validators additionally reject a non-empty option list).
    return []


def coerce_fill_blank_answer(raw: Any) -> list[str]:  # noqa: ANN401 -- raw LLM JSON
    """Return a list of blank strings, accepting list / semicolon /
    comma-separated string."""
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        candidate = raw.strip()
        if not candidate:
            return []
        for sep in (";", ","):
            if sep in candidate:
                return [piece.strip() for piece in candidate.split(sep) if piece.strip()]
        return [candidate]
    return []


def _normalize_fill_blank_options(
    options_raw: Any,  # noqa: ANN401 -- raw LLM JSON
    correct: Any,  # noqa: ANN401 -- raw LLM JSON
) -> list[dict[str, Any]]:
    """Return canonical word-bank option rows for ``fill_blank``.

    The LLM is asked to emit ``options`` as a JSON array of strings —
    the bank from which the learner drags into ``___`` slots. We:

    1. Coerce the bank to a deduped list of strings (case-insensitive
       dedup, original casing preserved).
    2. Coerce ``correct_answer`` via the shared blank-list parser and
       prepend any missing correct answers to the bank so the bank is
       guaranteed to contain every correct entry verbatim.
    3. Mark each entry ``is_correct=True`` iff its lowercased text
       matches a lowercased correct-answer token.
    4. Assign canonical keys ``O01..O99`` and 1-based positions. Keys
       fit inside the DB ``option_key VARCHAR(5)`` constraint.

    Bank entries beyond 99 are dropped (a bank that large is malformed
    anyway). Returns ``[]`` if neither bank nor correct answers parse.
    """
    correct_list = coerce_fill_blank_answer(correct)
    correct_lookup = {entry.lower() for entry in correct_list}

    bank: list[str] = []
    seen: set[str] = set()
    if isinstance(options_raw, list):
        for raw in options_raw:
            text = _coerce_fill_blank_option(raw)
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            bank.append(text)

    # Ensure every correct answer is present in the bank verbatim. If
    # the LLM forgot one, prepend it so the learner can still reach the
    # right answer; ``validate_fill_blank`` will still run and reject if
    # the bank ends up too small to be a meaningful exercise.
    for answer in correct_list:
        if answer.lower() in seen:
            continue
        seen.add(answer.lower())
        bank.insert(0, answer)

    bank = bank[:99]
    return [
        {
            "option_key": f"O{position:02d}",
            "option_text": text,
            "is_correct": text.lower() in correct_lookup,
            "position": position,
        }
        for position, text in enumerate(bank, start=1)
    ]


def _coerce_fill_blank_option(raw: Any) -> str:  # noqa: ANN401 -- raw LLM JSON
    """Coerce one bank entry into a stripped string, accepting either a
    bare string or ``{"option_text": "..."}`` / ``{"text": "..."}``.
    """
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        text = raw.get("option_text") or raw.get("text") or raw.get("value")
        if isinstance(text, str):
            return text.strip()
    return ""


def _coerce_correct_keys(correct: Any) -> set[str]:  # noqa: ANN401 -- raw LLM JSON
    """Return the set of option letters marked correct.

    Phase 7 multi-select: ``correct_answer`` may be a single letter ("B"), a
    list of letters (["A", "C"]), or a comma/slash-separated string ("A, C").
    Normalising to a set lets the same code path serve single- and multi-select.
    """
    tokens: list[str] = []
    if isinstance(correct, str):
        cleaned = correct.strip()
        for sep in (",", "/", ";", "|"):
            if sep in cleaned:
                tokens = cleaned.split(sep)
                break
        else:
            tokens = [cleaned]
    elif isinstance(correct, (list, tuple, set)):
        tokens = [str(item) for item in correct]
    return {token.strip().upper() for token in tokens if token and token.strip()}


def _normalize_mcq_options(
    options_raw: Any,  # noqa: ANN401 -- raw LLM JSON
    correct: Any,  # noqa: ANN401 -- raw LLM JSON
) -> list[dict[str, Any]]:
    correct_keys = _coerce_correct_keys(correct)
    if isinstance(options_raw, dict):
        cleaned = {
            str(key).strip().upper(): str(value).strip()
            for key, value in options_raw.items()
            if isinstance(value, str)
        }
        return [
            {
                "option_key": key,
                "option_text": cleaned.get(key, ""),
                "is_correct": key in correct_keys,
                "position": pos,
            }
            for pos, key in enumerate(["A", "B", "C", "D"], start=1)
            if key in cleaned
        ]
    if isinstance(options_raw, list):
        out: list[dict[str, Any]] = []
        for pos, item in enumerate(options_raw, start=1):
            if not isinstance(item, dict):
                continue
            key = item.get("option_key") or item.get("key")
            if not isinstance(key, str):
                continue
            out.append(
                {
                    "option_key": key.strip().upper(),
                    "option_text": str(item.get("option_text") or item.get("text") or "").strip(),
                    "is_correct": bool(item.get("is_correct")),
                    "position": int(item.get("position") or pos),
                }
            )
        return out
    return []


def _normalize_true_false_options(
    options_raw: Any,  # noqa: ANN401 -- raw LLM JSON
    correct: Any,  # noqa: ANN401 -- raw LLM JSON
) -> list[dict[str, Any]]:
    """Return canonical T/F option rows.

    Generators sometimes emit no options (just `correct_answer: "True"`)
    so we always synthesize the pair from the correct flag rather than
    trusting the LLM's option array verbatim.
    """
    correct_label = _coerce_true_false_correct(correct, options_raw)
    return [
        {
            "option_key": "T",
            "option_text": "True",
            "is_correct": correct_label is True,
            "position": 1,
        },
        {
            "option_key": "F",
            "option_text": "False",
            "is_correct": correct_label is False,
            "position": 2,
        },
    ]


def _coerce_true_false_correct(
    correct: Any,  # noqa: ANN401 -- raw LLM JSON
    options_raw: Any,  # noqa: ANN401 -- raw LLM JSON
) -> bool | None:
    """Coerce a true/false answer hint into a boolean."""
    if isinstance(correct, bool):
        return correct
    if isinstance(correct, str):
        token = correct.strip().lower()
        if token in {"true", "t", "1", "yes"}:
            return True
        if token in {"false", "f", "0", "no"}:
            return False
    if isinstance(options_raw, list):
        for item in options_raw:
            if isinstance(item, dict) and item.get("is_correct"):
                key = str(item.get("option_key") or item.get("key") or "").strip().lower()
                if key in {"t", "true"}:
                    return True
                if key in {"f", "false"}:
                    return False
    return None


__all__ = ["coerce_fill_blank_answer", "normalize_options"]
