"""Loader + schema validator for the interview-security red-team corpus.

The corpus is a versioned set of JSON files under ``corpus/``. Each case is a
labeled prompt-injection attempt (or a benign-but-scary academic answer, or an
output-leakage case) with the expected classifier decision. The corpus doubles
as:

* the **baseline** snapshot for currently-covered behavior (``status: covered``),
  and
* the **forward spec** for hardening work not yet shipped (``status: gap``,
  tagged with the ``target_phase`` that is expected to close it).

No case in this module executes, decodes, or follows embedded content — every
string is inert test data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

_CORPUS_DIR = Path(__file__).resolve().parent / "corpus"

# Input-layer files carry classification cases; the output file carries leakage
# cases with a different shape.
_INPUT_FILES = (
    "covered_input.json",
    "benign.json",
    "gap_input_vectors.json",
    "gap_behavioral.json",
)
_OUTPUT_FILE = "output_leakage.json"

_VALID_CATEGORIES = {
    "benign",
    "future_question_request",
    "answer_key_request",
    "rubric_exfiltration",
    "system_prompt_request",
    "instruction_override",
    "roleplay_bypass",
    "grading_manipulation",
    "hidden_state_request",
    "encoded_exfiltration",
    "cross_session_data_request",
}
# "covered": rules already catch it (baseline). "gap": a forward-spec case a
# later hardening phase must catch (expected to fail at baseline).
# "must_stay_benign": a false-positive guard — an academic answer that uses
# scary vocabulary but must NEVER be blocked, in any phase.
_VALID_STATUS = {"covered", "gap", "must_stay_benign"}


@dataclass(frozen=True)
class InputCase:
    """One input-classification corpus case."""

    id: str
    text: str
    lang: str
    expected_category: str
    expected_block: bool
    status: Literal["covered", "gap", "must_stay_benign"]
    target_phase: str
    tags: tuple[str, ...] = ()
    # When True the case is only expected to be caught via the semantic
    # classifier (not deterministic rules); the rules assertion is relaxed.
    classifier_only: bool = False
    # When True this is a KNOWN pre-existing false positive that the current
    # code exhibits and a later phase (see target_phase) will fix. The suite
    # records it as an xfail instead of hard-failing, so the baseline stays
    # green while the debt is tracked. Remove the flag when the phase ships.
    baseline_fp: bool = False
    notes: str = ""


@dataclass(frozen=True)
class OutputCase:
    """One output-leakage corpus case."""

    id: str
    text: str
    protected: tuple[dict[str, str], ...]
    expected_block: bool
    status: Literal["covered", "gap", "must_stay_benign"]
    target_phase: str
    expected_method: str | None = None
    tags: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class Corpus:
    input_cases: tuple[InputCase, ...] = field(default_factory=tuple)
    output_cases: tuple[OutputCase, ...] = field(default_factory=tuple)


def _read_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path.name}: expected a top-level JSON array")
    return payload


def _validate_input_case(raw: dict[str, Any], *, source: str) -> InputCase:
    missing = {
        "id",
        "text",
        "expected_category",
        "expected_block",
        "status",
        "target_phase",
    } - raw.keys()
    if missing:
        raise ValueError(f"{source}: case missing keys {sorted(missing)}: {raw!r}")
    category = str(raw["expected_category"])
    if category not in _VALID_CATEGORIES:
        raise ValueError(f"{source}: unknown category {category!r} in case {raw['id']!r}")
    status = str(raw["status"])
    if status not in _VALID_STATUS:
        raise ValueError(f"{source}: unknown status {status!r} in case {raw['id']!r}")
    return InputCase(
        id=str(raw["id"]),
        text=str(raw["text"]),
        lang=str(raw.get("lang", "en")),
        expected_category=category,
        expected_block=bool(raw["expected_block"]),
        status=status,  # type: ignore[arg-type]
        target_phase=str(raw["target_phase"]),
        tags=tuple(raw.get("tags", ())),
        classifier_only=bool(raw.get("classifier_only", False)),
        baseline_fp=bool(raw.get("baseline_fp", False)),
        notes=str(raw.get("notes", "")),
    )


def _validate_output_case(raw: dict[str, Any], *, source: str) -> OutputCase:
    missing = {"id", "text", "protected", "expected_block", "status", "target_phase"} - raw.keys()
    if missing:
        raise ValueError(f"{source}: output case missing keys {sorted(missing)}: {raw!r}")
    protected = tuple(
        {"category": str(item["category"]), "text": str(item["text"])}
        for item in raw.get("protected", [])
    )
    status = str(raw["status"])
    if status not in _VALID_STATUS:
        raise ValueError(f"{source}: unknown status {status!r} in case {raw['id']!r}")
    return OutputCase(
        id=str(raw["id"]),
        text=str(raw["text"]),
        protected=protected,
        expected_block=bool(raw["expected_block"]),
        status=status,  # type: ignore[arg-type]
        target_phase=str(raw["target_phase"]),
        expected_method=(
            str(raw["expected_method"]) if raw.get("expected_method") is not None else None
        ),
        tags=tuple(raw.get("tags", ())),
        notes=str(raw.get("notes", "")),
    )


@lru_cache(maxsize=1)
def load_corpus() -> Corpus:
    """Load, validate, and de-duplicate every corpus case (cached)."""
    input_cases: list[InputCase] = []
    seen_ids: set[str] = set()
    for name in _INPUT_FILES:
        path = _CORPUS_DIR / name
        for raw in _read_json(path):
            case = _validate_input_case(raw, source=name)
            if case.id in seen_ids:
                raise ValueError(f"duplicate corpus id {case.id!r} (in {name})")
            seen_ids.add(case.id)
            input_cases.append(case)

    output_cases: list[OutputCase] = []
    for raw in _read_json(_CORPUS_DIR / _OUTPUT_FILE):
        case = _validate_output_case(raw, source=_OUTPUT_FILE)
        if case.id in seen_ids:
            raise ValueError(f"duplicate corpus id {case.id!r} (in {_OUTPUT_FILE})")
        seen_ids.add(case.id)
        output_cases.append(case)

    return Corpus(input_cases=tuple(input_cases), output_cases=tuple(output_cases))


def input_cases() -> tuple[InputCase, ...]:
    return load_corpus().input_cases


def output_cases() -> tuple[OutputCase, ...]:
    return load_corpus().output_cases


__all__ = [
    "Corpus",
    "InputCase",
    "OutputCase",
    "input_cases",
    "load_corpus",
    "output_cases",
]
