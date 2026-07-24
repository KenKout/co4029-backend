"""Moodle GIFT import parser + export serializer (Phase 11).

Pure functions — no DB. ``parse_gift`` turns a GIFT file into a
:class:`ParseResult`; ``serialize_gift`` turns parsed questions back into GIFT.
Unsupported constructs (cloze / embedded) are collected as warnings rather than
failing the whole file; only a structurally broken block raises.
"""

from __future__ import annotations

import re

from abridgeai.features.quizzes.services.formats._types import (
    ParsedOption,
    ParsedQuestion,
    ParseResult,
)

_COMMENT = re.compile(r"^\s*//.*$", re.MULTILINE)
_TITLE = re.compile(r"^::(?P<title>.*?)::", re.DOTALL)


class GiftParseError(ValueError):
    """Raised when a GIFT block cannot be parsed at all."""


def _split_blocks(text: str) -> list[str]:
    text = _COMMENT.sub("", text)
    return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]


def _unescape(s: str) -> str:
    return re.sub(r"\\([=~{}#:])", r"\1", s).strip()


def parse_gift(text: str) -> ParseResult:
    questions: list[ParsedQuestion] = []
    warnings: list[str] = []
    for idx, block in enumerate(_split_blocks(text), start=1):
        q = _parse_block(block, idx, warnings)
        if q is not None:
            questions.append(q)
    return ParseResult(questions=questions, warnings=warnings)


def _parse_block(block: str, idx: int, warnings: list[str]) -> ParsedQuestion | None:
    m = _TITLE.match(block)
    if m:
        block = block[m.end() :]
    lbrace, rbrace = block.find("{"), block.rfind("}")
    if lbrace == -1 or rbrace == -1 or rbrace < lbrace:
        raise GiftParseError(f"Q{idx}: no answer block")
    prompt = _unescape(block[:lbrace])
    body = block[lbrace + 1 : rbrace].strip()

    if re.search(r"\d+:[A-Z]+:", body) or ":MULTICHOICE:" in body or ":SHORTANSWER:" in body:
        warnings.append(f"Q{idx}: embedded/cloze question unsupported, skipped")
        return None

    explanation = None
    if "####" in body:
        body, _, fb = body.partition("####")
        explanation = _unescape(fb)
        body = body.strip()

    upper = body.upper()
    if upper in {"T", "TRUE", "F", "FALSE"}:
        is_true = upper in {"T", "TRUE"}
        return ParsedQuestion(
            question_type="true_false",
            prompt_text=prompt,
            options=[
                ParsedOption(text="True", is_correct=is_true),
                ParsedOption(text="False", is_correct=not is_true),
            ],
            explanation=explanation,
        )

    entries = _split_answer_entries(body)
    has_wrong = any(sign == "~" for sign, _ in entries)
    parsed = [(sign, _unescape(re.sub(r"%[-\d.]+%", "", val))) for sign, val in entries]

    if not has_wrong and parsed and all(sign == "=" for sign, _ in parsed):
        return ParsedQuestion(
            question_type="short_answer",
            prompt_text=prompt,
            correct_answer=parsed[0][1],
            explanation=explanation,
        )

    opts = [ParsedOption(text=val, is_correct=(sign == "=")) for sign, val in parsed]
    if not opts:
        raise GiftParseError(f"Q{idx}: empty answer block")
    return ParsedQuestion(
        question_type="multiple_choice",
        prompt_text=prompt,
        options=opts,
        explanation=explanation,
    )


def _split_answer_entries(body: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    token = ""
    sign: str | None = None
    i = 0
    while i < len(body):
        c = body[i]
        if c in "=~" and (i == 0 or body[i - 1] != "\\"):
            if sign is not None:
                entries.append((sign, token.strip()))
            sign = c
            token = ""
        else:
            token += c
        i += 1
    if sign is not None:
        entries.append((sign, token.strip()))
    return entries


def _escape(s: str) -> str:
    return re.sub(r"([=~{}#:])", r"\\\1", s)


def serialize_gift(questions: list[ParsedQuestion]) -> str:
    """Serialize parsed questions back to GIFT text."""
    blocks: list[str] = []
    for q in questions:
        prompt = _escape(q.prompt_text)
        if q.question_type == "true_false":
            correct = next((o.is_correct for o in q.options if o.text.lower() == "true"), True)
            body = "TRUE" if correct else "FALSE"
        elif q.question_type == "short_answer":
            body = f"={_escape(q.correct_answer or '')}"
        elif q.question_type == "multiple_choice":
            body = " ".join(
                f"{'=' if o.is_correct else '~'}{_escape(o.text)}" for o in q.options
            )
        else:
            # Unsupported for GIFT export — emit a short-answer placeholder note.
            blocks.append(f"// Q ({q.question_type}) not exportable to GIFT, skipped")
            continue
        if q.explanation:
            body = f"{body} ####{_escape(q.explanation)}"
        blocks.append(f"{prompt}{{{body}}}")
    return "\n\n".join(blocks) + "\n"


__all__ = ["GiftParseError", "parse_gift", "serialize_gift"]
