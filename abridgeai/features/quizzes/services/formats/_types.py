"""Shared parse/serialize data structures for quiz import/export (Phase 11).

Format-agnostic intermediate representation. GIFT and Moodle-XML parsers both
produce a :class:`ParseResult`; the serializers consume the same shape. Zero DB
access — these are pure value objects so parsers/serializers are unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParsedOption:
    text: str
    is_correct: bool


@dataclass(frozen=True)
class ParsedQuestion:
    question_type: str
    prompt_text: str
    options: list[ParsedOption] = field(default_factory=list)
    correct_answer: str | None = None
    explanation: str | None = None


@dataclass(frozen=True)
class ParseResult:
    questions: list[ParsedQuestion] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


__all__ = ["ParseResult", "ParsedOption", "ParsedQuestion"]
