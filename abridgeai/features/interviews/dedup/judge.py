"""Stage 2 of duplicate detection: the semantic same-or-different judgement.

Takes the shortlist produced by :mod:`shortlist` and asks one LLM call whether the
proposed question actually duplicates any of them. One call per proposed question
regardless of shortlist size — every candidate goes in the same prompt.

Fail-open by design
-------------------
A failed or unparseable judgement returns "not a duplicate". This runs while a
teacher is saving or generating questions, and blocking a legitimate save because
a provider hiccuped is worse than letting one near-duplicate through — the teacher
can still see and delete it in the bank. The failure is recorded in ``error`` so a
caller can surface "duplicate check unavailable" instead of implying a clean pass.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from abridgeai.ai.llm import LLMRole
from abridgeai.ai.llm.gateway import LLMGateway
from abridgeai.ai.prompts import render_prompt

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.interviews.dedup.shortlist import ShortlistedQuestion

logger = logging.getLogger(__name__)

DEDUP_STAGE_NAME = "interview_dedup"


@dataclass(frozen=True, slots=True)
class DuplicateVerdict:
    """Outcome of the duplicate check for one proposed question."""

    is_duplicate: bool
    duplicate_of_id: UUID | None = None
    duplicate_of_text: str = ""
    rationale: str = ""
    error: str = ""
    """Non-empty when the check could not be completed. ``is_duplicate`` is then
    ``False`` (fail-open), so callers must read this before reporting "no
    duplicates found"."""

    @property
    def checked(self) -> bool:
        """True when a real judgement was made (no error)."""
        return not self.error

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_duplicate": self.is_duplicate,
            "duplicate_of_id": (
                str(self.duplicate_of_id) if self.duplicate_of_id is not None else None
            ),
            "duplicate_of_text": self.duplicate_of_text,
            "rationale": self.rationale,
            "error": self.error,
        }


NOT_DUPLICATE = DuplicateVerdict(is_duplicate=False)


async def judge_duplicate(
    db: AsyncSession,
    *,
    prompt_text: str,
    candidates: list[ShortlistedQuestion],
    gateway: LLMGateway | None = None,
    pipeline_run_id: UUID | None = None,
) -> DuplicateVerdict:
    """Decide whether ``prompt_text`` duplicates one of ``candidates``.

    Returns :data:`NOT_DUPLICATE` without an LLM call when there is nothing to
    compare against — an empty shortlist is the common case for a fresh bank and
    must not cost anything.
    """
    if not candidates:
        return NOT_DUPLICATE

    cleaned = (prompt_text or "").strip()
    if not cleaned:
        return NOT_DUPLICATE

    try:
        system_prompt = render_prompt("prompts/duplicate_system.j2")
        user_prompt = json.dumps(
            {
                "proposed_question": cleaned,
                "existing_questions": [c.prompt_text for c in candidates],
            },
            ensure_ascii=False,
        )
        gateway = gateway or LLMGateway()
        result = await gateway.generate_json(
            role=LLMRole.INTERVIEW_DEDUP,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            db=db,
            stage_name=DEDUP_STAGE_NAME,
            pipeline_run_id=pipeline_run_id,
        )
    except Exception as exc:  # noqa: BLE001 — fail open; see module docstring
        logger.warning("interview dedup judge failed; treating as not-duplicate: %s", exc)
        return DuplicateVerdict(is_duplicate=False, error=f"judge call failed: {exc}")

    payload = result.content_json if isinstance(result.content_json, dict) else None
    return parse_duplicate_verdict(payload, candidates=candidates)


def parse_duplicate_verdict(
    payload: dict[str, Any] | None,
    *,
    candidates: list[ShortlistedQuestion],
) -> DuplicateVerdict:
    """Validate a judge response into a :class:`DuplicateVerdict`.

    Strict about the index: a duplicate claim that points outside the candidate
    list is treated as an error rather than silently attached to the wrong
    question, because the whole value of the verdict is *which* question it
    duplicates.
    """
    if not isinstance(payload, dict):
        return DuplicateVerdict(is_duplicate=False, error="unparseable judge response")

    raw_flag = payload.get("is_duplicate")
    if not isinstance(raw_flag, bool):
        return DuplicateVerdict(is_duplicate=False, error="missing is_duplicate boolean")

    rationale = payload.get("rationale")
    rationale_text = rationale.strip() if isinstance(rationale, str) else ""

    if not raw_flag:
        return DuplicateVerdict(is_duplicate=False, rationale=rationale_text)

    raw_index = payload.get("duplicate_of_index")
    # bool is an int subclass in Python; True must not read as index 1.
    if isinstance(raw_index, bool) or not isinstance(raw_index, int):
        return DuplicateVerdict(
            is_duplicate=False,
            rationale=rationale_text,
            error="duplicate claimed without a usable duplicate_of_index",
        )
    if not 0 <= raw_index < len(candidates):
        return DuplicateVerdict(
            is_duplicate=False,
            rationale=rationale_text,
            error=f"duplicate_of_index {raw_index} out of range",
        )

    match = candidates[raw_index]
    return DuplicateVerdict(
        is_duplicate=True,
        duplicate_of_id=match.question_id,
        duplicate_of_text=match.prompt_text,
        rationale=rationale_text,
    )


__all__ = [
    "DEDUP_STAGE_NAME",
    "NOT_DUPLICATE",
    "DuplicateVerdict",
    "judge_duplicate",
    "parse_duplicate_verdict",
]
