"""Parser for the GAP REPORT stage LLM output (T6.9).

The stage prompt asks the synthesizer to return strict JSON of the shape::

    {
      "strengths": ["criterion: bullet", ...],
      "weaknesses": ["criterion: bullet", ...],
      "study_plan": [
        {
          "topic": "...",
          "weakness_summary": "...",
          "suggested_lesson_id": "<uuid or null>",
          "suggested_resource_ids": ["<uuid>", ...],
          "priority": "high|medium|low"
        }, ...
      ],
      "student_summary": "...",
      "teacher_summary": "..."
    }

This parser is intentionally permissive (mirrors the followup parser
philosophy in T6.7): malformed study-plan rows or invalid UUIDs are
dropped rather than fatal so a single bad item from the LLM doesn't
block an entire report. The caller (logic.py) is responsible for
ensuring the resource-coverage invariants hold (≥1 per item, ≥3 total).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal
from uuid import UUID

_DEFAULT_PRIORITY: Literal["high", "medium", "low"] = "medium"
_MAX_BULLET_LEN = 240
_MAX_TOPIC_LEN = 120
_MAX_SUMMARY_LEN = 1500


def parse_gap_report_response(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Coerce the gateway JSON dict into a normalised report dict.

    Returns a dict with the keys ``strengths``, ``weaknesses``,
    ``study_plan``, ``student_summary``, ``teacher_summary``. Missing
    keys default to empty values; malformed entries are dropped. The
    caller composes this with the discrepancy numbers it already owns.
    """

    if not isinstance(payload, Mapping):
        return _empty_report()

    return {
        "strengths": _coerce_bullets(payload.get("strengths")),
        "weaknesses": _coerce_bullets(payload.get("weaknesses")),
        "study_plan": _coerce_study_plan(payload.get("study_plan")),
        "student_summary": _coerce_summary(payload.get("student_summary")),
        "teacher_summary": _coerce_summary(payload.get("teacher_summary")),
    }


def _empty_report() -> dict[str, Any]:
    return {
        "strengths": [],
        "weaknesses": [],
        "study_plan": [],
        "student_summary": "",
        "teacher_summary": "",
    }


def _coerce_bullets(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    bullets: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned:
            continue
        bullets.append(cleaned[:_MAX_BULLET_LEN])
    return bullets


def _coerce_study_plan(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in value:
        item = _coerce_study_plan_item(entry)
        if item is not None:
            out.append(item)
    return out


def _coerce_study_plan_item(entry: object) -> dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    topic = entry.get("topic")
    if not isinstance(topic, str) or not topic.strip():
        return None
    weakness_summary = entry.get("weakness_summary")
    weakness_clean = weakness_summary.strip() if isinstance(weakness_summary, str) else ""
    return {
        "topic": topic.strip()[:_MAX_TOPIC_LEN],
        "weakness_summary": weakness_clean[:_MAX_BULLET_LEN],
        "suggested_lesson_id": _coerce_uuid_optional(entry.get("suggested_lesson_id")),
        "suggested_resource_ids": _coerce_uuid_list(entry.get("suggested_resource_ids")),
        "priority": _coerce_priority(entry.get("priority")),
    }


def _coerce_uuid_optional(value: object) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() == "null":
            return None
        try:
            return UUID(cleaned)
        except ValueError:
            return None
    return None


def _coerce_uuid_list(value: object) -> list[UUID]:
    if not isinstance(value, list):
        return []
    seen: set[UUID] = set()
    out: list[UUID] = []
    for item in value:
        parsed = _coerce_uuid_optional(item)
        if parsed is None or parsed in seen:
            continue
        seen.add(parsed)
        out.append(parsed)
    return out


def _coerce_priority(value: object) -> Literal["high", "medium", "low"]:
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned == "high":
            return "high"
        if cleaned == "medium":
            return "medium"
        if cleaned == "low":
            return "low"
    return _DEFAULT_PRIORITY


def _coerce_summary(value: object) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    return cleaned[:_MAX_SUMMARY_LEN]


__all__ = ["parse_gap_report_response"]
