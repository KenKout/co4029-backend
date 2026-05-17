"""Apply validator verdicts to a question batch (T5.7).

Ports ``_apply_verdicts`` from
``backend/app/ai/haystack/pipelines/quiz_generation.py:845-863``. The
new shape returns three values:

* ``accepted`` — questions whose verdict was ``accept``, in original
  order. The dict objects are returned untouched so callers can persist
  them as-is.
* ``rejected`` — minimal records carrying ``position``, ``prompt_text``,
  ``reasons`` (defect codes, possibly multiple), and ``evidence_excerpt``
  for teacher review.
* ``reasons`` — flat list of every rejection reason in order, useful for
  the audit row and quick-glance dashboards.

We deliberately keep ``reasons`` as a separate return so callers do not
have to re-derive it from the rejected list. The legacy two-tuple
return (``accepted, rejected``) is reachable via ``apply_verdicts(...)[:2]``.
"""

from __future__ import annotations

from typing import Any

from abridgeai.features.quizzes.ai.stages.validation.parsers import Verdict


def apply_verdicts(
    questions: list[dict[str, Any]],
    verdicts: list[Verdict],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Partition ``questions`` into accepted/rejected by ``verdicts``.

    Parameters
    ----------
    questions
        Output of the generation stage — each dict carries at minimum
        ``prompt_text`` and (optionally) a question identifier under
        ``id``, ``question_id`` or ``position``.
    verdicts
        Positional verdict list from
        :func:`abridgeai.features.quizzes.ai.stages.validation.parsers.parse_validation_response`.
        ``len(verdicts)`` should equal ``len(questions)`` — extra
        verdicts are ignored and missing ones default to ``accept`` so a
        flaky validator never silently rejects every question.

    Returns
    -------
    tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]
        ``(accepted, rejected, reasons)``.
    """

    by_position: dict[int, Verdict] = {v.position: v for v in verdicts}

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reasons: list[str] = []

    for index, question in enumerate(questions, start=1):
        verdict = by_position.get(index)
        if verdict is None or verdict.verdict == "accept":
            accepted.append(question)
            continue

        rejection = {
            "position": index,
            "question_id": _question_id(question, fallback=index),
            "prompt_text": question.get("prompt_text", ""),
            "reasons": list(verdict.reasons),
            "evidence_excerpt": verdict.evidence_excerpt,
        }
        rejected.append(rejection)
        reasons.extend(verdict.reasons)

    return accepted, rejected, reasons


def _question_id(question: dict[str, Any], *, fallback: int) -> object:
    """Pick the most stable identifier present on ``question``.

    Generation stage may attach ``id`` / ``question_id``; persistence
    rewrites either to a UUID. Fall back to the 1-based position so the
    rejected record always carries some referenceable handle.
    """

    for key in ("id", "question_id", "position"):
        value = question.get(key)
        if value is not None:
            return value
    return fallback


__all__ = ["apply_verdicts"]
