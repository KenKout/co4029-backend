"""Localized notification copy (EN/VI) rendered at creation time.

Notification ``title``/``body`` are persisted as plain strings on the
``notifications`` row (no ``payload`` column -- see ``models.py``), so the
text must be rendered in the recipient's language *before* the row is
written. This module is the single place that owns that copy.

Language resolution lives upstream: callers pass the recipient's
normalized locale (``'en'`` | ``'vi'``) obtained from
``identity.api.public.get_user_locale``. Unknown locales fall back to
``'en'`` here as a second backstop, mirroring the frontend i18next
``fallbackLng``.

Bilingual discipline mirrors ``interviews/orchestrator/utterance.py``:
every template ships an EN and a VI variant keyed by locale, and the
render helpers never emit a hardcoded English literal.
"""

from __future__ import annotations

from typing import Literal

Locale = Literal["en", "vi"]


def _norm(locale: str | None) -> Locale:
    """Normalize an arbitrary locale value to a supported one."""
    if locale == "vi":
        return "vi"
    return "en"


# ── Spaced-repetition: due cards (scan_due_cards worker) ──────────────────────


def due_cards_title(*, due_count: int, locale: str | None) -> str:
    """Title for the hourly "you have N cards due" notification."""
    lang = _norm(locale)
    if lang == "vi":
        # Vietnamese has no plural inflection on the noun ("thẻ" = card/cards).
        return f"Bạn có {due_count} thẻ cần ôn tập"
    plural = "s" if due_count != 1 else ""
    return f"You have {due_count} card{plural} due"


def due_cards_body(*, locale: str | None) -> str:
    """Body for the due-cards notification."""
    lang = _norm(locale)
    if lang == "vi":
        return "Hãy ôn tập ngay để ghi nhớ kiến thức lâu hơn."
    return "Review them now to keep your knowledge fresh."


# ── Spaced-repetition: remediation on card failure (remediation service) ──────


def remediation_title(
    *,
    missed_concepts: list[str],
    primary_resource: str | None,
    locale: str | None,
) -> str:
    """Title for the "review needed" remediation notification.

    Mirrors the concept-summary / resource-fallback logic that previously
    lived inline in ``remediation._compose_payload``.
    """
    lang = _norm(locale)
    if missed_concepts:
        head = ", ".join(missed_concepts[:2])
        if len(missed_concepts) > 2:
            extra = len(missed_concepts) - 2
            head += f" (+{extra} khái niệm khác)" if lang == "vi" else f" (+{extra} more)"
    elif lang == "vi":
        head = primary_resource or "thẻ này"
    else:
        head = primary_resource or "this card"

    title = f"Cần ôn tập: {head}" if lang == "vi" else f"Review needed: {head}"
    return title[:255]


def remediation_body(*, resource_links: list[tuple[str, str]], locale: str | None) -> str:
    """Body for the remediation notification.

    ``resource_links`` is a list of ``(label, deep_link)`` pairs; the body
    is rendered as Markdown (the learner inbox parses it), so the links are
    emitted as ``- [label](deep_link)`` regardless of locale -- only the
    lead-in sentence is translated.
    """
    lang = _norm(locale)
    if lang == "vi":
        lead = (
            "Bạn đã trả lời sai câu hỏi này. Hãy xem lại các tài nguyên bên dưới "
            "trước khi làm lại để nắm vững khái niệm."
        )
    else:
        lead = (
            "You missed this question. Review the linked resources before retry "
            "to lock the concept in."
        )
    lines = [lead, ""]
    for label, deep_link in resource_links:
        lines.append(f"- [{label}]({deep_link})")
    return "\n".join(lines)


__all__ = [
    "Locale",
    "due_cards_body",
    "due_cards_title",
    "remediation_body",
    "remediation_title",
]
