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


def due_cards_body(
    *,
    lesson_counts: list[tuple[str, int]] | None = None,
    due_count: int = 0,
    locale: str | None = None,
) -> str:
    """Body for the due-cards notification, naming the lessons involved.

    ``lesson_counts`` is ``[(lesson_title, card_count), ...]`` ordered with the
    largest backlog first (see ``scan_due_cards._dispatch_for_student``).

    A bare "review them now" body threw away the per-lesson structure that SM-2
    scheduling actually produces: a student enrolled in several courses could not
    tell which one needed attention without clicking through. Naming up to two
    lessons keeps the line readable while making it actionable.

    Falls back to the generic wording when no titles could be resolved (e.g. a
    lesson was deleted between the scan and the dispatch), so the notification
    still sends rather than rendering an empty body.
    """
    lang = _norm(locale)
    counts = lesson_counts or []

    if not counts:
        if lang == "vi":
            return "Hãy ôn tập ngay để ghi nhớ kiến thức lâu hơn."
        return "Review them now to keep your knowledge fresh."

    # Name at most two lessons; summarise any remainder as a count so the body
    # stays one short line regardless of how many lessons are involved.
    head = counts[:2]
    rest = len(counts) - len(head)

    if lang == "vi":
        parts = [f"{title} ({count} thẻ)" for title, count in head]
        joined = ", ".join(parts)
        if rest > 0:
            joined += f" và {rest} bài học khác"
        return f"Cần ôn tập: {joined}."

    parts = [f"{title} ({count} card{'s' if count != 1 else ''})" for title, count in head]
    joined = ", ".join(parts)
    if rest > 0:
        joined += f", and {rest} more lesson{'s' if rest != 1 else ''}"
    return f"Due for review: {joined}."


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


# ── Course assignment / publish (courses + enrollments features) ──────────────


def course_teacher_assigned_title(*, course_title: str, locale: str | None) -> str:
    """Title: a teacher was assigned to a course (draft or published)."""
    lang = _norm(locale)
    if lang == "vi":
        return f"Bạn được phân công dạy: {course_title}"[:255]
    return f"You've been assigned to teach: {course_title}"[:255]


def course_teacher_assigned_body(*, course_title: str, locale: str | None) -> str:
    lang = _norm(locale)
    if lang == "vi":
        return (
            f'Bạn đã được phân công làm giảng viên cho khoá học "{course_title}". '
            "Mở khoá học để bắt đầu quản lý nội dung."
        )
    return (
        f'You have been assigned as a teacher for "{course_title}". '
        "Open the course to start managing its content."
    )


def course_enrolled_title(*, course_title: str, locale: str | None) -> str:
    """Title: a student was enrolled in a (published) course."""
    lang = _norm(locale)
    if lang == "vi":
        return f"Bạn đã được ghi danh vào: {course_title}"[:255]
    return f"You're enrolled in: {course_title}"[:255]


def course_enrolled_body(*, course_title: str, locale: str | None) -> str:
    lang = _norm(locale)
    if lang == "vi":
        return (
            f'Bạn đã được ghi danh vào khoá học "{course_title}". '
            "Mở khoá học để bắt đầu học."
        )
    return f'You have been enrolled in "{course_title}". Open the course to start learning.'


def course_published_teacher_title(*, course_title: str, locale: str | None) -> str:
    """Title: a course you teach was published."""
    lang = _norm(locale)
    if lang == "vi":
        return f"Khoá học đã xuất bản: {course_title}"[:255]
    return f"Course published: {course_title}"[:255]


def course_published_teacher_body(*, course_title: str, locale: str | None) -> str:
    lang = _norm(locale)
    if lang == "vi":
        return (
            f'Khoá học bạn phụ trách "{course_title}" đã được xuất bản và hiện '
            "đã hiển thị với học viên."
        )
    return (
        f'The course you teach, "{course_title}", is now published and visible to learners.'
    )


def course_published_student_title(*, course_title: str, locale: str | None) -> str:
    """Title: a course you're enrolled in was published."""
    lang = _norm(locale)
    if lang == "vi":
        return f"Khoá học đã sẵn sàng: {course_title}"[:255]
    return f"Course now available: {course_title}"[:255]


def course_published_student_body(*, course_title: str, locale: str | None) -> str:
    lang = _norm(locale)
    if lang == "vi":
        return (
            f'Khoá học "{course_title}" bạn đã ghi danh đã được xuất bản. '
            "Mở khoá học để bắt đầu học."
        )
    return (
        f'The course you\'re enrolled in, "{course_title}", is now published. '
        "Open it to start learning."
    )


__all__ = [
    "Locale",
    "course_enrolled_body",
    "course_enrolled_title",
    "course_published_student_body",
    "course_published_student_title",
    "course_published_teacher_body",
    "course_published_teacher_title",
    "course_teacher_assigned_body",
    "course_teacher_assigned_title",
    "due_cards_body",
    "due_cards_title",
    "remediation_body",
    "remediation_title",
]
