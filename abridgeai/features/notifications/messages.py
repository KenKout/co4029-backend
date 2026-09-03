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


# ── Syllabus import (courses feature) ────────────────────────────────────────


def syllabus_import_succeeded_title(*, course_title: str, locale: str | None) -> str:
    """Title: a syllabus upload became a draft course."""
    lang = _norm(locale)
    if lang == "vi":
        return f"Đã nhập học phần: {course_title}"[:255]
    return f"Course imported: {course_title}"[:255]


def syllabus_import_succeeded_body(
    *,
    course_title: str,
    outcome_count: int,
    warnings: list[str] | None,
    locale: str | None,
) -> str:
    """Body: what was created, plus a nudge when the parser had reservations.

    The warnings themselves are NOT inlined — they are English parser codes
    and can run long. The notification says how many there are and the
    import screen shows them in full; that keeps the inbox row readable in
    both locales without half-translating machine strings.
    """
    lang = _norm(locale)
    warning_count = len(warnings or [])
    if lang == "vi":
        text = (
            f'Đã tạo bản nháp khoá học "{course_title}" từ đề cương, '
            f"kèm {outcome_count} chuẩn đầu ra."
        )
        if warning_count:
            text += f" Có {warning_count} cảnh báo cần xem lại trước khi xuất bản."
        else:
            text += " Mở khoá học để kiểm tra và bổ sung nội dung."
        return text
    text = (
        f'Created the draft course "{course_title}" from the syllabus, '
        f"with {outcome_count} learning outcome{'s' if outcome_count != 1 else ''}."
    )
    if warning_count:
        text += (
            f" {warning_count} warning{'s' if warning_count != 1 else ''} "
            "need a look before you publish."
        )
    else:
        text += " Open the course to review it and add content."
    return text


def syllabus_import_failed_title(*, filename: str | None, locale: str | None) -> str:
    """Title: a syllabus upload produced no course."""
    lang = _norm(locale)
    name = filename or ("tệp đã tải lên" if lang == "vi" else "the uploaded file")
    if lang == "vi":
        return f"Nhập học phần thất bại: {name}"[:255]
    return f"Course import failed: {name}"[:255]


def _rejection_reason_sentence(*, reason_code: str, locale: str | None) -> str:
    """The dean's canned rejection reason, in the student's language.

    The codes are a closed set (``PATH_CHANGE_REJECTION_REASON_CODES``) and the
    student must read the reason in their own language, so the mapping lives
    here rather than passing the dean's UI label through the API. ``other``
    carries no canned sentence — its detail text is mandatory and is appended
    by the caller.
    """
    lang = _norm(locale)
    if lang == "vi":
        table = {
            "insufficient_justification": "Lý do đề nghị chưa đủ thuyết phục.",
            "progress_loss_too_high": (
                "Tiến độ sẽ mất khi chuyển lộ trình là quá lớn ở thời điểm này."
            ),
            "target_path_not_suitable": (
                "Lộ trình muốn chuyển sang chưa phù hợp với hồ sơ học tập hiện tại."
            ),
            "preserve_remaining_switch": (
                "Nên giữ lại quyền đổi lộ trình còn lại cho một thay đổi cần thiết hơn."
            ),
            "advising_required": "Cần trao đổi trực tiếp với cố vấn học tập trước khi đổi.",
            "documentation_missing": "Thiếu minh chứng hoặc thông tin cần thiết cho đề nghị.",
        }
    else:
        table = {
            "insufficient_justification": "The stated justification was not sufficient.",
            "progress_loss_too_high": (
                "The progress you would lose by switching now is too high."
            ),
            "target_path_not_suitable": (
                "The target path is not a suitable fit for your current record."
            ),
            "preserve_remaining_switch": (
                "Your remaining path change is better kept for a more necessary switch."
            ),
            "advising_required": "An advising conversation is needed before a switch.",
            "documentation_missing": "Supporting information for the request was missing.",
        }
    return table.get(reason_code, "")


def discussion_comment_title(*, topic_title: str, locale: str | None) -> str:
    """New comment on a lesson discussion topic (title)."""
    lang = _norm(locale)
    if lang == "vi":
        return f"Bình luận mới trong thảo luận: {topic_title}"[:255]
    return f"New comment on: {topic_title}"[:255]


def discussion_comment_body(
    *,
    commenter_label: str,
    topic_title: str,
    course_title: str,
    comment_snippet: str,
    locale: str | None,
) -> str:
    """Who said what where (body) — the snippet answers the "do I care" test."""
    lang = _norm(locale)
    if lang == "vi":
        return (
            f'{commenter_label} vừa bình luận trong "{topic_title}" '
            f"({course_title}): “{comment_snippet}”"
        )
    return (
        f'{commenter_label} commented on "{topic_title}" in '
        f"{course_title}: “{comment_snippet}”"
    )


def path_change_requested_title(*, student_label: str, locale: str | None) -> str:
    """Dean-facing title: a student filed a path change request."""
    lang = _norm(locale)
    if lang == "vi":
        return f"Đề nghị đổi lộ trình từ {student_label}"[:255]
    return f"Path change request from {student_label}"[:255]


def path_change_requested_body(
    *,
    student_label: str,
    target_path_name: str,
    program_name: str,
    locale: str | None,
) -> str:
    """Dean-facing body: who wants to move where, and where to review it."""
    lang = _norm(locale)
    if lang == "vi":
        return (
            f'{student_label} muốn chuyển sang lộ trình "{target_path_name}" '
            f'trong chương trình {program_name}. Mở đề nghị để xem xét.'
        )
    return (
        f'{student_label} wants to switch to "{target_path_name}" in '
        f"{program_name}. Open the request to review it."
    )


def path_change_in_progress_title(*, program_name: str, locale: str | None) -> str:
    """Title: the Faculty Dean has picked the request up (no decision yet)."""
    lang = _norm(locale)
    if lang == "vi":
        return f"Đề nghị đổi lộ trình đang được xem xét: {program_name}"[:255]
    return f"Path change request under review: {program_name}"[:255]


def path_change_in_progress_body(*, target_path_name: str, locale: str | None) -> str:
    """Body: says explicitly that nothing has changed yet.

    The whole value of this signal is removing the "has anyone even looked at
    this?" doubt, so it states both facts: seen, and not decided.
    """
    lang = _norm(locale)
    if lang == "vi":
        return (
            f'Trưởng khoa đã nhận đề nghị chuyển sang "{target_path_name}" và đang '
            "kiểm tra dữ liệu học tập của bạn. Lộ trình hiện tại chưa thay đổi; "
            "bạn sẽ được thông báo khi có quyết định."
        )
    return (
        f'Your Faculty Dean has opened your request to switch to "{target_path_name}" '
        "and is checking your record. Nothing has changed yet — you will be "
        "notified when a decision is made."
    )


def path_change_rejected_title(*, program_name: str, locale: str | None) -> str:
    """Title: the request was rejected."""
    lang = _norm(locale)
    if lang == "vi":
        return f"Đề nghị đổi lộ trình bị từ chối: {program_name}"[:255]
    return f"Path change request rejected: {program_name}"[:255]


def path_change_rejected_body(
    *,
    target_path_name: str,
    reason_code: str,
    reason_detail: str | None,
    locale: str | None,
) -> str:
    """Body: the reason, then what it means for the student.

    Both the canned sentence and any free text are included — the code says
    which bucket, the detail says what specifically, and a student reading only
    the notification should not have to open the app to learn why.
    """
    lang = _norm(locale)
    canned = _rejection_reason_sentence(reason_code=reason_code, locale=locale)
    detail = (reason_detail or "").strip()
    parts: list[str] = []
    if lang == "vi":
        parts.append(f'Đề nghị chuyển sang "{target_path_name}" đã bị từ chối.')
        if canned:
            parts.append(f"Lý do: {canned}")
        if detail:
            parts.append(f"Ghi chú của trưởng khoa: {detail}")
        parts.append(
            "Bạn vẫn tiếp tục lộ trình hiện tại và quyền đổi lộ trình chưa bị trừ."
        )
        return " ".join(parts)
    parts.append(f'Your request to switch to "{target_path_name}" was rejected.')
    if canned:
        parts.append(f"Reason: {canned}")
    if detail:
        parts.append(f"Dean's note: {detail}")
    parts.append(
        "You stay on your current path, and this does not use up a path change."
    )
    return " ".join(parts)


def path_change_approved_title(*, program_name: str, locale: str | None) -> str:
    """Title: the request was approved and the switch has happened."""
    lang = _norm(locale)
    if lang == "vi":
        return f"Đề nghị đổi lộ trình được chấp thuận: {program_name}"[:255]
    return f"Path change approved: {program_name}"[:255]


def path_change_approved_body(*, target_path_name: str, locale: str | None) -> str:
    lang = _norm(locale)
    if lang == "vi":
        return (
            f'Bạn đã được chuyển sang lộ trình "{target_path_name}". Tiến độ trên lộ '
            "trình cũ đã được lưu lại; các khóa học đã hoàn thành vẫn được tính."
        )
    return (
        f'You have been moved to "{target_path_name}". Your progress on the previous '
        "path was snapshotted, and completed courses still count."
    )


def syllabus_import_failed_body(*, reason: str, locale: str | None) -> str:
    """Body: the failure reason verbatim.

    ``reason`` is the parser's own ``code: sentence`` message. It is passed
    through untranslated on purpose — an approximate translation of a
    diagnostic is worse than the exact one the import screen also shows,
    and the leading code is what a manager would quote when asking for
    help.
    """
    lang = _norm(locale)
    if lang == "vi":
        return f"Không thể tạo khoá học từ tệp đề cương. Lý do: {reason}"
    return f"No course was created from the syllabus file. Reason: {reason}"


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
    "path_change_approved_body",
    "path_change_approved_title",
    "path_change_in_progress_body",
    "path_change_in_progress_title",
    "path_change_rejected_body",
    "path_change_rejected_title",
    "remediation_body",
    "remediation_title",
]
