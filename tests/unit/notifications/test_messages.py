"""Unit tests for localized notification copy (EN/VI).

The ``messages`` module is the single owner of notification ``title`` /
``body`` text. These pure-function tests lock the bilingual contract so a
regression can't silently drop a locale back to English. No DB, no I/O.
"""

from __future__ import annotations

from abridgeai.features.notifications import messages


class TestDueCards:
    def test_title_en_plural(self) -> None:
        assert messages.due_cards_title(due_count=3, locale="en") == "You have 3 cards due"

    def test_title_en_singular(self) -> None:
        title = messages.due_cards_title(due_count=1, locale="en")
        assert title == "You have 1 card due"
        assert "cards" not in title

    def test_title_vi(self) -> None:
        # Vietnamese: no plural inflection; singular and multi share wording.
        assert messages.due_cards_title(due_count=3, locale="vi") == "Bạn có 3 thẻ cần ôn tập"
        assert messages.due_cards_title(due_count=1, locale="vi") == "Bạn có 1 thẻ cần ôn tập"

    def test_title_unknown_locale_falls_back_to_en(self) -> None:
        assert messages.due_cards_title(due_count=2, locale="fr") == "You have 2 cards due"

    def test_title_none_locale_falls_back_to_en(self) -> None:
        assert messages.due_cards_title(due_count=2, locale=None) == "You have 2 cards due"

    def test_body_en(self) -> None:
        assert messages.due_cards_body(locale="en") == (
            "Review them now to keep your knowledge fresh."
        )

    def test_body_vi(self) -> None:
        assert messages.due_cards_body(locale="vi") == (
            "Hãy ôn tập ngay để ghi nhớ kiến thức lâu hơn."
        )


class TestRemediation:
    def test_title_en_with_concepts(self) -> None:
        title = messages.remediation_title(
            missed_concepts=["Star schema", "Foreign keys"],
            primary_resource=None,
            locale="en",
        )
        assert title == "Review needed: Star schema, Foreign keys"

    def test_title_en_extra_concepts_suffix(self) -> None:
        title = messages.remediation_title(
            missed_concepts=["A", "B", "C", "D"],
            primary_resource=None,
            locale="en",
        )
        assert title == "Review needed: A, B (+2 more)"

    def test_title_vi_with_concepts(self) -> None:
        title = messages.remediation_title(
            missed_concepts=["Star schema", "Foreign keys"],
            primary_resource=None,
            locale="vi",
        )
        assert title == "Cần ôn tập: Star schema, Foreign keys"

    def test_title_vi_extra_concepts_suffix(self) -> None:
        title = messages.remediation_title(
            missed_concepts=["A", "B", "C", "D"],
            primary_resource=None,
            locale="vi",
        )
        assert title == "Cần ôn tập: A, B (+2 khái niệm khác)"

    def test_title_fallback_to_primary_resource(self) -> None:
        assert (
            messages.remediation_title(
                missed_concepts=[], primary_resource="Chapter 4", locale="en"
            )
            == "Review needed: Chapter 4"
        )
        assert (
            messages.remediation_title(
                missed_concepts=[], primary_resource="Chapter 4", locale="vi"
            )
            == "Cần ôn tập: Chapter 4"
        )

    def test_title_fallback_when_no_concepts_no_resource(self) -> None:
        assert (
            messages.remediation_title(missed_concepts=[], primary_resource=None, locale="en")
            == "Review needed: this card"
        )
        assert (
            messages.remediation_title(missed_concepts=[], primary_resource=None, locale="vi")
            == "Cần ôn tập: thẻ này"
        )

    def test_title_truncated_to_255(self) -> None:
        title = messages.remediation_title(
            missed_concepts=["x" * 400], primary_resource=None, locale="en"
        )
        assert len(title) == 255

    def test_body_en_lead_and_links(self) -> None:
        body = messages.remediation_body(
            resource_links=[("Lesson 1", "/c/slug/l/1"), ("Video", "/c/slug/l/2?t=30")],
            locale="en",
        )
        assert body.startswith("You missed this question.")
        assert "- [Lesson 1](/c/slug/l/1)" in body
        assert "- [Video](/c/slug/l/2?t=30)" in body

    def test_body_vi_lead_keeps_links_verbatim(self) -> None:
        body = messages.remediation_body(
            resource_links=[("Lesson 1", "/c/slug/l/1")],
            locale="vi",
        )
        assert body.startswith("Bạn đã trả lời sai câu hỏi này.")
        # Link markdown is locale-independent (labels are material titles).
        assert "- [Lesson 1](/c/slug/l/1)" in body
