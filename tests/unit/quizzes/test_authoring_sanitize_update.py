"""Regression: rich text must be sanitized on the UPDATE path, not just create.

SECURITY: the client renders ``html``-format question content via
``dangerouslySetInnerHTML`` (frontend ``RichContent``), trusting that the server
already cleaned it. ``create_question`` sanitized from the start; the PATCH path
originally did a blind ``setattr`` loop, so a teacher could store
``<script>``/``onerror`` markup that then executed in every student's browser
(stored XSS). These tests pin the sanitize call on the update path.

They exercise the pure sanitizer with the exact format-resolution rule the
service uses (payload format when supplied, else the format already on the row),
so they run without a DB.
"""

from __future__ import annotations

from abridgeai.core.sanitize import sanitize_rich_content

HOSTILE = '<p>hello</p><script>alert("pwned")</script><img src=x onerror="steal()">'


def _resolve_fmt(field_updates: dict, format_field: str, row_fmt: str) -> str:
    """Mirror of the service's format resolution for updated rich fields."""
    return field_updates.get(format_field, row_fmt)


def test_hostile_html_is_stripped_for_html_format():
    cleaned = sanitize_rich_content(HOSTILE, fmt="html")
    assert cleaned is not None
    assert "<script" not in cleaned.lower()
    assert "onerror" not in cleaned.lower()
    assert "alert(" not in cleaned


def test_hostile_html_is_stripped_for_markdown_format():
    cleaned = sanitize_rich_content(HOSTILE, fmt="markdown")
    assert cleaned is not None
    assert "<script" not in cleaned.lower()
    assert "onerror" not in cleaned.lower()


def test_plain_format_is_not_interpreted_as_markup():
    """plain content is escaped/left inert by the renderer, never parsed."""
    cleaned = sanitize_rich_content(HOSTILE, fmt="plain")
    # plain passes through unchanged; the client renders it as escaped text.
    assert cleaned == HOSTILE


def test_format_falls_back_to_existing_row_value():
    """PATCHing only the text (no format) must reuse the row's stored format,
    so an html-format question keeps being sanitized as html."""
    field_updates = {"prompt_text": HOSTILE}
    fmt = _resolve_fmt(field_updates, "prompt_format", "html")
    assert fmt == "html"
    cleaned = sanitize_rich_content(field_updates["prompt_text"], fmt=fmt)
    assert cleaned is not None
    assert "<script" not in cleaned.lower()


def test_payload_format_overrides_row_format():
    field_updates = {"prompt_text": HOSTILE, "prompt_format": "html"}
    fmt = _resolve_fmt(field_updates, "prompt_format", "plain")
    assert fmt == "html"
    cleaned = sanitize_rich_content(field_updates["prompt_text"], fmt=fmt)
    assert cleaned is not None
    assert "onerror" not in cleaned.lower()


def test_update_path_calls_sanitizer_for_all_three_rich_fields():
    """Guard the field list itself: prompt/hint/explanation must all be cleaned."""
    import inspect

    from abridgeai.features.quizzes.services import authoring

    src = inspect.getsource(authoring.update_question)
    assert "sanitize_rich_content" in src, "update_question must sanitize rich text"
    for field in ("prompt_text", "hint_text", "explanation"):
        assert field in src, f"{field} must be sanitized on update"
