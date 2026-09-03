"""Phase 3: rich-content sanitizer unit tests."""

from __future__ import annotations

from abridgeai.core.sanitize import sanitize_rich_content


def test_plain_is_returned_unchanged():
    assert sanitize_rich_content("2 < 3 & 4 > 1", fmt="plain") == "2 < 3 & 4 > 1"


def test_none_passes_through():
    assert sanitize_rich_content(None, fmt="html") is None


def test_html_strips_script_and_onerror():
    dirty = "<p>ok</p><script>alert(1)</script><img src=x onerror=alert(1)>"
    clean = sanitize_rich_content(dirty, fmt="html")
    assert "<script>" not in clean
    assert "onerror" not in clean
    assert "<p>ok</p>" in clean


def test_html_keeps_safe_img():
    dirty = '<img src="https://cdn.example/x.png" alt="fig">'
    clean = sanitize_rich_content(dirty, fmt="html")
    assert 'src="https://cdn.example/x.png"' in clean
    assert 'alt="fig"' in clean


def test_markdown_sanitizes_html_but_keeps_syntax_and_math():
    dirty = "# Title\n\n<script>x</script>\n\n$E = mc^2$"
    clean = sanitize_rich_content(dirty, fmt="markdown")
    assert "<script>" not in clean
    assert "$E = mc^2$" in clean
    assert "# Title" in clean
