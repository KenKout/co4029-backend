from pathlib import Path

import jinja2
import pytest

from abridgeai.ai.prompts import render_prompt

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "prompts"


def test_render_succeeds():
    rendered = render_prompt(
        str(FIXTURES_DIR / "test_template.j2"),
        name="Alice",
    )
    assert rendered.strip() == "Hello Alice"


def test_missing_var_strict():
    with pytest.raises(jinja2.UndefinedError):
        render_prompt(str(FIXTURES_DIR / "missing_var.j2"))
