"""Caller-relative prompt template loader.

A call from ``features/quizzes/ai/stages/ideation/logic.py``::

    render_prompt("prompts/system.j2", outline=..., chunks=...)

resolves to ``features/quizzes/ai/stages/ideation/prompts/system.j2``. This
keeps each stage's prompts co-located with the code that uses them, so
non-engineers can review or A/B-test prompts by editing ``.j2`` files without
touching Python.

Tradeoff
--------
A fresh ``jinja2.Environment`` is built per call instead of caching one global
env, because each caller has a different search path. The cost is a few
millis of object construction per render — negligible against LLM latency.
The benefit is that the API is impossible to misuse: relative paths always
resolve relative to the caller, never to a surprising cwd.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import jinja2


def render_prompt(template_path: str, **kwargs: object) -> str:
    """Render a Jinja2 template located relative to the caller's package.

    Parameters
    ----------
    template_path:
        Path relative to the calling module's directory. May contain
        subdirectories (e.g. ``"prompts/system.j2"``).
    **kwargs:
        Template variables. Any variable referenced in the template but
        missing here raises :class:`jinja2.UndefinedError` (StrictUndefined).

    Returns
    -------
    str
        The rendered template.

    Raises
    ------
    jinja2.TemplateNotFound
        If the template file does not exist relative to the caller.
    jinja2.UndefinedError
        If the template references a variable not provided in kwargs.
    """
    caller_frame = inspect.stack()[1]
    caller_dir = Path(caller_frame.filename).resolve().parent
    full_path = (caller_dir / template_path).resolve()

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(searchpath=[str(full_path.parent)]),
        autoescape=False,  # noqa: S701  -- prompts render LLM plain text, not HTML
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )
    template = env.get_template(full_path.name)
    return template.render(**kwargs)
