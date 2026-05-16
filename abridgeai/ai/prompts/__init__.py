"""Jinja2 prompt template infrastructure for AI stages."""

from abridgeai.ai.prompts.env import get_jinja_env
from abridgeai.ai.prompts.loader import render_prompt

__all__ = ["get_jinja_env", "render_prompt"]
