"""LLM-as-judge package.

Loads judge prompts from `prompts/` (Jinja2 templates) and applies them to
scenario outputs to produce numeric scores per criterion.

T8.1 ships scaffold only. T8.3 will ship the first prompt set
(faithfulness, coverage, difficulty alignment, ...).
"""
