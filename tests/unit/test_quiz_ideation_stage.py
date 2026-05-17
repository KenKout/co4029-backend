"""Unit tests for the quiz ideation stage (T5.5).

Covers acceptance items in plan §5634-5638:
* ``ideate_for_outline`` returns parsed Template objects from a mocked LLM.
* Audit metadata is threaded — gateway is called with ``stage_name="ideation"``
  + ``role=LLMRole.IDEATION`` + ``parent_run_id`` + ``pipeline_run_id``.
* ``parse_ideation_response`` raises ``ResponseFormatError`` on malformed JSON.
* ``_redistribute_chunk_anchors_within_section`` produces a balanced
  distribution within an outline section.
* No file in ``stages/ideation/`` exceeds 250 LOC.

We use ``SimpleNamespace`` stand-ins for ``LessonOutline`` / ``OutlineSection``
since those dataclasses still live in legacy ``backend/`` (T3.7 will port
them). The stage only reads attributes (``id``, ``depth``, ``title``,
``page_range``, ``content_role``, ``chunk_ids``, ``preview``) so a
namespace works.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.ai.llm.errors import ResponseFormatError
from abridgeai.ai.llm.roles import LLMRole
from abridgeai.features.quizzes.ai.stages.ideation import (
    Template,
    _redistribute_chunk_anchors_within_section,
    ideate_for_outline,
    parse_ideation_response,
)


def _section(
    *,
    section_id: str = "sec_a",
    chunk_ids: list[str] | None = None,
    title: str = "Intro",
    depth: int = 1,
    pages: tuple[int, int] = (1, 2),
    role: str = "body",
    preview: str = "preview text",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=section_id,
        title=title,
        depth=depth,
        chunk_ids=chunk_ids or ["c-1", "c-2", "c-3"],
        page_range=pages,
        content_role=role,
        preview=preview,
    )


def _outline(sections: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(lesson_id=uuid4(), title="Lesson", sections=sections)


def _llm_result(content_json: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(content_json=content_json)


def _make_gateway(content_json: dict[str, Any]) -> AsyncMock:
    gateway = AsyncMock()
    gateway.generate_json = AsyncMock(return_value=_llm_result(content_json))
    return gateway


@pytest.mark.asyncio
async def test_ideate_returns_parsed_templates() -> None:
    db = AsyncMock()
    run = SimpleNamespace(id=uuid4())
    outline = _outline([_section(section_id="sec_a", chunk_ids=["c-1", "c-2"])])
    budget = {"sec_a": 2}

    payload = {
        "templates": [
            {
                "position": 1,
                "section_id": "sec_a",
                "topic": "Definition of recursion",
                "question_type": "mcq",
                "bloom_level": "understand",
                "difficulty": "medium",
                "source_chunk_ids": ["c-1"],
                "rationale": "tests definition recall",
            },
            {
                "position": 2,
                "section_id": "sec_a",
                "topic": "Recursive call stack",
                "question_type": "mcq",
                "bloom_level": "apply",
                "difficulty": "medium",
                "source_chunk_ids": ["c-2"],
                "rationale": "tests application",
            },
        ]
    }
    gateway = _make_gateway(payload)

    templates = await ideate_for_outline(
        db,
        run,
        title="Sample Quiz",
        config={"difficulty": "medium"},
        outlines=[outline],
        budget=budget,
        pipeline_run_id=uuid4(),
        gateway=gateway,
    )

    assert len(templates) == 2
    assert all(isinstance(t, Template) for t in templates)
    assert [t.section_id for t in templates] == ["sec_a", "sec_a"]
    assert [t.source_chunk_ids[0] for t in templates] == ["c-1", "c-2"]


@pytest.mark.asyncio
async def test_ideate_passes_audit_metadata() -> None:
    db = AsyncMock()
    run_id = uuid4()
    pipeline_run_id = uuid4()
    run = SimpleNamespace(id=run_id)
    outline = _outline([_section()])
    budget = {"sec_a": 1}

    payload = {
        "templates": [
            {
                "section_id": "sec_a",
                "source_chunk_ids": ["c-1"],
            }
        ]
    }
    gateway = _make_gateway(payload)

    await ideate_for_outline(
        db,
        run,
        title="Audit Quiz",
        config={},
        outlines=[outline],
        budget=budget,
        pipeline_run_id=pipeline_run_id,
        gateway=gateway,
    )

    gateway.generate_json.assert_awaited_once()
    call_kwargs = gateway.generate_json.await_args.kwargs
    assert call_kwargs["role"] is LLMRole.IDEATION
    assert call_kwargs["stage_name"] == "ideation"
    assert call_kwargs["pipeline_run_id"] == pipeline_run_id
    assert call_kwargs["parent_run_id"] == run_id
    assert call_kwargs["db"] is db
    assert "Audit Quiz" in call_kwargs["user_prompt"]
    assert "sec_a" in call_kwargs["user_prompt"]


def test_parser_validates_required_fields() -> None:
    with pytest.raises(ResponseFormatError, match="missing required 'templates'"):
        parse_ideation_response({})

    with pytest.raises(ResponseFormatError, match="must be a JSON object"):
        parse_ideation_response([])  # type: ignore[arg-type]

    with pytest.raises(ResponseFormatError, match="must be a list"):
        parse_ideation_response({"templates": "not-a-list"})

    with pytest.raises(ResponseFormatError, match="failed validation"):
        parse_ideation_response({"templates": [{"topic": "missing section_id"}]})


def test_parser_skips_non_dict_entries_and_coerces_position() -> None:
    parsed = parse_ideation_response(
        {
            "templates": [
                "not a dict",
                {
                    "position": "3",
                    "section_id": "sec_a",
                    "source_chunk_ids": ["c-1"],
                },
                None,
            ]
        }
    )
    assert len(parsed) == 1
    assert parsed[0].position == 3
    assert parsed[0].source_chunk_ids == ["c-1"]


def test_redistribute_chunk_anchors_within_section_balanced() -> None:
    section = _section(section_id="sec_x", chunk_ids=["c-1", "c-2", "c-3"])
    sections_by_id = {"sec_x": section}

    templates = [
        Template(section_id="sec_x", source_chunk_ids=["c-1"]),
        Template(section_id="sec_x", source_chunk_ids=["c-1"]),
        Template(section_id="sec_x", source_chunk_ids=["c-1"]),
    ]
    redistributed = _redistribute_chunk_anchors_within_section(templates, sections_by_id)

    primaries = [t.source_chunk_ids[0] for t in redistributed]
    assert primaries == ["c-1", "c-2", "c-3"], (
        "collisions on c-1 should be reassigned to the section's unused chunks"
    )
    assert len(set(primaries)) == 3


def test_redistribute_falls_back_when_section_chunks_exhausted() -> None:
    section = _section(section_id="sec_y", chunk_ids=["c-1"])
    sections_by_id = {"sec_y": section}

    templates = [
        Template(section_id="sec_y", source_chunk_ids=["c-1"]),
        Template(section_id="sec_y", source_chunk_ids=["c-1"]),
    ]
    redistributed = _redistribute_chunk_anchors_within_section(templates, sections_by_id)

    assert [t.source_chunk_ids[0] for t in redistributed] == ["c-1", "c-1"]


@pytest.mark.asyncio
async def test_ideate_filters_templates_outside_outline() -> None:
    db = AsyncMock()
    run = SimpleNamespace(id=uuid4())
    outline = _outline([_section(section_id="sec_a", chunk_ids=["c-1"])])
    budget = {"sec_a": 1}

    payload = {
        "templates": [
            {"section_id": "ghost_section", "source_chunk_ids": ["c-99"]},
            {"section_id": "sec_a", "source_chunk_ids": ["c-99"]},
        ]
    }
    gateway = _make_gateway(payload)

    templates = await ideate_for_outline(
        db,
        run,
        title="Filter Quiz",
        config={},
        outlines=[outline],
        budget=budget,
        gateway=gateway,
    )

    assert len(templates) == 1
    assert templates[0].section_id == "sec_a"
    assert templates[0].source_chunk_ids == ["c-1"]


def test_no_god_file_in_ideation_stage() -> None:
    here = Path(__file__).resolve().parents[2]
    target = here / "abridgeai" / "features" / "quizzes" / "ai" / "stages" / "ideation"
    assert target.is_dir(), f"ideation stage dir not found at {target}"
    for path in target.rglob("*.py"):
        line_count = sum(1 for _ in path.open())
        assert line_count <= 250, f"{path.name} has {line_count} LOC > 250"
