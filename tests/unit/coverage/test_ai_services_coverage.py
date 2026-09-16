from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from abridgeai.ai.llm import embeddings
from abridgeai.ai.llm.embeddings import EmbeddingClient
from abridgeai.ai.llm.errors import ConfigError, ProviderError
from abridgeai.ai.preprocessing.base import ROLE_BODY, PageUnit, PreprocessReport
from abridgeai.ai.preprocessing.pipeline import (
    PreprocessConfig,
    _apply_teacher_restores,
    _is_ambiguous,
    _normalize_units,
    _page_role_map,
    _run_adjudication,
    _run_ocr,
)
from abridgeai.core.exceptions import AppError


def _settings(dimensions: int = 3) -> SimpleNamespace:
    return SimpleNamespace(embedding_dimensions=dimensions)


def _unit(page: int | None, body: str, *, marker: str | None = None) -> PageUnit:
    return PageUnit(
        marker=marker if marker is not None else (f"[Page {page}]" if page else ""),
        page_number=page,
        body=body,
    )


@pytest.mark.asyncio
async def test_embedding_dimension_validation() -> None:
    missing_db = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(first=lambda: None))
    )
    with pytest.raises(ConfigError, match="column not found"):
        await EmbeddingClient(_settings()).validate_dimensions(missing_db)

    mismatch_db = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(first=lambda: (5,)))
    )
    with pytest.raises(ConfigError, match="EMBEDDING_DIMENSIONS=3"):
        await EmbeddingClient(_settings()).validate_dimensions(mismatch_db)

    valid_db = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(first=lambda: (3,)))
    )
    await EmbeddingClient(_settings()).validate_dimensions(valid_db)


@pytest.mark.asyncio
async def test_embedding_success_sorts_provider_items_and_audits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = SimpleNamespace(model="embed-model", base_url="https://provider")
    provider = SimpleNamespace(
        embeddings=AsyncMock(
            return_value=(
                {
                    "data": [
                        {"index": 1, "embedding": [2.0]},
                        {"index": 0, "embedding": [1.0]},
                    ],
                    "usage": {"prompt_tokens": 12},
                },
                25,
            )
        )
    )
    audit = AsyncMock()
    monkeypatch.setattr(embeddings, "resolve_binding_overrides", AsyncMock(return_value=(4.0, 2)))
    monkeypatch.setattr(embeddings, "binding_for", lambda *_args, **_kwargs: binding)
    monkeypatch.setattr(embeddings, "OpenAICompatibleClient", lambda _binding: provider)
    monkeypatch.setattr(embeddings, "compute_cost", AsyncMock(return_value=0.001))
    monkeypatch.setattr(embeddings, "write_ai_model_call", audit)

    result = await EmbeddingClient(_settings()).embed(
        ["one", "two"], db=object(), organization_id=None
    )

    assert result == [[1.0], [2.0]]
    assert audit.await_args.kwargs["status"] == "success"
    assert audit.await_args.kwargs["input_tokens"] == 12


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_result", "message"),
    [
        (ProviderError("rate limited"), "rate limited"),
        (({"usage": {"prompt_tokens": 2}}, 9), "missing 'data' key"),
    ],
)
async def test_embedding_failures_are_audited_and_wrapped(
    monkeypatch: pytest.MonkeyPatch,
    provider_result: object,
    message: str,
) -> None:
    binding = SimpleNamespace(model="embed-model", base_url="https://provider")
    call = AsyncMock(
        side_effect=provider_result if isinstance(provider_result, Exception) else None,
        return_value=None if isinstance(provider_result, Exception) else provider_result,
    )
    monkeypatch.setattr(embeddings, "resolve_binding_overrides", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(embeddings, "binding_for", lambda *_args, **_kwargs: binding)
    monkeypatch.setattr(
        embeddings,
        "OpenAICompatibleClient",
        lambda _binding: SimpleNamespace(embeddings=call),
    )
    monkeypatch.setattr(embeddings, "compute_cost", AsyncMock(return_value=0.0))
    audit = AsyncMock()
    monkeypatch.setattr(embeddings, "write_ai_model_call", audit)

    with pytest.raises(AppError, match=message):
        await EmbeddingClient(_settings()).embed(["text"], db=object())

    assert audit.await_args.kwargs["status"] == "failed"


@pytest.mark.asyncio
async def test_embed_query_returns_single_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    client = EmbeddingClient(_settings())
    monkeypatch.setattr(client, "embed", AsyncMock(return_value=[[0.1, 0.2]]))
    assert await client.embed_query("question", db=object()) == [0.1, 0.2]


def test_normalize_units_tracks_dehyphenation() -> None:
    units = [_unit(1, "The decision sup-\nport system")]
    report = PreprocessReport(enabled=True)
    _normalize_units(
        units,
        "The decision sup-\nport system",
        PreprocessConfig(dehyphenation=True),
        report,
    )
    assert units[0].body == "The decision support system"
    assert report.hyphen_joins == 1


@pytest.mark.asyncio
async def test_ocr_recovers_text_and_leaves_empty_or_failed_pages() -> None:
    recovered = _unit(1, "")
    empty = _unit(2, "")
    failed = _unit(3, "")
    for unit in (recovered, empty, failed):
        unit.needs_ocr = True

    class _Ocr:
        async def ocr_page(self, page: int) -> str | None:
            if page == 1:
                return " recovered text "
            if page == 2:
                return " "
            raise RuntimeError("vision unavailable")

    report = PreprocessReport(enabled=True)
    await _run_ocr([recovered, empty, failed], _Ocr(), report, max_pages=1)

    assert recovered.body == "recovered text"
    assert recovered.needs_ocr is False
    assert "ocr_recovered" in recovered.noise_flags
    assert report.pages_ocr_routed == 1
    assert empty.needs_ocr is True
    assert failed.needs_ocr is True


@pytest.mark.asyncio
async def test_adjudication_accepts_only_valid_confident_verdicts() -> None:
    head = _unit(1, "word " * 20)
    tail = _unit(8, "word " * 20)
    middle = _unit(5, "word " * 20)
    units = [head, middle, tail]

    class _Adjudicator:
        async def classify(
            self, _pages: list[tuple[int, str]]
        ) -> dict[int, tuple[str, float]]:
            return {
                1: ("front_matter", 0.95),
                8: ("summary", 0.5),
                999: ("reference", 1.0),
            }

    report = PreprocessReport(enabled=True)
    await _run_adjudication(units, _Adjudicator(), report, min_confidence=0.8)

    assert head.role == "front_matter"
    assert tail.role == ROLE_BODY
    assert middle.role == ROLE_BODY
    assert report.llm_adjudicated == 1


@pytest.mark.asyncio
async def test_adjudication_failure_is_fail_open() -> None:
    unit = _unit(1, "word " * 20)
    adjudicator = SimpleNamespace(classify=AsyncMock(side_effect=RuntimeError("offline")))
    report = PreprocessReport(enabled=True)
    await _run_adjudication([unit], adjudicator, report, min_confidence=0.8)
    assert unit.role == ROLE_BODY
    assert report.llm_adjudicated == 0


def test_ambiguity_restore_and_page_role_projection() -> None:
    restored = _unit(1, "word " * 20)
    restored.dropped = True
    restored.retrieval_excluded = True
    restored.role = "front_matter"
    restored.topic_group_id = "topic"
    restored.slide_title = "Title"
    skipped = _unit(None, "word " * 20)
    report = PreprocessReport(enabled=True)

    assert not _is_ambiguous(restored, total_pages=5)
    restored.dropped = False
    restored.role = ROLE_BODY
    assert _is_ambiguous(restored, total_pages=5)

    restored.role = "front_matter"
    _apply_teacher_restores([restored, skipped], {1}, report)
    role_map = _page_role_map([restored, skipped])

    assert restored.role == ROLE_BODY
    assert restored.retrieval_excluded is False
    assert report.role_counts["teacher_restored"] == 1
    assert role_map["1"]["topic_group_id"] == "topic"
    assert role_map["1"]["slide_title"] == "Title"
