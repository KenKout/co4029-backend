"""The cross-encoder rerank, and its refusal to be load-bearing.

Voyage rerank-2.5 is a third-party call in the middle of quiz generation.
The module's whole design is that it may fail: any provider error is
logged and the MMR pool comes back unchanged, because a teacher's
generation must not block on someone else's outage. Every test here is
about a way that promise could quietly stop holding.

The sharpest one is the index bounds check. The provider returns
positions into the document list, and ``pool[row.index]`` with a negative
index is *valid Python* -- it silently returns a chunk from the other end
of the list. So a malformed response would not crash; it would hand the
model a different source than the one that was scored, and the questions
generated from it would be grounded in the wrong passage with nothing
anywhere saying so.

The client is a stub; no network.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from abridgeai.ai.llm.errors import ProviderError, ResponseFormatError
from abridgeai.ai.retrieval import ChunkWithDistance
from abridgeai.features.quizzes.ai.stages.retrieval import rerank as rerank_module
from abridgeai.features.quizzes.ai.stages.retrieval.rerank import rerank_pool


def _chunk(label: str) -> ChunkWithDistance:
    return ChunkWithDistance(
        chunk_id=uuid4(),
        material_version_id=uuid4(),
        course_id=uuid4(),
        lesson_id=uuid4(),
        content=label,
        distance=0.1,
    )


def _pool(size: int) -> list[ChunkWithDistance]:
    return [_chunk(f"chunk-{n}") for n in range(size)]


class _Client:
    """Stands in for ``VoyageRerankClient``."""

    def __init__(self, *, results: list[Any] | None = None, raises: Exception | None = None):
        self._results = results or []
        self._raises = raises
        self.calls: list[tuple[str, list[str], int]] = []

    async def rerank(self, anchor: str, documents: list[str], *, top_k: int):
        self.calls.append((anchor, documents, top_k))
        if self._raises:
            raise self._raises
        return self._results, 42


def _result(index: int, score: float = 0.9) -> SimpleNamespace:
    return SimpleNamespace(index=index, relevance_score=score)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        voyage_rerank_model="rerank-2.5",
        voyage_base_url="https://api.voyageai.com",
        voyage_rerank_timeout_seconds=10,
    )


class TestWhenThereIsNothingToDo:
    async def test_an_empty_pool_short_circuits(self) -> None:
        """Retrieval found nothing. Calling the provider with no documents
        spends a request to be told so."""
        client = _Client()

        assert await rerank_pool(
            [], anchor="a", final_top_k=5, client=client, settings=_settings(), voyage_key="k"
        ) == []
        assert client.calls == []

    async def test_no_api_key_means_the_pool_passes_through(self) -> None:
        """Rerank is optional infrastructure. A deployment without a key
        gets MMR ordering rather than an error, and the cap still applies
        so the caller's contract about length holds either way.
        """
        pool = _pool(5)

        result = await rerank_pool(
            pool, anchor="a", final_top_k=3, client=None, settings=_settings(), voyage_key=None
        )

        assert result == pool[:3]

    async def test_an_injected_client_is_used_even_without_a_key(self) -> None:
        """The key is only needed to *construct* a client; one passed in is
        already configured. Without this the test seam would be unusable.
        """
        pool = _pool(3)
        client = _Client(results=[_result(2), _result(0)])

        result = await rerank_pool(
            pool, anchor="a", final_top_k=2, client=client, settings=_settings(), voyage_key=None
        )

        assert result == [pool[2], pool[0]]


class TestTheProviderFailing:
    @pytest.mark.parametrize(
        "failure",
        [ProviderError("upstream 503"), ResponseFormatError("not json")],
    )
    async def test_a_provider_error_returns_the_mmr_ordering(
        self, failure: Exception
    ) -> None:
        """The promise the module exists to keep.

        A reranker outage costs the generation some ordering quality. It
        must not cost the teacher their quiz.
        """
        pool = _pool(5)
        client = _Client(raises=failure)

        result = await rerank_pool(
            pool, anchor="a", final_top_k=3, client=client, settings=_settings(), voyage_key="k"
        )

        assert result == pool[:3]

    async def test_an_empty_response_returns_the_mmr_ordering(self) -> None:
        """A 200 with no results is not an error, and it is not an answer
        either -- keeping it would return nothing at all."""
        pool = _pool(4)
        client = _Client(results=[])

        result = await rerank_pool(
            pool, anchor="a", final_top_k=2, client=client, settings=_settings(), voyage_key="k"
        )

        assert result == pool[:2]

    async def test_an_unexpected_error_is_not_swallowed(self) -> None:
        """Only the two provider errors are caught.

        A ``TypeError`` from our own code is a bug in this pipeline, and
        burying it under the fallback would make every future mistake here
        look like a Voyage outage.
        """
        client = _Client(raises=TypeError("our bug, not theirs"))

        with pytest.raises(TypeError):
            await rerank_pool(
                _pool(3),
                anchor="a",
                final_top_k=2,
                client=client,
                settings=_settings(),
                voyage_key="k",
            )


class TestReorderingThePool:
    async def test_the_pool_is_reordered_by_the_returned_positions(self) -> None:
        pool = _pool(4)
        client = _Client(results=[_result(3), _result(1), _result(0)])

        result = await rerank_pool(
            pool, anchor="a", final_top_k=3, client=client, settings=_settings(), voyage_key="k"
        )

        assert result == [pool[3], pool[1], pool[0]]

    async def test_the_anchor_and_documents_are_what_gets_scored(self) -> None:
        """The cross-encoder scores ``(anchor, document)`` pairs, so the
        documents must be the chunk texts in pool order -- the indices come
        back as positions into exactly that list.
        """
        pool = _pool(3)
        client = _Client(results=[_result(0)])

        await rerank_pool(
            pool,
            anchor="what is photosynthesis",
            final_top_k=1,
            client=client,
            settings=_settings(),
            voyage_key="k",
        )

        anchor, documents, top_k = client.calls[0]
        assert anchor == "what is photosynthesis"
        assert documents == [c.content for c in pool]
        assert top_k == 1

    async def test_more_results_than_asked_for_are_truncated(self) -> None:
        pool = _pool(5)
        client = _Client(results=[_result(i) for i in (4, 3, 2, 1, 0)])

        result = await rerank_pool(
            pool, anchor="a", final_top_k=2, client=client, settings=_settings(), voyage_key="k"
        )

        assert result == [pool[4], pool[3]]


class TestAMalformedResponseCannotSubstituteADocument:
    """The bounds and dedup checks, which have no visible failure mode.

    Each of these would produce a plausible-looking result set containing
    the wrong chunks, and the questions generated from them would be
    grounded in a passage the reranker never scored.
    """

    async def test_a_negative_index_is_discarded_rather_than_wrapping(self) -> None:
        """``pool[-1]`` is legal Python and returns the last chunk.

        Without the ``< 0`` guard a malformed response does not raise -- it
        silently swaps in a different document, which is the worst of the
        available outcomes.
        """
        pool = _pool(4)
        client = _Client(results=[_result(-1), _result(0)])

        result = await rerank_pool(
            pool, anchor="a", final_top_k=2, client=client, settings=_settings(), voyage_key="k"
        )

        assert pool[-1] not in result, "the last chunk was not what index -1 meant"
        assert result[0] == pool[0], "the one valid position still leads"
        assert result == [pool[0], pool[1]], "the discarded slot is backfilled in MMR order"

    async def test_an_index_past_the_end_does_not_raise(self) -> None:
        """This one *would* crash without the guard, taking the generation
        with it."""
        pool = _pool(3)
        client = _Client(results=[_result(99), _result(1)])

        result = await rerank_pool(
            pool, anchor="a", final_top_k=2, client=client, settings=_settings(), voyage_key="k"
        )

        assert result[0] == pool[1]
        assert len(result) == 2

    async def test_a_repeated_index_is_only_taken_once(self) -> None:
        """Kept twice, the same passage occupies two of the slots the model
        is given and the effective source coverage silently halves."""
        pool = _pool(4)
        client = _Client(results=[_result(2), _result(2), _result(0)])

        result = await rerank_pool(
            pool, anchor="a", final_top_k=3, client=client, settings=_settings(), voyage_key="k"
        )

        assert len(result) == len(set(id(c) for c in result))
        assert result[0] == pool[2]
        assert pool[0] in result


class TestToppingUpFromTheMmrOrdering:
    async def test_a_short_result_set_is_backfilled(self) -> None:
        """Voyage returning fewer positions than asked for should cost
        ordering, not chunks: the generation was promised ``final_top_k``
        sources and the MMR pool can still supply them.
        """
        pool = _pool(5)
        client = _Client(results=[_result(4)])

        result = await rerank_pool(
            pool, anchor="a", final_top_k=3, client=client, settings=_settings(), voyage_key="k"
        )

        assert len(result) == 3
        assert result[0] == pool[4], "the ranked one still leads"
        assert result[1:] == [pool[0], pool[1]], "the rest follow MMR order"

    async def test_backfill_never_repeats_a_ranked_chunk(self) -> None:
        pool = _pool(4)
        client = _Client(results=[_result(1)])

        result = await rerank_pool(
            pool, anchor="a", final_top_k=3, client=client, settings=_settings(), voyage_key="k"
        )

        assert len(result) == len({id(c) for c in result})
        assert result[0] == pool[1]

    async def test_a_pool_smaller_than_the_cap_is_returned_whole(self) -> None:
        """Backfill cannot invent chunks; it stops when the pool runs out."""
        pool = _pool(2)
        client = _Client(results=[_result(1)])

        result = await rerank_pool(
            pool, anchor="a", final_top_k=5, client=client, settings=_settings(), voyage_key="k"
        )

        assert len(result) == 2
        assert {id(c) for c in result} == {id(c) for c in pool}

    async def test_a_discarded_bad_index_is_made_up_from_the_pool(self) -> None:
        """The two defences compose: the wrong chunk is dropped and a real
        one takes its place, so the caller still gets a full set."""
        pool = _pool(4)
        client = _Client(results=[_result(99), _result(2)])

        result = await rerank_pool(
            pool, anchor="a", final_top_k=3, client=client, settings=_settings(), voyage_key="k"
        )

        assert len(result) == 3
        assert result[0] == pool[2]


async def test_a_client_is_built_from_settings_when_only_a_key_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestrator decides rerank on/off by passing the key; building
    the client here is what keeps ``SecretStr`` out of the caller.
    """
    captured: dict[str, Any] = {}

    class _Constructed(_Client):
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            super().__init__(results=[_result(0)])

    monkeypatch.setattr(rerank_module, "VoyageRerankClient", _Constructed)
    settings = _settings()

    await rerank_pool(
        _pool(2),
        anchor="a",
        final_top_k=1,
        client=None,
        settings=settings,
        voyage_key="secret-key",
    )

    assert captured["api_key"] == "secret-key"
    assert captured["model"] == settings.voyage_rerank_model
    assert captured["base_url"] == settings.voyage_base_url
    assert captured["timeout_s"] == settings.voyage_rerank_timeout_seconds
