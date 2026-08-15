"""Stage C — LLM enrichment for the SemanticChunker.

Cached on ``(content_hash, prompt_version)`` per Reconciliation §B9 / §C12;
threads ``pipeline_run_id`` and ``stage_name="chunking_enrichment"`` so
T2.4's audit columns get populated.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

from abridgeai.ai.chunking.base import EnrichedChunk, RawChunk
from abridgeai.ai.chunking.token_aware import truncate_to_tokens

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.ai.chunking.cache import ChunkingCache
    from abridgeai.ai.llm.gateway import LLMResult

    # A zero-arg callable returning an async-context-manager session — i.e.
    # ``async_sessionmaker`` (``get_sessionmaker()``). Each call yields a fresh
    # ``AsyncSession`` so concurrent enrichment coroutines never share one.
    SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v1"
_LLM_INPUT_TOKEN_BUDGET = 3000
# A large semaphore used as an effective no-op when the caller already gates
# concurrency at a higher level (``_enrich_one_isolated`` acquires the real
# bounded semaphore around the whole session, then hands this to the inner
# ``_enrich_one`` so the LLM call is not double-gated).
_UNBOUNDED_SEMAPHORE = asyncio.Semaphore(1_000_000)
_VALID_ROLES = frozenset({"body", "summary", "review", "front_matter"})

_ENRICHMENT_SYSTEM_PROMPT = (
    "You are a curriculum analyst preparing chunked educational content for a "
    "retrieval-augmented quiz pipeline.\n\nReturn JSON with fields: "
    "section_title (<=60 chars, real topic not page number), "
    "content_role (body|summary|review|front_matter), "
    "context_sentence (1-2 sentences situating the window in the wider document), "
    "key_concepts (<=6 lowercase noun phrases), "
    "propositions (<=8 atomic self-contained facts). "
    "Never invent content not in the window."
)


class LLMGatewayProto(Protocol):
    async def generate_json(
        self,
        *,
        role: object,
        system_prompt: str,
        user_prompt: str,
        db: AsyncSession,
        stage_name: str | None = ...,
        pipeline_run_id: UUID | None = ...,
        parent_job_id: UUID | None = ...,
        parent_run_id: UUID | None = ...,
    ) -> LLMResult: ...


@dataclass(frozen=True)
class WindowEnrichment:
    section_title: str
    content_role: str
    context_sentence: str = ""
    key_concepts: list[str] = field(default_factory=list)
    propositions: list[str] = field(default_factory=list)
    cached: bool = False
    model_name: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def enrich_with_llm(
    windows: list[RawChunk],
    *,
    llm_gateway: LLMGatewayProto,
    db: AsyncSession,
    cache: ChunkingCache | None = None,
    document_title: str | None = None,
    pipeline_run_id: UUID | None = None,
    parent_job_id: UUID | None = None,
    parallelism: int = 4,
    session_factory: SessionFactory | None = None,
) -> list[EnrichedChunk]:
    """Enrich each window with LLM-derived semantic metadata.

    Concurrency model (the crux of the multi-window PDF fix):

    Every ``_enrich_one`` touches a DB session — ``cache.get`` / ``cache.put``
    and the gateway's ``write_ai_model_call`` audit-row flush. A single
    ``AsyncSession`` CANNOT be driven by two coroutines at once: interleaved
    flushes raise ``InvalidRequestError: Session is already flushing``, which
    used to break every multi-window document (PDF/slides) — it stuck in
    ``pending`` forever. Single-window inputs escaped by luck.

    * When ``session_factory`` is provided (the production/worker path), each
      window runs CONCURRENTLY (bounded by ``parallelism``) inside its OWN
      short-lived session + its own :class:`ChunkingCache`. No two coroutines
      ever share a session, so the flush race is impossible AND throughput is
      restored — large PDFs no longer serialise dozens of LLM calls. Each
      per-window session commits independently (the ``ai_model_calls`` audit
      rows and cache writes are independent inserts).
    * When ``session_factory`` is ``None`` (unit tests passing a mock ``db`` /
      ``cache``, or any caller without a real sessionmaker), it falls back to
      the SEQUENTIAL shared-``db`` path — correct, just not parallel.
    """
    if not windows:
        return []

    if session_factory is not None:
        semaphore = asyncio.Semaphore(max(1, parallelism))
        enrichments = await asyncio.gather(
            *(
                _enrich_one_isolated(
                    window=w,
                    llm_gateway=llm_gateway,
                    session_factory=session_factory,
                    document_title=document_title,
                    pipeline_run_id=pipeline_run_id,
                    parent_job_id=parent_job_id,
                    semaphore=semaphore,
                )
                for w in windows
            )
        )
        return [_promote_to_enriched(w, e) for w, e in zip(windows, enrichments, strict=True)]

    # Fallback: sequential on the shared session (no per-call isolation
    # available). Semaphore(1) is belt-and-suspenders — the ``await`` in the
    # comprehension already serialises — so no two coroutines touch ``db``
    # concurrently.
    semaphore = asyncio.Semaphore(1)
    enrichments = [
        await _enrich_one(
            window=w,
            llm_gateway=llm_gateway,
            db=db,
            cache=cache,
            document_title=document_title,
            pipeline_run_id=pipeline_run_id,
            parent_job_id=parent_job_id,
            semaphore=semaphore,
        )
        for w in windows
    ]
    return [_promote_to_enriched(w, e) for w, e in zip(windows, enrichments, strict=True)]

async def _enrich_one_isolated(
    *,
    window: RawChunk,
    llm_gateway: LLMGatewayProto,
    session_factory: SessionFactory,
    document_title: str | None,
    pipeline_run_id: UUID | None,
    parent_job_id: UUID | None,
    semaphore: asyncio.Semaphore,
) -> WindowEnrichment:
    """Run one window's enrichment inside its OWN session.

    Isolates all DB I/O (cache get/put + the gateway audit flush) onto a
    dedicated session so concurrent windows never share one. Commits the
    session on the way out so the audit row + cache entry persist; a commit
    failure must never sink the ingest, so it degrades to the LLM result we
    already computed (the enrichment value is independent of the audit write).
    """
    from abridgeai.ai.chunking.cache import ChunkingCache  # noqa: PLC0415

    # CRITICAL: the semaphore must gate the ENTIRE session lifecycle, not just
    # the LLM call. ``asyncio.gather`` schedules all N window coroutines at
    # once; if each opened its session before acquiring the semaphore, an
    # N-window PDF would open N sessions simultaneously and exhaust the DB
    # connection pool (``QueuePool limit ... reached``). Acquiring FIRST caps
    # concurrent open sessions at ``parallelism``. The inner ``_enrich_one``
    # then runs with an unbounded semaphore so it does not double-gate.
    async with semaphore, session_factory() as session:
        cache = ChunkingCache(session)
        enrichment = await _enrich_one(
            window=window,
            llm_gateway=llm_gateway,
            db=session,
            cache=cache,
            document_title=document_title,
            pipeline_run_id=pipeline_run_id,
            parent_job_id=parent_job_id,
            semaphore=_UNBOUNDED_SEMAPHORE,
        )
        try:
            await session.commit()
        except Exception as exc:  # noqa: BLE001 -- audit persistence must not break ingest
            logger.warning(
                "chunking enrichment session commit failed (group=%s): %s",
                window.metadata.get("glue_group_id"),
                exc,
            )
        return enrichment


async def _enrich_one(
    *,
    window: RawChunk,
    llm_gateway: LLMGatewayProto,
    db: AsyncSession,
    cache: ChunkingCache | None,
    document_title: str | None,
    pipeline_run_id: UUID | None,
    parent_job_id: UUID | None,
    semaphore: asyncio.Semaphore,
) -> WindowEnrichment:
    fallback_role = str(window.metadata.get("content_role") or "body")
    chash = content_hash(window.content)

    if cache is not None:
        cached = await cache.get(chash, PROMPT_VERSION)
        if cached is not None:
            return _build_enrichment(
                _normalize_payload(cached["output_json"], fallback_role=fallback_role),
                cached=True,
                model_name=cached.get("model_name"),
                input_tokens=cached.get("input_tokens"),
                output_tokens=cached.get("output_tokens"),
            )

    async with semaphore:
        try:
            from abridgeai.ai.llm.roles import LLMRole

            result = await llm_gateway.generate_json(
                role=LLMRole.CHUNKING_ENRICHMENT,
                system_prompt=_ENRICHMENT_SYSTEM_PROMPT,
                user_prompt=_user_prompt(window, document_title),
                db=db,
                stage_name="chunking_enrichment",
                pipeline_run_id=pipeline_run_id,
                parent_job_id=parent_job_id,
            )
        except Exception as exc:  # noqa: BLE001 -- graceful Stage-C degrade per plan §3686
            logger.warning(
                "chunking enrichment LLM call failed (group=%s): %s",
                window.metadata.get("glue_group_id"),
                exc,
            )
            return WindowEnrichment(
                section_title=_fallback_title(window),
                content_role=fallback_role,
            )

    payload = result.content_json if isinstance(result.content_json, dict) else {}
    normalized = _normalize_payload(payload, fallback_role=fallback_role)

    if cache is not None:
        try:
            await cache.put(
                chash,
                PROMPT_VERSION,
                output_json=normalized,
                model_name=result.model_name,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
        except Exception as exc:  # noqa: BLE001 -- cache write must not break ingest
            logger.warning("chunking cache write failed: %s", exc)

    return _build_enrichment(
        normalized,
        cached=False,
        model_name=result.model_name,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


def _build_enrichment(
    normalized: dict[str, Any],
    *,
    cached: bool,
    model_name: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
) -> WindowEnrichment:
    return WindowEnrichment(
        section_title=normalized["section_title"],
        content_role=normalized["content_role"],
        context_sentence=normalized["context_sentence"],
        key_concepts=normalized["key_concepts"],
        propositions=normalized["propositions"],
        cached=cached,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _user_prompt(window: RawChunk, document_title: str | None) -> str:
    capped = truncate_to_tokens(window.content, _LLM_INPUT_TOKEN_BUDGET)
    pieces: list[str] = []
    if document_title:
        pieces.append(f"DOCUMENT: {document_title}")
    page_range = window.metadata.get("page_range")
    if isinstance(page_range, tuple) and page_range[0] and page_range[1]:
        lo, hi = page_range
        label = f"WINDOW LOCATION: page {lo}" if lo == hi else f"WINDOW LOCATION: pages {lo}-{hi}"
        pieces.append(label)
    pieces.append("WINDOW TEXT:")
    pieces.append(capped)
    return "\n\n".join(pieces)


def _normalize_payload(payload: dict[str, Any], *, fallback_role: str) -> dict[str, Any]:
    title = (payload.get("section_title") or "").strip() or "Untitled section"
    role = (payload.get("content_role") or fallback_role).strip().lower()
    if role not in _VALID_ROLES:
        role = fallback_role
    context = (payload.get("context_sentence") or "").strip()
    raw_concepts = payload.get("key_concepts") or []
    concepts = [
        c.strip().lower()
        for c in (raw_concepts if isinstance(raw_concepts, list) else [])
        if isinstance(c, str) and c.strip()
    ]
    raw_props = payload.get("propositions") or []
    propositions = [
        p.strip()
        for p in (raw_props if isinstance(raw_props, list) else [])
        if isinstance(p, str) and p.strip()
    ]
    return {
        "section_title": title[:120],
        "content_role": role,
        "context_sentence": context[:500],
        "key_concepts": concepts[:10],
        "propositions": propositions[:12],
    }


def _fallback_title(window: RawChunk) -> str:
    section = str(window.metadata.get("section") or "")
    if section:
        leaf = section.rsplit(">", 1)[-1].strip()
        if leaf:
            return leaf[:120]
    page_range = window.metadata.get("page_range")
    if isinstance(page_range, tuple) and page_range[0] and page_range[1]:
        lo, hi = page_range
        return f"Page {lo}" if lo == hi else f"Pages {lo}-{hi}"
    return "Untitled section"


def _promote_to_enriched(window: RawChunk, enrichment: WindowEnrichment) -> EnrichedChunk:
    semantic = {
        "section_title": enrichment.section_title,
        "content_role": enrichment.content_role,
        "context_sentence": enrichment.context_sentence,
        "key_concepts": list(enrichment.key_concepts),
        "propositions": list(enrichment.propositions),
        "cached": enrichment.cached,
        "model_name": enrichment.model_name,
        "input_tokens": enrichment.input_tokens,
        "output_tokens": enrichment.output_tokens,
        "glue_group_id": window.metadata.get("glue_group_id"),
        "member_indices": list(window.metadata.get("member_indices") or []),
    }
    return EnrichedChunk(
        content=window.content,
        chunk_index=window.chunk_index,
        metadata=dict(window.metadata),
        embedding=None,
        semantic_metadata=semantic,
    )


def promote_rule_based(window: RawChunk) -> EnrichedChunk:
    return _promote_to_enriched(
        window,
        WindowEnrichment(
            section_title=_fallback_title(window),
            content_role=str(window.metadata.get("content_role") or "body"),
        ),
    )


__all__ = [
    "PROMPT_VERSION",
    "LLMGatewayProto",
    "WindowEnrichment",
    "content_hash",
    "enrich_with_llm",
    "promote_rule_based",
]
