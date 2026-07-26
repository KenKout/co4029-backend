"""Stage B' — LLM-decided chunk boundaries.

Replaces the embedding similarity merge (:mod:`_glue`) when a gateway is
available. Cosine similarity between two adjacent slides answers "do these
use the same words", which is not the question. A definition slide and the
worked example that follows it score low; two consecutive bullet lists from
unrelated sections score high. The question is "does slide N+1 continue the
thought of slide N", and that is a reading-comprehension judgement.

Why this matters here: the ingestion pipeline constructs ``SemanticChunker()``
without an embedder, so Stage B has only ever taken its ``embedder is None``
branch — one window per page plus the tiny-window absorb. A 37-slide lecture
came out as 37 chunks with a median of 113 tokens, well under the size where
an embedding carries a usable amount of meaning.

Scale
-----
The naive shape ("show the LLM all N windows, ask for a partition") breaks on
long documents, and not for the reason people expect. Input size is not the
constraint: windows are sent as *digests* (index, slide title, token count,
opening ~150 chars), measured at ~47 tokens each, so even 500 windows fit a
context window comfortably. What degrades is the model's reliability at
partitioning a long list — indices get dropped, repeated, or reordered well
before the context runs out.

So the work is bounded two ways, in this order:

1. **Barriers.** Grouping never crosses a ``topic_group_id`` change (assigned
   upstream by ``preprocessing/deck.py`` from consecutive slides sharing a
   normalized title) or a ``content_role`` change (the rule :mod:`_glue`
   already enforced). Merging across either is merging across topics, which
   is wrong regardless of what the model says. On a real lecture this alone
   reduces the problem to runs of 1-7 windows.

2. **Batching with carry-over**, for the residual case of one barrier run
   longer than ``batch_size``. See :func:`_group_run`.

Everything the model returns is validated before use, and any violation
falls back to identity grouping for that run — never to a partial or
reordered document. A boundary model that is unavailable, slow, or wrong
must cost chunk quality, never content.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

from abridgeai.ai.chunking.base import RawChunk
from abridgeai.ai.llm.roles import LLMRole

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.ai.chunking.cache import ChunkingCache
    from abridgeai.ai.llm.gateway import LLMResult

logger = logging.getLogger(__name__)

BOUNDARY_STAGE_NAME = "chunk_boundary"
# Cache namespace. Shares ``chunking_enrichment_cache`` with Stage C, which is
# keyed on ``(content_hash, prompt_version)`` — a distinct version string keeps
# the two from colliding, and bumping it invalidates cached groupings when the
# prompt changes.
BOUNDARY_PROMPT_VERSION = "boundary-v1"

# Windows per LLM call. Digests run ~47 tokens, so 40 is ~1.9k input — the
# ceiling is the model's accuracy on a long partition, not the context window.
_BATCH_SIZE = 40
# Ceiling on windows carried into the next batch. Without it, a model that
# answers "all 40 are one group" carries all 40 forward, the read cursor never
# advances, and the loop spins forever. See :func:`_group_run`.
_MAX_CARRY = 8
# Digest excerpt per window. Enough to recognise a continuation, short enough
# that the list stays cheap.
_EXCERPT_CHARS = 150

BOUNDARY_SYSTEM_PROMPT = (
    "You segment lecture material into retrieval chunks.\n"
    "You are given consecutive windows from ONE document section, in reading "
    "order, each with an index.\n"
    "Group consecutive windows that develop a single idea, so each group can "
    "be read and understood on its own.\n"
    "\n"
    "Rules:\n"
    "- Groups must be CONSECUTIVE index runs. Never reorder, never skip.\n"
    "- Every index given to you must appear in exactly one group.\n"
    "- A slide that only introduces a topic belongs with the slides that "
    "develop it. A definition belongs with its example.\n"
    "- Start a new group when the subject changes, even if the wording is "
    "similar.\n"
    "- Prefer groups of 2-4 windows. A group of 1 is correct when the window "
    "stands alone.\n"
    'Return JSON: {"groups": [[0, 1], [2], [3, 4, 5]]}'
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
    ) -> LLMResult: ...


def _meta(chunk: RawChunk, key: str, default: Any = None) -> Any:  # noqa: ANN401
    value = chunk.metadata.get(key) if chunk.metadata else None
    return default if value is None else value


def _tokens_of(chunk: RawChunk) -> int:
    value = _meta(chunk, "token_count", 0)
    return int(value) if isinstance(value, int) else 0


def _barrier_key(chunk: RawChunk) -> tuple[Any, str]:
    """Windows may only be grouped with others sharing this key."""
    return (_meta(chunk, "topic_group_id"), str(_meta(chunk, "content_role", "body")))


def split_on_barriers(chunks: list[RawChunk]) -> list[list[int]]:
    """Split indices into consecutive runs that share a barrier key.

    Returned runs partition ``range(len(chunks))`` in order, so concatenating
    them reproduces the document exactly.
    """
    runs: list[list[int]] = []
    for index, chunk in enumerate(chunks):
        key = _barrier_key(chunk)
        if runs and _barrier_key(chunks[runs[-1][-1]]) == key:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def _digest(chunk: RawChunk, label: int) -> str:
    title = str(_meta(chunk, "slide_title", "") or "").strip()
    excerpt = " ".join(chunk.content.split())[:_EXCERPT_CHARS]
    return f"[{label}] ({_tokens_of(chunk)} tokens) {title} :: {excerpt}"


def validate_groups(
    raw: object,
    *,
    expected: list[int],
    tokens: dict[int, int],
    max_window_tokens: int,
) -> list[list[int]] | None:
    """Return validated groups, or ``None`` if the response cannot be trusted.

    Rejects — rather than repairs — on anything that would alter the document:

    * an index missing (content silently dropped) or repeated (duplicated),
    * a group that is not a strictly consecutive ascending run (reordering),
    * a group over ``max_window_tokens`` (unusable as an embedding target).

    Repairing these is tempting and wrong: every repair is a guess about what
    the model meant, applied to the one artefact the rest of the pipeline
    treats as ground truth. Identity grouping is a known-safe answer.
    """
    if not isinstance(raw, dict):
        return None
    groups_raw = raw.get("groups")
    if not isinstance(groups_raw, list):
        return None

    groups: list[list[int]] = []
    for group in groups_raw:
        if not isinstance(group, list) or not group:
            return None
        try:
            members = [int(i) for i in group]
        except (TypeError, ValueError):
            return None
        if members != list(range(members[0], members[0] + len(members))):
            return None  # not a consecutive ascending run
        if sum(tokens.get(i, 0) for i in members) > max_window_tokens:
            return None
        groups.append(members)

    flat = [i for group in groups for i in group]
    if flat != expected:
        # Covers missing, duplicated, out-of-range and out-of-order indices in
        # one comparison: the concatenation must reproduce the input exactly.
        return None
    return groups


async def _call_boundary_model(
    windows: list[RawChunk],
    labels: list[int],
    *,
    llm_gateway: LLMGatewayProto,
    db: AsyncSession,
    pipeline_run_id: UUID | None,
    parent_job_id: UUID | None,
) -> object:
    listing = "\n".join(_digest(w, label) for w, label in zip(windows, labels, strict=True))
    section = str(_meta(windows[0], "slide_title", "") or "").strip()
    header = f"Section: {section}\n\n" if section else ""
    result = await llm_gateway.generate_json(
        role=LLMRole.CHUNK_BOUNDARY,
        system_prompt=BOUNDARY_SYSTEM_PROMPT,
        user_prompt=(
            f"{header}Windows (indices {labels[0]}-{labels[-1]}), in reading order:\n"
            f"{listing}\n\nReturn the grouping as JSON."
        ),
        db=db,
        stage_name=BOUNDARY_STAGE_NAME,
        pipeline_run_id=pipeline_run_id,
        parent_job_id=parent_job_id,
    )
    content = result.content_json
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None
    return content


def _run_cache_key(chunks: list[RawChunk], run: list[int]) -> str:
    """Content hash for one barrier run's grouping decision.

    Hashes exactly what the model is shown — the digests, in order — so the
    cache hits when the source is unchanged and misses the moment any window
    in the run gains, loses or reorders content. Position-independent: the same
    run of slides re-ingested at a different offset still hits.
    """
    payload = "\n".join(_digest(chunks[i], position) for position, i in enumerate(run))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def _group_run(
    chunks: list[RawChunk],
    run: list[int],
    *,
    llm_gateway: LLMGatewayProto,
    db: AsyncSession,
    max_window_tokens: int,
    batch_size: int,
    max_carry: int,
    pipeline_run_id: UUID | None,
    parent_job_id: UUID | None,
) -> list[list[int]]:
    """Group one barrier run, batching with carry-over when it is long.

    The carry-over exists because the model cannot see past the end of its
    batch. Shown windows ``[i, i+B)``, the last group it forms is a guess: it
    has no way to know whether window ``i+B`` continues that thought. Committing
    that group anyway hard-splits a topic at a boundary chosen by the batch
    size. So the trailing group is withheld and re-shown at the head of the
    next batch, where the model can see what follows it.

    ``max_carry`` is what keeps this terminating. If the model answers "all B
    windows are one group", the trailing group IS the batch; carrying it whole
    advances the cursor by zero and the loop never ends. Past the cap the
    trailing group is committed as-is — by then it is large enough that the
    token ceiling would have split it regardless.
    """
    tokens = {i: _tokens_of(chunks[i]) for i in run}
    groups: list[list[int]] = []
    cursor = 0

    while cursor < len(run):
        batch = run[cursor : cursor + batch_size]
        raw = await _call_boundary_model(
            [chunks[i] for i in batch],
            batch,
            llm_gateway=llm_gateway,
            db=db,
            pipeline_run_id=pipeline_run_id,
            parent_job_id=parent_job_id,
        )
        validated = validate_groups(
            raw, expected=batch, tokens=tokens, max_window_tokens=max_window_tokens
        )
        if validated is None:
            logger.warning(
                "chunk_boundary: unusable grouping for indices %s-%s; "
                "falling back to one chunk per window for this run",
                batch[0],
                batch[-1],
            )
            groups.extend([i] for i in batch)
            cursor += len(batch)
            continue

        is_last_batch = cursor + len(batch) >= len(run)
        trailing = validated[-1]
        # ``len(trailing) >= len(batch)`` is the termination guard, and it is
        # not implied by ``max_carry``: with the defaults a 40-window batch
        # returned as one group trips the cap, but a caller passing
        # ``batch_size <= max_carry`` would not, and the cursor would advance by
        # zero forever. Withholding is only ever safe when something is left to
        # commit.
        if is_last_batch or len(trailing) > max_carry or len(trailing) >= len(batch):
            groups.extend(validated)
            cursor += len(batch)
        else:
            groups.extend(validated[:-1])
            cursor += len(batch) - len(trailing)

    return groups


def _offsets_from_cache(hit: dict[str, Any] | None, *, expected: int) -> list[list[int]] | None:
    """Re-validate a cached grouping before trusting it.

    A cache row can outlive the code that wrote it, so the stored partition is
    checked against the same coverage rule as a fresh response. An entry that
    no longer covers the run is ignored, not repaired.
    """
    if not hit:
        return None
    payload = hit.get("output_json")
    if not isinstance(payload, dict):
        return None
    groups = payload.get("groups")
    if not isinstance(groups, list):
        return None
    try:
        offsets = [[int(o) for o in group] for group in groups]
    except (TypeError, ValueError):
        return None
    if [o for group in offsets for o in group] != list(range(expected)):
        return None
    return offsets


async def group_by_llm_boundaries(
    chunks: list[RawChunk],
    *,
    llm_gateway: LLMGatewayProto | None,
    db: AsyncSession | None,
    cache: ChunkingCache | None = None,
    max_window_tokens: int = 2000,
    batch_size: int = _BATCH_SIZE,
    max_carry: int = _MAX_CARRY,
    pipeline_run_id: UUID | None = None,
    parent_job_id: UUID | None = None,
) -> list[list[int]] | None:
    """Decide chunk boundaries for ``chunks``; ``None`` means "caller decides".

    Returns a partition of ``range(len(chunks))`` into consecutive groups.
    ``None`` is returned when no gateway/session is available, so the caller
    can fall through to the existing embedding path rather than having a
    grouping decision silently made for it.

    Calls run sequentially. They share the caller's ``AsyncSession``, and two
    coroutines flushing one session raise ``Session is already flushing`` —
    the failure that used to hang every multi-window PDF (see
    ``_enrich.enrich_with_llm``). The volume does not justify the
    per-coroutine sessions that stage needs: barrier runs of length 1 need no
    call at all, so a 37-window lecture spends 4.
    """
    if not chunks or llm_gateway is None or db is None:
        return None

    runs = split_on_barriers(chunks)
    groups: list[list[int]] = []
    calls = 0
    cached = 0
    for run in runs:
        if len(run) == 1:
            groups.append(run)
            continue

        # Offsets within the run, not document indices: the cached decision is
        # about the shape of this run, so it stays valid if the run moves.
        key = _run_cache_key(chunks, run) if cache is not None else None
        if cache is not None and key is not None:
            hit = await cache.get(key, BOUNDARY_PROMPT_VERSION)
            offsets = _offsets_from_cache(hit, expected=len(run))
            if offsets is not None:
                cached += 1
                groups.extend([[run[o] for o in group] for group in offsets])
                continue

        run_groups = await _group_run(
            chunks,
            run,
            llm_gateway=llm_gateway,
            db=db,
            max_window_tokens=max_window_tokens,
            batch_size=batch_size,
            max_carry=max_carry,
            pipeline_run_id=pipeline_run_id,
            parent_job_id=parent_job_id,
        )
        calls += 1
        groups.extend(run_groups)

        if cache is not None and key is not None:
            position_of = {index: offset for offset, index in enumerate(run)}
            await cache.put(
                key,
                BOUNDARY_PROMPT_VERSION,
                output_json={
                    "groups": [[position_of[i] for i in group] for group in run_groups]
                },
            )

    flat = [i for group in groups for i in group]
    if flat != list(range(len(chunks))):
        # Defence in depth: per-run validation should make this unreachable.
        # If it ever fires, something reordered the document, and one chunk per
        # window is the only answer that cannot be wrong.
        logger.error(
            "chunk_boundary: assembled partition does not cover the document "
            "(%d of %d indices); falling back to one chunk per window",
            len(flat),
            len(chunks),
        )
        return [[i] for i in range(len(chunks))]

    logger.info(
        "chunk_boundary: %d windows -> %d chunks via %d LLM call(s) "
        "(%d run(s) from cache) over %d barrier run(s)",
        len(chunks),
        len(groups),
        calls,
        cached,
        len(runs),
    )
    return groups


__all__ = [
    "BOUNDARY_STAGE_NAME",
    "BOUNDARY_SYSTEM_PROMPT",
    "group_by_llm_boundaries",
    "split_on_barriers",
    "validate_groups",
]
