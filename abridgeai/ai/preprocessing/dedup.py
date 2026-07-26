"""Near-duplicate detection — link to a canonical, never delete.

The education-specific case that breaks the standard dedup playbook is the
recap slide. It is a near-duplicate of the original slide, but it is usually
phrased closest to how the exam question is worded, which makes it the best
*retrieval anchor* even though it is a poor *answer body*. Deleting it costs
recall on exactly the queries students ask.

So all three passes annotate rather than remove: a duplicate gets a
``canonical_chunk_hash`` pointing at the first occurrence. Retrieval collapses
the pair at query time — keep the higher-scoring member for ranking, expand to
the canonical body for the generation context.

SimHash (stdlib, 64-bit over 5-grams, Hamming <= 3) is the default because it
needs no extra dependency. The semantic pass deliberately lives downstream in
the pipeline, where embeddings already exist, and uses a 0.94 cosine gate —
not the usual 0.75, because within a single course everything is on-topic and
the intra-corpus baseline already runs 0.70-0.85.
"""

from __future__ import annotations

import hashlib
import re

_WS_RE = re.compile(r"\s+")
_NGRAM = 5
_HAMMING_MAX = 3
_SIMHASH_BITS = 64
_MIN_TOKENS_FOR_SIMHASH = 12


def content_hash(text: str) -> str:
    """SHA-256 of the whitespace-normalized text (exact-duplicate key)."""
    return hashlib.sha256(_WS_RE.sub(" ", text.strip()).lower().encode("utf-8")).hexdigest()


def _shingles(text: str) -> list[str]:
    tokens = _WS_RE.sub(" ", text.strip().lower()).split()
    if len(tokens) < _NGRAM:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + _NGRAM]) for i in range(len(tokens) - _NGRAM + 1)]


def simhash64(text: str) -> int:
    """64-bit SimHash over 5-gram shingles."""
    vector = [0] * _SIMHASH_BITS
    shingles = _shingles(text)
    if not shingles:
        return 0
    for shingle in shingles:
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(_SIMHASH_BITS):
            vector[bit] += 1 if (value >> bit) & 1 else -1
    out = 0
    for bit in range(_SIMHASH_BITS):
        if vector[bit] > 0:
            out |= 1 << bit
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def find_duplicates(texts: list[str]) -> dict[int, tuple[int, str]]:
    """Map ``index -> (canonical_index, kind)`` for exact and near duplicates.

    ``kind`` is ``"exact"`` or ``"lexical"``. The first occurrence in reading
    order is the canonical; later occurrences point back at it.
    """
    result: dict[int, tuple[int, str]] = {}
    seen_exact: dict[str, int] = {}
    signatures: list[tuple[int, int]] = []

    for index, text in enumerate(texts):
        digest = content_hash(text)
        if digest in seen_exact:
            result[index] = (seen_exact[digest], "exact")
            continue
        seen_exact[digest] = index

        # SimHash on very short text is dominated by a handful of shingles and
        # produces spurious collisions; short chunks fall through to exact-only.
        if len(text.split()) < _MIN_TOKENS_FOR_SIMHASH:
            continue

        signature = simhash64(text)
        for prior_index, prior_signature in signatures:
            if hamming(signature, prior_signature) <= _HAMMING_MAX:
                result[index] = (prior_index, "lexical")
                break
        signatures.append((index, signature))

    return result


def link_semantic_duplicates(
    embeddings: list[list[float]],
    *,
    threshold: float = 0.94,
) -> dict[int, int]:
    """Map ``index -> canonical_index`` for chunks that say the same thing.

    This is the pass that actually catches a reworded recap slide: swapping a
    single word moves a chunk ~15 bits in SimHash space (far outside the
    lexical gate) while barely moving its embedding.

    The 0.94 gate is deliberately high. The usual 0.75 advice assumes a
    heterogeneous corpus; within ONE course everything is on-topic, so the
    intra-corpus cosine baseline already sits around 0.70-0.85 and boilerplate
    vectors cluster near the centroid. 0.94 means "same content, reworded".

    O(n^2) over one document's chunks — a few hundred at most, and each
    comparison is a dot product over vectors already in memory.
    """
    canonical: dict[int, int] = {}
    norms = [_norm(v) for v in embeddings]
    for i in range(len(embeddings)):
        if norms[i] == 0.0:
            continue
        for j in range(i):
            if j in canonical or norms[j] == 0.0:
                continue
            if _cosine(embeddings[i], embeddings[j], norms[i], norms[j]) >= threshold:
                canonical[i] = j
                break
    return canonical


def _norm(vector: list[float]) -> float:
    return sum(v * v for v in vector) ** 0.5


def _cosine(a: list[float], b: list[float], norm_a: float, norm_b: float) -> float:
    if len(a) != len(b) or norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (norm_a * norm_b)


__all__ = [
    "content_hash",
    "find_duplicates",
    "hamming",
    "link_semantic_duplicates",
    "simhash64",
]
