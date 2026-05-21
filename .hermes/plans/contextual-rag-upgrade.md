# Contextual RAG Upgrade Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task. Each phase commits to its own feature branch, merges to master `--no-ff`, pushes to `origin/master`. Frontend is local-only (no remote).

**Goal:** Bring co4029 quiz generation up to Anthropic Contextual Retrieval quality bar — fix the FR-5 panel UX bug and progressively wire in contextual embeddings, hybrid BM25, and reranking.

**Architecture:** Three layers of changes that compose. Phase 1 fixes the immediate UX bug (panel shows useful section grouping). Phases 2-4 progressively upgrade retrieval quality without re-running expensive Stage C LLM enrichment (its output is already cached and contains the contextual sentences we need).

**Tech Stack:** Python 3.12 / FastAPI / SQLAlchemy 2.0 (async) / Postgres 16 + pgvector + tsvector / Vite + React + TS / tiktoken `o200k_base` / Voyage rerank-2.5

**Repos:**
- Backend: `/root/co4029/backend` (origin: `git@github.com:KenKout/co4029-backend.git`)
- Frontend: `/root/co4029/frontend` (local only)

**Process services (pm2):** `abridgeai-backend` :8000, `abridgeai-worker`, `abridgeai-frontend` :5173

---

## Phase 1 — UI knob: section granularity (UX fix)

**Why first:** Bug-shaped, no infra changes, immediate user value. Backend already accepts `coverage_options.slides_per_section` (default 4) but the new semantic-aware outline builder ignores it because every chunk now has a unique semantic title. Need to honor the knob explicitly when user passes it, and surface it in the panel UI.

### Task 1.1 — Honor `slides_per_section` knob in outline builder

**Objective:** When `slides_per_section` is explicitly passed and there's no real heading structure (PDF slide-decks), bundle by size like the legacy did. When user wants Auto, fall through to semantic-aware grouping.

**Files:**
- Modify: `abridgeai/features/quizzes/ai/outline.py:140` — `_group_sections()` add `force_bundle: bool` param
- Modify: `abridgeai/features/quizzes/ai/outline.py:126` — top-level `build_lesson_outline()` pass through
- Test: `tests/unit/test_quiz_outline.py` — new test asserting force-bundle ignores semantic titles

**Step 1: Write failing test**

```python
def test_group_sections_force_bundles_when_size_knob_set() -> None:
    """When force_bundle=True, builder bundles by slides_per_section
    regardless of semantic titles — gives user the legacy 'gọn'
    grouping for slide-deck PDFs.
    """
    chunks = [
        _chunk(
            chunk_id=UUID(int=i),
            content=f"page-{i}",
            section=f"Page {i}",
            page=i,
            page_at_top_level=True,
            semantic_title=f"Topic {i}",  # all unique
        )
        for i in range(1, 9)
    ]
    sections = _group_sections(chunks, slides_per_section=4, force_bundle=True)
    assert len(sections) == 2
    assert [len(s.chunk_ids) for s in sections] == [4, 4]
    # Title comes from first chunk's semantic title
    assert sections[0].title == "Topic 1"
    assert sections[0].page_range == (1, 4)
    assert sections[1].page_range == (5, 8)
```

**Step 2: Run test (expect FAIL)**

```bash
cd /root/co4029/backend && uv run --no-sync pytest tests/unit/test_quiz_outline.py::test_group_sections_force_bundles_when_size_knob_set -v --no-cov
```

**Step 3: Implementation**

Add `force_bundle: bool = False` to `_group_sections()` signature; when True, always go through `_group_slide_deck()`. Plumb through `build_lesson_outline()`.

**Step 4: Run all outline tests (expect PASS)**

```bash
uv run --no-sync pytest tests/unit/test_quiz_outline.py -v --no-cov
```

**Step 5: Lint + typecheck**

```bash
uv run --no-sync ruff check abridgeai/features/quizzes/ai/outline.py tests/unit/test_quiz_outline.py
uv run --no-sync mypy abridgeai/features/quizzes/ai/outline.py
```

### Task 1.2 — Wire `force_bundle` from generation config

**Objective:** `coverage_options` schema gets a new `section_grouping: Literal["auto", "fixed"]` field. When `"fixed"`, pass `force_bundle=True` to outline builder.

**Files:**
- Modify: `abridgeai/features/quizzes/schemas/run.py:96` — add `section_grouping` to `CoverageOptions`
- Modify: `abridgeai/features/quizzes/services/generation.py:120` — read knob, pass `force_bundle` through

**Step 1: Schema field**

```python
class CoverageOptions(BaseModel):
    # ... existing fields ...
    section_grouping: Literal["auto", "fixed"] = "fixed"
    """When 'fixed', force-bundle ``slides_per_section`` consecutive
    chunks per section regardless of semantic titles. When 'auto', the
    outline builder uses semantic enrichment to draw section
    boundaries (one section per topic). Default 'fixed' matches legacy
    UX where the panel shows ~ceil(N/slides_per_section) sections."""
    slides_per_section: Annotated[int, Field(ge=1, le=20)] = 4
```

**Step 2: Wire through service**

In `services/generation.py:120` (`_resolve_coverage_options`), pass both `slides_per_section` and `section_grouping` through to `build_lesson_outline()`.

**Step 3: Tests for both modes**

Add 2 router tests asserting outline shape differs between `auto` and `fixed`.

**Step 4: Verify**

```bash
uv run --no-sync pytest tests/integration/test_quiz_authoring_service.py tests/unit/test_quiz_outline.py -v --no-cov
```

### Task 1.3 — Frontend dropdown for `slides_per_section`

**Objective:** Quiz generation panel exposes "Section grouping" dropdown: `Auto (semantic)` | `4 pages/section` | `8 pages/section` | `16 pages/section`. Default `4 pages/section` (matches user's mental model from legacy).

**Files:**
- Modify: `frontend/src/routes/teacher/_components/quiz-generation-form-controls.tsx` — add `<Select>` for section_grouping + slides_per_section
- Modify: `frontend/src/routes/teacher/_components/quiz-generation-panel.tsx` — wire form state to mutation payload
- Run: `npm run codegen:api` after backend deploy

**Step 1: Codegen new types**

```bash
cd /root/co4029/frontend && curl -s http://localhost:8000/openapi.json -o openapi-snapshot.json && npm run codegen:api
```

**Step 2: Add UI control**

In form controls, render right next to existing coverage controls:

```tsx
<FormField name="sectionGrouping">
  <Label>Section grouping</Label>
  <Select value={value} onValueChange={onChange}>
    <SelectItem value="auto">Auto (semantic)</SelectItem>
    <SelectItem value="fixed-4">4 pages/section</SelectItem>
    <SelectItem value="fixed-8">8 pages/section</SelectItem>
    <SelectItem value="fixed-16">16 pages/section</SelectItem>
  </Select>
</FormField>
```

State maps to `{section_grouping: "fixed", slides_per_section: 4}` etc.

**Step 3: Verify in browser**

Refresh `https://abridgeai.tech/teacher/courses/.../quizzes/.../`, open panel, switch dropdown options, observe section count change.

**Step 4: Commit Phase 1**

Single branch `feat/section-grouping-knob`, both backend + frontend commits, merge to master no-ff, push backend.

**Verification checklist for Phase 1:**

- Outline endpoint with `slides_per_section=4, section_grouping=fixed` returns ~ceil(43/4)=11 sections for lesson `46741c12`
- Outline endpoint with `section_grouping=auto` returns ~41 sections (semantic-aware)
- Panel dropdown defaults to `4 pages/section`
- All unit + integration tests green

---

## Phase 2 — Contextual Embeddings (Anthropic technique)

**Why this:** Stage C enrichment already produces `metadata.semantic.context_sentence` (1-2 sentences situating the window in the wider document) — that's exactly what Anthropic's contextual retrieval prepends to chunks before embedding. Currently we ignore this output when calling the embedder. Wire it in. -35% retrieval failure rate per Anthropic's BEIR-style benchmark.

### Task 2.1 — Switch tokenizer to o200k_base

**Objective:** User explicitly requested GPT-5 tokenizer. Update the single constant.

**Files:**
- Modify: `abridgeai/ai/chunking/token_aware.py:34` — `_TIKTOKEN_ENCODING_NAME = "o200k_base"`

**Step 1: Verify tiktoken bundle**

```bash
cd /root/co4029/backend && uv run --no-sync python -c "import tiktoken; t = tiktoken.get_encoding('o200k_base'); print(t.encode('hello world'))"
```

Expected: a list of int tokens.

**Step 2: Update constant + run all chunking tests**

```bash
uv run --no-sync pytest tests/unit/test_token_aware.py tests/unit/test_quiz_outline.py -v --no-cov
```

Note: token counts will shift slightly (o200k tokenizes some Vietnamese characters more efficiently). Update any hardcoded `count_tokens(...) == N` test assertions to use ranges if they break.

### Task 2.2 — Build contextual prepend helper

**Objective:** Pure function that takes a `RawChunk` and returns the contextualized text to embed. Format:

```
[Topic: {section_title}] {context_sentence} {content}
```

Where `section_title` and `context_sentence` come from `metadata.semantic.*`. Falls back gracefully when enrichment is missing (returns just `content`).

**Files:**
- Create: `abridgeai/ai/chunking/contextual.py`
- Test: `tests/unit/test_contextual_chunking.py`

**Step 1: Test cases**

```python
def test_contextual_prepend_uses_semantic_metadata():
    chunk = RawChunk(
        chunk_index=0,
        content="The system handles 1M requests per second.",
        metadata={
            "semantic": {
                "section_title": "Performance Architecture",
                "context_sentence": "This section describes the load characteristics of the production deployment.",
            },
        },
    )
    result = build_contextual_text(chunk)
    assert result.startswith("[Topic: Performance Architecture]")
    assert "load characteristics" in result
    assert "1M requests per second" in result
    # Budget guard: prepend ≤ 150 tokens
    assert count_tokens(result) - count_tokens(chunk.content) <= 150


def test_contextual_prepend_returns_content_when_no_enrichment():
    chunk = RawChunk(chunk_index=0, content="bare content", metadata={})
    assert build_contextual_text(chunk) == "bare content"
```

**Step 2: Implementation**

```python
def build_contextual_text(chunk: RawChunk, *, max_prefix_tokens: int = 150) -> str:
    """Prepend Stage C semantic context to chunk content for embedding.
    
    Format: ``[Topic: {title}] {context_sentence} {content}``. When
    enrichment is missing, returns content unchanged. Truncates the
    prefix at ``max_prefix_tokens`` to bound embedding token cost.
    """
    metadata = chunk.metadata or {}
    semantic = metadata.get("semantic") or {}
    title = (semantic.get("section_title") or "").strip()
    ctx = (semantic.get("context_sentence") or "").strip()
    
    if not title and not ctx:
        return chunk.content
    
    parts: list[str] = []
    if title:
        parts.append(f"[Topic: {title}]")
    if ctx:
        parts.append(ctx)
    prefix = " ".join(parts)
    
    if count_tokens(prefix) > max_prefix_tokens:
        prefix = _truncate_to_tokens(prefix, max_prefix_tokens)
    
    return f"{prefix} {chunk.content}"
```

### Task 2.3 — Wire prepend into ingestion pipeline

**Objective:** Embedder receives contextualized text, but `DocumentChunk.content` column keeps the original (so retrieval result presentation isn't polluted with the prefix).

**Files:**
- Modify: `abridgeai/features/materials/ingestion/pipeline.py:459`

**Step 1: Update embed call**

```python
# Was:
# embeddings = await embed_client.embed([c.content for c in raw_chunks], ...)

# Now:
from abridgeai.ai.chunking.contextual import build_contextual_text

embed_inputs = [build_contextual_text(c) for c in raw_chunks]
embeddings = await embed_client.embed(
    embed_inputs,
    db=db,
    pipeline_run_id=pipeline_run_id,
    parent_job_id=job.id,
)
```

`_persist_chunks` continues to write `raw.content` (unchanged) so chunk content presentation in the panel is untouched.

**Step 2: Run integration tests**

```bash
uv run --no-sync pytest tests/integration/test_materials_ingestion.py -v --no-cov
```

### Task 2.4 — Re-embed material 66e6c128 to validate

**Objective:** Re-process the test lesson PDF to populate new embeddings without re-running expensive LLM enrichment (Stage C cache hits on content_hash).

**Files:**
- Use: `abridgeai/features/materials/api/admin.py` (or worker queue endpoint) — find existing reprocess endpoint

**Step 1: Trigger reprocess via API**

```bash
# Find the right endpoint — likely POST /api/v1/teacher/materials/{id}/reprocess
curl -X POST "http://localhost:8000/api/v1/teacher/materials/66e6c128-3229-4cc3-bd61-28dcbe8b1d4/reprocess" \
  -H "Authorization: Bearer $TEACHER_JWT"
```

**Step 2: Watch worker logs**

```bash
pm2 logs abridgeai-worker --lines 50
```

Expected: chunking_enrichment_cache hits (`cached: true` in metadata) — no LLM cost. Embedder runs fresh.

**Step 3: Verify embeddings updated**

```sql
SELECT updated_at, COUNT(*) 
FROM document_chunks 
WHERE lesson_id = '46741c12-ff9d-48ac-9804-24753f6386eb' 
GROUP BY 1;
-- All rows should share the same recent timestamp
```

**Step 4: Commit Phase 2**

Branch: `feat/contextual-embeddings`. Commits:
1. Tokenizer switch
2. Contextual prepend helper + tests
3. Pipeline integration
4. (No DB migration — embedding schema unchanged)

---

## Phase 3 — Hybrid BM25 retrieval

**Why this:** Embeddings excel at semantic similarity but miss exact-term matches (e.g. "TS-999"). BM25 catches lexical hits. Combining both with Reciprocal Rank Fusion gets a further -14 percentage points retrieval failure (-35% to -49% combined per Anthropic). Postgres native `tsvector` + GIN index = no new infra dependency.

### Task 3.1 — Add `tsvector` column + GIN index

**Files:**
- Create: `alembic/versions/<timestamp>_add_document_chunks_tsv.py`

**Step 1: Migration**

```python
def upgrade() -> None:
    op.execute("""
        ALTER TABLE document_chunks
        ADD COLUMN content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('simple', coalesce(content, ''))) STORED
    """)
    op.execute("CREATE INDEX ix_document_chunks_content_tsv ON document_chunks USING GIN (content_tsv)")


def downgrade() -> None:
    op.execute("DROP INDEX ix_document_chunks_content_tsv")
    op.execute("ALTER TABLE document_chunks DROP COLUMN content_tsv")
```

**Note:** Use `'simple'` dictionary (no stemming). Vietnamese has no Postgres dictionary and `'simple'` handles bilingual EN/VI without breaking either side.

**Step 2: Apply migration**

```bash
cd /root/co4029/backend && uv run --no-sync alembic upgrade head
```

**Step 3: Verify**

```sql
\d document_chunks
-- Should show content_tsv column
SELECT content_tsv FROM document_chunks LIMIT 1;
```

### Task 3.2 — Hybrid retrieval query helper

**Objective:** New function `retrieve_hybrid(db, query, lesson_id, *, top_k=150, vector_weight=0.5)` returns top-K chunks using RRF fusion of pgvector cosine + BM25 ts_rank.

**Files:**
- Modify: `abridgeai/features/quizzes/queries/published.py` (or wherever current retrieval lives)
- Test: `tests/integration/test_hybrid_retrieval.py`

**Step 1: RRF query**

```sql
-- Reciprocal Rank Fusion: score = sum(1/(k + rank_i)) where k=60 (Anthropic default)
WITH vec AS (
  SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> :query_embedding) AS rank
  FROM document_chunks WHERE lesson_id = :lesson_id LIMIT 200
),
bm25 AS (
  SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank(content_tsv, plainto_tsquery('simple', :query_text)) DESC) AS rank
  FROM document_chunks WHERE lesson_id = :lesson_id AND content_tsv @@ plainto_tsquery('simple', :query_text) LIMIT 200
)
SELECT id, COALESCE(1.0/(60+vec.rank), 0) + COALESCE(1.0/(60+bm25.rank), 0) AS score
FROM document_chunks dc
LEFT JOIN vec USING (id)
LEFT JOIN bm25 USING (id)
WHERE dc.lesson_id = :lesson_id AND (vec.rank IS NOT NULL OR bm25.rank IS NOT NULL)
ORDER BY score DESC
LIMIT :top_k
```

**Step 2: Test cases**

Lexical-only match (made-up technical term not in embedding training), semantic-only match (paraphrased query), both-match (canonical case).

### Task 3.3 — Wire hybrid retrieval into quiz generation pipeline

**Files:**
- Modify: caller of current vector-only retriever in quiz generation pipeline

**Step 1: Find caller**

```bash
search_files("retrieve_for_lesson|chunk_search", path="abridgeai/features/quizzes/")
```

**Step 2: Replace call**

Drop-in replacement: same signature, new internals.

**Step 3: A/B test on lesson `46741c12`**

Generate quiz with old retriever (off branch) and new retriever (this branch), eyeball quality. Manual eval acceptable for now.

**Step 4: Commit Phase 3**

Branch: `feat/bm25-hybrid-retrieval`. Migration is hard-to-reverse on a populated DB; double-check `downgrade()` works on a fresh test DB before merging.

---

## Phase 4 — Voyage Rerank-2.5

**Why this:** Even with great retrieval, top-150 has noise. Reranker scores each (query, chunk) pair with a cross-encoder, picks the cleanest top-20. -67% combined per Anthropic when stacked with contextual + BM25. Voyage gives 200M tokens free per account, balanced speed/quality per Agentset benchmark.

### Task 4.1 — Voyage HTTP client

**Files:**
- Create: `abridgeai/ai/rerank/voyage.py`
- Test: `tests/unit/test_voyage_rerank.py` (mocked HTTP)

**Step 1: Client implementation**

```python
import httpx

class VoyageReranker:
    def __init__(self, api_key: str, model: str = "rerank-2.5"):
        self._api_key = api_key
        self._model = model
        self._endpoint = "https://api.voyageai.com/v1/rerank"
    
    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int = 20,
        timeout_s: float = 30.0,
    ) -> list[tuple[int, float]]:
        """Returns [(original_index, score), ...] sorted by score desc."""
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(
                self._endpoint,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "query": query,
                    "documents": documents,
                    "model": self._model,
                    "top_k": top_k,
                    "return_documents": False,
                },
            )
            response.raise_for_status()
            data = response.json()
        return [(item["index"], item["relevance_score"]) for item in data["data"]]
```

**Step 2: Mocked tests**

```python
async def test_voyage_rerank_returns_scored_indices(httpx_mock):
    httpx_mock.add_response(
        url="https://api.voyageai.com/v1/rerank",
        json={"data": [{"index": 2, "relevance_score": 0.95}, {"index": 0, "relevance_score": 0.82}]},
    )
    reranker = VoyageReranker(api_key="test-key")
    result = await reranker.rerank("query", ["a", "b", "c"], top_k=2)
    assert result == [(2, 0.95), (0, 0.82)]
```

### Task 4.2 — Config + env-gated wiring

**Files:**
- Modify: `abridgeai/config.py` — add `VOYAGE_API_KEY: str | None = None`
- Modify: quiz generation retrieval — gate reranker behind config check

**Step 1: Config field**

```python
class Settings(BaseSettings):
    # ... existing ...
    VOYAGE_API_KEY: SecretStr | None = None
```

**Step 2: Graceful skip when key absent**

```python
def get_reranker(settings: Settings) -> VoyageReranker | None:
    if not settings.VOYAGE_API_KEY:
        return None
    return VoyageReranker(api_key=settings.VOYAGE_API_KEY.get_secret_value())


# In retrieval pipeline:
candidates = await retrieve_hybrid(db, query, lesson_id, top_k=150)
reranker = get_reranker(settings)
if reranker is not None:
    indices_scored = await reranker.rerank(
        query=query,
        documents=[c.content for c in candidates],
        top_k=20,
    )
    candidates = [candidates[i] for i, _score in indices_scored]
else:
    candidates = candidates[:20]
```

### Task 4.3 — User adds Voyage key

**Manual step (not subagent):**

User signs up at https://dash.voyageai.com (no card needed). Copies API key. Adds to `/root/co4029/backend/.env`:

```
VOYAGE_API_KEY=pa-xxxxxxxxxxxxxxxxxxxxxx
```

```bash
pm2 restart abridgeai-backend abridgeai-worker --update-env
```

### Task 4.4 — End-to-end smoke

**Step 1: Generate quiz**

Manually generate a quiz on lesson `46741c12` after Phase 4. Check logs for reranker call (e.g. log `voyage_rerank_completed top_k=20 input_count=150 latency_ms=NNN`).

**Step 2: Eyeball quality**

Are the questions tighter / more on-topic than Phase 3 baseline? If yes, we have validation. If not, increase `top_k` for retrieval (200) or tune RRF weights.

**Step 3: Commit Phase 4**

Branch: `feat/voyage-reranker`.

---

## Verification matrix (final smoke after all phases)

| Test | Expected after Phase 1 | After Phase 2 | After Phase 3 | After Phase 4 |
|------|------|------|------|------|
| Panel section count (lesson 46741c12, knob=4) | 11 | 11 | 11 | 11 |
| Panel section count (knob=auto) | 41 | 41 | 41 | 41 |
| Outline section title (auto mode) | semantic ("Definition and Nature of Data") | same | same | same |
| Quiz quality | unchanged baseline | improved (subjective eyeball) | improved | best |
| Logs — reranker call | n/a | n/a | n/a | one per generation when key set |

## Rollback plan

Each phase ships behind its own git branch. To roll back any phase:

```bash
cd /root/co4029/backend
git revert -m 1 <merge-commit-sha>  # for Phase 1, 2, 4
# Phase 3 also needs:
uv run --no-sync alembic downgrade -1
git push origin master
pm2 restart abridgeai-backend abridgeai-worker --update-env
```

## Cost summary

- Phase 1: $0
- Phase 2: $0 (Stage C cache hits on re-process; only embedding tokens, marginal)
- Phase 3: $0 (Postgres native)
- Phase 4: $0 (200M Voyage free tokens — at ~21k tokens/quiz that's 9.5M free quizzes)

Total: $0. Free tier covers far beyond co4029's expected scale.
