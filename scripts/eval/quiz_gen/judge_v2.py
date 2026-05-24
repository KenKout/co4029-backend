"""Improved single-judge LLM-as-a-judge — reference-guided CoT prompting.

Single judge: gpt-5-chat-latest. Improvements over judge.py are entirely on
the prompting side, drawn from Zheng et al. 2023 ("Judging LLM-as-a-Judge
with MT-Bench and Chatbot Arena", arXiv:2306.05685):

- Reference-guided procedure (§3.4 + Figure 8/10): the judge first solves
  the question independently using only the source chunks, then compares
  with the marked-correct option. Zheng et al. report this dropped GPT-4
  judge's failure rate on 10 math questions × 2 swap orders from 14/20
  (default prompt) → 6/20 (chain-of-thought) → 3/20 (reference-guided).

- Explicit anti-bias guardrails (Figures 5–7): tell the judge NOT to let
  length, naming, or position influence the verdict. Zheng et al. document
  position, verbosity, and self-enhancement biases in §3.3.

- Single-answer 1–10 scale with strict structured output (Figure 6) — for
  the new `overall_quality` field. The categorical fields keep the original
  rubric so existing consumers (compute_metrics.py) still work.

Usage:
  python judge_v2.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import httpx
import psycopg
from openai import AsyncOpenAI

DATA_DIR = Path(__file__).parent / "data"
LLM_BASE_URL = os.environ["LLM_BASE_URL"].rstrip("/")
LLM_API_KEY = os.environ["LLM_API_KEY"]
DATABASE_URL = (
    os.environ["DATABASE_URL"]
    .replace("+asyncpg", "")
    .replace("+psycopg", "")
)
JUDGE_MODEL = "gpt-5-chat-latest"

VERSIONS = [
    "e2f47693-3679-47b6-a84e-fbc4c99a5052",  # Ch.1
    "39806190-fe7f-415a-ba6d-3a96c6ec41c0",  # Ch.2
]

LESSON_TO_VERSION = {
    "46741c12-ff9d-48ac-9804-24753f6386eb": "e2f47693-3679-47b6-a84e-fbc4c99a5052",
    "db69e743-d24f-47b6-be6e-946628a542e7": "39806190-fe7f-415a-ba6d-3a96c6ec41c0",
}


JUDGE_SYSTEM = """You are an impartial expert judge of educational multiple-choice quiz questions.

Anti-bias guardrails (Zheng et al., 2023, arXiv:2306.05685, Figures 5-7):
- Do NOT let the length of the question, options, or explanation influence your evaluation.
- Do NOT favor questions that quote the source verbatim more than ones that paraphrase.
- Do NOT credit questions that require knowledge outside the provided source chunks.
- Be as objective as possible. Use ONLY the source chunks below as ground truth.

Reference-guided procedure (Zheng et al., 2023, §3.4 — Table 4 reports
failure rate 14/20 with default prompt -> 3/20 with reference-guided):
1. First, INDEPENDENTLY identify which option is correct using only the source chunks.
2. THEN compare your answer with the marked-correct option and the author's explanation.
3. Only after that, fill in the rubric.

Respond with ONE valid JSON object, no prose outside it."""


JUDGE_USER_TEMPLATE = """Source chunks (the only ground truth):
---
{chunks}
---

QUESTION:
{prompt}

OPTIONS:
{options_block}

Marked correct: {correct_key}
Author's explanation: {explanation}

Step 1 (reference-guided, Zheng et al. 2023 §3.4): work through the question
yourself using only the source chunks. Decide which option is correct BEFORE
looking at the marked-correct field. Step 2: compare your independent answer
with the marked-correct option. Step 3: fill the rubric below.

{{
  "judge_solved_key": "A" | "B" | "C" | "D" | "unsure",
  "judge_reasoning": "1-2 sentences naming the chunk(s) that decide it",
  "groundedness": "yes" | "partial" | "no",
  "groundedness_reason": "one sentence",
  "answer_correctness": "yes" | "no" | "unclear",
  "answer_correctness_reason": "one sentence",
  "distractor_plausibility": 1 | 2 | 3 | 4 | 5,
  "distractor_reason": "one sentence",
  "overall_quality": 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10,
  "schema_valid": true | false,
  "bloom_level": "remember" | "understand" | "apply" | "analyze" | "evaluate" | "create"
}}

Rubric:
- groundedness: "yes" = answer entailed by source. "partial" = consistent inference. "no" = unsupported.
- answer_correctness: is the marked-correct option the same as the one you picked in Step 1.
- distractor_plausibility 1=nonsense, 5=all distractors are defensibly-wrong but distinguishable.
- overall_quality: 1-10 single-answer grade (Zheng et al. Figure 6 scale)
  fusing groundedness + correctness + distractor quality + pedagogical value.
- schema_valid: exactly 4 options, exactly 1 correct, well-formed."""


def load_all_chunks() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        for v in VERSIONS:
            cur.execute(
                "SELECT chunk_index, content FROM document_chunks "
                "WHERE material_version_id = %s ORDER BY chunk_index",
                (v,),
            )
            out[v] = [c for _, c in cur.fetchall()]
    return out


async def judge_one(
    client: AsyncOpenAI,
    question: dict,
    chunks_by_version: dict[str, list[str]],
) -> dict:
    lesson_id = question["lesson_id"]
    version_id = LESSON_TO_VERSION[lesson_id]
    all_chunks = chunks_by_version[version_id]

    cited = question.get("source_chunk_indices") or list(range(len(all_chunks)))
    cited = [i for i in cited if 0 <= i < len(all_chunks)][:8]
    if not cited:
        cited = list(range(min(8, len(all_chunks))))
    chunks_text = "\n\n".join(f"[chunk {i}]\n{all_chunks[i]}" for i in cited)

    options = question.get("options", []) or []
    options_block = "\n".join(
        f"  {o.get('key', '?')}. {o.get('text', '')}" for o in options
    )
    correct = next(
        (o.get("key") for o in options if o.get("is_correct")),
        "?",
    )

    prompt = JUDGE_USER_TEMPLATE.format(
        chunks=chunks_text,
        prompt=question.get("prompt_text", ""),
        options_block=options_block,
        correct_key=correct,
        explanation=question.get("explanation", "(none)"),
    )

    t0 = time.perf_counter()
    err: str | None = None
    scores: dict | None = None
    usage = None
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,  # greedy decoding for determinism (judge stability)
                max_tokens=768,
                response_format={"type": "json_object"},
            )
            scores = json.loads(response.choices[0].message.content or "{}")
            usage = response.usage
            err = None
            break
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    dt = time.perf_counter() - t0
    return {
        "scores": scores,
        "judge_latency_s": dt,
        "judge_prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "judge_completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "judge_error": err,
    }


async def run_all() -> None:
    chunks = load_all_chunks()
    print(f"Loaded chunks for {len(chunks)} versions", flush=True)

    gen_files = sorted(DATA_DIR.glob("generated_*_run*.jsonl"))
    print(f"Found {len(gen_files)} generation files", flush=True)

    all_q: list[dict] = []
    for f in gen_files:
        for line in f.open():
            rec = json.loads(line)
            for q_idx, q in enumerate(rec.get("questions", [])):
                all_q.append({
                    **q,
                    "model": rec["model"],
                    "run": rec["run"],
                    "lesson_id": rec["lesson_id"],
                    "lesson_title": rec["lesson_title"],
                    "q_index": q_idx,
                })
    print(f"Total questions to judge: {len(all_q)}", flush=True)

    if not all_q:
        return

    client = AsyncOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        timeout=httpx.Timeout(120.0, connect=10.0),
        max_retries=2,
    )

    sem = asyncio.Semaphore(12)
    counter = {"done": 0, "err": 0}

    async def bounded(q: dict) -> dict:
        async with sem:
            res = await judge_one(client, q, chunks)
            counter["done"] += 1
            if res.get("judge_error"):
                counter["err"] += 1
            if counter["done"] % 50 == 0:
                print(f"  judged {counter['done']}/{len(all_q)} ({counter['err']} errors)", flush=True)
            return {**q, **res}

    t0 = time.perf_counter()
    judged = await asyncio.gather(*[bounded(q) for q in all_q])
    dt = time.perf_counter() - t0

    by_run: dict[tuple, list] = {}
    for j in judged:
        by_run.setdefault((j["model"], j["run"]), []).append(j)

    for (model, run_idx), items in by_run.items():
        safe = model.replace("/", "_").replace(":", "_")
        path = DATA_DIR / f"judged_{safe}_run{run_idx}.jsonl"
        with path.open("w") as f:
            for j in items:
                f.write(json.dumps(j) + "\n")

    print(f"\nJudged {len(judged)} questions in {dt:.1f}s, {counter['err']} errors")


if __name__ == "__main__":
    asyncio.run(run_all())
