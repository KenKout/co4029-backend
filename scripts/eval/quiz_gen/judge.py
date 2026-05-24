"""Score generated quiz questions with gpt-5-chat-latest as judge.

Usage:
  python judge.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, UTC

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
DATABASE_URL = os.environ["DATABASE_URL"].replace("+asyncpg", "").replace("+psycopg", "")
JUDGE_MODEL = "gpt-5-chat-latest"

VERSIONS = [
    "e2f47693-3679-47b6-a84e-fbc4c99a5052",  # Ch.1
    "39806190-fe7f-415a-ba6d-3a96c6ec41c0",  # Ch.2
]


JUDGE_SYSTEM = """You are an expert evaluator of educational multiple-choice quiz questions. Score each question on four dimensions, using ONLY the provided source chunks as ground truth. Be strict — do not credit questions that require external knowledge. Respond with valid JSON only, no prose."""


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

Score these dimensions and respond with JSON of EXACTLY this shape:
{{
  "groundedness": "yes" | "partial" | "no",
  "groundedness_reason": "one short sentence",
  "answer_correctness": "yes" | "no" | "unclear",
  "answer_correctness_reason": "one short sentence",
  "distractor_plausibility": 1 | 2 | 3 | 4 | 5,
  "distractor_reason": "one short sentence",
  "schema_valid": true | false,
  "bloom_level": "remember" | "understand" | "apply" | "analyze" | "evaluate" | "create"
}}

Definitions:
- groundedness: "yes" = answer directly entailed by source. "partial" = requires reasonable inference but consistent. "no" = contradicts source or unsupported.
- answer_correctness: is the marked-correct option actually correct given the source.
- distractor_plausibility 1=nonsense, 5=all distractors are defensibly-wrong but clearly distinguishable from correct.
- schema_valid: exactly 4 options, exactly 1 correct, well-formed."""


def load_all_chunks() -> dict[str, list[str]]:
    """Map version_id → list of chunk content strings."""
    out: dict[str, list[str]] = {}
    with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        for v in VERSIONS:
            cur.execute(
                "SELECT chunk_index, content FROM document_chunks "
                "WHERE material_version_id = %s ORDER BY chunk_index",
                (v,),
            )
            rows = cur.fetchall()
            out[v] = [c for _, c in rows]
    return out


# Lesson → material version
LESSON_TO_VERSION = {
    "46741c12-ff9d-48ac-9804-24753f6386eb": "e2f47693-3679-47b6-a84e-fbc4c99a5052",
    "db69e743-d24f-47b6-be6e-946628a542e7": "39806190-fe7f-415a-ba6d-3a96c6ec41c0",
}


async def judge_one(
    client: AsyncOpenAI,
    question: dict,
    chunks_by_lesson: dict[str, list[str]],
) -> dict:
    lesson_id = question["lesson_id"]
    version_id = LESSON_TO_VERSION[lesson_id]
    all_chunks = chunks_by_lesson[version_id]

    # Use cited chunk indices if available, else all chunks
    cited = question.get("source_chunk_indices") or list(range(len(all_chunks)))
    cited = [i for i in cited if 0 <= i < len(all_chunks)][:8]  # cap at 8 chunks
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
    err = None
    scores = None
    usage = None
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=512,
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
    t1 = time.perf_counter()

    return {
        "scores": scores,
        "judge_latency_s": t1 - t0,
        "judge_prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "judge_completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "judge_error": err,
    }


async def run_all():
    chunks_by_lesson = load_all_chunks()
    print(f"Loaded chunks for {len(chunks_by_lesson)} versions", flush=True)

    # Find all generated jsonl files
    gen_files = sorted(DATA_DIR.glob("generated_*_run*.jsonl"))
    print(f"Found {len(gen_files)} generation files", flush=True)

    # Flatten all questions
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
        print("No questions found — skipping judge.")
        return

    client = AsyncOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        timeout=httpx.Timeout(120.0, connect=10.0),
        max_retries=2,
    )

    sem = asyncio.Semaphore(12)
    counter = {"done": 0, "err": 0}

    async def bounded(q):
        async with sem:
            res = await judge_one(client, q, chunks_by_lesson)
            counter["done"] += 1
            if res.get("judge_error"):
                counter["err"] += 1
            if counter["done"] % 50 == 0:
                print(f"  judged {counter['done']}/{len(all_q)} ({counter['err']} errors)", flush=True)
            return {**q, **res}

    t0 = time.perf_counter()
    judged = await asyncio.gather(*[bounded(q) for q in all_q])
    t1 = time.perf_counter()

    # Group by (model, run) and write
    by_run: dict[tuple, list] = {}
    for j in judged:
        by_run.setdefault((j["model"], j["run"]), []).append(j)

    for (model, run_idx), items in by_run.items():
        safe = model.replace("/", "_").replace(":", "_")
        path = DATA_DIR / f"judged_{safe}_run{run_idx}.jsonl"
        with path.open("w") as f:
            for j in items:
                f.write(json.dumps(j) + "\n")

    print(f"\nJudged {len(judged)} questions in {t1-t0:.1f}s, {counter['err']} errors")


if __name__ == "__main__":
    asyncio.run(run_all())
