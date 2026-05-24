"""Generate quiz questions via direct LLM calls — model isolation eval.

Usage:
  python generate.py
  # Runs 3 models x 3 runs x 2 lessons in parallel
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, UTC
from uuid import UUID

# Allow running from script location
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import httpx
from openai import AsyncOpenAI

# Hand-rolled DB connection (sync psycopg) — avoid pulling in full app stack
import psycopg

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

LESSONS = [
    {
        "lesson_id": "46741c12-ff9d-48ac-9804-24753f6386eb",
        "version_id": "e2f47693-3679-47b6-a84e-fbc4c99a5052",
        "title": "Chapter 1 - Overview",
    },
    {
        "lesson_id": "db69e743-d24f-47b6-be6e-946628a542e7",
        "version_id": "39806190-fe7f-415a-ba6d-3a96c6ec41c0",
        "title": "Chapter 2 - Basic Issues in Data Warehousing",
    },
]

MODELS = [
    "gpt-oss-120b",
    "gemma-4-31b-it",
    "meta-llama/llama-3.3-70b-instruct:free",
]

QUESTIONS_PER_LESSON = 12
RUNS_PER_MODEL = 3

LLM_BASE_URL = os.environ["LLM_BASE_URL"].rstrip("/")
LLM_API_KEY = os.environ["LLM_API_KEY"]
DATABASE_URL = os.environ["DATABASE_URL"].replace("+asyncpg", "").replace("+psycopg", "")


SYSTEM_PROMPT = """You are an expert exam author. Given source course material, write multiple-choice quiz questions that test deep understanding of the content.

Requirements:
- Each question MUST be answerable from the source material alone.
- Exactly 4 options per question, exactly 1 correct.
- Distractors must be plausible to a learner who has only skimmed the material — not nonsense.
- Vary Bloom's levels: include remember, understand, apply, and analyze.
- Provide a 1-2 sentence explanation citing the relevant fact.
- Output strict JSON conforming to the schema provided. No prose."""


USER_TEMPLATE = """Course: Data Warehouses and Decision Support Systems
Lesson: {lesson_title}

Source material (paginated chunks):
---
{chunks}
---

Generate exactly {n} multiple-choice questions covering the breadth of the source material above.

Respond with JSON of this exact shape:
{{
  "questions": [
    {{
      "prompt_text": "string",
      "options": [
        {{"key": "A", "text": "string", "is_correct": false}},
        {{"key": "B", "text": "string", "is_correct": false}},
        {{"key": "C", "text": "string", "is_correct": true}},
        {{"key": "D", "text": "string", "is_correct": false}}
      ],
      "explanation": "string",
      "bloom_level": "remember|understand|apply|analyze",
      "source_chunk_indices": [0, 3]
    }}
  ]
}}

Source chunk indices reference the order of chunks given above (0-indexed)."""


def fetch_chunks(version_id: str) -> list[str]:
    with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT content
            FROM document_chunks
            WHERE material_version_id = %s
            ORDER BY chunk_index
            """,
            (version_id,),
        )
        return [r[0] for r in cur.fetchall()]


async def generate_one(
    client: AsyncOpenAI,
    model: str,
    lesson: dict,
    chunks: list[str],
    run_idx: int,
) -> tuple[dict, dict]:
    """Returns (output_record, latency_record)."""
    chunks_block = "\n\n".join(f"[chunk {i}]\n{c}" for i, c in enumerate(chunks))
    prompt = USER_TEMPLATE.format(
        lesson_title=lesson["title"],
        chunks=chunks_block,
        n=QUESTIONS_PER_LESSON,
    )

    t0 = time.perf_counter()
    err = None
    response = None
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            top_p=0.95,
            max_tokens=6144,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        err = str(e)
    t1 = time.perf_counter()

    wall = t1 - t0

    if err or response is None:
        return (
            {
                "model": model,
                "run": run_idx,
                "lesson_id": lesson["lesson_id"],
                "lesson_title": lesson["title"],
                "error": err,
                "questions": [],
            },
            {
                "model": model,
                "run": run_idx,
                "lesson_id": lesson["lesson_id"],
                "wall_seconds": wall,
                "error": err,
                "ts": datetime.now(UTC).isoformat(),
            },
        )

    raw = response.choices[0].message.content
    try:
        parsed = json.loads(raw)
        questions = parsed.get("questions", [])
    except Exception as e:
        # JSON repair pass — common Gemma failure mode is malformed
        # `options` array (duplicate keys collapsed into one object).
        # Ask a stronger model to repair the JSON without changing content.
        # Cited as a documented mitigation for output-format brittleness in
        # Li et al. 2024 (arXiv:2412.05579) §4.1.3 "Post-processing".
        questions = []
        err = f"json_parse_error: {e}"
        try:
            repair_resp = await client.chat.completions.create(
                model="gpt-5-chat-latest",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict JSON repair tool. The user pastes broken JSON. "
                            "Return a SINGLE valid JSON object that preserves all factual content "
                            "(question prompts, option texts, explanations, indices). Fix only "
                            "structural issues: duplicate keys, missing commas, trailing commas, "
                            "broken array/object nesting. Do not invent or remove content. "
                            "Output JSON only, no prose."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Repair this JSON so it parses. Expected schema: "
                            '{"questions":[{"prompt_text":..,"options":[{"key":..,"text":..,"is_correct":..},...],'
                            '"explanation":..,"bloom_level":..,"source_chunk_indices":[..]}]}'
                            "\n\nBroken JSON:\n"
                            + (raw or "")
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=8192,
                response_format={"type": "json_object"},
            )
            fixed_raw = repair_resp.choices[0].message.content or "{}"
            parsed = json.loads(fixed_raw)
            questions = parsed.get("questions", [])
            if questions:
                err = f"json_parse_error_repaired: {e}"
        except Exception as repair_e:
            err = f"json_parse_error: {e}; repair_failed: {repair_e}"

    usage = response.usage
    return (
        {
            "model": model,
            "run": run_idx,
            "lesson_id": lesson["lesson_id"],
            "lesson_title": lesson["title"],
            "questions": questions,
            "raw_response": raw,
            "error": err,
        },
        {
            "model": model,
            "run": run_idx,
            "lesson_id": lesson["lesson_id"],
            "wall_seconds": wall,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "n_questions": len(questions),
            "ts": datetime.now(UTC).isoformat(),
        },
    )


async def run_all():
    # Pre-fetch chunks once per lesson
    print("Fetching chunks…", flush=True)
    chunks_by_lesson = {
        l["lesson_id"]: fetch_chunks(l["version_id"]) for l in LESSONS
    }
    for l in LESSONS:
        print(f"  {l['title']}: {len(chunks_by_lesson[l['lesson_id']])} chunks", flush=True)

    client = AsyncOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        timeout=httpx.Timeout(180.0, connect=10.0),
        max_retries=2,
    )

    # Build flat task list: 3 models × 3 runs × 2 lessons = 18 calls
    tasks = []
    for model in MODELS:
        for run_idx in range(1, RUNS_PER_MODEL + 1):
            for lesson in LESSONS:
                tasks.append((model, run_idx, lesson, chunks_by_lesson[lesson["lesson_id"]]))

    print(f"\nDispatching {len(tasks)} parallel calls…", flush=True)
    sem = asyncio.Semaphore(8)  # cap concurrency on gateway

    async def bounded(model, run_idx, lesson, chunks):
        async with sem:
            print(f"  start: {model} run{run_idx} {lesson['title'][:30]}", flush=True)
            t0 = time.perf_counter()
            result = await generate_one(client, model, lesson, chunks, run_idx)
            print(f"  done  ({time.perf_counter()-t0:5.1f}s): {model} run{run_idx} {lesson['title'][:30]}", flush=True)
            return result

    results = await asyncio.gather(
        *[bounded(*t) for t in tasks],
        return_exceptions=False,
    )

    # Group + write per (model, run)
    by_run: dict[tuple[str, int], list] = {}
    lat_by_run: dict[tuple[str, int], list] = {}
    for output, latency in results:
        key = (output["model"], output["run"])
        by_run.setdefault(key, []).append(output)
        lat_by_run.setdefault(key, []).append(latency)

    for (model, run_idx), outputs in by_run.items():
        safe = model.replace("/", "_").replace(":", "_")
        out_path = DATA_DIR / f"generated_{safe}_run{run_idx}.jsonl"
        lat_path = DATA_DIR / f"latency_{safe}_run{run_idx}.jsonl"
        with out_path.open("w") as f:
            for o in outputs:
                f.write(json.dumps(o) + "\n")
        with lat_path.open("w") as f:
            for l in lat_by_run[(model, run_idx)]:
                f.write(json.dumps(l) + "\n")

    # Summary
    print("\n=== Generation summary ===")
    for (model, run_idx), outputs in sorted(by_run.items()):
        n_q = sum(len(o["questions"]) for o in outputs)
        n_err = sum(1 for o in outputs if o["error"])
        print(f"  {model} run{run_idx}: {n_q} questions, {n_err} errors")


if __name__ == "__main__":
    asyncio.run(run_all())
