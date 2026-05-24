"""Regenerate ONLY (model, run, lesson) cells that produced 0 questions.

Reads existing data/generated_*_run*.jsonl, finds rows with empty `questions`,
re-runs them through generate_one() (which now includes the JSON-repair
fallback), and patches those rows in place. Good data is left untouched.

Citations:
- Li et al. 2024 (arXiv:2412.05579) §4.1.3 Post-processing — text reprocessing
  / structural repair as a documented mitigation for output-format brittleness.
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
from openai import AsyncOpenAI

# Reuse helpers + constants from generate.py
sys.path.insert(0, str(Path(__file__).parent))
from generate import (  # noqa: E402
    DATA_DIR,
    LESSONS,
    QUESTIONS_PER_LESSON,
    LLM_BASE_URL,
    LLM_API_KEY,
    fetch_chunks,
    generate_one,
)


def find_failed_cells() -> list[tuple[str, int, str]]:
    """Return list of (model, run, lesson_id) cells with 0 questions."""
    failed: list[tuple[str, int, str]] = []
    for f in sorted(DATA_DIR.glob("generated_*_run*.jsonl")):
        for line in f.open():
            rec = json.loads(line)
            if not rec.get("questions"):
                failed.append((rec["model"], rec["run"], rec["lesson_id"]))
    return failed


def patch_row(model: str, run_idx: int, new_record: dict) -> None:
    """Replace the matching row in the per-(model, run) jsonl file."""
    safe = model.replace("/", "_").replace(":", "_")
    path = DATA_DIR / f"generated_{safe}_run{run_idx}.jsonl"
    rows: list[dict] = [json.loads(line) for line in path.open()]
    for i, r in enumerate(rows):
        if r["lesson_id"] == new_record["lesson_id"] and r["run"] == run_idx:
            rows[i] = new_record
            break
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def patch_latency_row(model: str, run_idx: int, new_lat: dict) -> None:
    safe = model.replace("/", "_").replace(":", "_")
    path = DATA_DIR / f"latency_{safe}_run{run_idx}.jsonl"
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.open()]
    for i, r in enumerate(rows):
        if r["lesson_id"] == new_lat["lesson_id"] and r["run"] == run_idx:
            rows[i] = new_lat
            break
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


async def run() -> None:
    failed = find_failed_cells()
    print(f"Failed cells found: {len(failed)}")
    for f in failed:
        print(f"  {f}")
    if not failed:
        print("Nothing to regenerate.")
        return

    lessons_by_id = {l["lesson_id"]: l for l in LESSONS}
    chunks_by_version = {l["lesson_id"]: fetch_chunks(l["version_id"]) for l in LESSONS}

    client = AsyncOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        timeout=httpx.Timeout(240.0, connect=10.0),
        max_retries=2,
    )

    sem = asyncio.Semaphore(4)

    async def worker(model: str, run_idx: int, lesson_id: str):
        async with sem:
            lesson = lessons_by_id[lesson_id]
            chunks = chunks_by_version[lesson_id]
            print(f"  regen: {model} run{run_idx} {lesson['title'][:30]}", flush=True)
            t0 = time.perf_counter()
            output, latency = await generate_one(client, model, lesson, chunks, run_idx)
            dt = time.perf_counter() - t0
            n = len(output.get("questions", []))
            err = output.get("error")
            print(f"  done ({dt:5.1f}s): {n}q, err={err!r}", flush=True)
            patch_row(model, run_idx, output)
            patch_latency_row(model, run_idx, latency)
            return output

    results = await asyncio.gather(*[worker(*c) for c in failed])

    print("\n=== Regen summary ===")
    for r in results:
        print(f"  {r['model']} run{r['run']} {r['lesson_title']}: "
              f"{len(r.get('questions', []))} q, err={r.get('error')!r}")


if __name__ == "__main__":
    asyncio.run(run())
