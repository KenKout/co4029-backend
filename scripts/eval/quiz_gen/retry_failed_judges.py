"""Retry only the panel-judge calls that errored on the first pass.

Llama-3.3-70b on the gateway returns 429 under burst load — by retrying with
serialized concurrency (1 at a time) and 5s backoff we recover the panel to
3-judge coverage without re-running the 527 calls that already succeeded.

Usage:
  python retry_failed_judges.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import httpx
from openai import AsyncOpenAI

from judge_panel import (  # noqa: E402
    DATA_DIR,
    LLM_BASE_URL,
    LLM_API_KEY,
    JUDGE_PANEL,
    aggregate_panel,
    judge_one,
    load_all_chunks,
)


async def run() -> None:
    chunks = load_all_chunks()
    files = sorted(DATA_DIR.glob("panel_judged_*_run*.jsonl"))
    print(f"Found {len(files)} panel-judge files")

    # Identify all (file, line_idx, judge_idx) triples that need retry
    retry_targets = []
    for f in files:
        rows = [json.loads(line) for line in f.open()]
        for row_idx, row in enumerate(rows):
            for j_idx, j in enumerate(row.get("per_judge", [])):
                if j.get("judge_error"):
                    retry_targets.append((f, row_idx, j_idx, j.get("judge_model")))

    if not retry_targets:
        print("Nothing to retry.")
        return

    from collections import Counter
    print(f"Retry targets: {len(retry_targets)}")
    print(f"  by judge: {dict(Counter(t[3] for t in retry_targets))}")

    client = AsyncOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        timeout=httpx.Timeout(45.0, connect=10.0),
        max_retries=2,
    )

    # Reload as fresh (file -> rows) cache so we can mutate + write once at end
    cache: dict[Path, list[dict]] = {f: [json.loads(line) for line in f.open()] for f in files}

    # Light parallel: 2 calls in flight, 0.5s pacing per dispatch — gentle
    # enough to avoid retripping the gateway 429 we hit on the burst pass,
    # while finishing 136 retries in ~3 minutes instead of ~5.
    sem = asyncio.Semaphore(2)
    counter = {"done": 0, "ok": 0, "still_err": 0}
    t0 = time.perf_counter()

    async def worker(f: Path, row_idx: int, j_idx: int, jm: str) -> None:
        async with sem:
            row = cache[f][row_idx]
            q = {
                "prompt_text": row.get("prompt_text"),
                "options": row.get("options"),
                "explanation": row.get("explanation"),
                "source_chunk_indices": row.get("source_chunk_indices"),
                "lesson_id": row["lesson_id"],
            }
            new_j = await judge_one(client, jm, q, chunks)
            row["per_judge"][j_idx] = new_j
            row["panel"] = aggregate_panel(row["per_judge"])
            counter["done"] += 1
            if new_j.get("judge_error"):
                counter["still_err"] += 1
            else:
                counter["ok"] += 1
            if counter["done"] % 20 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"  retried {counter['done']}/{len(retry_targets)} "
                    f"(ok={counter['ok']}, still_err={counter['still_err']}, "
                    f"elapsed={elapsed:.1f}s)",
                    flush=True,
                )

    async def dispatcher() -> None:
        tasks = []
        for f, row_idx, j_idx, jm in retry_targets:
            tasks.append(asyncio.create_task(worker(f, row_idx, j_idx, jm)))
            await asyncio.sleep(0.5)
        await asyncio.gather(*tasks)

    await dispatcher()

    # Write back
    for f, rows in cache.items():
        with f.open("w") as out:
            for r in rows:
                out.write(json.dumps(r) + "\n")

    print(
        f"\nFinished {counter['done']} retries: "
        f"ok={counter['ok']}, still_err={counter['still_err']}"
    )


if __name__ == "__main__":
    asyncio.run(run())
