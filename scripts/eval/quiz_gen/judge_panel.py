"""Panel-of-LLMs judge — PoLL aggregation with reference-guided CoT prompts.

Methodology backed by:
- Zheng et al. 2023, "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
  (arXiv:2306.05685): reference-guided grading reduced GPT-4 judge failure rate
  on 10 math questions × 2 swap orders from 14/20 (default prompt) → 6/20 (CoT)
  → 3/20 (reference-guided), §3.4 Table 4. Zheng et al. also document explicit
  anti-position-bias and anti-verbosity instructions in their default prompts
  (Figures 5, 6, 7) and a 1–10 single-answer scale with strict
  "Rating: [[score]]" output format (Figure 6).
- Li et al. 2024, "LLMs-as-Judges: A Comprehensive Survey" (arXiv:2412.05579):
  §4.2.2 Aggregation documents Panel-of-LLM-judges (PoLL) — "using a diverse
  panel of smaller models as judges through max voting and average pooling is
  not only an effective method ... but also reduces intra-model bias of a
  single large model" — and §6.2 lists Cohen's Kappa as the standard
  inter-judge agreement metric. §7.1 catalogues position, verbosity, and
  self-enhancement biases that motivate explicit anti-bias instructions
  and a multi-model panel.

Output: data/panel_judged_<model>_run<i>.jsonl — one record per question with
all per-judge raw scores plus aggregated panel verdict.

Usage:
  python judge_panel.py
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean

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

# Diverse panel — different model families (PoLL principle, Li et al. §4.2.2):
# OpenAI, Anthropic, Meta. Kept distinct from any of the candidate generator
# models (gpt-oss-120b, gemma-4-31b-it, meta-llama/llama-3.3-70b:free) to
# guard against the self-enhancement bias documented in Zheng et al. §3.3.
JUDGE_PANEL = [
    "gpt-5-chat-latest",
    "claude-opus-4-7",
    "meta/llama-3.3-70b-instruct",
]

VERSIONS = [
    "e2f47693-3679-47b6-a84e-fbc4c99a5052",  # Ch.1
    "39806190-fe7f-415a-ba6d-3a96c6ec41c0",  # Ch.2
]

LESSON_TO_VERSION = {
    "46741c12-ff9d-48ac-9804-24753f6386eb": "e2f47693-3679-47b6-a84e-fbc4c99a5052",
    "db69e743-d24f-47b6-be6e-946628a542e7": "39806190-fe7f-415a-ba6d-3a96c6ec41c0",
}


# Reference-guided CoT prompt. Anti-bias guardrails are taken verbatim in
# spirit from Zheng et al. (arXiv:2306.05685) Figures 5–7: the judge is told
# to (a) ignore length, (b) avoid favoring any name/position, (c) solve the
# question independently before grading, and (d) emit a strict structured
# verdict. The 1–10 scale on a single-answer task mirrors Zheng et al.
# Figure 6, while the rubric criteria (groundedness / correctness /
# distractor plausibility / schema) are domain-specific to MCQ generation
# but kept ordinal where Zheng et al. used ordinal scales.
JUDGE_SYSTEM = """You are an impartial expert judge of educational multiple-choice quiz questions.

Anti-bias guardrails (Zheng et al., 2023, arXiv:2306.05685, Fig. 5–7):
- Do NOT let the length of the question, options, or explanation influence your evaluation.
- Do NOT favor questions that name or quote the source verbatim more than ones that paraphrase.
- Do NOT credit questions that require knowledge outside the provided source chunks.
- Be as objective as possible. Use ONLY the source chunks below as ground truth.

Reference-guided procedure (Zheng et al., 2023, §3.4 — failure rate 14/20 → 3/20):
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

First, work through the question yourself using only the source chunks. Decide
which option is correct BEFORE looking at the marked-correct field. Then score:

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
- answer_correctness: is the marked-correct option the one your independent solve picked.
- distractor_plausibility: 1 nonsense; 5 all distractors are defensibly-wrong but distinguishable.
- overall_quality: 1–10 single-answer grade (Zheng et al. Fig. 6 scale) that fuses the above with pedagogical value.
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
    judge_model: str,
    question: dict,
    chunks_by_version: dict[str, list[str]],
) -> dict:
    """Run a single judge model on a single question."""
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
    err = None
    scores: dict | None = None
    usage = None
    raw_text: str | None = None
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,  # deterministic decoding for judges
                max_tokens=768,
                response_format={"type": "json_object"},
            )
            raw_text = response.choices[0].message.content or "{}"
            scores = json.loads(raw_text)
            usage = response.usage
            err = None
            break
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            # Some judges (Claude on this gateway) reject response_format —
            # fall back to plain JSON extraction with a regex.
            if "response_format" in str(e).lower() or "json_object" in str(e).lower():
                try:
                    response = await client.chat.completions.create(
                        model=judge_model,
                        messages=[
                            {"role": "system", "content": JUDGE_SYSTEM},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.0,
                        max_tokens=768,
                    )
                    raw_text = response.choices[0].message.content or ""
                    m = re.search(r"\{.*\}", raw_text, re.DOTALL)
                    if m:
                        scores = json.loads(m.group())
                        usage = response.usage
                        err = None
                        break
                except Exception as e2:
                    err = f"{type(e2).__name__}: {e2}"
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    dt = time.perf_counter() - t0
    return {
        "judge_model": judge_model,
        "scores": scores,
        "judge_latency_s": dt,
        "judge_prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "judge_completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "judge_error": err,
    }


def _majority(values: list) -> object:
    """Most-common value with stable tiebreak (alphabetical first)."""
    if not values:
        return None
    c = Counter(values)
    top = max(c.values())
    winners = sorted([k for k, v in c.items() if v == top], key=lambda x: str(x))
    return winners[0]


def aggregate_panel(per_judge: list[dict]) -> dict:
    """PoLL aggregation: majority vote (categorical) + mean (ordinal).

    Cited from Li et al. 2024 (arXiv:2412.05579) §4.2.2 Aggregation —
    "majority vote, weighted averages... allow each model to assess without
    interference, and eventually extract and combine the most effective
    elements from each model's response".
    """
    valid = [j for j in per_judge if j.get("scores") and not j.get("judge_error")]
    if not valid:
        return {
            "panel_groundedness": None,
            "panel_correctness": None,
            "panel_distractor": None,
            "panel_overall": None,
            "panel_schema_valid": None,
            "panel_bloom_level": None,
            "panel_consensus": None,
            "n_judges_valid": 0,
        }

    cat = lambda key: [j["scores"].get(key) for j in valid if j["scores"].get(key) is not None]  # noqa: E731
    ord_vals = lambda key: [j["scores"].get(key) for j in valid if isinstance(j["scores"].get(key), (int, float))]  # noqa: E731

    panel_ground = _majority(cat("groundedness"))
    panel_correct = _majority(cat("answer_correctness"))
    panel_schema = _majority(cat("schema_valid"))
    panel_bloom = _majority(cat("bloom_level"))
    distractor = ord_vals("distractor_plausibility")
    overall = ord_vals("overall_quality")

    # Consensus rate = fraction of judges that match the majority on the
    # primary correctness label. Closer to 1.0 = more reliable verdict.
    consensus_match = [
        1.0
        for j in valid
        if j["scores"].get("answer_correctness") == panel_correct
    ]
    consensus = sum(consensus_match) / len(valid) if valid else None

    return {
        "panel_groundedness": panel_ground,
        "panel_correctness": panel_correct,
        "panel_distractor": round(mean(distractor), 3) if distractor else None,
        "panel_overall": round(mean(overall), 3) if overall else None,
        "panel_schema_valid": panel_schema,
        "panel_bloom_level": panel_bloom,
        "panel_consensus": round(consensus, 3) if consensus is not None else None,
        "n_judges_valid": len(valid),
    }


async def run_all() -> None:
    chunks_by_version = load_all_chunks()
    print(f"Loaded chunks for {len(chunks_by_version)} versions", flush=True)

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
    print(f"Judges in panel: {JUDGE_PANEL}", flush=True)
    print(f"Total judge calls expected: {len(all_q) * len(JUDGE_PANEL)}", flush=True)

    if not all_q:
        return

    client = AsyncOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        timeout=httpx.Timeout(180.0, connect=10.0),
        max_retries=2,
    )

    # Concurrency: 12 simultaneous judge calls — same as legacy single-judge
    # cap. Each question expands to len(JUDGE_PANEL) calls.
    sem = asyncio.Semaphore(12)
    counter = {"done": 0, "err": 0, "total": len(all_q) * len(JUDGE_PANEL)}

    async def call_one(q: dict, jm: str) -> dict:
        async with sem:
            res = await judge_one(client, jm, q, chunks_by_version)
            counter["done"] += 1
            if res.get("judge_error"):
                counter["err"] += 1
            if counter["done"] % 50 == 0:
                print(
                    f"  progress: {counter['done']}/{counter['total']} "
                    f"({counter['err']} errors)",
                    flush=True,
                )
            return {"q_ref": id(q), "judge": res}

    t0 = time.perf_counter()
    judge_tasks = [call_one(q, jm) for q in all_q for jm in JUDGE_PANEL]
    judge_results = await asyncio.gather(*judge_tasks)
    dt = time.perf_counter() - t0

    # Group judge calls back per question (by id())
    by_q: dict[int, list[dict]] = {}
    for jr in judge_results:
        by_q.setdefault(jr["q_ref"], []).append(jr["judge"])

    judged: list[dict] = []
    for q in all_q:
        per_judge = by_q.get(id(q), [])
        agg = aggregate_panel(per_judge)
        judged.append({**q, "per_judge": per_judge, "panel": agg})

    # Group by (model, run) and write
    by_run: dict[tuple, list] = {}
    for j in judged:
        by_run.setdefault((j["model"], j["run"]), []).append(j)

    for (model, run_idx), items in by_run.items():
        safe = model.replace("/", "_").replace(":", "_")
        path = DATA_DIR / f"panel_judged_{safe}_run{run_idx}.jsonl"
        with path.open("w") as f:
            for j in items:
                f.write(json.dumps(j) + "\n")

    print(
        f"\nPanel-judged {len(judged)} questions × {len(JUDGE_PANEL)} judges "
        f"= {counter['done']} calls in {dt:.1f}s, {counter['err']} errors"
    )


if __name__ == "__main__":
    asyncio.run(run_all())
