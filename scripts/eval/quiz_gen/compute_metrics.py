"""Compute aggregate metrics, latency percentiles, cost; emit LaTeX table.

Usage:
  python compute_metrics.py
"""
from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
REPORT_DIR = Path(__file__).parent / "report"
REPORT_DIR.mkdir(exist_ok=True)

# Pricing per 1M tokens
PRICING = {
    "gpt-oss-120b":                            {"in": 0.039, "out": 0.19},
    "gemma-4-31b-it":                          {"in": 0.12,  "out": 0.37},
    "meta-llama/llama-3.3-70b-instruct:free":  {"in": 0.10,  "out": 0.32},
    "gpt-5-chat-latest":                       {"in": 1.25,  "out": 10.0},
}

MODELS = ["gpt-oss-120b", "gemma-4-31b-it", "meta-llama/llama-3.3-70b-instruct:free"]


def safe(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def percentiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    s = sorted(values)
    n = len(s)
    def pct(p):
        idx = max(0, min(n - 1, int(p * n / 100)))
        return s[idx]
    return {
        "n": n,
        "mean": statistics.mean(s),
        "stdev": statistics.stdev(s) if n > 1 else 0.0,
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "max": s[-1],
        "min": s[0],
    }


def bootstrap_ci(values: list[float], n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(42)
    means = []
    for _ in range(n_boot):
        sample = [rng.choice(values) for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return (lo, hi)


def load_judged() -> dict[str, list[dict]]:
    """Returns {model: [judged_questions, ...]}"""
    out: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(DATA_DIR.glob("judged_*_run*.jsonl")):
        for line in f.open():
            j = json.loads(line)
            out[j["model"]].append(j)
    return dict(out)


def load_latency() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(DATA_DIR.glob("latency_*_run*.jsonl")):
        for line in f.open():
            l = json.loads(line)
            out[l["model"]].append(l)
    return dict(out)


def compute_quality(judged: dict[str, list[dict]]) -> dict:
    out = {}
    for model, qs in judged.items():
        # Skip questions with judge errors or missing scores
        valid = [q for q in qs if q.get("scores") and not q.get("judge_error")]

        grounded_yes = [1.0 if (q["scores"].get("groundedness") == "yes") else 0.0 for q in valid]
        grounded_partial_or_yes = [1.0 if q["scores"].get("groundedness") in ("yes", "partial") else 0.0 for q in valid]
        correct = [1.0 if q["scores"].get("answer_correctness") == "yes" else 0.0 for q in valid]
        plaus = [float(q["scores"].get("distractor_plausibility", 0)) for q in valid]
        overall = [float(q["scores"].get("overall_quality") or 0) for q in valid]
        schema = [1.0 if q["scores"].get("schema_valid") else 0.0 for q in valid]

        # Independent-solve agreement: fraction of questions where the judge's
        # Step-1 answer (judge_solved_key) matches the marked-correct option.
        # This is the reference-guided cross-check from Zheng et al. 2023
        # (arXiv:2306.05685, §3.4 Table 4) — judge solves first, then grades.
        solve_match = []
        for q in valid:
            jk = q["scores"].get("judge_solved_key")
            mk = next((o.get("key") for o in (q.get("options") or []) if o.get("is_correct")), None)
            if jk and mk:
                solve_match.append(1.0 if jk == mk else 0.0)

        bloom = defaultdict(int)
        for q in valid:
            bl = q["scores"].get("bloom_level", "unknown")
            bloom[bl] += 1

        out[model] = {
            "n_total": len(qs),
            "n_valid": len(valid),
            "n_judge_errors": len(qs) - len(valid),
            "groundedness_strict": {
                "rate": statistics.mean(grounded_yes) if grounded_yes else 0,
                "ci": bootstrap_ci(grounded_yes),
            },
            "groundedness_lenient": {
                "rate": statistics.mean(grounded_partial_or_yes) if grounded_partial_or_yes else 0,
                "ci": bootstrap_ci(grounded_partial_or_yes),
            },
            "correctness": {
                "rate": statistics.mean(correct) if correct else 0,
                "ci": bootstrap_ci(correct),
            },
            "plausibility": {
                "mean": statistics.mean(plaus) if plaus else 0,
                "ci": bootstrap_ci(plaus),
            },
            "overall_quality": {
                "mean": statistics.mean(overall) if overall else 0,
                "ci": bootstrap_ci(overall),
            },
            "judge_solve_agreement": {
                "rate": statistics.mean(solve_match) if solve_match else 0,
                "ci": bootstrap_ci(solve_match),
                "n": len(solve_match),
            },
            "schema_compliance": statistics.mean(schema) if schema else 0,
            "bloom_distribution": dict(bloom),
        }
    return out


def compute_latency(latency: dict[str, list[dict]]) -> dict:
    """Per-call wall time + per-question implied (wall / n_questions)."""
    out = {}
    for model, calls in latency.items():
        wall = [c["wall_seconds"] for c in calls if c.get("wall_seconds") and not c.get("error")]
        per_q = []
        for c in calls:
            if c.get("wall_seconds") and c.get("n_questions"):
                per_q.append(c["wall_seconds"] / c["n_questions"])
        out[model] = {
            "per_lesson_call": percentiles(wall),
            "per_question_implied": percentiles(per_q),
        }
    return out


def compute_cost(latency: dict[str, list[dict]], judged: dict[str, list[dict]]) -> dict:
    out = {}
    for model, calls in latency.items():
        total_in = sum((c.get("prompt_tokens") or 0) for c in calls)
        total_out = sum((c.get("completion_tokens") or 0) for c in calls)
        n_questions = sum(c.get("n_questions") or 0 for c in calls)
        pricing = PRICING.get(model, {"in": 0, "out": 0})
        gen_cost = (total_in * pricing["in"] + total_out * pricing["out"]) / 1_000_000
        cost_per_q = gen_cost / n_questions if n_questions else 0

        # Judge cost for this model's questions
        judge_in = sum(j.get("judge_prompt_tokens") or 0 for j in judged.get(model, []))
        judge_out = sum(j.get("judge_completion_tokens") or 0 for j in judged.get(model, []))
        judge_pricing = PRICING["gpt-5-chat-latest"]
        judge_cost = (judge_in * judge_pricing["in"] + judge_out * judge_pricing["out"]) / 1_000_000

        out[model] = {
            "gen_input_tokens": total_in,
            "gen_output_tokens": total_out,
            "gen_cost_total": gen_cost,
            "gen_cost_per_question": cost_per_q,
            "gen_cost_per_100q": cost_per_q * 100,
            "judge_input_tokens": judge_in,
            "judge_output_tokens": judge_out,
            "judge_cost_total": judge_cost,
        }
    return out


def emit_latex(quality: dict, latency: dict, cost: dict, path: Path):
    """Render comparison_table.tex"""
    # Friendly model names for the table
    NAMES = {
        "gpt-oss-120b": "gpt-oss-120b",
        "gemma-4-31b-it": "gemma-4-31b-it",
        "meta-llama/llama-3.3-70b-instruct:free": r"llama-3.3-70b",
    }

    def pct(x: float) -> str:
        return f"{x*100:.1f}\\%"

    def ci_str(ci: tuple) -> str:
        return f"[{ci[0]*100:.1f}, {ci[1]*100:.1f}]"

    rows = []
    for m in MODELS:
        if m not in quality:
            continue
        q = quality[m]
        l = latency.get(m, {}).get("per_question_implied", {})
        c = cost.get(m, {})
        rows.append((m, q, l, c))

    body_lines = []
    for m, q, l, c in rows:
        name = NAMES.get(m, m)
        body_lines.append(
            f"{name} & "
            f"{pct(q['groundedness_strict']['rate'])} {ci_str(q['groundedness_strict']['ci'])} & "
            f"{pct(q['correctness']['rate'])} {ci_str(q['correctness']['ci'])} & "
            f"{pct(q['judge_solve_agreement']['rate'])} & "
            f"{q['plausibility']['mean']:.2f} & "
            f"{q['overall_quality']['mean']:.2f} & "
            f"{pct(q['schema_compliance'])} & "
            f"{l.get('p50', 0):.1f} / {l.get('p99', 0):.1f} & "
            f"\\${c.get('gen_cost_per_100q', 0):.4f} \\\\"
        )

    table = r"""\begin{table}[htbp]
\centering
\caption{Quiz generation model comparison ($n=72$ questions/model, 3 runs $\times$ 2 lessons $\times$ 12 questions). Quality scores produced by a single \texttt{gpt-5-chat-latest} judge using a reference-guided chain-of-thought protocol — the judge first solves the question independently from the source chunks, then grades the candidate~\cite{zheng2023judging}. Anti-bias guardrails (length-blind, paraphrase-blind, source-only) follow the same paper's Figures~5--7. Confidence intervals are 95\% bootstrap.}
\label{tab:gen_model_eval}
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{|l|c|c|c|c|c|c|c|c|}
\hline
\textbf{Model} & \textbf{Grounded [95\% CI]} & \textbf{Correct [95\% CI]} & \textbf{Solve-Match} & \textbf{Plaus.} & \textbf{Overall (1--10)} & \textbf{Schema} & \textbf{p50/p99 (s/q)} & \textbf{\$/100q} \\
\hline
""" + "\n".join(body_lines) + r"""
\hline
\end{tabular}
\end{table}
"""
    path.write_text(table)


def emit_latency_table(latency: dict, path: Path):
    NAMES = {
        "gpt-oss-120b": "gpt-oss-120b",
        "gemma-4-31b-it": "gemma-4-31b-it",
        "meta-llama/llama-3.3-70b-instruct:free": r"llama-3.3-70b",
    }
    lines = []
    for m in MODELS:
        if m not in latency:
            continue
        l = latency[m]["per_lesson_call"]
        lines.append(
            f"{NAMES.get(m, m)} & {l['mean']:.1f} & {l['p50']:.1f} & {l['p90']:.1f} & {l['p99']:.1f} & {l['max']:.1f} \\\\"
        )

    table = r"""\begin{table}[htbp]
\centering
\caption{Per-call generation latency (seconds, full lesson generation = 12 questions)}
\label{tab:gen_latency}
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{|l|c|c|c|c|c|}
\hline
\textbf{Model} & \textbf{mean} & \textbf{p50} & \textbf{p90} & \textbf{p99} & \textbf{max} \\
\hline
""" + "\n".join(lines) + r"""
\hline
\end{tabular}
\end{table}
"""
    path.write_text(table)


def main():
    print("Loading judged data…")
    judged = load_judged()
    print(f"  {sum(len(v) for v in judged.values())} judged questions across {len(judged)} models")

    print("Loading latency data…")
    latency = load_latency()
    print(f"  {sum(len(v) for v in latency.values())} latency records")

    print("\nComputing quality metrics…")
    quality = compute_quality(judged)

    print("Computing latency percentiles…")
    lat = compute_latency(latency)

    print("Computing cost…")
    cost = compute_cost(latency, judged)

    # Save aggregate JSON
    aggregate = {"quality": quality, "latency": lat, "cost": cost}
    (DATA_DIR / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2))

    # Console summary
    print("\n=== Quality Summary ===")
    for m in MODELS:
        if m not in quality:
            continue
        q = quality[m]
        print(f"\n{m} (n={q['n_valid']}):")
        print(f"  groundedness (strict): {q['groundedness_strict']['rate']*100:.1f}% [{q['groundedness_strict']['ci'][0]*100:.1f}, {q['groundedness_strict']['ci'][1]*100:.1f}]")
        print(f"  correctness:           {q['correctness']['rate']*100:.1f}% [{q['correctness']['ci'][0]*100:.1f}, {q['correctness']['ci'][1]*100:.1f}]")
        print(f"  judge solve match:     {q['judge_solve_agreement']['rate']*100:.1f}% [{q['judge_solve_agreement']['ci'][0]*100:.1f}, {q['judge_solve_agreement']['ci'][1]*100:.1f}]  (n={q['judge_solve_agreement']['n']})")
        print(f"  plausibility (1-5):    {q['plausibility']['mean']:.2f} [{q['plausibility']['ci'][0]:.2f}, {q['plausibility']['ci'][1]:.2f}]")
        print(f"  overall (1-10):        {q['overall_quality']['mean']:.2f} [{q['overall_quality']['ci'][0]:.2f}, {q['overall_quality']['ci'][1]:.2f}]")
        print(f"  schema compliance:     {q['schema_compliance']*100:.1f}%")
        print(f"  bloom: {q['bloom_distribution']}")

    print("\n=== Latency (per lesson, seconds) ===")
    for m in MODELS:
        if m not in lat:
            continue
        l = lat[m]["per_lesson_call"]
        print(f"  {m}: mean={l.get('mean', 0):.1f}, p50={l.get('p50', 0):.1f}, p90={l.get('p90', 0):.1f}, p99={l.get('p99', 0):.1f}, max={l.get('max', 0):.1f}")

    print("\n=== Cost ===")
    for m in MODELS:
        if m not in cost:
            continue
        c = cost[m]
        print(f"  {m}: ${c['gen_cost_per_100q']:.4f}/100q (gen) + ${c['judge_cost_total']:.2f} judge")

    # LaTeX
    emit_latex(quality, lat, cost, REPORT_DIR / "comparison_table.tex")
    emit_latency_table(lat, REPORT_DIR / "latency_table.tex")
    print(f"\nLaTeX tables written to {REPORT_DIR}/")


if __name__ == "__main__":
    main()
