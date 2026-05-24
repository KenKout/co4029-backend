"""Compute aggregate metrics from PoLL panel-judged data — v2.

Adds (vs. compute_metrics.py):
- Inter-judge agreement: pairwise Cohen's kappa per categorical dimension
  (Li et al. 2024 arXiv:2412.05579 §6.2 lists Cohen's Kappa as the standard
  meta-evaluation metric for LLM-as-a-judge agreement).
- Panel-aggregated quality scores (PoLL — Li et al. §4.2.2 Aggregation:
  majority vote categorical, mean ordinal).
- Per-judge breakdown so we can show how individual judges differ from the
  panel verdict (probes self-enhancement bias risk per Zheng et al. 2023
  arXiv:2306.05685 §3.3).
- Bootstrap 95% CI on panel rates.

Usage:
  python compute_metrics_v2.py
"""
from __future__ import annotations

import json
import random
import statistics
from collections import Counter, defaultdict
from itertools import combinations
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
    "claude-opus-4-7":                         {"in": 15.0,  "out": 75.0},
    "meta/llama-3.3-70b-instruct":             {"in": 0.10,  "out": 0.32},
}

MODELS = [
    "gpt-oss-120b",
    "gemma-4-31b-it",
    "meta-llama/llama-3.3-70b-instruct:free",
]

JUDGE_PANEL = [
    "gpt-5-chat-latest",
    "claude-opus-4-7",
    "meta/llama-3.3-70b-instruct",
]


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


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
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


def cohen_kappa(a: list, b: list) -> float | None:
    """Cohen's kappa for two raters on the same items.

    Cited from Cohen (1960) via Li et al. 2024 arXiv:2412.05579 §6.2 Metric
    table, which lists Cohen's Kappa among the standard inter-rater agreement
    metrics for LLM judges. Implementation is the canonical
        kappa = (po - pe) / (1 - pe)
    where po = observed agreement and pe = expected agreement under the
    assumption that each rater picks each label independently with their
    empirical marginal frequency.
    """
    paired = [(x, y) for x, y in zip(a, b, strict=False) if x is not None and y is not None]
    if not paired:
        return None
    n = len(paired)
    labels = sorted({x for p in paired for x in p}, key=str)
    if len(labels) < 2:
        return 1.0  # all raters agree on the only label observed
    po = sum(1 for x, y in paired if x == y) / n
    pe = 0.0
    for label in labels:
        pa = sum(1 for x, _ in paired if x == label) / n
        pb = sum(1 for _, y in paired if y == label) / n
        pe += pa * pb
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def load_panel_judged() -> dict[str, list[dict]]:
    """Returns {model: [judged_questions, ...]} from panel_judged_*.jsonl."""
    out: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(DATA_DIR.glob("panel_judged_*_run*.jsonl")):
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


def compute_panel_quality(panel_judged: dict[str, list[dict]]) -> dict:
    """Panel-aggregated quality (PoLL — majority + mean) with bootstrap CI."""
    out = {}
    for model, qs in panel_judged.items():
        valid = [q for q in qs if q.get("panel", {}).get("n_judges_valid", 0) >= 2]

        grounded_yes = [1.0 if (q["panel"].get("panel_groundedness") == "yes") else 0.0 for q in valid]
        grounded_loose = [1.0 if q["panel"].get("panel_groundedness") in ("yes", "partial") else 0.0 for q in valid]
        correct = [1.0 if q["panel"].get("panel_correctness") == "yes" else 0.0 for q in valid]
        plaus = [float(q["panel"].get("panel_distractor") or 0.0) for q in valid]
        overall = [float(q["panel"].get("panel_overall") or 0.0) for q in valid]
        schema = [1.0 if q["panel"].get("panel_schema_valid") else 0.0 for q in valid]
        consensus = [float(q["panel"].get("panel_consensus") or 0.0) for q in valid]

        bloom: dict[str, int] = defaultdict(int)
        for q in valid:
            bl = q["panel"].get("panel_bloom_level") or "unknown"
            bloom[bl] += 1

        out[model] = {
            "n_total": len(qs),
            "n_valid": len(valid),
            "n_panel_unusable": len(qs) - len(valid),
            "groundedness_strict": {
                "rate": statistics.mean(grounded_yes) if grounded_yes else 0,
                "ci": bootstrap_ci(grounded_yes),
            },
            "groundedness_lenient": {
                "rate": statistics.mean(grounded_loose) if grounded_loose else 0,
                "ci": bootstrap_ci(grounded_loose),
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
            "schema_compliance": statistics.mean(schema) if schema else 0,
            "bloom_distribution": dict(bloom),
            "panel_consensus_mean": statistics.mean(consensus) if consensus else 0,
        }
    return out


def compute_per_judge(panel_judged: dict[str, list[dict]]) -> dict:
    """Per-judge agreement / lenience / latency breakdown.

    For each (candidate-model, judge) we report the fraction the judge marked
    `groundedness=yes` and `answer_correctness=yes` independently. Differences
    between judges hint at residual self-enhancement / inter-model bias —
    see Zheng et al. 2023 arXiv:2306.05685 §3.3 ("self-enhancement bias")
    and Li et al. 2024 arXiv:2412.05579 §7.1.4.
    """
    out: dict[str, dict] = {}
    for model, qs in panel_judged.items():
        per_judge: dict[str, dict] = {jm: defaultdict(list) for jm in JUDGE_PANEL}
        latencies: dict[str, list[float]] = {jm: [] for jm in JUDGE_PANEL}
        errors: dict[str, int] = {jm: 0 for jm in JUDGE_PANEL}
        for q in qs:
            for j in q.get("per_judge", []):
                jm = j.get("judge_model")
                if jm not in per_judge:
                    continue
                latencies[jm].append(j.get("judge_latency_s") or 0.0)
                if j.get("judge_error"):
                    errors[jm] += 1
                    continue
                s = j.get("scores") or {}
                per_judge[jm]["grounded"].append(1.0 if s.get("groundedness") == "yes" else 0.0)
                per_judge[jm]["correct"].append(1.0 if s.get("answer_correctness") == "yes" else 0.0)
                per_judge[jm]["overall"].append(float(s.get("overall_quality") or 0))
        out[model] = {
            jm: {
                "n_calls": len(latencies[jm]),
                "n_errors": errors[jm],
                "grounded_yes_rate": statistics.mean(per_judge[jm]["grounded"]) if per_judge[jm]["grounded"] else 0.0,
                "correct_yes_rate": statistics.mean(per_judge[jm]["correct"]) if per_judge[jm]["correct"] else 0.0,
                "overall_mean": statistics.mean(per_judge[jm]["overall"]) if per_judge[jm]["overall"] else 0.0,
                "latency": percentiles(latencies[jm]),
            }
            for jm in JUDGE_PANEL
        }
    return out


def compute_inter_judge_kappa(panel_judged: dict[str, list[dict]]) -> dict:
    """Pairwise Cohen's kappa across all questions for each categorical key.

    Per Li et al. 2024 §6.2, kappa is the standard inter-judge agreement
    metric. We also include a 'fleiss-style' average of pairwise kappas
    across the 3-judge panel.
    """
    keys = ["groundedness", "answer_correctness", "schema_valid", "bloom_level"]
    out: dict[str, dict] = {k: {} for k in keys}

    # Pool all (q, judge_model) → score across all candidate models so kappa
    # has enough samples (~600+ items per judge pair).
    by_judge: dict[str, list] = {jm: [] for jm in JUDGE_PANEL}
    for qs in panel_judged.values():
        for q in qs:
            scores_by_judge = {j.get("judge_model"): (j.get("scores") or {}) for j in q.get("per_judge", [])}
            for jm in JUDGE_PANEL:
                by_judge[jm].append(scores_by_judge.get(jm) or {})

    for key in keys:
        pairwise = {}
        for a, b in combinations(JUDGE_PANEL, 2):
            va = [s.get(key) for s in by_judge[a]]
            vb = [s.get(key) for s in by_judge[b]]
            k = cohen_kappa(va, vb)
            pairwise[f"{a} vs {b}"] = round(k, 3) if k is not None else None
        valid_kappas = [v for v in pairwise.values() if v is not None]
        out[key] = {
            "pairwise": pairwise,
            "mean_kappa": round(statistics.mean(valid_kappas), 3) if valid_kappas else None,
        }
    return out


def compute_latency(latency: dict[str, list[dict]]) -> dict:
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


def compute_cost(latency: dict[str, list[dict]], panel_judged: dict[str, list[dict]]) -> dict:
    out = {}
    for model, calls in latency.items():
        total_in = sum((c.get("prompt_tokens") or 0) for c in calls)
        total_out = sum((c.get("completion_tokens") or 0) for c in calls)
        n_questions = sum(c.get("n_questions") or 0 for c in calls)
        pricing = PRICING.get(model, {"in": 0, "out": 0})
        gen_cost = (total_in * pricing["in"] + total_out * pricing["out"]) / 1_000_000
        cost_per_q = gen_cost / n_questions if n_questions else 0

        # Panel judge cost — sum across all 3 judges
        judge_cost = 0.0
        for q in panel_judged.get(model, []):
            for j in q.get("per_judge", []):
                jm = j.get("judge_model") or ""
                jp = PRICING.get(jm, {"in": 0, "out": 0})
                ji = j.get("judge_prompt_tokens") or 0
                jo = j.get("judge_completion_tokens") or 0
                judge_cost += (ji * jp["in"] + jo * jp["out"]) / 1_000_000

        out[model] = {
            "gen_input_tokens": total_in,
            "gen_output_tokens": total_out,
            "gen_cost_total": gen_cost,
            "gen_cost_per_question": cost_per_q,
            "gen_cost_per_100q": cost_per_q * 100,
            "panel_judge_cost_total": judge_cost,
        }
    return out


def emit_latex(quality: dict, latency: dict, cost: dict, kappa: dict, path: Path):
    NAMES = {
        "gpt-oss-120b": "gpt-oss-120b",
        "gemma-4-31b-it": "gemma-4-31b-it",
        "meta-llama/llama-3.3-70b-instruct:free": r"llama-3.3-70b",
    }

    def pct(x: float) -> str:
        return f"{x*100:.1f}\\%"

    def ci_str(ci: tuple) -> str:
        return f"[{ci[0]*100:.1f}, {ci[1]*100:.1f}]"

    body_lines = []
    for m in MODELS:
        if m not in quality:
            continue
        q = quality[m]
        l = latency.get(m, {}).get("per_question_implied", {})
        c = cost.get(m, {})
        name = NAMES.get(m, m)
        body_lines.append(
            f"{name} & "
            f"{pct(q['groundedness_strict']['rate'])} {ci_str(q['groundedness_strict']['ci'])} & "
            f"{pct(q['correctness']['rate'])} {ci_str(q['correctness']['ci'])} & "
            f"{q['plausibility']['mean']:.2f} & "
            f"{q['overall_quality']['mean']:.2f} & "
            f"{pct(q['schema_compliance'])} & "
            f"{l.get('p50', 0):.1f} / {l.get('p99', 0):.1f} & "
            f"\\${c.get('gen_cost_per_100q', 0):.4f} \\\\"
        )

    table = r"""\begin{table}[htbp]
\centering
\caption{Quiz generation model comparison ($n=72$ questions/model, 3 runs $\times$ 2 lessons $\times$ 12 questions). Quality scores are aggregated across a Panel of three LLM judges (gpt-5-chat-latest, claude-opus-4-7, llama-3.3-70b-instruct) using majority vote for categorical dimensions and mean for ordinal scores~\cite{li2024llmsasjudges}. Anti-bias and reference-guided judge prompting follows Zheng et al.~\cite{zheng2023judging}.}
\label{tab:gen_model_eval}
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{|l|c|c|c|c|c|c|c|}
\hline
\textbf{Model} & \textbf{Grounded [95\% CI]} & \textbf{Correct [95\% CI]} & \textbf{Plaus.} & \textbf{Overall (1--10)} & \textbf{Schema} & \textbf{p50/p99 (s/q)} & \textbf{\$/100q} \\
\hline
""" + "\n".join(body_lines) + r"""
\hline
\end{tabular}
\end{table}
"""
    path.write_text(table)


def emit_kappa_table(kappa: dict, path: Path):
    """Inter-judge agreement table."""
    NICE = {
        "groundedness": "Groundedness (3-class)",
        "answer_correctness": "Correctness (3-class)",
        "schema_valid": "Schema valid (binary)",
        "bloom_level": "Bloom level (6-class)",
    }
    body = []
    for key in ["groundedness", "answer_correctness", "schema_valid", "bloom_level"]:
        if key not in kappa:
            continue
        mean_k = kappa[key].get("mean_kappa")
        pw = kappa[key].get("pairwise", {})
        pw_str = ", ".join(f"{v:.2f}" if v is not None else "n/a" for v in pw.values())
        body.append(f"{NICE.get(key, key)} & {pw_str} & {mean_k if mean_k is not None else 'n/a'} \\\\")

    table = r"""\begin{table}[htbp]
\centering
\caption{Inter-judge agreement (Cohen's $\kappa$) across the three-LLM panel. Pairwise columns list $\kappa$ for (gpt-5/claude, gpt-5/llama, claude/llama). Mean $\kappa$ is the unweighted mean of the three pairs. Per Cohen's interpretation, $\kappa$ above 0.6 indicates substantial agreement; the survey of Li et al.~\cite{li2024llmsasjudges} cites Cohen's $\kappa$ as the standard meta-evaluation metric for LLM-as-a-judge studies.}
\label{tab:inter_judge_kappa}
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{|l|c|c|}
\hline
\textbf{Dimension} & \textbf{Pairwise $\kappa$} & \textbf{Mean $\kappa$} \\
\hline
""" + "\n".join(body) + r"""
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
    print("Loading panel-judged data…")
    panel = load_panel_judged()
    print(f"  {sum(len(v) for v in panel.values())} questions across {len(panel)} candidate models")

    print("Loading latency data…")
    latency = load_latency()
    print(f"  {sum(len(v) for v in latency.values())} latency records")

    print("\nComputing panel-aggregated quality…")
    quality = compute_panel_quality(panel)

    print("Computing per-judge breakdown…")
    per_judge = compute_per_judge(panel)

    print("Computing inter-judge Cohen's kappa…")
    kappa = compute_inter_judge_kappa(panel)

    print("Computing latency percentiles…")
    lat = compute_latency(latency)

    print("Computing cost…")
    cost = compute_cost(latency, panel)

    aggregate = {
        "quality": quality,
        "per_judge": per_judge,
        "inter_judge_kappa": kappa,
        "latency": lat,
        "cost": cost,
    }
    (DATA_DIR / "aggregate_metrics_v2.json").write_text(json.dumps(aggregate, indent=2))

    print("\n=== Panel Quality Summary ===")
    for m in MODELS:
        if m not in quality:
            continue
        q = quality[m]
        print(f"\n{m} (n={q['n_valid']}):")
        print(f"  groundedness (strict): {q['groundedness_strict']['rate']*100:.1f}% [{q['groundedness_strict']['ci'][0]*100:.1f}, {q['groundedness_strict']['ci'][1]*100:.1f}]")
        print(f"  correctness:           {q['correctness']['rate']*100:.1f}% [{q['correctness']['ci'][0]*100:.1f}, {q['correctness']['ci'][1]*100:.1f}]")
        print(f"  plausibility (1–5):    {q['plausibility']['mean']:.2f} [{q['plausibility']['ci'][0]:.2f}, {q['plausibility']['ci'][1]:.2f}]")
        print(f"  overall (1–10):        {q['overall_quality']['mean']:.2f} [{q['overall_quality']['ci'][0]:.2f}, {q['overall_quality']['ci'][1]:.2f}]")
        print(f"  schema compliance:     {q['schema_compliance']*100:.1f}%")
        print(f"  panel consensus:       {q['panel_consensus_mean']*100:.1f}%")
        print(f"  bloom: {q['bloom_distribution']}")

    print("\n=== Inter-judge Cohen's κ ===")
    for k, v in kappa.items():
        print(f"  {k}: mean κ = {v['mean_kappa']}")
        for pair, val in v["pairwise"].items():
            print(f"    {pair}: {val}")

    print("\n=== Per-judge breakdown ===")
    for m in MODELS:
        if m not in per_judge:
            continue
        print(f"\n{m}:")
        for jm in JUDGE_PANEL:
            d = per_judge[m][jm]
            print(f"  {jm}: grounded_yes={d['grounded_yes_rate']*100:.1f}%  correct_yes={d['correct_yes_rate']*100:.1f}%  overall={d['overall_mean']:.2f}  errors={d['n_errors']}")

    print("\n=== Cost ===")
    for m in MODELS:
        if m not in cost:
            continue
        c = cost[m]
        print(f"  {m}: ${c['gen_cost_per_100q']:.4f}/100q (gen) + ${c['panel_judge_cost_total']:.2f} panel-judge total")

    emit_latex(quality, lat, cost, kappa, REPORT_DIR / "comparison_table_v2.tex")
    emit_kappa_table(kappa, REPORT_DIR / "kappa_table.tex")
    emit_latency_table(lat, REPORT_DIR / "latency_table.tex")
    print(f"\nLaTeX tables written to {REPORT_DIR}/")


if __name__ == "__main__":
    main()
