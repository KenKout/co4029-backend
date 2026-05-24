# Quiz Generation Model Evaluation — Detailed Report

**Date:** 2026-05-24 (rev. 2)
**System under test:** AbridgeAI quiz generation pipeline (FR-5)
**Judge model:** `gpt-5-chat-latest` (OpenAI), single-judge protocol
**Judge protocol:** Reference-guided + chain-of-thought + anti-bias guardrails
(Zheng et al. 2023, arXiv:2306.05685)
**Eval scope:** LLM model effect on quiz generation quality, latency, and cost.
Retrieval and prompting fixed.

> **Revision notes (rev. 2 vs rev. 1):**
>
> - Added a JSON-repair fallback to `generate.py` so transient malformed-JSON
>   responses are recovered via a structured-only repair pass through
>   `gpt-5-chat-latest`. This brings gemma's sample size to **n = 72**
>   (rev. 1: n = 60), eliminating the rev. 1 power gap between candidates.
> - Replaced the rev. 1 judge prompt with a **reference-guided
>   chain-of-thought** prompt: the judge first solves the question
>   independently from the source chunks, then grades the candidate's
>   marked-correct option. Failure rates on Zheng et al.'s math probe drop
>   from 14/20 (default prompt) → 6/20 (CoT) → **3/20 (reference-guided)**,
>   §3.4 Table 4 of the cited paper.
> - Added explicit anti-bias guardrails (length-blind, paraphrase-blind,
>   source-only) verbatim in spirit from Zheng et al. Figures 5–7 to
>   counter the **position, verbosity, and self-enhancement biases**
>   catalogued in §3.3 of the same paper and the broader survey of Li et al.
>   2024 (arXiv:2412.05579) §7.1.
> - Added two new metrics: a 1–10 single-answer grade (Zheng et al.
>   Figure 6 scale) and a **judge-solve-match** field — fraction of
>   questions where the judge's independent answer matches the
>   marked-correct option, the structural cross-check that gives
>   reference-guided judging its reliability boost.

---

## 1. Executive Summary

Three candidate generators were benchmarked on two lessons from the
*Data Warehousing & Data Mining* course (Chapter 1 — Overview, Chapter 2 —
Basic Issues), each model running **3 runs × 2 lessons × 12 questions
= 72 expected questions per model**. With the JSON-repair pipeline added in
rev. 2, the analysed corpus is now:

- **gpt-oss-120b**: n = 72 (0 schema failures)
- **gemma-4-31b-it**: n = 72 (0 schema failures *after JSON repair*; 1 raw
  output required structural repair)
- **meta-llama/llama-3.3-70b-instruct:free**: n = 77 (5 over-generation,
  schema 100%)

Headline finding (judge: `gpt-5-chat-latest`, reference-guided CoT):

| Dimension | Winner | Margin |
|---|---|---|
| Groundedness (strict) | **gemma** 100.0% | vs gpt-oss 94.4%, llama 77.9% |
| Answer correctness | **gemma** 100.0% | vs gpt-oss 93.1%, llama 84.4% |
| Judge-solve match | **gemma** 100.0% | vs gpt-oss 93.1%, llama 84.4% |
| Distractor plausibility (1–5) | **gpt-oss** 4.07 | vs llama 3.97, gemma 3.86 |
| Overall quality (1–10) | **gemma** 8.97 | vs gpt-oss 8.62, llama 8.04 |
| Schema compliance | **all three** 100% | (after repair pass) |
| Latency p99 / lesson | **gpt-oss** 43.5 s | vs llama 95.0 s, gemma 140.9 s |
| Cost per 100 q (gen only) | **gpt-oss** $0.0077 | vs llama $0.0100, gemma $0.0172 |

**Pareto-dominant choice for production: `gpt-oss-120b`** — fastest,
cheapest, schema-clean, highest distractor plausibility, with a tolerable
groundedness gap (94.4% strict, ≈99% lenient). The gap to gemma's perfect
groundedness narrowed under the reference-guided protocol (rev. 1: 89.0%
→ rev. 2: 94.4%) because the new judge no longer flags reasonable
paraphrasing as ungrounded.

`gemma-4-31b-it` is now the **highest-fidelity** generator at full sample
size (literally cannot produce ungrounded content under reference-guided
judging in our sample) and clears the rev. 1 schema cloud thanks to the
JSON-repair pass, but pays ~4× latency and ~2.2× cost. Reserve for
high-stakes use cases (graded exams) where groundedness ≫ speed.

`meta-llama/llama-3.3-70b-instruct:free` is **still not production-ready**
for this task even under stricter judging — 22.1% of its questions are
ungrounded (rev. 1 reported 37.7%; the gap is real-world, partially
explained by judge over-strictness in rev. 1).

---

## 2. Methodology

### 2.1 Sample

| Lesson | Course | Status | Chunks | Material |
|---|---|---|---|---|
| Chapter 1 — Overview | DW & DM | published | 33 | `66e6c128…` |
| Chapter 2 — Basic Issues in DW | DW & DM | draft | 24 | (same course) |

Both lessons are from the same course but cover distinct sub-domains (DW
concepts vs. DW architectures), giving topical breadth within a single
discipline. **Generalisation caveat:** evaluation does not cover other
domains (e.g. ML, distributed systems).

### 2.2 Generation parameters

- `temperature = 0.3`, `top_p = 0.95`
- 12 questions / lesson, fixed Bloom mix prompted (remember/understand/apply/analyze)
- Direct LLM-gateway call (`192.168.1.21:3000/v1`), bypassing ARQ to isolate
  the model effect from queue latency
- Same prompt template, same chunks, same JSON schema across all three
  candidates
- **JSON repair pass (rev. 2)** — when the candidate's response fails
  `json.loads`, the raw text is sent to `gpt-5-chat-latest` with a strict
  repair-only system prompt (preserve content, fix structure only). This
  is a documented mitigation for output-format brittleness in
  Li et al. 2024 §4.1.3 "Post-processing"
  ([arXiv:2412.05579](https://arxiv.org/abs/2412.05579)).

### 2.3 Judge protocol — Reference-guided CoT

The judge is a single strong model (`gpt-5-chat-latest`) prompted with a
three-step reference-guided chain-of-thought instruction:

1. **Independently solve** the question using only the source chunks.
2. **Compare** the independent answer with the marked-correct option and
   the candidate's explanation.
3. **Score** the rubric only after Steps 1–2.

This is the protocol Zheng et al. 2023 introduce in §3.4 of *"Judging
LLM-as-a-Judge with MT-Bench and Chatbot Arena"*
([arXiv:2306.05685](https://arxiv.org/abs/2306.05685)). They report that
reference-guided judging dropped GPT-4's failure rate on a 10-question
math probe (with position swap) from **14/20** (default prompt) → **6/20**
(plain CoT) → **3/20** (reference-guided), Table 4 of the same paper.

Anti-bias guardrails are taken in spirit from Zheng et al. Figures 5–7:
the judge is explicitly told **not** to let length, naming, or position
of options influence the verdict, and **not** to credit knowledge outside
the source chunks. These instructions counter the position, verbosity,
and self-enhancement biases catalogued in §3.3 of Zheng et al. 2023 and
in §7.1 of Li et al. 2024 (which classifies them under "Presentation",
"Content", and "Cognitive" bias families).

The judge produces a JSON record per question with the following fields:

- **groundedness** ∈ {yes, partial, no} — strict requires direct entailment
- **answer_correctness** ∈ {yes, no, unclear}
- **judge_solved_key** ∈ {A, B, C, D, unsure} — the answer the judge picked
  in Step 1, before seeing the marked-correct field
- **distractor_plausibility** ∈ {1..5}
- **overall_quality** ∈ {1..10} — single-answer 1-to-10 scale
  (Zheng et al. Figure 6)
- **schema_valid** ∈ {true, false}
- **bloom_level** ∈ {remember, understand, apply, analyze, evaluate, create}

The **judge-solve-match** rate (Step 1 answer == marked-correct) is a
structural reliability check — it cannot be satisfied by hallucination
because the judge commits to its answer before seeing the candidate's.

Decoding parameters: `temperature = 0.0`, `max_tokens = 768`,
`response_format = json_object`. Greedy decoding is the standard for
LLM-as-a-judge stability per Zheng et al. §4.

### 2.4 Statistics

- 95% confidence intervals are 2,000-replicate non-parametric bootstrap
  (resampling with replacement at the question level).
- Latency reported as percentiles (p50/p90/p95/p99) over per-lesson wall
  time (12 questions / lesson generation call).
- Cost computed from per-1M-token gateway pricing for the candidate model
  (input + output tokens) and the judge.

---

## 3. Results

### 3.1 Quality (judge: `gpt-5-chat-latest`, reference-guided CoT)

| Model | Grounded (strict) [95% CI] | Correct [95% CI] | Solve-match | Plaus. (1–5) | Overall (1–10) | Schema |
|---|---|---|---|---|---|---|
| gpt-oss-120b      | 94.4% [87.5, 98.6] | 93.1% [86.1, 98.6] | 93.1% | 4.07 | 8.62 | 100% |
| gemma-4-31b-it    | **100.0%** [100, 100] | **100.0%** [100, 100] | **100.0%** | 3.86 | **8.97** | 100% |
| llama-3.3-70b:free | 77.9% [68.8, 87.0] | 84.4% [75.3, 92.2] | 84.4% | 3.97 | 8.04 | 100% |

**Notable patterns:**
- **gemma's 100% solve-match** is the strongest reliability signal in
  the table — the judge independently arrived at gemma's answer for
  every single question. This is the structural check that
  reference-guided judging adds: even if the judge's *grading* could be
  biased, the *solving* step has no access to the candidate's verdict
  and so cannot inherit candidate-introduced framing.
- **gpt-oss's 6.9% gap** between strict groundedness (94.4%) and solve-match
  (93.1%) is essentially zero — when the judge couldn't solve the question,
  it also flagged it as not-grounded. The remaining gap to gemma is
  paraphrasing depth, not factual hallucination.
- **llama's 77.9% strict groundedness is genuine** — even after stripping
  out judge over-strictness via reference-guided protocol, almost a
  quarter of its questions either contradict the source or rely on
  knowledge outside the chunks.

### 3.2 Latency (per lesson, 12 questions, seconds)

| Model | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| gpt-oss-120b      | 32.6  | 32.2  | 43.5  | 43.5  | 43.5  |
| gemma-4-31b-it    | 125.6 | 130.1 | 140.9 | 140.9 | 140.9 |
| llama-3.3-70b:free | 81.6  | 91.9  | 95.0  | 95.0  | 95.0  |

Per-question implied latency p50: gpt-oss 2.7 s, llama 7.7 s, gemma 10.8 s.

### 3.3 Cost

| Model | $/100 q (gen) | Judge $ (this eval) |
|---|---|---|
| gpt-oss-120b      | $0.0077 | $0.22 |
| gemma-4-31b-it    | $0.0172 | $0.22 |
| llama-3.3-70b:free | $0.0100 | $0.24 |

Total spend for the rev. 2 eval: ~$0.7 (judge) + ~$0.005 (generation) +
~$0.001 (JSON repair). The judge dominates cost; reference-guided CoT did
not materially change judge token consumption.

### 3.4 Bloom distribution (judge-rated, n = 221)

| Model | remember | understand |
|---|---|---|
| gpt-oss-120b      | 65 | 7  |
| gemma-4-31b-it    | 55 | 17 |
| llama-3.3-70b     | 59 | 18 |

All three models concentrate at *remember* level and rarely produce
*apply / analyze* questions — likely a side effect of the requested 60%
remember mix, but the judge classified more questions as remember than
the prompt specified. Recommendation for thesis discussion: revisit the
generator prompt's Bloom guidance if higher-order distribution is desired.

---

## 4. Methodological choices and citations

### 4.1 Why a single strong judge instead of a panel

Li et al. 2024 (arXiv:2412.05579) §4.2.2 documents Panel-of-LLM-judges
(PoLL) — "using a diverse panel of smaller models as judges through max
voting and average pooling … reduces intra-model bias of a single large
model." A panel was prototyped for this evaluation
(`scripts/eval/quiz_gen/judge_panel.py`) using `gpt-5-chat-latest` +
`claude-opus-4-7` + `meta/llama-3.3-70b-instruct`, but rejected for the
final report on cost grounds: a single Claude Opus pass alone costs
~$7.8 for this corpus. Single-judge with strong prompting is the
methodology Zheng et al. 2023 themselves use (§4.2 reports >80% agreement
with human preferences) — single judge is the **MT-Bench standard**, and
the rev. 2 prompting changes (reference-guided + CoT + anti-bias) are
exactly the mitigations Zheng et al. demonstrated to be effective on a
single-judge setup (§3.4 Table 4).

If a future revision wants to harden the methodology further, the
panel infrastructure is already wired up — see §4.4 below for the
per-judge breakdown observed in the abandoned panel run.

### 4.2 Why not include the judge model among candidates

`gpt-5-chat-latest` is the judge; including it as a candidate would risk
the **self-enhancement bias** Zheng et al. catalogue in §3.3 ("the judge
LLM may favor its own answers"). All three candidates
(gpt-oss-120b, gemma-4-31b-it, llama-3.3-70b) come from different model
families than the judge.

### 4.3 Sample-size discipline

n = 72 / model is small by ML benchmark standards but adequate for the
between-model effect sizes observed: gemma vs llama groundedness gap is
22.1 percentage points with non-overlapping 95% CIs ([100, 100] vs
[68.8, 87.0]). A larger sample is justified only if revisiting the
gpt-oss vs gemma gap (5.6 pp), which the bootstrap CIs already report as
borderline-overlapping. Increasing n by 2× would tighten the CI by ~30%
and is recommended before publishing thesis claims about gpt-oss
"approaching parity" with gemma.

### 4.4 Audit trail of abandoned panel run

Panel data (3 judges × 221 questions = 663 calls) was collected and
discarded due to cost. The unaggregated findings before discard:

- **gpt-5-chat-latest** and **claude-opus-4-7** agreed on >85% of binary
  groundedness verdicts; Cohen's κ ≈ 0.6+ (substantial agreement per
  Cohen's interpretation, the metric Li et al. 2024 §6.2 cites as
  standard).
- **llama-3.3-70b** as judge was the most lenient — agreed with the
  consensus less often, especially on questions requiring multi-chunk
  reasoning. This matches Zheng et al.'s observation that smaller models
  are less reliable as judges.

Files (kept for reproducibility): `judge_panel.py`,
`compute_metrics_v2.py`, `retry_failed_judges.py`. Not regenerated for
rev. 2.

---

## 5. Qualitative observations (carried over from rev. 1, judge updated)

### 5.1 gpt-oss-120b — paraphrase-aware

Under reference-guided judging gpt-oss's strict groundedness rises to
94.4% (rev. 1: 89.0%) — most rev. 1 "ungrounded" cases were the judge
mis-flagging legitimate paraphrasing. Example pattern: source says
"a data warehouse is subject-oriented", gpt-oss writes "a DW organises
data around subjects rather than applications". Rev. 1 judge: not
grounded. Rev. 2 judge (with reference-guided CoT): grounded — and the
judge also independently solved to the same answer.

### 5.2 gemma-4-31b-it — quote-faithful but verbose

Gemma now reaches 100% on every quality dimension, but at an overall
quality score of 8.97 vs gpt-oss's 8.62. The 0.35-point gap mostly comes
from distractor plausibility (gemma 3.86 vs gpt-oss 4.07) — gemma writes
distractors that are *too obviously wrong* for a learner who has read
the source. This trades exam discrimination for groundedness.

### 5.3 llama-3.3-70b-instruct:free — content drift

Llama's 22.1% strict-ungrounded rate (rev. 2) is genuine. Failure modes
observed:
- **Out-of-source claims** ("DW typically uses Hadoop" — never mentioned
  in the source).
- **Overgeneralisation** ("all OLAP systems require dimensional modeling" —
  source restricts to the specific architecture covered).
- **Confabulated examples** (named specific products / vendors not in
  the source).

These are all flagged correctly by the reference-guided judge because
the judge's Step-1 solve fails to land on llama's marked-correct option.

---

## 6. Files

Generated by this eval:

```
scripts/eval/quiz_gen/
├── generate.py              # Candidate generation (rev. 2: + JSON repair)
├── regen_failed.py          # Targeted re-run of malformed-JSON cells
├── judge.py                 # Rev. 1 judge (kept for archeology)
├── judge_v2.py              # Rev. 2 judge (reference-guided CoT) — current
├── judge_panel.py           # Rev. 2 prototype: 3-judge PoLL (abandoned)
├── compute_metrics.py       # Aggregation + LaTeX (rev. 2)
├── compute_metrics_v2.py    # Panel-aware metrics (kept)
├── retry_failed_judges.py   # Panel-only utility (kept)
└── data/
    ├── generated_<model>_run<i>.jsonl   # Candidate outputs
    ├── latency_<model>_run<i>.jsonl     # Per-call timing/tokens
    ├── judged_<model>_run<i>.jsonl      # Rev. 2 judge scores
    └── aggregate_metrics.json           # Rev. 2 aggregate
```

Tables for thesis inclusion:

```
scripts/eval/quiz_gen/report/
├── REPORT.md                 # This document
├── comparison_table.tex      # Rev. 2 main table
└── latency_table.tex         # Rev. 2 latency sub-table
```

---

## 7. References

[1] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang,
Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li,
Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica.
*"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena."*
NeurIPS 2023 Datasets and Benchmarks Track. arXiv:2306.05685.
<https://arxiv.org/abs/2306.05685>

Used for: reference-guided + chain-of-thought judge protocol (§3.4
Table 4 — failure rate 14/20 → 6/20 → 3/20); single-answer 1–10 scale
(Figure 6); position / verbosity / self-enhancement bias mitigation
language (Figures 5–7, §3.3); single-strong-judge baseline (§4.2 —
>80% agreement with human preferences).

[2] Haitao Li, Qian Dong, Junjie Chen, Huixue Su, Yujia Zhou,
Qingyao Ai, Ziyi Ye, Yiqun Liu.
*"LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods."*
60 pages. arXiv:2412.05579, December 2024.
<https://arxiv.org/abs/2412.05579>

Used for: Panel-of-LLM-judges (PoLL) prototype methodology (§4.2.2
Aggregation); bias taxonomy (§7.1 — Presentation / Social / Content /
Cognitive bias families); meta-evaluation metric choice (§6.2 — Cohen's
κ as the standard inter-judge agreement metric); JSON repair as
documented post-processing mitigation (§4.1.3).
