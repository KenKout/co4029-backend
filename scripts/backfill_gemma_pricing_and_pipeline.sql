-- Backfill B (gemma pricing + cost recompute) and A (pipeline_run_id for
-- historical interview_evaluation calls). Idempotent + transactional.
--
-- B) Add google/gemma-4-31b-it to ai_model_pricing at $0.10 / $0.35 per 1M
--    tokens (per-1M convention, matching migration 0041), then recompute
--    estimated_cost_usd for EVERY gemma call that has token counts. Gemma is
--    used across 4 stages (evaluation, gap_report, generation,
--    interview_generation) — all were NULL cost because the model had no
--    pricing row. Cost formula mirrors compute_cost():
--      input_tokens/1e6 * input_rate + output_tokens/1e6 * output_rate
--
-- A) Set pipeline_run_id = generation_run_id for the historical eval/gap
--    calls the earlier backfill linked to interview_evaluation runs but left
--    with a NULL pipeline_run_id (so the ai-costs "By Pipeline" tab shows
--    them, matching what the new service code now writes for fresh runs).

BEGIN;

-- B1) Insert gemma pricing (idempotent: skip if already present).
INSERT INTO ai_model_pricing (model_name, input_usd_per_1m, output_usd_per_1m, notes)
SELECT 'google/gemma-4-31b-it', 0.10, 0.35, 'per-1M; added 2026-07-24 for interview eval/gap + generation calls'
WHERE NOT EXISTS (
    SELECT 1 FROM ai_model_pricing WHERE model_name = 'google/gemma-4-31b-it'
);

-- B2) Recompute cost for all gemma calls that have token data.
--     Safe to re-run: it just rewrites the same computed value.
UPDATE ai_model_calls amc
SET estimated_cost_usd = ROUND(
        COALESCE(amc.input_tokens, 0)  / 1000000.0 * p.input_usd_per_1m
      + COALESCE(amc.output_tokens, 0) / 1000000.0 * p.output_usd_per_1m
    , 6)
FROM ai_model_pricing p
WHERE p.model_name = 'google/gemma-4-31b-it'
  AND amc.model_name = 'google/gemma-4-31b-it'
  AND amc.input_tokens IS NOT NULL;

-- A) Backfill pipeline_run_id for historical interview_evaluation calls.
UPDATE ai_model_calls amc
SET pipeline_run_id = amc.generation_run_id
FROM generation_runs gr
WHERE gr.id = amc.generation_run_id
  AND gr.generation_type = 'interview_evaluation'
  AND amc.pipeline_run_id IS NULL;

-- ---- Verification (printed inside the transaction, before COMMIT) ----
\echo '=== gemma pricing row (expect 1) ==='
SELECT model_name, input_usd_per_1m, output_usd_per_1m
FROM ai_model_pricing WHERE model_name = 'google/gemma-4-31b-it';

\echo '=== gemma calls: null cost remaining (expect 0 where tokens present) ==='
SELECT COUNT(*) FILTER (WHERE estimated_cost_usd IS NULL AND input_tokens IS NOT NULL) AS null_cost_with_tokens,
       COUNT(*) AS total_gemma_calls,
       ROUND(SUM(estimated_cost_usd), 4) AS gemma_total_usd
FROM ai_model_calls WHERE model_name = 'google/gemma-4-31b-it';

\echo '=== interview_evaluation calls missing pipeline_run_id (expect 0) ==='
SELECT COUNT(*) AS eval_calls_missing_pipeline_run
FROM ai_model_calls amc
JOIN generation_runs gr ON gr.id = amc.generation_run_id
WHERE gr.generation_type = 'interview_evaluation'
  AND amc.pipeline_run_id IS NULL;

COMMIT;
