-- Single job detail by id — checks processing_jobs first, then generation_runs.
SELECT
    pj.id               AS id,
    pj.entity_type      AS entity_type,
    pj.entity_id        AS entity_id,
    pj.job_type         AS job_type,
    pj.status           AS status,
    pj.progress_percent AS progress_percent,
    pj.error_message    AS error_message,
    pj.retry_count      AS retry_count,
    pj.started_at       AS started_at,
    pj.finished_at      AS finished_at,
    pj.created_at       AS created_at,
    pj.updated_at       AS updated_at,
    pj.request_id       AS request_id
FROM processing_jobs pj
WHERE pj.id = CAST(:job_id AS uuid)

UNION ALL

SELECT
    gr.id                                    AS id,
    'generation_run'                         AS entity_type,
    gr.id                                    AS entity_id,
    'generate_' || gr.generation_type        AS job_type,
    gr.status                                AS status,
    CASE gr.status
        WHEN 'completed' THEN 100
        WHEN 'running'   THEN 50
        ELSE 0
    END                                      AS progress_percent,
    CAST(gr.config_json->>'failure' AS text) AS error_message,
    0                                        AS retry_count,
    gr.started_at                            AS started_at,
    gr.finished_at                           AS finished_at,
    gr.created_at                            AS created_at,
    gr.updated_at                            AS updated_at,
    (
        SELECT amc.request_id
        FROM ai_model_calls amc
        WHERE amc.generation_run_id = gr.id
          AND amc.request_id IS NOT NULL
        ORDER BY amc.called_at ASC
        LIMIT 1
    )                                        AS request_id
FROM generation_runs gr
WHERE gr.id = CAST(:job_id AS uuid)

LIMIT 1;
