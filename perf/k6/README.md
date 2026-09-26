# k6 Load-Test Suite (Capstone Final)

Load tests for the aBridgeAI backend, designed to close the gaps the
specialized-project evaluation explicitly admitted (Conclusion, ch. 8):

> "Load testing exercised read paths only. … Write paths, concurrent attempts
> on one quiz, the AI generation pipelines, and the background worker pool
> are not represented, and these are precisely the paths where contention and
> cost would be expected to appear."

Target deployment: `https://abridgeai.tech` (single Uvicorn + PostgreSQL 16 +
Redis 7 + Neo4j on `192.168.1.41`), same as the previous iteration's
environment, so numbers are comparable year-over-year.

## Scenarios

| Script | Gap closed | What it does |
|---|---|---|
| `01_smoke` | — | 1 VU sanity pass over every picked endpoint (reads + one quiz journey + interview lifecycle) |
| `02_browse_mix` | read realism | Learner bootstrap + dashboard mixed reads (replaces single-endpoint GET tests) |
| `03_quiz_take` | **write paths, concurrent attempts on one quiz** | Full journey: start attempt → per-question answer writes (in-transaction SM-2) → heartbeats → integrity events → submit+grade, N VUs on the SAME quiz |
| `04_sr_review` | write paths | SR review session: queue reads + SM-2 review writes (self-limits via 429 cooldown) |
| `05_interview_control` | **AI pipelines** | Interview session lifecycle: start (LLM opening) → LiveKit token mint → state polling → finish (queues evaluation). ⚠ LLM cost — keep VUs tiny |
| `06_stress_ramp` | capacity headroom | Mixed read/write ramp 50→100→150→200 VUs to find the saturation point (previous cap was 100 VUs) |

Picked-endpoint rationale (hot paths per session):

* **Bootstrap reads** — `GET /users/me`, `/me/courses`, `/me/notifications/unread-count` (polled), `/me/progress/courses/{id}`, `/me/cards-due`
* **Course-learn reads** — `GET /courses/{id}/content`, `/courses` catalog, `/courses/{id}/quiz-progress`, `/courses/{id}/interview-progress`, `/me/sr-dashboard-summary`
* **Quiz writes (bursty)** — `POST /quizzes/{id}/attempts`, `POST /attempts/{id}/answers`, `POST /attempts/{id}/session/heartbeat`, `POST /attempts/{id}/integrity-events`, `POST /attempts/{id}/submit`
* **SR writes** — `POST /me/review/{question_id}`
* **Interview control plane** — `POST /interview-configs/{id}/sessions`, `POST /interview-sessions/{id}/realtime-token`, `GET /interview-sessions/{id}`, `POST /interview-sessions/{id}/finish`
* **Baseline kept for comparison** — `GET /api/v1/healthz`, `GET /api/v1/courses`

Not load-tested (deliberate): Google OAuth (external dependency), LiveKit
WebRTC media (not HTTP), S3 multipart uploads, admin/authoring routes
(low traffic).

## Usage

### Multi-user mode (recommended — true concurrent attempts)

All VUs sharing ONE token can only hold one active quiz attempt at a time
(one-active-attempt-per-user guard → 409s under contention). Multi-user mode
mints N synthetic students on the deployment server, each VU authenticates as
its own student:

```bash
# 1. Mint users (runs on the server, idempotent, 30-day JWTs):
scp bin/mint_loadtest_users.py ubuntu@192.168.1.41:/tmp/
ssh ubuntu@192.168.1.41 'sudo sh -c "cd /root/co4029/backend && \
    .venv/bin/python /tmp/mint_loadtest_users.py --count 20 \
    --course-id e5ba475d-140e-4a82-9a6b-b283b8e760ac"' \
    > .state/users.json

# 2. run.sh auto-detects .state/users.json — just run:
./run.sh 03_quiz_take                # 10 VUs = 10 distinct students, same quiz
VUS=100 DURATION=5m ./run.sh 03_quiz_take
```

Minted users are `k6-loadtest-NNN@abridgeai.local`, enrolled and org-membered
for the DWDSS course. If VUs outnumber minted users, tokens wrap around.

### Single-user mode (fallback)

```bash
cd perf/k6
REFRESH_TOKEN=<paste> ./run.sh 01_smoke    # seeds .state/refresh_token
./run.sh 02_browse_mix
VUS=100 DURATION=5m ./run.sh 03_quiz_take  # expect 409 contention (guard test)
```

Every scenario prints a per-endpoint table (count, RPS, p50/p95/p99/max,
error %) via `handleSummary` — screenshot/copy those into the evaluation
chapter.

## Auth notes

* Access-token TTL is 15 minutes; `run.sh` refreshes automatically when <60 s
  remain.
* **Refresh tokens rotate server-side on every refresh.** Never refresh twice
  with the same token — `refresh-token.sh` stores the latest in
  `.state/refresh_token` (git-ignored, chmod 600). Refreshing from elsewhere
  (e.g. the browser session of the same user) invalidates the stored one;
  re-seed with `REFRESH_TOKEN=... ./run.sh ...` if that happens.
* A single access token is shared by all VUs — fine for runs under ~14 min.

## Data-effect warnings

* `03_quiz_take` creates real attempts + answers on the demo quiz
  (`allow_retakes=true`, `max_attempts=null`). Expect the attempt history to
  grow; clean up in the teacher gradebook afterwards if needed.
* Minted load-test users accumulate attempts/SM-2 rows during runs. They are
  identifiable by the `k6-loadtest-%` email prefix; drop or keep them after
  the evaluation as you prefer.
* `04_sr_review` advances the caller's real SM-2 schedule; cards eventually go
  into cooldown and the API answers `429 all_cards_in_cooldown` (handled).
* `05_interview_control` consumes LLM tokens and enqueues evaluation jobs.
  Defaults: ≤5 VUs, 1 iteration, 0 text turns.

## Methodology suggestions for the report

1. Run `01_smoke` → validates everything, also the "before" screenshot.
2. Read scenarios at 50 / 100 VUs (`02_browse_mix`) → compare with the
   previous iteration's 136–431 RPS results.
3. Write scenarios at 50 / 100 VUs (`03_quiz_take`, `04_sr_review`) → the new
   evidence for NFR-2.1.
4. `06_stress_ramp` → report the saturation VU count + where p95 crosses the
   threshold. NFR-2.1 verdict = measured headroom vs target concurrent users.
5. `07_nfr_peak` (NFR-2.1 profile) at `TOTAL_VUS=50/100/200` → 1×/2×/4×
   headroom matrix; results from 2026-09-26 are archived in
   `results/nfr-peak-2026-09-26.md` (+ raw k6 logs under `results/logs-*`).
   Deployment now runs **4 Uvicorn workers** (`ecosystem.config.cjs`, PM2
   `pm2 save`d — permanent). Headline numbers, 1 worker → 4 workers:
   ceiling 35→111 RPS (~3×), p50@1× 661ms→73ms (9×), p95@1× 2.66s→464ms;
   NFR-2.1 verdict at 50 VUs = PASS (0% errors, all thresholds). Next
   bottleneck at 4× overload: PostgreSQL `max_connections` (0.55% HTTP 500,
   `psycopg.OperationalError` FATAL) — remediate with PgBouncer / higher
   `max_connections` if further scaling is needed.
