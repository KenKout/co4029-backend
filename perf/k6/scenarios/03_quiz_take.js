// 03_quiz_take.js — THE contention scenario.
// N concurrent VUs each repeatedly run the full quiz-taking journey against
// the SAME quiz: start attempt → answer every question (write + in-transaction
// SM-2 review) → heartbeats/integrity events → submit + grade.
// Directly targets the gap admitted in the previous evaluation: write paths
// and concurrent attempts on one quiz were never load-tested.
//
// NOTE: all VUs share ONE student token, and the backend enforces
// one-active-attempt-per-user-per-quiz. Concurrent starts therefore answer
// 409 (attempt already in progress) — the journey recovers by finalizing the
// stale attempt and retrying, which is exactly the single-session guard
// behaviour under contention. 409/429 are declared EXPECTED below so
// http_req_failed only counts real failures.

import http from 'k6/http';
import { sleep } from 'k6';

import { DURATION, THINK_TIME_S, VUS, defaultThresholds, summaryTrendStats } from '../config.js';
import { quizJourney } from '../lib/quiz_journey.js';
import { handleSummary } from '../lib/report.js';

http.setResponseCallback(http.expectedStatuses(200, 201, 202, 204, 409, 429));

export const options = {
  summaryTrendStats,
  scenarios: {
    quiz_take: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '15s', target: VUS },
        { duration: DURATION, target: VUS },
        { duration: '10s', target: 0 },
      ],
      gracefulRampDown: '15s',
      tags: { scenario: 'quiz_take' },
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.02'],
    'http_req_duration{name:POST /api/v1/quizzes/{quiz_id}/attempts}': ['p(95)<1500'],
    'http_req_duration{name:POST /api/v1/attempts/{id}/answers}': ['p(95)<800'],
    'http_req_duration{name:POST /api/v1/attempts/{id}/submit}': ['p(95)<2000'],
  },
};

export default function () {
  quizJourney();
  sleep(THINK_TIME_S + Math.random() * THINK_TIME_S);
}

export { handleSummary };
