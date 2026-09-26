// 07_nfr_peak.js — NFR-2.1 peak-load profile with headroom multiplier.
//
// NFR-2.1: "support on average 50 concurrent active users ... during peak
// hours without performance degradation" (TC-NFR-2.1). Video streams and AI
// interview sessions are out of scope for this run (no media material is
// published on the deployment; the interview pipeline is cost-gated and
// covered separately by 05_interview_control).
//
// The peak profile models a live class session: TOTAL_VUS concurrent active
// students, of which QUIZ_SHARE are actively taking the same quiz (write
// path) while the rest browse dashboards/curriculum (read path).
//
// Demonstrate headroom by scaling the multiplier:
//
//   TOTAL_VUS=50  DURATION=3m ./run.sh 07_nfr_peak   # 1×  NFR level
//   TOTAL_VUS=100 DURATION=3m ./run.sh 07_nfr_peak   # 2×
//   TOTAL_VUS=200 DURATION=3m ./run.sh 07_nfr_peak   # 4×  headroom proof
//
// Verdict criteria (mapped to the NFR tables):
//   * avg duration < 500 ms  → NFR-2.2 query target
//   * p(95)     < 2000 ms    → NFR-1.4 dashboard target
//   * error rate < 1%        → "without performance degradation"

import http from 'k6/http';
import { sleep } from 'k6';

import { DURATION, THINK_TIME_S, summaryTrendStats } from '../config.js';
import { browseIteration } from '../lib/browse_journey.js';
import { quizJourney } from '../lib/quiz_journey.js';
import { handleSummary } from '../lib/report.js';

const TOTAL_VUS = Number(__ENV.TOTAL_VUS || 50);
const QUIZ_SHARE = Number(__ENV.QUIZ_SHARE || 0.2);

const QUIZ_VUS = Math.max(1, Math.round(TOTAL_VUS * QUIZ_SHARE));
const BROWSE_VUS = Math.max(1, TOTAL_VUS - QUIZ_VUS);

// Quiz writes legitimately return 409 (guard) / 429 (cooldown) under stress.
http.setResponseCallback(http.expectedStatuses(200, 201, 202, 204, 409, 429));

export const options = {
  summaryTrendStats,
  scenarios: {
    browse: {
      executor: 'ramping-vus',
      exec: 'browse',
      startVUs: 0,
      stages: [
        { duration: '20s', target: BROWSE_VUS },
        { duration: DURATION, target: BROWSE_VUS },
        { duration: '10s', target: 0 },
      ],
      gracefulRampDown: '10s',
      tags: { scenario: 'nfr_peak_browse' },
    },
    quiz_take: {
      executor: 'ramping-vus',
      exec: 'quizTake',
      startVUs: 0,
      stages: [
        { duration: '20s', target: QUIZ_VUS },
        { duration: DURATION, target: QUIZ_VUS },
        { duration: '10s', target: 0 },
      ],
      gracefulRampDown: '15s',
      tags: { scenario: 'nfr_peak_quiz' },
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.01'],
    http_req_duration: ['avg<500', 'p(95)<2000'],
  },
};

export function browse() {
  browseIteration();
}

export function quizTake() {
  quizJourney();
  sleep(THINK_TIME_S + Math.random() * THINK_TIME_S);
}

export { handleSummary };
