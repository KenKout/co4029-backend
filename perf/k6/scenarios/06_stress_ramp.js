// 06_stress_ramp.js — capacity discovery.
// Mixed traffic (70% browse reads / 30% full quiz write journeys) ramped past
// the 100 VU ceiling of the previous evaluation to find the actual breaking
// point of the single-Uvicorn deployment and evidence NFR-2.1 headroom.
//
// Default profile: 0 → 50 → 100 → 150 → 200 VUs. Track where p95 blows past
// the threshold and the error rate leaves zero — that saturation point goes
// straight into the evaluation chapter.

import { check, sleep } from 'k6';

import { THINK_TIME_S, defaultThresholds, summaryTrendStats } from '../config.js';
import { apiGet } from '../lib/client.js';
import { quizJourney, randomOf } from '../lib/quiz_journey.js';
import { handleSummary } from '../lib/report.js';

const PROFILE = __ENV.RAMP_PROFILE || '50:30s,100:1m,150:1m,200:1m';

function stagesFrom(profile) {
  return profile.split(',').map((step) => {
    const [target, duration] = step.split(':');
    return { target: Number(target), duration };
  });
}

export const options = {
  summaryTrendStats,
  scenarios: {
    mixed: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: stagesFrom(PROFILE),
      gracefulRampDown: '20s',
      tags: { scenario: 'stress_ramp' },
    },
  },
  thresholds: {
    ...defaultThresholds,
    http_req_failed: ['rate<0.05'], // relaxation expected near saturation
  },
};

export default function () {
  if (Math.random() < 0.3) {
    quizJourney();
  } else {
    check(apiGet('/api/v1/users/me', 'GET /api/v1/users/me'), { 'users/me 200': (r) => r.status === 200 });
    check(apiGet('/api/v1/me/courses', 'GET /api/v1/me/courses'), { 'me/courses 200': (r) => r.status === 200 });
    check(
      apiGet('/api/v1/me/notifications/unread-count', 'GET /api/v1/me/notifications/unread-count'),
      { 'unread-count 200': (r) => r.status === 200 },
    );
    const browse = randomOf([
      { path: '/api/v1/courses', tag: 'GET /api/v1/courses' },
      { path: '/api/v1/me/cards-due', tag: 'GET /api/v1/me/cards-due' },
    ]);
    check(apiGet(browse.path, browse.tag), { 'browse 200': (r) => r.status === 200 });
  }
  sleep(THINK_TIME_S);
}

export { handleSummary };
