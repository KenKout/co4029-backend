// 01_smoke.js — 1 VU, 1 iteration sanity pass over every picked endpoint
// (reads + one full quiz write journey). Run this before any load test to
// validate auth, ids and payload shapes.

import { check, sleep } from 'k6';

import {
  ACCESS_TOKEN,
  COURSE_ID,
  INTERVIEW_CONFIG_ID,
  LOADTEST_USERS,
  QUIZ_ID,
  defaultThresholds,
  summaryTrendStats,
} from '../config.js';
import { apiGet } from '../lib/client.js';
import {
  completeOnboarding,
  finishInterviewSession,
  mintWarmToken,
  pollSessionState,
  startInterviewSession,
} from '../lib/interview_journey.js';
import { quizJourney } from '../lib/quiz_journey.js';
import { handleSummary } from '../lib/report.js';

export const options = {
  summaryTrendStats,
  vus: 1,
  iterations: 1,
  thresholds: { ...defaultThresholds, http_req_failed: ['rate<0.05'] },
  tags: { scenario: 'smoke' },
};

export function setup() {
  if (!ACCESS_TOKEN && !LOADTEST_USERS.length) {
    throw new Error('No token: export ACCESS_TOKEN, or use USERS_FILE (see README)');
  }
  return {};
}

export default function () {
  // Baseline + bootstrap reads
  check(apiGet('/api/v1/healthz', 'GET /api/v1/healthz'), { 'healthz 200': (r) => r.status === 200 });
  check(apiGet('/api/v1/users/me', 'GET /api/v1/users/me'), { 'users/me 200': (r) => r.status === 200 });
  check(apiGet('/api/v1/me/courses', 'GET /api/v1/me/courses'), { 'me/courses 200': (r) => r.status === 200 });
  check(apiGet('/api/v1/courses', 'GET /api/v1/courses'), { 'catalog 200': (r) => r.status === 200 });
  check(
    apiGet('/api/v1/me/notifications/unread-count', 'GET /api/v1/me/notifications/unread-count'),
    { 'unread-count 200': (r) => r.status === 200 },
  );

  // Course-learn reads
  check(
    apiGet(`/api/v1/courses/${COURSE_ID}/content`, 'GET /api/v1/courses/{id}/content'),
    { 'course content 200': (r) => r.status === 200 },
  );
  check(
    apiGet(`/api/v1/me/progress/courses/${COURSE_ID}`, 'GET /api/v1/me/progress/courses/{id}'),
    { 'course progress 200': (r) => r.status === 200 },
  );
  check(
    apiGet(`/api/v1/courses/${COURSE_ID}/quiz-progress`, 'GET /api/v1/courses/{id}/quiz-progress'),
    { 'quiz-progress 200': (r) => r.status === 200 },
  );
  check(apiGet('/api/v1/me/cards-due', 'GET /api/v1/me/cards-due'), { 'cards-due 200': (r) => r.status === 200 });
  check(apiGet('/api/v1/me/review/queue', 'GET /api/v1/me/review/queue'), { 'review queue 200': (r) => r.status === 200 });

  // Quiz reads + full write journey
  check(apiGet(`/api/v1/quizzes/${QUIZ_ID}`, 'GET /api/v1/quizzes/{quiz_id}'), { 'quiz meta 200': (r) => r.status === 200 });
  check(quizJourney(), { 'quiz journey completed': (ok) => ok });

  // Interview control-plane smoke (start mints an LLM opening turn — keep to 1)
  const session = startInterviewSession();
  if (session) {
    const sid = session.id;
    mintWarmToken(sid); // warm token is allowed during onboarding
    completeOnboarding(sid, session.onboardingStage);
    pollSessionState(sid, 1, 1);
    finishInterviewSession(sid);
  }

  sleep(1);
}

export { handleSummary };
