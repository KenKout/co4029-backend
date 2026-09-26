// 04_sr_review.js — spaced-repetition review session.
// GET /me/review/queue (Redis-cached backlog) → POST /me/review/{question_id}
// (SM-2 write per card). NOTE: reviews mutate this user's real SR schedule;
// sustained runs push cards into cooldown and the API answers 429
// (all_cards_in_cooldown) — the scenario treats that as a graceful stop.

import { check, sleep } from 'k6';

import { DURATION, SR_CARDS_PER_ITER, THINK_TIME_S, VUS, defaultThresholds, summaryTrendStats } from '../config.js';
import { apiGet, apiPost } from '../lib/client.js';
import { randomOf } from '../lib/quiz_journey.js';
import { handleSummary } from '../lib/report.js';

export const options = {
  summaryTrendStats,
  scenarios: {
    sr_review: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '15s', target: VUS },
        { duration: DURATION, target: VUS },
        { duration: '10s', target: 0 },
      ],
      gracefulRampDown: '10s',
      tags: { scenario: 'sr_review' },
    },
  },
  thresholds: {
    ...defaultThresholds,
    'http_req_duration{name:POST /api/v1/me/review/{question_id}}': ['p(95)<800'],
  },
};

export default function () {
  const queue = apiGet('/api/v1/me/review/queue', 'GET /api/v1/me/review/queue');
  if (!check(queue, { 'review queue 200': (r) => r.status === 200 })) return;

  const items = queue.json().items || [];
  if (!items.length) {
    sleep(THINK_TIME_S); // nothing due — idle student
    return;
  }

  const cards = items.slice(0, SR_CARDS_PER_ITER);
  for (const card of cards) {
    const optionId = card.options && card.options.length ? randomOf(card.options).id : null;
    const res = apiPost(
      `/api/v1/me/review/${card.question_id}`,
      {
        selected_option_id: optionId,
        t_actual_ms: 1500 + Math.floor(Math.random() * 6000),
        hint_used: false,
      },
      'POST /api/v1/me/review/{question_id}',
    );
    // 429 = every card now in cooldown (self-limiting) — stop reviewing.
    if (res.status === 429) break;
    check(res, { 'review submitted 200': (r) => r.status === 200 });
    sleep(THINK_TIME_S / 2);
  }

  check(apiGet('/api/v1/me/sr-dashboard-summary', 'GET /api/v1/me/sr-dashboard-summary'), {
    'sr summary 200': (r) => r.status === 200,
  });
  sleep(THINK_TIME_S);
}

export { handleSummary };
