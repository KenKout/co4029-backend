// 02_browse_mix.js — learner bootstrap + dashboard read mix.
// Replays the request pattern of a student opening the platform: profile,
// enrollments, notification badge, course dashboard, curriculum, SR state.
// This replaces the single-endpoint GET tests of the previous iteration with
// a realistic mixed read profile.

import { DURATION, VUS, defaultThresholds, summaryTrendStats } from '../config.js';
import { browseIteration } from '../lib/browse_journey.js';
import { handleSummary } from '../lib/report.js';

export const options = {
  summaryTrendStats,
  scenarios: {
    browse: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '20s', target: VUS },
        { duration: DURATION, target: VUS },
        { duration: '10s', target: 0 },
      ],
      gracefulRampDown: '10s',
      tags: { scenario: 'browse_mix' },
    },
  },
  thresholds: defaultThresholds,
};

export default function () {
  browseIteration();
}

export { handleSummary };
