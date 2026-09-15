# Interview audio recording deployment

Interview recording is **disabled by default**. The interview remains usable when
recording is disabled, consent is declined, LiveKit Egress is unavailable, or an
output fails validation.

## Provisioning order

1. Deploy the backend code and run `alembic upgrade head` on both the development
   and test databases. Confirm `alembic_version` is
   `0124_interview_recording_consent` and the `interview_recordings` table plus
   consent columns exist.
2. Provision LiveKit Egress separately from the interview agent. Use the same
   LiveKit API key/secret as the backend and configure the Egress webhook to
   `POST /internal/livekit/webhook`.
3. Provision a private, encrypted S3 bucket (or Garage bucket) and set
   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_BUCKET_NAME`. Set
   `AWS_ENDPOINT_URL` for server-side Garage access and
   `AWS_PUBLIC_ENDPOINT_URL` for browser presigned GET URLs when needed.
4. Restrict the Egress credential to `PutObject` under
   `interviews/recordings/*`. The API/ARQ credential needs `HeadObject`,
   `GetObject` presigning, and idempotent `DeleteObject` only for that prefix.
   Do not grant browser credentials or list-bucket access.
5. Add a bucket lifecycle rule for `interviews/recordings/*` at 30 days. The
   application reconciliation job is still authoritative for clearing the
   playback FKs and writing the `expired` tombstone.
6. Configure CORS for browser `GET`/`HEAD` only. Do not allow browser `PUT` for
   this provider-owned prefix.
7. Verify LiveKit webhook delivery with a consented pilot session while
   `INTERVIEW_RECORDING_ENABLED=false` first (signature rejection and route
   health), then enable `LIVEKIT_WEBHOOK_ENABLED=true` and confirm accepted
   signed events in the structured logs.
8. In a non-production environment set:

   ```dotenv
   INTERVIEW_RECORDING_ENABLED=true
   INTERVIEW_RECORDING_POLICY_VERSION=2026-09-15
   INTERVIEW_RECORDING_FORMAT=mp3
   INTERVIEW_RECORDING_S3_PREFIX=interviews/recordings
   INTERVIEW_RECORDING_RETENTION_DAYS=30
   LIVEKIT_WEBHOOK_ENABLED=true
   INTERVIEW_RECORDING_RECONCILE_MAX_ATTEMPTS=12
   ```

9. Run the opt-in smoke test: candidate accepts the current policy, joins a
   voice room, speaks briefly, finishes or times out, the Egress callback is
   accepted, `head_object` validates a non-empty allowed audio MIME, exactly one
   `storage_objects` row is attached, and the teacher Transcript tab plays the
   signed URL. Repeat with declined consent and verify no Egress job starts.
10. Only after the pilot verifies authorization, callback reconciliation,
    pointer clearing, and retention deletion should the feature be enabled for
    a broader course cohort.

## Operational invariants

- Consent is affirmative, policy-versioned, timestamped, and audio-only. A stale
  policy version never starts Egress.
- `InterviewSession.recording_object_id` is only the final playback pointer.
  Provider Egress IDs and storage coordinates stay in the recording lifecycle
  row and never enter transcript/session response DTOs.
- Callback delivery is authenticated before decoding. Duplicate and
  out-of-order callbacks are safe; provider/API outages are retried by ARQ with
  a bounded ceiling.
- URL responses are teacher-authorized, short-lived, inline, and `no-store`.
  Every URL mint is present in the HTTP audit trail and emits the structured
  `interview.recording.url_mint` event.
- Retention deletes the object before committing the `expired` tombstone and
  clears both playback references. The tombstone remains for audit.
