# Quiz session guard and required camera

The quiz take flow uses two layers of deadline-safe protection:

- Redis key `quiz_attempt_session:{attempt_id}` stores the owning
  `CurrentUser.session_id` for 90 seconds. Claim, heartbeat, takeover, and
  release are compare-and-set Lua operations. Answer saves, integrity events,
  and submission fail closed when Redis is unavailable.
- The frontend holds a Web Locks key for the attempt and coordinates transfer
  with `BroadcastChannel`. Browsers without Web Locks use the channel leader
  handshake fallback.

The server owns the authoritative attempt timer. Closing or losing focus never
pauses it. An accidental close leaves the attempt resumable; a deliberate exit
flushes the current answer, releases the Redis key, stops the local camera, and
leaves the attempt `in_progress`.

## Required camera

`Quiz.require_camera` defaults to `false`. When enabled, the learner must grant
`getUserMedia({ video: true, audio: false })` before starting or resuming. The
browser keeps only the local stream; no camera media is recorded, uploaded,
streamed, stored, or analysed. A missing, ended, muted, or otherwise inactive
video track blocks the workspace after a four-second recovery grace period.
The timer and Redis heartbeat continue during that block.

The attempt's `integrity_policy_snapshot` also records `require_camera`, so the
policy observed by an attempt is stable even if the quiz is edited later.

## Accepted boundary

This is not a persistent assessment lease. A copied access token reusing the
same authentication session id can bypass the distinction between two clients;
Redis loss or administrative flushes also lose ownership. A modified client can
bypass frontend tab coordination. These are accepted deadline risks.

Future work is the shared persistent assessment lease for quiz and interview,
with opaque client credentials, generations, audit history, Redis-reset
invalidation, and server-side interview/LiveKit enforcement.
