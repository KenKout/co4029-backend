-- Semantic auth-event search (FR-1.6): the typed trail behind /admin/audit/auth-events.
--
-- Unlike /http (whose rows are request-shaped and whose "login_failure" kind is
-- DERIVED from path + status), these rows carry the semantic event name. A
-- "MFA verified" row here was written BY the MFA service in the same
-- transaction as the verification, not inferred afterwards.
--
-- :since is required (bound by the router); :until is exclusive-optional.
-- Filters: subject (:user_id), performer (:actor_user_id), event type, and
-- organization (role/status events carry the org edge; login/MFA rows are
-- NULL there and pass the filter only when :organization_id IS NULL).
SELECT
    e.id,
    e.event_type,
    e.user_id,
    e.actor_user_id,
    e.organization_id,
    e.session_id,
    e.detail,
    e.occurred_at
FROM auth_events e
WHERE e.occurred_at >= CAST(:since AS timestamptz)
  AND (CAST(:until AS timestamptz) IS NULL
       OR e.occurred_at < CAST(:until AS timestamptz))
  AND (CAST(:user_id AS uuid) IS NULL OR e.user_id = CAST(:user_id AS uuid))
  AND (CAST(:actor_user_id AS uuid) IS NULL OR e.actor_user_id = CAST(:actor_user_id AS uuid))
  AND (CAST(:event_type AS text) IS NULL OR e.event_type = CAST(:event_type AS text))
  AND (CAST(:organization_id AS uuid) IS NULL
       OR e.organization_id = CAST(:organization_id AS uuid))
ORDER BY e.occurred_at DESC
LIMIT :limit
