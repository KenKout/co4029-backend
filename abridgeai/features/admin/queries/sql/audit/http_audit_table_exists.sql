-- Probe: does the T0.23 HTTP audit log table exist?
SELECT 1
FROM information_schema.tables
WHERE table_schema = current_schema()
  AND table_name = 'http_audit_log'
LIMIT 1;
