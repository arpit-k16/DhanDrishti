-- =====================================================================
-- 06 · Customer support — slow grievance redressal
-- =====================================================================

-- name: tickets_by_category
SELECT category,
       COUNT(*) AS tickets,
       ROUND(AVG(first_response_hours), 1) AS avg_first_response_h,
       ROUND(AVG((julianday(resolved_ts) - julianday(created_ts)) * 24), 1) AS avg_resolution_h,
       ROUND(100.0 * AVG(status = 'OPEN'), 1) AS still_open_pct,
       ROUND(AVG(csat), 2) AS avg_csat
FROM support_tickets
GROUP BY 1 ORDER BY tickets DESC;

-- name: csat_vs_resolution_time
SELECT CASE WHEN (julianday(resolved_ts) - julianday(created_ts)) * 24 <= 12 THEN '1) <= 12h'
            WHEN (julianday(resolved_ts) - julianday(created_ts)) * 24 <= 48 THEN '2) 12-48h'
            WHEN (julianday(resolved_ts) - julianday(created_ts)) * 24 <= 120 THEN '3) 2-5 days'
            ELSE '4) > 5 days' END AS resolution_bucket,
       COUNT(*) AS tickets,
       ROUND(AVG(csat), 2) AS avg_csat,
       ROUND(100.0 * AVG(csat <= 2), 1) AS detractor_pct
FROM support_tickets
WHERE resolved_ts IS NOT NULL AND csat IS NOT NULL
GROUP BY 1 ORDER BY 1;

-- name: tickets_per_1k_txns_by_bank
SELECT t.payer_bank,
       ROUND(1000.0 * COUNT(DISTINCT s.ticket_id) / COUNT(DISTINCT t.txn_id), 2) AS tickets_per_1k_txns
FROM transactions t LEFT JOIN support_tickets s ON s.txn_id = t.txn_id
GROUP BY 1 ORDER BY 2 DESC;
