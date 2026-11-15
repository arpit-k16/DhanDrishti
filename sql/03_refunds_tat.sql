-- =====================================================================
-- 03 · "Money debited, payment failed" — refund turnaround (TAT)
-- RBI's TAT harmonisation framework (2019): failed UPI debits must be
-- auto-reversed by T+1, else Rs 100/day compensation is payable.
-- =====================================================================

-- name: refund_tat_buckets
SELECT CASE WHEN tat_hours <= 4   THEN '1) <= 4h'
            WHEN tat_hours <= 24  THEN '2) 4-24h (within T+1)'
            WHEN tat_hours <= 72  THEN '3) 1-3 days (breach)'
            WHEN tat_hours <= 168 THEN '4) 3-7 days (breach)'
            ELSE                       '5) > 7 days (breach)' END AS tat_bucket,
       COUNT(*) AS refunds,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS share_pct,
       ROUND(SUM(amount) / 1e5, 2) AS stuck_value_lakh
FROM refunds GROUP BY 1 ORDER BY 1;

-- name: tat_breach_by_bank
SELECT t.payer_bank, b.bank_name,
       COUNT(*) AS debited_failures,
       ROUND(AVG(r.tat_hours), 1) AS avg_tat_h,
       ROUND(100.0 * AVG(r.tat_hours > 24), 1) AS breach_pct,
       SUM(r.compensation_due_inr) AS compensation_liability_inr
FROM refunds r
JOIN transactions t ON t.txn_id = r.txn_id
JOIN banks b ON b.bank_code = t.payer_bank
GROUP BY 1, 2 ORDER BY breach_pct DESC;

-- name: refund_percentiles
-- P50 / P90 / P99 TAT using window ranking (no PERCENTILE_CONT in SQLite)
WITH r AS (
    SELECT tat_hours, ROW_NUMBER() OVER (ORDER BY tat_hours) AS rn, COUNT(*) OVER () AS n FROM refunds
)
SELECT MAX(CASE WHEN rn = CAST(0.50 * n AS INT) THEN tat_hours END) AS p50_h,
       MAX(CASE WHEN rn = CAST(0.90 * n AS INT) THEN tat_hours END) AS p90_h,
       MAX(CASE WHEN rn = CAST(0.99 * n AS INT) THEN tat_hours END) AS p99_h
FROM r;

-- name: refund_delay_and_churn
-- Do users whose money was stuck > T+1 stop using the app?
WITH last_seen AS (SELECT user_id, MAX(ts) AS last_ts FROM transactions GROUP BY 1),
     exp AS (
        SELECT user_id, MAX(tat_hours > 24) AS had_breach, MIN(debit_ts) AS first_debit
        FROM refunds GROUP BY 1)
SELECT CASE WHEN had_breach = 1 THEN 'refund stuck > T+1' ELSE 'refund within T+1' END AS cohort,
       COUNT(*) AS users,
       ROUND(100.0 * AVG(julianday('2026-06-30') - julianday(l.last_ts) > 30), 1) AS churned_pct
FROM exp e JOIN last_seen l USING (user_id)
WHERE e.first_debit < '2026-05-01'
GROUP BY 1;
