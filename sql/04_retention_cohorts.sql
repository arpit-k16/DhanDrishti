-- =====================================================================
-- 04 · Retention — cohorts, acquisition channel, and the cost of a bad
-- first week
-- Churned = no transaction in the last 30 days of the window (Jun 2026)
-- =====================================================================

-- name: cohort_retention_matrix
-- % of each signup-month cohort that transacted N months after signup
WITH a AS (
    SELECT u.user_id,
           strftime('%Y-%m', u.signup_ts) AS cohort,
           (CAST(strftime('%Y', t.ts) AS INT) - CAST(strftime('%Y', u.signup_ts) AS INT)) * 12
         + (CAST(strftime('%m', t.ts) AS INT) - CAST(strftime('%m', u.signup_ts) AS INT)) AS m
    FROM users u JOIN transactions t ON t.user_id = u.user_id
    GROUP BY 1, 2, 3
),
size AS (SELECT cohort, COUNT(DISTINCT user_id) AS n FROM a WHERE m = 0 GROUP BY 1)
SELECT a.cohort, s.n AS cohort_size,
       CASE WHEN date(a.cohort || '-01', '+1 months') <= '2026-06-01'
            THEN ROUND(100.0 * COUNT(DISTINCT CASE WHEN m = 1 THEN user_id END) / s.n, 1) END AS m1,
       CASE WHEN date(a.cohort || '-01', '+2 months') <= '2026-06-01'
            THEN ROUND(100.0 * COUNT(DISTINCT CASE WHEN m = 2 THEN user_id END) / s.n, 1) END AS m2,
       CASE WHEN date(a.cohort || '-01', '+3 months') <= '2026-06-01'
            THEN ROUND(100.0 * COUNT(DISTINCT CASE WHEN m = 3 THEN user_id END) / s.n, 1) END AS m3,
       CASE WHEN date(a.cohort || '-01', '+4 months') <= '2026-06-01'
            THEN ROUND(100.0 * COUNT(DISTINCT CASE WHEN m = 4 THEN user_id END) / s.n, 1) END AS m4,
       CASE WHEN date(a.cohort || '-01', '+5 months') <= '2026-06-01'
            THEN ROUND(100.0 * COUNT(DISTINCT CASE WHEN m = 5 THEN user_id END) / s.n, 1) END AS m5
FROM a JOIN size s USING (cohort)
GROUP BY 1, 2 ORDER BY 1;

-- name: churn_by_channel
WITH l AS (SELECT user_id, MAX(ts) AS last_ts, COUNT(*) AS txns FROM transactions GROUP BY 1)
SELECT u.acquisition_channel,
       COUNT(*) AS users,
       ROUND(AVG(l.txns), 1) AS avg_lifetime_txns,
       ROUND(100.0 * AVG(julianday('2026-06-30') - julianday(l.last_ts) > 30), 1) AS churned_pct,
       ROUND(100.0 * AVG(l.txns = 1), 1) AS one_and_done_pct
FROM users u JOIN l USING (user_id)
WHERE u.signup_ts < '2026-04-30'
GROUP BY 1 ORDER BY churned_pct DESC;

-- name: first_week_failure_impact
-- Users who hit a technical decline in their first 14 days vs those who didn't
WITH fw AS (
    SELECT u.user_id,
           MAX(COALESCE(t.failure_reason,'') IN ('BANK_SERVER_DOWN','TIMEOUT','NETWORK_ERROR')
               AND julianday(t.ts) - julianday(u.signup_ts) <= 14) AS early_fail
    FROM users u JOIN transactions t ON t.user_id = u.user_id
    WHERE u.signup_ts < '2026-04-30'
    GROUP BY 1
),
l AS (SELECT user_id, MAX(ts) AS last_ts FROM transactions GROUP BY 1)
SELECT CASE WHEN early_fail = 1 THEN 'hit a failure in first 14 days' ELSE 'clean first 14 days' END AS cohort,
       COUNT(*) AS users,
       ROUND(100.0 * AVG(julianday('2026-06-30') - julianday(l.last_ts) > 30), 1) AS churned_pct
FROM fw JOIN l USING (user_id) GROUP BY 1;

-- name: campaign_roi
-- Cost per retained user: cashback spent / users still active in June
WITH cb AS (
    SELECT u.campaign_id, SUM(t.cashback_inr) AS cashback_spent
    FROM transactions t JOIN users u ON u.user_id = t.user_id
    WHERE t.cashback_inr > 0 GROUP BY 1
),
act AS (
    SELECT u.campaign_id, COUNT(*) AS acquired,
           SUM(EXISTS (SELECT 1 FROM transactions t WHERE t.user_id = u.user_id AND t.ts >= '2026-06-01')) AS active_in_june
    FROM users u WHERE u.campaign_id IS NOT NULL GROUP BY 1
)
SELECT c.code, a.acquired, a.active_in_june,
       ROUND(100.0 * a.active_in_june / a.acquired, 1) AS retained_pct,
       cb.cashback_spent,
       ROUND(cb.cashback_spent / a.acquired, 1) AS cost_per_acquired_inr,
       ROUND(cb.cashback_spent / NULLIF(a.active_in_june, 0), 1) AS cost_per_retained_inr
FROM act a JOIN cb USING (campaign_id) JOIN campaigns c USING (campaign_id);
