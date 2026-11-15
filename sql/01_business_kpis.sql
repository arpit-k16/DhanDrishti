-- =====================================================================
-- 01 · Business KPIs — the "health check" a payments analyst opens first
-- Dialect: SQLite (window functions; works the same in Postgres/BigQuery
-- with minor date-function changes)
-- =====================================================================

-- name: monthly_kpis
-- GMV counts only successful money movement. Success rate is the #1 UPI KPI.
SELECT
    strftime('%Y-%m', ts)                                        AS month,
    COUNT(*)                                                     AS txn_attempts,
    SUM(status = 'SUCCESS')                                      AS txn_success,
    ROUND(100.0 * SUM(status = 'SUCCESS') / COUNT(*), 2)         AS success_rate_pct,
    ROUND(SUM(CASE WHEN status = 'SUCCESS' THEN amount END) / 1e5, 2) AS gmv_lakh_inr,
    COUNT(DISTINCT user_id)                                      AS monthly_active_users,
    ROUND(1.0 * SUM(status = 'SUCCESS') / COUNT(DISTINCT user_id), 1) AS txns_per_mau,
    ROUND(SUM(cashback_inr), 0)                                  AS cashback_paid_inr
FROM transactions
GROUP BY 1
ORDER BY 1;

-- name: mom_growth
-- Month-over-month growth with LAG()
WITH m AS (
    SELECT strftime('%Y-%m', ts) AS month,
           SUM(CASE WHEN status = 'SUCCESS' THEN amount END) AS gmv,
           COUNT(DISTINCT user_id) AS mau
    FROM transactions GROUP BY 1
)
SELECT month,
       ROUND(gmv / 1e5, 2) AS gmv_lakh_inr,
       ROUND(100.0 * (gmv - LAG(gmv) OVER (ORDER BY month)) / LAG(gmv) OVER (ORDER BY month), 1) AS gmv_mom_pct,
       mau,
       ROUND(100.0 * (mau - LAG(mau) OVER (ORDER BY month)) / LAG(mau) OVER (ORDER BY month), 1) AS mau_mom_pct
FROM m ORDER BY month;

-- name: txn_type_mix
-- P2P is high-volume but earns nothing (zero MDR); P2M / bills are where the business is.
SELECT txn_type,
       COUNT(*)                                                     AS attempts,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)           AS share_of_volume_pct,
       ROUND(AVG(amount), 0)                                        AS avg_ticket_inr,
       ROUND(100.0 * SUM(CASE WHEN status='SUCCESS' THEN amount END)
             / SUM(SUM(CASE WHEN status='SUCCESS' THEN amount END)) OVER (), 1) AS share_of_gmv_pct,
       ROUND(100.0 * SUM(status = 'SUCCESS') / COUNT(*), 2)        AS success_rate_pct
FROM transactions
GROUP BY 1 ORDER BY attempts DESC;

-- name: merchant_category_gmv
SELECT m.category,
       COUNT(*) AS txns,
       ROUND(SUM(t.amount) / 1e5, 2) AS gmv_lakh_inr,
       ROUND(AVG(t.amount), 0) AS avg_ticket_inr,
       COUNT(DISTINCT t.user_id) AS unique_buyers
FROM transactions t
JOIN merchants m ON m.merchant_id = t.payee_id
WHERE t.payee_type = 'MERCHANT' AND t.status = 'SUCCESS'
GROUP BY 1 ORDER BY gmv_lakh_inr DESC;

-- name: city_tier_engagement
SELECT u.city_tier,
       COUNT(DISTINCT u.user_id) AS users,
       ROUND(1.0 * COUNT(t.txn_id) / COUNT(DISTINCT u.user_id), 1) AS txns_per_user,
       ROUND(AVG(t.amount), 0) AS avg_ticket_inr,
       ROUND(100.0 * AVG(t.status = 'SUCCESS'), 2) AS success_rate_pct
FROM users u
LEFT JOIN transactions t ON t.user_id = u.user_id
GROUP BY 1 ORDER BY 1;
