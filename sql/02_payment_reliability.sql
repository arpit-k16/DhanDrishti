-- =====================================================================
-- 02 · Payment reliability — where and when do payments break?
-- Technical declines (bank/network side) are separated from business
-- declines (wrong PIN, low balance) because only the first is fixable
-- by the app / bank partnership team.
-- =====================================================================

-- name: decline_reasons
SELECT failure_reason,
       CASE WHEN COALESCE(failure_reason,'') IN ('BANK_SERVER_DOWN','TIMEOUT','NETWORK_ERROR') THEN 'technical'
            WHEN failure_reason = 'RISK_BLOCKED' THEN 'risk'
            ELSE 'business (user-side)' END                    AS decline_type,
       COUNT(*)                                                AS declines,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)      AS share_pct,
       ROUND(SUM(amount) / 1e5, 2)                             AS value_lakh_inr
FROM transactions
WHERE status = 'FAILED'
GROUP BY 1 ORDER BY declines DESC;

-- name: technical_decline_by_bank
-- Remitter (payer) bank view. Compares each bank to the platform average.
WITH b AS (
    SELECT payer_bank,
           COUNT(*) AS attempts,
           SUM(COALESCE(failure_reason,'') IN ('BANK_SERVER_DOWN','TIMEOUT','NETWORK_ERROR')) AS tech_declines
    FROM transactions GROUP BY 1
)
SELECT b.payer_bank, bk.bank_name, attempts, tech_declines,
       ROUND(100.0 * tech_declines / attempts, 2) AS tech_decline_pct,
       ROUND((1.0 * tech_declines / attempts)
             / (SUM(tech_declines) OVER () * 1.0 / SUM(attempts) OVER ()), 2) AS x_platform_avg
FROM b JOIN banks bk ON bk.bank_code = b.payer_bank
ORDER BY tech_decline_pct DESC;

-- name: bank_peak_vs_offpeak
-- Which banks degrade under evening load (19:00–22:59)?
SELECT payer_bank,
       ROUND(100.0 * AVG(CASE WHEN CAST(strftime('%H', ts) AS INT) NOT BETWEEN 19 AND 22
             THEN COALESCE(failure_reason,'') IN ('BANK_SERVER_DOWN','TIMEOUT','NETWORK_ERROR') END), 2) AS offpeak_tech_decline_pct,
       ROUND(100.0 * AVG(CASE WHEN CAST(strftime('%H', ts) AS INT) BETWEEN 19 AND 22
             THEN COALESCE(failure_reason,'') IN ('BANK_SERVER_DOWN','TIMEOUT','NETWORK_ERROR') END), 2) AS peak_tech_decline_pct,
       ROUND(AVG(CASE WHEN CAST(strftime('%H', ts) AS INT) BETWEEN 19 AND 22 THEN latency_ms END), 0) AS peak_avg_latency_ms
FROM transactions
GROUP BY 1
ORDER BY peak_tech_decline_pct - offpeak_tech_decline_pct DESC;

-- name: hourly_decline_curve
SELECT CAST(strftime('%H', ts) AS INT) AS hour,
       COUNT(*) AS attempts,
       ROUND(100.0 * AVG(COALESCE(failure_reason,'') IN ('BANK_SERVER_DOWN','TIMEOUT','NETWORK_ERROR')), 2) AS tech_decline_pct
FROM transactions GROUP BY 1 ORDER BY 1;

-- name: month_start_effect
-- Salary-day load (days 1–3) vs rest of month
SELECT CASE WHEN CAST(strftime('%d', ts) AS INT) <= 3 THEN 'day 1-3' ELSE 'day 4+' END AS period,
       COUNT(*) AS attempts,
       ROUND(100.0 * AVG(COALESCE(failure_reason,'') IN ('BANK_SERVER_DOWN','TIMEOUT','NETWORK_ERROR')), 2) AS tech_decline_pct
FROM transactions GROUP BY 1;

-- name: lost_gmv_from_tech_declines
-- A technical decline is "lost" if the user did NOT pay the same payee successfully within 30 minutes
WITH f AS (
    SELECT f.*, EXISTS (SELECT 1 FROM transactions t
                        WHERE t.user_id = f.user_id AND t.payee_id = f.payee_id AND t.status = 'SUCCESS'
                          AND t.ts > f.ts AND t.ts <= datetime(f.ts, '+30 minutes')) AS recovered
    FROM transactions f
    WHERE COALESCE(f.failure_reason,'') IN ('BANK_SERVER_DOWN','TIMEOUT','NETWORK_ERROR')
)
SELECT strftime('%Y-%m', ts) AS month,
       COUNT(*) AS tech_declines,
       ROUND(SUM(amount) / 1e5, 2) AS declined_value_lakh,
       ROUND(SUM(CASE WHEN recovered = 0 THEN amount END) / 1e5, 2) AS lost_gmv_lakh,
       ROUND(100.0 * AVG(recovered = 0), 1) AS abandoned_pct
FROM f GROUP BY 1 ORDER BY 1;

-- name: retry_behaviour
-- Did the user successfully pay the same payee within 30 minutes of a technical failure?
WITH f AS (
    SELECT txn_id, user_id, payee_id, ts
    FROM transactions
    WHERE COALESCE(failure_reason,'') IN ('BANK_SERVER_DOWN','TIMEOUT','NETWORK_ERROR')
)
SELECT COUNT(*) AS tech_failures,
       SUM(EXISTS (SELECT 1 FROM transactions t
                   WHERE t.user_id = f.user_id AND t.payee_id = f.payee_id AND t.status = 'SUCCESS'
                     AND t.ts > f.ts AND t.ts <= datetime(f.ts, '+30 minutes'))) AS recovered_within_30m,
       ROUND(100.0 * SUM(EXISTS (SELECT 1 FROM transactions t
                   WHERE t.user_id = f.user_id AND t.payee_id = f.payee_id AND t.status = 'SUCCESS'
                     AND t.ts > f.ts AND t.ts <= datetime(f.ts, '+30 minutes'))) / COUNT(*), 1) AS recovery_pct
FROM f;
