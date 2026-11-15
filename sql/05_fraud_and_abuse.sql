-- =====================================================================
-- 05 · Fraud & promo abuse — what the current risk rules miss
-- is_fraud = confirmed via user complaint / chargeback (label lag ignored)
-- =====================================================================

-- name: fraud_overview
SELECT txn_type,
       SUM(is_fraud) AS fraud_txns,
       ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct,
       ROUND(SUM(CASE WHEN is_fraud = 1 AND status = 'SUCCESS' THEN amount END) / 1e5, 2) AS fraud_loss_lakh,
       ROUND(AVG(CASE WHEN is_fraud = 1 THEN amount END), 0) AS avg_fraud_ticket_inr,
       ROUND(AVG(CASE WHEN is_fraud = 0 THEN amount END), 0) AS avg_legit_ticket_inr
FROM transactions GROUP BY 1 ORDER BY fraud_txns DESC;

-- name: legacy_rule_performance
-- The static rules (new device + amount, late-night large transfers) as a classifier
SELECT SUM(failure_reason = 'RISK_BLOCKED')                               AS blocked,
       SUM(failure_reason = 'RISK_BLOCKED' AND is_fraud = 1)              AS true_positives,
       SUM(failure_reason = 'RISK_BLOCKED' AND is_fraud = 0)              AS legit_users_blocked,
       ROUND(100.0 * SUM(failure_reason = 'RISK_BLOCKED' AND is_fraud = 1) / SUM(failure_reason = 'RISK_BLOCKED'), 1) AS precision_pct,
       ROUND(100.0 * SUM(failure_reason = 'RISK_BLOCKED' AND is_fraud = 1) / SUM(is_fraud), 1) AS recall_pct
FROM transactions;

-- name: fraud_by_age_band
SELECT u.age_band,
       COUNT(DISTINCT u.user_id) AS users,
       COUNT(DISTINCT CASE WHEN t.is_fraud = 1 THEN u.user_id END) AS victims,
       ROUND(100.0 * COUNT(DISTINCT CASE WHEN t.is_fraud = 1 THEN u.user_id END) / COUNT(DISTINCT u.user_id), 2) AS victim_rate_pct
FROM users u JOIN transactions t ON t.user_id = u.user_id
GROUP BY 1 ORDER BY 1;

-- name: mule_account_candidates
-- Accounts receiving money from many *unrelated* payers, then sending it onward fast.
WITH inbound AS (
    SELECT payee_id AS user_id, COUNT(DISTINCT user_id) AS unique_payers, SUM(amount) AS inflow
    FROM transactions WHERE payee_type = 'USER' AND status = 'SUCCESS' GROUP BY 1
),
outbound AS (
    SELECT user_id, SUM(amount) AS outflow FROM transactions
    WHERE txn_type IN ('P2P','COLLECT') AND status = 'SUCCESS' GROUP BY 1
)
SELECT i.user_id, u.signup_ts, u.city, u.kyc_status, i.unique_payers,
       ROUND(i.inflow) AS inflow_inr, ROUND(COALESCE(o.outflow, 0)) AS outflow_inr,
       ROUND(COALESCE(o.outflow, 0) / i.inflow, 2) AS pass_through_ratio
FROM inbound i JOIN users u USING (user_id) LEFT JOIN outbound o USING (user_id)
WHERE i.unique_payers >= 8 AND i.inflow > 50000
ORDER BY i.unique_payers DESC
LIMIT 25;

-- name: shared_device_clusters
-- One device, many accounts, all on cashback campaigns = promo-abuse farm
SELECT u.device_id,
       COUNT(*) AS accounts,
       MIN(u.signup_ts) AS first_signup, MAX(u.signup_ts) AS last_signup,
       ROUND((julianday(MAX(u.signup_ts)) - julianday(MIN(u.signup_ts))) * 24, 1) AS signup_span_hours,
       SUM(u.campaign_id IS NOT NULL) AS campaign_accounts,
       (SELECT SUM(t.cashback_inr) FROM transactions t JOIN users u2 ON u2.user_id = t.user_id
         WHERE u2.device_id = u.device_id) AS cashback_extracted_inr
FROM users u
GROUP BY 1 HAVING COUNT(*) >= 3
ORDER BY cashback_extracted_inr DESC;

-- name: abuse_leakage_summary
WITH farm_devices AS (SELECT device_id FROM users GROUP BY 1 HAVING COUNT(*) >= 3)
SELECT ROUND(SUM(t.cashback_inr)) AS total_cashback_inr,
       ROUND(SUM(CASE WHEN u.device_id IN (SELECT device_id FROM farm_devices) THEN t.cashback_inr END)) AS farm_cashback_inr,
       ROUND(100.0 * SUM(CASE WHEN u.device_id IN (SELECT device_id FROM farm_devices) THEN t.cashback_inr END)
             / SUM(t.cashback_inr), 1) AS leakage_pct
FROM transactions t JOIN users u ON u.user_id = t.user_id;
