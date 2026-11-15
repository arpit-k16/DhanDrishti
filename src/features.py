"""Feature engineering shared by the notebooks.

Every fraud feature uses ONLY information available before the transaction
(no look-ahead leakage): cumulative counts are shifted, rolling windows are
closed on the left.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
TECH = ["BANK_SERVER_DOWN", "TIMEOUT", "NETWORK_ERROR"]
END = pd.Timestamp("2026-06-30 23:59:59")
AGE_ORD = {"18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55+": 4}


def load():
    d = {}
    d["users"] = pd.read_csv(f"{RAW}/users.csv", parse_dates=["signup_ts"])
    d["merchants"] = pd.read_csv(f"{RAW}/merchants.csv", parse_dates=["onboard_date"])
    d["banks"] = pd.read_csv(f"{RAW}/banks.csv")
    d["campaigns"] = pd.read_csv(f"{RAW}/campaigns.csv")
    d["tx"] = pd.read_csv(f"{RAW}/transactions.csv", parse_dates=["ts"])
    d["refunds"] = pd.read_csv(f"{RAW}/refunds.csv", parse_dates=["debit_ts", "refund_ts"])
    d["tickets"] = pd.read_csv(f"{RAW}/support_tickets.csv", parse_dates=["created_ts", "resolved_ts"])
    return d


def legacy_rule(df):
    """The app's current static rules, re-expressed on the feature table."""
    return (((df.is_new_device == 1) & (df.amount > 3000))
            | ((df.amount > 15000) & ((df.hour >= 23) | (df.hour < 5)))
            | (df.amount > 40000)).astype(int)


def fraud_features(d):
    # stable, deterministic order: payments in the same second are ordered by txn_id
    tx = d["tx"].sort_values(["ts", "txn_id"], kind="stable").reset_index(drop=True).copy()
    u = d["users"].set_index("user_id")
    m = d["merchants"].set_index("merchant_id")

    tx["hour"] = tx.ts.dt.hour
    tx["is_night"] = ((tx.hour >= 23) | (tx.hour < 5)).astype(int)
    tx["log_amount"] = np.log1p(tx.amount)
    tx["round_amount"] = ((tx.amount % 500 == 0) & (tx.amount >= 500)).astype(int)

    # payer profile
    tx["payer_age_days"] = (tx.ts - tx.user_id.map(u.signup_ts)).dt.total_seconds() / 86400
    tx["payer_age_band"] = tx.user_id.map(u.age_band).map(AGE_ORD)
    tx["payer_city_tier"] = tx.user_id.map(u.city_tier)
    tx["payer_kyc_min"] = (tx.user_id.map(u.kyc_status) == "MIN").astype(int)

    # payer history (strictly prior)
    g = tx.groupby("user_id")
    tx["payer_prior_txns"] = g.cumcount()
    prior_sum = g.amount.cumsum() - tx.amount
    tx["payer_prior_mean_amt"] = (prior_sum / tx.payer_prior_txns.replace(0, np.nan))
    tx["amt_vs_payer_mean"] = (tx.amount / tx.payer_prior_mean_amt).fillna(1.0).clip(0, 200)
    tx["payer_prior_max_amt"] = g.amount.transform(lambda s: s.shift().cummax()).fillna(0)
    tx["exceeds_prior_max"] = (tx.amount > tx.payer_prior_max_amt).astype(int)
    # velocity: payer txns in previous 1h and 24h
    # velocity: payer's own txns in the previous 1h / 24h (sorted-key searchsorted, O(n log n))
    ucode = tx.user_id.astype("category").cat.codes.values.astype(np.int64)
    secs = tx.ts.values.astype("datetime64[s]").astype(np.int64)
    order = np.lexsort((secs, ucode))
    key = ucode[order] * 10**11 + secs[order]
    pos = np.arange(len(key))
    for w, sec in [("1h", 3600), ("24h", 86400)]:
        left = np.searchsorted(key, key - sec, side="left")
        out = np.empty(len(key))
        out[order] = pos - left
        tx[f"payer_txns_{w}"] = out

    # payee profile
    is_user = tx.payee_type == "USER"
    tx["payee_is_merchant"] = (~is_user).astype(int)
    payee_start = np.where(is_user, tx.payee_id.map(u.signup_ts), tx.payee_id.map(m.onboard_date))
    tx["payee_age_days"] = ((tx.ts - pd.to_datetime(payee_start)).dt.total_seconds() / 86400).clip(lower=0)
    tx["payee_small_kyc"] = np.where(~is_user, (tx.payee_id.map(m.kyc_tier) == "SMALL").astype(int), 0)
    # first time this payer pays this payee; distinct payers the payee had before now
    pair_first = ~tx.duplicated(["user_id", "payee_id"])
    tx["first_time_payee"] = pair_first.astype(int)
    tx["payee_prior_unique_payers"] = pair_first.astype(int).groupby(tx.payee_id).cumsum() - pair_first.astype(int)
    tx["payee_prior_txns"] = tx.groupby("payee_id").cumcount()

    for t in ["P2P", "P2M", "COLLECT", "BILL", "RECHARGE"]:
        tx[f"type_{t}"] = (tx.txn_type == t).astype(int)
    tx["legacy_rule_flag"] = legacy_rule(tx)
    return tx


FRAUD_FEATURES = [
    "log_amount", "hour", "is_night", "round_amount", "is_new_device",
    "payer_age_days", "payer_age_band", "payer_city_tier", "payer_kyc_min",
    "payer_prior_txns", "amt_vs_payer_mean", "exceeds_prior_max", "payer_txns_1h", "payer_txns_24h",
    "payee_is_merchant", "payee_age_days", "payee_small_kyc", "first_time_payee",
    "payee_prior_unique_payers", "payee_prior_txns",
    "type_P2P", "type_P2M", "type_COLLECT", "type_BILL", "type_RECHARGE",
]


def churn_table(d, obs_days=30, label_days=30, signup_cutoff="2026-04-30"):
    """One row per user: behaviour in first `obs_days` after signup -> churn in the next `label_days`."""
    u = d["users"][d["users"].signup_ts <= signup_cutoff].copy()
    tx = d["tx"].merge(u[["user_id", "signup_ts"]], on="user_id")
    tx["dss"] = (tx.ts - tx.signup_ts).dt.total_seconds() / 86400
    obs = tx[tx.dss <= obs_days]
    lab = tx[(tx.dss > obs_days) & (tx.dss <= obs_days + label_days)]

    f = obs.groupby("user_id").agg(
        n_txns=("txn_id", "size"),
        active_days=("ts", lambda s: s.dt.date.nunique()),
        total_amount=("amount", "sum"),
        avg_amount=("amount", "mean"),
        n_unique_payees=("payee_id", "nunique"),
        share_p2m=("txn_type", lambda s: (s == "P2M").mean()),
        share_bill=("txn_type", lambda s: s.isin(["BILL", "RECHARGE"]).mean()),
        n_tech_fail=("failure_reason", lambda s: s.isin(TECH).sum()),
        n_user_fail=("failure_reason", lambda s: s.isin(["WRONG_PIN", "INSUFFICIENT_FUNDS"]).sum()),
        n_risk_blocked=("failure_reason", lambda s: (s == "RISK_BLOCKED").sum()),
        got_cashback=("cashback_inr", lambda s: int((s > 0).any())),
        last_txn_day=("dss", "max"),
    )
    f["days_since_last_txn"] = obs_days - f.last_txn_day
    f = f.drop(columns="last_txn_day")

    r = d["refunds"].merge(u[["user_id", "signup_ts"]], on="user_id")
    r = r[(r.debit_ts - r.signup_ts).dt.days <= obs_days]
    rf = r.groupby("user_id").agg(n_refunds=("refund_id", "size"), max_refund_tat_h=("tat_hours", "max"))
    k = d["tickets"].merge(u[["user_id", "signup_ts"]], on="user_id")
    k = k[(k.created_ts - k.signup_ts).dt.days <= obs_days]
    kf = k.groupby("user_id").agg(n_tickets=("ticket_id", "size"), min_csat=("csat", "min"))

    out = (u.set_index("user_id")[["acquisition_channel", "city_tier", "age_band", "primary_bank", "kyc_status"]]
           .join(f, how="inner").join(rf).join(kf))
    out[["n_refunds", "max_refund_tat_h", "n_tickets"]] = out[["n_refunds", "max_refund_tat_h", "n_tickets"]].fillna(0)
    out["min_csat"] = out.min_csat.fillna(5)
    out["refund_breached_tat"] = (out.max_refund_tat_h > 24).astype(int)
    out["churned"] = (~out.index.isin(lab.user_id)).astype(int)
    return out
