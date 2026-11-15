"""Recompute every number quoted in README.md straight from the raw CSVs and check it rounds to what the
README says. Deliberately written from scratch: it does NOT import src/features.py, so a bug there can't
hide a wrong claim.        Run:  python tests/test_readme_numbers.py
"""
import os
import sys
import numpy as np
import pandas as pd

R = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tx = pd.read_csv(f"{R}/data/raw/transactions.csv", parse_dates=["ts"])
us = pd.read_csv(f"{R}/data/raw/users.csv", parse_dates=["signup_ts"])
rf = pd.read_csv(f"{R}/data/raw/refunds.csv", parse_dates=["debit_ts", "refund_ts"])
tk = pd.read_csv(f"{R}/data/raw/support_tickets.csv", parse_dates=["created_ts", "resolved_ts"])
TECH = {"BANK_SERVER_DOWN", "TIMEOUT", "NETWORK_ERROR"}
tech = tx.failure_reason.isin(TECH)
ok = tx.status.eq("SUCCESS")
results = []


def check(name, actual, claimed, decimals):
    """PASS when the recomputed value, rounded the way the README displays it, equals the README figure."""
    passed = round(float(actual), decimals) == claimed
    results.append((name, round(float(actual), 3), claimed, "PASS" if passed else "FAIL"))


# --- scale
check("payments (3.16 lakh)", len(tx) / 1e5, 3.16, 2)
check("users (12.4k)", len(us) / 1e3, 12.4, 1)
check("GMV Rs Cr (dashboard)", tx.loc[ok, "amount"].sum() / 1e7, 24.6, 1)
check("success rate % (dashboard)", ok.mean() * 100, 95.52, 2)

# --- loophole 1
rate = tx.groupby("payer_bank").apply(lambda g: g.failure_reason.isin(TECH).mean())
worst3 = rate.nlargest(3).index
check("worst-3 banks traffic %", tx.payer_bank.isin(worst3).mean() * 100, 25, 0)
check("worst-3 banks share of tech declines %", tech[tx.payer_bank.isin(worst3)].sum() / tech.sum() * 100, 47, 0)
mrb = tx[tx.payer_bank == "MRB"]
pk = mrb.ts.dt.hour.between(19, 22)
check("MRB offpeak decline %", mrb[~pk].failure_reason.isin(TECH).mean() * 100, 1.9, 1)
check("MRB peak decline %", mrb[pk].failure_reason.isin(TECH).mean() * 100, 6.9, 1)
# retry recovery: brute force per failure
fails = tx[tech]
succ = tx[ok].sort_values("ts")
grp = {k: g.ts.values for k, g in succ.groupby(["user_id", "payee_id"])}
rec = []
for r in fails.itertuples():
    t = grp.get((r.user_id, r.payee_id))
    if t is None:
        rec.append(False); continue
    t0 = np.datetime64(r.ts)
    rec.append(bool(((t > t0) & (t <= t0 + np.timedelta64(30, "m"))).any()))
rec = np.array(rec)
check("retry recovery within 30 min %", rec.mean() * 100, 51, 0)
check("abandoned GMV Rs lakh", fails.amount.values[~rec].sum() / 1e5, 31, 0)

# --- loophole 2
check("refund T+1 breach %", (rf.tat_hours > 24).mean() * 100, 14.6, 1)
check("compensation Rs", rf.compensation_due_inr.sum(), 95900, 0)
# recompute compensation from timestamps (Rs100 per started day beyond 24h)
end = pd.Timestamp("2026-06-30 23:59:59")
h = (rf.refund_ts.fillna(end) - rf.debit_ts).dt.total_seconds() / 3600
comp = (np.maximum(np.ceil((h.round(1) - 24) / 24), 0) * 100).sum()
check("compensation recomputed from timestamps", comp, 95900, 0)
rb = rf.merge(tx[["txn_id", "payer_bank"]], on="txn_id")
br = rb.groupby("payer_bank").tat_hours.apply(lambda s: (s > 24).mean() * 100)
check("worst-3 min breach % (24-30%)", br[worst3].min(), 24, 0)
check("worst-3 max breach % (24-30%)", br[worst3].max(), 30, 0)
check("other banks min breach % (2-3%)", br.drop(worst3).min(), 2, 0)
check("other banks max breach % (2-3%)", br.drop(worst3).max(), 3, 0)
check("worst-3 share of compensation %", rb[rb.payer_bank.isin(worst3)].compensation_due_inr.sum() / rb.compensation_due_inr.sum() * 100, 96, 0)
check("refund tickets share %", (tk.category == "REFUND_NOT_RECEIVED").mean() * 100, 48, 0)

# --- loophole 3
last = tx.groupby("user_id").ts.max()
u = us.set_index("user_id")
u["churn"] = (pd.Timestamp("2026-06-30") - last.reindex(u.index)).dt.total_seconds() / 86400 > 30
m = u[u.signup_ts < "2026-04-30"]
c = m.groupby("acquisition_channel").churn.mean() * 100
check("churn cashback %", c["cashback_campaign"], 65, 0)
check("churn organic %", c["organic"], 34, 0)
check("churn referral %", c["referral"], 31, 0)
j = tx.merge(us[["user_id", "signup_ts"]], on="user_id")
ef = j[(j.failure_reason.isin(TECH)) & ((j.ts - j.signup_ts).dt.total_seconds() / 86400 <= 14)].user_id.unique()
m = m.assign(ef=m.index.isin(ef))
check("churn clean first 14d %", m[~m.ef].churn.mean() * 100, 44, 0)
check("churn early failure %", m[m.ef].churn.mean() * 100, 54, 0)
cmp1 = us[us.campaign_id == "CMP01"]
spent = tx[tx.user_id.isin(cmp1.user_id)].cashback_inr.sum()
active_jun = tx[(tx.user_id.isin(cmp1.user_id)) & (tx.ts >= "2026-06-01")].user_id.nunique()
check("NEWUSER50 cost per retained Rs", spent / active_jun, 192, 0)
check("NEWUSER50 cost per install Rs", spent / len(cmp1), 45, 0)

# --- loophole 4
fr = tx.groupby("txn_type").is_fraud.mean() * 100
check("COLLECT fraud rate %", fr["COLLECT"], 2.2, 1)
check("COLLECT / P2P fraud-rate ratio (~8x)", fr["COLLECT"] / fr["P2P"], 8, 0)
vic = set(tx[tx.is_fraud == 1].user_id)
ab = us.assign(v=us.user_id.isin(vic)).groupby("age_band").v.mean()
check("45+ vs under-35 victim ratio (~3.5x)", ab[["45-54", "55+"]].mean() / ab[["18-24", "25-34"]].mean(), 3.5, 1)
blk = tx.failure_reason.eq("RISK_BLOCKED")
check("rules recall %", blk[tx.is_fraud == 1].mean() * 100, 18, 0)
check("rules: legit share of blocks %", (tx[blk].is_fraud == 0).mean() * 100, 69, 0)
check("fraud loss Rs lakh", tx.loc[(tx.is_fraud == 1) & ok, "amount"].sum() / 1e5, 31.8, 1)

# --- loophole 5
dev = us.groupby("device_id").user_id.transform("size")
farm_users = us[dev >= 3]
check("farm devices", farm_users.device_id.nunique(), 42, 0)
check("farm accounts", len(farm_users), 324, 0)
span = farm_users.groupby("device_id").signup_ts.agg(lambda s: (s.max() - s.min()).total_seconds() / 3600)
check("farm accounts created within 10 hours", float(span.max() <= 10), 1.0, 0)
check("cashback leaked to farms %", tx[tx.user_id.isin(farm_users.user_id)].cashback_inr.sum() / tx.cashback_inr.sum() * 100, 10.8, 1)
# do the device-farm accounts really match the hidden ground truth?
truth = pd.read_csv(f"{R}/data/ground_truth/user_roles.csv")
true_farm = set(truth[truth.role.str.startswith("farm")].user_id)
check("farm rule precision vs ground truth", len(set(farm_users.user_id) & true_farm) / len(farm_users), 1.0, 3)
check("farm rule recall vs ground truth", len(set(farm_users.user_id) & true_farm) / len(true_farm), 1.0, 3)

# --- loophole 6
t = tk.dropna(subset=["resolved_ts", "csat"])
hh = (t.resolved_ts - t.created_ts).dt.total_seconds() / 3600
check("CSAT <=12h", t[hh <= 12].csat.mean(), 4.3, 1)
check("CSAT >5d", t[hh > 120].csat.mean(), 1.2, 1)
check("detractor % >5d", (t[hh > 120].csat <= 2).mean() * 100, 98, 0)
fh = tk[tk.category == "FRAUD_REPORT"]
check("fraud report resolution days", ((fh.resolved_ts - fh.created_ts).dt.total_seconds() / 86400).mean(), 6.5, 1)

out = pd.DataFrame(results, columns=["claim", "recomputed", "README says", "result"])
pd.set_option("display.width", 200)
print(out.to_string(index=False))
print(f"\n{(out.result == 'PASS').sum()}/{len(out)} README claims verified")
sys.exit(0 if (out.result == "PASS").all() else 1)
