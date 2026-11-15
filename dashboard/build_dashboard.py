"""Compute dashboard metrics and write a single self-contained dashboard/index.html (no internet needed).
Run from repo root:  python dashboard/build_dashboard.py
"""
import json
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from features import load, TECH, fraud_features, FRAUD_FEATURES  # noqa: E402
from models import FraudBlend  # noqa: E402

d = load()
tx, users, refunds, tickets, banks = d["tx"], d["users"], d["refunds"], d["tickets"], d["banks"]
tx["tech"] = tx.failure_reason.isin(TECH)
ok = tx.status == "SUCCESS"
END = pd.Timestamp("2026-06-30")

monthly = (tx.assign(month=tx.ts.dt.strftime("%b"), gmv=np.where(ok, tx.amount, 0))
             .groupby(tx.ts.dt.to_period("M"))
             .agg(month=("month", "first"), gmv=("gmv", "sum"), mau=("user_id", "nunique"),
                  sr=("status", lambda s: (s == "SUCCESS").mean() * 100)))
bank = (tx.groupby("payer_bank").tech.mean().mul(100).rename("rate").to_frame()
          .join(banks.set_index("bank_code").bank_name).sort_values("rate", ascending=False))
heat = tx.assign(h=tx.ts.dt.hour).pivot_table(index="payer_bank", columns="h", values="tech", aggfunc="mean").mul(100)
heat = heat.loc[bank.index]
r = refunds.tat_hours
tat = pd.cut(r, [0, 4, 24, 72, 168, 1e5], labels=["≤4h", "4–24h", "1–3d", "3–7d", ">7d"]).value_counts(sort=False)
last = tx.groupby("user_id").ts.max()
u = users.set_index("user_id").assign(churn=lambda x: (END - last.reindex(x.index)).dt.total_seconds() / 86400 > 30)
churn = u[u.signup_ts < "2026-04-30"].groupby("acquisition_channel").churn.mean().mul(100).sort_values(ascending=False)
age = users.assign(v=users.user_id.isin(tx[tx.is_fraud == 1].user_id)).groupby("age_band").v.mean().mul(100)
dev = users.groupby("device_id").user_id.size()
farm_dev = dev[dev >= 3].index
farm_cb = tx[tx.user_id.isin(users[users.device_id.isin(farm_dev)].user_id)].cashback_inr.sum()
tk = tickets.dropna(subset=["resolved_ts", "csat"]).assign(h=lambda t: (t.resolved_ts - t.created_ts).dt.total_seconds() / 3600)
csat = tk.groupby(pd.cut(tk.h, [0, 12, 48, 120, 1e5], labels=["≤12h", "12–48h", "2–5d", ">5d"]), observed=True).csat.mean()

# fraud: rules vs model on May–Jun, same number of flags
f = fraud_features(d)
tr, te = f[f.ts < "2026-05-01"], f[f.ts >= "2026-05-01"]
m = FraudBlend().fit(tr[FRAUD_FEATURES], tr.is_fraud)
p = m.predict_proba(te[FRAUD_FEATURES])[:, 1]
k = int(te.legacy_rule_flag.sum())
top = np.argsort(-p)[:k]
y = te.is_fraud.values
fraud = {"flags": k, "frauds": int(y.sum()),
         "rules_tp": int(((te.legacy_rule_flag.values == 1) & (y == 1)).sum()), "model_tp": int(y[top].sum())}

D = {
    "kpi": {
        "gmv_cr": round(tx.loc[ok, "amount"].sum() / 1e7, 1), "txns": int(len(tx)), "sr": round(ok.mean() * 100, 2),
        "users": int(users.shape[0]), "tech_decl": int(tx.tech.sum()),
        "breach": round((r > 24).mean() * 100, 1), "comp": int(refunds.compensation_due_inr.sum()),
        "fraud_lakh": round(tx.loc[(tx.is_fraud == 1) & ok, "amount"].sum() / 1e5, 1),
        "farm_pct": round(farm_cb / tx.cashback_inr.sum() * 100, 1), "farm_accts": int(dev[dev >= 3].sum()),
    },
    "monthly": monthly[["month", "gmv", "mau", "sr"]].round(2).to_dict("records"),
    "bank": [{"code": i, "name": n, "rate": round(v, 2)} for i, (v, n) in bank[["rate", "bank_name"]].iterrows()],
    "heat": {"rows": list(heat.index), "vals": heat.round(2).values.tolist()},
    "tat": [{"k": str(i), "v": int(v)} for i, v in tat.items()],
    "churn": [{"k": i.replace("_", " "), "v": round(v, 1)} for i, v in churn.items()],
    "age": [{"k": i, "v": round(v, 2)} for i, v in age.items()],
    "csat": [{"k": str(i), "v": round(v, 2)} for i, v in csat.items()],
    "fraud": fraud,
}

html = open(os.path.join(HERE, "template.html"), encoding="utf-8").read().replace("/*__DATA__*/{}", json.dumps(D))
open(os.path.join(HERE, "index.html"), "w", encoding="utf-8").write(html)
print("wrote dashboard/index.html", json.dumps(D["kpi"]), fraud)
