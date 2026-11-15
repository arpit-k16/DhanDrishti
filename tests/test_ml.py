"""ML tests: (1) leakage, (2) brute-force feature checks, (3) clean split, (4) re-train the models and check
every ML number quoted in README.md.        Run:  python tests/test_ml.py      (~3 min)
"""
import os
import sys
import numpy as np
import pandas as pd

R = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{R}/src")
import features as F  # noqa: E402
from models import FraudBlend, make_logit, make_gbm  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402

res = []


def check(name, ok, detail=""):
    res.append((name, "PASS" if ok else "FAIL", detail))


d = F.load()
full = F.fraud_features(d)

# 1) Look-ahead test: features for rows before a cutoff must not change when future rows are deleted
cut = pd.Timestamp("2026-05-01")
d_trunc = dict(d)
d_trunc["tx"] = d["tx"][d["tx"].ts < cut].copy()
trunc = F.fraud_features(d_trunc).set_index("txn_id")
a = full[full.ts < cut].set_index("txn_id").loc[trunc.index, F.FRAUD_FEATURES]
b = trunc[F.FRAUD_FEATURES]
diff_cols = [c for c in F.FRAUD_FEATURES if not np.allclose(a[c].fillna(-9).values, b[c].fillna(-9).values)]
check("no look-ahead: all 25 features identical when all post-May data is deleted", not diff_cols, f"changed: {diff_cols}")

# 2) Brute-force recomputation of history features on a random sample
tx = full[["txn_id", "ts", "user_id", "payee_id", "amount"]].reset_index(drop=True)
rng = np.random.default_rng(0)
idx = rng.choice(len(tx), 400, replace=False)
bad = {"payer_prior_txns": 0, "payer_txns_1h": 0, "payee_prior_unique_payers": 0, "first_time_payee": 0}
for i in idx:
    r = tx.iloc[i]
    before = tx.iloc[:i]  # rows strictly earlier in the (time-sorted) feature table
    mine = before[before.user_id == r.user_id]
    if len(mine) != full.payer_prior_txns.iloc[i]: bad["payer_prior_txns"] += 1
    n1h = (mine.ts >= r.ts - pd.Timedelta(hours=1)).sum()   # earlier payments by this payer within the last hour
    if n1h != full.payer_txns_1h.iloc[i]: bad["payer_txns_1h"] += 1
    pay = before[before.payee_id == r.payee_id]
    if pay.user_id.nunique() != full.payee_prior_unique_payers.iloc[i]: bad["payee_prior_unique_payers"] += 1
    if int(not (pay.user_id == r.user_id).any()) != full.first_time_payee.iloc[i]: bad["first_time_payee"] += 1
for k, v in bad.items():
    check(f"brute-force matches feature '{k}' on 400 random rows", v == 0, f"{v} mismatches")

# 3) No label / ground-truth columns among features; time split is clean
forbidden = {"is_fraud", "status", "failure_reason", "fraud_scheme", "role", "legacy_rule_flag"}
check("no label/outcome/ground-truth column used as a feature", not (forbidden & set(F.FRAUD_FEATURES)))
tr, te = full[full.ts < cut], full[full.ts >= cut]
check("train strictly before test (time split)", tr.ts.max() < te.ts.min(), f"{tr.ts.max()} < {te.ts.min()}")

# 4) Re-train exactly as notebook 02 and re-check README numbers
X = F.FRAUD_FEATURES
y, amt, rule = te.is_fraud.values, te.amount.values, te.legacy_rule_flag.values
check("504 frauds in test, ~240 in train", y.sum() == 504 and 230 <= tr.is_fraud.sum() <= 250, f"{y.sum()} / {tr.is_fraud.sum()}")
check("fraud share of payments 0.23%", round(full.is_fraud.mean() * 100, 2) == 0.23, f"{full.is_fraud.mean()*100:.3f}")
blend = FraudBlend().fit(tr[X], tr.is_fraud)
p = blend.predict_proba(te[X])[:, 1]
pl = make_logit().fit(tr[X], tr.is_fraud).predict_proba(te[X])[:, 1]
pg = make_gbm().fit(tr[X], tr.is_fraud).predict_proba(te[X])[:, 1]
check("test PR-AUC blend 0.63 > LR 0.60 > GB 0.56",
      (round(average_precision_score(y, p), 2), round(average_precision_score(y, pl), 2), round(average_precision_score(y, pg), 2)) == (0.63, 0.6, 0.56),
      f"{average_precision_score(y, p):.4f} / {average_precision_score(y, pl):.4f} / {average_precision_score(y, pg):.4f}")
# blend must also win on rolling CV (that is how it was selected)
cvw = 0
for a_, b_ in [("2026-02-01", "2026-03-01"), ("2026-03-01", "2026-04-01"), ("2026-04-01", "2026-05-01")]:
    f_, v_ = full[full.ts < a_], full[(full.ts >= a_) & (full.ts < b_)]
    s_ = {k: average_precision_score(v_.is_fraud, mk().fit(f_[X], f_.is_fraud).predict_proba(v_[X])[:, 1])
          for k, mk in [("blend", FraudBlend), ("lr", make_logit), ("gb", make_gbm)]}
    cvw += s_["blend"] >= max(s_["lr"], s_["gb"]) - 0.01
check("blend best or within 0.01 of best on all 3 CV folds", cvw == 3, f"{cvw}/3")

def stats(f):
    tp, fp = int((f & (y == 1)).sum()), int((f & (y == 0)).sum())
    return tp, fp, round(100 * tp / (tp + fp)), round(100 * tp / y.sum()), round(amt[f & (y == 1)].sum() / 1e5, 1)
check("rules: 135 caught, 250 legit, 35% prec, 27% rec, Rs 10.8 lakh", stats(rule == 1) == (135, 250, 35, 27, 10.8), str(stats(rule == 1)))
k = int(rule.sum()); same = np.zeros(len(y), bool); same[np.argsort(-p)[:k]] = True
check("same 385 flags: 267 caught, 118 legit, 69%, 53%, Rs 17.2 lakh", k == 385 and stats(same) == (267, 118, 69, 53, 17.2), str(stats(same)))
# threshold chosen on April validation with a Jan-Mar model, then frozen
fit, val = full[full.ts < "2026-04-01"], full[(full.ts >= "2026-04-01") & (full.ts < "2026-05-01")]
vs = FraudBlend().fit(fit[X], fit.is_fraud).predict_proba(val[X])[:, 1]
best, bestnet = None, -1e18
for t in np.linspace(0.5, 0.995, 100):
    f = vs >= t
    net = val.amount.values[f & (val.is_fraud.values == 1)].sum() - (f & (val.is_fraud.values == 0)).sum() * 100
    if net > bestnet: best, bestnet = t, net
check("validation threshold = 0.960", round(best, 3) == 0.96, f"{best:.4f}")
fz = p >= best
check("frozen threshold: 335 caught, 367 legit, 48%, 66%, Rs 20.8 lakh", stats(fz) == (335, 367, 48, 66, 20.8), str(stats(fz)))
sch = pd.read_csv(f"{R}/data/ground_truth/fraud_schemes.csv")
tt = te.assign(flag=fz, rule=rule).merge(sch, on="txn_id")
rc, rr = tt.groupby("fraud_scheme").flag.mean(), tt.groupby("fraud_scheme").rule.mean()
check("collect-scam recall 71%, rules 0%", round(rc["COLLECT_SCAM"] * 100) == 71 and rr["COLLECT_SCAM"] == 0, f"{rc['COLLECT_SCAM']:.3f} / {rr['COLLECT_SCAM']:.3f}")
check("fake-QR recall 34%", round(rc["FAKE_QR"] * 100) == 34, f"{rc['FAKE_QR']:.3f}")

# 5) Churn model claims: retrain exactly as notebook 03
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
ct = F.churn_table(d)
# label must only use days 31-60: confirm no observation-window feature uses later data
check("churn table: every user has >=60 days of history", (pd.Timestamp('2026-06-30 23:59') - d['users'].set_index('user_id').signup_ts.reindex(ct.index)).dt.days.min() >= 60)
cat = ['acquisition_channel', 'age_band', 'primary_bank', 'kyc_status']
num = ['city_tier', 'n_txns', 'active_days', 'total_amount', 'avg_amount', 'n_unique_payees', 'share_p2m', 'share_bill',
       'n_tech_fail', 'n_user_fail', 'n_risk_blocked', 'got_cashback', 'days_since_last_txn', 'n_refunds',
       'max_refund_tat_h', 'n_tickets', 'min_csat', 'refund_breached_tat']
X, yc = ct[cat + num].copy(), ct.churned
for c in ['n_txns', 'total_amount', 'avg_amount', 'n_unique_payees']:
    X[c] = np.log1p(X[c])
Xtr, Xte, ytr, yte = train_test_split(X, yc, test_size=0.25, stratify=yc, random_state=42)
cat_l = [c for c in cat if c != 'primary_bank']
num_l = [c for c in num if c not in ('total_amount', 'n_txns', 'n_unique_payees')]
lg = Pipeline([('pre', ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_l), ('num', StandardScaler(), num_l)])),
               ('m', LogisticRegression(max_iter=3000, C=0.5))]).fit(Xtr, ytr)
hg = Pipeline([('pre', ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat)], remainder='passthrough')),
               ('m', HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4, random_state=42))]).fit(Xtr, ytr)
ph = hg.predict_proba(Xte)[:, 1]
check("churn GB ROC-AUC 0.89", round(roc_auc_score(yte, ph), 2) == 0.89, f"{roc_auc_score(yte, ph):.3f}")
check("churn logistic ROC-AUC 0.87", round(roc_auc_score(yte, lg.predict_proba(Xte)[:, 1]), 2) == 0.87, f"{roc_auc_score(yte, lg.predict_proba(Xte)[:, 1]):.3f}")
order = np.argsort(-ph); n3 = int(round(0.3 * len(ph)))
cap = yte.values[order[:n3]].sum() / yte.sum()
check("top 3 deciles hold 75% of churners", round(cap * 100) == 75, f"{cap:.3f}")

out = pd.DataFrame(res, columns=["test", "result", "detail"])
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 80)
print(out.to_string(index=False))
print(f"\n{(out.result == 'PASS').sum()}/{len(out)} ML tests passed")
sys.exit(0 if (out.result == "PASS").all() else 1)
