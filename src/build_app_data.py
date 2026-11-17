"""Pre-compute the model outputs the Streamlit app needs, so the app starts in seconds and never trains a
model on page load. Same models, splits and threshold logic as notebooks 02 and 03.

Run from repo root:  python src/build_app_data.py      -> data/processed/
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import load, fraud_features, churn_table, FRAUD_FEATURES  # noqa: E402
from models import FraudBlend, make_logit, make_gbm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)
X = FRAUD_FEATURES

d = load()
df = fraud_features(d)

# ---------------- fraud: final model scores on the May–Jun test period -------------------------------
train, test = df[df.ts < "2026-05-01"], df[df.ts >= "2026-05-01"]
blend = FraudBlend().fit(train[X], train.is_fraud)
score = blend.predict_proba(test[X])[:, 1]
lr_s = make_logit().fit(train[X], train.is_fraud).predict_proba(test[X])[:, 1]
gb_s = make_gbm().fit(train[X], train.is_fraud).predict_proba(test[X])[:, 1]

# threshold chosen on April validation with a Jan–Mar model (never on test)
fit, val = df[df.ts < "2026-04-01"], df[(df.ts >= "2026-04-01") & (df.ts < "2026-05-01")]
vs = FraudBlend().fit(fit[X], fit.is_fraud).predict_proba(val[X])[:, 1]
best_t, best_net = None, -np.inf
for t in np.linspace(0.5, 0.995, 100):
    f = vs >= t
    net = val.amount.values[f & (val.is_fraud.values == 1)].sum() - (f & (val.is_fraud.values == 0)).sum() * 100
    if net > best_net:
        best_t, best_net = float(t), net

pd.DataFrame({"txn_id": test.txn_id.values, "score": score, "score_logit": lr_s,
              "score_gbm": gb_s}).to_csv(os.path.join(OUT, "fraud_scores.csv.gz"), index=False)

# ---------------- churn: held-out users, scores and drivers (as notebook 03) ---------------------------
ct = churn_table(d)
cat = ["acquisition_channel", "age_band", "primary_bank", "kyc_status"]
num = ["city_tier", "n_txns", "active_days", "total_amount", "avg_amount", "n_unique_payees", "share_p2m", "share_bill",
       "n_tech_fail", "n_user_fail", "n_risk_blocked", "got_cashback", "days_since_last_txn", "n_refunds",
       "max_refund_tat_h", "n_tickets", "min_csat", "refund_breached_tat"]
Xc, yc = ct[cat + num].copy(), ct.churned
for c in ["n_txns", "total_amount", "avg_amount", "n_unique_payees"]:
    Xc[c] = np.log1p(Xc[c])
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.25, stratify=yc, random_state=42)
cat_l = [c for c in cat if c != "primary_bank"]
num_l = [c for c in num if c not in ("total_amount", "n_txns", "n_unique_payees")]
logit = Pipeline([("pre", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_l),
                                             ("num", StandardScaler(), num_l)])),
                  ("m", LogisticRegression(max_iter=3000, C=0.5))]).fit(Xtr, ytr)
hgb = Pipeline([("pre", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)], remainder="passthrough")),
                ("m", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4, random_state=42))]).fit(Xtr, ytr)
p = hgb.predict_proba(Xte)[:, 1]
churn_out = ct.loc[Xte.index, ["acquisition_channel", "city_tier", "age_band", "primary_bank", "active_days",
                               "days_since_last_txn", "n_tech_fail", "refund_breached_tat", "got_cashback"]].copy()
churn_out["n_txns"] = ct.loc[Xte.index, "n_txns"]
churn_out["risk"] = p.round(4)
churn_out["churned"] = yte.values
churn_out.reset_index().to_csv(os.path.join(OUT, "churn_scores.csv"), index=False)

names = logit.named_steps["pre"].get_feature_names_out()
coef = pd.Series(logit.named_steps["m"].coef_[0], index=[n.split("__")[1] for n in names])
coef.rename("coef").rename_axis("feature").reset_index().to_csv(os.path.join(OUT, "churn_drivers.csv"), index=False)

meta = {
    "fraud": {"threshold": best_t, "fp_cost_inr": 100, "train_period": "2026-01-01 to 2026-04-30",
              "test_period": "2026-05-01 to 2026-06-30",
              "pr_auc": {"Blend (selected)": round(average_precision_score(test.is_fraud, score), 4),
                         "Logistic regression": round(average_precision_score(test.is_fraud, lr_s), 4),
                         "Gradient boosting": round(average_precision_score(test.is_fraud, gb_s), 4)}},
    "churn": {"roc_auc_gbm": round(roc_auc_score(yte, p), 4),
              "roc_auc_logit": round(roc_auc_score(yte, logit.predict_proba(Xte)[:, 1]), 4)},
}
json.dump(meta, open(os.path.join(OUT, "model_meta.json"), "w"), indent=2)
print(json.dumps(meta, indent=2))
