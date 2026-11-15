"""Builds and executes the three analysis notebooks (so they ship with outputs).
Run from repo root:  python notebooks/build_notebooks.py
"""
import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
SETUP = """import sys, warnings
sys.path.insert(0, '../src')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import plot_style as ps
from features import load, TECH
ps.apply()
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')
d = load()
tx, users, refunds, tickets, banks = d['tx'], d['users'], d['refunds'], d['tickets'], d['banks']
tx['is_tech_fail'] = tx.failure_reason.isin(TECH).astype(int)
tx['hour'] = tx.ts.dt.hour
END = pd.Timestamp('2026-06-30')"""


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


# =========================================================================== #
nb1 = [
md("""
# 01 · Payment health EDA — where a UPI app leaks money and users

**Business context.** UPI apps earn almost nothing per P2P payment, so growth depends on
*trust* (payments that just work), *retention* (users who stay after the cashback), and
*loss control* (fraud and promo abuse). This notebook profiles 6 months of activity on a
fictional UPI app (**316k transactions, 12.4k users, 8 partner banks**) and quantifies
six known weak spots:

| # | Loophole | Question |
|---|---|---|
| 1 | Bank-side failures | Which remitter banks / hours cause technical declines? |
| 2 | Stuck refunds | How often do "debited but failed" refunds break the T+1 rule? |
| 3 | Leaky acquisition | Do cashback-acquired users stick around? |
| 4 | Fraud | Who gets defrauded, and do the current rules catch it? |
| 5 | Promo abuse | How much cashback goes to device farms? |
| 6 | Grievance redressal | How does resolution time affect CSAT? |

> Data is synthetic (see `data/generate_data.py`), modelled on publicly reported UPI pain points.
"""),
code(SETUP),
md("## 0 · Data quality checks\nBefore any analysis: completeness, uniqueness, validity, consistency and referential integrity."),
code("""
checks = {
    'duplicate txn_id': tx.txn_id.duplicated().sum(),
    'txn with unknown user_id': (~tx.user_id.isin(users.user_id)).sum(),
    'FAILED without failure_reason': ((tx.status == 'FAILED') & tx.failure_reason.isna()).sum(),
    'SUCCESS with a failure_reason': ((tx.status == 'SUCCESS') & tx.failure_reason.notna()).sum(),
    'non-positive amount': (tx.amount <= 0).sum(),
    'amount above UPI limit (Rs 1 lakh)': (tx.amount > 100000).sum(),
    'txn before payer signup': (tx.ts < tx.user_id.map(users.set_index('user_id').signup_ts)).sum(),
    'refund for a non-failed txn': (~refunds.txn_id.isin(tx.loc[tx.status == 'FAILED', 'txn_id'])).sum(),
    'cashback on a failed txn': ((tx.cashback_inr > 0) & (tx.status != 'SUCCESS')).sum(),
}
pd.Series(checks, name='rows failing').to_frame()
"""),
code("""
print('Null share by column (transactions):')
tx.isna().mean().loc[lambda s: s > 0].round(3)
"""),
md("Only expected nulls: `failure_reason` is null for successful payments. All integrity checks pass, so we can proceed."),
md("## Headline KPIs"),
code("""
m = (tx.assign(month=tx.ts.dt.to_period('M'))
       .groupby('month')
       .agg(attempts=('txn_id', 'size'),
            success_rate=('status', lambda s: (s == 'SUCCESS').mean() * 100),
            gmv_lakh=('amount', lambda s: s[tx.loc[s.index, 'status'] == 'SUCCESS'].sum() / 1e5),
            mau=('user_id', 'nunique'),
            cashback=('cashback_inr', 'sum')))
m
"""),
code("""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
x = m.index.astype(str)
ax[0].bar(x, m.gmv_lakh, color=ps.BLUE, width=0.6)
ax[0].set_title('GMV (Rs lakh, successful payments)')
ax[1].plot(x, m.mau, color=ps.BLUE, marker='o', markersize=6)
ax[1].set_title('Monthly active users')
ax[1].set_ylim(0)
for a in ax: a.tick_params(axis='x', rotation=0)
fig.suptitle('Growth is real, but MAU dips in June once campaign users churn', x=0.01, ha='left', fontsize=10, color=ps.INK2)
fig.tight_layout(); ps.save(fig, '01_kpis');
"""),
md("""
## Loophole 1 · Technical declines are concentrated in three banks and the evening peak
Business declines (wrong PIN, low balance) belong to the user; **technical declines (bank server
down, timeout, network) are the platform's problem**. Each one is a payment the user blames on the app.
"""),
code("""
b = (tx.groupby('payer_bank').agg(attempts=('txn_id', 'size'), tech_fail_pct=('is_tech_fail', 'mean'))
       .assign(tech_fail_pct=lambda d: d.tech_fail_pct * 100)
       .join(banks.set_index('bank_code').bank_name)
       .sort_values('tech_fail_pct'))
avg = tx.is_tech_fail.mean() * 100
worst = b.tech_fail_pct.nlargest(3).index
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.barh(b.bank_name, b.tech_fail_pct, color=[ps.ORANGE if i in worst else ps.GREY for i in b.index], height=0.6)
ax.axvline(avg, color=ps.INK2, lw=1, ls='--'); ax.text(avg, len(b) - 0.45, f' platform avg {avg:.1f}%', color=ps.INK2, fontsize=9)
for i, v in enumerate(b.tech_fail_pct): ax.text(v + 0.08, i, f'{v:.1f}%', va='center', fontsize=9)
ax.set_title('Technical decline rate by remitter bank'); ax.set_xlabel('% of attempts'); ax.grid(axis='y', visible=False)
fig.tight_layout(); ps.save(fig, '02_bank_declines');
share = tx[tx.payer_bank.isin(worst)].is_tech_fail.sum() / tx.is_tech_fail.sum()
print(f'3 worst banks = {tx.payer_bank.isin(worst).mean():.0%} of traffic but {share:.0%} of technical declines')
"""),
code("""
hm = tx.pivot_table(index='payer_bank', columns='hour', values='is_tech_fail', aggfunc='mean') * 100
hm = hm.loc[b.index[::-1]]
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('seq', ps.SEQ_BLUE)
fig, ax = plt.subplots(figsize=(11, 3.4))
im = ax.imshow(hm.values, aspect='auto', cmap=cmap, vmin=0, vmax=10)
ax.set_yticks(range(len(hm)), hm.index); ax.set_xticks(range(24), range(24)); ax.grid(False)
ax.set_xlabel('hour of day'); ax.set_title('Technical decline % by bank × hour — Meridian (MRB) collapses 19:00–22:59')
fig.colorbar(im, ax=ax, label='% declined', shrink=0.9)
fig.tight_layout(); ps.save(fig, '03_bank_hour_heatmap');
"""),
code("""
# Does the user recover by retrying? (same payee, success within 30 min)
f = tx[tx.is_tech_fail == 1][['txn_id', 'user_id', 'payee_id', 'ts', 'amount']]
s = tx[tx.status == 'SUCCESS'][['user_id', 'payee_id', 'ts']].rename(columns={'ts': 'ts_ok'})
j = f.merge(s, on=['user_id', 'payee_id'])
j = j[(j.ts_ok > j.ts) & (j.ts_ok <= j.ts + pd.Timedelta(minutes=30))]
recovered = f.txn_id.isin(j.txn_id)
print(f'Technical declines: {len(f):,} | recovered by retry: {recovered.mean():.0%} | '
      f'abandoned GMV: Rs {f.loc[~recovered, "amount"].sum()/1e5:.1f} lakh in 6 months')
"""),
md("""
## Loophole 2 · "Money debited, payment failed" — refunds that break T+1
Under RBI's TAT framework a failed UPI debit must be auto-reversed by **T+1**; beyond that the
customer is owed **Rs 100/day**. These cases also drive most support tickets.
"""),
code("""
r = refunds.merge(tx[['txn_id', 'payer_bank']], on='txn_id')
fig, ax = plt.subplots(figsize=(9, 3.4))
bins = np.r_[0:24:2, 24:192:8, 400]
ax.hist(r.tat_hours.clip(upper=199), bins=np.r_[0:24:2, 24:200:8], color=ps.BLUE, edgecolor='#fcfcfb', linewidth=1)
ax.axvline(24, color=ps.RED, lw=1.5); ax.text(25, ax.get_ylim()[1]*0.85, 'T+1 limit (24h)', color=ps.INK2)
ax.set_xlabel('hours from debit to refund'); ax.set_ylabel('refunds'); ax.set_title('Refund turnaround time — a long tail past T+1')
fig.tight_layout(); ps.save(fig, '04_refund_tat');
rb = (r.groupby('payer_bank').agg(refunds=('refund_id', 'size'), breach_pct=('tat_hours', lambda s: (s > 24).mean() * 100),
                                 p90_tat_h=('tat_hours', lambda s: s.quantile(.9)), compensation_inr=('compensation_due_inr', 'sum'))
        .sort_values('breach_pct', ascending=False))
print(f"Breach rate overall: {(r.tat_hours > 24).mean():.1%} | compensation liability: Rs {r.compensation_due_inr.sum():,.0f}")
rb
"""),
md("""
## Loophole 3 · Cashback buys installs, not users
"""),
code("""
last = tx.groupby('user_id').ts.max()
u = users.set_index('user_id').assign(last_ts=last, n_txn=tx.groupby('user_id').size())
u['churned'] = (END - u.last_ts).dt.total_seconds() / 86400 > 30   # same definition as sql/04
mature = u[u.signup_ts < '2026-04-30']
ch = mature.groupby('acquisition_channel').agg(users=('churned', 'size'), churn_pct=('churned', 'mean'),
                                              median_txns=('n_txn', 'median'))
ch['churn_pct'] *= 100
fig, ax = plt.subplots(figsize=(7, 2.8))
ch = ch.sort_values('churn_pct')
ax.barh(ch.index, ch.churn_pct, color=[ps.ORANGE if i == 'cashback_campaign' else ps.GREY for i in ch.index], height=0.55)
for i, v in enumerate(ch.churn_pct): ax.text(v + 0.8, i, f'{v:.0f}%', va='center')
ax.set_xlim(0, 80); ax.set_title('Churn rate by acquisition channel (signed up before May)'); ax.grid(axis='y', visible=False)
fig.tight_layout(); ps.save(fig, '05_churn_by_channel');
ch
"""),
code("""
# monthly cohort retention
a = tx.merge(users[['user_id', 'signup_ts']], on='user_id')
a['cohort'] = a.signup_ts.dt.to_period('M'); a['m'] = (a.ts.dt.to_period('M') - a.cohort).apply(lambda x: x.n)
coh = a.groupby(['cohort', 'm']).user_id.nunique().unstack()
ret = coh.div(coh[0], axis=0) * 100
for c in ret.index:  # blank out months that haven't happened yet
    ret.loc[c, [k for k in ret.columns if (c + k) > pd.Period('2026-06', 'M')]] = np.nan
fig, ax = plt.subplots(figsize=(7, 3.2))
im = ax.imshow(ret.values, cmap=cmap, vmin=0, vmax=100, aspect='auto')
for i in range(ret.shape[0]):
    for k in range(ret.shape[1]):
        v = ret.values[i, k]
        if not np.isnan(v): ax.text(k, i, f'{v:.0f}', ha='center', va='center', fontsize=9, color='white' if v > 60 else ps.INK)
ax.set_xticks(range(ret.shape[1]), [f'M{k}' for k in ret.columns]); ax.set_yticks(range(len(ret)), ret.index.astype(str))
ax.grid(False); ax.set_title('Monthly cohort retention (% of cohort active)')
fig.tight_layout(); ps.save(fig, '06_cohort_retention');
"""),
code("""
# Does a bad first two weeks predict churn?
early = (tx.merge(users[['user_id', 'signup_ts']], on='user_id')
           .assign(early=lambda d: (d.is_tech_fail == 1) & ((d.ts - d.signup_ts).dt.total_seconds() / 86400 <= 14))
           .groupby('user_id').early.max())
mature = mature.assign(early_fail=early)
print((mature.groupby('early_fail').churned.mean() * 100).round(1).rename('churn %'))
"""),
md("## Loophole 4 · Fraud targets older users through collect requests, and the rules miss it"),
code("""
vic = tx[tx.is_fraud == 1].user_id.unique()
ab = users.assign(victim=users.user_id.isin(vic)).groupby('age_band').victim.mean() * 100
fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
ax[0].bar(ab.index, ab.values, color=[ps.ORANGE if a in ('45-54', '55+') else ps.GREY for a in ab.index], width=0.6)
for i, v in enumerate(ab.values): ax[0].text(i, v + 0.15, f'{v:.1f}%', ha='center', fontsize=9)
ax[0].set_title('% of users defrauded, by age band'); ax[0].grid(axis='x', visible=False)
ft = tx.groupby('txn_type').is_fraud.mean().mul(100).sort_values()
ax[1].barh(ft.index, ft.values, color=[ps.ORANGE if t == 'COLLECT' else ps.GREY for t in ft.index], height=0.55)
for i, v in enumerate(ft.values): ax[1].text(v + 0.03, i, f'{v:.2f}%', va='center', fontsize=9)
ax[1].set_title('Fraud rate by transaction type'); ax[1].grid(axis='y', visible=False)
fig.tight_layout(); ps.save(fig, '07_fraud_profile');
blocked = tx.failure_reason == 'RISK_BLOCKED'
print(f"Fraud loss (successful fraud): Rs {tx.loc[(tx.is_fraud==1)&(tx.status=='SUCCESS'),'amount'].sum()/1e5:.1f} lakh")
print(f"Legacy rules: blocked {blocked.sum()} | precision {tx[blocked].is_fraud.mean():.0%} | recall {blocked[tx.is_fraud==1].mean():.0%}")
"""),
md("## Loophole 5 · Device farms farming sign-up cashback"),
code("""
dev = users.groupby('device_id').agg(accounts=('user_id', 'size'), first=('signup_ts', 'min'), last=('signup_ts', 'max'))
dev['span_h'] = (dev['last'] - dev['first']).dt.total_seconds() / 3600
cb = tx[['user_id', 'cashback_inr']].merge(users[['user_id', 'device_id']], on='user_id').groupby('device_id').cashback_inr.sum()
dev = dev.join(cb.rename('cashback_inr'))
farms = dev[dev.accounts >= 3].sort_values('cashback_inr', ascending=False)
print(f'{len(farms)} devices with >=3 accounts ({farms.accounts.sum()} accounts, all created within {farms.span_h.max():.0f}h of each other)')
print(f'Cashback extracted: Rs {farms.cashback_inr.sum():,.0f} = {farms.cashback_inr.sum()/tx.cashback_inr.sum():.1%} of all cashback')
farms.head(8)
"""),
md("## Loophole 6 · Slow grievance redressal destroys satisfaction"),
code("""
t = tickets.dropna(subset=['resolved_ts', 'csat']).copy()
t['res_h'] = (t.resolved_ts - t.created_ts).dt.total_seconds() / 3600
t['bucket'] = pd.cut(t.res_h, [0, 12, 48, 120, 1e4], labels=['<=12h', '12-48h', '2-5 days', '>5 days'])
cs = t.groupby('bucket', observed=True).agg(tickets=('ticket_id', 'size'), avg_csat=('csat', 'mean'),
                                          detractor_pct=('csat', lambda s: (s <= 2).mean() * 100))
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(cs.index.astype(str), cs.avg_csat, color=ps.BLUE, width=0.55)
for i, v in enumerate(cs.avg_csat): ax.text(i, v + 0.06, f'{v:.1f}', ha='center')
ax.set_ylim(0, 5); ax.set_title('Average CSAT (1-5) by ticket resolution time'); ax.grid(axis='x', visible=False)
fig.tight_layout(); ps.save(fig, '08_csat_vs_resolution');
print(tickets.groupby('category').size().sort_values(ascending=False).rename('tickets'))
cs
"""),
md("""
## Summary
1. **Three banks (25% of traffic) cause 47% of technical declines**; one of them degrades mainly at peak hours, a routing / bank-switch nudge problem.
2. **Roughly half of users abandon after a technical decline** instead of retrying, which is lost GMV.
3. **~1 in 7 debited-but-failed refunds miss T+1**, and the same three banks own almost all the compensation liability.
4. **Cashback-acquired users churn at roughly twice the rate** of organic/referral users; a failure in week one adds more churn on top.
5. **Collect-request scams on 45+ users** are the biggest fraud vector; legacy rules catch < 1 in 5 frauds while wrongly blocking more than 2 legit payments for every fraud caught → see notebook 02.
6. **~10% of cashback is extracted by device farms** detectable with a simple device-graph rule.
7. **Tickets resolved after 5 days have CSAT ≈ 1**, with fraud reports the slowest category.
"""),
]

# =========================================================================== #
nb2 = [
md("""
# 02 · Fraud detection model — beating the static rules

**Goal:** score every payment in real time and send risky ones to a step-up check (cooling-off
period, re-authentication, or a warning screen for collect requests) instead of the current
static rules, which block legit users and miss most fraud.

**Method**
- 25 features, all computed from information available *before* the payment (verified: deleting all future
  rows leaves every feature value unchanged)
- **Time-based split:** train Jan–Apr, test May–Jun. The test set is used **once**, at the end
- **Model selection** by rolling time-series cross-validation (Feb, Mar, Apr folds) inside the training period
- **Operating threshold** chosen on an April validation month by net rupees saved, then frozen
- Fraud is ~0.23% of payments → evaluate with **PR-AUC**, not accuracy
"""),
code("""
import sys, warnings; sys.path.insert(0, '../src'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import plot_style as ps
from features import load, fraud_features, FRAUD_FEATURES
from models import make_logit, make_gbm, FraudBlend
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from sklearn.inspection import permutation_importance
ps.apply()
df = fraud_features(load())
X = FRAUD_FEATURES
train, test = df[df.ts < '2026-05-01'], df[df.ts >= '2026-05-01']
ytr, yte = train.is_fraud, test.is_fraud
print(f'train {len(train):,} rows, {ytr.sum()} fraud ({ytr.mean():.2%}) | test {len(test):,} rows, {yte.sum()} fraud ({yte.mean():.2%})')
"""),
md("## Feature intuition: fraud vs legit"),
code("""
cols = ['amt_vs_payer_mean', 'payee_prior_unique_payers', 'payee_age_days', 'is_new_device', 'is_night',
        'round_amount', 'payer_txns_1h', 'type_COLLECT', 'payer_age_band']
df.groupby('is_fraud')[cols].median().T.rename(columns={0: 'legit (median)', 1: 'fraud (median)'}).join(
    df.groupby('is_fraud')[cols].mean().T.rename(columns={0: 'legit (mean)', 1: 'fraud (mean)'}))
"""),
md("""
## Model selection: rolling time-series cross-validation (test set untouched)
Each fold trains on all months before the validation month. With only a few hundred frauds, a single
validation month is noisy, so we compare models across three folds.
"""),
code("""
cands = {'Logistic regression': make_logit, 'Gradient boosting (shallow)': make_gbm, 'Blend (LR + GB)': FraudBlend}
cv = {}
for start, end, name in [('2026-02-01', '2026-03-01', 'Feb'), ('2026-03-01', '2026-04-01', 'Mar'), ('2026-04-01', '2026-05-01', 'Apr')]:
    fit, val = df[df.ts < start], df[(df.ts >= start) & (df.ts < end)]
    for k, make in cands.items():
        p = make().fit(fit[X], fit.is_fraud).predict_proba(val[X])[:, 1]
        cv.setdefault(k, {})[f'{name} ({val.is_fraud.sum()} frauds)'] = average_precision_score(val.is_fraud, p)
cv = pd.DataFrame(cv).T
cv['mean PR-AUC'] = cv.mean(axis=1)
cv.sort_values('mean PR-AUC', ascending=False).round(3)
"""),
md("""
The blend (average of z-scored log-odds from both models, see `src/models.py`) is the best or near-best on
every fold: the linear model captures smooth risk gradients, the trees capture interactions. **We select it.**

## Final evaluation on May–Jun (the only time the test set is used)
"""),
code("""
lr, gb, blend = make_logit().fit(train[X], ytr), make_gbm().fit(train[X], ytr), FraudBlend().fit(train[X], ytr)
P = {'Logistic regression': lr.predict_proba(test[X])[:, 1], 'Gradient boosting': gb.predict_proba(test[X])[:, 1],
     'Blend (selected)': blend.predict_proba(test[X])[:, 1]}
rule = test.legacy_rule_flag.values
def recall_at_precision(y, p, target=0.8):
    pr, rc, _ = precision_recall_curve(y, p)
    return rc[pr >= target].max()
res = pd.DataFrame({k: {'PR-AUC': average_precision_score(yte, p), 'ROC-AUC': roc_auc_score(yte, p),
                        'Recall @ 80% precision': recall_at_precision(yte, p)} for k, p in P.items()}).T
res.loc['Legacy rules (binary)'] = [np.nan, roc_auc_score(yte, rule), np.nan]
print(f"Rules: precision {yte[rule == 1].mean():.0%}, recall {rule[yte == 1].mean():.0%}")
res.round(3)
"""),
code("""
fig, ax = plt.subplots(figsize=(6.5, 4))
for (name, p), c in zip(P.items(), [ps.GREY, ps.AQUA, ps.BLUE]):
    pr, rc, _ = precision_recall_curve(yte, p)
    ax.plot(rc, pr, color=c, label=f'{name} (PR-AUC {average_precision_score(yte, p):.2f})')
ax.scatter([rule[yte == 1].mean()], [yte[rule == 1].mean()], s=70, color=ps.ORANGE, zorder=5, label='Legacy rules')
ax.set_xlabel('recall (share of fraud caught)'); ax.set_ylabel('precision (share of flags that are fraud)')
ax.set_title('Precision–recall on May–Jun (unseen) data'); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
ax.annotate('Legacy rules', (rule[yte == 1].mean(), yte[rule == 1].mean()), xytext=(8, -14), textcoords='offset points', color=ps.INK2)
fig.tight_layout(); ps.save(fig, '09_fraud_pr_curve');
"""),
md("""
## Choosing the threshold by money, on validation data
Each fraud stopped saves its amount. Each legit payment sent to step-up costs friction; we assume
**Rs 100 per false flag** (support contact + the share of users who abandon). The threshold is picked on
**April** using a model trained on Jan–Mar, then frozen and applied to the final model on May–Jun. Choosing
it on the test set would overstate the savings.
"""),
code("""
FP_COST = 100
def money_curve(y, s, amt, ths):
    rows = []
    for t in ths:
        f = s >= t
        tp, fp = int((f & (y == 1)).sum()), int((f & (y == 0)).sum())
        saved = amt[f & (y == 1)].sum()
        rows.append((t, tp, fp, saved, saved - fp * FP_COST))
    return pd.DataFrame(rows, columns=['threshold', 'tp', 'fp', 'fraud_saved_inr', 'net_inr'])

fit, val = df[df.ts < '2026-04-01'], df[(df.ts >= '2026-04-01') & (df.ts < '2026-05-01')]
vb = FraudBlend().fit(fit[X], fit.is_fraud)
ths = np.linspace(0.5, 0.995, 100)
vcurve = money_curve(val.is_fraud.values, vb.predict_proba(val[X])[:, 1], val.amount.values, ths)
THRESHOLD = float(vcurve.loc[vcurve.net_inr.idxmax(), 'threshold'])
print(f'Threshold chosen on April validation: {THRESHOLD:.3f}  (frozen from here on)')

y, s, amt = yte.values, P['Blend (selected)'], test.amount.values
flag = s >= THRESHOLD
k = int(rule.sum()); topk = np.argsort(-s)[:k]
def row(f):
    tp, fp = int((f & (y == 1)).sum()), int((f & (y == 0)).sum()); saved = amt[f & (y == 1)].sum()
    return [tp, fp, f'{tp / max(tp + fp, 1):.0%}', f'{tp / y.sum():.0%}', saved, saved - fp * FP_COST]
same = np.zeros(len(y), bool); same[topk] = True
summary = pd.DataFrame({'Legacy rules': row(rule == 1), f'Blend @ {THRESHOLD:.3f} (frozen)': row(flag),
                        f'Blend, same # flags as rules ({k})': row(same)},
                       index=['fraud caught', 'legit flagged', 'precision', 'recall', 'fraud value stopped (Rs)', 'net savings (Rs)'])
print(f'Total fraud value in test period: Rs {amt[y == 1].sum():,.0f}')
summary
"""),
code("""
fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.plot(vcurve.threshold, vcurve.net_inr / 1e5, color=ps.BLUE)
ax.axvline(THRESHOLD, color=ps.INK2, ls='--', lw=1)
ax.text(THRESHOLD - 0.01, vcurve.net_inr.max() / 1e5 * 0.5, f'chosen = {THRESHOLD:.3f}', color=ps.INK2, ha='right')
ax.set_xlabel('blend risk score threshold'); ax.set_ylabel('net savings (Rs lakh)')
ax.set_title('Net savings vs threshold on April validation (threshold then frozen)')
fig.tight_layout(); ps.save(fig, '10_fraud_threshold');
"""),
md("## What drives the score? (permutation importance on test data)"),
code("""
sample = test.sample(60000, random_state=1)
imp = permutation_importance(blend, sample[X], sample.is_fraud, scoring='average_precision',
                             n_repeats=4, random_state=1, n_jobs=-1)
fi = pd.Series(imp.importances_mean, index=X).sort_values().tail(10)
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.barh(fi.index, fi.values, color=ps.BLUE, height=0.6); ax.grid(axis='y', visible=False)
ax.set_title('Top 10 features (drop in PR-AUC when shuffled)')
fig.tight_layout(); ps.save(fig, '11_fraud_importance');
"""),
code("""
# Recall by fraud scheme (ground-truth scheme labels are used ONLY here, for diagnosis)
schemes = pd.read_csv('../data/ground_truth/fraud_schemes.csv')
tt = test.assign(flag=flag, rule=rule).merge(schemes, on='txn_id')
tt.groupby('fraud_scheme').agg(frauds=('txn_id', 'size'), model_recall=('flag', 'mean'), rules_recall=('rule', 'mean')).round(2)
"""),
md("""
## Takeaways
- The blended model beats the static rules on every axis: **for the same number of flags it catches about
  twice as much fraud**, and at the validation-chosen threshold it stops far more fraud value net of friction cost.
- Beyond the amount itself, the strongest signals are about the **payee** (how new the account is, its KYC tier) and
  the **payer's behaviour** (how established the account is, bursts of payments in the last hour, a new device). A static rule set
  can't express these well.
- Collect-request scams, which the rules miss entirely, are caught most of the time. Fake-merchant QR fraud is the
  weak spot: merchant-side features (refund rate, complaint history) are the next thing to add.
- **Deployment idea:** score → (a) high risk: hold + re-authenticate; (b) medium + COLLECT: full-screen
  "you are about to SEND money" warning; (c) confirmed mule accounts feed a blocklist.
- Limitations: synthetic labels arrive instantly; in production fraud labels lag by weeks, so the training
  window must exclude the most recent days and the model needs drift monitoring.
"""),
]

# =========================================================================== #
nb3 = [
md("""
# 03 · Early churn prediction — who will leave after their first month?

**Definition.** Using only a user's **first 30 days**, predict whether they make **zero transactions in
days 31–60** (churn). Users who signed up by 30 Apr 2026 are used, so every label is fully observed.
**Use case:** a weekly list of high-risk new users for the retention team (nudges, bill-pay onboarding,
proactive refund updates) instead of blanket cashback.
"""),
code("""
import sys, warnings; sys.path.insert(0, '../src'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import plot_style as ps
from features import load, churn_table
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
ps.apply()
ct = churn_table(load())
print(ct.shape, f'churn rate {ct.churned.mean():.1%}')
ct.head()
"""),
code("""
cat = ['acquisition_channel', 'age_band', 'primary_bank', 'kyc_status']
num = ['city_tier', 'n_txns', 'active_days', 'total_amount', 'avg_amount', 'n_unique_payees', 'share_p2m', 'share_bill',
       'n_tech_fail', 'n_user_fail', 'n_risk_blocked', 'got_cashback', 'days_since_last_txn', 'n_refunds',
       'max_refund_tat_h', 'n_tickets', 'min_csat', 'refund_breached_tat']
X, y = ct[cat + num].copy(), ct.churned
for c in ['n_txns', 'total_amount', 'avg_amount', 'n_unique_payees']:
    X[c] = np.log1p(X[c])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat), ('num', StandardScaler(), num)])
# Interpretable model: drop bank dummies (noisy) and the activity counts that are collinear with
# active_days (n_txns, n_unique_payees, total_amount): with them in, coefficients flip sign and stop being readable
cat_l = [c for c in cat if c != 'primary_bank']
num_l = [c for c in num if c not in ('total_amount', 'n_txns', 'n_unique_payees')]
pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_l), ('num', StandardScaler(), num_l)])
logit = Pipeline([('pre', pre), ('m', LogisticRegression(max_iter=3000, C=0.5))]).fit(Xtr, ytr)
hgb = Pipeline([('pre', ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat)], remainder='passthrough')),
                ('m', HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4, random_state=42))]).fit(Xtr, ytr)
p_l, p_h = logit.predict_proba(Xte)[:, 1], hgb.predict_proba(Xte)[:, 1]
pd.DataFrame({'ROC-AUC': [roc_auc_score(yte, p_l), roc_auc_score(yte, p_h)],
              'PR-AUC': [average_precision_score(yte, p_l), average_precision_score(yte, p_h)]},
             index=['Logistic regression', 'Gradient boosting']).round(3)
"""),
md("## Explainability: logistic-regression odds ratios\nInterpretable coefficients matter here: the retention team needs to know *why* a user is at risk."),
code("""
names = logit.named_steps['pre'].get_feature_names_out()
coef = pd.Series(logit.named_steps['m'].coef_[0], index=[n.split('__')[1] for n in names])
top = pd.concat([coef.nlargest(7), coef.nsmallest(6)]).sort_values()
fig, ax = plt.subplots(figsize=(7.5, 4.3))
ax.barh(top.index, top.values, color=[ps.ORANGE if v > 0 else ps.BLUE for v in top.values], height=0.6)
ax.axvline(0, color=ps.INK2, lw=0.8); ax.grid(axis='y', visible=False)
ax.set_title('Churn drivers (standardised log-odds) — orange raises risk, blue lowers it')
fig.tight_layout(); ps.save(fig, '12_churn_drivers');
np.exp(top).rename('odds ratio per 1 SD / vs baseline').round(2).sort_values(ascending=False)
"""),
md("## Is the score useful? Lift by risk decile"),
code("""
dec = pd.DataFrame({'p': p_h, 'y': yte.values})
dec['decile'] = pd.qcut(dec.p.rank(method='first', ascending=False), 10, labels=range(1, 11))
lift = dec.groupby('decile', observed=True).agg(users=('y', 'size'), churn_rate=('y', 'mean'))
lift['lift'] = lift.churn_rate / yte.mean()
lift['cum_share_of_churners'] = dec.groupby('decile', observed=True).y.sum().cumsum() / dec.y.sum()
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.bar(lift.index.astype(str), lift.churn_rate * 100, color=[ps.ORANGE if i <= 3 else ps.GREY for i in lift.index], width=0.6)
ax.axhline(yte.mean() * 100, color=ps.INK2, ls='--', lw=1); ax.text(9.6, yte.mean() * 100 + 1.5, 'avg', color=ps.INK2)
ax.set_xlabel('risk decile (1 = highest predicted risk)'); ax.set_ylabel('actual churn %'); ax.grid(axis='x', visible=False)
ax.set_title('Actual churn by predicted-risk decile'); fig.tight_layout(); ps.save(fig, '13_churn_lift');
print(f"Top 3 deciles (30% of users) contain {lift.cum_share_of_churners.iloc[2]:.0%} of churners")
lift.round(2)
"""),
md("""
## Takeaways
- Early behaviour predicts churn well. The strongest drivers are **low activity / days since last payment**,
  **cashback-only acquisition**, **technical failures and refunds that breached T+1**, while **bill payments and
  more unique payees** (habit formation) protect.
- Targeting the **top 3 risk deciles** reaches a large majority of future churners while contacting only 30% of users. That is a cheaper
  alternative to blanket cashback.
- Next step: an A/B test of retention nudges on the high-risk group to measure *incremental* retention (uplift),
  because a churn score alone says who will leave, not who can be saved.
"""),
]

for name, cells in [("01_eda_payment_health", nb1), ("02_fraud_detection_model", nb2), ("03_churn_prediction", nb3)]:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    print(f"executing {name} ...")
    ExecutePreprocessor(timeout=1200, kernel_name="python3").preprocess(nb, {"metadata": {"path": HERE}})
    nbf.write(nb, os.path.join(HERE, f"{name}.ipynb"))
print("done")
