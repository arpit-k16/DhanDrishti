"""DhanDrishti · interactive payment-health dashboard (Streamlit + Plotly).

Run:  streamlit run app.py
Model outputs are pre-computed by `python src/build_app_data.py`, so the app never trains a model on load.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).parent
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
TECH = ["BANK_SERVER_DOWN", "TIMEOUT", "NETWORK_ERROR"]
END = pd.Timestamp("2026-06-30")

BLUE, ORANGE, AQUA, GREY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2", "#e34948"
SEQ = ["#eef5fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

st.set_page_config(page_title="DhanDrishti · Payment Health", page_icon="📊", layout="wide")


# --------------------------------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading 3 lakh payments…")
def load_data():
    users = pd.read_csv(RAW / "users.csv", parse_dates=["signup_ts"])
    banks = pd.read_csv(RAW / "banks.csv")
    merchants = pd.read_csv(RAW / "merchants.csv")
    campaigns = pd.read_csv(RAW / "campaigns.csv")
    tx = pd.read_csv(RAW / "transactions.csv", parse_dates=["ts"])
    refunds = pd.read_csv(RAW / "refunds.csv", parse_dates=["debit_ts", "refund_ts"])
    tickets = pd.read_csv(RAW / "support_tickets.csv", parse_dates=["created_ts", "resolved_ts"])
    tx = tx.merge(users[["user_id", "acquisition_channel", "city_tier", "age_band", "signup_ts"]], on="user_id", how="left")
    tx["is_tech_fail"] = tx.failure_reason.isin(TECH)
    tx["ok"] = tx.status.eq("SUCCESS")
    tx["gmv_ok"] = np.where(tx.ok, tx.amount, 0.0)
    tx["hour"] = tx.ts.dt.hour
    tx["date"] = tx.ts.dt.date
    bank_name = banks.set_index("bank_code").bank_name.to_dict()
    tx["payer_bank_name"] = tx.payer_bank.map(bank_name)
    return users, banks, merchants, campaigns, tx, refunds, tickets, bank_name


@st.cache_data
def load_models():
    scores = pd.read_csv(PROC / "fraud_scores.csv.gz")
    churn = pd.read_csv(PROC / "churn_scores.csv")
    drivers = pd.read_csv(PROC / "churn_drivers.csv")
    meta = json.load(open(PROC / "model_meta.json"))
    schemes = pd.read_csv(ROOT / "data" / "ground_truth" / "fraud_schemes.csv")
    return scores, churn, drivers, meta, schemes


users, banks, merchants, campaigns, TX, REFUNDS, TICKETS, BANK_NAME = load_data()
SCORES, CHURN, DRIVERS, META, SCHEMES = load_models()


def style(fig, height=360, legend=False):
    fig.update_layout(height=height, margin=dict(l=0, r=10, t=60 if legend else 40, b=0), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(size=12, color="#2b2b29"), showlegend=legend,
                      title=dict(x=0, xanchor="left", font=dict(size=14)), hoverlabel=dict(font_size=12),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title_text=""))
    fig.update_xaxes(showgrid=False, automargin=True, title_standoff=8)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", automargin=True, title_standoff=8)
    return fig


def show(fig, **kw):
    st.plotly_chart(fig, width="stretch", theme=None, **kw)


def inr(x, unit="lakh"):
    return f"₹{x / 1e5:,.1f} L" if unit == "lakh" else f"₹{x / 1e7:,.2f} Cr"


# --------------------------------------------------------------------------------------------------- #
# Sidebar filters (drive every tab)
# --------------------------------------------------------------------------------------------------- #
st.sidebar.title("DhanDrishti")
st.sidebar.caption("Payment reliability, refunds, retention, fraud & promo abuse on a simulated UPI app · Jan–Jun 2026")
st.sidebar.header("Filters")
d0, d1 = TX.ts.min().date(), TX.ts.max().date()
dates = st.sidebar.date_input("Date range", (d0, d1), min_value=d0, max_value=d1)
if not isinstance(dates, (list, tuple)) or len(dates) != 2:
    st.sidebar.info("Pick an end date to apply the range.")
    dates = (d0, d1)
bank_opts = sorted(BANK_NAME.values())
sel_banks = st.sidebar.multiselect("Payer bank", bank_opts, placeholder="All banks")
type_opts = ["P2M", "P2P", "BILL", "RECHARGE", "COLLECT"]
sel_types = st.sidebar.multiselect("Payment type", type_opts, placeholder="All types")
sel_ch = st.sidebar.multiselect("Acquisition channel", ["organic", "referral", "cashback_campaign"], placeholder="All channels")
sel_tier = st.sidebar.multiselect("City tier", [1, 2, 3], placeholder="All tiers")
sel_age = st.sidebar.multiselect("Age band", ["18-24", "25-34", "35-44", "45-54", "55+"], placeholder="All ages")
st.sidebar.caption("Empty filter = everything. Retention and the ML tabs use user-level filters only "
                   "(their time windows are fixed by the model design).")


def apply_user_filters(df):
    m = np.ones(len(df), bool)
    if sel_ch:
        m &= df.acquisition_channel.isin(sel_ch).values
    if sel_tier:
        m &= df.city_tier.isin(sel_tier).values
    if sel_age:
        m &= df.age_band.isin(sel_age).values
    return df[m]


def apply_tx_filters(df, use_dates=True):
    df = apply_user_filters(df)
    m = np.ones(len(df), bool)
    if use_dates:
        m &= (df.date >= dates[0]).values & (df.date <= dates[1]).values
    if sel_banks:
        m &= df.payer_bank_name.isin(sel_banks).values
    if sel_types:
        m &= df.txn_type.isin(sel_types).values
    return df[m]


tx = apply_tx_filters(TX)
if tx.empty:
    st.warning("No payments match these filters. Widen the filters in the sidebar.")
    st.stop()
tx_ids = set(tx.txn_id)
refunds = REFUNDS[REFUNDS.txn_id.isin(tx_ids)].merge(TX[["txn_id", "payer_bank"]], on="txn_id")
tickets = TICKETS[TICKETS.txn_id.isin(tx_ids)]

# --------------------------------------------------------------------------------------------------- #
# Header + KPIs
# --------------------------------------------------------------------------------------------------- #
st.title("DhanDrishti · Payment Health")
st.caption(f"{len(tx):,} payments in view · {dates[0]:%d %b} – {dates[1]:%d %b %Y} · synthetic data modelled on "
           "publicly reported UPI pain points")
k = st.columns(6)
k[0].metric("GMV (successful)", inr(tx.gmv_ok.sum(), "cr"))
k[1].metric("Payments", f"{len(tx):,}")
k[2].metric("Success rate", f"{tx.ok.mean():.2%}")
k[3].metric("Active users", f"{tx.user_id.nunique():,}")
k[4].metric("Refunds breaching T+1", f"{(refunds.tat_hours > 24).mean():.1%}" if len(refunds) else "–")
k[5].metric("Fraud losses", inr(tx.loc[(tx.is_fraud == 1) & tx.ok, "amount"].sum()))

tabs = st.tabs(["📈 Overview", "🏦 Reliability", "↩️ Refunds & support", "👥 Retention", "🚨 Fraud & abuse", "🤖 Fraud model lab"])

# --------------------------------------------------------------------------------------------------- #
# 1 · Overview
# --------------------------------------------------------------------------------------------------- #
with tabs[0]:
    grain = st.radio("Time grain", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True)
    freq = {"Daily": "D", "Weekly": "W-MON", "Monthly": "MS"}[grain]
    ts = (tx.set_index("ts").groupby(pd.Grouper(freq=freq, label="left", closed="left"))
            .agg(gmv=("gmv_ok", "sum"), attempts=("txn_id", "size"), success=("ok", "mean"), users=("user_id", "nunique"))
            .reset_index())
    # keep only complete periods, so a 2-day stub week doesn't look like a crash
    off = pd.tseries.frequencies.to_offset(freq)
    start, stop = pd.Timestamp(dates[0]), pd.Timestamp(dates[1]) + pd.Timedelta(days=1)
    ts = ts[(ts.attempts > 0) & (ts.ts >= start) & (ts.ts + off <= stop)] if grain != "Daily" else ts[ts.attempts > 0]
    if ts.empty:
        st.info(f"The selected date range doesn't contain a complete {grain.lower()[:-2] if grain != 'Daily' else 'day'}. Try a finer time grain.")
        st.stop()
    st.caption("Partial periods at the edges of the date range are left out.")
    c1, c2 = st.columns(2)
    with c1:
        f = px.bar(ts, x="ts", y=ts.gmv / 1e5, title=f"{grain} GMV (₹ lakh)", color_discrete_sequence=[BLUE],
                   labels={"y": "₹ lakh", "ts": ""})
        f.update_traces(hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.1f} lakh<extra></extra>")
        show(style(f), key="ov_gmv")
    with c2:
        f = px.line(ts, x="ts", y="users", title=f"{grain} active users", markers=True, color_discrete_sequence=[BLUE],
                    labels={"users": "users", "ts": ""})
        f.update_yaxes(rangemode="tozero")
        f.update_traces(hovertemplate="%{x|%d %b %Y}<br>%{y:,} users<extra></extra>")
        show(style(f), key="ov_users")
    f = px.line(ts, x="ts", y=ts.success * 100, title=f"{grain} payment success rate (%)", markers=True,
                color_discrete_sequence=[BLUE], labels={"y": "%", "ts": ""})
    f.update_traces(hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}%<extra></extra>")
    show(style(f, 280), key="ov_sr")
    c1, c2 = st.columns(2)
    with c1:
        mix = (tx.groupby("txn_type").agg(payments=("txn_id", "size"), gmv=("gmv_ok", "sum"), avg_ticket=("amount", "mean"))
                 .reset_index().sort_values("payments"))
        f = px.bar(mix, y="txn_type", x="payments", orientation="h", title="Payments by type",
                   color_discrete_sequence=[BLUE], custom_data=["gmv", "avg_ticket"], labels={"txn_type": "", "payments": "payments"})
        f.update_traces(hovertemplate="%{y}: %{x:,} payments<br>GMV ₹%{customdata[0]:,.0f}<br>avg ticket ₹%{customdata[1]:,.0f}<extra></extra>")
        show(style(f, 320), key="ov_mix")
    with c2:
        mc = (tx[(tx.payee_type == "MERCHANT") & tx.ok].merge(merchants[["merchant_id", "category"]], left_on="payee_id", right_on="merchant_id")
                .groupby("category").amount.sum().sort_values().tail(10) / 1e5)
        f = px.bar(x=mc.values, y=mc.index, orientation="h", title="Merchant GMV by category (₹ lakh, top 10)",
                   color_discrete_sequence=[BLUE], labels={"x": "₹ lakh", "y": ""})
        f.update_traces(hovertemplate="%{y}: ₹%{x:,.1f} lakh<extra></extra>")
        show(style(f, 320), key="ov_cat")

# --------------------------------------------------------------------------------------------------- #
# 2 · Reliability
# --------------------------------------------------------------------------------------------------- #
with tabs[1]:
    st.subheader("Where and when do payments break?")
    st.caption("Technical declines (bank server down, timeout, network) are the platform's problem. Wrong PIN and low balance are the user's.")


    @st.cache_data
    def retry_stats(ids):
        t = TX[TX.txn_id.isin(ids)]
        fails = t[t.is_tech_fail][["txn_id", "user_id", "payee_id", "ts", "amount"]].sort_values("ts")
        succ = TX[TX.ok][["user_id", "payee_id", "ts"]].rename(columns={"ts": "ts_ok"}).sort_values("ts_ok")
        m = pd.merge_asof(fails, succ, left_on="ts", right_on="ts_ok", by=["user_id", "payee_id"], direction="forward",
                          tolerance=pd.Timedelta(minutes=30), allow_exact_matches=False)
        rec = m.ts_ok.notna()
        return len(m), rec.mean() if len(m) else np.nan, m.loc[~rec, "amount"].sum()

    n_f, rec_rate, lost = retry_stats(frozenset(tx_ids))
    k = st.columns(4)
    k[0].metric("Technical decline rate", f"{tx.is_tech_fail.mean():.2%}")
    k[1].metric("Technical declines", f"{n_f:,}")
    k[2].metric("Recovered by retry < 30 min", f"{rec_rate:.0%}" if n_f else "–")
    k[3].metric("GMV abandoned", inr(lost))

    c1, c2 = st.columns([1, 1.4])
    with c1:
        b = tx.groupby("payer_bank_name").agg(rate=("is_tech_fail", "mean"), attempts=("txn_id", "size")).reset_index()
        b["rate"] *= 100
        b = b.sort_values("rate")
        avg = tx.is_tech_fail.mean() * 100
        worst = b.nlargest(min(3, len(b)), "rate").payer_bank_name
        f = px.bar(b, y="payer_bank_name", x="rate", orientation="h", title="Technical decline rate by payer bank (%)",
                   color=b.payer_bank_name.isin(worst).map({True: "Worst 3", False: "Others"}),
                   color_discrete_map={"Worst 3": ORANGE, "Others": GREY}, custom_data=["attempts"],
                   labels={"payer_bank_name": "", "rate": "%"})
        f.update_traces(hovertemplate="%{y}: %{x:.2f}% of %{customdata[0]:,} attempts<extra></extra>")
        f.add_vline(x=avg, line_dash="dash", line_color="#52514e", annotation_text=f"avg {avg:.1f}%", annotation_position="top")
        show(style(f, 380), key="rel_bank")
    with c2:
        hm = tx.pivot_table(index="payer_bank_name", columns="hour", values="is_tech_fail", aggfunc="mean") * 100
        hm = hm.reindex(index=b.payer_bank_name[::-1], columns=range(24))   # same order as the bar chart
        f = px.imshow(hm, color_continuous_scale=SEQ, zmin=0, zmax=10, aspect="auto",
                      title="Technical decline % · bank × hour of day", labels=dict(x="hour", y="", color="% declined"))
        f.update_traces(hovertemplate="%{y} · %{x}:00<br>%{z:.1f}% declined<extra></extra>")
        show(style(f, 380), key="rel_heat")
    c1, c2 = st.columns(2)
    with c1:
        hr = tx.groupby("hour").agg(rate=("is_tech_fail", "mean"), attempts=("txn_id", "size")).reset_index()
        f = go.Figure()
        f.add_bar(x=hr.hour, y=hr.attempts, marker_color=GREY, name="attempts", hovertemplate="%{x}:00 · %{y:,} attempts<extra></extra>")
        f.update_layout(title="Payment volume by hour")
        show(style(f, 300), key="rel_vol")
    with c2:
        f = px.line(hr, x="hour", y=hr.rate * 100, markers=True, title="Technical decline % by hour",
                    color_discrete_sequence=[ORANGE], labels={"y": "%", "hour": "hour"})
        f.update_yaxes(rangemode="tozero")
        show(style(f, 300), key="rel_hour")
    reasons = (tx[tx.status == "FAILED"].groupby("failure_reason")
                 .agg(declines=("txn_id", "size"), value_inr=("amount", "sum")).sort_values("declines", ascending=False).reset_index())
    reasons["type"] = np.where(reasons.failure_reason.isin(TECH), "technical",
                               np.where(reasons.failure_reason == "RISK_BLOCKED", "risk rules", "user-side"))
    reasons["share"] = reasons.declines / reasons.declines.sum()
    st.dataframe(reasons, hide_index=True, width="stretch",
                 column_config={"value_inr": st.column_config.NumberColumn("value (₹)", format="₹%d"),
                                "share": st.column_config.ProgressColumn("share", format="%.0f%%", min_value=0, max_value=1)})

# --------------------------------------------------------------------------------------------------- #
# 3 · Refunds & support
# --------------------------------------------------------------------------------------------------- #
with tabs[2]:
    st.subheader('"Money debited, payment failed" refunds')
    st.caption("RBI's TAT framework: failed UPI debits must be reversed by T+1, else ₹100/day compensation is due.")
    if refunds.empty:
        st.info("No debited-but-failed payments in this selection.")
    else:
        k = st.columns(4)
        k[0].metric("Debited-but-failed payments", f"{len(refunds):,}")
        k[1].metric("Breach T+1", f"{(refunds.tat_hours > 24).mean():.1%}")
        k[2].metric("P90 refund time", f"{refunds.tat_hours.quantile(.9):.0f} h")
        k[3].metric("Compensation owed", f"₹{refunds.compensation_due_inr.sum():,.0f}")
        c1, c2 = st.columns(2)
        with c1:
            cap = st.slider("Show refunds up to (hours)", 24, 400, 200, step=8)
            r = refunds[refunds.tat_hours <= cap]
            f = px.histogram(r, x="tat_hours", nbins=50, title="Hours from debit to refund",
                             color=np.where(r.tat_hours > 24, "Breached T+1", "Within T+1"),
                             color_discrete_map={"Within T+1": BLUE, "Breached T+1": ORANGE}, labels={"tat_hours": "hours"})
            f.add_vline(x=24, line_color=RED, annotation_text="T+1", annotation_position="top right")
            show(style(f, 340, legend=True), key="ref_hist")
        with c2:
            rb = refunds.groupby("payer_bank").agg(breach=("tat_hours", lambda s: (s > 24).mean() * 100),
                                                   comp=("compensation_due_inr", "sum"), n=("refund_id", "size")).reset_index()
            rb["bank"] = rb.payer_bank.map(BANK_NAME)
            rb = rb.sort_values("breach")
            f = px.bar(rb, y="bank", x="breach", orientation="h", title="T+1 breach rate by bank (%)", custom_data=["comp", "n"],
                       color_discrete_sequence=[ORANGE], labels={"bank": "", "breach": "%"})
            f.update_traces(hovertemplate="%{y}: %{x:.1f}% of %{customdata[1]} refunds<br>compensation ₹%{customdata[0]:,.0f}<extra></extra>")
            show(style(f, 400), key="ref_bank")
    st.divider()
    st.subheader("Customer support")
    if tickets.empty:
        st.info("No tickets in this selection.")
    else:
        t = tickets.assign(res_h=(tickets.resolved_ts - tickets.created_ts).dt.total_seconds() / 3600)
        c1, c2 = st.columns(2)
        with c1:
            cat = t.groupby("category").agg(tickets=("ticket_id", "size"), avg_res_h=("res_h", "mean"),
                                            csat=("csat", "mean")).reset_index().sort_values("tickets")
            f = px.bar(cat, y="category", x="tickets", orientation="h", title="Tickets by category", custom_data=["avg_res_h", "csat"],
                       color_discrete_sequence=[BLUE], labels={"category": ""})
            f.update_traces(hovertemplate="%{y}: %{x} tickets<br>avg resolution %{customdata[0]:.0f} h · CSAT %{customdata[1]:.1f}<extra></extra>")
            show(style(f, 340), key="sup_cat")
        with c2:
            tt = t.dropna(subset=["csat", "res_h"])
            tt["bucket"] = pd.cut(tt.res_h, [0, 12, 48, 120, 1e5], labels=["≤ 12 h", "12–48 h", "2–5 days", "> 5 days"])
            cs = tt.groupby("bucket", observed=True).agg(csat=("csat", "mean"), n=("csat", "size")).reset_index()
            f = px.bar(cs, x="bucket", y="csat", title="Average CSAT (1–5) by resolution time", custom_data=["n"],
                       color_discrete_sequence=[BLUE], labels={"bucket": "", "csat": "CSAT"}, text_auto=".1f")
            f.update_yaxes(range=[0, 5])
            f.update_traces(hovertemplate="%{x}: CSAT %{y:.2f} (%{customdata[0]} rated tickets)<extra></extra>")
            show(style(f, 340), key="sup_csat")

# --------------------------------------------------------------------------------------------------- #
# 4 · Retention
# --------------------------------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("Do users stay?")
    st.caption("Churned = no payment in the last 30 days of the window. Only users who signed up before 30 Apr are counted, "
               "so everyone had at least 2 months to show it.")
    U = apply_user_filters(users)
    last = TX.groupby("user_id").ts.max()
    U = U.assign(last_ts=U.user_id.map(last))
    U["churned"] = (END - U.last_ts).dt.total_seconds() / 86400 > 30
    mature = U[U.signup_ts < "2026-04-30"]
    if mature.empty:
        st.info("No users match these filters.")
    else:
        early = TX[TX.is_tech_fail & ((TX.ts - TX.signup_ts).dt.total_seconds() / 86400 <= 14)].user_id.unique()
        mature = mature.assign(early_fail=mature.user_id.isin(early))
        k = st.columns(3)
        k[0].metric("Users (signed up before May)", f"{len(mature):,}")
        k[1].metric("Churn rate", f"{mature.churned.mean():.0%}")
        ef = mature.groupby("early_fail").churned.mean()
        if len(ef) == 2:
            k[2].metric("Churn after a failure in first 14 days", f"{ef[True]:.0%}", f"{(ef[True] - ef[False]) * 100:+.0f} pts vs clean start",
                        delta_color="inverse")
        c1, c2 = st.columns(2)
        with c1:
            ch = mature.groupby("acquisition_channel").churned.mean().mul(100).sort_values().reset_index()
            f = px.bar(ch, y="acquisition_channel", x="churned", orientation="h", title="Churn rate by acquisition channel (%)",
                       color=ch.acquisition_channel.eq("cashback_campaign").map({True: "cashback", False: "other"}),
                       color_discrete_map={"cashback": ORANGE, "other": GREY}, labels={"acquisition_channel": "", "churned": "%"},
                       text_auto=".0f")
            show(style(f, 300), key="ret_ch")
        with c2:
            a = TX[TX.user_id.isin(U.user_id)][["user_id", "ts", "signup_ts"]]
            a = a.assign(cohort=a.signup_ts.dt.to_period("M"), m=(a.ts.dt.to_period("M") - a.signup_ts.dt.to_period("M")).apply(lambda x: x.n))
            coh = a.groupby(["cohort", "m"]).user_id.nunique().unstack()
            ret = coh.div(coh[0], axis=0) * 100
            for c in ret.index:
                ret.loc[c, [m for m in ret.columns if c + m > pd.Period("2026-06", "M")]] = np.nan
            ret.index = ret.index.astype(str)
            ret.columns = [f"M{m}" for m in ret.columns]
            f = px.imshow(ret, text_auto=".0f", color_continuous_scale=SEQ, zmin=0, zmax=100, aspect="auto",
                          title="Monthly cohort retention (% of cohort active)", labels=dict(x="months since signup", y="signup cohort", color="%"))
            show(style(f, 300), key="ret_coh")
        # campaign ROI
        cb = TX[TX.cashback_inr > 0].merge(users[["user_id", "campaign_id"]], on="user_id")
        roi = []
        for cmp in campaigns.itertuples():
            acq = U[U.campaign_id == cmp.campaign_id]
            if acq.empty:
                continue
            spent = cb[cb.user_id.isin(acq.user_id)].cashback_inr.sum()
            june = TX[(TX.user_id.isin(acq.user_id)) & (TX.ts >= "2026-06-01")].user_id.nunique()
            roi.append({"campaign": cmp.code, "users acquired": len(acq), "active in June": june,
                        "cashback spent (₹)": spent, "₹ per install": spent / len(acq), "₹ per retained user": spent / june if june else np.nan})
        if roi:
            st.markdown("**Cashback campaign ROI**")
            st.dataframe(pd.DataFrame(roi), hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.0f") for c in ["cashback spent (₹)", "₹ per install", "₹ per retained user"]})

    st.divider()
    st.subheader("Early-churn model: who to contact")
    st.caption(f"Gradient boosting on each user's first 30 days predicts no payment in days 31–60. Held-out users only. "
               f"ROC-AUC {META['churn']['roc_auc_gbm']:.2f}.")
    C = apply_user_filters(CHURN)
    if len(C) < 20:
        st.info("Too few held-out users for these filters.")
    else:
        pct = st.slider("Contact the riskiest … % of new users", 5, 100, 30, step=5)
        C = C.sort_values("risk", ascending=False)
        n = int(np.ceil(len(C) * pct / 100))
        cap = C.churned.iloc[:n].sum() / max(C.churned.sum(), 1)
        k = st.columns(3)
        k[0].metric("Users contacted", f"{n:,} of {len(C):,}")
        k[1].metric("Future churners reached", f"{cap:.0%}")
        k[2].metric("Lift vs random", f"{cap / (pct / 100):.1f}×")
        c1, c2 = st.columns(2)
        with c1:
            C["decile"] = pd.qcut(C.risk.rank(method="first", ascending=False), 10, labels=[str(i) for i in range(1, 11)])
            lift = C.groupby("decile", observed=True).churned.mean().mul(100).reset_index()
            lift["in"] = lift.decile.astype(int) <= int(np.ceil(pct / 10))
            f = px.bar(lift, x="decile", y="churned", title="Actual churn % by predicted-risk decile (1 = riskiest)",
                       color=lift["in"].map({True: "contacted", False: "not contacted"}),
                       color_discrete_map={"contacted": ORANGE, "not contacted": GREY}, labels={"churned": "% churned", "decile": "risk decile"})
            f.add_hline(y=C.churned.mean() * 100, line_dash="dash", line_color="#52514e", annotation_text="average")
            show(style(f, 340, legend=True), key="ret_lift")
        with c2:
            dv = DRIVERS.set_index("feature").coef
            dv = pd.concat([dv.nlargest(6), dv.nsmallest(6)]).sort_values()
            f = px.bar(x=dv.values, y=dv.index, orientation="h", title="What drives churn risk (logistic regression, standardised)",
                       color=np.where(dv.values > 0, "raises risk", "lowers risk"),
                       color_discrete_map={"raises risk": ORANGE, "lowers risk": BLUE}, labels={"x": "log-odds", "y": ""})
            show(style(f, 340, legend=True), key="ret_drv")
        top = C.iloc[:n].drop(columns=["decile"])
        st.dataframe(top.head(200), hide_index=True, width="stretch",
                     column_config={"risk": st.column_config.ProgressColumn("risk", min_value=0, max_value=1, format="%.2f")})
        st.download_button("Download contact list (CSV)", top.to_csv(index=False), "high_risk_users.csv", "text/csv")

# --------------------------------------------------------------------------------------------------- #
# 5 · Fraud & abuse
# --------------------------------------------------------------------------------------------------- #
with tabs[4]:
    st.subheader("Fraud: who gets hit, and what the current rules miss")
    fr = tx[tx.is_fraud == 1]
    blocked = tx.failure_reason.eq("RISK_BLOCKED")
    k = st.columns(4)
    k[0].metric("Fraudulent payments", f"{len(fr):,}")
    k[1].metric("Fraud losses", inr(fr.loc[fr.ok, "amount"].sum()))
    k[2].metric("Rules catch (recall)", f"{blocked[tx.is_fraud == 1].mean():.0%}" if len(fr) else "–")
    k[3].metric("Rule blocks that were legit", f"{(tx[blocked].is_fraud == 0).mean():.0%}" if blocked.any() else "–")
    c1, c2 = st.columns(2)
    with c1:
        vic = set(fr.user_id)
        ab = apply_user_filters(users).assign(v=lambda d: d.user_id.isin(vic)).groupby("age_band").v.mean().mul(100).reset_index()
        f = px.bar(ab, x="age_band", y="v", title="% of users defrauded, by age band", text_auto=".1f",
                   color=ab.age_band.isin(["45-54", "55+"]).map({True: "45+", False: "under 45"}),
                   color_discrete_map={"45+": ORANGE, "under 45": GREY}, labels={"v": "%", "age_band": ""})
        show(style(f, 320), key="fr_age")
    with c2:
        ft = tx.groupby("txn_type").is_fraud.mean().mul(100).sort_values().reset_index()
        f = px.bar(ft, y="txn_type", x="is_fraud", orientation="h", title="Fraud rate by payment type (%)", text_auto=".2f",
                   color=ft.txn_type.eq("COLLECT").map({True: "collect", False: "other"}),
                   color_discrete_map={"collect": ORANGE, "other": GREY}, labels={"is_fraud": "%", "txn_type": ""})
        show(style(f, 320), key="fr_type")
    st.divider()
    st.subheader("Promo abuse: one device, many accounts")
    min_acc = st.slider("Flag devices with at least … accounts", 2, 8, 3)
    U = apply_user_filters(users)
    dev = users.groupby("device_id").agg(accounts=("user_id", "size"), first=("signup_ts", "min"), last=("signup_ts", "max"))
    dev["signup_span_h"] = (dev["last"] - dev["first"]).dt.total_seconds() / 3600
    cbu = TX.groupby("user_id").cashback_inr.sum()
    dev = dev.join(users.assign(cb=users.user_id.map(cbu).fillna(0)).groupby("device_id").cb.sum().rename("cashback_inr"))
    farms = dev[(dev.accounts >= min_acc) & dev.index.isin(U.device_id)].sort_values("cashback_inr", ascending=False)
    total_cb = TX[TX.user_id.isin(U.user_id)].cashback_inr.sum()
    k = st.columns(3)
    k[0].metric("Flagged devices", f"{len(farms):,}")
    k[1].metric("Accounts on them", f"{int(farms.accounts.sum()):,}")
    k[2].metric("Share of cashback they took", f"{farms.cashback_inr.sum() / total_cb:.1%}" if total_cb else "–")
    st.dataframe(farms.reset_index().head(100), hide_index=True, width="stretch",
                 column_config={"signup_span_h": st.column_config.NumberColumn("accounts created within (h)", format="%.1f"),
                                "cashback_inr": st.column_config.NumberColumn("cashback taken (₹)", format="₹%d")})
    st.divider()
    st.subheader("Mule-account candidates")
    st.caption("Accounts receiving money from many unrelated people, then passing it on.")
    mp = st.slider("Minimum distinct payers", 3, 30, 8)
    inbound = (TX[(TX.payee_type == "USER") & TX.ok].groupby("payee_id")
                 .agg(unique_payers=("user_id", "nunique"), inflow=("amount", "sum"), fraud_inbound=("is_fraud", "sum")))
    outbound = TX[TX.txn_type.isin(["P2P", "COLLECT"]) & TX.ok].groupby("user_id").amount.sum()
    mules = inbound[inbound.unique_payers >= mp].assign(outflow=lambda d: d.index.map(outbound).fillna(0))
    mules["pass_through"] = mules.outflow / mules.inflow
    mules = mules.join(users.set_index("user_id")[["signup_ts", "city", "kyc_status"]]).sort_values("unique_payers", ascending=False)
    st.dataframe(mules.reset_index().rename(columns={"payee_id": "account"}).head(100), hide_index=True, width="stretch",
                 column_config={"inflow": st.column_config.NumberColumn("inflow (₹)", format="₹%d"),
                                "outflow": st.column_config.NumberColumn("outflow (₹)", format="₹%d"),
                                "pass_through": st.column_config.NumberColumn(format="%.2f")})

# --------------------------------------------------------------------------------------------------- #
# 6 · Fraud model lab
# --------------------------------------------------------------------------------------------------- #
with tabs[5]:
    st.subheader("Fraud model lab: tune the operating point")
    fm = META["fraud"]
    st.caption(f"Model trained on {fm['train_period']}, scored on unseen {fm['test_period']}. Blend of logistic regression + "
               f"shallow gradient boosting, chosen by rolling time-series CV. The production threshold "
               f"({fm['threshold']:.3f}) was fixed on April validation data. Moving the slider here is exploration on the test period.")
    T = TX[TX.ts >= "2026-05-01"].merge(SCORES, on="txn_id")
    T = apply_tx_filters(T, use_dates=False)
    if T.is_fraud.sum() == 0:
        st.info("No fraud in the test period for these filters.")
    else:
        c1, c2 = st.columns(2)
        thr = c1.slider("Risk-score threshold", 0.50, 0.999, float(fm["threshold"]), step=0.001, format="%.3f")
        fp_cost = c2.slider("Friction cost per legit payment flagged (₹)", 0, 500, int(fm["fp_cost_inr"]), step=10)
        y, s, amt = T.is_fraud.values, T.score.values, T.amount.values
        rule = (((T.is_new_device == 1) & (T.amount > 3000)) | ((T.amount > 15000) & ((T.hour >= 23) | (T.hour < 5)))
                | (T.amount > 40000)).values

        def perf(flag):
            tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
            saved = amt[flag & (y == 1)].sum()
            return {"frauds caught": tp, "legit flagged": fp, "precision": tp / max(tp + fp, 1), "recall": tp / y.sum(),
                    "fraud value stopped (₹)": saved, "net savings (₹)": saved - fp * fp_cost}

        flag = s >= thr
        pm, pr_ = perf(flag), perf(rule)
        k = st.columns(5)
        k[0].metric("Frauds caught", f"{pm['frauds caught']:,} / {int(y.sum()):,}", f"{pm['frauds caught'] - pr_['frauds caught']:+,} vs rules")
        k[1].metric("Legit payments flagged", f"{pm['legit flagged']:,}", f"{pm['legit flagged'] - pr_['legit flagged']:+,} vs rules", delta_color="inverse")
        k[2].metric("Precision", f"{pm['precision']:.0%}", f"{(pm['precision'] - pr_['precision']) * 100:+.0f} pts vs rules")
        k[3].metric("Recall", f"{pm['recall']:.0%}", f"{(pm['recall'] - pr_['recall']) * 100:+.0f} pts vs rules")
        k[4].metric("Net savings", inr(pm["net savings (₹)"]), f"{(pm['net savings (₹)'] - pr_['net savings (₹)']) / 1e5:+.1f} L vs rules")

        c1, c2 = st.columns(2)
        with c1:
            f = go.Figure()
            for col, name, colr in [("score_logit", "Logistic regression", GREY), ("score_gbm", "Gradient boosting", AQUA), ("score", "Blend (selected)", BLUE)]:
                p_, r_, _ = precision_recall_curve(y, T[col].values)
                f.add_scatter(x=r_, y=p_, mode="lines", name=f"{name} · PR-AUC {average_precision_score(y, T[col].values):.2f}",
                              line=dict(color=colr, width=2), hovertemplate="recall %{x:.0%}<br>precision %{y:.0%}<extra></extra>")
            f.add_scatter(x=[pr_["recall"]], y=[pr_["precision"]], mode="markers", name="Legacy rules",
                          marker=dict(color=ORANGE, size=12), hovertemplate="rules: recall %{x:.0%}, precision %{y:.0%}<extra></extra>")
            f.add_scatter(x=[pm["recall"]], y=[pm["precision"]], mode="markers", name="Your threshold",
                          marker=dict(color=BLUE, size=14, symbol="diamond", line=dict(color="white", width=2)),
                          hovertemplate="threshold: recall %{x:.0%}, precision %{y:.0%}<extra></extra>")
            f.update_layout(title="Precision–recall on May–Jun", xaxis_title="recall (share of fraud caught)", yaxis_title="precision")
            f.update_xaxes(range=[0, 1.02]); f.update_yaxes(range=[0, 1.02])
            style(f, 420, legend=True)
            f.update_layout(legend=dict(orientation="v", x=0.01, y=0.02, xanchor="left", yanchor="bottom",
                                        bgcolor="rgba(252,252,251,0.85)"), margin=dict(t=40))
            show(f, key="lab_pr")
        with c2:
            grid = np.linspace(0.5, 0.999, 120)
            nets = [perf(s >= t)["net savings (₹)"] / 1e5 for t in grid]
            f = px.line(x=grid, y=nets, title=f"Net savings vs threshold (friction ₹{fp_cost}/flag)",
                        labels={"x": "threshold", "y": "net savings (₹ lakh)"}, color_discrete_sequence=[BLUE])
            f.add_vline(x=thr, line_color=BLUE, annotation_text="your threshold")
            f.add_vline(x=fm["threshold"], line_dash="dash", line_color="#52514e", annotation_text="validation pick",
                        annotation_position="bottom left")
            f.add_hline(y=pr_["net savings (₹)"] / 1e5, line_dash="dot", line_color=ORANGE, annotation_text="rules")
            show(style(f, 420), key="lab_net")

        c1, c2 = st.columns([1, 1.3])
        with c1:
            fmt = lambda k, v: f"{v:.0%}" if k in ("precision", "recall") else (f"₹{v:,.0f}" if "₹" in k else f"{v:,}")
            cmp = pd.DataFrame({"metric": list(pm), "Legacy rules": [fmt(k, v) for k, v in pr_.items()],
                                f"Model @ {thr:.3f}": [fmt(k, v) for k, v in pm.items()]})
            st.markdown("**Rules vs model**")
            st.dataframe(cmp, hide_index=True, width="stretch")
            sc = T[T.is_fraud == 1].merge(SCHEMES, on="txn_id").assign(model=lambda d: d.score >= thr)
            sc["rules"] = sc.txn_id.isin(T.txn_id[rule])
            rs = sc.groupby("fraud_scheme")[["model", "rules"]].mean().mul(100).reset_index().melt("fraud_scheme", var_name="who", value_name="recall")
            f = px.bar(rs, x="fraud_scheme", y="recall", color="who", barmode="group", title="Recall by fraud scheme (%)",
                       color_discrete_map={"model": BLUE, "rules": ORANGE}, labels={"fraud_scheme": "", "recall": "% caught"}, text_auto=".0f")
            show(style(f, 320, legend=True), key="lab_scheme")
        with c2:
            st.markdown("**Payments the model would flag** (highest risk first)")
            fl = T[flag].sort_values("score", ascending=False)[["txn_id", "ts", "txn_type", "amount", "payer_bank", "is_new_device", "score", "is_fraud"]]
            st.dataframe(fl.head(300), hide_index=True, width="stretch", height=420,
                         column_config={"score": st.column_config.ProgressColumn("risk score", min_value=0.5, max_value=1, format="%.3f"),
                                        "amount": st.column_config.NumberColumn(format="₹%d"),
                                        "is_fraud": st.column_config.CheckboxColumn("confirmed fraud")})
            st.download_button("Download flagged payments (CSV)", fl.to_csv(index=False), "flagged_payments.csv", "text/csv")
