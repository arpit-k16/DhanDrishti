"""
UPI Pulse — synthetic data generator
====================================
Simulates 6 months (Jan–Jun 2026) of activity on a fictional UPI payments app.

The data is synthetic, but the patterns built into it come from publicly
reported pain points of Indian UPI apps:

  1. Technical failures concentrated in a few remitter banks and at peak hours
  2. "Debited but failed" transactions whose refunds break the T+1 reversal TAT
  3. Fraud: collect-request scams, account takeover (new device), fake-merchant QR
  4. Cashback abuse: device farms creating many accounts to farm sign-up rewards
  5. Churn: cashback-acquired users and users hit by early failures leave faster
  6. A rule-based risk engine that blocks legit users yet misses most fraud

Run:  python data/generate_data.py            (writes CSVs to data/raw/)
"""
import os
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

START = pd.Timestamp("2026-01-01")
END = pd.Timestamp("2026-06-30 23:59:59")
N_USERS = 12000
N_MERCHANTS = 2000
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
TRUTH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth")
os.makedirs(OUT, exist_ok=True)
os.makedirs(TRUTH, exist_ok=True)

HOUR = np.timedelta64(1, "h")
DAY = np.timedelta64(1, "D")

# --------------------------------------------------------------------------- #
# Reference tables
# --------------------------------------------------------------------------- #
banks = pd.DataFrame(
    [
        # code, name, user share, base tech-failure prob, extra failure at peak (19-22h)
        ("NSB", "Northstar Bank",            0.22, 0.010, 0.002, False),
        ("UNB", "Unity National Bank",       0.20, 0.013, 0.004, False),
        ("SFB", "Saffron Bank",              0.15, 0.015, 0.004, False),
        ("MRB", "Meridian Bank",             0.13, 0.016, 0.050, True),   # collapses at peak
        ("HBR", "Harbor Bank",               0.10, 0.014, 0.006, False),
        ("CCB", "Coastal Co-operative Bank", 0.08, 0.052, 0.020, True),   # chronically bad
        ("PNC", "Pinnacle Bank",             0.07, 0.015, 0.004, False),
        ("GRB", "Granite Rural Bank",        0.05, 0.034, 0.012, True),
    ],
    columns=["bank_code", "bank_name", "user_share", "base_tech_fail", "peak_extra", "slow_refunds"],
)
BANK_CODES = banks.bank_code.values
B = banks.set_index("bank_code")

cities = pd.DataFrame(
    [
        ("Delhi", 1), ("Mumbai", 1), ("Bengaluru", 1), ("Hyderabad", 1), ("Chennai", 1), ("Pune", 1),
        ("Jaipur", 2), ("Lucknow", 2), ("Indore", 2), ("Bhopal", 2), ("Patna", 2), ("Nagpur", 2), ("Kochi", 2),
        ("Ghaziabad", 2), ("Varanasi", 3), ("Ranchi", 3), ("Dehradun", 3), ("Siliguri", 3), ("Bareilly", 3),
        ("Guntur", 3), ("Muzaffarpur", 3),
    ],
    columns=["city", "tier"],
)
city_w = np.where(cities.tier == 1, 3.0, np.where(cities.tier == 2, 1.6, 1.0))
city_w = city_w / city_w.sum()

campaigns = pd.DataFrame(
    [
        ("CMP01", "NEWUSER50", "2026-01-15", "2026-02-15", "Flat Rs 50 on first merchant payment >= Rs 100", 50, 1500000),
        ("CMP02", "SUMMER100", "2026-04-01", "2026-04-30", "Up to Rs 100 on first merchant payment >= Rs 100", 100, 2500000),
    ],
    columns=["campaign_id", "code", "start_date", "end_date", "rule", "max_reward", "budget_inr"],
)
CAMP_WIN = [(pd.Timestamp(r.start_date), pd.Timestamp(r.end_date) + pd.Timedelta(hours=23, minutes=59), r.campaign_id)
            for r in campaigns.itertuples()]

# --------------------------------------------------------------------------- #
# Merchants
# --------------------------------------------------------------------------- #
cat_list = ["Grocery", "Food & Dining", "Fuel", "Pharmacy", "Fashion", "Electronics", "Travel", "Kirana",
            "Utilities", "Telecom", "Education", "Entertainment"]
cat_w = np.array([14, 14, 7, 6, 7, 4, 4, 22, 5, 5, 4, 8], float)
cat_w /= cat_w.sum()
m_ids = [f"M{i:05d}" for i in range(1, N_MERCHANTS + 1)]
merchants = pd.DataFrame({
    "merchant_id": m_ids,
    "category": rng.choice(cat_list, N_MERCHANTS, p=cat_w),
    "city": rng.choice(cities.city, N_MERCHANTS, p=city_w),
    "settlement_bank": rng.choice(BANK_CODES, N_MERCHANTS, p=banks.user_share),
    "onboard_date": (pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 1400, N_MERCHANTS), unit="D")).normalize(),
    "kyc_tier": rng.choice(["FULL", "SMALL"], N_MERCHANTS, p=[0.55, 0.45]),
})
# fake merchants used in QR scams: recently onboarded, small KYC
fake_idx = rng.choice(N_MERCHANTS, 15, replace=False)
merchants.loc[fake_idx, "category"] = rng.choice(["Electronics", "Fashion", "Travel"], 15)
merchants.loc[fake_idx, "kyc_tier"] = "SMALL"
merchants.loc[fake_idx, "onboard_date"] = (pd.Timestamp("2025-12-01") + pd.to_timedelta(rng.integers(0, 40, 15), unit="D")).normalize()
# colluding merchants used by cashback farms
coll_pool = np.setdiff1d(np.arange(N_MERCHANTS), fake_idx)
coll_idx = rng.choice(coll_pool, 10, replace=False)
merchants.loc[coll_idx, "category"] = "Kirana"
merchants.loc[coll_idx, "kyc_tier"] = "SMALL"
FAKE_M = merchants.merchant_id.values[fake_idx]
COLL_M = merchants.merchant_id.values[coll_idx]

normal_m_mask = ~merchants.merchant_id.isin(np.concatenate([FAKE_M, COLL_M]))
retail_m = merchants[normal_m_mask & ~merchants.category.isin(["Utilities", "Telecom"])]
biller_m = merchants[normal_m_mask & merchants.category.isin(["Utilities", "Telecom"])]
retail_ids = retail_m.merchant_id.values
retail_w = 1.0 / np.arange(1, len(retail_ids) + 1) ** 0.9  # Zipf-ish popularity
retail_w /= retail_w.sum()
biller_ids = biller_m.merchant_id.values
M_BANK = merchants.set_index("merchant_id").settlement_bank.to_dict()

# --------------------------------------------------------------------------- #
# Users (normal + mules + cashback-farm accounts)
# --------------------------------------------------------------------------- #
def rand_ts(start, end, n):
    s, e = start.value, end.value
    return pd.to_datetime(rng.integers(s, e, n))

age_bands = ["18-24", "25-34", "35-44", "45-54", "55+"]
age_w = [0.24, 0.36, 0.22, 0.11, 0.07]

ch = rng.choice(["organic", "referral", "cashback_campaign"], N_USERS, p=[0.45, 0.20, 0.35])
signup = np.array(rand_ts(START, pd.Timestamp("2026-05-31 23:00"), N_USERS), dtype=object)
# 75% of campaign users sign up inside a campaign window
camp_id = np.array([None] * N_USERS, dtype=object)
for i in np.where(ch == "cashback_campaign")[0]:
    if rng.random() < 0.75:
        s, e, cid = CAMP_WIN[rng.integers(0, 2)]
        signup[i] = rand_ts(s, e, 1)[0]
        camp_id[i] = cid
    else:
        for s, e, cid in CAMP_WIN:
            if s <= signup[i] <= e:
                camp_id[i] = cid
        if camp_id[i] is None:
            ch[i] = "organic"  # signed up outside any campaign -> not campaign-acquired

city_idx = rng.choice(len(cities), N_USERS, p=city_w)
users = pd.DataFrame({
    "user_id": [f"U{i:06d}" for i in range(1, N_USERS + 1)],
    "signup_ts": pd.to_datetime(signup),
    "city": cities.city.values[city_idx],
    "city_tier": cities.tier.values[city_idx],
    "age_band": rng.choice(age_bands, N_USERS, p=age_w),
    "acquisition_channel": ch,
    "campaign_id": camp_id,
    "primary_bank": rng.choice(BANK_CODES, N_USERS, p=banks.user_share),
    "device_id": [f"D{x:07d}" for x in rng.choice(9_000_000, N_USERS, replace=False) + 1_000_000],
    "kyc_status": rng.choice(["FULL", "MIN"], N_USERS, p=[0.72, 0.28]),
})
users["role"] = "normal"

# mule accounts: newish, full of inbound transfers from strangers
N_MULE = 60
mules = pd.DataFrame({
    "user_id": [f"U{i:06d}" for i in range(N_USERS + 1, N_USERS + N_MULE + 1)],
    "signup_ts": rand_ts(START, pd.Timestamp("2026-05-15"), N_MULE),
    "city": rng.choice(["Jamtara", "Nuh", "Bharatpur", "Deoghar"], N_MULE),
    "city_tier": 3,
    "age_band": rng.choice(["18-24", "25-34"], N_MULE),
    "acquisition_channel": rng.choice(["organic", "referral"], N_MULE),
    "campaign_id": None,
    "primary_bank": rng.choice(BANK_CODES, N_MULE, p=banks.user_share),
    "device_id": [f"D{x:07d}" for x in rng.choice(900_000, N_MULE, replace=False) + 100_000],
    "kyc_status": rng.choice(["FULL", "MIN"], N_MULE, p=[0.3, 0.7]),
    "role": "mule",
})

# cashback farms: one device -> many accounts, burst sign-ups inside campaign windows
farm_rows = []
uid = N_USERS + N_MULE
for f in range(42):
    s, e, cid = CAMP_WIN[rng.integers(0, 2)]
    day0 = rand_ts(s, e - pd.Timedelta(days=2), 1)[0]
    dev = f"D{rng.integers(10_000, 99_999):07d}"
    k = rng.integers(5, 12)
    city = cities.city.values[rng.integers(0, len(cities))]
    for j in range(k):
        uid += 1
        farm_rows.append({
            "user_id": f"U{uid:06d}", "signup_ts": day0 + pd.Timedelta(minutes=int(rng.integers(0, 600))),
            "city": city, "city_tier": int(cities.set_index("city").tier[city]),
            "age_band": rng.choice(["18-24", "25-34"]), "acquisition_channel": "cashback_campaign",
            "campaign_id": cid, "primary_bank": rng.choice(BANK_CODES, p=banks.user_share),
            "device_id": dev, "kyc_status": "MIN", "role": f"farm_{f:02d}",
        })
farms = pd.DataFrame(farm_rows)
users = pd.concat([users, mules, farms], ignore_index=True)
USER_BANK = users.set_index("user_id").primary_bank.to_dict()
USER_SIGNUP = users.set_index("user_id").signup_ts.to_dict()
normal_ids = users.loc[users.role == "normal", "user_id"].values

# contact graph for P2P
contacts = {u: rng.choice(normal_ids, rng.integers(4, 15), replace=False) for u in normal_ids}

# --------------------------------------------------------------------------- #
# Transaction simulation
# --------------------------------------------------------------------------- #
hour_w = np.array([0.6, 0.3, 0.2, 0.15, 0.15, 0.3, 0.8, 1.6, 2.6, 3.2, 3.6, 4.0,
                   4.6, 4.8, 4.2, 3.8, 3.8, 4.2, 5.0, 5.8, 6.4, 6.0, 4.4, 2.2])
hour_w /= hour_w.sum()
TYPES = np.array(["P2M", "P2P", "BILL", "RECHARGE", "COLLECT"])
TYPE_W = np.array([0.46, 0.30, 0.09, 0.10, 0.05])
RECH = np.array([149, 199, 239, 299, 479, 719, 2999])

T = {k: [] for k in ["txn_id", "ts", "user_id", "txn_type", "amount", "payee_type", "payee_id",
                     "payer_bank", "payee_bank", "device_id", "is_new_device", "status",
                     "failure_reason", "latency_ms", "cashback_inr", "is_fraud", "fraud_scheme"]}
REF = []      # refunds for debited-but-failed
TICK = []     # support tickets
txn_counter = [0]


def next_id():
    txn_counter[0] += 1
    return f"T{txn_counter[0]:08d}"


def tech_fail_prob(payer_bank, payee_bank, hours, days, ptype):
    p = np.full(len(hours), B.base_tech_fail[payer_bank])
    peak = (hours >= 19) & (hours <= 22)
    p = p + peak * B.peak_extra[payer_bank]
    p = p + (days <= 3) * 0.008                       # salary-day load at month start
    pb = np.array([B.base_tech_fail[b] for b in payee_bank])
    p = p + 0.35 * pb * (ptype != "MERCHANT")          # beneficiary-bank side failures
    return np.clip(p, 0, 0.5)


def refund_tat_hours(bank):
    if B.slow_refunds[bank] and rng.random() < 0.30:
        return float(rng.lognormal(np.log(70), 0.6))  # long tail: days
    return float(rng.lognormal(np.log(5), 0.8))


def add_ticket(user, txn, created, category, resolve_h, first_resp_h=None):
    if first_resp_h is None:
        first_resp_h = float(rng.lognormal(np.log(3), 0.9))
    resolved = created + pd.Timedelta(hours=resolve_h)
    csat = None
    if rng.random() < 0.65:
        csat = int(np.clip(round(4.6 - 0.025 * resolve_h + rng.normal(0, 0.7)), 1, 5))
    TICK.append({"user_id": user, "txn_id": txn, "created_ts": created, "category": category,
                 "first_response_hours": round(first_resp_h, 1),
                 "resolved_ts": resolved if resolved <= END + pd.Timedelta(days=3) else pd.NaT,
                 "csat": csat})


def legacy_rule(new_dev, amt, hr):
    """The app's current static risk rules (what the ML model has to beat)."""
    return (new_dev and amt > 3000) or (amt > 15000 and (hr >= 23 or hr < 5)) or amt > 40000


def emit(user, ts, ttype, amount, payee_type, payee_id, payer_bank, payee_bank, device, new_dev,
         status, reason, latency, cashback=0.0, fraud=0, scheme=None):
    tid = next_id()
    for k, v in zip(T.keys(), [tid, ts, user, ttype, amount, payee_type, payee_id, payer_bank, payee_bank,
                               device, new_dev, status, reason, latency, cashback, fraud, scheme]):
        T[k].append(v)
    return tid


def settle(user, tid, ts, amount, payer_bank, reason, lat):
    """For a technical failure decide whether money was debited and simulate the refund."""
    debit_p = 0.55 if reason == "TIMEOUT" else 0.25 if reason == "BANK_SERVER_DOWN" else 0.15
    if rng.random() >= debit_p:
        return 0.0
    tat = refund_tat_hours(payer_bank)
    refund_ts = ts + pd.Timedelta(hours=tat)
    REF.append({"txn_id": tid, "user_id": user, "amount": amount, "debit_ts": ts,
                "refund_ts": refund_ts if refund_ts <= END else pd.NaT,
                "status": "REVERSED" if refund_ts <= END else "PENDING"})
    if rng.random() < (0.45 + (0.40 if tat > 24 else 0)):
        created = ts + pd.Timedelta(hours=float(rng.uniform(0.3, min(tat, 30) + 0.5)))
        add_ticket(user, tid, created, "REFUND_NOT_RECEIVED",
                   max(tat - (created - ts).total_seconds() / 3600, 0) + float(rng.uniform(1, 18)))
    return tat


def sim_normal_user(u):
    signup_ts = u.signup_ts
    lam = float(np.exp(rng.normal(0.6, 0.8)))
    hazard = {"organic": 0.040, "referral": 0.035, "cashback_campaign": 0.11}[u.acquisition_channel]
    if u.acquisition_channel == "cashback_campaign":
        lam *= 0.7
    bank, dev = u.primary_bank, u.device_id
    early_fail_hit, refund_hit, cashback_given = False, False, False
    week_start = signup_ts
    week = 0
    while week_start < END:
        n = rng.poisson(lam * (1.5 if week == 0 else 1.0))
        if week == 0:
            n = max(n, 1)
        if n:
            offs = rng.uniform(0, 7 * 24, n) if week else np.sort(rng.uniform(0, 72, n))
            base = (week_start + pd.to_timedelta(offs, unit="h")).normalize()
            hrs = rng.choice(24, n, p=hour_w)
            ts = base + pd.to_timedelta(hrs, unit="h") + pd.to_timedelta(rng.integers(0, 3600, n), unit="s")
            early = ts < signup_ts   # first-week payments can't precede signup: spread them over the first 3h instead
            if early.any():
                ts = ts.where(~early, signup_ts + pd.to_timedelta(rng.uniform(300, 3 * 3600, n), unit="s").floor("s"))
            keep = ts <= END
            ts, hrs = ts[keep], hrs[keep]
            n = len(ts)
            if n == 0:
                break
            types = rng.choice(TYPES, n, p=TYPE_W)
            if week == 0 and u.acquisition_channel == "cashback_campaign":
                types[0] = "P2M"
            amt = np.empty(n)
            payee_type, payee_id, payee_bank = [], [], []
            for i, t in enumerate(types):
                if t == "P2M":
                    amt[i] = round(np.clip(rng.lognormal(5.3, 1.0), 10, 60000))
                    m = rng.choice(retail_ids, p=retail_w)
                    payee_type.append("MERCHANT"); payee_id.append(m); payee_bank.append(M_BANK[m])
                elif t in ("BILL", "RECHARGE"):
                    amt[i] = rng.choice(RECH) if t == "RECHARGE" else round(np.clip(rng.lognormal(7.0, 0.7), 100, 40000))
                    m = rng.choice(biller_ids)
                    payee_type.append("MERCHANT"); payee_id.append(m); payee_bank.append(M_BANK[m])
                else:
                    amt[i] = round(np.clip(rng.lognormal(6.5 if t == "P2P" else 6.3, 1.2), 1, 100000))
                    if amt[i] > 300 and rng.random() < 0.35:
                        amt[i] = max(100, round(amt[i], -2))   # rent, splits, transfers to family
                    c = rng.choice(contacts[u.user_id]) if rng.random() < 0.85 else rng.choice(normal_ids)
                    if USER_SIGNUP[c] > ts[i]:   # payee must already be on the app
                        c = next((x for x in rng.permutation(normal_ids)[:50] if USER_SIGNUP[x] <= ts[i]), u.user_id)
                        if c == u.user_id:
                            types[i] = "P2M"
                            m = rng.choice(retail_ids, p=retail_w)
                            payee_type.append("MERCHANT"); payee_id.append(m); payee_bank.append(M_BANK[m])
                            continue
                    payee_type.append("USER"); payee_id.append(c); payee_bank.append(USER_BANK[c])
            payee_type = np.array(payee_type)
            p_tech = tech_fail_prob(bank, payee_bank, hrs, ts.day.values, payee_type)
            r = rng.random(n)
            p_funds = 0.008 + 0.035 * (amt > 5000)
            p_pin = 0.012
            for i in range(n):
                status, reason = "SUCCESS", None
                if r[i] < p_tech[i]:
                    status = "FAILED"
                    reason = rng.choice(["BANK_SERVER_DOWN", "TIMEOUT", "NETWORK_ERROR"], p=[0.45, 0.40, 0.15])
                elif r[i] < p_tech[i] + p_funds[i]:
                    status, reason = "FAILED", "INSUFFICIENT_FUNDS"
                elif r[i] < p_tech[i] + p_funds[i] + p_pin:
                    status, reason = "FAILED", "WRONG_PIN"
                new_dev = int(rng.random() < 0.03)   # reinstall / phone change
                # legacy rule engine: blocks big transfers from new device / at night -> false positives
                if status == "SUCCESS" and legacy_rule(new_dev, amt[i], hrs[i]) and rng.random() < 0.7:
                    status, reason = "FAILED", "RISK_BLOCKED"
                if reason == "TIMEOUT":
                    lat = int(rng.uniform(30000, 45000))
                else:
                    lat = int(rng.lognormal(np.log(1800 + 9000 * B.base_tech_fail[bank]), 0.45))
                cb = 0.0
                if (status == "SUCCESS" and not cashback_given and u.campaign_id is not None
                        and types[i] == "P2M" and amt[i] >= 100):
                    for s, e, cid in CAMP_WIN:
                        if cid == u.campaign_id and s <= ts[i] <= e + pd.Timedelta(days=7):
                            cb = 50.0 if cid == "CMP01" else float(rng.integers(10, 101))
                            cashback_given = True
                tid = emit(u.user_id, ts[i], types[i], float(amt[i]), payee_type[i], payee_id[i], bank,
                           payee_bank[i], dev, new_dev, status, reason, lat, cb)
                if reason in ("BANK_SERVER_DOWN", "TIMEOUT", "NETWORK_ERROR"):
                    tat = settle(u.user_id, tid, ts[i], float(amt[i]), bank, reason, lat)
                    # ~55% of users retry within minutes; the rest abandon (lost GMV)
                    if rng.random() < 0.55:
                        rts = ts[i] + pd.Timedelta(minutes=float(rng.uniform(0.5, 12)))
                        ok = rng.random() > p_tech[i] * 1.5
                        emit(u.user_id, rts, types[i], float(amt[i]), payee_type[i], payee_id[i], bank,
                             payee_bank[i], dev, 0, "SUCCESS" if ok else "FAILED",
                             None if ok else "BANK_SERVER_DOWN", int(rng.lognormal(np.log(2200), 0.45)))
                    if week <= 1 and not early_fail_hit:
                        early_fail_hit = True
                        hazard *= 1.7
                    if tat > 24 and not refund_hit:
                        refund_hit = True
                        hazard *= 1.8
                elif reason == "RISK_BLOCKED" and rng.random() < 0.5:
                    add_ticket(u.user_id, tid, ts[i] + pd.Timedelta(hours=float(rng.uniform(0.2, 6))),
                               "PAYMENT_BLOCKED", float(rng.lognormal(np.log(30), 0.7)))
                elif status == "SUCCESS" and rng.random() < 0.003:
                    cat = rng.choice(["CASHBACK_NOT_RECEIVED", "KYC_ISSUE", "APP_BUG", "OTHER"], p=[0.35, 0.2, 0.25, 0.2])
                    add_ticket(u.user_id, tid, ts[i] + pd.Timedelta(hours=float(rng.uniform(0.2, 48))), cat,
                               float(rng.lognormal(np.log(20), 0.9)))
        if rng.random() < hazard:
            break
        hazard = max(hazard * 0.94, 0.012)
        week_start += pd.Timedelta(days=7)
        week += 1


print("Simulating normal users ...")
for k, u in enumerate(users[users.role == "normal"].itertuples(index=False)):
    sim_normal_user(u)
    if k % 2000 == 0:
        print(f"  {k:>6} users, {txn_counter[0]:>8} txns")

# --------------------------------------------------------------------------- #
# Cashback farms: 1 qualifying payment per account, then sweep balance to master
# --------------------------------------------------------------------------- #
print("Injecting cashback farms ...")
for role, g in farms.groupby("role"):
    master = g.user_id.iloc[0]
    for u in g.itertuples(index=False):
        ts = u.signup_ts + pd.Timedelta(minutes=int(rng.integers(3, 240)))
        m = rng.choice(COLL_M)
        amt = float(rng.integers(100, 131))
        cb = 50.0 if u.campaign_id == "CMP01" else float(rng.integers(60, 101))
        emit(u.user_id, ts, "P2M", amt, "MERCHANT", m, u.primary_bank, M_BANK[m], u.device_id, 0,
             "SUCCESS", None, int(rng.lognormal(np.log(1900), 0.4)), cb)
        if u.user_id != master and rng.random() < 0.8:
            emit(u.user_id, ts + pd.Timedelta(hours=float(rng.uniform(2, 30))), "P2P", float(cb + rng.integers(0, 20)),
                 "USER", master, u.primary_bank, USER_BANK[master], u.device_id, 0, "SUCCESS", None,
                 int(rng.lognormal(np.log(1900), 0.4)))

# --------------------------------------------------------------------------- #
# Fraud: collect-request scams, account takeover, fake-merchant QR
# --------------------------------------------------------------------------- #
print("Injecting fraud ...")
u_norm = users[(users.role == "normal") & (users.signup_ts < pd.Timestamp("2026-06-10"))].copy()
w = np.where(u_norm.age_band.isin(["45-54", "55+"]), 2.8, 1.0) * np.where(u_norm.city_tier > 1, 1.3, 1.0)
victims = u_norm.sample(480, weights=w, random_state=SEED)
mule_ids = mules.user_id.values
for v in victims.itertuples(index=False):
    scheme = rng.choice(["COLLECT_SCAM", "ACCOUNT_TAKEOVER", "FAKE_QR"], p=[0.45, 0.35, 0.20])
    t0 = rand_ts(max(v.signup_ts + pd.Timedelta(days=5), START), END - pd.Timedelta(hours=6), 1)[0]
    ok_mules = mules.user_id[mules.signup_ts < t0 - pd.Timedelta(hours=12)].values
    if len(ok_mules) == 0:
        t0 = mules.signup_ts.min() + pd.Timedelta(days=int(rng.integers(1, 30)))
        ok_mules = mules.user_id[mules.signup_ts < t0].values
    mule = rng.choice(ok_mules)
    k = int(rng.choice([1, 2, 3], p=[0.55, 0.3, 0.15]))
    first_tid = None
    h_collect = int(rng.integers(10, 21))
    h_ato = int(rng.choice([23, 0, 1, 2, 3])) if rng.random() < 0.6 else int(rng.integers(8, 23))
    h_qr = int(rng.integers(9, 22))
    for j in range(k):
        if scheme == "COLLECT_SCAM":
            ts = t0.normalize() + pd.Timedelta(hours=h_collect, minutes=12 * j + int(rng.integers(0, 10)))
            ttype, ptype, pid, pbank = "COLLECT", "USER", mule, USER_BANK[mule]
            amt, new_dev, dev = float(round(rng.lognormal(8.3, 0.7))), 0, v.device_id
            if rng.random() < 0.6:
                amt = float(max(500, round(amt / 500) * 500))   # "pay Rs 5,000 to receive your refund"
        elif scheme == "ACCOUNT_TAKEOVER":
            ts = t0.normalize() + pd.Timedelta(hours=h_ato, minutes=4 * j + int(rng.integers(0, 4)))
            ttype, ptype, pid, pbank = "P2P", "USER", mule, USER_BANK[mule]
            amt = float(round(rng.lognormal(8.7, 0.6)))
            new_dev = int(rng.random() < 0.85)
            dev = f"D{rng.integers(100, 99999):07d}" if new_dev else v.device_id
        else:
            ts = t0.normalize() + pd.Timedelta(hours=h_qr, minutes=7 * j + int(rng.integers(0, 6)))
            m = rng.choice(FAKE_M)
            ttype, ptype, pid, pbank = "P2M", "MERCHANT", m, M_BANK[m]
            amt, new_dev, dev = float(round(rng.lognormal(8.0, 0.8))), 0, v.device_id
        amt = float(np.clip(amt, 200, 100000))
        status, reason = "SUCCESS", None
        if legacy_rule(new_dev, amt, ts.hour) and rng.random() < 0.7:
            status, reason = "FAILED", "RISK_BLOCKED"   # the old rules do catch a few
        tid = emit(v.user_id, ts, ttype, amt, ptype, pid, v.primary_bank, pbank, dev, new_dev, status, reason,
                   int(rng.lognormal(np.log(2000), 0.4)), 0.0, 1, scheme)
        first_tid = first_tid or tid
    if rng.random() < 0.65:
        add_ticket(v.user_id, first_tid, t0 + pd.Timedelta(hours=float(rng.uniform(1, 72))), "FRAUD_REPORT",
                   float(rng.lognormal(np.log(140), 0.6)), float(rng.lognormal(np.log(10), 0.8)))

# mules cash out: frequent outbound transfers between mules
for m in mule_ids:
    for _ in range(int(rng.integers(3, 15))):
        dst = rng.choice(mule_ids)
        lo = max(USER_SIGNUP[m], USER_SIGNUP[dst]) + pd.Timedelta(days=2)
        ts = rand_ts(lo, END, 1)[0]
        emit(m, ts, "P2P", float(round(rng.lognormal(9.0, 0.5))), "USER", dst, USER_BANK[m], USER_BANK[dst],
             users.set_index("user_id").device_id[m], 0, "SUCCESS", None, int(rng.lognormal(np.log(1900), 0.4)))

# --------------------------------------------------------------------------- #
# Write out
# --------------------------------------------------------------------------- #
tx = pd.DataFrame(T).sort_values("ts").reset_index(drop=True)
tx["ts"] = pd.to_datetime(tx["ts"]).dt.floor("s")

ref = pd.DataFrame(REF)
ref.insert(0, "refund_id", [f"R{i:06d}" for i in range(1, len(ref) + 1)])
ref["tat_hours"] = ((ref.refund_ts.fillna(END) - ref.debit_ts).dt.total_seconds() / 3600).round(1)
# RBI TAT framework: auto-reversal by T+1, Rs 100/day compensation for each day of delay beyond that
ref["days_beyond_tat"] = np.maximum(np.ceil((ref.tat_hours - 24) / 24), 0).astype(int)
ref["compensation_due_inr"] = ref.days_beyond_tat * 100

tk = pd.DataFrame(TICK).sort_values("created_ts").reset_index(drop=True)
tk.insert(0, "ticket_id", [f"TK{i:06d}" for i in range(1, len(tk) + 1)])
tk["status"] = np.where(tk.resolved_ts.isna(), "OPEN", "RESOLVED")

truth = users[["user_id", "role"]].copy()
users_out = users.drop(columns=["role"])
users_out["signup_ts"] = users_out.signup_ts.dt.floor("s")

users_out.to_csv(f"{OUT}/users.csv", index=False)
merchants.to_csv(f"{OUT}/merchants.csv", index=False)
banks.drop(columns=["user_share", "base_tech_fail", "peak_extra", "slow_refunds"]).to_csv(f"{OUT}/banks.csv", index=False)
campaigns.to_csv(f"{OUT}/campaigns.csv", index=False)
tx.drop(columns=["fraud_scheme"]).to_csv(f"{OUT}/transactions.csv", index=False)
ref.to_csv(f"{OUT}/refunds.csv", index=False)
tk.to_csv(f"{OUT}/support_tickets.csv", index=False)
# ground truth kept separate: used only to *validate* the detection work, never as a feature
truth.to_csv(f"{TRUTH}/user_roles.csv", index=False)
tx.loc[tx.is_fraud == 1, ["txn_id", "fraud_scheme"]].to_csv(f"{TRUTH}/fraud_schemes.csv", index=False)

print(f"users={len(users_out):,} merchants={len(merchants):,} txns={len(tx):,} "
      f"refunds={len(ref):,} tickets={len(tk):,} fraud_txns={int(tx.is_fraud.sum()):,}")
