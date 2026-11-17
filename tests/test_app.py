"""Dashboard tests: run app.py headless with Streamlit's AppTest, exercise filters and sliders, and check that
the numbers it shows match the figures verified in README.md.        Run:  python tests/test_app.py   (~1 min)
"""
import os
import sys

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
res = []


def check(name, ok, detail=""):
    res.append((name, "PASS" if ok else "FAIL", detail))


def run():
    return AppTest.from_file(APP, default_timeout=300).run()


def metrics(at):
    return {m.label: m.value for m in at.metric}


# 1) default view: no errors, all tabs, and the headline numbers match the README
at = run()
check("default run: no exceptions", len(at.exception) == 0, "; ".join(e.value[:200] for e in at.exception))
check("6 tabs rendered", len(at.tabs) == 6, str(len(at.tabs)))
check("at least 20 interactive charts", len(at.get("plotly_chart")) >= 20, str(len(at.get("plotly_chart"))))
m = metrics(at)
expected = {
    "GMV (successful)": "₹24.62 Cr", "Payments": "315,822", "Success rate": "95.52%", "Active users": "12,384",
    "Refunds breaching T+1": "14.6%", "Fraud losses": "₹31.8 L", "Recovered by retry < 30 min": "51%",
    "GMV abandoned": "₹31.4 L", "Compensation owed": "₹95,900", "Churn after a failure in first 14 days": "54%",
    "Future churners reached": "75%", "Flagged devices": "42", "Accounts on them": "324",
    "Share of cashback they took": "10.8%", "Rules catch (recall)": "18%", "Rule blocks that were legit": "69%",
    "Frauds caught": "335 / 504", "Legit payments flagged": "367", "Precision": "48%", "Recall": "66%",
}
for k, v in expected.items():
    check(f"shows {k} = {v}", m.get(k) == v, f"got {m.get(k)}")

# 2) every filter set at once
at.sidebar.multiselect[0].select("Meridian Bank").run()
at.sidebar.multiselect[2].select("cashback_campaign").run()
at.sidebar.multiselect[4].select("55+").run()
check("filtered run: no exceptions", len(at.exception) == 0, "; ".join(e.value[:200] for e in at.exception))
check("filters change the KPIs", metrics(at).get("Payments") != "315,822", metrics(at).get("Payments"))

# 3) a filter combination with almost no data must degrade gracefully, not crash
at.sidebar.multiselect[1].select("RECHARGE").run()
at.sidebar.multiselect[3].select(1).run()
check("near-empty filter: no exceptions", len(at.exception) == 0, "; ".join(e.value[:200] for e in at.exception))

# 4) sliders and time grain
at = run()
for label, value in [("Risk-score", 0.9), ("Friction", 300), ("Contact", 10), ("Show refunds", 72),
                     ("Flag devices", 5), ("Minimum distinct", 15)]:
    for s in at.slider:
        if s.label.startswith(label):
            s.set_value(value).run()
for g in ["Daily", "Monthly"]:
    at.radio[0].set_value(g).run()
check("sliders + time grain: no exceptions", len(at.exception) == 0, "; ".join(e.value[:200] for e in at.exception))
m = metrics(at)
check("threshold 0.90 catches more fraud than 0.96", int(m["Frauds caught"].split(" /")[0]) > 335, m.get("Frauds caught"))

for name, r, detail in res:
    print(f"{r}  {name}" + (f"   [{detail}]" if r == "FAIL" and detail else ""))
print(f"\n{sum(r == 'PASS' for _, r, _ in res)}/{len(res)} dashboard tests passed")
sys.exit(0 if all(r == "PASS" for _, r, _ in res) else 1)
