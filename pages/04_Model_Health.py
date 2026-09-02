"""
Model Health — anomaly detection, drift monitoring and campaign uplift.
pages/04_Model_Health.py

Three questions this page answers:
  1. Which days did the model badly miss, and what happened there?
  2. Is the model drifting — consistently wrong in one direction lately?
  3. Did a promotion actually cause incremental orders, or discount existing ones?

All three need YOUR data. Upload a CSV or point the files at
data/training/actuals.csv and data/training/campaigns.csv.
"""

import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ml.anomaly import explain_anomaly, scan                      # noqa: E402
from ml.config import ACTUALS_CSV, CAMPAIGNS_CSV                  # noqa: E402
from ml.uplift import estimate                                    # noqa: E402
from utils.restaurant_impact import engine_status                 # noqa: E402

st.set_page_config(page_title="Model Health", page_icon="🩺", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.hero{background:linear-gradient(135deg,#0d1b2a 0%,#1b263b 100%);border-radius:18px;
  padding:22px 26px;color:white;margin-bottom:18px}
.hero h1{margin:0;font-size:23px;font-weight:600}
.hero p{margin:6px 0 0;opacity:.65;font-size:13px}
.alert{background:rgba(242,84,91,.12);border-left:4px solid #F2545B;border-radius:10px;
  padding:10px 14px;margin-bottom:8px;color:white;font-size:13px}
.okbox{background:rgba(46,125,50,.14);border-left:4px solid #2E7D32;border-radius:10px;
  padding:10px 14px;color:white;font-size:13px}
section[data-testid="stSidebar"]{background:rgba(13,27,42,.97)!important}
section[data-testid="stSidebar"] *{color:white!important}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🩺 Model Health</h1>
  <p>Anomalies, drift and campaign incrementality — the checks that tell you whether
     to trust today's forecast and whether your promotions are paying for themselves.</p>
</div>
""", unsafe_allow_html=True)

info = engine_status()
with st.sidebar:
    st.markdown("### 🤖 Live engine")
    st.metric("Engine", info["active_engine"].upper())
    v = info.get("model", {}).get("validation", {})
    if v.get("orders_mape_pct"):
        st.metric("Holdout MAPE", f"{v['orders_mape_pct']}%")
    if info.get("model", {}).get("interval_coverage_pct"):
        st.metric("80% interval coverage", f"{info['model']['interval_coverage_pct']}%")
    st.caption("Coverage should sit near 80. Far below means intervals are too "
               "narrow to trust for staffing decisions.")

tab_anom, tab_uplift, tab_schema = st.tabs(
    ["🚨 Anomalies & drift", "📈 Campaign uplift", "📋 Data formats"])

# ── Anomalies ─────────────────────────────────────────────────────────────────
with tab_anom:
    st.markdown("Compare actual orders against what the model expected. Isolated "
                "misses usually mean something happened at the outlet; a sustained "
                "one-sided gap means the model needs retraining.")

    up = st.file_uploader("Actuals CSV — date, city, restaurant_type, orders",
                          type=["csv"], key="anom")
    level = st.slider("Sensitivity (conformal level)", 0.90, 0.999, 0.99, 0.005,
                      help="0.99 flags roughly the worst 1% of days for each series")

    path = None
    if up is not None:
        path = os.path.join("/tmp", "uploaded_actuals.csv")
        with open(path, "wb") as fh:
            fh.write(up.getbuffer())
    elif os.path.exists(ACTUALS_CSV):
        path = ACTUALS_CSV
        st.caption(f"Using {ACTUALS_CSV}")

    if not path:
        st.info("Upload a CSV of actual daily orders to scan. Format is in the "
                "**Data formats** tab.")
    elif st.button("Run scan", type="primary"):
        with st.spinner("Scoring every outlet-day against the model…"):
            try:
                res = scan(path=path, level=level)
            except Exception as exc:                              # noqa: BLE001
                res = {"error": str(exc)}
        if "error" in res:
            st.error(res["error"])
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Outlet-days scanned", f"{res['scored_rows']:,}")
            c2.metric("MAPE vs actuals", f"{res['overall_mape_pct']}%")
            c3.metric("Bias", f"{res['overall_bias_pct']:+.1f}%")
            c4.metric("Anomalies", res["n_anomalies"])

            if res["drift"]:
                st.markdown("#### ⚠️ Drift warnings")
                for d in res["drift"]:
                    st.markdown(
                        f"<div class='alert'><b>{d['city']} / {d['restaurant_type']}</b> — "
                        f"{d['recent_bias_pct']:+.1f}% over the last 28 days. "
                        f"{d['diagnosis']}</div>", unsafe_allow_html=True)
                st.caption("Fix: retrain on newer history — `python -m ml.train`")
            else:
                st.markdown("<div class='okbox'>No drift detected — recent errors are "
                            "balanced around zero.</div>", unsafe_allow_html=True)

            if res["anomalies"]:
                st.markdown("#### 🚨 Flagged days")
                for a in res["anomalies"][:30]:
                    st.markdown(f"<div class='alert'>{explain_anomaly(a)}</div>",
                                unsafe_allow_html=True)
                st.download_button(
                    "Download flagged days (CSV)",
                    pd.DataFrame(res["anomalies"]).to_csv(index=False),
                    "anomalies.csv", "text/csv")

            st.markdown("#### Per-series accuracy")
            st.dataframe(pd.DataFrame(res["series"]), use_container_width=True,
                         hide_index=True)

# ── Uplift ────────────────────────────────────────────────────────────────────
with tab_uplift:
    st.markdown("Did the promotion **cause** orders, or discount people who were "
                "coming anyway? Campaign days are usually already-busy days, so the "
                "raw difference always flatters the campaign.")

    upc = st.file_uploader(
        "Campaign CSV — date, city, restaurant_type, orders, treated [, discount_pct, channel]",
        type=["csv"], key="uplift")
    seg = st.selectbox("Extra breakdown", ["none", "city", "restaurant_type", "channel"])

    cpath = None
    if upc is not None:
        cpath = os.path.join("/tmp", "uploaded_campaigns.csv")
        with open(cpath, "wb") as fh:
            fh.write(upc.getbuffer())
    elif os.path.exists(CAMPAIGNS_CSV):
        cpath = CAMPAIGNS_CSV
        st.caption(f"Using {CAMPAIGNS_CSV}")

    if not cpath:
        st.info("Upload a campaign log with both campaign and non-campaign days. "
                "The control days are what make the estimate possible.")
    elif st.button("Estimate uplift", type="primary"):
        with st.spinner("Fitting treated / control models…"):
            try:
                res = estimate(cpath, None if seg == "none" else seg)
            except Exception as exc:                              # noqa: BLE001
                res = {"error": str(exc)}
        if "error" in res:
            st.error(res["error"])
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Naive difference", f"{res['naive_difference_pct']:+.1f}%",
                      help="What a dashboard would report — inflated by confounding")
            c2.metric("Causal uplift (ATT)", f"{res['uplift_att_pct']:+.1f}%",
                      delta=f"{-res['naive_overstatement_pct_points']:.1f} pts vs naive")
            c3.metric("IPW cross-check", f"{res['uplift_ipw_pct']:+.1f}%")

            verdict = res["verdict"]
            box = "okbox" if verdict.startswith("REASONABLE") else "alert"
            st.markdown(f"<div class='{box}'>{verdict}</div>", unsafe_allow_html=True)

            st.markdown(f"""
            **Incrementality.** Of {res['observed_orders_on_campaign_days']:,} orders on
            campaign days, about **{res['incremental_orders_on_campaign_days']:,}** were
            caused by the campaign. {res['interpretation']}
            """)
            for key in list(res):
                if key.startswith("uplift_by_"):
                    st.markdown(f"**{key.replace('_', ' ').title()}**")
                    st.dataframe(pd.DataFrame(res[key]), use_container_width=True,
                                 hide_index=True)
            st.caption(res["caveat"])

# ── Schema help ───────────────────────────────────────────────────────────────
with tab_schema:
    st.markdown("""
#### Actuals — for anomaly & drift detection
`data/training/actuals.csv`

| column | required | notes |
|---|---|---|
| `date` | ✅ | YYYY-MM-DD |
| `city` | ✅ | must match a city in the geo database |
| `restaurant_type` | ✅ | QSR, Fine Dining, PBCL, Casual Dining, Cloud Kitchen, Cafe |
| `orders` | ✅ | actual orders that day |
| `aov` | — | average order value |
| `temp_max_c`, `precipitation_mm`, `weather_code` | — | improves the comparison |

#### Campaigns — for uplift
`data/training/campaigns.csv`

| column | required | notes |
|---|---|---|
| `date`, `city`, `restaurant_type`, `orders` | ✅ | as above |
| `treated` | ✅ | 1 on campaign days, 0 otherwise — **export both** |
| `discount_pct` | — | enables the uplift-by-discount-band breakdown |
| `channel` | — | swiggy / zomato / own-app |

#### Also available from the command line
```bash
python -m ml.anomaly --csv data/training/actuals.csv
python -m ml.uplift  --csv data/training/campaigns.csv --segment city
```
""")
