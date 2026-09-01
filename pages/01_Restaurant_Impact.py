"""
Restaurant Impact Page — QSR, Fine Dining, PBCL, Casual Dining, Cloud Kitchen, Cafe
pages/01_Restaurant_Impact.py
"""

import streamlit as st
import sys, os
from datetime import date, timedelta, datetime
import pytz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.india_geo import CITIES
from utils.restaurant_impact import predict_date_range, BASELINES, ALL_RT

st.set_page_config(page_title="Dotpe Demand Pulse — Restaurant Impact", page_icon="🍽️", layout="wide")

IST   = pytz.timezone("Asia/Kolkata")
now_i = datetime.now(IST)
today = now_i.date()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.rt-card{border-radius:16px;padding:16px 20px;color:white;margin-bottom:12px;border-left:4px solid}
.sig-badge{display:inline-block;padding:4px 12px;border-radius:16px;font-size:12px;font-weight:600;margin-top:6px}
.kpi-big{text-align:center;background:rgba(255,255,255,.08);border-radius:14px;
  padding:14px 8px;border:1px solid rgba(255,255,255,.12);color:white}
.kpi-val{font-size:24px;font-weight:600}
.kpi-lbl{font-size:10px;opacity:.6;margin-top:3px}
.section-hdr{color:white;font-size:15px;font-weight:600;margin:18px 0 10px;
  padding-bottom:5px;border-bottom:1px solid rgba(255,255,255,.12)}
section[data-testid="stSidebar"]{background:rgba(13,27,42,.97)!important}
section[data-testid="stSidebar"] *{color:white!important}
</style>
""", unsafe_allow_html=True)

RT_COLORS = {
    "QSR":           "#FF6F00",
    "Fine Dining":   "#1565C0",
    "PBCL":          "#4A148C",
    "Casual Dining": "#2E7D32",
    "Cloud Kitchen": "#00838F",
    "Cafe":          "#6A1B9A",
}
RT_ICONS = {
    "QSR":"🍔","Fine Dining":"🍽️","PBCL":"🍺",
    "Casual Dining":"🥘","Cloud Kitchen":"📦","Cafe":"☕",
}
RT_DESC = {
    "QSR":           "Quick Service — counter orders, delivery, takeaway (Mad Over Donuts, 99 Pancakes, McDonald's)",
    "Fine Dining":   "Upscale table dining — à la carte, reservations, white-tablecloth (ITC, Taj restaurants)",
    "PBCL":          "Pub, Bar, Café, Lounge — beverages + light bites (Social, Barista, brewpubs)",
    "Casual Dining": "Mid-range sit-down — Chili's, Barbeque Nation, Mainland China, Olive",
    "Cloud Kitchen": "Delivery-only brands — Rebel Foods, Biryani By Kilo, Faasos, Freshmenu",
    "Cafe":          "Coffee shops & bakery-cafes — Starbucks, Blue Tokai, Third Wave Coffee, Chaayos",
}
SIGNAL_COLOR = {
    "very_high_up":"#00C853","high_up":"#43A047","slight_up":"#81C784",
    "neutral":"#78909C","slight_down":"#FF8A65","low":"#EF5350","very_low":"#B71C1C",
}
SIGNAL_LABEL = {
    "very_high_up":"🚀 Very High Demand","high_up":"📈 High Uplift",
    "slight_up":"↑ Slight Uplift","neutral":"➡️ Neutral",
    "slight_down":"↓ Slight Drop","low":"📉 Low Demand","very_low":"🔴 Very Low",
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍽️ Restaurant Impact")
    st.markdown("Predicted order volume & price impact vs normal baseline")
    st.markdown(f"<div style='font-size:11px;opacity:.5'>🕐 IST: {now_i.strftime('%d %b %Y, %I:%M %p')}</div>", unsafe_allow_html=True)
    st.markdown("---")
    city = st.selectbox("📍 City", sorted(CITIES.keys()),
                        index=sorted(CITIES.keys()).index("New Delhi"))
    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1: start_date = st.date_input("From", today)
    with c2: end_date   = st.date_input("To",   today+timedelta(days=14))
    st.markdown("---")
    st.markdown("### 🌡️ Weather Override")
    override  = st.toggle("Apply custom weather")
    temp_max  = st.slider("Max Temp (°C)", 10, 50, 32) if override else None
    precip_mm = st.slider("Rainfall (mm)",  0, 80,  0) if override else None
    st.markdown("---")
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear(); st.rerun()

# ── Weather fetch ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_weather(city, start_str, end_str):
    """
    Fetch weather for the full date range.
    - Historical dates  → Open-Meteo archive API (actual observed data)
    - Today + future    → Open-Meteo forecast API
    For forecast dates, precipitation_sum can be None — we use
    precipitation_probability_max to estimate mm when sum is missing.
    """
    import requests; coords=CITIES[city]; temp_map={}
    try:
        # ── Historical (up to yesterday) ──────────────────────────────────────
        hist_end = min(date.fromisoformat(end_str), today - timedelta(days=1))
        if date.fromisoformat(start_str) <= hist_end:
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
                "latitude": coords["lat"], "longitude": coords["lon"],
                "daily": ["temperature_2m_max","temperature_2m_min",
                          "precipitation_sum","weathercode"],
                "start_date": start_str, "end_date": hist_end.isoformat(),
                "timezone": "Asia/Kolkata"}, timeout=12)
            if r.status_code == 200:
                d = r.json()["daily"]
                for i in range(len(d["time"])):
                    temp_map[d["time"][i]] = {
                        "temp_max_c":      d["temperature_2m_max"][i],
                        "temp_min_c":      d["temperature_2m_min"][i],
                        "precipitation_mm":d["precipitation_sum"][i] or 0,
                        "weather_code":    d["weathercode"][i],
                    }
        # ── Forecast (today + future) ─────────────────────────────────────────
        fcst_start = max(date.fromisoformat(start_str), today)
        if fcst_start <= date.fromisoformat(end_str):
            days = (date.fromisoformat(end_str) - today).days + 3
            r2 = requests.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": coords["lat"], "longitude": coords["lon"],
                "daily": ["temperature_2m_max","temperature_2m_min",
                          "precipitation_sum","weathercode",
                          "precipitation_probability_max"],
                "forecast_days": min(days, 16),
                "timezone": "Asia/Kolkata"}, timeout=10)
            if r2.status_code == 200:
                d2 = r2.json()["daily"]
                for i in range(len(d2["time"])):
                    precip = d2["precipitation_sum"][i]
                    rain_prob = d2["precipitation_probability_max"][i] or 0
                    # If precipitation_sum is None (common for forecast),
                    # estimate from probability: >60% → ~8mm, >30% → ~3mm, else 0
                    if precip is None:
                        if rain_prob >= 60:   precip = 10.0
                        elif rain_prob >= 40: precip = 5.0
                        elif rain_prob >= 25: precip = 2.0
                        else:                precip = 0.0
                    temp_map[d2["time"][i]] = {
                        "temp_max_c":           d2["temperature_2m_max"][i],
                        "temp_min_c":           d2["temperature_2m_min"][i],
                        "precipitation_mm":     precip,
                        "weather_code":         d2["weathercode"][i],
                        "rain_probability_pct": rain_prob,
                    }
    except Exception as e:
        pass
    return temp_map

with st.spinner("Calculating impact for all 6 restaurant types..."):
    weather_data = get_weather(city, start_date.isoformat(), end_date.isoformat())
    if override:
        for k in weather_data:
            weather_data[k]["temp_max_c"]=temp_max; weather_data[k]["precipitation_mm"]=precip_mm
    predictions = predict_date_range(city, start_date, end_date, weather_data)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"## 🍽️ Restaurant Demand Impact — {city}")
st.markdown(f"*{start_date.strftime('%d %b %Y')} → {end_date.strftime('%d %b %Y')} · IST: {now_i.strftime('%d %b %Y, %I:%M %p')} · Index: 100 = typical weekday · 💰 Price shown as avg. order value (AOV)*")

# ── KPI row — all 6 types ──────────────────────────────────────────────────────
cols = st.columns(6)
for col,rt in zip(cols, ALL_RT):
    preds=predictions[rt]
    avg_i=sum(p["predicted_index"] for p in preds)/len(preds) if preds else 100
    avg_p=sum(p["pct_change"] for p in preds)/len(preds) if preds else 0
    avg_aov=sum(p["predicted_aov"] for p in preds)/len(preds) if preds else 0
    avg_rev_p=sum(p["revenue_pct_change"] for p in preds)/len(preds) if preds else 0
    best =max(preds,key=lambda x:x["pct_change"]) if preds else {}
    worst=min(preds,key=lambda x:x["pct_change"]) if preds else {}
    c=RT_COLORS[rt]
    with col:
        st.markdown(f"""
        <div class="kpi-big" style="border-top:3px solid {c}">
          <div style="font-size:16px;margin-bottom:4px">{RT_ICONS[rt]} {rt}</div>
          <div class="kpi-val" style="color:{c}">{avg_i:.0f}</div>
          <div class="kpi-lbl">avg order index</div>
          <div style="font-size:12px;margin-top:5px;color:{'#43A047' if avg_p>=0 else '#EF5350'};font-weight:500">{avg_p:+.1f}% orders</div>
          <div style="margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,.1)">
            <div style="font-size:15px;font-weight:600">₹{avg_aov:,.0f}</div>
            <div class="kpi-lbl">avg price / order</div>
            <div style="font-size:11px;margin-top:2px;color:{'#43A047' if avg_rev_p>=0 else '#EF5350'};font-weight:500">{avg_rev_p:+.1f}% revenue</div>
          </div>
          <div style="font-size:9px;opacity:.35;margin-top:8px">
            ↑ {best.get('pct_change',0):+.0f}% {best.get('date','')[:10]}<br>
            ↓ {worst.get('pct_change',0):+.0f}% {worst.get('date','')[:10]}
          </div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
    "🍔 QSR","🍽️ Fine Dining","🍺 PBCL",
    "🥘 Casual Dining","📦 Cloud Kitchen","☕ Cafe","📊 Compare All"
])

def render_tab(rt):
    preds=predictions[rt]; c=RT_COLORS[rt]
    st.markdown(f"<div style='color:rgba(255,255,255,.5);font-size:12px;margin-bottom:12px'>{RT_DESC[rt]}</div>",unsafe_allow_html=True)
    try:
        import plotly.graph_objects as go
        dates=[p["date"] for p in preds]; idx=[p["predicted_index"] for p in preds]
        base=[p["base_index"] for p in preds]; pct=[p["pct_change"] for p in preds]
        rev_pct=[p["revenue_pct_change"] for p in preds]
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=dates,y=base,name="Baseline",line=dict(color="rgba(255,255,255,.2)",width=1.5,dash="dot")))
        fig.add_trace(go.Scatter(x=dates,y=idx,name="Predicted Orders",
            line=dict(color=c,width=2.5),fill="tonexty",
            fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},.12)",
            mode="lines+markers",marker=dict(size=7,color=c)))
        fig.add_trace(go.Bar(x=dates,y=pct,name="Order % Change",
            marker_color=[SIGNAL_COLOR.get(p["signal"],"#78909C") for p in preds],
            opacity=0.55,yaxis="y2",
            text=[f"{v:+.0f}%" for v in pct],textposition="outside",
            textfont=dict(color="rgba(255,255,255,.7)",size=10)))
        fig.add_trace(go.Scatter(x=dates,y=rev_pct,name="Revenue % Change (price-adjusted)",
            line=dict(color="#FFD54F",width=2,dash="dash"),mode="lines+markers",
            marker=dict(size=6,color="#FFD54F"),yaxis="y2"))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.03)",
            font=dict(color="white",family="Inter"),height=320,margin=dict(l=0,r=0,t=10,b=0),
            hovermode="x unified",
            legend=dict(orientation="h",yanchor="bottom",y=1.02,bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="rgba(255,255,255,.08)",tickfont=dict(color="rgba(255,255,255,.6)")),
            yaxis=dict(title="Order Index",gridcolor="rgba(255,255,255,.08)",
                       tickfont=dict(color="rgba(255,255,255,.6)"),range=[0,280]),
            yaxis2=dict(title="% Change",overlaying="y",side="right",
                        tickfont=dict(color="rgba(255,255,255,.4)"),showgrid=False,
                        zeroline=True,zerolinecolor="rgba(255,255,255,.2)"))
        st.plotly_chart(fig,use_container_width=True)
    except Exception as e: st.info(f"Chart: {e}")

    st.markdown('<div class="section-hdr">Notable Impact Days (±8% change)</div>',unsafe_allow_html=True)
    notable=[p for p in preds if abs(p["pct_change"])>=8]
    if notable:
        c1,c2=st.columns(2)
        for i,p in enumerate(notable):
            col=c1 if i%2==0 else c2
            sc=SIGNAL_COLOR.get(p["signal"],"#78909C"); sl=SIGNAL_LABEL.get(p["signal"],p["signal"])
            facts="<br>".join([f"• {f}" for f in p["factors"]]) if p["factors"] else "No major factors"
            with col:
                st.markdown(f"""
                <div class="rt-card" style="background:{c}18;border-left-color:{c}">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                      <div style="font-size:14px;font-weight:600">{p['weekday']}, {p['date']}</div>
                      <div class="sig-badge" style="background:{sc}22;border:1px solid {sc};color:{sc}">{sl}</div>
                    </div>
                    <div style="text-align:right">
                      <div style="font-size:28px;font-weight:300;color:{sc}">{p['pct_change']:+.0f}%</div>
                      <div style="font-size:12px;opacity:.5">Index: {p['predicted_index']:.0f}</div>
                    </div>
                  </div>
                  <div style="display:flex;gap:8px;margin-top:8px">
                    <span style="background:rgba(255,213,79,.15);color:#FFD54F;border:1px solid rgba(255,213,79,.3);
                      border-radius:8px;padding:3px 9px;font-size:11px;font-weight:500">
                      💰 ₹{p['predicted_aov']:,.0f}/order
                    </span>
                    <span style="background:rgba(255,213,79,.15);color:#FFD54F;border:1px solid rgba(255,213,79,.3);
                      border-radius:8px;padding:3px 9px;font-size:11px;font-weight:500">
                      Revenue {p['revenue_pct_change']:+.0f}%
                    </span>
                  </div>
                  <div style="margin-top:8px;font-size:11px;opacity:.65;line-height:1.7">{facts}</div>
                  <div style="font-size:10px;opacity:.35;margin-top:5px">
                    Confidence: {p['confidence']} · {len(p['active_events'])} event(s)
                  </div>
                </div>""",unsafe_allow_html=True)
    else:
        st.info("No significant impact days in this range. Try extending the date range.")

with tab1: render_tab("QSR")
with tab2: render_tab("Fine Dining")
with tab3: render_tab("PBCL")
with tab7: render_tab("Casual Dining")
with tab5: render_tab("Cloud Kitchen")
with tab6: render_tab("Cafe")

with tab7:
    st.markdown("**All 6 restaurant types — order index + signal heatmap + event table**")
    try:
        import plotly.graph_objects as go
        dates=[p["date"] for p in predictions["QSR"]]
        fig3=go.Figure()
        for rt in ALL_RT:
            idx=[p["predicted_index"] for p in predictions[rt]]
            fig3.add_trace(go.Scatter(x=dates,y=idx,name=f"{RT_ICONS[rt]} {rt}",
                line=dict(color=RT_COLORS[rt],width=2),mode="lines+markers",marker=dict(size=5)))
        fig3.add_hline(y=100,line_dash="dot",line_color="rgba(255,255,255,.2)",
                       annotation_text="Weekday baseline",annotation_font_color="rgba(255,255,255,.35)")
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.03)",
            font=dict(color="white",family="Inter"),height=340,margin=dict(l=0,r=0,t=10,b=0),
            hovermode="x unified",
            legend=dict(orientation="h",yanchor="bottom",y=1.02,bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="rgba(255,255,255,.08)",tickfont=dict(color="rgba(255,255,255,.6)")),
            yaxis=dict(title="Order Index (100=baseline)",gridcolor="rgba(255,255,255,.08)",
                       tickfont=dict(color="rgba(255,255,255,.6)")))
        st.plotly_chart(fig3,use_container_width=True)

        st.markdown('<div class="section-hdr">📅 Signal Heatmap</div>',unsafe_allow_html=True)
        sig_val={"very_high_up":3,"high_up":2,"slight_up":1,"neutral":0,"slight_down":-1,"low":-2,"very_low":-3}
        z_data=[]; y_labels=[]; hover_text=[]
        for rt in ALL_RT:
            z_data.append([sig_val.get(p["signal"],0) for p in predictions[rt]])
            y_labels.append(f"{RT_ICONS[rt]} {rt}")
            hover_text.append([
                f"<b>{RT_ICONS[rt]} {rt}</b><br>{p['date']} ({p['weekday']})<br>"
                f"Signal: {SIGNAL_LABEL.get(p['signal'],p['signal'])}<br>"
                f"Change: {p['pct_change']:+.1f}%<br>Index: {p['predicted_index']:.0f}"
                for p in predictions[rt]])
        fig4=go.Figure(go.Heatmap(
            z=z_data,x=dates,y=y_labels,text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            colorscale=[[0,"#B71C1C"],[0.17,"#EF5350"],[0.33,"#FF8A65"],
                        [0.5,"#455A64"],[0.67,"#81C784"],[0.83,"#43A047"],[1,"#00C853"]],
            zmin=-3,zmax=3,showscale=True,
            colorbar=dict(
                title=dict(text="Impact",font=dict(color="white",size=12)),
                tickvals=[-3,-2,-1,0,1,2,3],
                ticktext=["Very Low","Low","Slight↓","Neutral","Slight↑","High","Very High"],
                tickfont=dict(color="white",size=10))))
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white",family="Inter"),height=260,margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(tickfont=dict(color="rgba(255,255,255,.7)"),gridcolor="rgba(255,255,255,.05)"),
            yaxis=dict(tickfont=dict(color="rgba(255,255,255,.9)",size=11)))
        st.plotly_chart(fig4,use_container_width=True)
    except Exception as e: st.warning(f"Chart: {e}")

    st.markdown('<div class="section-hdr">📋 Event Impact Reference Table</div>',unsafe_allow_html=True)
    from data.events_db import EVENTS_DB, _event_applies_to_geo
    from datetime import date as dt_date
    from utils.restaurant_impact import EVENT_MULTIPLIERS, aov_event_multiplier
    active_evts=[]
    s=dt_date.fromisoformat(start_date.isoformat()); e_d=dt_date.fromisoformat(end_date.isoformat())
    for ev in EVENTS_DB:
        ev_s=dt_date.fromisoformat(ev["start_date"]); ev_e=dt_date.fromisoformat(ev["end_date"])
        if ev_e<s or ev_s>e_d: continue
        if not _event_applies_to_geo(ev,city=city): continue
        active_evts.append(ev)
    if active_evts:
        rows=[]
        for ev in active_evts:
            cat=ev["category"]; sub=ev.get("subcategory","*")
            cm=EVENT_MULTIPLIERS.get(cat,{}); sm=cm.get(sub,cm.get("*",{}))
            def fmt(rt): v=(sm.get(rt,1.0)-1)*100; return f"{'↑' if v>0 else '↓' if v<0 else '→'} {abs(v):.0f}%"
            avg_order_mult = sum(sm.get(rt,1.0) for rt in ALL_RT)/len(ALL_RT)
            avg_aov_mult   = aov_event_multiplier(avg_order_mult)
            avg_rev_pct    = (avg_order_mult*avg_aov_mult-1)*100
            rows.append({"Event":ev["name"],"Category":cat,
                "Dates":f"{ev['start_date']} → {ev['end_date']}",
                "🍔 QSR":fmt("QSR"),"🍽️ Fine Dining":fmt("Fine Dining"),
                "🍺 PBCL":fmt("PBCL"),"🥘 Casual":fmt("Casual Dining"),
                "📦 Cloud":fmt("Cloud Kitchen"),"☕ Cafe":fmt("Cafe"),
                "💰 Avg Revenue Impact":f"{'↑' if avg_rev_pct>0 else '↓' if avg_rev_pct<0 else '→'} {abs(avg_rev_pct):.0f}%"})
        import pandas as pd
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    else: st.info("No events for this city and date range.")

st.markdown("---")
st.markdown(f"<div style='font-size:10px;color:rgba(255,255,255,.3)'>Order baselines: QSR Sat 145, Fine Dining Sat 165, PBCL Sat 175, Casual Sat 155, Cloud Kitchen Sat 130, Cafe Sat 140 · AOV baselines: QSR ₹330, Fine Dining ₹2,100, PBCL ₹1,350, Casual ₹950, Cloud Kitchen ₹480, Cafe ₹420. IST: {now_i.strftime('%d %b %Y %H:%M')}. Predictive model — weather is used internally, not displayed.</div>",unsafe_allow_html=True)
