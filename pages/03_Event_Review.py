"""
Event Review — the human gate in front of LLM-proposed calendar events.
pages/03_Event_Review.py

Nothing an LLM proposes reaches the events database without someone approving it
here. Candidates that fail hard validation (unknown city, impossible dates, wrong
category, duplicate) never even appear — they are rejected in Python first.
"""

import os
import sys
from datetime import date

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.events_db import EVENTS_DB                                   # noqa: E402
from ml.event_ingest import (approve_ids, load_pending, propose_for_year,  # noqa: E402
                             propose_from_text, propose_from_urls, reject_ids)
from ml.llm import LLMUnavailable                                     # noqa: E402
from pages._llm_sidebar import llm_sidebar                            # noqa: E402

st.set_page_config(page_title="Event Review", page_icon="🗂️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.hero{background:linear-gradient(135deg,#0d1b2a 0%,#1b263b 100%);border-radius:18px;
  padding:22px 26px;color:white;margin-bottom:18px}
.hero h1{margin:0;font-size:23px;font-weight:600}
.hero p{margin:6px 0 0;opacity:.65;font-size:13px}
.cand{background:rgba(255,255,255,.05);border-left:4px solid #F2545B;border-radius:12px;
  padding:14px 18px;margin-bottom:10px;color:white}
.cand b{font-size:15px}
.meta{opacity:.6;font-size:12px;margin-top:4px}
section[data-testid="stSidebar"]{background:rgba(13,27,42,.97)!important}
section[data-testid="stSidebar"] *{color:white!important}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🗂️ Event Review</h1>
  <p>The calendar is what the forecast rests on. An LLM reads sources and proposes
     events; validation rejects the impossible ones; you approve the rest.</p>
</div>
""", unsafe_allow_html=True)

llm = llm_sidebar()

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Database")
    st.metric("Events in DB", len(EVENTS_DB))
    years = sorted({e["start_date"][:4] for e in EVENTS_DB})
    st.caption(f"Covering {years[0]} – {years[-1]}")
    st.markdown("---")
    st.caption("Approved events are written to `data/events_db.py`. "
               "Retrain afterwards so the model sees them: `python -m ml.train`")

tab_propose, tab_review, tab_browse = st.tabs(
    ["✨ Propose events", "✅ Review queue", "📚 Browse database"])

# ── Propose ───────────────────────────────────────────────────────────────────
with tab_propose:
    if not llm["available"]:
        st.warning("Proposing events needs a language model — Ollama runs locally with "
                   "no API key (see the sidebar). The rest of the app works without "
                   "one: you can still browse and hand-edit the database.")
    mode = st.radio("Source", ["Whole year calendar", "Web pages", "Pasted text"],
                    horizontal=True)

    if mode == "Whole year calendar":
        y = st.number_input("Year", min_value=2023, max_value=date.today().year + 3,
                            value=date.today().year + 1, step=1)
        st.caption("Proposes the festival and holiday calendar for that year — the "
                   "job that was previously done by hand, and got nine 2026 dates wrong.")
        go = st.button("Propose calendar", type="primary", disabled=not llm["available"])
        if go:
            with st.spinner(f"Reading the {y} calendar…"):
                try:
                    st.session_state.last = propose_for_year(int(y))
                except LLMUnavailable as e:
                    st.error(str(e))
                except Exception as e:                             # noqa: BLE001
                    st.error(f"Extraction failed: {e}")

    elif mode == "Web pages":
        urls = st.text_area("URLs, one per line",
                            placeholder="https://www.drikpanchang.com/…\nhttps://state-gazette…")
        go = st.button("Extract from pages", type="primary", disabled=not llm["available"])
        if go and urls.strip():
            with st.spinner("Fetching and extracting…"):
                try:
                    st.session_state.last = propose_from_urls(
                        [u.strip() for u in urls.splitlines() if u.strip()])
                except Exception as e:                             # noqa: BLE001
                    st.error(f"Extraction failed: {e}")

    else:
        txt = st.text_area("Paste a notice, circular, schedule or article", height=200)
        go = st.button("Extract from text", type="primary", disabled=not llm["available"])
        if go and txt.strip():
            with st.spinner("Extracting…"):
                try:
                    st.session_state.last = propose_from_text(txt)
                except Exception as e:                             # noqa: BLE001
                    st.error(f"Extraction failed: {e}")

    res = st.session_state.get("last")
    if res:
        c1, c2, c3 = st.columns(3)
        c1.metric("Proposed", res["proposed"])
        c2.metric("Queued for review", res["queued_for_review"])
        c3.metric("Auto-rejected", len(res["rejected_by_validation"]))
        if res["rejected_by_validation"]:
            with st.expander("Rejected by validation (never reached the queue)"):
                for r in res["rejected_by_validation"]:
                    st.markdown(f"- **{r['candidate'].get('name','?')}** — {r['reason']}")
        if res["queued_for_review"]:
            st.success("Candidates are in the Review queue tab.")

# ── Review ────────────────────────────────────────────────────────────────────
with tab_review:
    pending = load_pending()
    if not pending:
        st.info("Review queue is empty. Propose some events in the first tab.")
    else:
        st.markdown(f"**{len(pending)} candidate(s) awaiting your decision**")
        ca, cb = st.columns(2)
        if ca.button("✅ Approve all", use_container_width=True):
            out = approve_ids([e["id"] for e in pending])
            st.success(f"Merged {len(out['approved'])} events. {out['note']}")
            st.rerun()
        if cb.button("🗑️ Reject all", use_container_width=True):
            reject_ids([e["id"] for e in pending])
            st.rerun()
        st.markdown("---")

        for ev in pending:
            geo = (", ".join(ev.get("cities") or ev.get("states") or ev.get("zones") or [])
                   or "Pan-India")
            st.markdown(f"""
            <div class="cand">
              <b>{ev['name']}</b> &nbsp;·&nbsp; {ev['start_date']} → {ev['end_date']}
              <div class="meta">{ev['category']} / {ev['subcategory']} ·
                {ev['scope']} ({geo}) · expected: {ev['impact_on_demand']}</div>
              <div class="meta">{ev['description']}</div>
              <div class="meta">source: {ev.get('source','—')} ·
                proposed {ev.get('_review',{}).get('proposed_at','')}</div>
            </div>""", unsafe_allow_html=True)
            a, b, _ = st.columns([1, 1, 6])
            if a.button("Approve", key=f"a{ev['id']}"):
                out = approve_ids([ev["id"]])
                st.success(f"Merged {ev['name']}. {out['note']}")
                st.rerun()
            if b.button("Reject", key=f"r{ev['id']}"):
                reject_ids([ev["id"]])
                st.rerun()

# ── Browse ────────────────────────────────────────────────────────────────────
with tab_browse:
    df = pd.DataFrame([{
        "id": e["id"], "name": e["name"], "category": e["category"],
        "subcategory": e.get("subcategory"), "start": e["start_date"],
        "end": e["end_date"], "scope": e.get("scope"),
        "impact": e.get("impact_on_demand"),
    } for e in EVENTS_DB]).sort_values("start")

    c1, c2 = st.columns(2)
    cat = c1.selectbox("Category", ["All"] + sorted(df["category"].unique()))
    yr = c2.selectbox("Year", ["All"] + sorted({s[:4] for s in df["start"]}))
    view = df
    if cat != "All":
        view = view[view["category"] == cat]
    if yr != "All":
        view = view[view["start"].str.startswith(yr)]
    st.caption(f"{len(view)} of {len(df)} events")
    st.dataframe(view, use_container_width=True, hide_index=True, height=520)
