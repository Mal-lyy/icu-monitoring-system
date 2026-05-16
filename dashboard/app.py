import streamlit as st
from pymongo import MongoClient
import pandas as pd
import plotly.graph_objects as go
import time

st.set_page_config(
    page_title="ICU Monitor",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
/* Dark Theme Core */
.stApp {
    background-color: #0f1116;
    color: #e0e6ed;
}
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}
hr {
    border-color: #2a2d35 !important;
}

/* Badge Styling */
.badge {
    display: inline-block; border-radius: 6px;
    padding: 4px 8px; font-size: 0.8rem;
    font-weight: 600; margin: 3px;
    letter-spacing: 0.5px;
}

/* Risk Score Highlights */
.risk-pill {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 10px; border-radius: 10px;
    font-weight: bold; margin: 5px 0;
    text-align: center;
}
.risk-low { background: rgba(40, 167, 69, 0.15); border: 1px solid #28a745; color: #28a745; }
.risk-medium { background: rgba(253, 126, 20, 0.15); border: 1px solid #fd7e14; color: #fd7e14; }
.risk-high { background: rgba(220, 53, 69, 0.15); border: 1px solid #dc3545; color: #dc3545; }
.risk-val { font-size: 1.6rem; line-height: 1.2; }
.risk-label { font-size: 0.75rem; text-transform: uppercase; opacity: 0.9; }

/* Custom Cards */
.patient-card {
    background-color: #1a1d24; border-radius: 12px;
    padding: 16px; margin-bottom: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    transition: transform 0.2s ease;
}
.patient-card:hover { transform: translateY(-2px); }

.vitals-table { width: 100%; font-size: 0.9rem; border-collapse: separate; border-spacing: 0 8px; }
.vitals-table td { padding: 6px 10px; background: #252830; }
.vitals-table td:first-child { border-radius: 6px 0 0 6px; color: #a0aabf; width: 25%; }
.vitals-table td:nth-child(2) { width: 20%; font-weight: bold; }
.vitals-table td:nth-child(3) { width: 10%; background: transparent; }
.vitals-table td:nth-child(4) { border-radius: 6px 0 0 6px; color: #a0aabf; width: 25%; }
.vitals-table td:last-child { border-radius: 0 6px 6px 0; font-weight: bold; width: 20%; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_db():
    client = MongoClient("mongodb://localhost:27017/")
    return client["icu_monitoring"]

db = get_db()

STATUS_COLOR = {"NORMAL": "#28a745", "WARNING": "#fd7e14", "CRITICAL": "#dc3545"}
STATUS_EMOJI = {"NORMAL": "🟢", "WARNING": "🟡", "CRITICAL": "🔴"}
VITALS = ["heart_rate", "systolic_bp", "diastolic_bp",
          "spo2", "temperature", "respiratory_rate"]

def fmt(val, unit=""):
    if val is None or str(val).strip() in ("", "None", "—"):
        return "—"
    try:
        return f"{round(float(val), 1)}{unit}"
    except Exception:
        return f"{val}{unit}"

def badge(label, color="#6c757d"):
    return (f"<span class='badge' "
            f"style='background:{color}22; color:{color}; "
            f"border:1px solid {color}55;'>{label}</span>")

def news2_color(score):
    if score is None: return "#6c757d"
    if score <= 4:    return "#28a745"
    if score <= 6:    return "#fd7e14"
    return "#dc3545"

def mort_color(val):
    if val is None: return "#6c757d"
    if val >= 30:   return "#dc3545"
    if val >= 10:   return "#fd7e14"
    return "#28a745"

def mort_class(val):
    if val is None: return ""
    if val >= 30:   return "risk-high"
    if val >= 10:   return "risk-medium"
    return "risk-low"

def get_risk_pill(label, val, val_str=None):
    if val is None:
        return f"<div class='risk-pill' style='background: #252830; color: #6c757d; border: 1px solid #3d424d;'><div class='risk-val'>—</div><div class='risk-label'>{label}</div></div>"
    return f"<div class='risk-pill {mort_class(val)}'><div class='risk-val'>{val_str if val_str else f'{val}%'}</div><div class='risk-label'>{label}</div></div>"

def get_large_risk_pill(label, val, subtitle=""):
    if val is None:
        return f"<div class='risk-pill' style='padding: 20px; background: #252830; color: #6c757d; border: 1px solid #3d424d;'><div class='risk-label' style='font-size: 1rem; margin-bottom: 5px;'>{label}</div><div class='risk-val' style='font-size: 2.5rem;'>—</div><div style='font-size: 0.8rem; margin-top: 10px; opacity: 0.8;'>{subtitle}</div></div>"
    return f"<div class='risk-pill {mort_class(val)}' style='padding: 20px;'><div class='risk-label' style='font-size: 1rem; margin-bottom: 5px;'>{label}</div><div class='risk-val' style='font-size: 2.5rem;'>{val}%</div><div style='font-size: 0.8rem; margin-top: 10px; opacity: 0.8;'>{subtitle}</div></div>"

def get_last_known(patient_id):
    result = {v: None for v in VITALS}
    docs = list(db["vitals_log"].find(
        {"patient_id": patient_id},
        {"_id": 0, "heart_rate": 1, "systolic_bp": 1,
         "diastolic_bp": 1, "spo2": 1,
         "temperature": 1, "respiratory_rate": 1}
    ).sort("timestamp", -1).limit(20))
    for doc in docs:
        for vital in VITALS:
            if result[vital] is None and doc.get(vital) is not None:
                result[vital] = doc[vital]
        if all(result[v] is not None for v in VITALS):
            break
    return result

@st.cache_data(ttl=30)
def fetch_sepsis(_db):
    try:
        return {r["patient_id"]: r
                for r in _db["sepsis_risk"].find({}, {"_id": 0})}
    except Exception:
        return {}

@st.cache_data(ttl=30)
def fetch_charlson(_db):
    try:
        return {r["patient_id"]: r
                for r in _db["charlson_scores"].find({}, {"_id": 0})}
    except Exception:
        return {}

@st.cache_data(ttl=30)
def fetch_profiles(_db):
    try:
        return {r["patient_id"]: r
                for r in _db["patient_profiles"].find({}, {"_id": 0})}
    except Exception:
        return {}

def get_patients():
    all_p = list(db["vitals_log"].aggregate([
        {"$sort":        {"timestamp": -1}},
        {"$group":       {"_id": "$patient_id",
                          "latest": {"$first": "$$ROOT"}}},
        {"$replaceRoot": {"newRoot": "$latest"}},
    ]))
    crit = [p for p in all_p if p.get("overall_status") == "CRITICAL"]
    warn = [p for p in all_p if p.get("overall_status") == "WARNING"]
    norm = [p for p in all_p if p.get("overall_status") == "NORMAL"]
    sel  = crit[:6] + warn[:6] + norm[:8]
    return sorted(sel, key=lambda p: {
        "CRITICAL": 0, "WARNING": 1, "NORMAL": 2
    }.get(p.get("overall_status"), 3))

# ── Static Header (never rerenders) ──────────────────────────
st.html("""
<h1 style='text-align:center; color:#ff4b4b; font-weight:800; letter-spacing:-1px;'>
    🏥 ICU Real-Time Vitals Monitoring System
</h1>
<p style='text-align:center; color:#8b949e; margin-top:-10px; font-size:1.1rem;'>
    MIMIC-III · Kafka · Spark · MongoDB · NEWS2 · qSOFA · XGBoost · LSTM
</p><hr style='border-color:#2a2d35;'>
""")

tab1, tab2, tab3 = st.tabs(
    ["📊 Patient Overview", "🔬 Patient Detail", "🚨 Alerts"]
)

# ═══════════════════════════════════════════════════════════════
# TAB 1 — Patient Overview (auto-refreshes every 5s)
# ═══════════════════════════════════════════════════════════════
with tab1:
    @st.fragment(run_every=5)
    def overview_tab():
        patients    = get_patients()
        sepsis      = fetch_sepsis(db)
        charlson    = fetch_charlson(db)
        profiles    = fetch_profiles(db)
        n_critical  = sum(1 for p in patients if p.get("overall_status") == "CRITICAL")
        n_warning   = sum(1 for p in patients if p.get("overall_status") == "WARNING")
        total_alerts= db["alert_history"].count_documents({})
        high_sepsis = sum(1 for v in sepsis.values()
                          if v.get("sepsis_risk") in ("HIGH","CRITICAL"))

        if n_critical > 0:
            st.error(f"🚨 CRITICAL — {n_critical} patient(s) require immediate attention!")
        elif n_warning > 0:
            st.warning(f"⚠️ WARNING — {n_warning} patient(s) have abnormal vitals")
        else:
            st.success("✅ All patients stable")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("👥 Monitored",    len(patients))
        m2.metric("🔴 Critical",     n_critical)
        m3.metric("🟡 Warning",      n_warning)
        m4.metric("📋 Total Alerts", total_alerts)
        m5.metric("🦠 High Sepsis",  high_sepsis)
        st.markdown("---")

        if not patients:
            st.info("⏳ Waiting for data from pipeline...")
            return

        cols = st.columns(2)
        for i, p in enumerate(patients):
            pid      = p.get("patient_id", "")
            bare     = str(pid).lstrip("P")
            status   = p.get("overall_status", "NORMAL")
            color    = STATUS_COLOR.get(status, "#6c757d")
            emoji    = STATUS_EMOJI.get(status, "⚪")
            known    = get_last_known(pid)
            sep      = sepsis.get(pid, sepsis.get(bare, {}))
            cci      = charlson.get(bare, charlson.get(pid, {}))
            prof     = profiles.get(bare, profiles.get(pid, {}))

            hr   = fmt(known.get("heart_rate"),       " bpm")
            bp   = fmt(known.get("systolic_bp"),      " mmHg")
            spo2 = fmt(known.get("spo2"),             "%")
            temp = fmt(known.get("temperature"),      " °C")
            rr   = fmt(known.get("respiratory_rate"), " /min")
            ts   = str(p.get("timestamp", ""))[:19]

            news2    = p.get("news2_score")
            qsofa    = p.get("qsofa_score")
            shock    = p.get("shock_index")
            red      = p.get("red_score_triggered", False)
            sep_risk = sep.get("sepsis_risk", "—")
            cci_sc   = cci.get("cci_score")
            age      = prof.get("age_at_first_icu")
            gender   = prof.get("gender", "")
            mort     = p.get("mortality_risk_pct")
            lstm_48h = p.get("lstm_48h_risk_pct")
            demo     = " · ".join(filter(None,
                        [f"{int(age)}y" if age else "", gender]))

            sep_colors  = {"HIGH": "#dc3545", "MODERATE": "#fd7e14",
                           "LOW":  "#ffc107", "NONE":     "#28a745"}
            qsofa_color = ("#dc3545" if qsofa and qsofa >= 2
                           else "#fd7e14" if qsofa == 1 else "#28a745")
            shock_color = "#dc3545" if shock and shock > 1.0 else "#28a745"

            badges_html = "".join([
                badge(f"NEWS2 {news2 if news2 is not None else '—'}" +
                      (" 🔴" if red else ""), news2_color(news2)),
                badge(f"qSOFA {qsofa if qsofa is not None else '—'}",
                      qsofa_color),
                badge(f"SI {round(shock,2) if shock else '—'}",
                      shock_color),
                badge(f"Sepsis {sep_risk}",
                      sep_colors.get(sep_risk, "#6c757d")),
                badge(f"CCI {cci_sc if cci_sc is not None else '—'}",
                      "#6c757d"),
                badge(f"🤖 {mort}%" if mort is not None
                      else "🤖 —", mort_color(mort)),
                badge(f"⏱️ 48h {lstm_48h}%" if lstm_48h is not None
                      else "⏱️ 48h —", mort_color(lstm_48h)),
            ])

            risk_html = f"""
            <div style='display:flex; gap:10px; margin-top:12px;'>
                <div style='flex:1;'>{get_risk_pill("XGB Mortality", mort)}</div>
                <div style='flex:1;'>{get_risk_pill("LSTM 48h Risk", lstm_48h)}</div>
            </div>
            """

            with cols[i % 2]:
                st.html(f"""
<div class='patient-card' style='border-left: 5px solid {color};'>
  <div style='display:flex; justify-content:space-between; align-items:center;'>
      <h4 style='color:{color};margin:0;font-size:1.2rem;'>
        {emoji} {pid}
      </h4>
      {f"<span style='font-size:0.85em;color:#a0aabf;background:#252830;padding:2px 8px;border-radius:10px;'>{demo}</span>" if demo else ""}
  </div>
  <hr style='margin:12px 0;border-color:#2a2d35;'>
  
  <table class='vitals-table'>
    <tr>
      <td>❤️ HR</td><td>{hr}</td>
      <td></td>
      <td>🩸 SBP</td><td>{bp}</td>
    </tr><tr>
      <td>🫁 SpO₂</td><td>{spo2}</td>
      <td></td>
      <td>🌡️ Temp</td><td>{temp}</td>
    </tr><tr>
      <td>💨 RR</td><td>{rr}</td>
      <td></td>
      <td colspan='2' style='text-align:right;font-size:0.8em;font-weight:normal;color:#6c757d;'>⏱️ {ts}</td>
    </tr>
  </table>
  
  <div style='margin-top:12px; display:flex; flex-wrap:wrap; gap:4px;'>
      {badges_html}
  </div>
  {risk_html}
</div>
""")

        st.markdown("---")
        st.caption(
            f"Auto-refresh every 5s · "
            f"Last update: {pd.Timestamp.now().strftime('%H:%M:%S')} · "
            f"MIMIC-III Full Dataset · BIA 678-WS · Stevens Institute of Technology"
        )

    overview_tab()

# ═══════════════════════════════════════════════════════════════
# TAB 2 — Patient Detail
# ═══════════════════════════════════════════════════════════════
with tab2:
    @st.fragment(run_every=3)
    def refresh_tab2_data():
        st.session_state["tab2_patients"] = get_patients()
        st.session_state["tab2_sepsis"]   = fetch_sepsis(db)
        st.session_state["tab2_charlson"] = fetch_charlson(db)
        st.session_state["tab2_profiles"] = fetch_profiles(db)

    refresh_tab2_data()

    patients = st.session_state.get("tab2_patients", get_patients())
    sepsis   = st.session_state.get("tab2_sepsis",   fetch_sepsis(db))
    charlson = st.session_state.get("tab2_charlson", fetch_charlson(db))
    profiles = st.session_state.get("tab2_profiles", fetch_profiles(db))

    pid_options = [p.get("patient_id") for p in patients]
    if not pid_options:
        st.info("⏳ No patients yet.")
    else:
        if "selected_pid" not in st.session_state:
            st.session_state.selected_pid = pid_options[0]

        if st.session_state.selected_pid not in pid_options:
            st.session_state.selected_pid = pid_options[0]

        sel_pid = st.selectbox(
            "Select patient",
            pid_options,
            index=pid_options.index(st.session_state.selected_pid),
            key="patient_selector",
            on_change=lambda: st.session_state.update(
                {"selected_pid": st.session_state.patient_selector}
            )
        )

        bare     = str(sel_pid).lstrip("P")
        latest_v = db["vitals_log"].find_one(
            {"patient_id": sel_pid},
            sort=[("timestamp", -1)]
        )

        known    = get_last_known(sel_pid)
        prof     = profiles.get(bare, profiles.get(sel_pid, {}))
        sep_d    = sepsis.get(sel_pid, sepsis.get(bare, {}))
        cci_d    = charlson.get(bare, charlson.get(sel_pid, {}))
        status   = latest_v.get("overall_status", "NORMAL")
        color    = STATUS_COLOR.get(status, "#6c757d")

        st.markdown(
            f"### {STATUS_EMOJI.get(status,'⚪')} {sel_pid} — "
            f"<span style='color:{color};'>{status}</span>",
            unsafe_allow_html=True
        )

        if prof:
            st.markdown("#### 👤 Demographics")
            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("Age", f"{int(prof['age_at_first_icu'])}y"
                      if prof.get("age_at_first_icu") else "—")
            d2.metric("Gender",    prof.get("gender") or "—")
            d3.metric("Ethnicity", (prof.get("ethnicity") or "—")[:16])
            d4.metric("ICU Stays", prof.get("total_icu_stays") or "—")
            d5.metric("ICU LOS",   f"{prof.get('total_icu_los_days','—')} d")
            if prof.get("primary_diagnosis"):
                st.caption(f"Diagnosis: {prof['primary_diagnosis']} · "
                           f"Unit: {prof.get('last_careunit','—')}")
            st.markdown("---")

        st.markdown("#### 🧮 Clinical Scores")
        sc1, sc2, sc3, sc4 = st.columns(4)

        news2 = latest_v.get("news2_score")
        red   = latest_v.get("red_score_triggered", False)
        sc1.metric("NEWS2 Score", news2 if news2 is not None else "—")
        if red: sc1.caption("🔴 Red score triggered")
        sc1.caption(f"RR:{latest_v.get('rr_score','—')} "
                    f"SpO₂:{latest_v.get('spo2_score','—')} "
                    f"BP:{latest_v.get('bp_score','—')} "
                    f"HR:{latest_v.get('hr_score','—')} "
                    f"Temp:{latest_v.get('temp_score','—')}")

        qsofa = latest_v.get("qsofa_score")
        sc2.metric("qSOFA Score", qsofa if qsofa is not None else "—")
        sc2.caption(f"RR≥22:{latest_v.get('qsofa_rr','—')} "
                    f"SBP≤100:{latest_v.get('qsofa_bp','—')}")
        if qsofa and qsofa >= 2:
            sc2.caption("⚠️ Sepsis suspected")

        si   = latest_v.get("shock_index")
        hemo = latest_v.get("hemodynamic_instability", False)
        sc3.metric("Shock Index", round(si, 2) if si else "—")
        sc3.caption("⚠️ Hemodynamic instability"
                    if hemo else "✅ Hemodynamically stable")

        sep_risk = sep_d.get("sepsis_risk", "—")
        sc4.metric("Sepsis-3 Risk", sep_risk)
        sc4.caption(f"Critical labs: {sep_d.get('critical_labs',0)} · "
                    f"Warning: {sep_d.get('warning_labs',0)}")
        if sep_d.get("abnormal_flags"):
            flags    = sep_d["abnormal_flags"][:4]
            flag_str = ", ".join(
                f.get("lab","") if isinstance(f, dict) else str(f)
                for f in flags)
            sc4.caption(f"Flags: {flag_str}")

        st.markdown("---")
        st.markdown("#### 🤖 ML Mortality Predictions")
        ml1, ml2, ml3 = st.columns(3)

        if cci_d:
            cci_sc = cci_d.get("cci_score")
            surv   = cci_d.get("predicted_10yr_survival")
            ml1.metric("Charlson CCI",
                       f"{cci_sc} ({cci_d.get('cci_risk','—')})"
                       if cci_sc is not None else "—")
            if surv:
                ml1.caption(f"10-yr survival: {round(surv*100,1)}%")
            if cci_d.get("conditions_present"):
                ml1.caption("Conditions: " + ", ".join(
                    cci_d["conditions_present"][:3]) +
                    ("…" if len(cci_d["conditions_present"]) > 3 else ""))

        mort = latest_v.get("mortality_risk_pct")
        xgb_sub = "Will this patient die this admission?<br><span style='color:#8b949e'>AUC 0.9467 · Chen & Guestrin 2016</span>"
        with ml2:
            st.html(get_large_risk_pill("🤖 XGBoost — Hospital Mortality", mort, xgb_sub))

        lstm_48h = latest_v.get("lstm_48h_risk_pct")
        lstm_sub = "Will this patient die in next 48 hours?<br><span style='color:#8b949e'>AUC 0.9469 · Hochreiter & Schmidhuber 1997</span>"
        with ml3:
            st.html(get_large_risk_pill("⏱️ LSTM — 48h Mortality Risk", lstm_48h, lstm_sub))

        top_factors = latest_v.get("mortality_top_factors", [])
        if top_factors:
            st.markdown("#### 🧠 SHAP — Top XGBoost Risk Factors")
            st.caption("Each factor shows its contribution to the "
                       "mortality prediction. Positive = increases risk.")
            shap_cols = st.columns(len(top_factors[:5]))
            for idx, factor in enumerate(top_factors[:5]):
                feat           = factor.get("feature","").replace("_"," ").title()
                impact         = factor.get("impact", 0)
                increases_risk = impact > 0
                shap_cols[idx].metric(
                    feat, f"{abs(round(impact, 3))}",
                    delta="↑ Increases risk" if increases_risk
                          else "↓ Decreases risk",
                    delta_color="inverse" if increases_risk else "normal"
                )
            st.caption("Reference: Lundberg & Lee. NeurIPS 2017.")

        st.markdown("---")
        st.markdown("#### 📈 Risk Trajectory Over Time")
        st.caption("XGBoost hospital mortality % and LSTM 48h risk % "
                   "tracked across readings for this patient")

        trajectory = list(
            db["vitals_log"]
            .find({"patient_id": sel_pid,
                   "mortality_risk_pct": {"$ne": None}},
                  {"_id": 0, "timestamp": 1, "mortality_risk_pct": 1,
                   "lstm_48h_risk_pct": 1, "overall_status": 1})
            .sort("timestamp", 1).limit(100)
        )

        if trajectory:
            tdf = pd.DataFrame(trajectory)
            tdf["timestamp"] = pd.to_datetime(
                tdf["timestamp"], errors="coerce")
            tdf = tdf.dropna(subset=["timestamp"]).sort_values("timestamp")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=tdf["timestamp"], y=tdf["mortality_risk_pct"],
                mode="lines+markers", name="🤖 XGBoost Hospital Mortality %",
                line=dict(color="#e63946", width=2), marker=dict(size=4),
            ))
            if "lstm_48h_risk_pct" in tdf.columns:
                if tdf["lstm_48h_risk_pct"].dropna().shape[0] > 0:
                    fig.add_trace(go.Scatter(
                        x=tdf["timestamp"], y=tdf["lstm_48h_risk_pct"],
                        mode="lines+markers", name="⏱️ LSTM 48h Risk %",
                        line=dict(color="#fd7e14", width=2, dash="dash"),
                        marker=dict(size=4),
                    ))
            fig.add_hline(y=30, line_dash="dot", line_color="#dc3545",
                          annotation_text="High risk (30%)",
                          annotation_position="top right")
            crit_pts = tdf[tdf["overall_status"] == "CRITICAL"]
            if len(crit_pts) > 0:
                fig.add_trace(go.Scatter(
                    x=crit_pts["timestamp"],
                    y=crit_pts["mortality_risk_pct"],
                    mode="markers", name="🔴 CRITICAL",
                    marker=dict(color="#dc3545", size=10, symbol="x"),
                ))
            fig.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=1.15),
                yaxis=dict(title="Risk %", range=[0, 100], gridcolor="#2a2d35"),
                xaxis=dict(title="Time", gridcolor="#2a2d35"),
                font=dict(size=11, color="#e0e6ed"),
            )
            st.plotly_chart(fig, use_container_width=True)
            c_count = (tdf["overall_status"] == "CRITICAL").sum()
            w_count = (tdf["overall_status"] == "WARNING").sum()
            st.caption(f"{c_count} CRITICAL · {w_count} WARNING "
                       f"out of {len(tdf)} total readings")
        else:
            st.info("No trajectory data yet — readings accumulate "
                    "as the pipeline runs.")

# ═══════════════════════════════════════════════════════════════
# TAB 3 — Alerts
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 🚨 Recent Alerts (last 30)")
    alerts = list(
        db["alert_history"]
        .find({}, {
            "_id": 0, "patient_id": 1, "overall_status": 1,
            "news2_score": 1, "qsofa_score": 1, "shock_index": 1,
            "heart_rate": 1, "systolic_bp": 1, "spo2": 1,
            "respiratory_rate": 1, "red_score_triggered": 1,
            "mortality_risk_pct": 1, "lstm_48h_risk_pct": 1,
            "timestamp": 1,
        })
        .sort("timestamp", -1).limit(30)
    )
    if alerts:
        df = pd.DataFrame(alerts)
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce"
        ).dt.strftime("%H:%M:%S")
        for col_name, dec in [("shock_index", 2),
                               ("mortality_risk_pct", 1),
                               ("lstm_48h_risk_pct", 1)]:
            if col_name in df.columns:
                df[col_name] = pd.to_numeric(
                    df[col_name], errors="coerce").round(dec)
        df = df.astype(str).replace("nan","—").replace("None","—")
        df.columns = [c.replace("_"," ").title() for c in df.columns]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No alerts yet.")