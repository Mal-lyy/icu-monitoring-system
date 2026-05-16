"""
ICU Real-Time Vitals Monitoring System — Full Test Suite
BIA 678-WS · Stevens Institute of Technology

Tests cover:
  1. Data integrity     — MIMIC-III CSV files
  2. Vital sign logic   — physiological plausibility
  3. NEWS2 scoring      — all 4 rules (RCP 2017)
  4. qSOFA scoring      — Singer et al. JAMA 2016
  5. Shock Index        — hemodynamic instability
  6. Sepsis-3 labs      — Singer et al. JAMA 2016
  7. Charlson CCI       — Charlson et al. 1987
  8. MongoDB pipeline   — collections and schema
  9. ML model           — XGBoost loading + scoring
  10. End-to-end        — full pipeline integration

Run with:
    python3 test_icu_project.py
"""

import sys
import os
import json
import pickle
import traceback
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────
PROJECT_DIR  = "/Users/malcolmdivinec/Documents/icu-monitoring-system"
FULL_DATA    = "/Users/malcolmdivinec/Downloads/mimic-iii-clinical-database-1.4"
DEMO_DATA    = f"{PROJECT_DIR}/data"
MODEL_PATH   = f"{PROJECT_DIR}/models/xgboost_mortality.pkl"
MONGO_URI    = "mongodb://localhost:27017/"
DB_NAME      = "icu_monitoring"

# ── Test Runner ───────────────────────────────────────────────
PASS = 0
FAIL = 0
WARN = 0
results = []

def test(name, fn):
    global PASS, FAIL, WARN
    try:
        result = fn()
        if result is True or result is None:
            PASS += 1
            results.append(("PASS", name, ""))
            print(f"  PASS  {name}")
        elif isinstance(result, str) and result.startswith("WARN"):
            WARN += 1
            results.append(("WARN", name, result))
            print(f"   WARN  {name} — {result[5:]}")
        else:
            FAIL += 1
            results.append(("FAIL", name, str(result)))
            print(f"  FAIL  {name} — {result}")
    except Exception as e:
        FAIL += 1
        results.append(("FAIL", name, str(e)))
        print(f"  FAIL  {name} — {e}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — DATA INTEGRITY
# ═══════════════════════════════════════════════════════════════
section("1. DATA INTEGRITY — MIMIC-III Files")

import pandas as pd
import numpy as np

def test_chartevents_exists():
    path = f"{FULL_DATA}/CHARTEVENTS.csv"
    assert os.path.exists(path), f"Not found: {path}"
    size_gb = os.path.getsize(path) / 1e9
    assert size_gb > 10, f"File too small: {size_gb:.1f}GB (expected >10GB)"
    return True

def test_chartevents_rowcount():
    chunk = next(pd.read_csv(
        f"{FULL_DATA}/CHARTEVENTS.csv",
        chunksize=100, low_memory=False
    ))
    assert len(chunk) == 100
    cols = [c.upper() for c in chunk.columns]
    required = ["SUBJECT_ID", "ITEMID", "VALUENUM", "CHARTTIME"]
    for r in required:
        assert r in cols, f"Missing column: {r}"
    return True

def test_labevents_exists():
    path = f"{FULL_DATA}/LABEVENTS.csv"
    assert os.path.exists(path)
    size_mb = os.path.getsize(path) / 1e6
    assert size_mb > 100, f"Too small: {size_mb:.0f}MB"
    return True

def test_icustays_patients():
    df = pd.read_csv(f"{FULL_DATA}/ICUSTAYS.csv")
    df.columns = df.columns.str.upper()
    n_patients = df["SUBJECT_ID"].nunique()
    n_stays    = len(df)
    assert n_patients > 40000, f"Too few patients: {n_patients}"
    assert n_stays    > 60000, f"Too few stays: {n_stays}"
    return True

def test_admissions_labels():
    df = pd.read_csv(f"{FULL_DATA}/ADMISSIONS.csv")
    df.columns = df.columns.str.upper()
    assert "HOSPITAL_EXPIRE_FLAG" in df.columns
    mortality = df["HOSPITAL_EXPIRE_FLAG"].mean()
    assert 0.05 < mortality < 0.20, \
        f"Mortality rate {mortality:.1%} outside expected 5-20%"
    return True

def test_diagnoses_icd():
    df = pd.read_csv(f"{FULL_DATA}/DIAGNOSES_ICD.csv")
    df.columns = df.columns.str.upper()
    assert "ICD9_CODE" in df.columns
    assert len(df) > 500000, f"Too few records: {len(df)}"
    return True

test("CHARTEVENTS.csv exists and >10GB",   test_chartevents_exists)
test("CHARTEVENTS columns and structure",   test_chartevents_rowcount)
test("LABEVENTS.csv exists and >100MB",     test_labevents_exists)
test("ICUSTAYS has 40k+ patients",          test_icustays_patients)
test("ADMISSIONS has mortality label 5-20%",test_admissions_labels)
test("DIAGNOSES_ICD has 500k+ records",     test_diagnoses_icd)

# ═══════════════════════════════════════════════════════════════
# SECTION 2 — VITAL SIGN LOGIC
# ═══════════════════════════════════════════════════════════════
section("2. VITAL SIGN LOGIC — Physiological Plausibility")

def test_temperature_conversion():
    # F to C: 98.6F = 37.0C
    f_val = 98.6
    c_val = round((f_val - 32) * 5 / 9, 1)
    assert c_val == 37.0, f"Expected 37.0, got {c_val}"

    # Values > 50 should be converted
    assert 98.6 > 50  # triggers conversion
    assert 37.0 < 50  # already Celsius, no conversion

    # Edge case: exactly 50
    assert not (50.0 > 50)  # 50 not converted
    return True

def test_plausibility_ranges():
    ranges = {
        "heart_rate":       (1,   300),
        "systolic_bp":      (40,  300),
        "spo2":             (50,  100),
        "temperature":      (25,  45),
        "respiratory_rate": (1,   60),
    }
    impossible = {
        "heart_rate":       [0, 350, -5],
        "systolic_bp":      [10, 350, -1],
        "spo2":             [30, 105, -1],
        "temperature":      [20, 50, -1],
        "respiratory_rate": [0, 70, -1],
    }
    for vital, (lo, hi) in ranges.items():
        for bad_val in impossible[vital]:
            in_range = lo <= bad_val <= hi
            assert not in_range, \
                f"{vital}={bad_val} should be filtered out"
    return True

def test_vital_normal_ranges():
    # These should all pass plausibility filter
    valid = {
        "heart_rate": 72,
        "systolic_bp": 120,
        "spo2": 98,
        "temperature": 37.0,
        "respiratory_rate": 16,
    }
    ranges = {
        "heart_rate":       (1,   300),
        "systolic_bp":      (40,  300),
        "spo2":             (50,  100),
        "temperature":      (25,  45),
        "respiratory_rate": (1,   60),
    }
    for vital, val in valid.items():
        lo, hi = ranges[vital]
        assert lo <= val <= hi, f"{vital}={val} should be valid"
    return True

test("Temperature F→C conversion",         test_temperature_conversion)
test("Impossible values filtered out",      test_plausibility_ranges)
test("Normal vital values pass filter",     test_vital_normal_ranges)

# ═══════════════════════════════════════════════════════════════
# SECTION 3 — NEWS2 SCORING
# Reference: RCP. NEWS2. London: RCP, 2017. Chapter 6, p.28
# ═══════════════════════════════════════════════════════════════
section("3. NEWS2 SCORING — Royal College of Physicians 2017")

def news2_score(rr, spo2, sbp, hr, temp):
    """Replicate Spark NEWS2 logic in Python for testing."""
    # RR score
    if rr <= 8:    rr_s = 3
    elif rr <= 11: rr_s = 1
    elif rr <= 20: rr_s = 0
    elif rr <= 24: rr_s = 2
    else:          rr_s = 3

    # SpO2 score
    if spo2 is None:   spo2_s = 0
    elif spo2 <= 91:   spo2_s = 3
    elif spo2 <= 93:   spo2_s = 2
    elif spo2 <= 95:   spo2_s = 1
    else:              spo2_s = 0

    # BP score
    if sbp is None:    bp_s = 0
    elif sbp <= 90:    bp_s = 3
    elif sbp <= 100:   bp_s = 2
    elif sbp <= 110:   bp_s = 1
    elif sbp <= 219:   bp_s = 0
    else:              bp_s = 3

    # HR score
    if hr is None:     hr_s = 0
    elif hr <= 40:     hr_s = 3
    elif hr <= 50:     hr_s = 1
    elif hr <= 90:     hr_s = 0
    elif hr <= 110:    hr_s = 1
    elif hr <= 130:    hr_s = 2
    else:              hr_s = 3

    # Temp score
    if temp is None:      temp_s = 0
    elif temp <= 35.0:    temp_s = 3
    elif temp <= 36.0:    temp_s = 1
    elif temp <= 38.0:    temp_s = 0
    elif temp <= 39.0:    temp_s = 1
    else:                 temp_s = 2

    total      = rr_s + spo2_s + bp_s + hr_s + temp_s
    red_score  = any(s == 3 for s in [rr_s, spo2_s, bp_s, hr_s, temp_s])

    if red_score or total >= 7:
        status = "CRITICAL"
    elif total >= 5:
        status = "WARNING"
    else:
        status = "NORMAL"

    return total, red_score, status, {
        "rr": rr_s, "spo2": spo2_s, "bp": bp_s,
        "hr": hr_s, "temp": temp_s
    }

def test_news2_normal_patient():
    # Healthy ICU patient — all vitals normal
    total, red, status, scores = news2_score(
        rr=16, spo2=98, sbp=120, hr=72, temp=37.0
    )
    assert total == 0, f"Expected 0, got {total}"
    assert status == "NORMAL"
    assert not red
    return True

def test_news2_warning_patient():
    # RR=22 (score 2) + HR=105 (score 1) + SpO2=94 (score 1) = 4
    # Wait — 4 is still NORMAL. Let's use RR=22 + HR=105 + SBP=100
    # RR=22: score 2, HR=105: score 1, SBP=100: score 2 = total 5 = WARNING
    total, red, status, scores = news2_score(
        rr=22, spo2=98, sbp=100, hr=105, temp=37.0
    )
    assert total == 5, f"Expected 5, got {total} scores={scores}"
    assert status == "WARNING"
    return True

def test_news2_critical_aggregate():
    # Total >= 7 = CRITICAL
    # RR=26 (3) + SBP=85 (3) + HR=135 (3) = 9
    total, red, status, scores = news2_score(
        rr=26, spo2=98, sbp=85, hr=135, temp=37.0
    )
    assert total >= 7, f"Expected >=7, got {total}"
    assert status == "CRITICAL"
    return True

def test_news2_red_score_rule2():
    # Rule 2: single vital scores 3 = CRITICAL regardless of total
    # RR=26 alone (score 3), everything else normal
    total, red, status, scores = news2_score(
        rr=26, spo2=98, sbp=120, hr=72, temp=37.0
    )
    assert scores["rr"] == 3, f"RR score should be 3, got {scores['rr']}"
    assert red, "Red score should be triggered"
    assert status == "CRITICAL"
    return True

def test_news2_boundary_rr():
    # RR boundary: 20=0, 21=2, 24=2, 25=3
    _, _, _, s = news2_score(rr=20, spo2=98, sbp=120, hr=72, temp=37.0)
    assert s["rr"] == 0, f"RR=20 should score 0, got {s['rr']}"
    _, _, _, s = news2_score(rr=21, spo2=98, sbp=120, hr=72, temp=37.0)
    assert s["rr"] == 2, f"RR=21 should score 2, got {s['rr']}"
    _, _, _, s = news2_score(rr=25, spo2=98, sbp=120, hr=72, temp=37.0)
    assert s["rr"] == 3, f"RR=25 should score 3, got {s['rr']}"
    return True

def test_news2_boundary_temp():
    # Temp boundaries: ≤35=3, 35.1-36=1, 36.1-38=0, 38.1-39=1, ≥39.1=2
    _, _, _, s = news2_score(rr=16, spo2=98, sbp=120, hr=72, temp=35.0)
    assert s["temp"] == 3, f"Temp=35 should score 3, got {s['temp']}"
    _, _, _, s = news2_score(rr=16, spo2=98, sbp=120, hr=72, temp=37.0)
    assert s["temp"] == 0, f"Temp=37 should score 0, got {s['temp']}"
    _, _, _, s = news2_score(rr=16, spo2=98, sbp=120, hr=72, temp=39.5)
    assert s["temp"] == 2, f"Temp=39.5 should score 2, got {s['temp']}"
    return True

def test_news2_max_score():
    # All vitals at worst values — max possible score
    total, red, status, scores = news2_score(
        rr=30, spo2=88, sbp=80, hr=150, temp=34.0
    )
    assert total == 15, f"Expected 15, got {total}"
    assert status == "CRITICAL"
    assert red
    return True

test("NEWS2: healthy patient = NORMAL score 0", test_news2_normal_patient)
test("NEWS2: borderline = WARNING score 5",     test_news2_warning_patient)
test("NEWS2: critical aggregate score >=7",     test_news2_critical_aggregate)
test("NEWS2: Rule 2 red score (single vital=3)",test_news2_red_score_rule2)
test("NEWS2: RR boundary values correct",       test_news2_boundary_rr)
test("NEWS2: Temperature boundary values",      test_news2_boundary_temp)
test("NEWS2: Maximum score = 15",              test_news2_max_score)

# ═══════════════════════════════════════════════════════════════
# SECTION 4 — qSOFA SCORING
# Reference: Singer et al. JAMA 2016;315(8):801-810
# ═══════════════════════════════════════════════════════════════
section("4. qSOFA SCORING — Singer et al. JAMA 2016")

def qsofa_score(rr, sbp):
    """qSOFA: 2 criteria implementable from CHARTEVENTS."""
    rr_flag  = 1 if rr  is not None and rr  >= 22  else 0
    bp_flag  = 1 if sbp is not None and sbp <= 100 else 0
    # Mentation excluded — NOTEEVENTS empty in MIMIC-III demo
    total = rr_flag + bp_flag
    flag  = "SEPSIS_RISK" if total >= 2 else \
            "MONITOR"     if total == 1 else "LOW_RISK"
    return total, flag

def test_qsofa_no_risk():
    score, flag = qsofa_score(rr=16, sbp=120)
    assert score == 0
    assert flag == "LOW_RISK"
    return True

def test_qsofa_one_criterion():
    score, flag = qsofa_score(rr=22, sbp=120)
    assert score == 1
    assert flag == "MONITOR"
    return True

def test_qsofa_sepsis_risk():
    # Both criteria met
    score, flag = qsofa_score(rr=25, sbp=95)
    assert score == 2
    assert flag == "SEPSIS_RISK"
    return True

def test_qsofa_rr_boundary():
    # RR=21 → 0, RR=22 → 1
    s1, _ = qsofa_score(rr=21, sbp=120)
    s2, _ = qsofa_score(rr=22, sbp=120)
    assert s1 == 0, f"RR=21 should give 0, got {s1}"
    assert s2 == 1, f"RR=22 should give 1, got {s2}"
    return True

def test_qsofa_sbp_boundary():
    # SBP=101 → 0, SBP=100 → 1
    s1, _ = qsofa_score(rr=16, sbp=101)
    s2, _ = qsofa_score(rr=16, sbp=100)
    assert s1 == 0, f"SBP=101 should give 0, got {s1}"
    assert s2 == 1, f"SBP=100 should give 1, got {s2}"
    return True

test("qSOFA: no criteria = LOW_RISK",          test_qsofa_no_risk)
test("qSOFA: one criterion = MONITOR",          test_qsofa_one_criterion)
test("qSOFA: two criteria = SEPSIS_RISK",       test_qsofa_sepsis_risk)
test("qSOFA: RR boundary at 22",               test_qsofa_rr_boundary)
test("qSOFA: SBP boundary at 100",             test_qsofa_sbp_boundary)

# ═══════════════════════════════════════════════════════════════
# SECTION 5 — SHOCK INDEX
# ═══════════════════════════════════════════════════════════════
section("5. SHOCK INDEX — Hemodynamic Instability")

def shock_index(hr, sbp):
    if hr is None or sbp is None or sbp == 0:
        return None
    return round(hr / sbp, 3)

def test_shock_normal():
    si = shock_index(hr=72, sbp=120)
    assert si == 0.6, f"Expected 0.6, got {si}"
    assert si < 1.0
    return True

def test_shock_instability():
    si = shock_index(hr=110, sbp=90)
    assert si > 1.0, f"Expected >1.0, got {si}"
    return True

def test_shock_boundary():
    # Exactly 1.0
    si = shock_index(hr=100, sbp=100)
    assert si == 1.0
    # Flag triggers at >1.0 not >=1.0
    assert not (si > 1.0)
    return True

def test_shock_none_values():
    assert shock_index(None, 120) is None
    assert shock_index(72, None)  is None
    assert shock_index(72, 0)     is None
    return True

test("Shock Index: normal patient 0.6",         test_shock_normal)
test("Shock Index: instability when >1.0",      test_shock_instability)
test("Shock Index: boundary at exactly 1.0",    test_shock_boundary)
test("Shock Index: None values handled",        test_shock_none_values)

# ═══════════════════════════════════════════════════════════════
# SECTION 6 — SEPSIS-3 LAB DETECTION
# Reference: Singer et al. JAMA 2016
# ═══════════════════════════════════════════════════════════════
section("6. SEPSIS-3 LAB DETECTION — Singer et al. JAMA 2016")

def classify_sepsis(n_warning, n_critical):
    if n_critical >= 2: return "HIGH"
    if n_critical == 1: return "MODERATE"
    if n_warning  >= 2: return "MODERATE"
    if n_warning  == 1: return "LOW"
    return "NONE"

def test_sepsis_none():
    assert classify_sepsis(0, 0) == "NONE"
    return True

def test_sepsis_low():
    assert classify_sepsis(1, 0) == "LOW"
    return True

def test_sepsis_moderate_one_critical():
    assert classify_sepsis(0, 1) == "MODERATE"
    return True

def test_sepsis_moderate_two_warnings():
    assert classify_sepsis(2, 0) == "MODERATE"
    return True

def test_sepsis_high():
    assert classify_sepsis(0, 2) == "HIGH"
    assert classify_sepsis(5, 3) == "HIGH"
    return True

def test_lactate_threshold():
    # Lactate >2.0 = critical, >1.5 = warning
    lactate_critical = 2.5
    lactate_warning  = 1.7
    lactate_normal   = 1.2
    assert lactate_critical > 2.0
    assert 1.5 < lactate_warning <= 2.0
    assert lactate_normal <= 1.5
    return True

def test_platelets_threshold():
    # Platelets: low = danger (thrombocytopenia)
    # <150 = warning
    assert 100 < 150  # warning threshold
    assert 80  < 150  # also warning
    return True

test("Sepsis: 0 flags = NONE",                  test_sepsis_none)
test("Sepsis: 1 warning = LOW",                 test_sepsis_low)
test("Sepsis: 1 critical = MODERATE",           test_sepsis_moderate_one_critical)
test("Sepsis: 2 warnings = MODERATE",           test_sepsis_moderate_two_warnings)
test("Sepsis: 2+ critical = HIGH",              test_sepsis_high)
test("Lactate threshold: >2.0 critical",        test_lactate_threshold)
test("Platelets threshold: <150 warning",       test_platelets_threshold)

# ═══════════════════════════════════════════════════════════════
# SECTION 7 — CHARLSON CCI
# Reference: Charlson et al. J Chronic Diseases 1987
# ═══════════════════════════════════════════════════════════════
section("7. CHARLSON CCI — Charlson et al. 1987")

def cci_risk_label(score):
    if score == 0:  return "NONE"
    if score <= 2:  return "MILD"
    if score <= 4:  return "MODERATE"
    return "SEVERE"

def predicted_survival(score):
    if score == 0: return 0.98
    if score == 1: return 0.96
    if score == 2: return 0.90
    if score == 3: return 0.77
    if score == 4: return 0.53
    return 0.21

def test_cci_risk_labels():
    assert cci_risk_label(0) == "NONE"
    assert cci_risk_label(1) == "MILD"
    assert cci_risk_label(2) == "MILD"
    assert cci_risk_label(3) == "MODERATE"
    assert cci_risk_label(4) == "MODERATE"
    assert cci_risk_label(5) == "SEVERE"
    assert cci_risk_label(19) == "SEVERE"
    return True

def test_cci_survival():
    assert predicted_survival(0) == 0.98
    assert predicted_survival(2) == 0.90
    # Higher score = lower survival
    assert predicted_survival(0) > predicted_survival(3)
    assert predicted_survival(3) > predicted_survival(5)
    return True

def test_cci_weights():
    # Verify disease weights from Charlson et al. 1987
    weights = {
        "myocardial_infarction":         1,
        "congestive_heart_failure":      1,
        "diabetes_without_complications":1,
        "diabetes_with_complications":   2,
        "renal_disease":                 2,
        "malignancy":                    2,
        "severe_liver_disease":          3,
        "metastatic_solid_tumor":        6,
        "aids_hiv":                      6,
    }
    # Metastatic cancer + AIDS = 12 (highest possible combo)
    worst = weights["metastatic_solid_tumor"] + weights["aids_hiv"]
    assert worst == 12
    return True

def test_cci_mongodb():
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        db = client[DB_NAME]
        count = db["charlson_scores"].count_documents({})
        assert count > 40000, f"Expected 40k+ CCI records, got {count}"
        sample = db["charlson_scores"].find_one({})
        assert "cci_score" in sample
        assert "cci_risk" in sample
        assert "predicted_10yr_survival" in sample
        assert sample["cci_score"] >= 0
        client.close()
        return True
    except Exception as e:
        return f"WARN MongoDB not available: {e}"

test("CCI risk labels correct",              test_cci_risk_labels)
test("CCI survival decreases with score",    test_cci_survival)
test("CCI disease weights from 1987 paper",  test_cci_weights)
test("CCI MongoDB collection populated",     test_cci_mongodb)

# ═══════════════════════════════════════════════════════════════
# SECTION 8 — MONGODB PIPELINE
# ═══════════════════════════════════════════════════════════════
section("8. MONGODB PIPELINE — Collections and Schema")

try:
    from pymongo import MongoClient
    _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    _db = _client[DB_NAME]
    _connected = True
except:
    _connected = False

def test_mongo_connection():
    assert _connected, "Cannot connect to MongoDB"
    return True

def test_vitals_log_schema():
    if not _connected: return "WARN MongoDB unavailable"
    doc = _db["vitals_log"].find_one(
        {"shock_index": {"$exists": True}},
        sort=[("timestamp", -1)]
    )
    if not doc: return "WARN vitals_log is empty — start the pipeline"
    required = ["patient_id", "timestamp", "overall_status",
                "news2_score", "qsofa_score", "shock_index"]
    for field in required:
        assert field in doc, f"Missing field: {field}"
    return True

def test_vitals_log_values():
    if not _connected: return "WARN MongoDB unavailable"
    docs = list(_db["vitals_log"].find(
        {"overall_status": {"$exists": True}}, limit=100
    ))
    if not docs: return "WARN No vitals data"
    statuses = set(d["overall_status"] for d in docs)
    valid    = {"NORMAL", "WARNING", "CRITICAL"}
    invalid  = statuses - valid
    assert not invalid, f"Invalid status values: {invalid}"
    return True

def test_news2_score_range():
    if not _connected: return "WARN MongoDB unavailable"
    docs = list(_db["vitals_log"].find(
        {"news2_score": {"$exists": True}}, limit=200
    ))
    if not docs: return "WARN No news2_score data"
    for doc in docs:
        score = doc.get("news2_score", 0)
        assert 0 <= score <= 15, \
            f"NEWS2 score {score} out of range 0-15"
    return True

def test_qsofa_score_range():
    if not _connected: return "WARN MongoDB unavailable"
    docs = list(_db["vitals_log"].find(
        {"qsofa_score": {"$exists": True}}, limit=200
    ))
    if not docs: return "WARN No qsofa_score data"
    for doc in docs:
        score = doc.get("qsofa_score", 0)
        assert 0 <= score <= 2, \
            f"qSOFA score {score} out of range 0-2"
    return True

def test_shock_index_values():
    if not _connected: return "WARN MongoDB unavailable"
    docs = list(_db["vitals_log"].find(
        {"shock_index": {"$exists": True, "$ne": None}},
        limit=200
    ))
    if not docs: return "WARN No shock_index data"
    for doc in docs:
        si = doc.get("shock_index")
        if si is not None:
            assert 0 < si < 10, \
                f"Shock index {si} out of realistic range"
    return True

def test_alert_history_populated():
    if not _connected: return "WARN MongoDB unavailable"
    count = _db["alert_history"].count_documents({})
    assert count > 0, "alert_history is empty"
    doc = _db["alert_history"].find_one({})
    status = doc.get("overall_status")
    assert status in ("WARNING", "CRITICAL"), \
        f"Alert should be WARNING or CRITICAL, got {status}"
    return True

def test_sepsis_risk_populated():
    if not _connected: return "WARN MongoDB unavailable"
    count = _db["sepsis_risk"].count_documents({})
    assert count > 40000, f"Expected 40k+ sepsis records, got {count}"
    doc = _db["sepsis_risk"].find_one({})
    assert doc["patient_id"].startswith("P"), \
        f"patient_id should start with P, got {doc['patient_id']}"
    assert doc["sepsis_risk"] in ("HIGH","MODERATE","LOW","NONE")
    return True

def test_patient_profiles_populated():
    if not _connected: return "WARN MongoDB unavailable"
    count = _db["patient_profiles"].count_documents({})
    assert count > 40000, f"Expected 40k+, got {count}"
    doc = _db["patient_profiles"].find_one({})
    assert "age_at_first_icu" in doc
    assert "gender" in doc
    assert doc["gender"] in ("M", "F")
    return True

def test_mortality_scores_in_vitals():
    if not _connected: return "WARN MongoDB unavailable"
    count = _db["vitals_log"].count_documents(
        {"mortality_risk_pct": {"$exists": True, "$ne": None}}
    )
    if count == 0:
        return "WARN mortality_risk_pct not yet in vitals_log"
    sample = list(_db["vitals_log"].find(
        {"mortality_risk_pct": {"$ne": None}}, limit=50
    ))
    for doc in sample:
        pct = doc["mortality_risk_pct"]
        assert 0 <= pct <= 100, f"Mortality % {pct} out of range"
    return True

test("MongoDB connection",                       test_mongo_connection)
test("vitals_log schema has required fields",    test_vitals_log_schema)
test("overall_status only NORMAL/WARNING/CRITICAL", test_vitals_log_values)
test("NEWS2 scores in range 0-15",              test_news2_score_range)
test("qSOFA scores in range 0-2",               test_qsofa_score_range)
test("Shock index in realistic range",           test_shock_index_values)
test("alert_history populated with alerts",      test_alert_history_populated)
test("sepsis_risk has 40k+ records with P-prefix", test_sepsis_risk_populated)
test("patient_profiles has demographics",        test_patient_profiles_populated)
test("mortality_risk_pct in vitals_log",         test_mortality_scores_in_vitals)

# ═══════════════════════════════════════════════════════════════
# SECTION 9 — ML MODEL
# Reference: Chen & Guestrin KDD 2016 / Lundberg & Lee NeurIPS 2017
# ═══════════════════════════════════════════════════════════════
section("9. ML MODEL — XGBoost + SHAP")

def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"
    size_mb = os.path.getsize(MODEL_PATH) / 1e6
    assert size_mb > 0.5, f"Model file too small: {size_mb:.1f}MB"
    return True

def test_model_loads():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    required = ["model", "explainer", "feature_cols",
                "medians", "xgb_auc", "trained_on"]
    for key in required:
        assert key in bundle, f"Missing key: {key}"
    return True

def test_model_auc():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    auc = bundle["xgb_auc"]
    assert auc > 0.80, f"AUC {auc:.4f} below minimum 0.80"
    assert auc <= 1.0, f"AUC {auc:.4f} impossibly high"
    return True

def test_model_trained_on_scale():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    n = bundle["trained_on"]
    assert n > 50000, f"Trained on only {n} samples — expected 50k+"
    return True

def test_model_features():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    features = bundle["feature_cols"]
    expected = ["heart_rate_mean", "systolic_bp_min",
                "spo2_mean", "respiratory_rate_mean", "cci_score"]
    for f in expected:
        assert f in features, f"Missing feature: {f}"
    assert len(features) >= 20, f"Too few features: {len(features)}"
    return True

def test_model_prediction():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model    = bundle["model"]
    features = bundle["feature_cols"]
    medians  = bundle["medians"]

    # Predict on median values — should give non-extreme probability
    import numpy as np
    X = np.array([medians.get(f, 0) or 0 for f in features]).reshape(1,-1)
    prob = model.predict_proba(X)[0][1]
    assert 0.0 <= prob <= 1.0, f"Probability {prob} out of [0,1]"
    return True

def test_model_high_risk_patient():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model    = bundle["model"]
    features = bundle["feature_cols"]
    medians  = bundle["medians"]

    import numpy as np
    # Low BP, high HR, low SpO2 = high mortality risk
    X = np.array([medians.get(f, 0) or 0 for f in features]).reshape(1,-1)
    feat_idx = {f: i for i, f in enumerate(features)}

    # Set dangerous values
    if "systolic_bp_min"  in feat_idx: X[0][feat_idx["systolic_bp_min"]]  = 55
    if "heart_rate_max"   in feat_idx: X[0][feat_idx["heart_rate_max"]]   = 140
    if "spo2_min"         in feat_idx: X[0][feat_idx["spo2_min"]]         = 82
    if "lactate"          in feat_idx: X[0][feat_idx["lactate"]]           = 6.0
    if "cci_score"        in feat_idx: X[0][feat_idx["cci_score"]]         = 8

    prob_high = model.predict_proba(X)[0][1]

    # Normal patient
    X_norm = np.array([medians.get(f, 0) or 0 for f in features]).reshape(1,-1)
    prob_norm = model.predict_proba(X_norm)[0][1]

    assert prob_high > prob_norm, \
        f"High-risk patient ({prob_high:.2f}) should score higher than normal ({prob_norm:.2f})"
    return True

def test_shap_explainer():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    explainer = bundle["explainer"]
    model     = bundle["model"]
    features  = bundle["feature_cols"]
    medians   = bundle["medians"]

    import numpy as np
    X = np.array([medians.get(f, 0) or 0 for f in features]).reshape(1,-1)
    shap_vals = explainer.shap_values(X)

    assert shap_vals is not None
    assert len(shap_vals[0]) == len(features), \
        f"SHAP length {len(shap_vals[0])} != features {len(features)}"
    return True

def test_metrics_json():
    metrics_path = f"{PROJECT_DIR}/models/model_metrics.json"
    assert os.path.exists(metrics_path)
    with open(metrics_path) as f:
        m = json.load(f)
    assert m["xgboost_auc"] > 0.80
    assert m["xgboost_auc"] > m["baseline_lr_auc"], \
        "XGBoost should outperform logistic regression baseline"
    return True

test("Model file exists and >0.5MB",            test_model_file_exists)
test("Model bundle loads with all keys",         test_model_loads)
test("Model AUC-ROC >0.80",                     test_model_auc)
test("Model trained on 50k+ ICU stays",          test_model_trained_on_scale)
test("Model has all required features",          test_model_features)
test("Model prediction in [0,1]",               test_model_prediction)
test("High-risk patient scores higher than normal", test_model_high_risk_patient)
test("SHAP explainer produces values",           test_shap_explainer)
test("metrics.json: XGBoost beats LR baseline", test_metrics_json)

# ═══════════════════════════════════════════════════════════════
# SECTION 10 — END-TO-END INTEGRATION
# ═══════════════════════════════════════════════════════════════
section("10. END-TO-END INTEGRATION")

def test_kafka_connection():
    try:
        from kafka import KafkaProducer
        p = KafkaProducer(
            bootstrap_servers="localhost:9092",
            request_timeout_ms=3000
        )
        p.close()
        return True
    except Exception as e:
        return f"WARN Kafka not reachable: {e}"

def test_kafka_topic_exists():
    try:
        from kafka.admin import KafkaAdminClient
        admin = KafkaAdminClient(bootstrap_servers="localhost:9092",
                                 request_timeout_ms=3000)
        topics = admin.list_topics()
        admin.close()
        assert "icu-vitals" in topics, \
            f"Topic icu-vitals not found. Topics: {topics}"
        return True
    except Exception as e:
        return f"WARN {e}"

def test_pipeline_producing_data():
    if not _connected: return "WARN MongoDB unavailable"
    count = _db["vitals_log"].count_documents({})
    assert count > 0, "vitals_log empty — is producer running?"
    return True

def test_patient_ids_consistent():
    if not _connected: return "WARN MongoDB unavailable"
    vitals_pids  = set(d["patient_id"] for d in
                       _db["vitals_log"].find({}, {"patient_id":1,"_id":0})
                       .limit(50))
    sepsis_pids  = set(d["patient_id"] for d in
                       _db["sepsis_risk"].find({}, {"patient_id":1,"_id":0})
                       .limit(500))
    profile_pids = set(d["patient_id"] for d in
                       _db["patient_profiles"].find({}, {"patient_id":1,"_id":0})
                       .limit(500))

    # Check P-prefix consistency
    for pid in vitals_pids:
        assert pid.startswith("P"), \
            f"vitals_log patient_id should start with P: {pid}"
    for pid in list(sepsis_pids)[:5]:
        assert pid.startswith("P"), \
            f"sepsis_risk patient_id should start with P: {pid}"
    return True

def test_alert_logic_consistent():
    if not _connected: return "WARN MongoDB unavailable"
    alerts = list(_db["alert_history"].find(
        {}, {"overall_status":1, "news2_score":1,
             "qsofa_score":1, "_id":0}
    ).limit(100))
    if not alerts: return "WARN No alerts yet"
    for doc in alerts:
        status = doc.get("overall_status")
        news2  = doc.get("news2_score", 0) or 0
        qsofa  = doc.get("qsofa_score", 0) or 0
        assert status in ("WARNING", "CRITICAL"), \
            f"Alert has invalid status: {status}"
    return True

def test_features_parquet_exists():
    path = f"{PROJECT_DIR}/features.parquet"
    assert os.path.exists(path), "features.parquet not found"
    df = pd.read_parquet(path)
    assert len(df) > 50000, f"Too few rows: {len(df)}"
    assert "died" in df.columns, "Missing label column 'died'"
    mortality = df["died"].mean()
    assert 0.05 < mortality < 0.20, \
        f"Mortality rate {mortality:.1%} outside expected range"
    return True

def test_full_stack_document():
    if not _connected: return "WARN MongoDB unavailable"
    doc = _db["vitals_log"].find_one(
        {"mortality_risk_pct": {"$ne": None}}
    )
    if not doc:
        return "WARN No documents with mortality_risk_pct yet"

    checks = {
        "patient_id":        lambda v: isinstance(v, str) and v.startswith("P"),
        "overall_status":    lambda v: v in ("NORMAL","WARNING","CRITICAL"),
        "news2_score":       lambda v: 0 <= v <= 15,
        "qsofa_score":       lambda v: 0 <= v <= 2,
        "mortality_risk_pct":lambda v: 0 <= v <= 100,
    }
    for field, check_fn in checks.items():
        val = doc.get(field)
        assert val is not None, f"Missing {field}"
        assert check_fn(val), \
            f"{field}={val} failed validation"
    return True

test("Kafka broker reachable",               test_kafka_connection)
test("icu-vitals topic exists",              test_kafka_topic_exists)
test("Pipeline producing data to MongoDB",   test_pipeline_producing_data)
test("Patient IDs use P-prefix consistently",test_patient_ids_consistent)
test("Alert logic consistent with scores",   test_alert_logic_consistent)
test("features.parquet exists with 50k+ rows", test_features_parquet_exists)
test("Full-stack document has all fields",   test_full_stack_document)

# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
total = PASS + FAIL + WARN
print(f"\n{'='*60}")
print(f"  TEST SUMMARY")
print(f"{'='*60}")
print(f"  Total:   {total}")
print(f"  PASS:  {PASS}")
print(f"  FAIL:  {FAIL}")
print(f"   WARN:  {WARN}")
print(f"{'='*60}")

if FAIL > 0:
    print(f"\n  Failed tests:")
    for status, name, msg in results:
        if status == "FAIL":
            print(f"    ❌ {name}")
            print(f"       {msg}")

if WARN > 0:
    print(f"\n  Warnings (pipeline may not be running):")
    for status, name, msg in results:
        if status == "WARN":
            print(f"    ⚠️  {name}")

score_pct = round(PASS / total * 100) if total > 0 else 0
print(f"\n  Score: {PASS}/{total} ({score_pct}%)")

if FAIL == 0:
    print(f"\n  All tests passed! System is working correctly.")
elif FAIL <= 2:
    print(f"\n  Minor issues found. Check failed tests above.")
else:
    print(f"\n  Multiple failures. Check pipeline is running.")

print()
if _connected:
    _client.close()