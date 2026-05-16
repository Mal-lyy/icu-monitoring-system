import pandas as pd
import numpy as np
from pymongo import MongoClient
import pickle
import os

# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  — Improved for XGBoost + LSTM
# ═══════════════════════════════════════════════════════════════

DATA_DIR    = "/Users/malcolmdivinec/Downloads/mimic-iii-clinical-database-1.4"
CHART_PATH  = f"{DATA_DIR}/CHARTEVENTS.csv"
LAB_PATH    = f"{DATA_DIR}/LABEVENTS.csv"
ADM_PATH    = f"{DATA_DIR}/ADMISSIONS.csv"
ICU_PATH    = f"{DATA_DIR}/ICUSTAYS.csv"
OUT_PATH    = "/Users/malcolmdivinec/Documents/icu-monitoring-system/features.parquet"
CKPT_PATH   = "/Users/malcolmdivinec/Documents/icu-monitoring-system/stay_vitals.pkl"
MONGO_URI   = "mongodb://localhost:27017/"
CHUNK_SIZE  = 500000

VITAL_ITEMS = {
    211:    'heart_rate',    220045: 'heart_rate',
    51:     'systolic_bp',   442:    'systolic_bp',
    455:    'systolic_bp',   6701:   'systolic_bp',
    220179: 'systolic_bp',   220050: 'systolic_bp',
    646:    'spo2',          220277: 'spo2',
    678:    'temperature',   679:    'temperature',
    223761: 'temperature',   224642: 'temperature',
    618:    'respiratory_rate', 615: 'respiratory_rate',
    220210: 'respiratory_rate', 224690: 'respiratory_rate'
}

VITAL_RANGES = {
    'heart_rate':       (1,   300),
    'systolic_bp':      (40,  300),
    'spo2':             (50,  100),
    'temperature':      (25,  45),
    'respiratory_rate': (1,   60),
}

LAB_ITEMS = {
    50813: 'lactate',
    51301: 'wbc',
    50912: 'creatinine',
    51265: 'platelets',
    50820: 'ph',
}

print("=" * 60)
print("FEATURE ENGINEERING — XGBoost Mortality Predictor")
print("Reference: Liu et al. IEEE ICDM 2008 / Chen & Guestrin KDD 2016")
print("=" * 60)

# ── Load Reference Tables ─────────────────────────────────────
print("\nLoading reference tables...")
icustays   = pd.read_csv(ICU_PATH,  low_memory=False)
admissions = pd.read_csv(ADM_PATH,  low_memory=False)
icustays.columns   = icustays.columns.str.upper()
admissions.columns = admissions.columns.str.upper()

icustays["INTIME"]  = pd.to_datetime(icustays["INTIME"],  errors="coerce")
icustays["OUTTIME"] = pd.to_datetime(icustays["OUTTIME"], errors="coerce")
admissions["ADMITTIME"]   = pd.to_datetime(admissions["ADMITTIME"],   errors="coerce")
admissions["DISCHTIME"]   = pd.to_datetime(admissions["DISCHTIME"],   errors="coerce")
admissions["DEATHTIME"]   = pd.to_datetime(admissions["DEATHTIME"],   errors="coerce")

valid_stays = set(icustays["ICUSTAY_ID"].dropna().astype(int))
valid_sids  = set(icustays["SUBJECT_ID"].dropna().astype(int))
print(f"ICU stays: {len(valid_stays):,} | Patients: {len(valid_sids):,}")

# ── Build 48-hour Mortality Label ─────────────────────────────
# A patient "died within 48h of ICU admission" if DEATHTIME
# is within 48 hours of their ICU INTIME
print("Building 48-hour mortality label...")
icu_adm = icustays[["ICUSTAY_ID","SUBJECT_ID","HADM_ID","INTIME"]].merge(
    admissions[["HADM_ID","DEATHTIME","HOSPITAL_EXPIRE_FLAG"]],
    on="HADM_ID", how="left"
)
icu_adm["died_48h"] = (
    icu_adm["DEATHTIME"].notna() &
    ((icu_adm["DEATHTIME"] - icu_adm["INTIME"]).dt.total_seconds() <= 48*3600) &
    ((icu_adm["DEATHTIME"] - icu_adm["INTIME"]).dt.total_seconds() >= 0)
).astype(int)
icu_adm["died_hospital"] = icu_adm["HOSPITAL_EXPIRE_FLAG"].fillna(0).astype(int)

died_48h_map      = dict(zip(icu_adm["ICUSTAY_ID"], icu_adm["died_48h"]))
died_hospital_map = dict(zip(icu_adm["ICUSTAY_ID"], icu_adm["died_hospital"]))

n_48h = sum(died_48h_map.values())
print(f"48h mortality: {n_48h:,} ({n_48h/len(died_48h_map)*100:.1f}%)")

# ── CCI from MongoDB ──────────────────────────────────────────
print("Loading CCI scores...")
client  = MongoClient(MONGO_URI)
db      = client["icu_monitoring"]
cci_map = {
    r["patient_id"].lstrip("P"): r.get("cci_score", 0)
    for r in db["charlson_scores"].find({}, {"_id":0,
        "patient_id":1, "cci_score":1})
}
client.close()
print(f"CCI loaded: {len(cci_map):,}")

# ── Load CHARTEVENTS (use checkpoint if available) ────────────
if os.path.exists(CKPT_PATH):
    print(f"\nLoading stay_vitals from checkpoint...")
    with open(CKPT_PATH, "rb") as f:
        stay_vitals = pickle.load(f)
    print(f"Loaded {len(set(k[0] for k in stay_vitals)):,} stays")
else:
    print("\nAggregating CHARTEVENTS (this will take hours)...")
    stay_vitals  = {}
    chunk_num = 0

    for chunk in pd.read_csv(
        CHART_PATH, low_memory=False, chunksize=CHUNK_SIZE,
        usecols=["SUBJECT_ID","ICUSTAY_ID","ITEMID",
                 "VALUENUM","CHARTTIME"]
    ):
        chunk_num += 1
        chunk.columns = chunk.columns.str.upper()
        chunk = chunk[
            chunk["SUBJECT_ID"].isin(valid_sids) &
            chunk["ITEMID"].isin(VITAL_ITEMS.keys())
        ].dropna(subset=["VALUENUM","ICUSTAY_ID"])

        if chunk.empty:
            if chunk_num % 50 == 0:
                print(f"  Chunk {chunk_num}...")
            continue

        chunk["ICUSTAY_ID"] = chunk["ICUSTAY_ID"].astype(int)
        chunk["vital_name"] = chunk["ITEMID"].map(VITAL_ITEMS)
        chunk["CHARTTIME"]  = pd.to_datetime(chunk["CHARTTIME"],
                                              errors="coerce")

        temp_mask = (chunk["vital_name"] == "temperature") & \
                    (chunk["VALUENUM"] > 50)
        chunk.loc[temp_mask, "VALUENUM"] = \
            (chunk.loc[temp_mask, "VALUENUM"] - 32) * 5 / 9

        for vital, (lo, hi) in VITAL_RANGES.items():
            mask  = chunk["vital_name"] == vital
            chunk = chunk[~(mask & ~chunk["VALUENUM"].between(lo, hi))]

        for (icustay_id, vital), grp in chunk.groupby(
            ["ICUSTAY_ID","vital_name"]
        ):
            key  = (icustay_id, vital)
            vals = grp["VALUENUM"].values.tolist()
            times = grp["CHARTTIME"].values.tolist()
            if key not in stay_vitals:
                stay_vitals[key] = []
            stay_vitals[key].extend(vals)

        if chunk_num % 10 == 0:
            n = len(set(k[0] for k in stay_vitals))
            print(f"  Chunk {chunk_num} — stays: {n:,}")

    with open(CKPT_PATH, "wb") as f:
        pickle.dump(stay_vitals, f)
    print("Checkpoint saved.")

# ── Build Feature Matrix ──────────────────────────────────────
print("\nBuilding enhanced feature matrix...")
vitals_list = ["heart_rate","systolic_bp","spo2",
               "temperature","respiratory_rate"]
all_stays   = list(set(k[0] for k in stay_vitals))
rows        = []

def news2_score_val(rr, spo2, sbp, hr, temp):
    """Compute NEWS2 score for a single reading."""
    def rr_s(v):
        if v is None: return 0
        if v <= 8:  return 3
        if v <= 11: return 1
        if v <= 20: return 0
        if v <= 24: return 2
        return 3
    def spo2_s(v):
        if v is None: return 0
        if v <= 91: return 3
        if v <= 93: return 2
        if v <= 95: return 1
        return 0
    def bp_s(v):
        if v is None: return 0
        if v <= 90:  return 3
        if v <= 100: return 2
        if v <= 110: return 1
        if v <= 219: return 0
        return 3
    def hr_s(v):
        if v is None: return 0
        if v <= 40:  return 3
        if v <= 50:  return 1
        if v <= 90:  return 0
        if v <= 110: return 1
        if v <= 130: return 2
        return 3
    def temp_s(v):
        if v is None: return 0
        if v <= 35.0: return 3
        if v <= 36.0: return 1
        if v <= 38.0: return 0
        if v <= 39.0: return 1
        return 2
    return rr_s(rr) + spo2_s(spo2) + bp_s(sbp) + hr_s(hr) + temp_s(temp)

for stay_id in all_stays:
    row = {"icustay_id": stay_id}
    vital_vals = {}

    for vital in vitals_list:
        vals = stay_vitals.get((stay_id, vital), [])
        if vals:
            arr = np.array(vals)
            n   = len(arr)
            row[f"{vital}_mean"] = round(np.mean(arr), 2)
            row[f"{vital}_max"]  = round(np.max(arr),  2)
            row[f"{vital}_min"]  = round(np.min(arr),  2)
            row[f"{vital}_std"]  = round(np.std(arr),  2)

            # ── NEW: Rate of change (last 25% vs first 25%) ───
            quarter = max(1, n // 4)
            early   = np.mean(arr[:quarter])
            late    = np.mean(arr[-quarter:])
            row[f"{vital}_trend"] = round(late - early, 2)

            # ── NEW: Instability = readings outside normal range
            if vital == "heart_rate":
                abnormal = np.sum((arr < 60) | (arr > 100))
            elif vital == "systolic_bp":
                abnormal = np.sum((arr < 90) | (arr > 140))
            elif vital == "spo2":
                abnormal = np.sum(arr < 95)
            elif vital == "respiratory_rate":
                abnormal = np.sum((arr < 12) | (arr > 20))
            elif vital == "temperature":
                abnormal = np.sum((arr < 36.0) | (arr > 38.0))
            else:
                abnormal = 0
            row[f"{vital}_pct_abnormal"] = round(
                abnormal / n if n > 0 else 0, 3
            )
            vital_vals[vital] = arr
        else:
            row[f"{vital}_mean"]         = np.nan
            row[f"{vital}_max"]          = np.nan
            row[f"{vital}_min"]          = np.nan
            row[f"{vital}_std"]          = np.nan
            row[f"{vital}_trend"]        = np.nan
            row[f"{vital}_pct_abnormal"] = np.nan
            vital_vals[vital] = np.array([])

    # ── NEW: NEWS2 score statistics per stay ──────────────────
    n_readings = min(
        len(vital_vals.get("heart_rate", [])),
        len(vital_vals.get("systolic_bp", [])),
        len(vital_vals.get("spo2", [])),
        len(vital_vals.get("respiratory_rate", []))
    )
    if n_readings > 0:
        news2_scores = []
        rrs  = vital_vals.get("respiratory_rate", [])
        spo2s = vital_vals.get("spo2", [])
        sbps  = vital_vals.get("systolic_bp", [])
        hrs   = vital_vals.get("heart_rate", [])
        temps = vital_vals.get("temperature", [])

        for j in range(min(n_readings, len(rrs))):
            rr   = rrs[j]   if j < len(rrs)   else None
            spo2 = spo2s[j] if j < len(spo2s) else None
            sbp  = sbps[j]  if j < len(sbps)  else None
            hr   = hrs[j]   if j < len(hrs)   else None
            temp = temps[j] if j < len(temps) else None
            news2_scores.append(
                news2_score_val(rr, spo2, sbp, hr, temp)
            )

        arr_n2 = np.array(news2_scores)
        row["news2_mean"]     = round(np.mean(arr_n2), 2)
        row["news2_max"]      = round(np.max(arr_n2),  2)
        row["news2_pct_high"] = round(
            np.sum(arr_n2 >= 5) / len(arr_n2), 3
        )
    else:
        row["news2_mean"]     = np.nan
        row["news2_max"]      = np.nan
        row["news2_pct_high"] = np.nan

    # ── NEW: Shock Index statistics ───────────────────────────
    hrs  = vital_vals.get("heart_rate", np.array([]))
    sbps = vital_vals.get("systolic_bp", np.array([]))
    n_si = min(len(hrs), len(sbps))
    if n_si > 0:
        si_vals = hrs[:n_si] / np.where(sbps[:n_si] > 0, sbps[:n_si], 1)
        row["shock_index_mean"] = round(np.mean(si_vals), 3)
        row["shock_index_max"]  = round(np.max(si_vals),  3)
        row["pct_shock"]        = round(
            np.sum(si_vals > 1.0) / n_si, 3
        )
    else:
        row["shock_index_mean"] = np.nan
        row["shock_index_max"]  = np.nan
        row["pct_shock"]        = np.nan

    rows.append(row)

features = pd.DataFrame(rows)
print(f"Feature matrix: {features.shape}")

# ── Merge ICU Stay Info ───────────────────────────────────────
icu_info = icustays[["ICUSTAY_ID","SUBJECT_ID","LOS"]].copy()
icu_info.columns = ["icustay_id","subject_id","los_hours"]
features = features.merge(icu_info, on="icustay_id", how="left")

# ── Merge CCI ─────────────────────────────────────────────────
features["cci_score"] = features["subject_id"].apply(
    lambda s: cci_map.get(str(int(s)), 0) if pd.notna(s) else 0
)

# ── Merge Labels ──────────────────────────────────────────────
print("Merging labels...")
features["died_hospital"] = features["icustay_id"].map(
    died_hospital_map
).fillna(0).astype(int)
features["died_48h"] = features["icustay_id"].map(
    died_48h_map
).fillna(0).astype(int)

# Use hospital mortality as primary label (48h as secondary)
features["died"] = features["died_hospital"]

# ── Merge Lab Features ────────────────────────────────────────
print("Adding lab features...")
labs = pd.read_csv(LAB_PATH, low_memory=False)
labs.columns = labs.columns.str.upper()
labs = labs[labs["ITEMID"].isin(LAB_ITEMS.keys())]
labs = labs.dropna(subset=["VALUENUM","SUBJECT_ID"])
labs["SUBJECT_ID"] = labs["SUBJECT_ID"].astype(int)
labs["lab_name"]   = labs["ITEMID"].map(LAB_ITEMS)
lab_agg = labs.groupby(["SUBJECT_ID","lab_name"])["VALUENUM"] \
              .agg(["mean","max","min"]).unstack()
lab_agg.columns = [f"{lab}_{stat}" for lab, stat in lab_agg.columns]
lab_agg = lab_agg.reset_index().rename(
    columns={"SUBJECT_ID":"subject_id"}
)
features = features.merge(lab_agg, on="subject_id", how="left")

# ── Deduplicate ───────────────────────────────────────────────
features = features.drop_duplicates(subset=["icustay_id"], keep="first")

# ── Save ──────────────────────────────────────────────────────
print(f"\nSaving to {OUT_PATH}...")
features.to_parquet(OUT_PATH, index=False)

print(f"\n{'='*60}")
print(f"FEATURE ENGINEERING COMPLETE")
print(f"{'='*60}")
print(f"Total ICU stays:       {len(features):,}")
print(f"Total features:        {features.shape[1]-4}")
print(f"Hospital mortality:    {features['died_hospital'].mean()*100:.1f}%")
print(f"48h mortality:         {features['died_48h'].mean()*100:.1f}%")
print(f"\nNew temporal features added:")
print(f"  *_trend          (rate of change early vs late)")
print(f"  *_pct_abnormal   (% readings outside normal range)")
print(f"  news2_mean/max/pct_high  (NEWS2 stats per stay)")
print(f"  shock_index_mean/max/pct_shock")
print(f"\nNext step: python3 train_xgboost.py")