import pandas as pd
from pymongo import MongoClient
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# SEPSIS-3 BATCH DETECTOR
# Reference: Singer M et al. JAMA 2016;315(8):801-810.
#            doi:10.1001/jama.2016.0287
#
# 10 lab indicators evaluated per patient:
#   Lactate, WBC, Creatinine, Bilirubin, Platelets,
#   pH, pCO2, Glucose, BUN, Chloride
#
# Thresholds from Singer et al. Table 2 and supplementary data.
# Worst (most abnormal) value per patient used — consistent
# with Sepsis-3 definition of organ dysfunction over time.
#
# Classification:
#   HIGH     — 2+ critical lab flags
#   MODERATE — 1 critical flag OR 2+ warning flags
#   LOW      — 1 warning flag
#   NONE     — 0 abnormal flags
#
# Input:  Full MIMIC-III LABEVENTS.csv (chunked) + ICUSTAYS.csv
# Output: MongoDB collection → sepsis_risk
# ═══════════════════════════════════════════════════════════════

FULL_DATA_PATH = "/Users/malcolmdivinec/Downloads/mimic-iii-clinical-database-1.4/"
LAB_PATH       = FULL_DATA_PATH + "LABEVENTS.csv"
ICUSTAYS_PATH  = FULL_DATA_PATH + "ICUSTAYS.csv"
CHUNK_SIZE     = 500_000

# ── ITEMID Mappings ───────────────────────────────────────────
# Both MetaVision and CareVue ITEMIDs included where applicable
LAB_ITEMIDS = {
    "lactate":    [50813],
    "wbc":        [51301, 51300],
    "creatinine": [50912],
    "bilirubin":  [50885],
    "platelets":  [51265],
    "ph":         [50820],
    "pco2":       [50818],
    "glucose":    [50931, 50809],
    "bun":        [51006],
    "chloride":   [50902],
}

ALL_LAB_IDS = set(id for ids in LAB_ITEMIDS.values() for id in ids)

# ── Sepsis-3 Thresholds ───────────────────────────────────────
# Each entry: (low_warning, low_critical, high_warning, high_critical)
# None = threshold not applicable on that side
THRESHOLDS = {
    "lactate":    (None, None,  2.0,   4.0),   # mmol/L
    "wbc":        (4.0,  None, 12.0,   None),  # ×10³/μL
    "creatinine": (None, None,  1.2,   2.0),   # mg/dL
    "bilirubin":  (None, None,  1.2,   2.0),   # mg/dL
    "platelets":  (100,  None, None,  150.0),  # ×10³/μL (low = danger, reversed)
    "ph":         (7.30, None,  7.45,  None),  # pH units
    "pco2":       (None, None, 45.0,   None),  # mmHg
    "glucose":    (70.0, None, 180.0,  None),  # mg/dL
    "bun":        (None, None, 20.0,  40.0),   # mg/dL
    "chloride":   (98.0, None, 106.0,  None),  # mEq/L
}

# Platelets: abnormality is LOW values (thrombocytopenia)
# pH: abnormality is LOW values (acidosis)
# Handled explicitly in score_lab()

def score_lab(name, value) -> tuple:
    """
    Returns (warning: bool, critical: bool) for a given lab value.
    Uses worst-case logic: critical implies warning.
    """
    if value is None or pd.isnull(value):
        return False, False

    lo_w, lo_c, hi_w, hi_c = THRESHOLDS[name]

    # Platelets: low = danger
    if name == "platelets":
        critical = (lo_c is not None and value < lo_c)
        warning  = (hi_c is not None and value < hi_c)
        return warning or critical, critical

    # pH: low = danger (acidosis)
    if name == "ph":
        critical = (lo_w is not None and value < lo_w)
        warning  = (hi_w is not None and value > hi_w) or \
                   (lo_w is not None and value < lo_w)
        # pCO2 high = warning
        return warning or critical, critical

    # WBC: both low and high are warning
    if name == "wbc":
        warning  = (lo_w is not None and value < lo_w) or \
                   (hi_w is not None and value > hi_w)
        critical = False
        return warning, critical

    # Glucose, Chloride: both low and high are warning
    if name in ("glucose", "chloride"):
        warning  = (lo_w is not None and value < lo_w) or \
                   (hi_w is not None and value > hi_w)
        return warning, False

    # All others: high side only, graded warning/critical
    warning  = (hi_w is not None and value >= hi_w)
    critical = (hi_c is not None and value >= hi_c)
    return warning or critical, critical


def classify_sepsis(n_warning: int, n_critical: int) -> str:
    if n_critical >= 2:          return "HIGH"
    if n_critical == 1:          return "MODERATE"
    if n_warning  >= 2:          return "MODERATE"
    if n_warning  == 1:          return "LOW"
    return "NONE"


def main():
    print("=" * 60)
    print("  Sepsis-3 Batch Detector")
    print("  Reference: Singer et al. JAMA 2016;315(8):801-810")
    print("=" * 60)

    # ── ICU patient filter ─────────────────────────────────────
    print("\n[1/4] Loading ICUSTAYS for ICU patient filter...")
    icustays = pd.read_csv(ICUSTAYS_PATH)
    icustays.columns = icustays.columns.str.lower()
    icu_sids = set(icustays["subject_id"].unique())
    print(f"      ICU patients: {len(icu_sids):,}")

    # ── Load LABEVENTS — chunked ───────────────────────────────
    print("\n[2/4] Reading LABEVENTS in chunks...")
    lab_chunks = []
    rows_scanned = 0

    for i, chunk in enumerate(pd.read_csv(
            LAB_PATH,
            chunksize=CHUNK_SIZE,
            low_memory=False)):
        chunk.columns = chunk.columns.str.lower()
        rows_scanned += len(chunk)

        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        mask = (chunk["itemid"].isin(ALL_LAB_IDS) &
                chunk["subject_id"].isin(icu_sids))
        lab_chunks.append(chunk[mask].dropna(subset=["valuenum"]))

        if (i + 1) % 10 == 0:
            kept = sum(len(c) for c in lab_chunks)
            print(f"      Chunks: {i+1} | "
                  f"Scanned: {rows_scanned:,} | "
                  f"Kept: {kept:,}")

    lab = pd.concat(lab_chunks, ignore_index=True)
    del lab_chunks

    # Reverse-map itemid → lab name
    itemid_to_name = {
        iid: name
        for name, ids in LAB_ITEMIDS.items()
        for iid in ids
    }
    lab["lab_name"] = lab["itemid"].map(itemid_to_name)
    patients = sorted(lab["subject_id"].unique().tolist())

    print(f"      Lab rows kept:     {len(lab):,}")
    print(f"      Patients to score: {len(patients):,}")

    # ── Score Each Patient ─────────────────────────────────────
    print("\n[3/4] Scoring each patient...")
    docs = []

    for pid in patients:
        pat_rows = lab[lab["subject_id"] == pid]

        warning_flags  = []
        critical_flags = []
        raw_values     = {}

        for lab_name, itemids in LAB_ITEMIDS.items():
            rows = pat_rows[pat_rows["itemid"].isin(itemids)]["valuenum"].dropna()
            if len(rows) == 0:
                raw_values[lab_name] = None
                continue

            # Worst-case: most abnormal value
            if lab_name in ("platelets", "ph", "wbc"):
                worst = rows.min()    # low = danger
            else:
                worst = rows.max()    # high = danger

            raw_values[lab_name] = round(float(worst), 3)
            is_warn, is_crit = score_lab(lab_name, worst)

            if is_warn: warning_flags.append(lab_name)
            if is_crit: critical_flags.append(lab_name)

        n_warning  = len(warning_flags)
        n_critical = len(critical_flags)
        risk       = classify_sepsis(n_warning, n_critical)

        docs.append({
            "patient_id":    f"P{pid}",
            "sepsis_risk":   risk,
            "critical_labs": n_critical,
            "warning_labs":  n_warning,
            "abnormal_flags": warning_flags + [
                f"{f}(CRITICAL)" for f in critical_flags
            ],
            "critical_flags": critical_flags,
            "warning_flags":  warning_flags,

            # Raw worst values per lab
            "lactate_mmol_l":    raw_values.get("lactate"),
            "wbc_k_ul":          raw_values.get("wbc"),
            "creatinine_mg_dl":  raw_values.get("creatinine"),
            "bilirubin_mg_dl":   raw_values.get("bilirubin"),
            "platelets_k_ul":    raw_values.get("platelets"),
            "ph":                raw_values.get("ph"),
            "pco2_mmhg":         raw_values.get("pco2"),
            "glucose_mg_dl":     raw_values.get("glucose"),
            "bun_mg_dl":         raw_values.get("bun"),
            "chloride_meq_l":    raw_values.get("chloride"),

            "labs_evaluated": sum(1 for v in raw_values.values()
                                  if v is not None),
            "scored_at": datetime.utcnow().isoformat(),
        })

    # ── Save to MongoDB ────────────────────────────────────────
    print("\n[4/4] Saving to MongoDB (collection: sepsis_risk)...")
    client = MongoClient("mongodb://localhost:27017/")
    db     = client["icu_monitoring"]

    db["sepsis_risk"].drop()
    db["sepsis_risk"].insert_many(docs)
    client.close()
    print(f"      {len(docs):,} documents written.")

    # ── Summary ────────────────────────────────────────────────
    risk_counts = {}
    for d in docs:
        r = d["sepsis_risk"]
        risk_counts[r] = risk_counts.get(r, 0) + 1

    print(f"\n      Risk breakdown: {risk_counts}")
    print(f"      High risk:      {risk_counts.get('HIGH', 0):,}")
    print(f"      Moderate risk:  {risk_counts.get('MODERATE', 0):,}")

    print("\n      Top 5 highest risk patients:")
    priority = {"HIGH": 0, "MODERATE": 1, "LOW": 2, "NONE": 3}
    top = sorted(docs, key=lambda x: (priority.get(x["sepsis_risk"], 9),
                                      -x["critical_labs"],
                                      -x["warning_labs"]))[:5]
    for d in top:
        print(f"        Patient {d['patient_id']:>8} | "
              f"Risk={d['sepsis_risk']:<8} | "
              f"Critical={d['critical_labs']} | "
              f"Warning={d['warning_labs']} | "
              f"Flags: {', '.join(d['critical_flags'] or d['warning_flags'] or ['none'])}")

    print("\nDone. sepsis_risk collection ready.")


if __name__ == "__main__":
    main()
