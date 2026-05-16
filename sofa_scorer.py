# NOTE: Partial SOFA scoring implemented but not integrated into
# the real-time pipeline due to FiO2/PaO2 data availability
# constraints in the MIMIC-III demo dataset.
# Planned for full dataset integration as future work.
# Run manually: python3 sofa_scorer.py

import pandas as pd
from pymongo import MongoClient
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# PARTIAL SOFA SCORER — BATCH SCRIPT
# Reference: Vincent JL et al. Intensive Care Med 1996;22(7):707-710
#            doi:10.1007/BF01709751
#
# 4 of 6 organ systems implemented (MIMIC-III demo constraints):
#   1. Coagulation   — Platelets       (LABEVENTS  51265)
#   2. Liver         — Bilirubin       (LABEVENTS  50885)
#   3. Renal         — Creatinine      (LABEVENTS  50912)
#   4. Cardiovascular — MAP            (CHARTEVENTS 456 / 220052)
#
# Excluded with documented rationale:
#   Respiratory: FiO2 available for only 28/100 patients in demo;
#                PaO2/FiO2 ratio not computable for 72% of cohort
#   CNS: GCS MetaVision (ITEMID 226755) has 1 row in demo;
#        CareVue GCS Total (ITEMID 198) available but composite
#        E+V+M breakdown absent — excluded to avoid misclassification
#
# Scoring convention: worst (most abnormal) value per patient
# across all ICU stays. Aligns with Vincent 1996 worst-case approach.
# Max achievable score = 16 (4 systems × max 4 per system).
# ═══════════════════════════════════════════════════════════════

FULL_DATA_PATH = "/Users/malcolmdivinec/Downloads/mimic-iii-clinical-database-1.4/"
LAB_PATH       = FULL_DATA_PATH + "LABEVENTS.csv"
CHART_PATH     = FULL_DATA_PATH + "CHARTEVENTS.csv"
ICUSTAYS_PATH  = FULL_DATA_PATH + "ICUSTAYS.csv"
CHUNK_SIZE     = 500_000

# ── ITEMID Mappings ───────────────────────────────────────────
LAB_ITEMIDS = {
    "platelets":  [51265],
    "bilirubin":  [50885],
    "creatinine": [50912],
}

CHART_ITEMIDS = {
    "map": [456, 220052],   # CareVue + MetaVision
}

# ── SOFA Scoring Functions ────────────────────────────────────

def score_platelets(val) -> int:
    """Coagulation — Platelets (×10³/μL). Vincent 1996 Table 1."""
    if val is None or pd.isnull(val): return None
    if val >= 150: return 0
    if val >= 100: return 1
    if val >=  50: return 2
    if val >=  20: return 3
    return 4

def score_bilirubin(val) -> int:
    """Liver — Bilirubin (mg/dL). Vincent 1996 Table 1."""
    if val is None or pd.isnull(val): return None
    if val <  1.2: return 0
    if val <  2.0: return 1
    if val <  6.0: return 2
    if val < 12.0: return 3
    return 4

def score_creatinine(val) -> int:
    """Renal — Creatinine (mg/dL). Vincent 1996 Table 1."""
    if val is None or pd.isnull(val): return None
    if val <  1.2: return 0
    if val <  2.0: return 1
    if val <  3.5: return 2
    if val <  5.0: return 3
    return 4

def score_map(val) -> int:
    """
    Cardiovascular — MAP (mmHg). Vincent 1996 Table 1.
    Note: Full cardiovascular scoring includes vasopressor dose
    (dopamine, epinephrine, norepinephrine). Vasopressor component
    excluded here — INPUTEVENTS parsing outside scope of demo.
    MAP-only scoring: 0 (≥70) or 1 (<70).
    """
    if val is None or pd.isnull(val): return None
    return 0 if val >= 70 else 1

def classify_partial_sofa(score) -> str:
    """
    Risk classification for partial (4-system) SOFA.
    Thresholds scaled from Vincent 1996 full SOFA mortality data.
    Max partial score = 16.
    """
    if score is None: return "UNKNOWN"
    if score == 0:    return "NONE"
    if score <= 3:    return "LOW"
    if score <= 7:    return "MODERATE"
    if score <= 11:   return "HIGH"
    return "CRITICAL"


def worst_lab(df_lab, subject_id, itemids):
    """Returns the single worst (most abnormal = highest) numeric value
    for the given patient and ITEMID list. Lab abnormalities generally
    worsen as values rise (exception: platelets — handled by inverting
    the worst-case logic at call site)."""
    rows = df_lab[
        (df_lab["subject_id"] == subject_id) &
        (df_lab["itemid"].isin(itemids))
    ]["valuenum"].dropna()
    return rows.max() if len(rows) else None

def worst_platelets(df_lab, subject_id, itemids):
    """Platelets: worst = LOWEST value (thrombocytopenia = danger)."""
    rows = df_lab[
        (df_lab["subject_id"] == subject_id) &
        (df_lab["itemid"].isin(itemids))
    ]["valuenum"].dropna()
    return rows.min() if len(rows) else None

def worst_map(df_chart, subject_id, itemids):
    """MAP: worst = LOWEST value (hypotension = danger)."""
    rows = df_chart[
        (df_chart["subject_id"] == subject_id) &
        (df_chart["itemid"].isin(itemids))
    ]["valuenum"].dropna()
    # Filter physiologically plausible MAP values
    rows = rows[(rows >= 20) & (rows <= 200)]
    return rows.min() if len(rows) else None


def main():
    print("=" * 60)
    print("  Partial SOFA Scorer (4 of 6 organ systems)")
    print("  Reference: Vincent et al. Intensive Care Med 1996")
    print("=" * 60)

    # ── Load Data ─────────────────────────────────────────────
    print("\n[1/4] Loading data (chunked for full MIMIC-III)...")

    # ICU patient filter
    icustays = pd.read_csv(ICUSTAYS_PATH)
    icustays.columns = icustays.columns.str.lower()
    icu_sids = set(icustays["subject_id"].unique())
    print(f"      ICU patients: {len(icu_sids):,}")

    all_lab_ids   = [id for ids in LAB_ITEMIDS.values()   for id in ids]
    all_chart_ids = [id for ids in CHART_ITEMIDS.values() for id in ids]

    # LABEVENTS — chunked
    print("      Reading LABEVENTS in chunks...")
    lab_chunks = []
    for chunk in pd.read_csv(LAB_PATH, chunksize=CHUNK_SIZE, low_memory=False):
        chunk.columns = chunk.columns.str.lower()
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        mask = (chunk["itemid"].isin(all_lab_ids) &
                chunk["subject_id"].isin(icu_sids))
        lab_chunks.append(chunk[mask])
    lab_filtered = pd.concat(lab_chunks, ignore_index=True)
    del lab_chunks
    print(f"      Lab rows kept: {len(lab_filtered):,}")

    # CHARTEVENTS — chunked (33GB)
    print("      Reading CHARTEVENTS in chunks (33GB — may take a few minutes)...")
    chart_chunks = []
    for chunk in pd.read_csv(CHART_PATH, chunksize=CHUNK_SIZE, low_memory=False,
                              usecols=["SUBJECT_ID", "ITEMID", "VALUENUM"]):
        chunk.columns = chunk.columns.str.lower()
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        mask = (chunk["itemid"].isin(all_chart_ids) &
                chunk["subject_id"].isin(icu_sids))
        chart_chunks.append(chunk[mask])
    chart_filtered = pd.concat(chart_chunks, ignore_index=True)
    del chart_chunks
    print(f"      Chart rows kept: {len(chart_filtered):,}")

    patients = sorted(lab_filtered["subject_id"].unique().tolist())
    print(f"      Patients to score: {len(patients):,}")

    # ── Score Each Patient ────────────────────────────────────
    print("\n[2/4] Scoring each patient...")
    docs = []

    for pid in patients:
        # Raw worst values
        plt_val  = worst_platelets(lab_filtered,   pid, LAB_ITEMIDS["platelets"])
        bili_val = worst_lab(lab_filtered,         pid, LAB_ITEMIDS["bilirubin"])
        cr_val   = worst_lab(lab_filtered,         pid, LAB_ITEMIDS["creatinine"])
        map_val  = worst_map(chart_filtered,       pid, CHART_ITEMIDS["map"])

        # Component scores
        plt_score  = score_platelets(plt_val)
        bili_score = score_bilirubin(bili_val)
        cr_score   = score_creatinine(cr_val)
        map_score  = score_map(map_val)

        # Partial SOFA total (only sum available components)
        component_scores = [s for s in [plt_score, bili_score, cr_score, map_score]
                            if s is not None]
        total_score  = sum(component_scores) if component_scores else None
        systems_scored = len(component_scores)

        # Organ failure flags (score >= 2 per Vincent 1996 = organ failure)
        organ_failures = []
        if plt_score  is not None and plt_score  >= 2: organ_failures.append("coagulation")
        if bili_score is not None and bili_score >= 2: organ_failures.append("liver")
        if cr_score   is not None and cr_score   >= 2: organ_failures.append("renal")
        if map_score  is not None and map_score  >= 2: organ_failures.append("cardiovascular")

        docs.append({
            "patient_id":         str(pid),

            # Raw worst values
            "worst_platelets_k_ul":   float(plt_val)  if plt_val  is not None else None,
            "worst_bilirubin_mg_dl":  float(bili_val) if bili_val is not None else None,
            "worst_creatinine_mg_dl": float(cr_val)   if cr_val   is not None else None,
            "worst_map_mmhg":         float(map_val)  if map_val  is not None else None,

            # Component SOFA scores (0-4 each, None if no data)
            "sofa_coagulation":    plt_score,
            "sofa_liver":          bili_score,
            "sofa_renal":          cr_score,
            "sofa_cardiovascular": map_score,

            # Totals
            "partial_sofa_score":  total_score,
            "systems_scored":      systems_scored,
            "organ_failures":      organ_failures,
            "organ_failure_count": len(organ_failures),
            "sofa_risk":           classify_partial_sofa(total_score),

            # Excluded systems (for report transparency)
            "excluded_systems": ["respiratory", "cns"],
            "exclusion_reason": {
                "respiratory": "FiO2 available for 28/100 patients only",
                "cns":         "GCS MetaVision has 1 row in demo dataset",
            },

            "scored_at": datetime.utcnow().isoformat(),
        })

    # ── Save to MongoDB ───────────────────────────────────────
    print("\n[3/4] Saving to MongoDB (collection: sofa_scores)...")
    client = MongoClient("mongodb://localhost:27017/")
    db     = client["icu_monitoring"]

    db["sofa_scores"].drop()
    db["sofa_scores"].insert_many(docs)
    client.close()
    print(f"      {len(docs)} documents written.")

    # ── Summary ───────────────────────────────────────────────
    print("\n[4/4] Summary:")
    scored   = [d for d in docs if d["partial_sofa_score"] is not None]
    scores   = [d["partial_sofa_score"] for d in scored]
    risk_cnt = {}
    for d in docs:
        r = d["sofa_risk"]
        risk_cnt[r] = risk_cnt.get(r, 0) + 1

    print(f"      Patients with any score: {len(scored)}/{len(docs)}")
    if scores:
        print(f"      Score range:  {min(scores)} – {max(scores)}")
        print(f"      Mean score:   {sum(scores)/len(scores):.2f}")
    print(f"      Risk breakdown: {risk_cnt}")

    mof = [d for d in docs if d["organ_failure_count"] >= 2]
    print(f"      Multi-organ failure (≥2 systems): {len(mof)} patients")

    print("\n      Top 5 highest SOFA scores:")
    for d in sorted(scored, key=lambda x: -x["partial_sofa_score"])[:5]:
        print(f"        Patient {d['patient_id']:>6} | "
              f"SOFA={d['partial_sofa_score']} | "
              f"Risk={d['sofa_risk']:<8} | "
              f"Failures: {', '.join(d['organ_failures']) or 'none'}")

    print("\nDone. sofa_scores collection ready.")


if __name__ == "__main__":
    main()
