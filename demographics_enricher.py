import pandas as pd
from pymongo import MongoClient
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# PATIENT DEMOGRAPHICS ENRICHMENT
# Sources: PATIENTS.csv + ADMISSIONS.csv + ICUSTAYS.csv
# MIMIC-III Clinical Database (Full Dataset)
# ═══════════════════════════════════════════════════════════════

DATA_DIR   = "/Users/malcolmdivinec/Downloads/mimic-iii-clinical-database-1.4"
MONGO_URI  = "mongodb://localhost:27017/"

print("=" * 60)
print("PATIENT DEMOGRAPHICS ENRICHMENT")
print("Sources: PATIENTS + ADMISSIONS + ICUSTAYS")
print("=" * 60)

# ── Load Files ────────────────────────────────────────────────
print("\nLoading files...")
patients   = pd.read_csv(f"{DATA_DIR}/PATIENTS.csv",   low_memory=False)
admissions = pd.read_csv(f"{DATA_DIR}/ADMISSIONS.csv", low_memory=False)
icustays   = pd.read_csv(f"{DATA_DIR}/ICUSTAYS.csv",   low_memory=False)

patients.columns   = patients.columns.str.upper()
admissions.columns = admissions.columns.str.upper()
icustays.columns   = icustays.columns.str.upper()

print(f"Patients:   {len(patients):,}")
print(f"Admissions: {len(admissions):,}")
print(f"ICU Stays:  {len(icustays):,}")

# ── Compute Age at First ICU Stay ─────────────────────────────
patients["DOB"]  = pd.to_datetime(patients["DOB"],  errors="coerce")
icustays["INTIME"] = pd.to_datetime(icustays["INTIME"], errors="coerce")

first_icu = icustays.sort_values("INTIME") \
                     .groupby("SUBJECT_ID") \
                     .first() \
                     .reset_index()[["SUBJECT_ID", "INTIME",
                                     "LAST_CAREUNIT", "LOS"]]

merged = patients.merge(first_icu, on="SUBJECT_ID", how="inner")
merged["age_at_first_icu"] = merged.apply(
    lambda row: round(
        (row["INTIME"].year - row["DOB"].year), 1
    ) if pd.notna(row["INTIME"]) and pd.notna(row["DOB"]) else None,
    axis=1
)
merged["age_at_first_icu"] = merged["age_at_first_icu"].apply(
    lambda a: 91.4 if a is not None and a > 150 else a
)

# MIMIC-III caps ages > 89 at 300 for de-identification
# Convert those back to 91.4 (median age of 90+ group per paper)
merged["age_at_first_icu"] = merged["age_at_first_icu"].apply(
    lambda a: 91.4 if a and a > 150 else a
)

# ── ICU Stay Aggregates ───────────────────────────────────────
icu_agg = icustays.groupby("SUBJECT_ID").agg(
    total_icu_stays    = ("ICUSTAY_ID", "count"),
    total_icu_los_days = ("LOS", "sum"),
    last_careunit      = ("LAST_CAREUNIT", "last")
).reset_index()
icu_agg["total_icu_los_days"] = icu_agg["total_icu_los_days"].round(2)

# ── Primary Diagnosis + Admission Info ────────────────────────
# Get first admission per patient
first_adm = admissions.sort_values("ADMITTIME") \
                       .groupby("SUBJECT_ID") \
                       .first() \
                       .reset_index()[[
                           "SUBJECT_ID", "ADMISSION_TYPE",
                           "INSURANCE", "ETHNICITY",
                           "DIAGNOSIS", "HOSPITAL_EXPIRE_FLAG"
                       ]]

# ── Merge Everything ──────────────────────────────────────────
final = merged.merge(icu_agg,   on="SUBJECT_ID", how="left")
final = final.merge(first_adm,  on="SUBJECT_ID", how="left")

# ── Build Result Documents ────────────────────────────────────
print("\nBuilding patient profiles...")
results = []
for _, row in final.iterrows():
    results.append({
        "patient_id":            str(int(row["SUBJECT_ID"])),
        "gender":                row.get("GENDER", ""),
        "age_at_first_icu":      row.get("age_at_first_icu"),
        "ethnicity":             str(row.get("ETHNICITY", ""))[:50],
        "insurance":             str(row.get("INSURANCE", ""))[:30],
        "admission_type":        str(row.get("ADMISSION_TYPE", "")),
        "primary_diagnosis":     str(row.get("DIAGNOSIS", ""))[:100],
        "total_icu_stays":       int(row.get("total_icu_stays", 0)),
        "total_icu_los_days":    row.get("total_icu_los_days"),
        "last_careunit":         str(row.get("last_careunit", "")),
        "hospital_expire_flag":  int(row.get("HOSPITAL_EXPIRE_FLAG", 0)),
        "last_updated":          datetime.now().isoformat()
    })

# ── Save to MongoDB ───────────────────────────────────────────
print("Saving to MongoDB...")
client = MongoClient(MONGO_URI)
db     = client["icu_monitoring"]
db["patient_profiles"].drop()
if results:
    db["patient_profiles"].insert_many(results)
client.close()

# ── Summary ───────────────────────────────────────────────────
male      = sum(1 for r in results if r["gender"] == "M")
female    = sum(1 for r in results if r["gender"] == "F")
expired   = sum(1 for r in results if r["hospital_expire_flag"] == 1)
ages      = [r["age_at_first_icu"] for r in results
             if r["age_at_first_icu"]]
mean_age  = round(sum(ages) / len(ages), 1) if ages else 0

print(f"\n{'=' * 60}")
print(f"DEMOGRAPHICS COMPLETE")
print(f"{'=' * 60}")
print(f"Total patients:         {len(results):,}")
print(f"Male:                   {male:,}")
print(f"Female:                 {female:,}")
print(f"Mean age at ICU:        {mean_age} years")
print(f"Hospital mortality:     {expired:,} "
      f"({round(expired/len(results)*100,1)}%)")
print(f"Saved to: icu_monitoring.patient_profiles")