import pandas as pd
from pymongo import MongoClient
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# CHARLSON COMORBIDITY INDEX (CCI)
# Reference: Charlson et al. J Chronic Diseases 1987;40(5):373-383
# Updated ICD-9 mapping: Quan et al. Medical Care 2005;43(11):1130-9
#
# Maps ICD-9 diagnosis codes to 17 disease categories.
# Each category carries a weight (1-6).
# Total score predicts 10-year mortality risk.
# ═══════════════════════════════════════════════════════════════

DATA_DIR   = "/Users/malcolmdivinec/Downloads/mimic-iii-clinical-database-1.4"
DIAG_PATH  = f"{DATA_DIR}/DIAGNOSES_ICD.csv"
MONGO_URI  = "mongodb://localhost:27017/"

# ── ICD-9 Code Mappings (Quan et al. 2005) ───────────────────
CCI_MAP = {
    "myocardial_infarction": {
        "weight": 1,
        "codes": ["410","412"]
    },
    "congestive_heart_failure": {
        "weight": 1,
        "codes": ["39891","40201","40211","40291","40401","40403",
                  "40411","40413","40491","40493","4254","4255",
                  "4257","4258","4259","428"]
    },
    "peripheral_vascular_disease": {
        "weight": 1,
        "codes": ["0930","4373","440","441","4431","4432","4433",
                  "4434","4435","4436","4437","4438","4439","4471",
                  "5571","5579","V434"]
    },
    "cerebrovascular_disease": {
        "weight": 1,
        "codes": ["36234","430","431","432","433","434","435",
                  "436","437","438"]
    },
    "dementia": {
        "weight": 1,
        "codes": ["290","2941","3312"]
    },
    "chronic_pulmonary_disease": {
        "weight": 1,
        "codes": ["4168","4169","490","491","492","493","494",
                  "495","496","500","501","502","503","504","505",
                  "5064","5081","5088"]
    },
    "rheumatic_disease": {
        "weight": 1,
        "codes": ["4465","7100","7101","7102","7103","7104",
                  "7140","7141","7142","7148","725"]
    },
    "peptic_ulcer_disease": {
        "weight": 1,
        "codes": ["531","532","533","534"]
    },
    "mild_liver_disease": {
        "weight": 1,
        "codes": ["07022","07023","07032","07033","07044","07054",
                  "0706","0709","570","571","5733","5734","5738",
                  "5739","V427"]
    },
    "diabetes_without_complications": {
        "weight": 1,
        "codes": ["2500","2501","2502","2503","2508","2509"]
    },
    "diabetes_with_complications": {
        "weight": 2,
        "codes": ["2504","2505","2506","2507"]
    },
    "hemiplegia_paraplegia": {
        "weight": 2,
        "codes": ["3341","3440","3441","3442","3443","3444",
                  "3445","3446","3449","438"]
    },
    "renal_disease": {
        "weight": 2,
        "codes": ["40301","40311","40391","40402","40403","40412",
                  "40413","40492","40493","582","5830","5831","5832",
                  "5834","5836","5837","585","586","5880","V420",
                  "V451","V56"]
    },
    "malignancy": {
        "weight": 2,
        "codes": ["140","141","142","143","144","145","146","147",
                  "148","149","150","151","152","153","154","155",
                  "156","157","158","159","160","161","162","163",
                  "164","165","170","171","172","174","175","176",
                  "179","180","181","182","183","184","185","186",
                  "187","188","189","190","191","192","193","194",
                  "195","200","201","202","203","204","205","206",
                  "207","208","2386"]
    },
    "severe_liver_disease": {
        "weight": 3,
        "codes": ["4560","4561","4562","5722","5723","5724",
                  "5725","5726","5727","5728"]
    },
    "metastatic_solid_tumor": {
        "weight": 6,
        "codes": ["196","197","198","199"]
    },
    "aids_hiv": {
        "weight": 6,
        "codes": ["042","043","044"]
    },
}

# ── 10-year survival lookup (Charlson et al. 1987) ───────────
def predicted_survival(score):
    if score == 0:   return 0.98
    if score == 1:   return 0.96
    if score == 2:   return 0.90
    if score == 3:   return 0.77
    if score == 4:   return 0.53
    if score >= 5:   return 0.21

def cci_risk_label(score):
    if score == 0:  return "NONE"
    if score <= 2:  return "MILD"
    if score <= 4:  return "MODERATE"
    return "SEVERE"

def icd_matches(icd_code, prefix_list):
    code = str(icd_code).strip().upper()
    for prefix in prefix_list:
        if code.startswith(prefix.upper()):
            return True
    return False

print("=" * 60)
print("CHARLSON COMORBIDITY INDEX — Batch Scoring")
print("Reference: Charlson et al. J Chronic Diseases 1987")
print("ICD mapping: Quan et al. Medical Care 2005")
print("=" * 60)

# ── Load DIAGNOSES_ICD ────────────────────────────────────────
print("\nLoading DIAGNOSES_ICD.csv...")
diag = pd.read_csv(DIAG_PATH, low_memory=False)
diag.columns = diag.columns.str.upper()
diag = diag[["SUBJECT_ID", "ICD9_CODE"]].dropna()
diag["ICD9_CODE"] = diag["ICD9_CODE"].astype(str).str.strip()
print(f"Loaded {len(diag):,} diagnosis records")
print(f"Patients with diagnoses: {diag['SUBJECT_ID'].nunique():,}")

# ── Score Each Patient ────────────────────────────────────────
print("\nScoring patients...")
results = []

for subject_id, group in diag.groupby("SUBJECT_ID"):
    codes        = group["ICD9_CODE"].tolist()
    cci_score    = 0
    conditions   = []

    for condition, info in CCI_MAP.items():
        matched = any(
            icd_matches(code, info["codes"])
            for code in codes
        )
        if matched:
            cci_score += info["weight"]
            conditions.append(condition.replace("_", " ").title())

    results.append({
        "patient_id":              f"P{subject_id}",
        "cci_score":               cci_score,
        "cci_risk":                cci_risk_label(cci_score),
        "predicted_10yr_survival": predicted_survival(cci_score),
        "conditions_present":      conditions,
        "condition_count":         len(conditions),
        "last_updated":            datetime.now().isoformat()
    })

# ── Save to MongoDB ───────────────────────────────────────────
print("Saving to MongoDB...")
client = MongoClient(MONGO_URI)
db     = client["icu_monitoring"]
db["charlson_scores"].drop()
if results:
    db["charlson_scores"].insert_many(results)
client.close()

# ── Summary ───────────────────────────────────────────────────
scores = [r["cci_score"] for r in results]
none     = sum(1 for r in results if r["cci_risk"] == "NONE")
mild     = sum(1 for r in results if r["cci_risk"] == "MILD")
moderate = sum(1 for r in results if r["cci_risk"] == "MODERATE")
severe   = sum(1 for r in results if r["cci_risk"] == "SEVERE")

print(f"\n{'=' * 60}")
print(f"CCI SCORING COMPLETE")
print(f"{'=' * 60}")
print(f"Total patients scored:  {len(results):,}")
print(f"Mean CCI score:         {sum(scores)/len(scores):.2f}")
print(f"Max CCI score:          {max(scores)}")
print(f"NONE   (score 0):       {none:,}")
print(f"MILD   (score 1-2):     {mild:,}")
print(f"MODERATE (score 3-4):   {moderate:,}")
print(f"SEVERE (score 5+):      {severe:,}")
print(f"Saved to: icu_monitoring.charlson_scores")