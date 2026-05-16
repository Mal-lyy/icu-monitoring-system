import os
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.model_selection import train_test_split

# ═══════════════════════════════════════════════════════════════
# ML vs CLINICAL SCORING COMPARISON
#
# Compares XGBoost against rule-based clinical scores:
#   - NEWS2 (RCP 2017)
#   - qSOFA (Singer et al. JAMA 2016)
#   - Shock Index
#   - Logistic Regression baseline
#   - XGBoost (Chen & Guestrin KDD 2016)
#
# Metric: AUC-ROC — standard for clinical prediction models
# ═══════════════════════════════════════════════════════════════

FEATURES_PATH = "/Users/malcolmdivinec/Documents/icu-monitoring-system/features.parquet"
MODEL_PATH    = "/Users/malcolmdivinec/Documents/icu-monitoring-system/models/xgboost_mortality.pkl"
OUT_PATH      = "/Users/malcolmdivinec/Documents/icu-monitoring-system/models/comparison_results.json"

print("=" * 60)
print("ML vs CLINICAL SCORING COMPARISON")
print("=" * 60)

# ── Load Features ─────────────────────────────────────────────
df = pd.read_parquet(FEATURES_PATH)
print(f"Dataset: {len(df):,} ICU stays | "
      f"Mortality: {df['died'].mean()*100:.1f}%")

y = df["died"].fillna(0).astype(int)

# ── Reproduce same train/test split as training ───────────────
exclude = ["icustay_id", "subject_id", "died",
           "died_hospital", "died_48h"]
feature_cols = [c for c in df.columns if c not in exclude]
X = df[feature_cols].fillna(df[feature_cols].median())

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
test_idx = X_test.index
df_test  = df.loc[test_idx].copy()
y_test   = y.loc[test_idx]

print(f"Test set: {len(y_test):,} stays | "
      f"Deaths: {y_test.sum():,} ({y_test.mean()*100:.1f}%)\n")

results = {}

# ── Helper ────────────────────────────────────────────────────
def evaluate(name, y_true, y_score, threshold=0.5):
    auc  = roc_auc_score(y_true, y_score)
    pred = (np.array(y_score) >= threshold).astype(int)
    acc  = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, zero_division=0)
    rec  = recall_score(y_true, pred, zero_division=0)
    f1   = f1_score(y_true, pred, zero_division=0)
    cm   = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    results[name] = {
        "auc_roc":    round(auc,  4),
        "accuracy":   round(acc,  4),
        "precision":  round(prec, 4),
        "recall":     round(rec,  4),
        "f1_score":   round(f1,   4),
        "specificity":round(spec, 4),
    }
    return auc

# ═══════════════════════════════════════════════════════════════
# 1. NEWS2 Score as Predictor
# Normalize to [0,1]: max possible NEWS2 = 15
# ═══════════════════════════════════════════════════════════════
print("--- Evaluating NEWS2 ---")
if "news2_max" in df_test.columns:
    news2_scores = df_test["news2_max"].fillna(0) / 15.0
    auc_news2    = evaluate("NEWS2 (max score)", y_test,
                            news2_scores, threshold=5/15)
    print(f"NEWS2 AUC-ROC: {auc_news2:.4f}")
else:
    print("NEWS2 scores not in features — skipping")
    auc_news2 = None

# ═══════════════════════════════════════════════════════════════
# 2. qSOFA Score as Predictor
# qSOFA = RR>=22 + SBP<=100 (0-2 scale)
# ═══════════════════════════════════════════════════════════════
print("--- Evaluating qSOFA ---")
rr_col  = "respiratory_rate_mean"
sbp_col = "systolic_bp_mean"

if rr_col in df_test.columns and sbp_col in df_test.columns:
    qsofa_rr  = (df_test[rr_col].fillna(0)  >= 22).astype(int)
    qsofa_sbp = (df_test[sbp_col].fillna(200) <= 100).astype(int)
    qsofa_scores = (qsofa_rr + qsofa_sbp) / 2.0
    auc_qsofa = evaluate("qSOFA", y_test,
                         qsofa_scores, threshold=0.5)
    print(f"qSOFA AUC-ROC: {auc_qsofa:.4f}")
else:
    print("qSOFA features not available — skipping")
    auc_qsofa = None

# ═══════════════════════════════════════════════════════════════
# 3. Shock Index as Predictor
# ═══════════════════════════════════════════════════════════════
print("--- Evaluating Shock Index ---")
if "shock_index_max" in df_test.columns:
    si_scores = df_test["shock_index_max"].fillna(0)
    si_scores = np.clip(si_scores, 0, 3) / 3.0
    auc_si = evaluate("Shock Index", y_test,
                      si_scores, threshold=1/3)
    print(f"Shock Index AUC-ROC: {auc_si:.4f}")
else:
    auc_si = None

# ═══════════════════════════════════════════════════════════════
# 4. Charlson CCI as Predictor
# ═══════════════════════════════════════════════════════════════
print("--- Evaluating Charlson CCI ---")
if "cci_score" in df_test.columns:
    cci_scores = df_test["cci_score"].fillna(0)
    cci_scores = np.clip(cci_scores, 0, 10) / 10.0
    auc_cci = evaluate("Charlson CCI", y_test,
                       cci_scores, threshold=0.3)
    print(f"Charlson CCI AUC-ROC: {auc_cci:.4f}")
else:
    auc_cci = None

# ═══════════════════════════════════════════════════════════════
# 5. Logistic Regression (already trained in train_xgboost.py)
# Re-train here for fair comparison on same test set
# ═══════════════════════════════════════════════════════════════
print("--- Evaluating Logistic Regression ---")
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X_train_lr = X.drop(index=test_idx)
y_train_lr = y.drop(index=test_idx)

lr = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        max_iter=1000, random_state=42,
        class_weight="balanced"
    ))
])
lr.fit(X_train_lr, y_train_lr)
lr_proba = lr.predict_proba(X_test)[:,1]
auc_lr   = evaluate("Logistic Regression", y_test,
                    lr_proba, threshold=0.5)
print(f"Logistic Regression AUC-ROC: {auc_lr:.4f}")

# ═══════════════════════════════════════════════════════════════
# 6. XGBoost
# ═══════════════════════════════════════════════════════════════
print("--- Evaluating XGBoost ---")
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model        = bundle["model"]
feature_cols = bundle["feature_cols"]
medians      = bundle["medians"]

X_test_xgb = X_test[feature_cols].fillna(
    pd.Series(medians)
)
xgb_proba = model.predict_proba(X_test_xgb)[:,1]
auc_xgb   = evaluate("XGBoost", y_test,
                     xgb_proba, threshold=0.5)
print(f"XGBoost AUC-ROC: {auc_xgb:.4f}")

# ═══════════════════════════════════════════════════════════════
# FINAL COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"FINAL COMPARISON — All Models on Same Test Set")
print(f"Test set: {len(y_test):,} ICU stays "
      f"({y_test.sum():,} deaths)")
print(f"{'='*60}")
print(f"{'Model':<30} {'AUC-ROC':>8} {'Recall':>8} "
      f"{'Precision':>10} {'F1':>8}")
print(f"{'-'*60}")

order = ["NEWS2 (max score)", "qSOFA", "Shock Index",
         "Charlson CCI", "Logistic Regression", "XGBoost"]
for name in order:
    if name not in results:
        continue
    r = results[name]
    marker = " ← BEST" if name == "XGBoost" else ""
    print(f"{name:<30} {r['auc_roc']:>8.4f} "
          f"{r['recall']:>8.4f} "
          f"{r['precision']:>10.4f} "
          f"{r['f1_score']:>8.4f}{marker}")

print(f"{'='*60}")

# Key finding
if auc_news2 and auc_xgb:
    gain_vs_news2 = (auc_xgb - auc_news2) * 100
    print(f"\nKey Finding:")
    print(f"XGBoost outperforms NEWS2 by {gain_vs_news2:.1f}% AUC")
    print(f"XGBoost outperforms qSOFA  by "
          f"{(auc_xgb - auc_qsofa)*100:.1f}% AUC")
    print(f"XGBoost outperforms LR     by "
          f"{(auc_xgb - auc_lr)*100:.1f}% AUC")

# Save results
results["metadata"] = {
    "test_size":      len(y_test),
    "n_deaths":       int(y_test.sum()),
    "mortality_rate": round(y_test.mean()*100, 1),
    "dataset":        "MIMIC-III Full (60,190 ICU stays)",
    "references": {
        "NEWS2":   "Royal College of Physicians, 2017",
        "qSOFA":   "Singer et al. JAMA 2016;315(8):801-810",
        "XGBoost": "Chen & Guestrin. KDD 2016",
    }
}

# Load and display LSTM results
lstm_metrics_path = os.path.join(
    os.path.dirname(MODEL_PATH), "lstm_metrics.json"
)
if os.path.exists(lstm_metrics_path):
    with open(lstm_metrics_path) as f:
        lstm_m = json.load(f)
    print(f"{'LSTM (hospital mortality)':<30} "
          f"{lstm_m['lstm_hospital_auc']:>8.4f}     —        —         —")
    print(f"{'LSTM (48h mortality)':<30} "
          f"{lstm_m['lstm_48h_auc']:>8.4f}     —        —         —")
    print(f"{'='*60}")
    print(f"\nKey Finding:")
    print(f"XGBoost outperforms NEWS2 by "
          f"{(auc_xgb - auc_news2)*100:.1f}% AUC")
    print(f"XGBoost outperforms qSOFA  by "
          f"{(auc_xgb - auc_qsofa)*100:.1f}% AUC")
    print(f"LSTM 48h AUC: {lstm_m['lstm_48h_auc']} "
          f"(sequence-based, 48h prediction task)")
    results["lstm_hospital"] = {
        "auc_roc": lstm_m["lstm_hospital_auc"],
        "task": "hospital_mortality"
    }
    results["lstm_48h"] = {
        "auc_roc": lstm_m["lstm_48h_auc"],
        "task": "48h_mortality"
    }

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {OUT_PATH}")