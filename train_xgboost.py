import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             classification_report,
                             confusion_matrix)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# XGBOOST MORTALITY PREDICTOR + SHAP EXPLAINABILITY
# Reference: Chen & Guestrin. XGBoost. KDD 2016.
#            doi:10.1145/2939672.2939785
# SHAP: Lundberg & Lee. NeurIPS 2017.
#       doi:10.5555/3295222.3295230
#
# Label:    hospital_expire_flag (1=died, 0=survived)
# Features: vital aggregates + labs + CCI per ICU stay
# Dataset:  MIMIC-III Full (60,190 ICU stays, 53 features)
# ═══════════════════════════════════════════════════════════════

FEATURES_PATH  = "/Users/malcolmdivinec/Documents/icu-monitoring-system/features.parquet"
MODEL_DIR      = "/Users/malcolmdivinec/Documents/icu-monitoring-system/models"
MODEL_PATH     = f"{MODEL_DIR}/xgboost_mortality.pkl"
METRICS_PATH   = f"{MODEL_DIR}/model_metrics.json"

import os
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("XGBOOST MORTALITY PREDICTOR — Training")
print("Reference: Chen & Guestrin, KDD 2016")
print("=" * 60)

# ── Load Features ─────────────────────────────────────────────
print("\nLoading features.parquet...")
df = pd.read_parquet(FEATURES_PATH)
print(f"Shape: {df.shape}")
print(f"Mortality rate: {df['died'].mean()*100:.1f}%")

# ── Define Feature Columns ────────────────────────────────────
exclude = ["icustay_id", "subject_id", "died",
           "died_hospital", "died_48h"]
feature_cols = [c for c in df.columns if c not in exclude]
print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")

# ── Prepare X and y ───────────────────────────────────────────
X = df[feature_cols].copy()
y = df["died"].copy()

# Fill NaN with median per column
medians = X.median()
X = X.fillna(medians)

print(f"\nClass distribution:")
print(f"  Survived (0): {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
print(f"  Died     (1): {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")

# ── Train / Test Split ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

# ── Class Weight for Imbalance ────────────────────────────────
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Scale pos weight: {scale_pos_weight:.2f}")

# ── Baseline Logistic Regression ─────────────────────────────
print("\n--- Baseline: Logistic Regression ---")
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

lr = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        max_iter=1000, random_state=42,
        class_weight="balanced"
    ))
])
lr.fit(X_train, y_train)
lr_proba = lr.predict_proba(X_test)[:, 1]
lr_auc   = roc_auc_score(y_test, lr_proba)
lr_acc   = accuracy_score(y_test, lr.predict(X_test))
print(f"Logistic Regression AUC-ROC: {lr_auc:.4f}")
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

# ── XGBoost Training ──────────────────────────────────────────
print("\n--- XGBoost Training ---")
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# ── Evaluation ────────────────────────────────────────────────
print("\n--- Model Evaluation ---")
y_proba = model.predict_proba(X_test)[:, 1]
y_pred  = model.predict(X_test)

xgb_auc = roc_auc_score(y_test, y_proba)
xgb_acc = accuracy_score(y_test, y_pred)

print(f"\nXGBoost AUC-ROC:  {xgb_auc:.4f}")
print(f"XGBoost Accuracy: {xgb_acc:.4f}")
print(f"\nImprovement over LR baseline: "
      f"+{(xgb_auc - lr_auc)*100:.2f}% AUC")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Survived", "Died"]))

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")

# ── Feature Importance ────────────────────────────────────────
print("\n--- Top 10 Feature Importances ---")
importances = pd.Series(
    model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)
for feat, imp in importances.head(10).items():
    print(f"  {feat:<35} {imp:.4f}")

# ── SHAP Values ───────────────────────────────────────────────
print("\n--- Computing SHAP Values ---")
print("(This may take a few minutes...)")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Get mean absolute SHAP per feature
shap_importance = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=feature_cols
).sort_values(ascending=False)

print("\nTop 10 SHAP Feature Importances:")
for feat, val in shap_importance.head(10).items():
    print(f"  {feat:<35} {val:.4f}")

# ── Save Model Bundle ─────────────────────────────────────────
print(f"\nSaving model to {MODEL_PATH}...")
bundle = {
    "model":            model,
    "explainer":        explainer,
    "feature_cols":     feature_cols,
    "medians":          medians.to_dict(),
    "xgb_auc":          xgb_auc,
    "lr_auc":           lr_auc,
    "shap_importance":  shap_importance.head(10).to_dict(),
    "trained_on":       len(X),
    "trained_at":       pd.Timestamp.now().isoformat(),
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(bundle, f)

# ── Save Metrics JSON ─────────────────────────────────────────
metrics = {
    "xgboost_auc":      round(xgb_auc, 4),
    "xgboost_accuracy": round(xgb_acc, 4),
    "baseline_lr_auc":  round(lr_auc, 4),
    "improvement_pct":  round((xgb_auc - lr_auc) * 100, 2),
    "train_size":       len(X_train),
    "test_size":        len(X_test),
    "n_features":       len(feature_cols),
    "mortality_rate":   round(y.mean() * 100, 1),
    "top_features":     shap_importance.head(5).index.tolist(),
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n{'=' * 60}")
print(f"TRAINING COMPLETE")
print(f"{'=' * 60}")
print(f"XGBoost AUC-ROC:     {xgb_auc:.4f}")
print(f"LR Baseline AUC:     {lr_auc:.4f}")
print(f"Trained on:          {len(X):,} ICU stays")
print(f"Model saved to:      {MODEL_PATH}")
print(f"Metrics saved to:    {METRICS_PATH}")
print(f"\nNext step: integrate model into stream_processor.py")