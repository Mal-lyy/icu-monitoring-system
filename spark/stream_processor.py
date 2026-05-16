from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, lit, when
from pyspark.sql.types import (StructType, StructField,
                                StringType, DoubleType)
from pymongo import MongoClient
import json
import pickle
import numpy as np
import os

# ═══════════════════════════════════════════════════════════════
# ICU REAL-TIME VITALS PROCESSOR
# Pipeline: Kafka → Spark Structured Streaming → MongoDB
#
# Layer 1: NEWS2 Clinical Risk Scoring (RCP 2017)
# Layer 2: qSOFA Sepsis Screening (Singer et al. JAMA 2016)
# Layer 3: Shock Index (HR/SBP)
# Layer 4: XGBoost Mortality Predictor (Chen & Guestrin KDD 2016)
#           + SHAP Explainability (Lundberg & Lee NeurIPS 2017)
# Layer 5: LSTM 48h Mortality (Hochreiter & Schmidhuber 1997)
# ═══════════════════════════════════════════════════════════════

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models"
)

# ── Load XGBoost ──────────────────────────────────────────────
_MODEL_BUNDLE = None
try:
    with open(os.path.join(MODELS_DIR, "xgboost_mortality.pkl"), "rb") as f:
        _MODEL_BUNDLE = pickle.load(f)
    print(f"✅ XGBoost loaded — AUC {_MODEL_BUNDLE['xgb_auc']:.4f} — "
          f"trained on {_MODEL_BUNDLE['trained_on']:,} ICU stays")
except Exception as e:
    print(f"⚠️  XGBoost not found: {e}")

# ── Load LSTM Models ──────────────────────────────────────────
_LSTM_48H        = None
_LSTM_48H_BUNDLE = None
_TORCH_AVAILABLE = False

try:
    import torch
    import torch.nn as nn

    class ICU_LSTM(nn.Module):
        def __init__(self, input_size=5, hidden_size=64,
                     num_layers=2, dropout=0.3):
            super(ICU_LSTM, self).__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=True
            )
            self.dropout = nn.Dropout(dropout)
            self.fc1     = nn.Linear(hidden_size * 2, 32)
            self.relu    = nn.ReLU()
            self.fc2     = nn.Linear(32, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            out = lstm_out[:, -1, :]
            out = self.dropout(out)
            out = self.relu(self.fc1(out))
            out = self.dropout(out)
            out = self.sigmoid(self.fc2(out))
            return out.squeeze(1)

    def load_lstm(path):
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        cfg   = bundle["model_config"]
        model = ICU_LSTM(
            input_size  = cfg["input_size"],
            hidden_size = cfg["hidden_size"],
            num_layers  = cfg["num_layers"],
            dropout     = cfg["dropout"]
        )
        model.load_state_dict(bundle["model_state"])
        model.eval()
        return model, bundle

    lstm_path = os.path.join(MODELS_DIR, "lstm_48h.pkl")
    _LSTM_48H, _LSTM_48H_BUNDLE = load_lstm(lstm_path)
    _TORCH_AVAILABLE = True
    print(f"✅ LSTM 48h loaded — AUC {_LSTM_48H_BUNDLE['auc']:.4f}")

except Exception as e:
    print(f"⚠️  LSTM not loaded: {e}")

# ── XGBoost Scoring ───────────────────────────────────────────
def score_mortality(doc, patient_vitals_cache):
    if _MODEL_BUNDLE is None:
        return None, []

    feature_cols = _MODEL_BUNDLE["feature_cols"]
    medians      = _MODEL_BUNDLE["medians"]
    model        = _MODEL_BUNDLE["model"]
    explainer    = _MODEL_BUNDLE["explainer"]
    pid          = doc.get("patient_id")

    if pid not in patient_vitals_cache:
        patient_vitals_cache[pid] = {
            "heart_rate": [], "systolic_bp": [],
            "spo2": [], "temperature": [],
            "respiratory_rate": []
        }

    for vital in ["heart_rate", "systolic_bp", "spo2",
                  "temperature", "respiratory_rate"]:
        val = doc.get(vital)
        if val is not None:
            patient_vitals_cache[pid][vital].append(float(val))

    cache = patient_vitals_cache[pid]

    def agg(vals):
        if not vals:
            return None, None, None, None
        arr = np.array(vals)
        return (round(np.mean(arr), 2), round(np.max(arr), 2),
                round(np.min(arr), 2), round(np.std(arr), 2))

    feature_map = {}
    for vital in ["heart_rate", "systolic_bp", "spo2",
                  "temperature", "respiratory_rate"]:
        mn, mx, mi, sd = agg(cache[vital])
        feature_map[f"{vital}_mean"] = mn
        feature_map[f"{vital}_max"]  = mx
        feature_map[f"{vital}_min"]  = mi
        feature_map[f"{vital}_std"]  = sd

    feature_map["los_hours"]  = medians.get("los_hours", 24.0)
    feature_map["cci_score"]  = medians.get("cci_score", 2.0)
    feature_map["lactate"]    = medians.get("lactate")
    feature_map["wbc"]        = medians.get("wbc")
    feature_map["creatinine"] = medians.get("creatinine")
    feature_map["platelets"]  = medians.get("platelets")
    feature_map["ph"]         = medians.get("ph")

    X = np.array([
        feature_map.get(f, medians.get(f, 0.0)) or medians.get(f, 0.0)
        for f in feature_cols
    ]).reshape(1, -1)

    for j, f in enumerate(feature_cols):
        if X[0][j] is None or np.isnan(X[0][j]):
            X[0][j] = medians.get(f, 0.0) or 0.0

    X = X.astype(float)

    try:
        mortality_prob = float(model.predict_proba(X)[0][1])
        mortality_pct  = round(mortality_prob * 100, 1)

        shap_vals  = explainer.shap_values(X)[0]
        shap_pairs = sorted(
            zip(feature_cols, shap_vals),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        top_factors = [
            {"feature": f, "impact": round(float(v), 3)}
            for f, v in shap_pairs[:5]
        ]
        return mortality_pct, top_factors
    except Exception:
        return None, []

# ── LSTM 48h Scoring ──────────────────────────────────────────
VITAL_MEANS = {
    "heart_rate": 80.0, "systolic_bp": 120.0, "spo2": 97.0,
    "temperature": 37.0, "respiratory_rate": 16.0,
}
VITAL_STDS = {
    "heart_rate": 15.0, "systolic_bp": 20.0, "spo2": 3.0,
    "temperature": 0.8, "respiratory_rate": 4.0,
}
VITALS_ORDER = ["heart_rate", "systolic_bp", "spo2",
                "temperature", "respiratory_rate"]
TIME_STEPS   = 24

def score_lstm_48h(pid, patient_vitals_cache):
    if not _TORCH_AVAILABLE or _LSTM_48H is None:
        return None

    cache = patient_vitals_cache.get(pid, {})
    seq   = np.zeros((TIME_STEPS, 5), dtype=np.float32)

    for v_idx, vital in enumerate(VITALS_ORDER):
        vals = cache.get(vital, [])
        if not vals:
            seq[:, v_idx] = 0.0
            continue
        arr = np.array(vals, dtype=np.float32)
        arr = (arr - VITAL_MEANS[vital]) / VITAL_STDS[vital]
        arr = np.clip(arr, -4, 4)
        if len(arr) >= TIME_STEPS:
            seq[:, v_idx] = arr[-TIME_STEPS:]
        else:
            pad = TIME_STEPS - len(arr)
            seq[:, v_idx] = np.concatenate(
                [np.full(pad, arr[0]), arr]
            )

    try:
        import torch
        with torch.no_grad():
            x     = torch.FloatTensor(seq).unsqueeze(0)
            prob  = float(_LSTM_48H(x).item())
            return round(prob * 100, 1)
    except Exception:
        return None

# ── Spark Session ─────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("ICU_Vitals_Processor") \
    .config("spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ── Schema ────────────────────────────────────────────────────
schema = StructType([
    StructField("patient_id",       StringType(), True),
    StructField("timestamp",        StringType(), True),
    StructField("charttime",        StringType(), True),
    StructField("heart_rate",       DoubleType(), True),
    StructField("systolic_bp",      DoubleType(), True),
    StructField("diastolic_bp",     DoubleType(), True),
    StructField("spo2",             DoubleType(), True),
    StructField("temperature",      DoubleType(), True),
    StructField("respiratory_rate", DoubleType(), True),
])

# ── Read From Kafka ───────────────────────────────────────────
stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "icu-vitals") \
    .option("startingOffsets", "latest") \
    .load()

parsed = stream.select(
    from_json(col("value").cast("string"), schema).alias("d")
).select("d.*")

# ── Temperature Conversion ────────────────────────────────────
converted = parsed.withColumn(
    "temperature",
    when(col("temperature") > 50,
         (col("temperature") - 32) * 5 / 9
    ).otherwise(col("temperature"))
)

# ── Physiological Plausibility Filter ────────────────────────
converted = converted.filter(
    (col("respiratory_rate").isNull() |
     col("respiratory_rate").between(1, 60))   &
    (col("heart_rate").isNull() |
     col("heart_rate").between(1, 300))         &
    (col("systolic_bp").isNull() |
     col("systolic_bp").between(40, 300))       &
    (col("temperature").isNull() |
     col("temperature").between(25, 45))        &
    (col("spo2").isNull() |
     col("spo2").between(50, 100))
)

# ═══════════════════════════════════════════════════════════════
# LAYER 1 — NEWS2 (RCP 2017)
# ═══════════════════════════════════════════════════════════════
scored = converted.withColumn("rr_score",
    when(col("respiratory_rate") <= 8,   lit(3))
    .when(col("respiratory_rate") <= 11, lit(1))
    .when(col("respiratory_rate") <= 20, lit(0))
    .when(col("respiratory_rate") <= 24, lit(2))
    .when(col("respiratory_rate") >= 25, lit(3))
    .otherwise(lit(0))
)
scored = scored.withColumn("spo2_score",
    when(col("spo2") <= 91, lit(3))
    .when(col("spo2") <= 93, lit(2))
    .when(col("spo2") <= 95, lit(1))
    .when(col("spo2") >= 96, lit(0))
    .otherwise(lit(0))
)
scored = scored.withColumn("bp_score",
    when(col("systolic_bp") <= 90,   lit(3))
    .when(col("systolic_bp") <= 100, lit(2))
    .when(col("systolic_bp") <= 110, lit(1))
    .when(col("systolic_bp") <= 219, lit(0))
    .when(col("systolic_bp") >= 220, lit(3))
    .otherwise(lit(0))
)
scored = scored.withColumn("hr_score",
    when(col("heart_rate") <= 40,   lit(3))
    .when(col("heart_rate") <= 50,  lit(1))
    .when(col("heart_rate") <= 90,  lit(0))
    .when(col("heart_rate") <= 110, lit(1))
    .when(col("heart_rate") <= 130, lit(2))
    .when(col("heart_rate") >= 131, lit(3))
    .otherwise(lit(0))
)
scored = scored.withColumn("temp_score",
    when(col("temperature") <= 35.0, lit(3))
    .when(col("temperature") <= 36.0, lit(1))
    .when(col("temperature") <= 38.0, lit(0))
    .when(col("temperature") <= 39.0, lit(1))
    .when(col("temperature") >= 39.1, lit(2))
    .otherwise(lit(0))
)
scored = scored.withColumn("news2_score",
    col("rr_score")   +
    col("spo2_score") +
    col("bp_score")   +
    col("hr_score")   +
    col("temp_score")
)
scored = scored.withColumn("red_score_triggered",
    (col("rr_score")   == 3) |
    (col("spo2_score") == 3) |
    (col("bp_score")   == 3) |
    (col("hr_score")   == 3) |
    (col("temp_score") == 3)
)

# ═══════════════════════════════════════════════════════════════
# LAYER 2 — qSOFA (Singer et al. JAMA 2016)
# ═══════════════════════════════════════════════════════════════
scored = scored.withColumn("qsofa_rr",
    when(col("respiratory_rate") >= 22, lit(1)).otherwise(lit(0))
)
scored = scored.withColumn("qsofa_bp",
    when(col("systolic_bp") <= 100, lit(1)).otherwise(lit(0))
)
scored = scored.withColumn("qsofa_score",
    col("qsofa_rr") + col("qsofa_bp")
)

# ═══════════════════════════════════════════════════════════════
# LAYER 3 — SHOCK INDEX
# ═══════════════════════════════════════════════════════════════
scored = scored.withColumn("shock_index",
    when(
        (col("heart_rate").isNotNull()) &
        (col("systolic_bp").isNotNull()) &
        (col("systolic_bp") > 0),
        col("heart_rate") / col("systolic_bp")
    ).otherwise(lit(None).cast(DoubleType()))
)
scored = scored.withColumn("hemodynamic_instability",
    when(col("shock_index") > 1.0, lit(True))
    .otherwise(lit(False))
)

# ═══════════════════════════════════════════════════════════════
# COMBINED ALERT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

classified = scored.withColumn("overall_status",
    when(
        col("red_score_triggered") |
        (col("news2_score") >= 7)  |
        (col("qsofa_score") >= 2),
        lit("CRITICAL")
    ).when(
        (col("news2_score") >= 5) |
        (col("qsofa_score") == 1) |
        col("hemodynamic_instability"),
        lit("WARNING")
    ).otherwise(lit("NORMAL"))
)

# ── Per-patient cache for ML scoring ─────────────────────────
_patient_cache = {}

# ── Save to MongoDB ───────────────────────────────────────────
def save_to_mongo(batch_df, batch_id):
    records = batch_df.toJSON().collect()
    if not records:
        return

    client = MongoClient("mongodb://localhost:27017/")
    db     = client["icu_monitoring"]
    docs   = [json.loads(r) for r in records]

    for doc in docs:
        pid = doc.get("patient_id")

        # Update patient cache first
        if pid not in _patient_cache:
            _patient_cache[pid] = {
                "heart_rate": [], "systolic_bp": [],
                "spo2": [], "temperature": [],
                "respiratory_rate": []
            }
        for vital in ["heart_rate", "systolic_bp", "spo2",
                      "temperature", "respiratory_rate"]:
            val = doc.get(vital)
            if val is not None:
                _patient_cache[pid][vital].append(float(val))

        # LAYER 4: XGBoost
        mort_pct, top_factors = score_mortality(doc, _patient_cache)
        doc["mortality_risk_pct"]     = mort_pct
        doc["mortality_top_factors"]  = top_factors

        # LAYER 5: LSTM 48h
        lstm_pct = score_lstm_48h(pid, _patient_cache)
        doc["lstm_48h_risk_pct"] = lstm_pct

        # Upgrade status if either ML model flags high risk
        if mort_pct is not None and mort_pct >= 70:
            if doc.get("overall_status") == "NORMAL":
                doc["overall_status"] = "WARNING"
        if lstm_pct is not None and lstm_pct >= 50:
            if doc.get("overall_status") == "NORMAL":
                doc["overall_status"] = "WARNING"

    db["vitals_log"].insert_many(docs)

    alerts = [d for d in docs
              if d.get("overall_status") in ("CRITICAL", "WARNING")]
    if alerts:
        db["alert_history"].insert_many(alerts)

    client.close()

    ml_scored   = sum(1 for d in docs
                      if d.get("mortality_risk_pct") is not None)
    lstm_scored = sum(1 for d in docs
                      if d.get("lstm_48h_risk_pct") is not None)
    print(f"Batch {batch_id}: {len(docs)} records | "
          f"{len(alerts)} alerts | "
          f"XGB: {ml_scored} | LSTM: {lstm_scored}")

# ── Start Streaming ───────────────────────────────────────────
print("\nSpark Streaming started — 5-layer detection active")
print("Layer 1: NEWS2        (RCP 2017)")
print("Layer 2: qSOFA        (Singer et al. JAMA 2016)")
print("Layer 3: Shock Index  (HR/SBP > 1.0)")
print("Layer 4: XGBoost      (Chen & Guestrin KDD 2016)"
      " + SHAP (Lundberg & Lee NeurIPS 2017)")
print("Layer 5: LSTM 48h     (Hochreiter & Schmidhuber 1997)")

classified.writeStream \
    .foreachBatch(save_to_mongo) \
    .outputMode("append") \
    .trigger(processingTime="3 seconds") \
    .start() \
    .awaitTermination()