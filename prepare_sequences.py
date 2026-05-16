import pandas as pd
import numpy as np
import pickle
import os

# ═══════════════════════════════════════════════════════════════
# SEQUENCE PREPARATION FOR LSTM
#
# Reshapes per-stay vital sign data into fixed-length sequences
# Each patient → matrix of shape (TIME_STEPS, N_FEATURES)
#
# Time steps: last 24 readings per vital sign per stay
# Features:   HR, SBP, SpO2, Temp, RR (5 vitals)
#
# Output:
#   sequences.pkl — X sequences (n_stays, 24, 5)
#   seq_labels.pkl — y labels (n_stays,) hospital + 48h
# ═══════════════════════════════════════════════════════════════

CKPT_PATH     = "/Users/malcolmdivinec/Documents/icu-monitoring-system/stay_vitals.pkl"
FEATURES_PATH = "/Users/malcolmdivinec/Documents/icu-monitoring-system/features.parquet"
SEQ_PATH      = "/Users/malcolmdivinec/Documents/icu-monitoring-system/sequences.pkl"
DATA_DIR      = "/Users/malcolmdivinec/Downloads/mimic-iii-clinical-database-1.4"
TIME_STEPS    = 24
VITALS        = ["heart_rate", "systolic_bp", "spo2",
                 "temperature", "respiratory_rate"]

# Normal ranges for normalization
VITAL_MEANS = {
    "heart_rate":       80.0,
    "systolic_bp":      120.0,
    "spo2":             97.0,
    "temperature":      37.0,
    "respiratory_rate": 16.0,
}
VITAL_STDS = {
    "heart_rate":       15.0,
    "systolic_bp":      20.0,
    "spo2":             3.0,
    "temperature":      0.8,
    "respiratory_rate": 4.0,
}

print("=" * 60)
print("SEQUENCE PREPARATION FOR LSTM")
print(f"Time steps per sequence: {TIME_STEPS}")
print(f"Features per time step:  {len(VITALS)}")
print("=" * 60)

# ── Load checkpoint ───────────────────────────────────────────
print("\nLoading stay_vitals checkpoint...")
with open(CKPT_PATH, "rb") as f:
    stay_vitals = pickle.load(f)

all_stays = list(set(k[0] for k in stay_vitals))
print(f"Total stays in checkpoint: {len(all_stays):,}")

# ── Load labels from features.parquet ────────────────────────
print("Loading labels...")
df = pd.read_parquet(FEATURES_PATH)
died_hospital = dict(zip(df["icustay_id"], df["died_hospital"]))
died_48h      = dict(zip(df["icustay_id"], df["died_48h"]))
print(f"Labels loaded: {len(died_hospital):,} stays")

# ── Build Sequences ───────────────────────────────────────────
print("\nBuilding sequences...")
X_seqs     = []
y_hospital = []
y_48h      = []
stay_ids   = []
skipped    = 0

for stay_id in all_stays:
    if stay_id not in died_hospital:
        skipped += 1
        continue

    # Build (TIME_STEPS, 5) matrix
    seq = np.zeros((TIME_STEPS, len(VITALS)), dtype=np.float32)

    has_any = False
    for v_idx, vital in enumerate(VITALS):
        vals = stay_vitals.get((stay_id, vital), [])
        if len(vals) == 0:
            # Fill with normal value normalized to 0
            seq[:, v_idx] = 0.0
            continue

        has_any = True
        arr = np.array(vals, dtype=np.float32)

        # Normalize using clinical normal ranges
        arr = (arr - VITAL_MEANS[vital]) / VITAL_STDS[vital]
        arr = np.clip(arr, -4, 4)

        # Take last TIME_STEPS readings
        if len(arr) >= TIME_STEPS:
            seq[:, v_idx] = arr[-TIME_STEPS:]
        else:
            # Pad with first value on the left
            pad_len = TIME_STEPS - len(arr)
            seq[:, v_idx] = np.concatenate([
                np.full(pad_len, arr[0]), arr
            ])

    if not has_any:
        skipped += 1
        continue

    X_seqs.append(seq)
    y_hospital.append(died_hospital[stay_id])
    y_48h.append(died_48h.get(stay_id, 0))
    stay_ids.append(stay_id)

X_seqs     = np.array(X_seqs,     dtype=np.float32)
y_hospital = np.array(y_hospital, dtype=np.float32)
y_48h      = np.array(y_48h,      dtype=np.float32)

print(f"\nSequences built: {len(X_seqs):,}")
print(f"Skipped:         {skipped:,}")
print(f"Shape:           {X_seqs.shape}  "
      f"(stays, time_steps, features)")
print(f"Hospital mortality: {y_hospital.mean()*100:.1f}%")
print(f"48h mortality:      {y_48h.mean()*100:.1f}%")

# ── Save ──────────────────────────────────────────────────────
print(f"\nSaving to {SEQ_PATH}...")
with open(SEQ_PATH, "wb") as f:
    pickle.dump({
        "X":          X_seqs,
        "y_hospital": y_hospital,
        "y_48h":      y_48h,
        "stay_ids":   stay_ids,
        "time_steps": TIME_STEPS,
        "vitals":     VITALS,
        "vital_means":VITAL_MEANS,
        "vital_stds": VITAL_STDS,
    }, f)

print(f"{'='*60}")
print(f"SEQUENCE PREPARATION COMPLETE")
print(f"{'='*60}")
print(f"X shape:   {X_seqs.shape}")
print(f"Sequences: {len(X_seqs):,}")
print(f"\nNext step: python3 train_lstm.py")