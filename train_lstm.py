import numpy as np
import pickle
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import os

# ═══════════════════════════════════════════════════════════════
# LSTM MORTALITY PREDICTOR
#
# Architecture: 2-layer Bidirectional LSTM + Dropout + FC
# Device: Apple Silicon MPS (GPU) if available
#
# Trains twice:
#   Run 1 — hospital_expire_flag (compare with XGBoost)
#   Run 2 — 48h mortality (clinically actionable)
#
# Reference: Hochreiter & Schmidhuber.
#   "Long Short-Term Memory." Neural Computation 1997.
#   doi:10.1162/neco.1997.9.8.1735
# ═══════════════════════════════════════════════════════════════

SEQ_PATH   = "/Users/malcolmdivinec/Documents/icu-monitoring-system/sequences.pkl"
MODEL_DIR  = "/Users/malcolmdivinec/Documents/icu-monitoring-system/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Device ────────────────────────────────────────────────────
device = (torch.device("mps")
          if torch.backends.mps.is_available()
          else torch.device("cpu"))
print(f"Device: {device}")

# ── Load Sequences ────────────────────────────────────────────
print("\nLoading sequences...")
with open(SEQ_PATH, "rb") as f:
    data = pickle.load(f)

X          = data["X"]
y_hospital = data["y_hospital"]
y_48h      = data["y_48h"]
print(f"X shape: {X.shape}")
print(f"Hospital mortality: {y_hospital.mean()*100:.1f}%")
print(f"48h mortality:      {y_48h.mean()*100:.1f}%")

# ═══════════════════════════════════════════════════════════════
# LSTM ARCHITECTURE
# ═══════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════
# TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════
def train_model(X, y, label_name, model_path,
                epochs=50, batch_size=256, lr=0.001):

    print(f"\n{'='*60}")
    print(f"TRAINING LSTM — Label: {label_name}")
    print(f"{'='*60}")
    print(f"Positive rate: {y.mean()*100:.1f}%")

    # Train/test split — same seed as XGBoost for fair comparison
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y.astype(int)
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t  = torch.FloatTensor(X_test).to(device)
    y_test_t  = torch.FloatTensor(y_test).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True)

    # Model
    model     = ICU_LSTM().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_auc   = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            preds = model(xb)
            loss  = criterion(preds, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_t).cpu().numpy()
            y_test_np  = y_test_t.cpu().numpy()

        try:
            auc = roc_auc_score(y_test_np, test_preds)
        except Exception:
            auc = 0.5

        scheduler.step(total_loss)

        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {total_loss/len(train_dl):.4f} | "
                  f"AUC: {auc:.4f} | Best: {best_auc:.4f}")

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        final_preds = model(X_test_t).cpu().numpy()

    final_auc = roc_auc_score(y_test_np, final_preds)
    binary_preds = (final_preds >= 0.5).astype(int)

    print(f"\nFinal AUC-ROC: {final_auc:.4f}")
    print(classification_report(
        y_test_np.astype(int), binary_preds,
        target_names=["Survived", "Died"]
    ))

    # Save model bundle
    bundle = {
        "model_state":  best_state,
        "model_config": {
            "input_size":  5,
            "hidden_size": 64,
            "num_layers":  2,
            "dropout":     0.3
        },
        "auc":          final_auc,
        "label":        label_name,
        "time_steps":   24,
        "vitals":       data["vitals"],
        "vital_means":  data["vital_means"],
        "vital_stds":   data["vital_stds"],
        "trained_on":   len(X_train),
    }
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Model saved: {model_path}")

    return final_auc

# ═══════════════════════════════════════════════════════════════
# RUN 1 — Hospital Mortality (compare with XGBoost)
# ═══════════════════════════════════════════════════════════════
auc_hospital = train_model(
    X, y_hospital,
    label_name  = "hospital_mortality",
    model_path  = f"{MODEL_DIR}/lstm_hospital.pkl",
    epochs      = 50,
    batch_size  = 256,
    lr          = 0.001
)

# ═══════════════════════════════════════════════════════════════
# RUN 2 — 48h Mortality (clinically actionable)
# ═══════════════════════════════════════════════════════════════
auc_48h = train_model(
    X, y_48h,
    label_name  = "48h_mortality",
    model_path  = f"{MODEL_DIR}/lstm_48h.pkl",
    epochs      = 50,
    batch_size  = 256,
    lr          = 0.001
)

# ── Final Summary ─────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"LSTM TRAINING COMPLETE")
print(f"{'='*60}")
print(f"Hospital mortality AUC: {auc_hospital:.4f}")
print(f"48h mortality AUC:      {auc_48h:.4f}")
print(f"\nReference: Hochreiter & Schmidhuber.")
print(f"  Long Short-Term Memory. Neural Computation 1997.")
print(f"\nNext step: python3 compare_models.py")

# Save summary
summary = {
    "lstm_hospital_auc": round(auc_hospital, 4),
    "lstm_48h_auc":      round(auc_48h, 4),
    "architecture":      "2-layer Bidirectional LSTM",
    "hidden_size":       64,
    "time_steps":        24,
    "trained_on":        int(len(X) * 0.8),
}
with open(f"{MODEL_DIR}/lstm_metrics.json", "w") as f:
    json.dump(summary, f, indent=2)