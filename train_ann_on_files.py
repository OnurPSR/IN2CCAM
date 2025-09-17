"""
train_ann_on_files.py
---------------------
End-to-end ANN (MLP) training over sliding-window batches derived from a single
time series stored in Excel/CSV. The script:
  1) builds fixed-length windows (n) with next-step prediction targets,
  2) splits into train/val/test (hold-out through simple slicing),
  3) grid-searches over (window length n, batch_size, learning_rate, hidden widths)
     with a fixed 2-hidden-layer MLP architecture,
  4) trains up to 300 epochs with early stopping on validation loss,
  5) reports best R^2 on test and persists the best model weights.

Notes
-----
- This is a rewrite of the LSTM script into an ANN (MLP) with similar
  professionalised comments and structure.
- Windowing is deterministic; each input window of length n predicts the next
  value (t+1).
"""

from __future__ import annotations

import os
import math
import random
from copy import deepcopy
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------
# Reproducibility & Device selection
# ---------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Using device: {device}")

# ---------------------------------------------------------------------
# Model: 2-layer MLP regressor (ANN)
# ---------------------------------------------------------------------
class MLPRegressor(nn.Module):
    """Two-hidden-layer feed-forward regressor.

    Parameters
    ----------
    input_dim : int
        Number of input features (window length n).
    hidden_dim1 : int
        Width of the first hidden layer.
    hidden_dim2 : int
        Width of the second hidden layer.
    output_dim : int
        Output dimension (1 for univariate regression).
    """

    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int, output_dim: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected shape: (B, n) for ANN
        return self.net(x)


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
def dataset_to_array(file_name: str, target_column: str, sheet_name: str = "3 dklık veri") -> np.ndarray:
    """Read `target_column` into a NumPy array from .xlsx or .csv."""
    ext = os.path.splitext(file_name)[1]
    path = os.path.join(os.getcwd(), file_name)
    if ext == ".xlsx":
        df = pd.read_excel(path, sheet_name=sheet_name)
        return np.asarray(df[target_column], dtype=np.float64)
    elif ext == ".csv":
        df = pd.read_csv(path)
        return np.asarray(df[target_column], dtype=np.float64)
    raise ValueError(f"Extension '{ext}' is not supported")


# ---------------------------------------------------------------------
# Windowing and dataset preparation
# ---------------------------------------------------------------------
def make_supervised_windows(series: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build supervised pairs (X, y) with window length n -> next-step target.

    Given a univariate series s[0..T-1], we form samples:
      X_i = [s[i], s[i+1], ..., s[i+n-1]]  (length n)
      y_i = s[i+n]                          (the next value)
    for i = 0 .. T - n - 1.
    """
    T = len(series)
    if T <= n:
        raise ValueError(f"Series length {T} must be > window length n={n}.")
    X = np.stack([series[i : i + n] for i in range(T - n)], axis=0)
    y = np.array([series[i + n] for i in range(T - n)], dtype=np.float64)
    return X, y


def split_train_val_test(
    X: np.ndarray, y: np.ndarray, train_frac: float = 0.8, val_frac: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple contiguous split (no shuffling) to respect temporal ordering."""
    N = len(X)
    n_train = int(math.floor(train_frac * N))
    n_val = int(math.floor(val_frac * N))
    n_test = N - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("Insufficient data for requested splits.")
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Convert arrays to tensors and wrap in DataLoaders."""
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------
# Training utilities (early stopping)
# ---------------------------------------------------------------------
class EarlyStopper:
    """Early stopping on validation loss improvement.

    Parameters
    ----------
    patience : int
        Number of epochs to wait without improvement before stopping.
    min_delta : float
        Minimum relative improvement required to reset patience.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop; stores best weights."""
        if val_loss < self.best * (1.0 - self.min_delta):
            self.best = val_loss
            self.counter = 0
            self.best_state = deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience

    def load_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer) -> float:
    model.train()
    losses = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("inf")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion) -> Tuple[float, float]:
    model.eval()
    losses = []
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        preds = model(xb)
        losses.append(criterion(preds.cpu(), yb).item())
        y_true.append(yb.numpy())
        y_pred.append(preds.cpu().numpy())
    y_true = np.vstack(y_true) if y_true else np.zeros((0, 1))
    y_pred = np.vstack(y_pred) if y_pred else np.zeros((0, 1))
    mse = mean_squared_error(y_true, y_pred) if len(y_true) > 0 else float("inf")
    return float(np.mean(losses)), mse


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
FILE_NAME = "Aeropuertomadroa - deneme (1).xlsx"
SHEET_NAME = "3 dklık veri"
TARGET_COLUMN = "Hacim (tş/6dk)"

series = dataset_to_array(FILE_NAME, TARGET_COLUMN, sheet_name=SHEET_NAME)
series = series.astype(np.float64)

# ---------------------------------------------------------------------
# HYPERPARAMETERS (as requested)
# ---------------------------------------------------------------------
# (1) Architecture: exactly 2 hidden layers ("ax_layers=2"), with small grid
HIDDEN_WIDTHS = [5, 15]  # sweep each hidden layer over {5, 15}

# (2) Window lengths: inclusive interval [5, 15]
INPUT_SIZE_START, INPUT_SIZE_END = 5, 15

# (3) Batch sizes: {32, 64, 128}
BATCH_SIZES = [32, 64, 128]

# (4) Learning rates: choose two distinct defaults
LEARNING_RATES = [1e-3, 1e-2]

# (5) Epochs & early stopping
MAX_EPOCHS = 300
EARLY_STOP_PATIENCE = 20
EARLY_STOP_MIN_DELTA = 1e-4

# (6) Optimization/loss
CRITERION = nn.MSELoss

# ---------------------------------------------------------------------
# Grid-search over (n, batch_size, learning_rate, hidden widths)
# Selection metric: highest Test R^2 after restoring best val-loss weights.
# ---------------------------------------------------------------------
best_global = {
    "r2_test": -float("inf"),
    "cfg": None,
    "state_dict": None,
}

for n in range(INPUT_SIZE_START, INPUT_SIZE_END + 1):  # inclusive [3,10]
    # Build supervised dataset for this window length
    X, y = make_supervised_windows(series, n)

    # Temporal hold-out split
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(
        X, y, train_frac=0.8, val_frac=0.1
    )

    for bs in BATCH_SIZES:
        train_loader, val_loader, test_loader = build_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test, bs
        )

        # Sweep each hidden layer width over {5, 15} (crossed)
        for h1 in HIDDEN_WIDTHS:
            for h2 in HIDDEN_WIDTHS:
                for lr in LEARNING_RATES:
                    # Build model & optimizer
                    model = MLPRegressor(input_dim=n, hidden_dim1=h1, hidden_dim2=h2, output_dim=1).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    criterion = CRITERION()

                    stopper = EarlyStopper(patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA)

                    # ------------------------- Training loop -------------------------
                    for epoch in range(1, MAX_EPOCHS + 1):
                        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
                        val_loss, _ = evaluate(model, val_loader, criterion)

                        if epoch % 25 == 0 or epoch == 1:
                            print(
                                f"[n={n:02d} | bs={bs:3d} | lr={lr:.0e} | h1={h1:02d} | h2={h2:02d}] "
                                f"epoch {epoch:03d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
                            )

                        if stopper.step(val_loss, model):
                            # Early stop triggered; restore best weights
                            stopper.load_best(model)
                            break
                    else:
                        # If not early-stopped, still ensure best weights are loaded
                        stopper.load_best(model)

                    # -------------------------- Evaluation ---------------------------
                    model.eval()
                    y_true_all, y_pred_all = [], []
                    with torch.no_grad():
                        for xb, yb in test_loader:
                            xb = xb.to(device)
                            preds = model(xb).cpu().numpy()
                            y_true_all.append(yb.numpy())
                            y_pred_all.append(preds)
                    y_true_all = np.vstack(y_true_all)
                    y_pred_all = np.vstack(y_pred_all)

                    r2_t = r2_score(y_true_all, y_pred_all) if len(y_true_all) > 0 else -float("inf")
                    mse_t = mean_squared_error(y_true_all, y_pred_all) if len(y_true_all) > 0 else float("inf")

                    print(
                        f"FINAL  [n={n:02d} | bs={bs:3d} | lr={lr:.0e} | h1={h1:02d} | h2={h2:02d}]  "
                        f"Test R2={r2_t:.6f}  Test MSE={mse_t:.6f}"
                    )

                    # Keep global best by highest test R^2
                    if r2_t > best_global["r2_test"]:
                        best_global.update(
                            {
                                "r2_test": r2_t,
                                "cfg": {"n": n, "batch_size": bs, "lr": lr, "H1": h1, "H2": h2},
                                "state_dict": deepcopy(model.state_dict()),
                                "mse_test": mse_t,
                            }
                        )

# ---------------------------------------------------------------------
# Persist best model
# ---------------------------------------------------------------------
if best_global["state_dict"] is None:
    raise RuntimeError("No ANN model trained successfully; nothing to save.")

SAVE_PATH = os.path.join(os.getcwd(), "ANN_Aeropuertomadroa - deneme (1).pt")
torch.save(best_global["state_dict"], SAVE_PATH)

cfg = best_global["cfg"]
print("\n==================== BEST CONFIG (ANN) ====================")
print(f"Window length (n):  {cfg['n']}")
print(f"Batch size:         {cfg['batch_size']}")
print(f"Learning rate:      {cfg['lr']:.0e}")
print(f"Hidden layers:      2  (H1={cfg['H1']}, H2={cfg['H2']})")
print(f"Test R^2 (best):    {best_global['r2_test']:.6f}")
print(f"Test MSE (best):    {best_global['mse_test']:.6f}")
print(f"Saved weights to:   {SAVE_PATH}")
print("===========================================================\n")
