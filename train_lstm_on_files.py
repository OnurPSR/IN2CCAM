"""
train_lstm_on_files
-------------------
End-to-end LSTM training over sliding-window batches derived from a single
time series stored in Excel/CSV. The script:
  1) builds fixed-length windows (n) with next-step prediction targets,
  2) splits into train/val/test (hold-out through simple slicing),
  3) grid-searches over (num_layers, window length n, hidden_size),
  4) reports best R^2 on test and persists the best model weights.

Notes
-----
- Batching uses overlapping windows via enumerate-based slicing.
- The dataset column name and sheet name are specified below.
- This file intentionally keeps the original structure; comments and
  docstrings were professionalised and redundant imports removed.
"""

from __future__ import annotations

import itertools
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics import R2Score

# ---------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA :{device}")


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class LSTM(nn.Module):
    """Sequence regressor with an LSTM encoder and linear head.

    Parameters
    ----------
    input_size : int
        Feature dimension per time step (1 for univariate series).
    hidden_size : int
        Hidden state size of the LSTM.
    num_layers : int
        Number of stacked LSTM layers.
    output_size : int
        Output dimension (1 for univariate regression).
    device : torch.device
        Target device for initial hidden/cell states.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.Lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial hidden and cell states are allocated on the target device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=self.device)
        out, _ = self.Lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # use the last time-step representation
        return out


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
def dataset_to_array(file_name: str, target_column: str) -> np.ndarray:
    """Read a column into a NumPy array from .xlsx or .csv.

    Parameters
    ----------
    file_name : str
        Path to the data file.
    target_column : str
        Column to extract.

    Returns
    -------
    np.ndarray
        1-D numeric array.
    """
    ext = os.path.splitext(file_name)[1]
    path = os.path.join(os.getcwd(), file_name)
    if ext == ".xlsx":
        df = pd.read_excel(path, sheet_name="3 dklık veri")
        return np.asarray(df[target_column])
    elif ext == ".csv":
        df = pd.read_csv(path)
        return np.asarray(df[target_column])
    else:
        raise ValueError(f"Extension '{ext}' is not supported")


# ---------------------------------------------------------------------
# Windowing and dataset preparation
# ---------------------------------------------------------------------
def array_to_n_len_batches(array: np.ndarray, n: int) -> list[np.ndarray]:
    """Return all overlapping windows of length n from `array`."""
    return [array[index : index + n] for index, _ in enumerate(array)]


def value_count_batches(data: list[np.ndarray]) -> tuple[dict, str, list[int]]:
    """Count batch lengths and mark the first occurrence of each as an outlier.

    This mirrors the original behaviour: the first seen length is flagged
    and removed to keep only repeated, consistent window lengths.
    """
    outlier_indexs = []
    dict1 = {"Batches": {"": ""}}

    for index, batch in enumerate(data):
        key = f"{len(batch)} lenghts of batch"
        if key not in dict1["Batches"]:
            dict1["Batches"][key] = 1
            outlier_indexs.append(index)
        else:
            dict1["Batches"][key] += 1

    del dict1["Batches"][""]
    if outlier_indexs:
        del outlier_indexs[0]
    return dict1, "Outlier Indexs : ", outlier_indexs


def discard_outliers_batches(data: list[np.ndarray], indexs: list[int]) -> np.ndarray:
    """Remove batches at specified indices (returns NumPy array)."""
    extracted_list = []
    for index, element in enumerate(data):
        if index not in indexs:
            extracted_list.append(element)
    return np.array(extracted_list)


def split_to_target_and_label(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 2D array of windows-with-label into (X, y).

    Each row is a window where the last element is the label (y) and the
    preceding elements are the input sequence (X).
    """
    data_copy = list(data)
    temp = []
    for index, batch in enumerate(data_copy):
        temp.append(batch[-1])
        data_copy[index] = data_copy[index][:-1]
    return np.array(data_copy, dtype=np.int64), np.array(temp, np.int64)


def generate_batches(array: np.ndarray, n: int, batch_size: int, device: torch.device):
    """Prepare DataLoaders for training and testing using window length `n`."""
    array_of_batches_n = array_to_n_len_batches(array, n)
    dict1, str1, outliers = value_count_batches(array_of_batches_n)
    array_of_batches_extracted = list(discard_outliers_batches(array_of_batches_n, outliers))

    # Next-step target: take the last element of the *next* window
    array_of_outputs = np.array(
        [array_of_batches_extracted[i + 1][-1] for i in range(len(array_of_batches_extracted) - 1)]
    )
    del array_of_batches_extracted[-1]

    data = np.column_stack((array_of_batches_extracted, array_of_outputs))

    train_size = len(data) * 4 // 5
    val_size = len(data) // 10
    test_size = len(data) - train_size - val_size

    train_data = data[:train_size]
    val_data = data[train_size : train_size + val_size]
    test_data = data[train_size + val_size :]

    train_target, train_label = split_to_target_and_label(train_data)
    val_target, val_label = split_to_target_and_label(val_data)
    test_target, test_label = split_to_target_and_label(test_data)

    sequence_length = n
    input_size = 1

    X_train, X_test, y_train, y_test = train_test_split(
        train_target, train_label, train_size=0.7, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, sequence_length, 1).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).view(-1, sequence_length, 1).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(device)
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ---------------------------------------------------------------------
# MLP builder (kept for parity with original; unused here)
# ---------------------------------------------------------------------
def create_model(input_size: int, hidden_layers: list[int], output_size: int) -> nn.Sequential:
    """Build a simple feed-forward network."""
    model = nn.Sequential()
    last_size = input_size
    for i, layer_size in enumerate(hidden_layers):
        model.add_module(f"Linear_{i}", nn.Linear(last_size, layer_size))
        model.add_module(f"ReLU_{i}", nn.ReLU())
        last_size = layer_size
    model.add_module("Output", nn.Linear(last_size, output_size))
    return model


def all_permutations_subsets(data: list[int], layers: int) -> dict[str, list[tuple[int, ...]]]:
    """Enumerate permutations across all subset sizes up to `layers`."""
    all_perms: dict[str, list[tuple[int, ...]]] = {}
    for r in range(1, layers + 1):
        all_perms[f"{r}"] = []
        for subset in itertools.combinations(data, r):
            for perm in itertools.permutations(subset):
                all_perms[f"{r}"].append(perm)
    return all_perms


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
file_path = os.path.join(os.getcwd(), "Aeropuertomadroa - deneme (1).xlsx")
df = pd.read_excel(file_path, sheet_name="3 dklık veri")
full_array = np.array(df["Hacim (tş/6dk)"])

# Demonstrative windows
array_of_batches = [full_array[index : index + 10] for index, _ in enumerate(full_array)]
array_of_batches_10 = array_to_n_len_batches(full_array, 10)

dict1, str1, outlier_indexs = value_count_batches(array_of_batches_10)
array_of_batches_extracted = discard_outliers_batches(array_of_batches_10, outlier_indexs)
array_of_outputs = np.array(
    [array_of_batches_extracted[i + 1][-1] for i in range(len(array_of_batches_extracted) - 1)]
)
array_of_batches_extracted = array_of_batches_extracted[:-1]

# Sanity check for aligned features/labels
if len(array_of_batches_extracted) == len(array_of_outputs):
    print("---------EQUAL---------")
else:
    print("---------NOT EQUAL---------")
    print(len(array_of_batches_extracted), len(array_of_outputs))
    raise Exception("Window/label length mismatch")


# Build the full design matrix for the hold-out split sizes below
data = np.column_stack((array_of_batches_extracted, array_of_outputs))

train_size = len(data) * 4 // 5
val_size = len(data) // 10
test_size = len(data) - train_size - val_size
print(f"Train      size :\t{train_size}")
print(f"Validation size :\t{val_size}")
print(f"Test       size :\t{test_size}")

train_data = data[:train_size]
val_data = data[train_size : train_size + val_size]
test_data = data[train_size + val_size :]

train_target, train_label = split_to_target_and_label(train_data)
val_target, val_label = split_to_target_and_label(val_data)
test_target, test_label = split_to_target_and_label(test_data)

_ = R2Score()  # placeholder to mirror the original import usage

# ---------------------------------------------------------------------
# Grid-search configuration
# ---------------------------------------------------------------------
input_sizes = np.arange(3, 10)      # window lengths
max_layers = 4                      # stacked LSTM layers
batch_size = 64
num_epochs = 100
neuron_counts = all_permutations_subsets(list(np.arange(5, 15, 1)), max_layers)  # kept (unused)
learning_rate = 0.01
output_size = 1
hidden_sizes = np.arange(5, 50, 5)

best_models = {}
best_r2_scores = []

# ---------------------------------------------------------------------
# Grid-search over (num_layers, n, hidden_size)
# ---------------------------------------------------------------------
lstm_model_index = 0
best_model_obj = None
best_score = -float("inf")
best_cfg = {"num_layers": None, "n": None, "hidden_size": None, "mse": None, "r2": None}

for num_layers in range(1, max_layers + 1):
    for n in input_sizes:
        for hidden_size in hidden_sizes:
            model = LSTM(input_size=1, hidden_size=hidden_size, output_size=output_size,
                         num_layers=num_layers, device=device)
            train_load, test_load = generate_batches(array=full_array, batch_size=batch_size, n=int(n), device=device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            best_mse = float("inf")
            for epoch in range(num_epochs):
                epoch_losses = []
                model.train()
                for inputs, labels in train_load:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(loss.item())
                epoch_mse = sum(epoch_losses) / len(epoch_losses)
                if epoch_mse < best_mse:
                    best_mse = epoch_mse

            with torch.no_grad():
                model.eval()
                r2_tests = []
                for X_test, y_test in test_load:
                    y_pred_test = model(X_test.to(device))
                    r2_tests.append(r2_score(y_test.cpu().numpy(), y_pred_test.cpu().numpy()))
                best_r2_test = float(np.max(r2_tests)) if r2_tests else -float("inf")

            best_r2_scores.append((f"model_{lstm_model_index}", model, best_r2_test, num_layers, int(n), best_mse))
            best_models[f"model_{lstm_model_index}"] = {
                str(num_layers): num_layers,
                str(int(n)): int(n),
                str(hidden_size): hidden_size,
                str(best_r2_test): best_r2_test,
                str(best_mse): best_mse,
            }
            print(
                f"No. of Layers: {num_layers}, Input Size {int(n)}, Hidden Size: {hidden_size}, "
                f"Best R2 Score (Test): {best_r2_test}, Best MSE: {best_mse}"
            )
            if best_r2_test > best_score:
                best_score = best_r2_test
                best_model_obj = model
                best_cfg.update(
                    {"num_layers": num_layers, "n": int(n), "hidden_size": hidden_size, "mse": best_mse, "r2": best_r2_test}
                )
            lstm_model_index += 1

# ---------------------------------------------------------------------
# Persist best model
# ---------------------------------------------------------------------
if best_model_obj is None:
    raise RuntimeError("No model trained successfully; best_model_obj is None.")

save_path = os.path.join(os.getcwd(), "LSTM_Aeropuertomadroa - deneme (1).pt")
torch.save(best_model_obj.state_dict(), save_path)

print(
    f"No. of Layers: {best_cfg['num_layers']},\n"
    f"      Input Size: {best_cfg['n']},\n"
    f"      Hidden Size: {best_cfg['hidden_size']},\n"
    f"      Best R2 Score (Test): {best_cfg['r2']},\n"
    f"      Best MSE: {best_cfg['mse']}"
)
