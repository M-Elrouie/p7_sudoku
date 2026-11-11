"""Loading the preprocessed digit data + the brightness rule for blank cells."""
from __future__ import annotations

import os
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "digits19.npz")
BLANK_MAX_THRESHOLD = 0.3  # a cell whose brightest pixel is below this is blank


def load_digits(path: str = DATA_PATH) -> dict:
    """Returns dict with x_train/x_val/x_test (N,784) float32 in [0,1] and y_* in 0..8."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run `python make_data.py` first")
    z = np.load(path)
    out = {}
    for k in ("train", "val", "test"):
        out[f"x_{k}"] = (z[f"x_{k}"].reshape(len(z[f"x_{k}"]), -1).astype(np.float32) / 255.0)
        out[f"y_{k}"] = z[f"y_{k}"].astype(np.int64)
    out["source"] = str(z["source"])
    return out


def is_blank(cell_img: np.ndarray) -> bool:
    """Blank-cell rule (NOT the network): no pixel brighter than the threshold."""
    return float(np.max(cell_img)) < BLANK_MAX_THRESHOLD
