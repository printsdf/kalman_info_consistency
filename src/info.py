from __future__ import annotations

import numpy as np

from .linalg_utils import (
    as_float64,
    ensure_row_observation,
)


def info_update(Y_prior: np.ndarray, H: np.ndarray, R: np.ndarray) -> np.ndarray:
    
    Y = as_float64(Y_prior)
    H = ensure_row_observation(H)
    R = as_float64(R)

    R_inv = 1.0 / R
    return Y + H.T @ (R_inv @ H)
