from __future__ import annotations

import numpy as np


def cov_update(P: np.ndarray, K: np.ndarray, H: np.ndarray, R: np.ndarray | None = None, mode: str = "simple") -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    if H.ndim == 1:
        H = H.reshape(1, -1)
    elif H.ndim == 2 and H.shape[0] != 1 and H.shape[1] == 1:
        H = H.T
    if R is None:
        R = np.zeros((1, 1), dtype=np.float64)
    else:
        R = np.asarray(R, dtype=np.float64)

    I = np.eye(P.shape[0], dtype=np.float64)
    # Joseph form
    return (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T

