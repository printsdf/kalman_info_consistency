from __future__ import annotations

import numpy as np

from .covariance import cov_update
from .linalg_utils import as_float64, ensure_row_observation


class KalmanFilter:
    def __init__(self):
        self.n = 2
        self.A = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        self.Q = np.diag([1e-5, 1e-5]).astype(np.float64)
        self.x = np.zeros((self.n, 1), dtype=np.float64)
        self.P = np.diag([1.0, 1.0]).astype(np.float64)

    def reset(self, x0: np.ndarray | None = None, P0: np.ndarray | None = None) -> None:
        if x0 is not None:
            x0 = as_float64(x0)
            self.x = x0.reshape(self.n, 1)
        else:
            self.x = np.zeros((self.n, 1), dtype=np.float64)
        if P0 is not None:
            self.P = as_float64(P0)
        else:
            self.P = np.diag([1.0, 1.0]).astype(np.float64)

    def predict(self) -> None:
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        H = ensure_row_observation(H)
        R = as_float64(R)
        
        # 计算卡尔曼增益
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # 更新状态估计
        innovation = z - H @ self.x
        self.x = self.x + K @ innovation
        
        # 更新协方差矩阵
        self.P = cov_update(self.P, K, H, R)
