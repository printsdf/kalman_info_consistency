import numpy as np


def as_float64(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def ensure_row_observation(H: np.ndarray) -> np.ndarray:
    H = as_float64(H)
    if H.ndim == 1:
        H = H.reshape(1, -1)
    elif H.ndim == 2 and H.shape[0] != 1 and H.shape[1] == 1:
        H = H.T
    return H


def make_spd(n: int, rng: np.random.Generator, eps: float = 1e-8) -> np.ndarray:
    A = rng.standard_normal((n, n))
    P = A @ A.T
    P.flat[:: n + 1] += eps
    return as_float64(P)


def is_symmetric(a: np.ndarray, tol: float = 1e-12) -> bool:
    a = as_float64(a)
    return np.max(np.abs(a - a.T)) <= tol


def stable_inverse_spd(Y: np.ndarray) -> np.ndarray:
    Y = as_float64(Y)
    I = np.eye(Y.shape[0], dtype=np.float64)
    return np.linalg.solve(Y, I)


def assert_shapes_cov(P: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
    P = as_float64(P)
    H = as_float64(H)
    R = as_float64(R)
    assert P.ndim == 2 and P.shape[0] == P.shape[1]
    assert H.ndim == 2 and H.shape[0] == 1
    assert R.ndim == 2 and R.shape == (1, 1)
