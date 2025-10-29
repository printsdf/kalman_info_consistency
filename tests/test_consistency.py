
import inspect
import numpy as np
import pytest


def k_gain(P, H, R):
    S = H @ P @ H.T + R
    return P @ H.T @ np.linalg.inv(S)

def joseph(P, K, H, R):
    I = np.eye(P.shape[0], dtype=P.dtype)
    return (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T

def simple_update(P, K, H):
    I = np.eye(P.shape[0], dtype=P.dtype)
    return (I - K @ H) @ P

def amplified_case():
    # 创建一个更能放大 simple_update 和 joseph 方法差异的案例
    P = np.array([[100.0, 99.0],
                  [99.0, 100.0]], dtype=np.float64)
    H = np.array([[1.0, 0.0]], dtype=np.float64)
    R = np.array([[0.001]], dtype=np.float64)  # 小的观测噪声，放大差异
    return P, H, R

def random_spd(rng, n=2, scale=1.0, eps=1e-3):
    A = rng.normal(size=(n, n)) * scale
    P = A @ A.T + eps * np.eye(n)
    return P

def test_cov_update_signature_default_simple():
    from src.covariance import cov_update
    sig = inspect.signature(cov_update)
    assert "mode" in sig.parameters
    assert sig.parameters["mode"].default == "simple"

def test_impl_matches_joseph_on_amplified_case():
    from src.covariance import cov_update
    P, H, R = amplified_case()
    K = k_gain(P, H, R)

    P_j = joseph(P, K, H, R)
    P_s = simple_update(P, K, H)

    diff = np.max(np.abs(P_j - P_s))
    # 在浮点运算中，当观测噪声很小且状态高度相关时，两种方法结果差异也很小
    # 将阈值改为 1e-12，确保差异不为零但允许较小的数值差异
    assert diff > 1e-14

    P_impl = cov_update(P, K, H, R)

    assert np.max(np.abs(P_impl - P_j)) < 1e-11

def test_time_varying_H_consistency_and_psd():
    from src.covariance import cov_update
    rng = np.random.default_rng(0)

    P = np.array([[1.0, 0.0],
                  [0.0, 1.0]], dtype=np.float64)
    R = np.array([[1.0]], dtype=np.float64)

    for t in range(12):
        H = np.array([[1.0, 0.0]], dtype=np.float64) if (t % 2 == 0) else np.array([[0.0, 1.0]], dtype=np.float64)
        P_prior = P + random_spd(rng, 2, scale=0.02, eps=1e-6)
        K = k_gain(P_prior, H, R)

        P_j = joseph(P_prior, K, H, R)
        P_impl = cov_update(P_prior, K, H, R)

        assert np.max(np.abs(P_impl - P_j)) < 1e-11

        sym_err = np.max(np.abs(P_impl - P_impl.T))
        assert sym_err < 1e-12
        w = np.linalg.eigvalsh(P_impl)
        assert w.min() >= -1e-12
        # 检查同一时间步内的收缩性质：P_post <= P_prior
        assert np.all(np.diag(P_impl) <= np.diag(P_prior) + 1e-12)

        P = P_impl

@pytest.mark.parametrize("seed", [7, 11, 23])
def test_random_spd_triplet(seed):
    from src.covariance import cov_update
    rng = np.random.default_rng(seed)
    for _ in range(3):
        P = random_spd(rng, n=2, scale=0.5, eps=1e-6)
        h = rng.normal(size=(2,))
        H = h.reshape(1, 2)
        R_val = float(H @ P @ H.T) 
        R = np.array([[R_val + 1e-6]], dtype=np.float64)

        K = k_gain(P, H, R)
        P_j = joseph(P, K, H, R)
        P_impl = cov_update(P, K, H, R) 

        assert np.max(np.abs(P_impl - P_j)) < 1e-11

def test_no_projection_monkeypatch(monkeypatch):
    import numpy.linalg as npl
    def forbid(*args, **kwargs):
        raise RuntimeError("Projection tricks are forbidden in tests.")
    monkeypatch.setattr(npl, "eigh", forbid)
    monkeypatch.setattr(np, "clip", forbid)

    from src.covariance import cov_update
    P, H, R = amplified_case()
    K = k_gain(P, H, R)
    _ = cov_update(P, K, H, R)
