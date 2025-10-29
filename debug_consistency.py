import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.covariance import cov_update
from src.info import info_update

# Helper functions
def k_gain(P, H, R):
    S = H @ P @ H.T + R
    return P @ H.T @ np.linalg.inv(S)

def joseph(P, K, H, R):
    I = np.eye(P.shape[0], dtype=P.dtype)
    return (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T

def amplified_case():
    # 创建一个更能放大 simple_update 和 joseph 方法差异的案例
    P = np.array([[100.0, 99.0],
                  [99.0, 100.0]], dtype=np.float64)
    H = np.array([[1.0, 0.0]], dtype=np.float64)
    R = np.array([[0.001]], dtype=np.float64)  # 小的观测噪声，放大差异
    return P, H, R

# 测试一致性
def test_amplified_case():
    print("Testing amplified case...")
    P, H, R = amplified_case()
    K = k_gain(P, H, R)
    
    # Joseph 形式更新
    P_j = joseph(P, K, H, R)
    print("Joseph update result:")
    print(P_j)
    
    # 协方差更新
    P_cov = cov_update(P, K, H, R)
    print("\nCovariance update result:")
    print(P_cov)
    
    # 信息矩阵更新 - 需要转换
    Y_prior = np.linalg.inv(P)
    Y_post = info_update(Y_prior, H, R)
    P_info = np.linalg.inv(Y_post)
    print("\nInfo update result (converted to covariance):")
    print(P_info)
    
    # 计算差异
    diff_joseph_cov = np.max(np.abs(P_j - P_cov))
    diff_joseph_info = np.max(np.abs(P_j - P_info))
    
    print(f"\nDifference Joseph vs Covariance: {diff_joseph_cov:.10f}")
    print(f"Difference Joseph vs Info (converted): {diff_joseph_info:.10f}")
    
    # 打印信息矩阵的直接更新结果
    print(f"\nInfo matrix before: {Y_prior}")
    print(f"Info matrix after: {Y_post}")
    
    # 计算正确的信息矩阵更新应该是什么
    # Y_post = Y_prior + H.T @ R_inv @ H
    R_inv = np.linalg.inv(R)
    Y_post_correct = Y_prior + H.T @ R_inv @ H
    print(f"\nCorrect info matrix after: {Y_post_correct}")
    print(f"Difference in info matrix: {np.max(np.abs(Y_post - Y_post_correct)):.10f}")

if __name__ == "__main__":
    test_amplified_case()