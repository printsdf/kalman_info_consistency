import numpy as np
import pytest
from src.consistency import (
    calculate_nees, calculate_nis, check_matrix_properties,
    evaluate_offline, monitor_online, load_config, ConsistencyReport
)
from src.kalman import KalmanFilter

def test_calculate_nees():
    """测试 NEES 计算"""
    x_true = np.array([[1.0], [2.0]])
    x_estimate = np.array([[1.1], [2.2]])
    P = np.diag([0.1, 0.1])
    
    nees = calculate_nees(x_true, x_estimate, P)
    
    # 手动计算预期值
    x_error = x_true - x_estimate
    expected_nees = x_error.T @ np.linalg.inv(P) @ x_error
    
    assert np.isclose(nees, expected_nees)

def test_calculate_nis():
    """测试 NIS 计算"""
    z = np.array([[3.0]])
    z_estimate = np.array([[2.8]])
    S = np.array([[0.01]])
    
    nis = calculate_nis(z, z_estimate, S)
    
    # 手动计算预期值
    innovation = z - z_estimate
    expected_nis = innovation.T @ np.linalg.inv(S) @ innovation
    
    assert np.isclose(nis, expected_nis)

def test_check_matrix_properties():
    """测试矩阵性质检查"""
    # 测试正定矩阵
    P_positive_definite = np.diag([1.0, 1.0])
    is_positive_definite, condition_number = check_matrix_properties(P_positive_definite)
    
    assert is_positive_definite
    assert np.isclose(condition_number, 1.0)
    
    # 测试非正定矩阵
    P_not_positive_definite = np.array([[1.0, 2.0], [2.0, 3.0]])
    is_positive_definite, condition_number = check_matrix_properties(P_not_positive_definite)
    
    assert not is_positive_definite
    assert condition_number > 0

def test_load_config():
    """测试配置文件加载"""
    config = load_config("configs/consistency.yaml")
    
    assert "thresholds" in config
    assert "nees" in config["thresholds"]
    assert "nis" in config["thresholds"]
    assert "condition_number" in config["thresholds"]
    assert "significance_level" in config
    assert "sliding_window" in config
    assert "cold_start" in config

def test_monitor_online():
    """测试在线监控"""
    # 创建简单的测试数据
    x_true = np.array([[1.0], [2.0]])
    x_estimate = np.array([[1.1], [2.2]])
    P = np.diag([0.1, 0.1])
    z = np.array([[3.0]])
    z_estimate = np.array([[2.8]])
    S = np.array([[0.01]])
    
    report = monitor_online(x_true, x_estimate, P, z, z_estimate, S)
    
    assert isinstance(report, ConsistencyReport)
    assert len(report.nees_values) == 1
    assert len(report.nis_values) == 1
    assert len(report.condition_numbers) == 1
    assert report.overall_status in ["PASS", "FAIL"]

def test_evaluate_offline():
    """测试离线评估"""
    # 创建卡尔曼滤波器实例
    kf = KalmanFilter()
    
    # 模拟时间步数
    n_steps = 50
    
    # 初始化历史记录
    x_true_history = []
    x_estimate_history = []
    P_history = []
    z_history = []
    z_estimate_history = []
    S_history = []
    
    # 真实状态初始化
    x_true = np.array([[0.0], [0.0]], dtype=np.float64)
    
    # 过程噪声和测量噪声标准差
    process_noise_std = np.array([[1e-2], [1e-2]], dtype=np.float64)
    measurement_noise_std = 1e-1
    
    # 模拟卡尔曼滤波过程
    for t in range(n_steps):
        # 生成真实状态（带有过程噪声）
        A = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        process_noise = np.random.normal(0, process_noise_std, size=(2, 1))
        x_true = A @ x_true + process_noise
        
        # 生成观测值（带有测量噪声）
        if t % 2 == 0:
            # 偶数时间步观测位置
            H = np.array([[1.0, 0.0]], dtype=np.float64)
        else:
            # 奇数时间步观测速度
            H = np.array([[0.0, 1.0]], dtype=np.float64)
        
        measurement_noise = np.random.normal(0, measurement_noise_std, size=(1, 1))
        z = H @ x_true + measurement_noise
        
        # 卡尔曼滤波预测
        kf.predict()
        
        # 计算卡尔曼增益
        S = H @ kf.P @ H.T + np.array([[measurement_noise_std**2]], dtype=np.float64)
        K = kf.P @ H.T @ np.linalg.inv(S)
        
        # 记录预测状态和协方差
        x_estimate_history.append(kf.x.copy())
        P_history.append(kf.P.copy())
        
        # 计算预测观测值
        z_estimate = H @ kf.x.copy()
        z_estimate_history.append(z_estimate)
        S_history.append(S.copy())
        
        # 卡尔曼滤波更新
        kf.update(z, H, np.array([[measurement_noise_std**2]], dtype=np.float64))
        
        # 记录真实状态和观测值
        x_true_history.append(x_true.copy())
        z_history.append(z.copy())
    
    # 离线评估一致性
    report = evaluate_offline(
        x_true_history=x_true_history,
        x_estimate_history=x_estimate_history,
        P_history=P_history,
        z_history=z_history,
        z_estimate_history=z_estimate_history,
        S_history=S_history
    )
    
    assert isinstance(report, ConsistencyReport)
    assert len(report.nees_values) == n_steps
    assert len(report.nis_values) == n_steps
    assert len(report.condition_numbers) == n_steps
    assert 0 <= report.nees_coverage <= 1
    assert 0 <= report.nis_coverage <= 1
    assert report.overall_status in ["PASS", "FAIL"]

def test_generate_report():
    """测试报告生成"""
    # 创建一个简单的报告
    report = ConsistencyReport(
        nees_values=[1.0, 2.0, 3.0, 4.0, 5.0],
        nis_values=[1.5, 2.5, 3.5, 4.5, 5.5],
        condition_numbers=[1.0, 1.0, 1.0, 1.0, 1.0],
        nees_outliers=[2],
        nis_outliers=[3],
        condition_number_outliers=[],
        nees_coverage=0.8,
        nis_coverage=0.8,
        overall_status="PASS",
        message="测试报告"
    )
    
    # 生成报告
    from src.consistency import generate_report
    generate_report(report)
    
    # 检查报告文件是否存在
    import os
    assert os.path.exists("reports/consistency_report.md")
    assert os.path.exists("reports/consistency_report.html")
    assert os.path.exists("reports/consistency_report.png")
    
    # 清理生成的文件
    os.remove("reports/consistency_report.md")
    os.remove("reports/consistency_report.html")
    os.remove("reports/consistency_report.png")