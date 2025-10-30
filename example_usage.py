import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.kalman import KalmanFilter
from src.consistency import evaluate_offline, generate_report

# 创建卡尔曼滤波器实例
kf = KalmanFilter()

# 模拟时间步数
n_steps = 200

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
    
    # 卡尔曼滤波更新
    kf.update(z, H, np.array([[measurement_noise_std**2]], dtype=np.float64))
    
    # 记录历史数据
    x_true_history.append(x_true.copy())
    x_estimate_history.append(kf.x.copy())
    P_history.append(kf.P.copy())
    z_history.append(z.copy())
    z_estimate_history.append(H @ kf.x.copy())
    S_history.append(S.copy())

# 离线评估一致性
report = evaluate_offline(
    x_true_history=x_true_history,
    x_estimate_history=x_estimate_history,
    P_history=P_history,
    z_history=z_history,
    z_estimate_history=z_estimate_history,
    S_history=S_history
)

# 生成报告
generate_report(report)

# 打印报告摘要
print("一致性评估报告摘要:")
print(f"总体状态: {report.overall_status}")
print(f"消息: {report.message}")
print(f"NEES 覆盖率: {report.nees_coverage:.2%}")
print(f"NIS 覆盖率: {report.nis_coverage:.2%}")
print(f"NEES 异常点数量: {len(report.nees_outliers)}")
print(f"NIS 异常点数量: {len(report.nis_outliers)}")
print(f"条件数异常点数量: {len(report.condition_number_outliers)}")