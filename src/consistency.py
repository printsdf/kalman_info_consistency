from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import ruamel.yaml as yaml
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

@dataclass
class ConsistencyReport:
    """一致性报告类，存储评估结果"""
    nees_values: List[float] = field(default_factory=list)
    nis_values: List[float] = field(default_factory=list)
    condition_numbers: List[float] = field(default_factory=list)
    nees_outliers: List[int] = field(default_factory=list)
    nis_outliers: List[int] = field(default_factory=list)
    condition_number_outliers: List[int] = field(default_factory=list)
    nees_coverage: float = 0.0
    nis_coverage: float = 0.0
    overall_status: str = ""
    message: str = ""

def load_config(config_path: str = "configs/consistency.yaml") -> Dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        yaml_obj = yaml.YAML(typ='safe', pure=True)
        config = yaml_obj.load(f)
    return config


def calculate_nees(x_true: np.ndarray, x_estimate: np.ndarray, P: np.ndarray) -> float:
    """计算 NEES (Normalized Estimation Error Squared)"""
    x_error = x_true - x_estimate
    nees = x_error.T @ np.linalg.inv(P) @ x_error
    return float(nees.item())


def calculate_nis(z: np.ndarray, z_estimate: np.ndarray, S: np.ndarray) -> float:
    """计算 NIS (Normalized Innovation Squared)"""
    innovation = z - z_estimate
    nis = innovation.T @ np.linalg.inv(S) @ innovation
    return float(nis.item())


def check_matrix_properties(P: np.ndarray) -> Tuple[bool, float]:
    """检查矩阵的正定性和条件数"""
    # 检查正定性
    is_positive_definite = np.all(np.linalg.eigvalsh(P) > 0)
    
    # 计算条件数
    condition_number = np.linalg.cond(P)
    
    return is_positive_definite, condition_number


def evaluate_offline(
    x_true_history: List[np.ndarray],
    x_estimate_history: List[np.ndarray],
    P_history: List[np.ndarray],
    z_history: List[np.ndarray],
    z_estimate_history: List[np.ndarray],
    S_history: List[np.ndarray],
    config_path: str = "configs/consistency.yaml"
) -> ConsistencyReport:
    """离线评估卡尔曼滤波一致性"""
    # 加载配置
    config = load_config(config_path)
    thresholds = config['thresholds']
    significance_level = config['significance_level']
    cold_start = config['cold_start']
    
    report = ConsistencyReport()
    
    # 计算各项指标
    for i in range(len(x_true_history)):
        # 计算 NEES
        nees = calculate_nees(x_true_history[i], x_estimate_history[i], P_history[i])
        report.nees_values.append(nees)
        
        # 计算 NIS
        nis = calculate_nis(z_history[i], z_estimate_history[i], S_history[i])
        report.nis_values.append(nis)
        
        # 检查信息矩阵性质
        is_positive_definite, condition_number = check_matrix_properties(P_history[i])
        report.condition_numbers.append(condition_number)
        
        # 检测异常值
        if i >= cold_start:
            if nees > thresholds['nees']['upper'] or nees < thresholds['nees']['lower']:
                report.nees_outliers.append(i)
            
            if nis > thresholds['nis']['upper'] or nis < thresholds['nis']['lower']:
                report.nis_outliers.append(i)
            
            if condition_number > thresholds['condition_number']['upper']:
                report.condition_number_outliers.append(i)
    
    # 计算覆盖率
    if len(report.nees_values) > cold_start:
        nees_within_threshold = sum(1 for nees in report.nees_values[cold_start:] 
                                    if thresholds['nees']['lower'] <= nees <= thresholds['nees']['upper'])
        report.nees_coverage = nees_within_threshold / (len(report.nees_values) - cold_start)
        
        nis_within_threshold = sum(1 for nis in report.nis_values[cold_start:] 
                                   if thresholds['nis']['lower'] <= nis <= thresholds['nis']['upper'])
        report.nis_coverage = nis_within_threshold / (len(report.nis_values) - cold_start)
    
    # 生成总体结论
    if (report.nees_coverage > 1 - significance_level and 
        report.nis_coverage > 1 - significance_level and 
        len(report.condition_number_outliers) == 0):
        report.overall_status = "PASS"
        report.message = "卡尔曼滤波一致性良好，所有指标均在可接受范围内。"
    else:
        report.overall_status = "FAIL"
        report.message = "卡尔曼滤波一致性不满足要求，存在异常指标。"
    
    return report


def monitor_online(
    x_true: np.ndarray,
    x_estimate: np.ndarray,
    P: np.ndarray,
    z: np.ndarray,
    z_estimate: np.ndarray,
    S: np.ndarray,
    config_path: str = "configs/consistency.yaml"
) -> ConsistencyReport:
    """在线监控卡尔曼滤波一致性"""
    # 加载配置
    config = load_config(config_path)
    thresholds = config['thresholds']
    
    report = ConsistencyReport()
    
    # 计算各项指标
    nees = calculate_nees(x_true, x_estimate, P)
    report.nees_values.append(nees)
    
    nis = calculate_nis(z, z_estimate, S)
    report.nis_values.append(nis)
    
    is_positive_definite, condition_number = check_matrix_properties(P)
    report.condition_numbers.append(condition_number)
    
    # 检测异常值
    if nees > thresholds['nees']['upper'] or nees < thresholds['nees']['lower']:
        report.nees_outliers.append(0)
    
    if nis > thresholds['nis']['upper'] or nis < thresholds['nis']['lower']:
        report.nis_outliers.append(0)
    
    if condition_number > thresholds['condition_number']['upper']:
        report.condition_number_outliers.append(0)
    
    # 生成总体结论
    if (len(report.nees_outliers) == 0 and 
        len(report.nis_outliers) == 0 and 
        len(report.condition_number_outliers) == 0):
        report.overall_status = "PASS"
        report.message = "当前时刻卡尔曼滤波一致性良好。"
    else:
        report.overall_status = "FAIL"
        report.message = "当前时刻卡尔曼滤波一致性不满足要求，存在异常指标。"
    
    return report


def generate_visualization(
    report: ConsistencyReport,
    config_path: str = "configs/consistency.yaml"
) -> plt.Figure:
    """生成可视化报告"""
    # 加载配置
    config = load_config(config_path)
    thresholds = config['thresholds']
    visualization_config = config['visualization']
    
    # 创建画布
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=visualization_config['figure_size'], dpi=visualization_config['dpi'])
    
    # 绘制 NEES 曲线
    ax1.plot(report.nees_values, color=visualization_config['colors']['nees'], label='NEES')
    ax1.axhline(y=thresholds['nees']['upper'], color=visualization_config['colors']['threshold'], linestyle='--', label='Upper Threshold')
    ax1.axhline(y=thresholds['nees']['lower'], color=visualization_config['colors']['threshold'], linestyle='--', label='Lower Threshold')
    
    # 标记异常点
    nees_outlier_values = [report.nees_values[i] for i in report.nees_outliers]
    ax1.scatter(report.nees_outliers, nees_outlier_values, color=visualization_config['colors']['anomaly'], label='Outliers')
    
    ax1.set_title('Normalized Estimation Error Squared (NEES)')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('NEES Value')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制 NIS 曲线
    ax2.plot(report.nis_values, color=visualization_config['colors']['nis'], label='NIS')
    ax2.axhline(y=thresholds['nis']['upper'], color=visualization_config['colors']['threshold'], linestyle='--', label='Upper Threshold')
    ax2.axhline(y=thresholds['nis']['lower'], color=visualization_config['colors']['threshold'], linestyle='--', label='Lower Threshold')
    
    # 标记异常点
    nis_outlier_values = [report.nis_values[i] for i in report.nis_outliers]
    ax2.scatter(report.nis_outliers, nis_outlier_values, color=visualization_config['colors']['anomaly'], label='Outliers')
    
    ax2.set_title('Normalized Innovation Squared (NIS)')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('NIS Value')
    ax2.legend()
    ax2.grid(True)
    
    # 绘制条件数曲线
    ax3.plot(report.condition_numbers, color=visualization_config['colors']['nees'], label='Condition Number')
    ax3.axhline(y=thresholds['condition_number']['upper'], color=visualization_config['colors']['threshold'], linestyle='--', label='Upper Threshold')
    
    # 标记异常点
    condition_outlier_values = [report.condition_numbers[i] for i in report.condition_number_outliers]
    ax3.scatter(report.condition_number_outliers, condition_outlier_values, color=visualization_config['colors']['anomaly'], label='Outliers')
    
    ax3.set_title('Condition Number of Covariance Matrix')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Condition Number')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    
    return fig


def generate_report(
    report: ConsistencyReport,
    config_path: str = "configs/consistency.yaml"
) -> None:
    """生成 Markdown 和 HTML 报告"""
    # 加载配置
    config = load_config(config_path)
    report_config = config['report']
    
    # 创建报告目录
    os.makedirs(os.path.dirname(report_config['save_path']), exist_ok=True)
    
    # 生成 Markdown 报告
    markdown_content = f"""# {report_config['title']}

## 总体结论
- 状态: {report.overall_status}
- 消息: {report.message}

## 覆盖率统计
- NEES 覆盖率: {report.nees_coverage:.2%}
- NIS 覆盖率: {report.nis_coverage:.2%}

## 异常点统计
- NEES 异常点数量: {len(report.nees_outliers)}
- NIS 异常点数量: {len(report.nis_outliers)}
- 条件数异常点数量: {len(report.condition_number_outliers)}

## 异常点详情
- NEES 异常点索引: {report.nees_outliers}
- NIS 异常点索引: {report.nis_outliers}
- 条件数异常点索引: {report.condition_number_outliers}

## 可视化结果
![一致性评估曲线]({os.path.basename(report_config['save_path'])}.png)
"""
    
    # 保存 Markdown 报告
    with open(f"{report_config['save_path']}.md", 'w') as f:
        f.write(markdown_content)
    
    # 生成可视化
    fig = generate_visualization(report, config_path)
    fig.savefig(f"{report_config['save_path']}.png", dpi=config['visualization']['dpi'], bbox_inches='tight')
    plt.close(fig)
    
    # 生成 HTML 报告
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report_config['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2E86AB; }}
        h2 {{ color: #A23B72; }}
        .summary {{ background-color: #F0F0F0; padding: 10px; border-radius: 5px; }}
        .metric {{ margin: 20px 0; }}
        .outliers {{ background-color: #FFE6E6; padding: 10px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>{report_config['title']}</h1>
    
    <div class="summary">
        <h2>总体结论</h2>
        <p><strong>状态:</strong> {report.overall_status}</p>
        <p><strong>消息:</strong> {report.message}</p>
    </div>
    
    <div class="metric">
        <h2>覆盖率统计</h2>
        <p>NEES 覆盖率: {report.nees_coverage:.2%}</p>
        <p>NIS 覆盖率: {report.nis_coverage:.2%}</p>
    </div>
    
    <div class="metric">
        <h2>异常点统计</h2>
        <p>NEES 异常点数量: {len(report.nees_outliers)}</p>
        <p>NIS 异常点数量: {len(report.nis_outliers)}</p>
        <p>条件数异常点数量: {len(report.condition_number_outliers)}</p>
    </div>
    
    <div class="outliers">
        <h2>异常点详情</h2>
        <p>NEES 异常点索引: {report.nees_outliers}</p>
        <p>NIS 异常点索引: {report.nis_outliers}</p>
        <p>条件数异常点索引: {report.condition_number_outliers}</p>
    </div>
    
    <div class="visualization">
        <h2>可视化结果</h2>
        <img src="{os.path.basename(report_config['save_path'])}.png" alt="一致性评估曲线">
    </div>
</body>
</html>"""
    
    # 保存 HTML 报告
    with open(f"{report_config['save_path']}.html", 'w') as f:
        f.write(html_content)