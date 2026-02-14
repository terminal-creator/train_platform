"""
Demo梯度数据 - 梯度可视化和训练诊断

提供:
- 梯度热力图数据（各层梯度分布）
- 梯度统计（范数、方差等）
- 异常检测结果
"""
import math
import random
from typing import Dict, List, Optional
import numpy as np

from .jobs import DEMO_JOB_UUIDS


# Transformer层名称模板
LAYER_NAMES = [
    "embed_tokens",
    *[f"layers.{i}.self_attn.q_proj" for i in range(32)],
    *[f"layers.{i}.self_attn.k_proj" for i in range(32)],
    *[f"layers.{i}.self_attn.v_proj" for i in range(32)],
    *[f"layers.{i}.self_attn.o_proj" for i in range(32)],
    *[f"layers.{i}.mlp.gate_proj" for i in range(32)],
    *[f"layers.{i}.mlp.up_proj" for i in range(32)],
    *[f"layers.{i}.mlp.down_proj" for i in range(32)],
    *[f"layers.{i}.input_layernorm" for i in range(32)],
    *[f"layers.{i}.post_attention_layernorm" for i in range(32)],
    "norm",
    "lm_head",
]

# 简化的层名称（用于展示）
SIMPLIFIED_LAYERS = [
    "embed_tokens",
    *[f"layer_{i}_attn" for i in range(32)],
    *[f"layer_{i}_mlp" for i in range(32)],
    *[f"layer_{i}_norm" for i in range(32)],
    "final_norm",
    "lm_head",
]


def _generate_gradient_heatmap(
    num_layers: int = 32,
    num_steps: int = 50,
    training_progress: float = 0.5,
) -> Dict:
    """
    生成梯度热力图数据

    展示一个健康的训练过程，有明显的特征:
    - 早期训练梯度较大（偏红/黄）
    - 随着训练进行梯度逐渐稳定（偏绿）
    - 不同层有不同的梯度特征
    - embed层和head层梯度较大
    - 中间层梯度较小且稳定
    """
    # 层名称（简化版）
    layers = [
        "embed",
        *[f"L{i}_attn" for i in range(num_layers)],
        *[f"L{i}_ffn" for i in range(num_layers)],
        "head",
    ]

    # 步骤范围
    steps = list(range(0, num_steps * 100, 100))
    total_layers = len(layers)

    # 生成热力图数据 (layers x steps)
    # 值域: 梯度范数的对数 (log10)，范围 [-3, 0]
    # -3 = 0.001 (小梯度，绿色)
    # -2 = 0.01 (中等，黄色)
    # -1 = 0.1 (较大，橙色)
    # 0 = 1.0 (大梯度，红色)
    heatmap_data = []

    for layer_idx, layer_name in enumerate(layers):
        row = []
        # 层类型特征
        is_embed = layer_idx == 0
        is_head = layer_idx == total_layers - 1
        is_attn = 1 <= layer_idx <= num_layers
        layer_depth = layer_idx / total_layers  # 0 到 1

        for step_idx, step in enumerate(steps):
            progress = step_idx / num_steps  # 训练进度 0 到 1

            # 基础梯度值（log10尺度）
            if is_embed:
                # embed层: 梯度较大，从 -1 逐渐稳定到 -2
                base = -1.0 - progress * 1.0
            elif is_head:
                # head层: 梯度最大，从 -0.5 逐渐稳定到 -1.5
                base = -0.5 - progress * 1.0
            elif is_attn:
                # attention层: 中等梯度，深层略小
                depth_in_attn = (layer_idx - 1) / num_layers
                base = -1.5 - depth_in_attn * 0.8 - progress * 0.5
            else:
                # FFN层: 梯度较小，更稳定
                depth_in_ffn = (layer_idx - num_layers - 1) / num_layers
                base = -2.0 - depth_in_ffn * 0.5 - progress * 0.3

            # 早期波动（前20%的步数波动较大）
            if progress < 0.2:
                early_variation = 0.5 * (1 - progress / 0.2) * math.sin(step_idx * 0.5)
            else:
                early_variation = 0

            # 周期性波动（模拟batch间差异）
            periodic = 0.1 * math.sin(step_idx * 0.3 + layer_idx * 0.2)

            # 随机噪声
            noise = random.gauss(0, 0.1)

            # 最终值
            log_norm = base + early_variation + periodic + noise

            # 限制在 [-3, 0] 范围
            log_norm = max(-3.0, min(0.0, log_norm))

            row.append(round(log_norm, 4))

        heatmap_data.append(row)

    return {
        "layers": layers,
        "steps": steps,
        "values": heatmap_data,
        "value_range": {"min": -3.0, "max": 0.0},
        "colorscale": "RdYlGn",  # 红黄绿色阶
        "title": "Gradient Norm Heatmap (log10 scale)",
        "xlabel": "Training Step",
        "ylabel": "Layer",
    }


def _generate_gradient_stats(num_steps: int = 50) -> List[Dict]:
    """
    生成梯度统计时序数据

    包括:
    - 全局梯度范数
    - 最大/最小层梯度
    - 梯度方差
    """
    stats = []

    for step_idx in range(num_steps):
        step = step_idx * 100
        progress = step_idx / num_steps

        # 全局梯度范数（随训练逐渐稳定）
        global_norm = 1.2 * (1 - 0.4 * progress) + random.gauss(0, 0.05)

        # 层间梯度范数统计
        layer_norms = [
            0.1 + 0.05 * random.random() + 0.02 * math.sin(i * 0.1)
            for i in range(32)
        ]

        # 梯度方差
        grad_variance = 0.05 * (1 - 0.3 * progress) + random.gauss(0, 0.005)

        # 梯度裁剪触发率
        clip_rate = 0.15 * (1 - 0.5 * progress) + random.gauss(0, 0.02)

        stats.append({
            "step": step,
            "global_grad_norm": round(max(global_norm, 0.3), 4),
            "max_layer_grad_norm": round(max(layer_norms), 4),
            "min_layer_grad_norm": round(min(layer_norms), 4),
            "mean_layer_grad_norm": round(sum(layer_norms) / len(layer_norms), 4),
            "grad_variance": round(max(grad_variance, 0.01), 5),
            "clip_rate": round(min(max(clip_rate, 0), 0.5), 3),
            "has_nan": False,
            "has_inf": False,
        })

    return stats


def _generate_layer_analysis(layer_name: str, step: int) -> Dict:
    """生成单层的详细梯度分析"""
    # 生成梯度分布直方图数据
    num_bins = 50
    # 正态分布的梯度
    mean = random.gauss(0, 0.01)
    std = 0.05 + random.random() * 0.02

    bin_edges = np.linspace(-0.3, 0.3, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 模拟直方图计数（正态分布）
    counts = [
        int(1000 * math.exp(-((x - mean) ** 2) / (2 * std ** 2)))
        for x in bin_centers
    ]

    return {
        "layer_name": layer_name,
        "step": step,
        "statistics": {
            "mean": round(mean, 6),
            "std": round(std, 6),
            "min": round(mean - 3 * std, 6),
            "max": round(mean + 3 * std, 6),
            "l2_norm": round(std * math.sqrt(1000), 4),
            "sparsity": round(0.02 + random.random() * 0.03, 4),
        },
        "histogram": {
            "bin_edges": [round(x, 4) for x in bin_edges.tolist()],
            "counts": counts,
        },
        "health_status": "healthy",
        "warnings": [],
    }


# ============ 缓存 ============

_HEATMAP_CACHE: Dict[str, Dict] = {}
_STATS_CACHE: Dict[str, List[Dict]] = {}


def get_gradient_heatmap(job_id: str) -> Dict:
    """获取任务的梯度热力图"""
    if job_id not in _HEATMAP_CACHE:
        if job_id == DEMO_JOB_UUIDS["grpo"]:
            _HEATMAP_CACHE[job_id] = _generate_gradient_heatmap(
                num_layers=32,
                num_steps=64,
                training_progress=0.64,
            )
        elif job_id == DEMO_JOB_UUIDS["sft"]:
            _HEATMAP_CACHE[job_id] = _generate_gradient_heatmap(
                num_layers=32,
                num_steps=125,
                training_progress=1.0,
            )
        else:
            _HEATMAP_CACHE[job_id] = _generate_gradient_heatmap()

    return _HEATMAP_CACHE[job_id]


def get_gradient_stats(job_id: str) -> List[Dict]:
    """
    获取任务的梯度统计 - 按层统计格式

    返回各层的梯度统计信息，供前端表格展示
    """
    # 层名称
    layers = [
        'model.embed_tokens',
        'model.layers.0.self_attn.q_proj',
        'model.layers.0.self_attn.k_proj',
        'model.layers.0.self_attn.v_proj',
        'model.layers.0.self_attn.o_proj',
        'model.layers.0.mlp.gate_proj',
        'model.layers.0.mlp.up_proj',
        'model.layers.0.mlp.down_proj',
        'model.layers.15.self_attn.q_proj',
        'model.layers.15.mlp.gate_proj',
        'model.layers.31.self_attn.q_proj',
        'model.layers.31.mlp.down_proj',
        'model.norm',
        'lm_head'
    ]

    stats = []
    for layer in layers:
        # 生成合理的梯度统计
        mean = random.gauss(0, 0.001)
        std = 0.001 + random.random() * 0.01
        min_val = mean - 3 * std
        max_val = mean + 3 * std
        norm = std * math.sqrt(1000)  # L2 norm approximation

        stats.append({
            'layer_name': layer,
            'mean': round(mean, 8),
            'std': round(std, 6),
            'min': round(min_val, 6),
            'max': round(max_val, 6),
            'norm': round(norm, 4)
        })

    return stats


def get_gradient_stats_timeseries(job_id: str) -> List[Dict]:
    """获取任务的梯度统计时序数据"""
    if job_id not in _STATS_CACHE:
        if job_id == DEMO_JOB_UUIDS["grpo"]:
            _STATS_CACHE[job_id] = _generate_gradient_stats(64)
        elif job_id == DEMO_JOB_UUIDS["sft"]:
            _STATS_CACHE[job_id] = _generate_gradient_stats(125)
        else:
            _STATS_CACHE[job_id] = _generate_gradient_stats(50)

    return _STATS_CACHE[job_id]


def get_layer_gradient_analysis(job_id: str, layer_name: str, step: int) -> Dict:
    """获取单层的梯度详细分析"""
    return _generate_layer_analysis(layer_name, step)


def get_gradient_health_report(job_id: str) -> Dict:
    """获取梯度健康报告"""
    stats = get_gradient_stats(job_id)
    if not stats:
        return {"status": "unknown", "issues": []}

    latest = stats[-1]

    issues = []
    warnings = []

    # 检查梯度范数
    if latest["global_grad_norm"] > 5.0:
        issues.append("梯度范数过大，可能存在梯度爆炸风险")
    elif latest["global_grad_norm"] < 0.01:
        issues.append("梯度范数过小，可能存在梯度消失问题")

    # 检查裁剪率
    if latest["clip_rate"] > 0.3:
        warnings.append("梯度裁剪频率较高，建议降低学习率")

    # 检查NaN/Inf
    if latest["has_nan"] or latest["has_inf"]:
        issues.append("检测到NaN或Inf梯度，训练可能不稳定")

    status = "healthy"
    if issues:
        status = "critical"
    elif warnings:
        status = "warning"

    return {
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "summary": {
            "avg_grad_norm": round(sum(s["global_grad_norm"] for s in stats) / len(stats), 4),
            "max_grad_norm": max(s["global_grad_norm"] for s in stats),
            "avg_clip_rate": round(sum(s["clip_rate"] for s in stats) / len(stats), 4),
            "nan_count": sum(1 for s in stats if s["has_nan"]),
            "inf_count": sum(1 for s in stats if s["has_inf"]),
        },
        "recommendations": [
            "当前训练状态良好，梯度分布健康",
            "建议继续监控KL散度，保持在0.02以下",
        ] if status == "healthy" else [
            f"建议: {issue}" for issue in issues + warnings
        ],
    }
