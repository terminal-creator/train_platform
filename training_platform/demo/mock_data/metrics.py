"""
Demo训练指标数据 - 匹配前端MonitoringView期望的格式
"""
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .jobs import DEMO_JOB_UUIDS, BASE_TIME


def _generate_grpo_metrics(total_steps: int, current_step: int) -> List[Dict]:
    """生成GRPO训练指标 - 完整格式"""
    metrics = []

    for step in range(0, min(current_step + 1, total_steps + 1), 50):
        progress = step / total_steps

        # Loss: 逐渐下降
        base_loss = 2.5 * math.exp(-2 * progress) + 0.3
        policy_loss = base_loss * 0.7 + random.gauss(0, 0.02)
        value_loss = base_loss * 0.3 + random.gauss(0, 0.01)
        total_loss = policy_loss + value_loss

        # Reward: S型增长
        sigmoid = 1 / (1 + math.exp(-8 * (progress - 0.4)))
        reward_mean = -0.3 + 2.5 * sigmoid + random.gauss(0, 0.05)
        reward_std = 0.3 + random.gauss(0, 0.05)

        # KL散度
        kl_mean = 0.001 + 0.025 * progress + random.gauss(0, 0.002)
        kl_max = kl_mean * 1.5

        # 梯度范数
        grad_actor = 0.5 + random.gauss(0, 0.1) + (2.0 if random.random() > 0.95 else 0)
        grad_critic = 0.8 + random.gauss(0, 0.15) + (1.5 if random.random() > 0.95 else 0)

        # 吞吐量和显存
        warmup = min(1, step / 500)
        tokens_per_sec = (15000 + random.gauss(0, 1500)) * warmup
        gpu_memory = 72 + random.gauss(0, 2)

        timestamp = BASE_TIME + timedelta(hours=66 + 6 * progress)

        metrics.append({
            "step": step,
            "epoch": int(progress * 3) + 1,
            "timestamp": timestamp.isoformat(),
            # 嵌套格式 (前端图表用)
            "loss": {
                "total_loss": round(total_loss, 4),
                "policy_loss": round(policy_loss, 4),
                "value_loss": round(value_loss, 4)
            },
            "reward": {
                "mean": round(reward_mean, 4),
                "std": round(reward_std, 4),
                "min": round(reward_mean - 1, 4),
                "max": round(reward_mean + 1.5, 4)
            },
            "kl": {
                "mean": round(kl_mean, 5),
                "max": round(kl_max, 5)
            },
            "gradient": {
                "actor_norm": round(grad_actor, 4),
                "critic_norm": round(grad_critic, 4)
            },
            "performance": {
                "tokens_per_second": round(tokens_per_sec, 1),
                "gpu_memory_allocated": round(gpu_memory, 1)
            },
            # 扁平格式 (兼容)
            "total_loss": round(total_loss, 4),
            "policy_loss": round(policy_loss, 4),
            "value_loss": round(value_loss, 4),
            "reward_mean": round(reward_mean, 4),
            "reward_std": round(reward_std, 4),
            "kl_divergence": round(kl_mean, 5),
            "grad_norm_actor": round(grad_actor, 4),
            "grad_norm_critic": round(grad_critic, 4),
            "tokens_per_second": round(tokens_per_sec, 1),
            "gpu_memory_allocated_gib": round(gpu_memory, 1),
            "learning_rate": round(5e-7 * (1 - 0.4 * progress), 9),
            "entropy": round(2.1 - 0.25 * progress + random.gauss(0, 0.03), 4),
            "clip_fraction": round(0.2 * (1 - 0.25 * progress) + random.gauss(0, 0.015), 4),
        })

    return metrics


def _generate_sft_metrics(total_steps: int, current_step: int) -> List[Dict]:
    """生成SFT训练指标"""
    metrics = []

    for step in range(0, min(current_step + 1, total_steps + 1), 50):
        progress = step / total_steps

        # Loss曲线
        decay = math.exp(-3 * progress)
        train_loss = 0.42 + 2.38 * decay + random.gauss(0, 0.03)
        eval_loss = train_loss * 1.12 + random.gauss(0, 0.02)

        # 梯度
        grad_norm = 0.8 + 0.5 * decay + random.gauss(0, 0.1)

        # 吞吐量
        tokens_per_sec = 11000 + random.gauss(0, 500)
        gpu_memory = 72 + random.gauss(0, 1)

        # Learning rate
        warmup = 0.03
        if progress < warmup:
            lr = 1e-5 * (progress / warmup)
        else:
            lr = 1e-5 * 0.5 * (1 + math.cos(math.pi * (progress - warmup) / (1 - warmup)))

        timestamp = BASE_TIME + timedelta(hours=48 - 8.5 * (1 - progress))

        metrics.append({
            "step": step,
            "epoch": int(progress * 2) + 1,
            "timestamp": timestamp.isoformat(),
            "loss": {
                "total_loss": round(train_loss, 4),
                "train_loss": round(train_loss, 4),
                "eval_loss": round(eval_loss, 4)
            },
            "gradient": {
                "norm": round(grad_norm, 4)
            },
            "performance": {
                "tokens_per_second": round(tokens_per_sec, 1),
                "gpu_memory_allocated": round(gpu_memory, 1)
            },
            "total_loss": round(train_loss, 4),
            "train_loss": round(train_loss, 4),
            "eval_loss": round(eval_loss, 4),
            "perplexity": round(math.exp(train_loss), 4),
            "grad_norm": round(grad_norm, 4),
            "tokens_per_second": round(tokens_per_sec, 1),
            "gpu_memory_allocated_gib": round(gpu_memory, 1),
            "learning_rate": round(lr, 9),
        })

    return metrics


def _generate_dpo_metrics(total_steps: int, current_step: int) -> List[Dict]:
    """生成DPO训练指标"""
    metrics = []

    for step in range(0, min(current_step + 1, total_steps + 1), 25):
        progress = step / total_steps

        loss = 0.69 - 0.35 * progress + random.gauss(0, 0.02)
        rewards_chosen = 1.5 + 1.5 * progress + random.gauss(0, 0.1)
        rewards_rejected = -0.5 - 1.0 * progress + random.gauss(0, 0.1)
        accuracy = 0.55 + 0.35 * progress + random.gauss(0, 0.02)

        tokens_per_sec = 14000 + random.gauss(0, 600)
        gpu_memory = 68 + random.gauss(0, 0.5)

        timestamp = BASE_TIME + timedelta(hours=72 + 4 * progress)

        metrics.append({
            "step": step,
            "epoch": 1,
            "timestamp": timestamp.isoformat(),
            "loss": {"total_loss": round(loss, 4)},
            "reward": {
                "chosen": round(rewards_chosen, 4),
                "rejected": round(rewards_rejected, 4),
                "margin": round(rewards_chosen - rewards_rejected, 4)
            },
            "performance": {
                "tokens_per_second": round(tokens_per_sec, 1),
                "gpu_memory_allocated": round(gpu_memory, 1)
            },
            "total_loss": round(loss, 4),
            "rewards_chosen": round(rewards_chosen, 4),
            "rewards_rejected": round(rewards_rejected, 4),
            "reward_margin": round(rewards_chosen - rewards_rejected, 4),
            "accuracy": round(min(accuracy, 0.95), 4),
            "tokens_per_second": round(tokens_per_sec, 1),
            "gpu_memory_allocated_gib": round(gpu_memory, 1),
            "learning_rate": round(5e-7 * (1 - 0.3 * progress), 9),
        })

    return metrics


# 缓存
_METRICS_CACHE: Dict[str, List[Dict]] = {}


def get_demo_metrics(job_id: str, start_step: int = 0, limit: int = 1000) -> List[Dict]:
    """获取Demo任务的训练指标"""
    global _METRICS_CACHE

    if job_id not in _METRICS_CACHE:
        if job_id == DEMO_JOB_UUIDS["sft"] or "sft" in job_id.lower():
            _METRICS_CACHE[job_id] = _generate_sft_metrics(6250, 6250)
        elif job_id == DEMO_JOB_UUIDS["grpo"] or "grpo" in job_id.lower():
            _METRICS_CACHE[job_id] = _generate_grpo_metrics(5000, 3200)
        elif job_id == DEMO_JOB_UUIDS["dpo"] or "dpo" in job_id.lower():
            _METRICS_CACHE[job_id] = _generate_dpo_metrics(2500, 0)
        elif job_id == DEMO_JOB_UUIDS["sft_code"]:
            _METRICS_CACHE[job_id] = _generate_sft_metrics(12500, 12500)
        elif job_id == DEMO_JOB_UUIDS["grpo_reasoning"]:
            _METRICS_CACHE[job_id] = _generate_grpo_metrics(7500, 7500)
        else:
            # 默认返回GRPO指标
            _METRICS_CACHE[job_id] = _generate_grpo_metrics(5000, 3200)

    metrics = _METRICS_CACHE[job_id]
    filtered = [m for m in metrics if m["step"] >= start_step]
    return filtered[:limit]


def get_realtime_metrics(job_id: str, current_step: int) -> Optional[Dict]:
    """获取实时指标"""
    metrics = get_demo_metrics(job_id)
    if not metrics:
        return None

    for m in reversed(metrics):
        if m["step"] <= current_step:
            return m

    return metrics[0] if metrics else None


def get_metrics_summary(job_id: str) -> Dict:
    """获取指标汇总"""
    metrics = get_demo_metrics(job_id)
    if not metrics:
        return {}

    if "grpo" in job_id.lower() or job_id == DEMO_JOB_UUIDS["grpo"]:
        rewards = [m.get("reward_mean", 0) for m in metrics]
        kls = [m.get("kl_divergence", 0) for m in metrics]
        return {
            "initial_reward": rewards[0] if rewards else 0,
            "final_reward": rewards[-1] if rewards else 0,
            "max_reward": max(rewards) if rewards else 0,
            "avg_kl": round(sum(kls) / len(kls), 5) if kls else 0,
            "max_kl": max(kls) if kls else 0,
            "total_steps": len(metrics) * 50,
        }
    else:
        losses = [m.get("total_loss") or m.get("train_loss", 0) for m in metrics]
        return {
            "initial_loss": losses[0] if losses else 0,
            "final_loss": losses[-1] if losses else 0,
            "min_loss": min(losses) if losses else 0,
            "total_steps": len(metrics) * 50,
        }
