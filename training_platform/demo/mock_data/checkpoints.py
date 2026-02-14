"""
Demo检查点数据 - 模型训练检查点

展示训练过程中保存的检查点，用于:
- 检查点选择
- SWA (Stochastic Weight Averaging)
- 模型恢复
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .jobs import DEMO_JOB_UUIDS, BASE_TIME


def _generate_checkpoints(
    job_id: str,
    job_name: str,
    total_steps: int,
    checkpoint_interval: int,
    completed_steps: int,
    base_reward: float = 0.5,
    reward_growth: float = 0.3,
) -> List[Dict]:
    """生成检查点列表"""
    checkpoints = []

    for step in range(checkpoint_interval, completed_steps + 1, checkpoint_interval):
        progress = step / total_steps

        # 计算该检查点的指标
        reward = base_reward + reward_growth * progress + (0.05 if step % 1000 == 0 else 0)
        eval_loss = 0.8 - 0.3 * progress

        # 模拟一些检查点有更好的性能（用于检查点选择演示）
        is_best = step in [int(total_steps * 0.6), int(total_steps * 0.8)]
        if is_best:
            reward += 0.05
            eval_loss -= 0.05

        checkpoints.append({
            "id": f"{job_id}-ckpt-{step}",
            "job_id": job_id,
            "job_name": job_name,
            "step": step,
            "epoch": int(step / (total_steps / 3)) + 1,
            "path": f"/outputs/{job_id}/checkpoint-{step}",
            "size_gb": round(14.2 + (step % 3) * 0.1, 2),
            "metrics": {
                "reward_mean": round(min(reward, 0.95), 4),
                "eval_loss": round(max(eval_loss, 0.35), 4),
                "kl_divergence": round(0.01 + 0.02 * progress, 5),
                "perplexity": round(2.5 - 1.0 * progress, 3),
            },
            "benchmark_scores": {
                "gsm8k": round(0.55 + 0.35 * progress, 3),
                "math": round(0.25 + 0.25 * progress, 3),
                "humaneval": round(0.45 + 0.25 * progress, 3),
            } if step >= total_steps * 0.3 else None,
            "is_best": is_best,
            "created_at": (BASE_TIME + timedelta(hours=48 - 8 * (1 - progress))).isoformat(),
        })

    return checkpoints


# ============ 预定义检查点 ============

DEMO_CHECKPOINTS: Dict[str, List[Dict]] = {
    DEMO_JOB_UUIDS["sft"]: _generate_checkpoints(
        job_id=DEMO_JOB_UUIDS["sft"],
        job_name="Qwen2.5-7B-Math-SFT",
        total_steps=6250,
        checkpoint_interval=500,
        completed_steps=6250,
        base_reward=0.0,  # SFT没有reward
        reward_growth=0.0,
    ),

    DEMO_JOB_UUIDS["grpo"]: _generate_checkpoints(
        job_id=DEMO_JOB_UUIDS["grpo"],
        job_name="Qwen2.5-7B-Math-GRPO",
        total_steps=5000,
        checkpoint_interval=500,
        completed_steps=3000,  # 还在训练中
        base_reward=0.25,
        reward_growth=0.55,
    ),

    DEMO_JOB_UUIDS["grpo_reasoning"]: _generate_checkpoints(
        job_id=DEMO_JOB_UUIDS["grpo_reasoning"],
        job_name="Qwen2.5-7B-Reasoning-GRPO",
        total_steps=7500,
        checkpoint_interval=500,
        completed_steps=7500,
        base_reward=0.30,
        reward_growth=0.60,
    ),

    DEMO_JOB_UUIDS["sft_code"]: _generate_checkpoints(
        job_id=DEMO_JOB_UUIDS["sft_code"],
        job_name="Qwen2.5-7B-Code-SFT",
        total_steps=12500,
        checkpoint_interval=1000,
        completed_steps=12500,
        base_reward=0.0,
        reward_growth=0.0,
    ),
}


def get_demo_checkpoints(job_id: str) -> List[Dict]:
    """获取任务的检查点列表"""
    return DEMO_CHECKPOINTS.get(job_id, [])


def get_best_checkpoint(job_id: str, criteria: str = "highest_reward") -> Optional[Dict]:
    """获取最佳检查点"""
    checkpoints = get_demo_checkpoints(job_id)
    if not checkpoints:
        return None

    if criteria == "highest_reward":
        return max(checkpoints, key=lambda x: x["metrics"].get("reward_mean", 0))
    elif criteria == "lowest_loss":
        return min(checkpoints, key=lambda x: x["metrics"].get("eval_loss", float("inf")))
    elif criteria == "highest_benchmark":
        scored = [c for c in checkpoints if c.get("benchmark_scores")]
        if not scored:
            return checkpoints[-1]
        return max(scored, key=lambda x: sum(x["benchmark_scores"].values()) / len(x["benchmark_scores"]))
    elif criteria == "latest":
        return checkpoints[-1]

    return checkpoints[-1]


def get_checkpoints_for_swa(job_id: str, num_checkpoints: int = 5) -> List[Dict]:
    """获取用于SWA的检查点（通常是最后N个）"""
    checkpoints = get_demo_checkpoints(job_id)
    return checkpoints[-num_checkpoints:] if len(checkpoints) >= num_checkpoints else checkpoints
