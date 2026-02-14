"""
Demo训练任务数据 - 展示完整的LLM训练流程

故事线: 训练一个数学推理增强的Qwen2.5-7B模型
1. SFT: 使用高质量数学指令数据进行监督微调
2. GRPO: 使用奖励模型进行策略优化
3. DPO: 直接偏好优化进一步对齐
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import uuid

# 演示任务的固定UUID（保证一致性）
DEMO_JOB_UUIDS = {
    "sft": "demo-sft-qwen7b-math-001",
    "grpo": "demo-grpo-qwen7b-math-002",
    "dpo": "demo-dpo-qwen7b-math-003",
    "sft_code": "demo-sft-qwen7b-code-004",
    "grpo_reasoning": "demo-grpo-qwen7b-reasoning-005",
}

# 基础时间（演示从3天前开始）
BASE_TIME = datetime.now() - timedelta(days=3)


def _make_job(
    uuid: str,
    name: str,
    algorithm: str,
    status: str,
    model: str,
    dataset: str,
    current_step: int,
    total_steps: int,
    current_epoch: int,
    total_epochs: int,
    hours_ago: float,
    duration_hours: float = 0,
    config_overrides: Dict = None,
    latest_metrics: Dict = None,
) -> Dict:
    """创建标准化的Job数据"""
    started_at = BASE_TIME + timedelta(hours=72 - hours_ago)
    completed_at = None
    if status == "completed":
        completed_at = started_at + timedelta(hours=duration_hours)

    base_config = {
        "model_path": f"/models/{model}",
        "train_data_path": f"/datasets/{dataset}",
        "learning_rate": 1e-5 if algorithm == "sft" else 5e-7,
        "batch_size": 4,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 4096,
        "num_epochs": total_epochs,
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "fp16": True,
        "deepspeed_stage": 2,
    }

    if algorithm in ["grpo", "ppo"]:
        base_config.update({
            "kl_coef": 0.05,
            "clip_range": 0.2,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "rollout_batch_size": 512,
            "ppo_epochs": 4,
        })
    elif algorithm == "dpo":
        base_config.update({
            "beta": 0.1,
            "label_smoothing": 0.0,
            "loss_type": "sigmoid",
        })

    if config_overrides:
        base_config.update(config_overrides)

    default_metrics = {
        "step": current_step,
        "epoch": current_epoch,
        "loss": 0.35,
        "learning_rate": base_config["learning_rate"],
    }

    if algorithm == "sft":
        default_metrics.update({
            "train_loss": 0.82,
            "eval_loss": 0.91,
            "perplexity": 2.24,
        })
    elif algorithm in ["grpo", "ppo"]:
        default_metrics.update({
            "policy_loss": 0.12,
            "value_loss": 0.08,
            "reward_mean": 0.76,
            "reward_std": 0.15,
            "kl_divergence": 0.023,
            "entropy": 1.85,
            "advantages_mean": 0.12,
            "clip_fraction": 0.18,
        })
    elif algorithm == "dpo":
        default_metrics.update({
            "loss": 0.45,
            "rewards_chosen": 2.3,
            "rewards_rejected": -1.2,
            "reward_margin": 3.5,
            "accuracy": 0.78,
        })

    if latest_metrics:
        default_metrics.update(latest_metrics)

    return {
        "id": uuid,
        "uuid": uuid,
        "name": name,
        "description": f"Demo {algorithm.upper()} training for {model}",
        "status": status,
        "algorithm": algorithm,
        "model_path": base_config["model_path"],
        "train_data_path": base_config["train_data_path"],
        "eval_data_path": f"/datasets/{dataset}_eval",
        "config": base_config,
        "current_step": current_step,
        "total_steps": total_steps,
        "current_epoch": current_epoch,
        "total_epochs": total_epochs,
        "progress": round(current_step / total_steps * 100, 1) if total_steps > 0 else 0,
        "created_at": (started_at - timedelta(minutes=30)).isoformat(),
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat() if completed_at else None,
        "output_path": f"/outputs/{uuid}",
        "checkpoint_paths": [
            f"/outputs/{uuid}/checkpoint-{i}"
            for i in range(500, current_step + 1, 500)
        ][:10],
        "latest_metrics": default_metrics,
        "ray_job_id": f"raysubmit_{uuid[:8]}",
        "gpu_count": 8,
        "gpu_type": "A100-80G",
        "estimated_time_remaining": "2h 15m" if status == "running" else None,
        "throughput": 1250.5 if status == "running" else None,
    }


# ============ 主要演示任务 ============

DEMO_JOBS: List[Dict] = [
    # 阶段3: SFT预训练 (已完成)
    _make_job(
        uuid=DEMO_JOB_UUIDS["sft"],
        name="Qwen2.5-7B-Math-SFT",
        algorithm="sft",
        status="completed",
        model="Qwen2.5-7B-Instruct",
        dataset="math_instruction_50k",
        current_step=6250,
        total_steps=6250,
        current_epoch=2,
        total_epochs=2,
        hours_ago=48,
        duration_hours=8.5,
        latest_metrics={
            "train_loss": 0.42,
            "eval_loss": 0.48,
            "perplexity": 1.62,
            "accuracy": 0.89,
        }
    ),

    # 阶段5: GRPO训练 (正在运行 - 主要展示)
    _make_job(
        uuid=DEMO_JOB_UUIDS["grpo"],
        name="Qwen2.5-7B-Math-GRPO",
        algorithm="grpo",
        status="running",
        model="Qwen2.5-7B-Math-SFT",  # 基于SFT模型
        dataset="math_preference_20k",
        current_step=3200,
        total_steps=5000,
        current_epoch=2,
        total_epochs=3,
        hours_ago=6,
        config_overrides={
            "reward_model_path": "/models/math-reward-model-7b",
            "kl_coef": 0.04,
            "rollout_batch_size": 256,
        },
        latest_metrics={
            "policy_loss": 0.089,
            "value_loss": 0.052,
            "reward_mean": 0.823,
            "reward_std": 0.12,
            "kl_divergence": 0.018,
            "entropy": 1.92,
            "advantages_mean": 0.15,
            "clip_fraction": 0.14,
            "response_length_mean": 385.2,
            "throughput_tokens_per_sec": 12500,
        }
    ),

    # 阶段6: DPO对齐 (待运行)
    _make_job(
        uuid=DEMO_JOB_UUIDS["dpo"],
        name="Qwen2.5-7B-Math-DPO",
        algorithm="dpo",
        status="pending",
        model="Qwen2.5-7B-Math-GRPO",  # 基于GRPO模型
        dataset="math_dpo_pairs_10k",
        current_step=0,
        total_steps=2500,
        current_epoch=0,
        total_epochs=1,
        hours_ago=0,
        config_overrides={
            "beta": 0.1,
            "reference_model_path": "/models/Qwen2.5-7B-Math-SFT",
        },
        latest_metrics={}
    ),

    # 额外演示: 代码能力SFT (已完成)
    _make_job(
        uuid=DEMO_JOB_UUIDS["sft_code"],
        name="Qwen2.5-7B-Code-SFT",
        algorithm="sft",
        status="completed",
        model="Qwen2.5-7B-Instruct",
        dataset="code_instruction_100k",
        current_step=12500,
        total_steps=12500,
        current_epoch=2,
        total_epochs=2,
        hours_ago=72,
        duration_hours=16,
        latest_metrics={
            "train_loss": 0.38,
            "eval_loss": 0.45,
            "perplexity": 1.57,
        }
    ),

    # 额外演示: 推理增强GRPO (已完成)
    _make_job(
        uuid=DEMO_JOB_UUIDS["grpo_reasoning"],
        name="Qwen2.5-7B-Reasoning-GRPO",
        algorithm="grpo",
        status="completed",
        model="Qwen2.5-7B-Code-SFT",
        dataset="reasoning_preference_30k",
        current_step=7500,
        total_steps=7500,
        current_epoch=3,
        total_epochs=3,
        hours_ago=24,
        duration_hours=12,
        latest_metrics={
            "policy_loss": 0.065,
            "value_loss": 0.038,
            "reward_mean": 0.91,
            "reward_std": 0.08,
            "kl_divergence": 0.012,
            "entropy": 1.78,
        }
    ),
]


def get_demo_job(job_id: str) -> Optional[Dict]:
    """获取单个Demo任务"""
    for job in DEMO_JOBS:
        if job["id"] == job_id or job["uuid"] == job_id:
            return job
    return None


def get_all_demo_jobs() -> List[Dict]:
    """获取所有Demo任务"""
    return DEMO_JOBS


def get_demo_jobs_by_status(status: str) -> List[Dict]:
    """按状态筛选Demo任务"""
    return [job for job in DEMO_JOBS if job["status"] == status]


def get_demo_jobs_by_algorithm(algorithm: str) -> List[Dict]:
    """按算法筛选Demo任务"""
    return [job for job in DEMO_JOBS if job["algorithm"] == algorithm]
