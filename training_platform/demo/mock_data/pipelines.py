"""
Demo流水线数据 - 完整的训练流水线

展示端到端的训练流程:
数据准备 → SFT → GRPO → DPO → 评估 → 模型融合 → 部署
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .jobs import DEMO_JOB_UUIDS, BASE_TIME


# ============ 演示流水线 ============

DEMO_PIPELINES: List[Dict] = [
    # 主流水线：数学推理增强
    {
        "id": "pipeline-math-enhancement-001",
        "uuid": "pipeline-math-enhancement-001",
        "name": "Qwen2.5-7B 数学推理增强流水线",
        "description": "完整的数学能力增强训练流程：SFT预训练 → GRPO策略优化 → DPO对齐 → 模型融合",
        "status": "running",
        "current_stage": 4,  # GRPO训练中
        "total_stages": 8,
        "progress": 55,  # 总进度百分比
        "stages": [
            {
                "id": "stage-1",
                "name": "数据准备",
                "stage_type": "data_preparation",
                "status": "completed",
                "order": 1,
                "config": {
                    "dataset": "math_instruction_50k",
                    "preprocessing": ["tokenization", "quality_filter", "deduplication"],
                },
                "result": {
                    "samples_processed": 50000,
                    "samples_filtered": 2150,
                    "quality_score": 0.95,
                },
                "started_at": (BASE_TIME).isoformat(),
                "completed_at": (BASE_TIME + timedelta(hours=2)).isoformat(),
                "duration_seconds": 7200,
            },
            {
                "id": "stage-2",
                "name": "计算配置",
                "stage_type": "compute_setup",
                "status": "completed",
                "order": 2,
                "config": {
                    "gpu_type": "A100-80G",
                    "gpu_count": 8,
                    "auto_optimize": True,
                },
                "result": {
                    "recommended_batch_size": 4,
                    "gradient_accumulation": 8,
                    "zero_stage": 2,
                    "estimated_memory": "72GB per GPU",
                },
                "started_at": (BASE_TIME + timedelta(hours=2)).isoformat(),
                "completed_at": (BASE_TIME + timedelta(hours=2, minutes=5)).isoformat(),
                "duration_seconds": 300,
            },
            {
                "id": "stage-3",
                "name": "SFT预训练",
                "stage_type": "training",
                "algorithm": "sft",
                "status": "completed",
                "order": 3,
                "config": {
                    "model": "Qwen2.5-7B-Instruct",
                    "dataset": "math_instruction_50k",
                    "epochs": 2,
                    "learning_rate": 1e-5,
                },
                "result": {
                    "final_loss": 0.42,
                    "final_perplexity": 1.62,
                    "best_checkpoint": "checkpoint-6250",
                    "gsm8k_score": 0.752,
                },
                "job_id": DEMO_JOB_UUIDS["sft"],
                "started_at": (BASE_TIME + timedelta(hours=3)).isoformat(),
                "completed_at": (BASE_TIME + timedelta(hours=11, minutes=30)).isoformat(),
                "duration_seconds": 30600,
            },
            {
                "id": "stage-4",
                "name": "GRPO训练",
                "stage_type": "training",
                "algorithm": "grpo",
                "status": "running",
                "order": 4,
                "config": {
                    "model": "Qwen2.5-7B-Math-SFT",
                    "dataset": "math_preference_20k",
                    "reward_model": "math-reward-model-7b",
                    "epochs": 3,
                    "kl_coef": 0.04,
                },
                "progress": 64,  # 3200/5000
                "current_metrics": {
                    "step": 3200,
                    "reward_mean": 0.823,
                    "kl_divergence": 0.018,
                    "policy_loss": 0.089,
                },
                "job_id": DEMO_JOB_UUIDS["grpo"],
                "started_at": (BASE_TIME + timedelta(hours=66)).isoformat(),
                "completed_at": None,
                "estimated_completion": (BASE_TIME + timedelta(hours=75)).isoformat(),
            },
            {
                "id": "stage-5",
                "name": "中间评估",
                "stage_type": "evaluation",
                "status": "pending",
                "order": 5,
                "config": {
                    "benchmarks": ["GSM8K", "MATH"],
                    "checkpoints": "last_3",
                },
                "depends_on": ["stage-4"],
            },
            {
                "id": "stage-6",
                "name": "DPO对齐",
                "stage_type": "training",
                "algorithm": "dpo",
                "status": "pending",
                "order": 6,
                "config": {
                    "model": "Qwen2.5-7B-Math-GRPO",
                    "dataset": "math_dpo_pairs_10k",
                    "beta": 0.1,
                },
                "depends_on": ["stage-5"],
                "job_id": DEMO_JOB_UUIDS["dpo"],
            },
            {
                "id": "stage-7",
                "name": "模型融合",
                "stage_type": "model_surgery",
                "status": "pending",
                "order": 7,
                "config": {
                    "method": "slerp",
                    "source_models": ["GRPO-best", "DPO-final"],
                    "interpolation_factor": 0.6,
                },
                "depends_on": ["stage-6"],
            },
            {
                "id": "stage-8",
                "name": "最终评估",
                "stage_type": "evaluation",
                "status": "pending",
                "order": 8,
                "config": {
                    "benchmarks": ["GSM8K", "MATH", "MMLU", "HumanEval"],
                    "compare_to_baseline": True,
                },
                "depends_on": ["stage-7"],
            },
        ],
        "created_at": (BASE_TIME - timedelta(hours=1)).isoformat(),
        "started_at": (BASE_TIME).isoformat(),
        "estimated_completion": (BASE_TIME + timedelta(hours=80)).isoformat(),
        "created_by": "demo_user",
    },

    # 已完成的流水线：代码增强
    {
        "id": "pipeline-code-enhancement-001",
        "uuid": "pipeline-code-enhancement-001",
        "name": "Qwen2.5-7B 代码能力增强流水线",
        "description": "代码生成和推理能力增强流程",
        "status": "completed",
        "current_stage": 6,
        "total_stages": 6,
        "progress": 100,
        "stages": [
            {
                "id": "stage-1",
                "name": "数据准备",
                "stage_type": "data_preparation",
                "status": "completed",
                "order": 1,
                "result": {"samples_processed": 100000},
                "duration_seconds": 10800,
            },
            {
                "id": "stage-2",
                "name": "代码SFT训练",
                "stage_type": "training",
                "algorithm": "sft",
                "status": "completed",
                "order": 2,
                "job_id": DEMO_JOB_UUIDS["sft_code"],
                "result": {
                    "final_loss": 0.38,
                    "humaneval_score": 0.689,
                },
                "duration_seconds": 57600,
            },
            {
                "id": "stage-3",
                "name": "推理GRPO训练",
                "stage_type": "training",
                "algorithm": "grpo",
                "status": "completed",
                "order": 3,
                "job_id": DEMO_JOB_UUIDS["grpo_reasoning"],
                "result": {
                    "final_reward": 0.91,
                    "humaneval_score": 0.720,
                },
                "duration_seconds": 43200,
            },
            {
                "id": "stage-4",
                "name": "检查点选择",
                "stage_type": "model_surgery",
                "status": "completed",
                "order": 4,
                "result": {
                    "selected_checkpoint": "checkpoint-7500",
                    "criteria": "highest_benchmark",
                },
                "duration_seconds": 600,
            },
            {
                "id": "stage-5",
                "name": "SWA优化",
                "stage_type": "model_surgery",
                "status": "completed",
                "order": 5,
                "result": {
                    "improvement": "+1.2%",
                },
                "duration_seconds": 900,
            },
            {
                "id": "stage-6",
                "name": "最终评估",
                "stage_type": "evaluation",
                "status": "completed",
                "order": 6,
                "result": {
                    "humaneval": 0.732,
                    "mbpp": 0.768,
                    "gsm8k": 0.862,
                },
                "duration_seconds": 3600,
            },
        ],
        "created_at": (BASE_TIME - timedelta(days=4)).isoformat(),
        "started_at": (BASE_TIME - timedelta(days=4)).isoformat(),
        "completed_at": (BASE_TIME - timedelta(days=2)).isoformat(),
        "created_by": "demo_user",
        "total_duration_hours": 32.5,
    },
]


# ============ 流水线模板 ============

DEMO_PIPELINE_TEMPLATES: List[Dict] = [
    {
        "id": "template-math-full",
        "name": "数学能力增强（完整流程）",
        "description": "SFT → GRPO → DPO → 融合的完整数学训练流程",
        "stages": ["data_prep", "sft", "grpo", "dpo", "merge", "eval"],
        "estimated_hours": 48,
        "recommended_gpu": "A100-80G x 8",
    },
    {
        "id": "template-code-quick",
        "name": "代码能力快速增强",
        "description": "SFT → GRPO的快速代码训练流程",
        "stages": ["data_prep", "sft", "grpo", "eval"],
        "estimated_hours": 24,
        "recommended_gpu": "A100-80G x 4",
    },
    {
        "id": "template-alignment-only",
        "name": "对齐微调",
        "description": "仅进行GRPO/DPO对齐的快速流程",
        "stages": ["grpo", "dpo", "eval"],
        "estimated_hours": 16,
        "recommended_gpu": "A100-80G x 4",
    },
]


def get_demo_pipeline(pipeline_id: str) -> Optional[Dict]:
    """获取单个流水线"""
    for pipeline in DEMO_PIPELINES:
        if pipeline["id"] == pipeline_id or pipeline["uuid"] == pipeline_id:
            return pipeline
    return None


def get_all_demo_pipelines() -> List[Dict]:
    """获取所有流水线"""
    return DEMO_PIPELINES


def get_pipeline_templates() -> List[Dict]:
    """获取流水线模板"""
    return DEMO_PIPELINE_TEMPLATES


def get_pipeline_stage(pipeline_id: str, stage_id: str) -> Optional[Dict]:
    """获取流水线的特定阶段"""
    pipeline = get_demo_pipeline(pipeline_id)
    if not pipeline:
        return None

    for stage in pipeline.get("stages", []):
        if stage["id"] == stage_id:
            return stage
    return None


def get_running_pipelines() -> List[Dict]:
    """获取正在运行的流水线"""
    return [p for p in DEMO_PIPELINES if p["status"] == "running"]


def get_completed_pipelines() -> List[Dict]:
    """获取已完成的流水线"""
    return [p for p in DEMO_PIPELINES if p["status"] == "completed"]
