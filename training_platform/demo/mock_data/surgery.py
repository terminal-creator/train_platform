"""
Demo模型手术数据 - 模型融合、检查点选择、SWA

展示模型优化的各种操作:
- 检查点选择（基于reward/loss/benchmark）
- SWA权重平均
- 模型融合（SLERP/TIES/DARE）
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import random

from .jobs import DEMO_JOB_UUIDS, BASE_TIME
from .checkpoints import get_demo_checkpoints, get_best_checkpoint


# ============ 模型融合结果 ============

DEMO_MERGE_RESULTS: List[Dict] = [
    # SLERP融合：Math-GRPO + Reasoning-GRPO
    {
        "id": "merge-slerp-001",
        "method": "slerp",
        "method_display": "SLERP (Spherical Linear Interpolation)",
        "source_models": [
            {
                "path": f"/outputs/{DEMO_JOB_UUIDS['grpo']}/checkpoint-2500",
                "name": "Qwen2.5-7B-Math-GRPO",
                "weight": 0.6,
            },
            {
                "path": f"/outputs/{DEMO_JOB_UUIDS['grpo_reasoning']}/checkpoint-7500",
                "name": "Qwen2.5-7B-Reasoning-GRPO",
                "weight": 0.4,
            },
        ],
        "output_path": "/models/merged/Qwen2.5-7B-MathReasoning-SLERP",
        "output_name": "Qwen2.5-7B-MathReasoning-SLERP",
        "config": {
            "interpolation_factor": 0.6,
            "normalize": True,
        },
        "status": "completed",
        "evaluation_scores": {
            "GSM8K": 0.868,  # 比单个模型更好
            "MATH": 0.478,
            "HumanEval": 0.695,
            "MMLU": 0.712,
        },
        "comparison_to_sources": {
            "GSM8K": {"model_1": 0.823, "model_2": 0.856, "merged": 0.868, "improvement": "+1.4%"},
            "MATH": {"model_1": 0.456, "model_2": 0.412, "merged": 0.478, "improvement": "+4.8%"},
        },
        "created_at": (BASE_TIME + timedelta(hours=70)).isoformat(),
        "duration_seconds": 245,
    },

    # TIES融合：多个checkpoint
    {
        "id": "merge-ties-001",
        "method": "ties",
        "method_display": "TIES (Trim, Elect Sign & Merge)",
        "source_models": [
            {
                "path": f"/outputs/{DEMO_JOB_UUIDS['grpo']}/checkpoint-2000",
                "name": "GRPO-checkpoint-2000",
                "weight": 0.25,
            },
            {
                "path": f"/outputs/{DEMO_JOB_UUIDS['grpo']}/checkpoint-2500",
                "name": "GRPO-checkpoint-2500",
                "weight": 0.35,
            },
            {
                "path": f"/outputs/{DEMO_JOB_UUIDS['grpo']}/checkpoint-3000",
                "name": "GRPO-checkpoint-3000",
                "weight": 0.40,
            },
        ],
        "output_path": "/models/merged/Qwen2.5-7B-Math-TIES",
        "output_name": "Qwen2.5-7B-Math-TIES",
        "config": {
            "density": 0.5,  # 保留50%的参数差异
            "normalize": True,
        },
        "status": "completed",
        "evaluation_scores": {
            "GSM8K": 0.835,
            "MATH": 0.465,
        },
        "created_at": (BASE_TIME + timedelta(hours=71)).isoformat(),
        "duration_seconds": 320,
    },

    # DARE融合
    {
        "id": "merge-dare-001",
        "method": "dare",
        "method_display": "DARE (Drop And REscale)",
        "source_models": [
            {
                "path": f"/outputs/{DEMO_JOB_UUIDS['sft']}/checkpoint-6250",
                "name": "Qwen2.5-7B-Math-SFT",
                "weight": 0.3,
            },
            {
                "path": f"/outputs/{DEMO_JOB_UUIDS['grpo']}/checkpoint-2500",
                "name": "Qwen2.5-7B-Math-GRPO",
                "weight": 0.7,
            },
        ],
        "output_path": "/models/merged/Qwen2.5-7B-Math-DARE",
        "output_name": "Qwen2.5-7B-Math-DARE",
        "config": {
            "drop_rate": 0.9,  # 丢弃90%的delta
            "rescale": True,
        },
        "status": "completed",
        "evaluation_scores": {
            "GSM8K": 0.842,
            "MATH": 0.472,
        },
        "created_at": (BASE_TIME + timedelta(hours=71, minutes=30)).isoformat(),
        "duration_seconds": 285,
    },
]


# ============ SWA结果 ============

DEMO_SWA_RESULTS: List[Dict] = [
    {
        "id": "swa-grpo-001",
        "job_id": DEMO_JOB_UUIDS["grpo"],
        "source_checkpoints": [
            f"/outputs/{DEMO_JOB_UUIDS['grpo']}/checkpoint-2000",
            f"/outputs/{DEMO_JOB_UUIDS['grpo']}/checkpoint-2500",
            f"/outputs/{DEMO_JOB_UUIDS['grpo']}/checkpoint-3000",
        ],
        "output_path": f"/outputs/{DEMO_JOB_UUIDS['grpo']}/swa-model",
        "output_name": "Qwen2.5-7B-Math-GRPO-SWA",
        "config": {
            "swa_start_step": 2000,
            "swa_freq": 500,
            "swa_lr": 1e-7,
        },
        "status": "completed",
        "evaluation_scores": {
            "GSM8K": 0.838,  # SWA通常能提升1-2%
            "MATH": 0.468,
        },
        "improvement_vs_last_checkpoint": {
            "GSM8K": "+1.5%",
            "MATH": "+1.2%",
        },
        "created_at": (BASE_TIME + timedelta(hours=69, minutes=30)).isoformat(),
        "duration_seconds": 180,
    },
]


# ============ 检查点选择结果 ============

DEMO_CHECKPOINT_SELECTIONS: List[Dict] = [
    {
        "id": "select-reward-001",
        "job_id": DEMO_JOB_UUIDS["grpo"],
        "criteria": "highest_reward",
        "criteria_display": "最高奖励分数",
        "candidates": [
            {"step": 2000, "reward_mean": 0.756, "eval_loss": 0.52},
            {"step": 2500, "reward_mean": 0.798, "eval_loss": 0.48},
            {"step": 3000, "reward_mean": 0.823, "eval_loss": 0.45},
        ],
        "selected": {
            "step": 3000,
            "path": f"/outputs/{DEMO_JOB_UUIDS['grpo']}/checkpoint-3000",
            "reward_mean": 0.823,
            "reason": "该检查点具有最高的奖励均值(0.823)，表明模型在该点生成的响应质量最佳",
        },
        "created_at": (BASE_TIME + timedelta(hours=68)).isoformat(),
    },
    {
        "id": "select-benchmark-001",
        "job_id": DEMO_JOB_UUIDS["grpo"],
        "criteria": "highest_benchmark",
        "criteria_display": "最高Benchmark分数",
        "candidates": [
            {"step": 2000, "gsm8k": 0.785, "math": 0.412},
            {"step": 2500, "gsm8k": 0.808, "math": 0.438},
            {"step": 3000, "gsm8k": 0.823, "math": 0.456},
        ],
        "selected": {
            "step": 3000,
            "path": f"/outputs/{DEMO_JOB_UUIDS['grpo']}/checkpoint-3000",
            "benchmark_avg": 0.6395,
            "reason": "该检查点在GSM8K(82.3%)和MATH(45.6%)上均达到最佳表现",
        },
        "created_at": (BASE_TIME + timedelta(hours=68, minutes=15)).isoformat(),
    },
]


# ============ RM提示词配置 ============

DEMO_RM_PROMPT_CONFIGS: List[Dict] = [
    {
        "id": "rm-prompt-math-001",
        "name": "数学推理奖励模型提示词",
        "description": "用于评估数学问题解答质量的奖励模型",
        "prompt_template": """你是一个数学答案质量评估专家。请根据以下标准评估给定的数学问题回答：

## 评估标准

1. **正确性 (40%)**
   - 最终答案是否正确
   - 计算过程是否准确

2. **推理质量 (30%)**
   - 推理步骤是否清晰
   - 是否有完整的解题思路
   - 是否使用了合适的数学方法

3. **表达清晰度 (20%)**
   - 解答是否易于理解
   - 数学符号使用是否规范
   - 步骤之间是否有逻辑衔接

4. **完整性 (10%)**
   - 是否回答了所有子问题
   - 是否给出了最终结论

## 问题
{question}

## 回答
{answer}

## 评估
请给出0-1之间的分数，并简要说明原因。
""",
        "variables": ["question", "answer"],
        "scoring_range": [0, 1],
        "created_at": (BASE_TIME + timedelta(days=-2)).isoformat(),
        "updated_at": (BASE_TIME + timedelta(hours=65)).isoformat(),
        "is_active": True,
    },
    {
        "id": "rm-prompt-code-001",
        "name": "代码质量奖励模型提示词",
        "description": "用于评估代码生成质量的奖励模型",
        "prompt_template": """你是一个代码质量评估专家。请根据以下标准评估给定的代码实现：

## 评估标准

1. **功能正确性 (50%)**
   - 代码是否实现了要求的功能
   - 边界情况处理是否正确

2. **代码质量 (25%)**
   - 代码是否简洁高效
   - 命名是否规范
   - 是否有合适的错误处理

3. **可读性 (15%)**
   - 代码结构是否清晰
   - 是否有必要的注释

4. **最佳实践 (10%)**
   - 是否遵循语言惯用法
   - 是否考虑了性能

## 任务描述
{task}

## 代码实现
{code}

## 评估
请给出0-1之间的分数，并简要说明原因。
""",
        "variables": ["task", "code"],
        "scoring_range": [0, 1],
        "created_at": (BASE_TIME + timedelta(days=-3)).isoformat(),
        "updated_at": (BASE_TIME + timedelta(days=-1)).isoformat(),
        "is_active": True,
    },
]


def get_merge_result(merge_id: str) -> Optional[Dict]:
    """获取融合结果"""
    for result in DEMO_MERGE_RESULTS:
        if result["id"] == merge_id:
            return result
    return None


def get_all_merge_results() -> List[Dict]:
    """获取所有融合结果"""
    return DEMO_MERGE_RESULTS


def get_swa_result(swa_id: str) -> Optional[Dict]:
    """获取SWA结果"""
    for result in DEMO_SWA_RESULTS:
        if result["id"] == swa_id:
            return result
    return None


def get_all_swa_results() -> List[Dict]:
    """获取所有SWA结果"""
    return DEMO_SWA_RESULTS


def get_checkpoint_selection(selection_id: str) -> Optional[Dict]:
    """获取检查点选择结果"""
    for result in DEMO_CHECKPOINT_SELECTIONS:
        if result["id"] == selection_id:
            return result
    return None


def get_all_checkpoint_selections() -> List[Dict]:
    """获取所有检查点选择结果"""
    return DEMO_CHECKPOINT_SELECTIONS


def get_rm_prompt_config(config_id: str) -> Optional[Dict]:
    """获取RM提示词配置"""
    for config in DEMO_RM_PROMPT_CONFIGS:
        if config["id"] == config_id:
            return config
    return None


def get_all_rm_prompt_configs() -> List[Dict]:
    """获取所有RM提示词配置"""
    return DEMO_RM_PROMPT_CONFIGS


def get_active_rm_prompt_config() -> Optional[Dict]:
    """获取当前激活的RM提示词配置"""
    for config in DEMO_RM_PROMPT_CONFIGS:
        if config.get("is_active"):
            return config
    return DEMO_RM_PROMPT_CONFIGS[0] if DEMO_RM_PROMPT_CONFIGS else None


def simulate_merge_operation(
    method: str,
    source_models: List[Dict],
    config: Dict,
) -> Dict:
    """模拟融合操作结果"""
    merge_id = f"merge-{method}-{random.randint(100, 999)}"

    # 根据方法生成预期的评估分数
    base_scores = {"GSM8K": 0.82, "MATH": 0.45}

    if method == "slerp":
        improvement = 0.02
    elif method == "ties":
        improvement = 0.015
    elif method == "dare":
        improvement = 0.018
    else:
        improvement = 0.01

    return {
        "id": merge_id,
        "method": method,
        "source_models": source_models,
        "config": config,
        "status": "completed",
        "output_path": f"/models/merged/merged-{merge_id}",
        "evaluation_scores": {
            k: round(v + improvement + random.uniform(-0.01, 0.01), 3)
            for k, v in base_scores.items()
        },
        "created_at": datetime.now().isoformat(),
        "duration_seconds": random.randint(200, 400),
    }
