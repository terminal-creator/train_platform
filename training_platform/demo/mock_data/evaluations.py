"""
Demo评估结果 - Benchmark测试结果

展示模型在各个阶段的评估表现，讲述一个完整的能力提升故事:
- 基座模型基线
- SFT后提升
- GRPO后继续提升
- 最终融合模型的最佳表现
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .jobs import DEMO_JOB_UUIDS, BASE_TIME


# ============ 评估结果 ============

DEMO_EVALUATIONS: Dict[str, List[Dict]] = {
    # SFT模型评估结果
    DEMO_JOB_UUIDS["sft"]: [
        {
            "id": "eval-sft-gsm8k-001",
            "job_id": DEMO_JOB_UUIDS["sft"],
            "benchmark": "GSM8K",
            "checkpoint_step": 6250,
            "score": 0.752,
            "baseline_score": 0.583,  # 基座模型
            "improvement": "+16.9%",
            "num_samples": 1319,
            "details": {
                "accuracy": 0.752,
                "correct_answers": 992,
                "total_questions": 1319,
                "avg_response_length": 285.3,
                "timeout_rate": 0.02,
            },
            "by_difficulty": {
                "easy": {"accuracy": 0.89, "count": 400},
                "medium": {"accuracy": 0.76, "count": 600},
                "hard": {"accuracy": 0.58, "count": 319},
            },
            "created_at": (BASE_TIME + timedelta(hours=50)).isoformat(),
        },
        {
            "id": "eval-sft-math-001",
            "job_id": DEMO_JOB_UUIDS["sft"],
            "benchmark": "MATH",
            "checkpoint_step": 6250,
            "score": 0.385,
            "baseline_score": 0.245,
            "improvement": "+14.0%",
            "num_samples": 5000,
            "details": {
                "accuracy": 0.385,
                "correct_answers": 1925,
                "total_questions": 5000,
            },
            "by_level": {
                "Level 1": {"accuracy": 0.72, "count": 1000},
                "Level 2": {"accuracy": 0.55, "count": 1000},
                "Level 3": {"accuracy": 0.38, "count": 1000},
                "Level 4": {"accuracy": 0.22, "count": 1000},
                "Level 5": {"accuracy": 0.065, "count": 1000},
            },
            "created_at": (BASE_TIME + timedelta(hours=51)).isoformat(),
        },
    ],

    # GRPO模型评估结果（当前运行中，中间checkpoint评估）
    DEMO_JOB_UUIDS["grpo"]: [
        {
            "id": "eval-grpo-gsm8k-001",
            "job_id": DEMO_JOB_UUIDS["grpo"],
            "benchmark": "GSM8K",
            "checkpoint_step": 2500,
            "score": 0.823,
            "baseline_score": 0.752,  # SFT模型
            "improvement": "+7.1%",
            "num_samples": 1319,
            "details": {
                "accuracy": 0.823,
                "correct_answers": 1086,
                "total_questions": 1319,
                "avg_response_length": 342.8,
                "cot_usage_rate": 0.95,  # Chain-of-Thought使用率
            },
            "by_difficulty": {
                "easy": {"accuracy": 0.94, "count": 400},
                "medium": {"accuracy": 0.84, "count": 600},
                "hard": {"accuracy": 0.67, "count": 319},
            },
            "created_at": (BASE_TIME + timedelta(hours=68)).isoformat(),
        },
        {
            "id": "eval-grpo-math-001",
            "job_id": DEMO_JOB_UUIDS["grpo"],
            "benchmark": "MATH",
            "checkpoint_step": 2500,
            "score": 0.456,
            "baseline_score": 0.385,
            "improvement": "+7.1%",
            "num_samples": 5000,
            "details": {
                "accuracy": 0.456,
                "correct_answers": 2280,
                "total_questions": 5000,
            },
            "by_level": {
                "Level 1": {"accuracy": 0.81, "count": 1000},
                "Level 2": {"accuracy": 0.65, "count": 1000},
                "Level 3": {"accuracy": 0.48, "count": 1000},
                "Level 4": {"accuracy": 0.28, "count": 1000},
                "Level 5": {"accuracy": 0.06, "count": 1000},
            },
            "created_at": (BASE_TIME + timedelta(hours=69)).isoformat(),
        },
    ],

    # 代码SFT模型评估
    DEMO_JOB_UUIDS["sft_code"]: [
        {
            "id": "eval-code-humaneval-001",
            "job_id": DEMO_JOB_UUIDS["sft_code"],
            "benchmark": "HumanEval",
            "checkpoint_step": 12500,
            "score": 0.689,
            "baseline_score": 0.524,
            "improvement": "+16.5%",
            "num_samples": 164,
            "details": {
                "pass_at_1": 0.689,
                "pass_at_10": 0.835,
                "pass_at_100": 0.921,
                "avg_attempts": 1.8,
            },
            "by_category": {
                "algorithms": {"pass_rate": 0.72, "count": 50},
                "data_structures": {"pass_rate": 0.68, "count": 40},
                "string_manipulation": {"pass_rate": 0.75, "count": 35},
                "math": {"pass_rate": 0.62, "count": 39},
            },
            "created_at": (BASE_TIME + timedelta(hours=20)).isoformat(),
        },
        {
            "id": "eval-code-mbpp-001",
            "job_id": DEMO_JOB_UUIDS["sft_code"],
            "benchmark": "MBPP",
            "checkpoint_step": 12500,
            "score": 0.724,
            "baseline_score": 0.586,
            "improvement": "+13.8%",
            "num_samples": 500,
            "details": {
                "pass_at_1": 0.724,
                "pass_at_10": 0.856,
            },
            "created_at": (BASE_TIME + timedelta(hours=21)).isoformat(),
        },
    ],

    # 推理GRPO模型评估
    DEMO_JOB_UUIDS["grpo_reasoning"]: [
        {
            "id": "eval-reasoning-gsm8k-001",
            "job_id": DEMO_JOB_UUIDS["grpo_reasoning"],
            "benchmark": "GSM8K",
            "checkpoint_step": 7500,
            "score": 0.856,
            "baseline_score": 0.689,  # Code SFT model
            "improvement": "+16.7%",
            "num_samples": 1319,
            "details": {
                "accuracy": 0.856,
                "correct_answers": 1129,
                "total_questions": 1319,
            },
            "created_at": (BASE_TIME + timedelta(hours=40)).isoformat(),
        },
        {
            "id": "eval-reasoning-humaneval-001",
            "job_id": DEMO_JOB_UUIDS["grpo_reasoning"],
            "benchmark": "HumanEval",
            "checkpoint_step": 7500,
            "score": 0.720,
            "baseline_score": 0.689,
            "improvement": "+3.1%",
            "num_samples": 164,
            "details": {
                "pass_at_1": 0.720,
                "pass_at_10": 0.862,
            },
            "created_at": (BASE_TIME + timedelta(hours=41)).isoformat(),
        },
    ],
}


# ============ 对比分析数据 ============

EVALUATION_COMPARISON = {
    "models": [
        {"name": "Qwen2.5-7B (Base)", "id": "baseline"},
        {"name": "Qwen2.5-7B-Math-SFT", "id": DEMO_JOB_UUIDS["sft"]},
        {"name": "Qwen2.5-7B-Math-GRPO", "id": DEMO_JOB_UUIDS["grpo"]},
        {"name": "Qwen2.5-7B-Code-SFT", "id": DEMO_JOB_UUIDS["sft_code"]},
        {"name": "Qwen2.5-7B-Reasoning-GRPO", "id": DEMO_JOB_UUIDS["grpo_reasoning"]},
    ],
    "benchmarks": {
        "GSM8K": {
            "baseline": 0.583,
            DEMO_JOB_UUIDS["sft"]: 0.752,
            DEMO_JOB_UUIDS["grpo"]: 0.823,
            DEMO_JOB_UUIDS["sft_code"]: 0.615,
            DEMO_JOB_UUIDS["grpo_reasoning"]: 0.856,
        },
        "MATH": {
            "baseline": 0.245,
            DEMO_JOB_UUIDS["sft"]: 0.385,
            DEMO_JOB_UUIDS["grpo"]: 0.456,
            DEMO_JOB_UUIDS["sft_code"]: 0.268,
            DEMO_JOB_UUIDS["grpo_reasoning"]: 0.412,
        },
        "HumanEval": {
            "baseline": 0.524,
            DEMO_JOB_UUIDS["sft"]: 0.542,
            DEMO_JOB_UUIDS["grpo"]: 0.558,
            DEMO_JOB_UUIDS["sft_code"]: 0.689,
            DEMO_JOB_UUIDS["grpo_reasoning"]: 0.720,
        },
        "MBPP": {
            "baseline": 0.586,
            DEMO_JOB_UUIDS["sft"]: 0.598,
            DEMO_JOB_UUIDS["grpo"]: 0.612,
            DEMO_JOB_UUIDS["sft_code"]: 0.724,
            DEMO_JOB_UUIDS["grpo_reasoning"]: 0.756,
        },
        "MMLU": {
            "baseline": 0.682,
            DEMO_JOB_UUIDS["sft"]: 0.695,
            DEMO_JOB_UUIDS["grpo"]: 0.708,
            DEMO_JOB_UUIDS["sft_code"]: 0.688,
            DEMO_JOB_UUIDS["grpo_reasoning"]: 0.715,
        },
    },
}


def get_demo_evaluation_results(job_id: str) -> List[Dict]:
    """获取任务的评估结果"""
    return DEMO_EVALUATIONS.get(job_id, [])


def get_evaluation_by_benchmark(job_id: str, benchmark: str) -> Optional[Dict]:
    """获取特定benchmark的评估结果"""
    results = DEMO_EVALUATIONS.get(job_id, [])
    for r in results:
        if r["benchmark"] == benchmark:
            return r
    return None


def get_evaluation_comparison() -> Dict:
    """获取评估对比数据"""
    return EVALUATION_COMPARISON


def get_evaluation_history(benchmark: str) -> List[Dict]:
    """获取某个benchmark的历史评估记录（展示提升趋势）"""
    history = []

    if benchmark == "GSM8K":
        history = [
            {"model": "Base", "score": 0.583, "stage": "baseline"},
            {"model": "SFT", "score": 0.752, "stage": "sft"},
            {"model": "GRPO-1k", "score": 0.785, "stage": "grpo_early"},
            {"model": "GRPO-2k", "score": 0.808, "stage": "grpo_mid"},
            {"model": "GRPO-2.5k", "score": 0.823, "stage": "grpo_current"},
        ]
    elif benchmark == "MATH":
        history = [
            {"model": "Base", "score": 0.245, "stage": "baseline"},
            {"model": "SFT", "score": 0.385, "stage": "sft"},
            {"model": "GRPO-1k", "score": 0.412, "stage": "grpo_early"},
            {"model": "GRPO-2k", "score": 0.438, "stage": "grpo_mid"},
            {"model": "GRPO-2.5k", "score": 0.456, "stage": "grpo_current"},
        ]

    return history


def get_all_demo_evaluations() -> Dict[str, List[Dict]]:
    """获取所有评估结果"""
    return DEMO_EVALUATIONS
