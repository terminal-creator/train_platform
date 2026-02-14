"""
Demo数据集 - 高质量训练数据展示

展示数据准备阶段的各类数据集:
- SFT指令数据集
- 偏好对比数据集
- DPO训练数据集
- 评估数据集
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional

BASE_TIME = datetime.now() - timedelta(days=7)


# ============ 训练数据集 ============

DEMO_DATASETS: List[Dict] = [
    # 数学指令数据集 (SFT用)
    {
        "id": "demo-dataset-math-sft-001",
        "uuid": "demo-dataset-math-sft-001",
        "name": "math_instruction_50k",
        "display_name": "数学指令数据集 50K",
        "description": "高质量数学推理指令数据，包含代数、几何、概率统计等多领域",
        "format": "openai_messages",
        "path": "/datasets/math_instruction_50k",
        "size_bytes": 125_000_000,  # 125MB
        "num_samples": 50000,
        "version": "v1.2",
        "version_hash": "a3f8c2d1e5b4",
        "quality_score": 0.95,
        "quality_details": {
            "avg_instruction_length": 85.3,
            "avg_response_length": 420.6,
            "diversity_score": 0.89,
            "complexity_distribution": {
                "easy": 0.15,
                "medium": 0.45,
                "hard": 0.30,
                "expert": 0.10
            }
        },
        "statistics": {
            "total_tokens": 28_500_000,
            "avg_tokens_per_sample": 570,
            "max_tokens": 4096,
            "language_distribution": {"zh": 0.6, "en": 0.4},
            "topic_distribution": {
                "algebra": 0.25,
                "geometry": 0.20,
                "probability": 0.15,
                "calculus": 0.15,
                "number_theory": 0.10,
                "logic": 0.15
            }
        },
        "created_at": (BASE_TIME).isoformat(),
        "updated_at": (BASE_TIME + timedelta(days=1)).isoformat(),
    },

    # 数学偏好数据集 (GRPO/PPO用)
    {
        "id": "demo-dataset-math-pref-002",
        "uuid": "demo-dataset-math-pref-002",
        "name": "math_preference_20k",
        "display_name": "数学偏好对比数据集 20K",
        "description": "数学问题的多个回答对比，标注了正确性和推理质量偏好",
        "format": "preference_pairs",
        "path": "/datasets/math_preference_20k",
        "size_bytes": 85_000_000,
        "num_samples": 20000,
        "version": "v1.0",
        "version_hash": "b7d9e3f2a1c6",
        "quality_score": 0.93,
        "quality_details": {
            "agreement_rate": 0.91,
            "avg_responses_per_prompt": 3.2,
            "annotation_source": "expert_mathematician"
        },
        "statistics": {
            "total_pairs": 64000,
            "chosen_avg_length": 450,
            "rejected_avg_length": 280,
            "margin_distribution": {
                "clear_win": 0.65,
                "slight_win": 0.25,
                "tie": 0.10
            }
        },
        "created_at": (BASE_TIME + timedelta(days=2)).isoformat(),
        "updated_at": (BASE_TIME + timedelta(days=2)).isoformat(),
    },

    # DPO数据集
    {
        "id": "demo-dataset-math-dpo-003",
        "uuid": "demo-dataset-math-dpo-003",
        "name": "math_dpo_pairs_10k",
        "display_name": "数学DPO训练数据 10K",
        "description": "精选的数学问答DPO对比对，chosen和rejected差异明显",
        "format": "dpo_pairs",
        "path": "/datasets/math_dpo_pairs_10k",
        "size_bytes": 42_000_000,
        "num_samples": 10000,
        "version": "v1.0",
        "version_hash": "c5a2b8d4e7f9",
        "quality_score": 0.96,
        "quality_details": {
            "chosen_correctness": 0.98,
            "rejected_error_types": {
                "calculation_error": 0.35,
                "logic_error": 0.30,
                "incomplete": 0.20,
                "wrong_method": 0.15
            }
        },
        "statistics": {
            "avg_prompt_length": 120,
            "avg_chosen_length": 480,
            "avg_rejected_length": 350,
        },
        "created_at": (BASE_TIME + timedelta(days=3)).isoformat(),
        "updated_at": (BASE_TIME + timedelta(days=3)).isoformat(),
    },

    # 代码指令数据集
    {
        "id": "demo-dataset-code-sft-004",
        "uuid": "demo-dataset-code-sft-004",
        "name": "code_instruction_100k",
        "display_name": "代码指令数据集 100K",
        "description": "多语言编程指令数据，覆盖Python、JavaScript、Go等主流语言",
        "format": "openai_messages",
        "path": "/datasets/code_instruction_100k",
        "size_bytes": 320_000_000,
        "num_samples": 100000,
        "version": "v2.1",
        "version_hash": "d8e6f4c3a2b1",
        "quality_score": 0.94,
        "quality_details": {
            "syntax_valid_rate": 0.99,
            "test_pass_rate": 0.92,
        },
        "statistics": {
            "language_distribution": {
                "python": 0.40,
                "javascript": 0.20,
                "java": 0.15,
                "go": 0.10,
                "rust": 0.08,
                "other": 0.07
            },
            "task_distribution": {
                "implementation": 0.35,
                "debugging": 0.20,
                "optimization": 0.15,
                "explanation": 0.15,
                "testing": 0.15
            }
        },
        "created_at": (BASE_TIME - timedelta(days=3)).isoformat(),
        "updated_at": (BASE_TIME).isoformat(),
    },

    # 推理偏好数据集
    {
        "id": "demo-dataset-reasoning-pref-005",
        "uuid": "demo-dataset-reasoning-pref-005",
        "name": "reasoning_preference_30k",
        "display_name": "推理偏好数据集 30K",
        "description": "Chain-of-Thought推理质量偏好数据，强调推理过程的清晰度",
        "format": "preference_pairs",
        "path": "/datasets/reasoning_preference_30k",
        "size_bytes": 150_000_000,
        "num_samples": 30000,
        "version": "v1.1",
        "version_hash": "e9f7a5b3c1d2",
        "quality_score": 0.92,
        "quality_details": {
            "cot_quality_score": 0.94,
            "step_by_step_rate": 0.88,
        },
        "statistics": {
            "avg_reasoning_steps": 5.2,
            "domains": ["math", "logic", "commonsense", "science"]
        },
        "created_at": (BASE_TIME - timedelta(days=1)).isoformat(),
        "updated_at": (BASE_TIME + timedelta(days=1)).isoformat(),
    },
]


# ============ 评估数据集 ============

DEMO_EVAL_DATASETS: List[Dict] = [
    {
        "id": "demo-eval-gsm8k-001",
        "uuid": "demo-eval-gsm8k-001",
        "name": "GSM8K",
        "display_name": "GSM8K 数学推理测试",
        "description": "Grade School Math 8K - 小学数学应用题测试集",
        "format": "benchmark",
        "path": "/eval_datasets/gsm8k",
        "num_samples": 1319,
        "capability": "math_reasoning",
        "metadata": {
            "source": "OpenAI",
            "difficulty": "medium",
            "requires_cot": True
        }
    },
    {
        "id": "demo-eval-math-002",
        "uuid": "demo-eval-math-002",
        "name": "MATH",
        "display_name": "MATH 竞赛数学测试",
        "description": "高中及竞赛级别数学问题",
        "format": "benchmark",
        "path": "/eval_datasets/math",
        "num_samples": 5000,
        "capability": "math_competition",
        "metadata": {
            "source": "Hendrycks et al.",
            "difficulty": "hard",
            "levels": ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
        }
    },
    {
        "id": "demo-eval-humaneval-003",
        "uuid": "demo-eval-humaneval-003",
        "name": "HumanEval",
        "display_name": "HumanEval 代码生成测试",
        "description": "Python函数生成能力测试",
        "format": "benchmark",
        "path": "/eval_datasets/humaneval",
        "num_samples": 164,
        "capability": "code_generation",
        "metadata": {
            "source": "OpenAI",
            "language": "python",
            "metric": "pass@1"
        }
    },
    {
        "id": "demo-eval-mbpp-004",
        "uuid": "demo-eval-mbpp-004",
        "name": "MBPP",
        "display_name": "MBPP Python编程测试",
        "description": "Mostly Basic Python Problems",
        "format": "benchmark",
        "path": "/eval_datasets/mbpp",
        "num_samples": 500,
        "capability": "code_generation",
        "metadata": {
            "source": "Google",
            "language": "python"
        }
    },
    {
        "id": "demo-eval-mmlu-005",
        "uuid": "demo-eval-mmlu-005",
        "name": "MMLU",
        "display_name": "MMLU 多领域知识测试",
        "description": "Massive Multitask Language Understanding",
        "format": "benchmark",
        "path": "/eval_datasets/mmlu",
        "num_samples": 14042,
        "capability": "knowledge",
        "metadata": {
            "source": "Hendrycks et al.",
            "subjects": 57,
            "format": "multiple_choice"
        }
    },
]


# ============ 数据集样本预览 ============

DEMO_DATASET_SAMPLES: Dict[str, List[Dict]] = {
    "demo-dataset-math-sft-001": [
        {
            "messages": [
                {"role": "system", "content": "你是一个专业的数学老师，请用清晰的步骤解答问题。"},
                {"role": "user", "content": "一个圆的面积是78.5平方厘米，求这个圆的周长。（π取3.14）"},
                {"role": "assistant", "content": "让我来一步步解答这个问题。\n\n**已知条件：**\n- 圆的面积 S = 78.5 平方厘米\n- π = 3.14\n\n**解题步骤：**\n\n1. **根据面积公式求半径**\n   圆的面积公式：S = πr²\n   78.5 = 3.14 × r²\n   r² = 78.5 ÷ 3.14 = 25\n   r = 5 厘米\n\n2. **根据半径求周长**\n   圆的周长公式：C = 2πr\n   C = 2 × 3.14 × 5\n   C = 31.4 厘米\n\n**答案：这个圆的周长是31.4厘米。**"}
            ],
            "quality_score": 0.96
        },
        {
            "messages": [
                {"role": "user", "content": "解方程：2x + 5 = 3x - 7"},
                {"role": "assistant", "content": "解这个一元一次方程：\n\n2x + 5 = 3x - 7\n\n**移项（把x项移到左边，常数移到右边）：**\n2x - 3x = -7 - 5\n\n**合并同类项：**\n-x = -12\n\n**系数化为1：**\nx = 12\n\n**验证：**\n- 左边 = 2(12) + 5 = 29\n- 右边 = 3(12) - 7 = 29\n- 左边 = 右边 ✓\n\n**答案：x = 12**"}
            ],
            "quality_score": 0.94
        },
        {
            "messages": [
                {"role": "user", "content": "甲乙两地相距450公里，一辆汽车从甲地开往乙地，前3小时行驶了180公里，照这样的速度，还需要多少小时才能到达乙地？"},
                {"role": "assistant", "content": "这是一道行程问题，让我们分步解答。\n\n**分析题目：**\n- 甲乙两地距离：450公里\n- 已行驶时间：3小时\n- 已行驶距离：180公里\n- 求：剩余时间\n\n**解题步骤：**\n\n1. **计算汽车速度**\n   速度 = 距离 ÷ 时间\n   速度 = 180 ÷ 3 = 60 公里/小时\n\n2. **计算剩余距离**\n   剩余距离 = 总距离 - 已行驶距离\n   剩余距离 = 450 - 180 = 270 公里\n\n3. **计算剩余时间**\n   时间 = 距离 ÷ 速度\n   时间 = 270 ÷ 60 = 4.5 小时\n\n**答案：还需要4.5小时（或4小时30分钟）才能到达乙地。**"}
            ],
            "quality_score": 0.97
        }
    ],
    "demo-dataset-math-pref-002": [
        {
            "prompt": "计算：123 × 45 + 123 × 55",
            "chosen": "观察这道题，我发现可以用乘法分配律来简化计算。\n\n123 × 45 + 123 × 55\n= 123 × (45 + 55)  【提取公因数123】\n= 123 × 100\n= 12300\n\n答案：12300\n\n这种方法比直接计算两个乘法再相加要简便得多！",
            "rejected": "123 × 45 = 5535\n123 × 55 = 6765\n5535 + 6765 = 12300\n\n答案是12300",
            "preference_reason": "chosen回答展示了数学思维（乘法分配律），而rejected只是机械计算"
        }
    ]
}


def get_demo_dataset(dataset_id: str) -> Optional[Dict]:
    """获取单个数据集信息"""
    for ds in DEMO_DATASETS:
        if ds["id"] == dataset_id or ds["uuid"] == dataset_id:
            return ds
    for ds in DEMO_EVAL_DATASETS:
        if ds["id"] == dataset_id or ds["uuid"] == dataset_id:
            return ds
    return None


def get_demo_dataset_samples(dataset_id: str, limit: int = 10) -> List[Dict]:
    """获取数据集样本预览"""
    samples = DEMO_DATASET_SAMPLES.get(dataset_id, [])
    return samples[:limit]


def get_all_demo_datasets() -> List[Dict]:
    """获取所有训练数据集"""
    return DEMO_DATASETS


def get_all_demo_eval_datasets() -> List[Dict]:
    """获取所有评估数据集"""
    return DEMO_EVAL_DATASETS
