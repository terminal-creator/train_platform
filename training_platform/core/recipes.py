"""
Training Recipe System (Phase 2)

提供预定义的训练配置模板，降低训练门槛，提高实验可复现性。

配方（Recipe）包含：
- 算法类型（SFT/DPO/GRPO等）
- 推荐的超参数配置
- 数据格式要求
- 最佳实践建议

设计理念：
1. 配方是不可变的模板（immutable templates）
2. 用户可以基于配方创建训练任务
3. 配方可以被版本化和扩展
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field
import json


class TaskType(str, Enum):
    """训练任务类型"""
    SFT = "sft"  # Supervised Fine-Tuning
    RLHF = "rlhf"  # Reinforcement Learning from Human Feedback
    DPO = "dpo"  # Direct Preference Optimization
    GRPO = "grpo"  # Group Relative Policy Optimization
    PRETRAIN = "pretrain"  # Pre-training

    # Legacy task types (保持向后兼容)
    GENERAL_CHAT = "general_chat"
    MATH_REASONING = "math_reasoning"
    CODE_GENERATION = "code_generation"
    CREATIVE_WRITING = "creative_writing"


class ModelSize(str, Enum):
    """模型大小分类"""
    SMALL = "small"  # < 1B parameters
    MEDIUM = "medium"  # 1B - 10B parameters
    LARGE = "large"  # 10B - 100B parameters
    XLARGE = "xlarge"  # > 100B parameters


@dataclass
class TrainingRecipe:
    """
    训练配方

    配方是预定义的训练配置模板，包含算法、超参数、资源配置等。
    """
    # 基本信息
    name: str  # 配方唯一标识
    description: str  # 配方描述
    task_type: TaskType  # 任务类型
    recommended_algorithm: str  # 推荐算法

    # 配置
    default_config: Dict[str, Any]  # 默认配置参数

    # 要求和建议
    data_requirements: str  # 数据格式要求
    tips: List[str]  # 最佳实践建议

    # 资源建议
    model_size_hint: ModelSize = ModelSize.MEDIUM  # 推荐的模型大小
    min_gpus: int = 1  # 最小 GPU 数量
    recommended_gpus: int = 4  # 推荐 GPU 数量

    # 元信息
    tags: List[str] = field(default_factory=list)  # 标签（如 "beginner", "advanced"）
    author: str = "platform"  # 作者
    version: str = "1.0"  # 版本号

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type.value if isinstance(self.task_type, Enum) else self.task_type,
            "recommended_algorithm": self.recommended_algorithm,
            "default_config": self.default_config,
            "data_requirements": self.data_requirements,
            "model_size_hint": self.model_size_hint.value if isinstance(self.model_size_hint, Enum) else self.model_size_hint,
            "min_gpus": self.min_gpus,
            "recommended_gpus": self.recommended_gpus,
            "tips": self.tips,
            "tags": self.tags,
            "author": self.author,
            "version": self.version,
        }

    def get_config(self, model_size: str = None, num_gpus: int = None) -> Dict[str, Any]:
        """
        Get configuration adapted to model size and resources.

        Args:
            model_size: Model size hint (e.g., "7B", "70B")
            num_gpus: Number of available GPUs

        Returns:
            Adapted configuration dictionary
        """
        config = json.loads(json.dumps(self.default_config))  # Deep copy

        # Apply model-size specific heuristics
        if model_size:
            if "7B" in model_size or "1B" in model_size:
                # Smaller models can use higher learning rates
                if "learning_rate" in config:
                    config["learning_rate"] = min(config["learning_rate"] * 2, 5e-5)
            elif "70B" in model_size or "100B" in model_size:
                # Larger models need lower learning rates
                if "learning_rate" in config:
                    config["learning_rate"] = max(config["learning_rate"] * 0.5, 1e-7)

        # Apply GPU-specific heuristics
        if num_gpus:
            # Adjust batch size based on GPU count
            if "batch_size" in config and num_gpus != self.recommended_gpus:
                ratio = num_gpus / self.recommended_gpus
                config["batch_size"] = int(config["batch_size"] * ratio)

        return config


class RecipeRegistry:
    """
    配方注册表

    管理所有预定义的训练配方。
    采用单例模式，确保全局唯一。
    """
    _recipes: Dict[str, TrainingRecipe] = {}

    @classmethod
    def register(cls, recipe: TrainingRecipe) -> None:
        """注册一个配方"""
        cls._recipes[recipe.name] = recipe

    @classmethod
    def get(cls, name: str) -> Optional[TrainingRecipe]:
        """获取指定配方"""
        return cls._recipes.get(name)

    @classmethod
    def list_all(cls) -> List[Dict[str, Any]]:
        """列出所有配方（简化版）"""
        return [
            {
                "id": recipe.name,
                "name": recipe.name.replace("_", " ").title(),
                "description": recipe.description,
                "task_type": recipe.task_type.value if isinstance(recipe.task_type, Enum) else recipe.task_type,
                "algorithm": recipe.recommended_algorithm,
                "tags": recipe.tags,
            }
            for recipe in cls._recipes.values()
        ]

    @classmethod
    def list_by_task_type(cls, task_type: TaskType) -> List[TrainingRecipe]:
        """按任务类型筛选配方"""
        return [
            recipe
            for recipe in cls._recipes.values()
            if recipe.task_type == task_type
        ]

    @classmethod
    def list_by_tag(cls, tag: str) -> List[TrainingRecipe]:
        """按标签筛选配方"""
        return [
            recipe
            for recipe in cls._recipes.values()
            if tag in recipe.tags
        ]


# ============== 预定义配方 ==============

# SFT: 基础监督微调
sft_basic = TrainingRecipe(
    name="sft_basic",
    description="适用于小规模数据集的监督微调，适合初学者",
    task_type=TaskType.SFT,
    recommended_algorithm="sft",
    default_config={
        "learning_rate": 2e-5,
        "batch_size": 128,
        "num_epochs": 3,
        "max_steps": None,
        "context_length": 2048,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "lr_scheduler": "cosine",
        "lora_enabled": True,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    },
    data_requirements="JSON Lines 格式，每行包含 'prompt' 和 'response' 字段",
    model_size_hint=ModelSize.SMALL,
    min_gpus=1,
    recommended_gpus=4,
    tips=[
        "适用于 1B-7B 参数的模型",
        "数据量建议 1K-100K 条",
        "使用 LoRA 可以显著降低显存需求",
        "建议先在小数据集上验证配置",
    ],
    tags=["beginner", "sft", "lora"],
)

# SFT: 大规模微调
sft_large_scale = TrainingRecipe(
    name="sft_large_scale",
    description="适用于大规模数据集和大模型的监督微调",
    task_type=TaskType.SFT,
    recommended_algorithm="sft",
    default_config={
        "learning_rate": 1e-5,
        "batch_size": 256,
        "num_epochs": 1,
        "max_steps": 10000,
        "context_length": 4096,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "lr_scheduler": "cosine",
        "lora_enabled": False,  # 全参数微调
        "fp16": True,
        "gradient_checkpointing": True,
    },
    data_requirements="JSON Lines 格式，每行包含 'prompt' 和 'response' 字段",
    model_size_hint=ModelSize.LARGE,
    min_gpus=8,
    recommended_gpus=32,
    tips=[
        "适用于 10B+ 参数的模型",
        "数据量建议 100K+ 条",
        "全参数微调需要大量 GPU 资源",
        "建议使用 gradient checkpointing 降低显存",
        "使用 mixed precision (FP16) 加速训练",
    ],
    tags=["advanced", "sft", "full-parameter"],
)

# GRPO: 基础配方
grpo_basic = TrainingRecipe(
    name="grpo_basic",
    description="Group Relative Policy Optimization，适合小规模 RLHF",
    task_type=TaskType.RLHF,
    recommended_algorithm="grpo",
    default_config={
        "learning_rate": 5e-7,
        "batch_size": 128,
        "num_epochs": 1,
        "max_steps": 1000,
        "context_length": 2048,
        "kl_coef": 0.02,
        "rollout_n": 8,
        "rollout_batch_size": 512,
        "value_loss_coef": 0.1,
        "entropy_coef": 0.01,
        "gamma": 1.0,
        "lam": 0.95,
        "warmup_steps": 50,
        "lora_enabled": True,
        "lora_rank": 16,
    },
    data_requirements="JSON Lines 格式，每行包含 'prompt' 字段",
    model_size_hint=ModelSize.MEDIUM,
    min_gpus=4,
    recommended_gpus=8,
    tips=[
        "GRPO 适用于 1B-10B 参数的模型",
        "需要配置 Reward Model 或 Reward Function",
        "rollout_n 决定每个 prompt 生成的样本数",
        "KL 系数控制策略更新幅度，建议从 0.01-0.05 调整",
        "建议先用 SFT 预训练模型",
    ],
    tags=["intermediate", "rlhf", "grpo"],
)

# GRPO: 大规模配方
grpo_large_scale = TrainingRecipe(
    name="grpo_large_scale",
    description="适用于大模型和大数据集的 GRPO 训练",
    task_type=TaskType.RLHF,
    recommended_algorithm="grpo",
    default_config={
        "learning_rate": 1e-7,
        "batch_size": 256,
        "num_epochs": 1,
        "max_steps": 5000,
        "context_length": 4096,
        "kl_coef": 0.01,
        "rollout_n": 16,
        "rollout_batch_size": 1024,
        "value_loss_coef": 0.1,
        "entropy_coef": 0.005,
        "gamma": 1.0,
        "lam": 0.95,
        "warmup_steps": 200,
        "lora_enabled": False,
        "fp16": True,
        "gradient_checkpointing": True,
    },
    data_requirements="JSON Lines 格式，每行包含 'prompt' 字段",
    model_size_hint=ModelSize.LARGE,
    min_gpus=16,
    recommended_gpus=64,
    tips=[
        "适用于 10B+ 参数的模型",
        "数据量建议 10K+ prompts",
        "使用更大的 rollout_n 可以提高样本多样性",
        "KL 系数需要仔细调整，避免策略崩溃",
        "建议监控 KL 散度和 Reward 变化趋势",
    ],
    tags=["advanced", "rlhf", "grpo", "large-scale"],
)

# DPO: 基础配方
dpo_basic = TrainingRecipe(
    name="dpo_basic",
    description="Direct Preference Optimization，无需 Reward Model 的对齐方法",
    task_type=TaskType.DPO,
    recommended_algorithm="dpo",
    default_config={
        "learning_rate": 5e-7,
        "batch_size": 64,
        "num_epochs": 3,
        "max_steps": None,
        "context_length": 2048,
        "beta": 0.1,  # DPO temperature parameter
        "label_smoothing": 0.0,
        "warmup_steps": 100,
        "lora_enabled": True,
        "lora_rank": 16,
        "lora_alpha": 32,
    },
    data_requirements="JSON Lines 格式，每行包含 'prompt', 'chosen', 'rejected' 字段",
    model_size_hint=ModelSize.MEDIUM,
    min_gpus=2,
    recommended_gpus=8,
    tips=[
        "DPO 适用于 1B-10B 参数的模型",
        "需要偏好数据（chosen vs rejected pairs）",
        "beta 参数控制对齐强度，建议从 0.1 开始调整",
        "相比 PPO/GRPO，DPO 训练更稳定、更高效",
        "建议先用 SFT 预训练模型",
    ],
    tags=["intermediate", "alignment", "dpo"],
)

# PPO: 经典 RLHF
ppo_classic = TrainingRecipe(
    name="ppo_classic",
    description="经典的 PPO 算法，适用于 RLHF 训练",
    task_type=TaskType.RLHF,
    recommended_algorithm="ppo",
    default_config={
        "learning_rate": 1e-6,
        "batch_size": 128,
        "num_epochs": 1,
        "max_steps": 2000,
        "context_length": 2048,
        "kl_coef": 0.02,
        "rollout_n": 4,
        "rollout_batch_size": 256,
        "ppo_epochs": 4,
        "clip_range": 0.2,
        "value_loss_coef": 0.1,
        "entropy_coef": 0.01,
        "gamma": 1.0,
        "lam": 0.95,
        "warmup_steps": 50,
        "lora_enabled": True,
        "lora_rank": 16,
    },
    data_requirements="JSON Lines 格式，每行包含 'prompt' 字段",
    model_size_hint=ModelSize.MEDIUM,
    min_gpus=4,
    recommended_gpus=16,
    tips=[
        "PPO 是最经典的 RLHF 算法，稳定性较好",
        "需要配置 Reward Model",
        "clip_range 控制策略更新幅度",
        "ppo_epochs 决定每批数据训练轮数",
        "建议监控 KL 散度和 Reward 变化",
    ],
    tags=["classic", "rlhf", "ppo"],
)

# Legacy recipes (保持向后兼容)
math_recipe = TrainingRecipe(
    name="math_reasoning_grpo",
    description="Train a math reasoning model using Group Relative Policy Optimization (GRPO). Best for tasks with verifiable answers.",
    task_type=TaskType.MATH_REASONING,
    recommended_algorithm="grpo",
    data_requirements="Dataset with 'prompt', 'answer', and 'solution' columns.",
    tips=[
        "Use a large rollout_n (8-16) for better exploration.",
        "Ensure your reward function strictly checks the final answer.",
        "Start with a strong SFT base model (e.g., Qwen-Math) for faster convergence."
    ],
    default_config={
        "algorithm": "grpo",
        "learning_rate": 1e-6,
        "rollout_n": 8,
        "kl_coef": 0.04,
        "max_prompt_length": 512,
        "max_response_length": 1024,
        "reward_fn_type": "math_verify",
    },
    model_size_hint=ModelSize.MEDIUM,
    min_gpus=4,
    recommended_gpus=8,
    tags=["math", "grpo"],
)

chat_sft_recipe = TrainingRecipe(
    name="general_chat_sft",
    description="Standard Supervised Fine-Tuning for general chat capabilities.",
    task_type=TaskType.GENERAL_CHAT,
    recommended_algorithm="sft",
    data_requirements="Instruction-Response pairs in JSONL or Parquet.",
    tips=[
        "Quality > Quantity. 10k high-quality samples beat 1M noisy ones.",
        "Use packing (sequence packing) to speed up training by 2x.",
        "Monitor loss spikes; if it spikes, lower LR or check data quality."
    ],
    default_config={
        "algorithm": "sft",
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "batch_size": 128,
        "max_prompt_length": 1024,
        "max_response_length": 2048,
    },
    model_size_hint=ModelSize.MEDIUM,
    min_gpus=2,
    recommended_gpus=4,
    tags=["chat", "sft"],
)

code_ppo_recipe = TrainingRecipe(
    name="code_generation_ppo",
    description="Optimize code generation using PPO with unit test feedback.",
    task_type=TaskType.CODE_GENERATION,
    recommended_algorithm="ppo",
    data_requirements="Prompts with unit tests for reward calculation.",
    tips=[
        "PPO requires a Critic model (2x memory usage). Ensure you have enough GPUs.",
        "Keep KL penalty low (0.01) to prevent the model from forgetting syntax.",
        "Use a small learning rate for the Critic."
    ],
    default_config={
        "algorithm": "ppo",
        "learning_rate": 1e-6,
        "critic_lr": 1e-5,
        "rollout_n": 4,
        "kl_coef": 0.02,
        "reward_fn_type": "code_test",
    },
    model_size_hint=ModelSize.MEDIUM,
    min_gpus=4,
    recommended_gpus=8,
    tags=["code", "ppo"],
)


# 注册所有预定义配方
def register_builtin_recipes():
    """注册所有内置配方"""
    # Phase 2 recipes
    RecipeRegistry.register(sft_basic)
    RecipeRegistry.register(sft_large_scale)
    RecipeRegistry.register(grpo_basic)
    RecipeRegistry.register(grpo_large_scale)
    RecipeRegistry.register(dpo_basic)
    RecipeRegistry.register(ppo_classic)

    # Legacy recipes
    RecipeRegistry.register(math_recipe)
    RecipeRegistry.register(chat_sft_recipe)
    RecipeRegistry.register(code_ppo_recipe)


# 自动注册
register_builtin_recipes()


# ============== 配方应用工具 ==============

def apply_recipe_to_job_config(
    recipe: TrainingRecipe,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    将配方应用到任务配置

    Args:
        recipe: 配方对象
        overrides: 用户自定义覆盖参数

    Returns:
        合并后的配置字典
    """
    # 深拷贝配方的默认配置
    config = json.loads(json.dumps(recipe.default_config))

    # 应用用户覆盖
    if overrides:
        config.update(overrides)

    return config


def validate_recipe_config(recipe: TrainingRecipe, config: Dict[str, Any]) -> List[str]:
    """
    验证配置是否符合配方要求

    Args:
        recipe: 配方对象
        config: 用户配置

    Returns:
        警告信息列表（空列表表示无警告）
    """
    warnings = []

    # 检查必需参数
    required_params = ["learning_rate", "batch_size"]
    for param in required_params:
        if param not in config:
            warnings.append(f"缺少必需参数: {param}")

    # 检查 GPU 数量
    if "num_gpus" in config:
        num_gpus = config["num_gpus"]
        if num_gpus < recipe.min_gpus:
            warnings.append(
                f"GPU 数量 ({num_gpus}) 低于推荐最小值 ({recipe.min_gpus})"
            )

    # 检查学习率范围（简单检查）
    if "learning_rate" in config:
        lr = config["learning_rate"]
        if lr > 1e-3:
            warnings.append(f"学习率 ({lr}) 可能过大，建议 < 1e-3")
        elif lr < 1e-8:
            warnings.append(f"学习率 ({lr}) 可能过小，建议 > 1e-8")

    return warnings
