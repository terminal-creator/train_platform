"""
Compute Calculator for verl Training
Generates optimal verl configurations based on hardware and model specs
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum
import math

from .memory_estimator import (
    MemoryEstimator,
    ModelConfig,
    GPUConfig,
    LoRAConfig,
    Precision,
    ZeROStage,
    TrainingType,
    estimate_memory,
)


class ShardingStrategy(Enum):
    NO_SHARD = "NO_SHARD"
    SHARD_GRAD_OP = "SHARD_GRAD_OP"  # ZeRO-2
    FULL_SHARD = "FULL_SHARD"  # ZeRO-3


@dataclass
class VerlConfig:
    """Generated verl configuration"""
    # Actor config
    actor_strategy: str
    actor_fsdp_sharding: str
    actor_micro_batch_size: int
    actor_micro_batch_size_per_gpu: int
    actor_lr: float

    # Rollout config
    rollout_tp_size: int
    rollout_gpu_memory_utilization: float
    rollout_n: int  # Number of samples per prompt

    # Trainer config
    gradient_accumulation_steps: int
    global_batch_size: int
    total_epochs: int

    # Algorithm config
    algorithm: str
    kl_coef: float
    entropy_coef: float
    clip_ratio: float

    # Critic config (for PPO)
    critic_enabled: bool
    critic_lr: Optional[float]

    # LoRA config
    lora_enabled: bool
    lora_rank: int
    lora_alpha: int

    # Offload config
    cpu_offload: bool
    activation_checkpointing: bool

    # Reference model
    ref_offload: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actor": {
                "strategy": self.actor_strategy,
                "fsdp_config": {
                    "sharding_strategy": self.actor_fsdp_sharding,
                },
                "micro_batch_size": self.actor_micro_batch_size,
                "micro_batch_size_per_gpu": self.actor_micro_batch_size_per_gpu,
                "optim": {
                    "lr": self.actor_lr,
                },
                "lora": {
                    "enabled": self.lora_enabled,
                    "rank": self.lora_rank,
                    "alpha": self.lora_alpha,
                } if self.lora_enabled else None,
                "activation_checkpointing": self.activation_checkpointing,
            },
            "critic": {
                "enabled": self.critic_enabled,
                "optim": {
                    "lr": self.critic_lr,
                } if self.critic_enabled else None,
            },
            "rollout": {
                "tensor_parallel_size": self.rollout_tp_size,
                "gpu_memory_utilization": self.rollout_gpu_memory_utilization,
                "n": self.rollout_n,
            },
            "ref": {
                "offload": self.ref_offload,
            },
            "trainer": {
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "total_epochs": self.total_epochs,
            },
            "algorithm": {
                "name": self.algorithm,
                "kl_coef": self.kl_coef,
                "entropy_coef": self.entropy_coef,
                "clip_ratio": self.clip_ratio,
            },
            "computed": {
                "global_batch_size": self.global_batch_size,
                "cpu_offload": self.cpu_offload,
            },
        }

    def to_yaml_string(self) -> str:
        """Convert to YAML string for verl"""
        import yaml
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)


class ComputeCalculator:
    """
    Compute Calculator that generates optimal verl configurations
    """

    # Default learning rates by model size
    DEFAULT_LR = {
        "0.5B": 5e-6,
        "1.5B": 3e-6,
        "3B": 2e-6,
        "7B": 1e-6,
        "14B": 5e-7,
        "32B": 3e-7,
        "72B": 1e-7,
    }

    # Algorithm specific configs
    ALGORITHM_CONFIGS = {
        "ppo": {
            "kl_coef": 0.02,
            "entropy_coef": 0.01,
            "clip_ratio": 0.2,
            "needs_critic": True,
            "rollout_n": 1,
        },
        "grpo": {
            "kl_coef": 0.02,
            "entropy_coef": 0.0,
            "clip_ratio": 0.2,
            "needs_critic": False,
            "rollout_n": 8,
        },
        "dpo": {
            "kl_coef": 0.1,
            "entropy_coef": 0.0,
            "clip_ratio": 0.0,
            "needs_critic": False,
            "rollout_n": 1,
        },
        "sft": {
            "kl_coef": 0.0,
            "entropy_coef": 0.0,
            "clip_ratio": 0.0,
            "needs_critic": False,
            "rollout_n": 1,
        },
        "gspo": {
            "kl_coef": 0.02,
            "entropy_coef": 0.0,
            "clip_ratio": 0.2,
            "needs_critic": False,
            "rollout_n": 8,
        },
    }

    def __init__(
        self,
        model_size: str = "7B",
        gpu_type: str = "A100-80G",
        num_gpus: int = 8,
        context_length: int = 4096,
        training_type: str = "grpo",
        lora_enabled: bool = False,
        lora_rank: int = 8,
        target_global_batch_size: int = 256,
    ):
        self.model_size = model_size
        self.gpu_type = gpu_type
        self.num_gpus = num_gpus
        self.context_length = context_length
        self.training_type = training_type
        self.lora_enabled = lora_enabled
        self.lora_rank = lora_rank
        self.target_global_batch_size = target_global_batch_size

        self.model_config = ModelConfig.from_model_size(model_size)
        self.gpu_config = GPUConfig.from_gpu_type(gpu_type)

    def calculate(self) -> Dict[str, Any]:
        """
        Calculate optimal configuration and return comprehensive result
        """
        # Get algorithm config
        algo_config = self.ALGORITHM_CONFIGS.get(
            self.training_type,
            self.ALGORITHM_CONFIGS["grpo"]
        )

        # Determine optimal ZeRO stage
        zero_stage = self._determine_zero_stage()

        # Estimate memory and find max batch size
        lora_config = LoRAConfig(enabled=self.lora_enabled, rank=self.lora_rank)

        estimator = MemoryEstimator(
            model_config=self.model_config,
            gpu_config=self.gpu_config,
            num_gpus=self.num_gpus,
            precision=Precision.BF16,
            zero_stage=zero_stage,
            training_type=TrainingType(self.training_type),
            lora_config=lora_config,
        )

        # Find optimal micro batch size
        max_batch = estimator.recommend_max_batch_size(
            self.context_length,
            target_utilization=0.85
        )
        micro_batch = min(max_batch, 4)  # Cap at 4 for stability
        micro_batch = max(micro_batch, 1)  # At least 1

        # Calculate gradient accumulation
        effective_batch_per_step = micro_batch * self.num_gpus
        if algo_config["rollout_n"] > 1:
            effective_batch_per_step *= algo_config["rollout_n"]

        gradient_accumulation = max(
            1,
            self.target_global_batch_size // effective_batch_per_step
        )
        actual_global_batch = effective_batch_per_step * gradient_accumulation

        # Determine tensor parallel size for rollout
        tp_size = self._determine_tp_size()

        # Determine if CPU offload is needed
        memory_estimate = estimator.estimate_total(micro_batch, self.context_length)
        cpu_offload = memory_estimate.utilization_percent > 90

        # Get learning rate
        lr = self.DEFAULT_LR.get(self.model_size, 1e-6)
        if self.lora_enabled:
            lr *= 10  # LoRA typically uses higher LR

        # Determine strategy based on hardware
        if self.gpu_config.is_apple_silicon:
            # Apple Silicon uses MPS backend, single GPU only
            actor_strategy = "mps"
            sharding_strategy = "NO_SHARD"
            effective_num_gpus = 1  # Force single GPU for Apple Silicon
        else:
            actor_strategy = "fsdp"
            sharding_strategy = self._get_sharding_strategy(zero_stage)
            effective_num_gpus = self.num_gpus

        # Build config
        verl_config = VerlConfig(
            actor_strategy=actor_strategy,
            actor_fsdp_sharding=sharding_strategy,
            actor_micro_batch_size=micro_batch * effective_num_gpus,
            actor_micro_batch_size_per_gpu=micro_batch,
            actor_lr=lr,
            rollout_tp_size=tp_size,
            rollout_gpu_memory_utilization=0.85 if not self.gpu_config.is_apple_silicon else 0.7,
            rollout_n=algo_config["rollout_n"],
            gradient_accumulation_steps=gradient_accumulation,
            global_batch_size=actual_global_batch,
            total_epochs=3,
            algorithm=self.training_type,
            kl_coef=algo_config["kl_coef"],
            entropy_coef=algo_config["entropy_coef"],
            clip_ratio=algo_config["clip_ratio"],
            critic_enabled=algo_config["needs_critic"] and not self.gpu_config.is_apple_silicon,
            critic_lr=lr * 2 if algo_config["needs_critic"] and not self.gpu_config.is_apple_silicon else None,
            lora_enabled=self.lora_enabled,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_rank * 2,
            cpu_offload=cpu_offload,
            activation_checkpointing=True,
            ref_offload=memory_estimate.utilization_percent > 75,
        )

        # Generate warnings and recommendations
        warnings = self._generate_warnings(memory_estimate, verl_config)
        recommendations = self._generate_recommendations(memory_estimate, verl_config)

        return {
            "config": verl_config.to_dict(),
            "yaml": verl_config.to_yaml_string(),
            "memory_estimate": memory_estimate.to_dict(),
            "zero_stage": zero_stage.value,
            "warnings": warnings,
            "recommendations": recommendations,
            "summary": {
                "model": self.model_size,
                "gpus": f"{self.num_gpus}x {self.gpu_type}",
                "training_type": self.training_type,
                "lora": f"rank={self.lora_rank}" if self.lora_enabled else "disabled",
                "micro_batch_size": micro_batch,
                "global_batch_size": actual_global_batch,
                "estimated_memory_per_gpu": f"{memory_estimate.per_gpu_gb:.1f} GB",
                "memory_utilization": f"{memory_estimate.utilization_percent:.1f}%",
            },
        }

    def _determine_zero_stage(self) -> ZeROStage:
        """Determine optimal ZeRO stage"""
        # Apple Silicon doesn't support ZeRO
        if self.gpu_config.is_apple_silicon:
            return ZeROStage.ZERO_0

        model_memory_gb = self.model_config.params_billion * 2  # FP16
        single_gpu_memory = self.gpu_config.memory_gb
        total_gpu_memory = single_gpu_memory * self.num_gpus

        # If using LoRA, we need much less memory
        if self.lora_enabled:
            return ZeROStage.ZERO_2  # ZeRO-2 is usually sufficient for LoRA

        if model_memory_gb < single_gpu_memory * 0.3:
            return ZeROStage.ZERO_0
        elif model_memory_gb < single_gpu_memory * 0.5:
            return ZeROStage.ZERO_1
        elif model_memory_gb * 4 < total_gpu_memory:
            return ZeROStage.ZERO_2
        else:
            return ZeROStage.ZERO_3

    def _determine_tp_size(self) -> int:
        """Determine tensor parallel size for inference/rollout"""
        # Apple Silicon doesn't support tensor parallelism
        if self.gpu_config.is_apple_silicon:
            return 1

        model_memory_gb = self.model_config.params_billion * 2
        single_gpu_memory = self.gpu_config.memory_gb * 0.85  # Leave some headroom

        if model_memory_gb < single_gpu_memory:
            return 1
        elif model_memory_gb < single_gpu_memory * 2:
            return 2
        elif model_memory_gb < single_gpu_memory * 4:
            return 4
        else:
            return 8

    def _get_sharding_strategy(self, zero_stage: ZeROStage) -> str:
        """Convert ZeRO stage to FSDP sharding strategy"""
        mapping = {
            ZeROStage.ZERO_0: "NO_SHARD",
            ZeROStage.ZERO_1: "NO_SHARD",  # FSDP doesn't have ZeRO-1 equivalent
            ZeROStage.ZERO_2: "SHARD_GRAD_OP",
            ZeROStage.ZERO_3: "FULL_SHARD",
        }
        return mapping.get(zero_stage, "SHARD_GRAD_OP")

    def _generate_warnings(self, memory: Any, config: VerlConfig) -> List[Dict[str, str]]:
        """Generate warnings based on configuration"""
        warnings = []

        if memory.utilization_percent > 95:
            warnings.append({
                "level": "error",
                "message": "显存使用率超过 95%，存在 OOM 风险。建议减小批量大小或启用 LoRA。",
            })
        elif memory.utilization_percent > 85:
            warnings.append({
                "level": "warning",
                "message": "显存使用率较高 (>85%)，建议减小批量大小以提高稳定性。",
            })

        if config.cpu_offload:
            warnings.append({
                "level": "info",
                "message": "已启用 CPU 卸载，训练速度可能变慢但内存效率更高。",
            })

        if self.model_size in ["32B", "72B"] and not self.lora_enabled:
            warnings.append({
                "level": "warning",
                "message": f"{self.model_size} 模型全参数训练需要大量资源，建议使用 LoRA。",
            })

        # Apple Silicon specific warnings
        if self.gpu_config.is_apple_silicon:
            warnings.append({
                "level": "info",
                "message": "Apple Silicon 使用 MPS 后端进行训练。",
            })
            if self.num_gpus > 1:
                warnings.append({
                    "level": "warning",
                    "message": "Apple Silicon 不支持多 GPU 并行，将使用单 GPU 训练。",
                })
            if not self.lora_enabled and self.model_config.params_billion > 3:
                warnings.append({
                    "level": "warning",
                    "message": "在 Apple Silicon 上训练较大模型，强烈建议启用 LoRA。",
                })

        return warnings

    def _generate_recommendations(self, memory: Any, config: VerlConfig) -> List[str]:
        """Generate recommendations for optimization"""
        recommendations = []

        if memory.utilization_percent < 60:
            recommendations.append(
                "GPU 利用率较低，可以增大批量大小以提高吞吐量。"
            )

        if not self.lora_enabled and self.model_config.params_billion > 7:
            recommendations.append(
                "建议使用 LoRA 以加快训练速度并降低显存需求。"
            )

        if self.training_type == "grpo" and config.rollout_n < 8:
            recommendations.append(
                "GRPO 算法在更多采样下效果更好，建议将 rollout.n 增加到 8-16。"
            )

        if config.gradient_accumulation_steps > 16:
            recommendations.append(
                "梯度累积步数较高可能降低训练速度，建议增加 GPU 数量。"
            )

        # Apple Silicon specific recommendations
        if self.gpu_config.is_apple_silicon:
            recommendations.append(
                "Apple Silicon 建议使用 FP16 精度（BF16 支持有限）。"
            )
            if self.model_config.params_billion <= 3:
                recommendations.append(
                    "0.5B-3B 模型可在 Apple Silicon 上进行全参数微调。"
                )
            elif self.model_config.params_billion <= 7:
                recommendations.append(
                    "7B 模型建议使用 LoRA（rank 8-16）进行高效微调。"
                )
            else:
                recommendations.append(
                    "大于 7B 的模型在 Apple Silicon 上可能需要量化（如 4-bit QLoRA）。"
                )

        return recommendations


def calculate_compute_config(
    model_size: str = "7B",
    gpu_type: str = "A100-80G",
    num_gpus: int = 8,
    context_length: int = 4096,
    training_type: str = "grpo",
    lora_enabled: bool = False,
    lora_rank: int = 8,
    target_global_batch_size: int = 256,
) -> Dict[str, Any]:
    """
    Convenience function to calculate optimal compute configuration
    """
    calculator = ComputeCalculator(
        model_size=model_size,
        gpu_type=gpu_type,
        num_gpus=num_gpus,
        context_length=context_length,
        training_type=training_type,
        lora_enabled=lora_enabled,
        lora_rank=lora_rank,
        target_global_batch_size=target_global_batch_size,
    )
    return calculator.calculate()
