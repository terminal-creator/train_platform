"""
Memory Estimator for LLM Training
Estimates GPU memory requirements for various training configurations
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
import math


class Precision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"


class ZeROStage(Enum):
    ZERO_0 = 0
    ZERO_1 = 1
    ZERO_2 = 2
    ZERO_3 = 3


class TrainingType(Enum):
    SFT = "sft"
    PPO = "ppo"
    GRPO = "grpo"
    DPO = "dpo"
    GSPO = "gspo"


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    name: str
    params_billion: float
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int = 152064  # Qwen default

    @classmethod
    def from_model_size(cls, size: str) -> "ModelConfig":
        """Create config from model size string like '7B', '14B', etc."""
        configs = {
            # Qwen 系列
            "0.5B": cls("Qwen2.5-0.5B", 0.5, 896, 24, 14, 4864),
            "0.6B": cls("Qwen2.5-0.6B", 0.6, 1024, 24, 16, 5504),
            "1.5B": cls("Qwen2.5-1.5B", 1.5, 1536, 28, 12, 8960),
            "3B": cls("Qwen2.5-3B", 3, 2048, 36, 16, 11008),
            "7B": cls("Qwen2.5-7B", 7, 3584, 28, 28, 18944),
            "8B": cls("Llama-3.1-8B", 8, 4096, 32, 32, 14336),
            "14B": cls("Qwen2.5-14B", 14, 5120, 40, 40, 13824),
            "32B": cls("Qwen2.5-32B", 32, 5120, 64, 40, 27648),
            "70B": cls("Llama-3.1-70B", 70, 8192, 80, 64, 28672),
            "72B": cls("Qwen2.5-72B", 72, 8192, 80, 64, 29568),
            "110B": cls("Qwen2.5-110B", 110, 10240, 96, 80, 36864),
            "235B": cls("Qwen2.5-235B", 235, 12288, 120, 96, 49152),
            "405B": cls("Llama-3.1-405B", 405, 16384, 126, 128, 53248),
            "671B": cls("DeepSeek-V3-671B", 671, 16384, 160, 128, 65536),
        }
        return configs.get(size, configs["7B"])


@dataclass
class GPUConfig:
    """GPU hardware configuration"""
    name: str
    memory_gb: float
    memory_bandwidth_gbps: float
    fp16_tflops: float
    is_apple_silicon: bool = False  # MPS backend flag

    @classmethod
    def from_gpu_type(cls, gpu_type: str) -> "GPUConfig":
        """Create config from GPU type string"""
        configs = {
            # Consumer GPUs (CUDA backend)
            "RTX3090": cls("RTX3090", 24, 936, 71),
            "RTX4080": cls("RTX4080", 16, 717, 65),
            "RTX4090": cls("RTX4090", 24, 1008, 82.6),
            "RTX5090": cls("RTX5090", 32, 1792, 209),
            # Data Center GPUs - A-series
            "A100-40G": cls("A100-40G", 40, 1555, 312),
            "A100-80G": cls("A100-80G", 80, 2039, 312),
            "A800-80G": cls("A800-80G", 80, 2039, 312),
            # Data Center GPUs - H-series (Hopper)
            "H100-80G": cls("H100-80G", 80, 3350, 989),
            "H100-SXM": cls("H100-SXM", 80, 3350, 989),
            "H100-NVL": cls("H100-NVL", 94, 3938, 835),
            "H800-80G": cls("H800-80G", 80, 3350, 989),
            "H200-141G": cls("H200-141G", 141, 4800, 989),
            # Data Center GPUs - L-series
            "L40": cls("L40", 48, 864, 181),
            "L40S": cls("L40S", 48, 864, 362),
            # Data Center GPUs - B-series (Blackwell)
            "B100": cls("B100", 192, 8000, 1800),
            "B200": cls("B200", 192, 8000, 2250),
            "GB200": cls("GB200", 384, 16000, 4500),
            # Apple Silicon (MPS backend) - unified memory architecture
            "M1-Max-32G": cls("M1-Max-32G", 32, 400, 10.4, is_apple_silicon=True),
            "M1-Max-64G": cls("M1-Max-64G", 64, 400, 10.4, is_apple_silicon=True),
            "M2-Max-32G": cls("M2-Max-32G", 32, 400, 13.6, is_apple_silicon=True),
            "M2-Max-96G": cls("M2-Max-96G", 96, 400, 13.6, is_apple_silicon=True),
            "M2-Ultra-128G": cls("M2-Ultra-128G", 128, 800, 27.2, is_apple_silicon=True),
            "M2-Ultra-192G": cls("M2-Ultra-192G", 192, 800, 27.2, is_apple_silicon=True),
            "M3-Max-36G": cls("M3-Max-36G", 36, 400, 14.2, is_apple_silicon=True),
            "M3-Max-128G": cls("M3-Max-128G", 128, 400, 14.2, is_apple_silicon=True),
            "M4-Max-36G": cls("M4-Max-36G", 36, 546, 16.0, is_apple_silicon=True),
            "M4-Max-48G": cls("M4-Max-48G", 48, 546, 16.0, is_apple_silicon=True),
            "M4-Max-128G": cls("M4-Max-128G", 128, 546, 16.0, is_apple_silicon=True),
        }
        return configs.get(gpu_type, configs["A100-80G"])


@dataclass
class LoRAConfig:
    """LoRA configuration"""
    enabled: bool = False
    rank: int = 8
    alpha: int = 16
    target_modules: list = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"]


@dataclass
class MemoryBreakdown:
    """Detailed memory breakdown"""
    model_weights_gb: float
    optimizer_states_gb: float
    gradients_gb: float
    activations_gb: float
    kv_cache_gb: float
    total_gb: float
    per_gpu_gb: float
    utilization_percent: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_weights_gb": round(self.model_weights_gb, 2),
            "optimizer_states_gb": round(self.optimizer_states_gb, 2),
            "gradients_gb": round(self.gradients_gb, 2),
            "activations_gb": round(self.activations_gb, 2),
            "kv_cache_gb": round(self.kv_cache_gb, 2),
            "total_gb": round(self.total_gb, 2),
            "per_gpu_gb": round(self.per_gpu_gb, 2),
            "utilization_percent": round(self.utilization_percent, 1),
        }


class MemoryEstimator:
    """
    Estimates GPU memory requirements for LLM training
    Supports various training types: SFT, PPO, GRPO, DPO, GSPO
    """

    BYTES_PER_PARAM = {
        Precision.FP32: 4,
        Precision.FP16: 2,
        Precision.BF16: 2,
        Precision.FP8: 1,
    }

    def __init__(
        self,
        model_config: ModelConfig,
        gpu_config: GPUConfig,
        num_gpus: int = 8,
        precision: Precision = Precision.BF16,
        zero_stage: ZeROStage = ZeROStage.ZERO_2,
        training_type: TrainingType = TrainingType.GRPO,
        lora_config: Optional[LoRAConfig] = None,
    ):
        self.model = model_config
        self.gpu = gpu_config
        self.num_gpus = num_gpus
        self.precision = precision
        self.zero_stage = zero_stage
        self.training_type = training_type
        self.lora = lora_config or LoRAConfig()

    def estimate_model_weights(self) -> float:
        """Estimate model weights memory in GB"""
        bytes_per_param = self.BYTES_PER_PARAM[self.precision]
        total_bytes = self.model.params_billion * 1e9 * bytes_per_param

        # ZeRO-3 shards model weights
        if self.zero_stage == ZeROStage.ZERO_3:
            total_bytes /= self.num_gpus

        return total_bytes / (1024**3)

    def estimate_optimizer_states(self) -> float:
        """
        Estimate optimizer states memory in GB
        AdamW: master weights (fp32) + momentum (fp32) + variance (fp32) = 12 bytes/param
        """
        if self.lora.enabled:
            # Only LoRA params need optimizer states
            lora_params = self._estimate_lora_params()
            total_bytes = lora_params * 12
        else:
            total_bytes = self.model.params_billion * 1e9 * 12

        # ZeRO-1/2/3 shard optimizer states
        if self.zero_stage.value >= 1:
            total_bytes /= self.num_gpus

        return total_bytes / (1024**3)

    def estimate_gradients(self) -> float:
        """Estimate gradients memory in GB"""
        bytes_per_param = self.BYTES_PER_PARAM[self.precision]

        if self.lora.enabled:
            lora_params = self._estimate_lora_params()
            total_bytes = lora_params * bytes_per_param
        else:
            total_bytes = self.model.params_billion * 1e9 * bytes_per_param

        # ZeRO-2/3 shard gradients
        if self.zero_stage.value >= 2:
            total_bytes /= self.num_gpus

        return total_bytes / (1024**3)

    def estimate_activations(self, batch_size: int, seq_len: int) -> float:
        """
        Estimate activation memory in GB
        Activations scale with batch_size * seq_len * hidden_size * num_layers
        """
        # Approximate activation memory per token per layer
        # Includes: attention scores, intermediate activations, etc.
        bytes_per_token_per_layer = (
            self.model.hidden_size * 4 +  # attention input/output
            self.model.hidden_size * 4 +  # attention scores (approx)
            self.model.intermediate_size * 2  # FFN intermediate
        ) * self.BYTES_PER_PARAM[self.precision]

        total_bytes = (
            batch_size * seq_len *
            self.model.num_layers *
            bytes_per_token_per_layer
        )

        # Apply activation checkpointing factor (saves ~60% activation memory)
        # Most frameworks enable this by default for large models
        checkpoint_factor = 0.4  # Only keep 40% of activations
        total_bytes *= checkpoint_factor

        return total_bytes / (1024**3)

    def estimate_kv_cache(self, batch_size: int, seq_len: int) -> float:
        """
        Estimate KV cache memory in GB (mainly for inference/generation)
        KV cache = 2 * num_layers * hidden_size * seq_len * batch_size * bytes
        """
        bytes_per_param = self.BYTES_PER_PARAM[self.precision]
        head_dim = self.model.hidden_size // self.model.num_attention_heads

        # KV cache size per token
        kv_per_token = 2 * self.model.num_layers * self.model.num_attention_heads * head_dim
        total_bytes = batch_size * seq_len * kv_per_token * bytes_per_param

        return total_bytes / (1024**3)

    def estimate_total(
        self,
        batch_size: int,
        seq_len: int,
        include_kv_cache: bool = True,
    ) -> MemoryBreakdown:
        """
        Estimate total memory and return detailed breakdown
        """
        model_weights = self.estimate_model_weights()
        optimizer_states = self.estimate_optimizer_states()
        gradients = self.estimate_gradients()
        activations = self.estimate_activations(batch_size, seq_len)
        kv_cache = self.estimate_kv_cache(batch_size, seq_len) if include_kv_cache else 0

        # Additional overhead for training type
        overhead_factor = self._get_training_overhead_factor()

        total = (model_weights + optimizer_states + gradients + activations + kv_cache) * overhead_factor
        per_gpu = total  # Already divided by num_gpus where applicable

        # Add fixed overhead (CUDA/MPS context, etc.)
        if self.gpu.is_apple_silicon:
            # MPS has lower overhead, but unified memory leaves less available
            # Reserve ~10% for system + ~1GB for MPS context
            per_gpu += 1.0
        else:
            per_gpu += 1.5  # ~1.5GB CUDA overhead

        utilization = (per_gpu / self.gpu.memory_gb) * 100

        return MemoryBreakdown(
            model_weights_gb=model_weights,
            optimizer_states_gb=optimizer_states,
            gradients_gb=gradients,
            activations_gb=activations,
            kv_cache_gb=kv_cache,
            total_gb=total,
            per_gpu_gb=per_gpu,
            utilization_percent=utilization,
        )

    def _estimate_lora_params(self) -> int:
        """Estimate number of LoRA parameters"""
        # Each target module: 2 * hidden_size * rank
        params_per_module = 2 * self.model.hidden_size * self.lora.rank
        # Multiply by number of modules and layers
        total_params = (
            params_per_module *
            len(self.lora.target_modules) *
            self.model.num_layers
        )
        return total_params

    def _get_training_overhead_factor(self) -> float:
        """Get overhead factor based on training type"""
        factors = {
            TrainingType.SFT: 1.0,
            TrainingType.PPO: 1.5,   # Actor + Critic + Reference
            TrainingType.GRPO: 1.2,  # Actor + Reference (no Critic)
            TrainingType.DPO: 1.3,   # Actor + Reference
            TrainingType.GSPO: 1.25, # Similar to GRPO
        }
        return factors.get(self.training_type, 1.0)

    def recommend_max_batch_size(self, seq_len: int, target_utilization: float = 0.85) -> int:
        """
        Recommend maximum batch size given sequence length and target GPU utilization
        """
        target_memory = self.gpu.memory_gb * target_utilization

        # Binary search for max batch size
        low, high = 1, 128
        best_batch = 1

        while low <= high:
            mid = (low + high) // 2
            breakdown = self.estimate_total(mid, seq_len)

            if breakdown.per_gpu_gb <= target_memory:
                best_batch = mid
                low = mid + 1
            else:
                high = mid - 1

        return best_batch

    def recommend_zero_stage(self) -> ZeROStage:
        """
        Recommend ZeRO stage based on model size and GPU configuration
        """
        # Apple Silicon doesn't support ZeRO (no DeepSpeed/FSDP on MPS)
        if self.gpu.is_apple_silicon:
            return ZeROStage.ZERO_0  # Single GPU training only

        total_model_memory = self.model.params_billion * 2  # FP16
        single_gpu_memory = self.gpu.memory_gb
        total_gpu_memory = single_gpu_memory * self.num_gpus

        if total_model_memory < single_gpu_memory * 0.3:
            # Model fits easily, use ZeRO-0 for max throughput
            return ZeROStage.ZERO_0
        elif total_model_memory < single_gpu_memory * 0.6:
            # Model fits but tight, use ZeRO-1
            return ZeROStage.ZERO_1
        elif total_model_memory * 4 < total_gpu_memory:
            # Need optimizer sharding, use ZeRO-2
            return ZeROStage.ZERO_2
        else:
            # Need full sharding
            return ZeROStage.ZERO_3


def estimate_memory(
    model_size: str = "7B",
    gpu_type: str = "A100-80G",
    num_gpus: int = 8,
    batch_size: int = 4,
    seq_len: int = 4096,
    precision: str = "bf16",
    zero_stage: int = 2,
    training_type: str = "grpo",
    lora_enabled: bool = False,
    lora_rank: int = 8,
) -> Dict[str, Any]:
    """
    Convenience function to estimate memory with simple parameters
    """
    model_config = ModelConfig.from_model_size(model_size)
    gpu_config = GPUConfig.from_gpu_type(gpu_type)

    lora_config = LoRAConfig(enabled=lora_enabled, rank=lora_rank)

    estimator = MemoryEstimator(
        model_config=model_config,
        gpu_config=gpu_config,
        num_gpus=num_gpus,
        precision=Precision(precision),
        zero_stage=ZeROStage(zero_stage),
        training_type=TrainingType(training_type),
        lora_config=lora_config,
    )

    breakdown = estimator.estimate_total(batch_size, seq_len)
    recommended_zero = estimator.recommend_zero_stage()
    max_batch = estimator.recommend_max_batch_size(seq_len)

    return {
        "breakdown": breakdown.to_dict(),
        "recommended_zero_stage": recommended_zero.value,
        "recommended_max_batch_size": max_batch,
        "model_info": {
            "name": model_config.name,
            "params_billion": model_config.params_billion,
            "hidden_size": model_config.hidden_size,
            "num_layers": model_config.num_layers,
        },
        "gpu_info": {
            "name": gpu_config.name,
            "memory_gb": gpu_config.memory_gb,
        },
        "warnings": _generate_warnings(breakdown, gpu_config),
    }


def _generate_warnings(breakdown: MemoryBreakdown, gpu: GPUConfig) -> list:
    """Generate warnings based on memory estimates"""
    warnings = []

    if breakdown.utilization_percent > 95:
        warnings.append({
            "level": "error",
            "message": "显存使用率超过 95%，存在 OOM 风险。",
        })
    elif breakdown.utilization_percent > 85:
        warnings.append({
            "level": "warning",
            "message": "显存使用率较高，建议减小批量大小。",
        })

    if breakdown.activations_gb > breakdown.model_weights_gb * 2:
        warnings.append({
            "level": "info",
            "message": "激活值占用较大，建议启用激活检查点或减小序列长度。",
        })

    # Apple Silicon specific warnings
    if gpu.is_apple_silicon:
        warnings.append({
            "level": "info",
            "message": "Apple Silicon 使用 MPS 后端，不支持多卡并行和 ZeRO。建议使用 LoRA 微调。",
        })
        if breakdown.total_gb > gpu.memory_gb * 0.7:
            warnings.append({
                "level": "warning",
                "message": "统一内存架构下，建议保留更多内存给系统。考虑使用更小的模型或启用 LoRA。",
            })

    return warnings
