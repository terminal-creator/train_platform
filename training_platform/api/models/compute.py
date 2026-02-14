"""
Pydantic models for Compute Calculator API
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class GPUType(str, Enum):
    # Consumer GPUs
    RTX3090 = "RTX3090"
    RTX4080 = "RTX4080"
    RTX4090 = "RTX4090"
    RTX5090 = "RTX5090"
    # Data Center GPUs - NVIDIA A-series
    A100_40G = "A100-40G"
    A100_80G = "A100-80G"
    A800_80G = "A800-80G"
    # Data Center GPUs - NVIDIA H-series
    H100_80G = "H100-80G"
    H100_SXM = "H100-SXM"
    H100_NVL = "H100-NVL"
    H800_80G = "H800-80G"
    H200_141G = "H200-141G"
    # Data Center GPUs - NVIDIA L-series
    L40 = "L40"
    L40S = "L40S"
    # Data Center GPUs - NVIDIA B-series (Blackwell)
    B100 = "B100"
    B200 = "B200"
    GB200 = "GB200"
    # Apple Silicon (MPS backend)
    M1_MAX_32G = "M1-Max-32G"
    M1_MAX_64G = "M1-Max-64G"
    M2_MAX_32G = "M2-Max-32G"
    M2_MAX_96G = "M2-Max-96G"
    M2_ULTRA_128G = "M2-Ultra-128G"
    M2_ULTRA_192G = "M2-Ultra-192G"
    M3_MAX_36G = "M3-Max-36G"
    M3_MAX_128G = "M3-Max-128G"
    M4_MAX_36G = "M4-Max-36G"
    M4_MAX_48G = "M4-Max-48G"
    M4_MAX_128G = "M4-Max-128G"


class ModelSize(str, Enum):
    B05 = "0.5B"
    B06 = "0.6B"
    B15 = "1.5B"
    B3 = "3B"
    B7 = "7B"
    B8 = "8B"
    B14 = "14B"
    B32 = "32B"
    B70 = "70B"
    B72 = "72B"
    B110 = "110B"
    B235 = "235B"
    B405 = "405B"
    B671 = "671B"


class TrainingType(str, Enum):
    SFT = "sft"
    PPO = "ppo"
    GRPO = "grpo"
    DPO = "dpo"
    GSPO = "gspo"


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"


class ComputeRequest(BaseModel):
    """Request for compute configuration calculation"""
    model_size: ModelSize = Field(default=ModelSize.B7, description="Model size")
    gpu_type: GPUType = Field(default=GPUType.A100_80G, description="GPU type")
    num_gpus: int = Field(default=8, ge=1, le=1024, description="Number of GPUs")
    context_length: int = Field(default=4096, ge=128, le=131072, description="Context length")
    training_type: TrainingType = Field(default=TrainingType.GRPO, description="Training type")
    lora_enabled: bool = Field(default=False, description="Enable LoRA")
    lora_rank: int = Field(default=8, ge=1, le=256, description="LoRA rank")
    target_global_batch_size: int = Field(default=256, ge=1, description="Target global batch size")
    precision: Precision = Field(default=Precision.BF16, description="Training precision")

    class Config:
        json_schema_extra = {
            "example": {
                "model_size": "7B",
                "gpu_type": "A100-80G",
                "num_gpus": 8,
                "context_length": 4096,
                "training_type": "grpo",
                "lora_enabled": False,
                "lora_rank": 8,
                "target_global_batch_size": 256,
                "precision": "bf16",
            }
        }


class MemoryBreakdownResponse(BaseModel):
    """Memory breakdown response"""
    model_weights_gb: float
    optimizer_states_gb: float
    gradients_gb: float
    activations_gb: float
    kv_cache_gb: float
    total_gb: float
    per_gpu_gb: float
    utilization_percent: float


class WarningResponse(BaseModel):
    """Warning response"""
    level: str  # error, warning, info
    message: str


class ComputeConfigResponse(BaseModel):
    """Response for compute configuration"""
    actor: Dict[str, Any]
    critic: Dict[str, Any]
    rollout: Dict[str, Any]
    ref: Dict[str, Any]
    trainer: Dict[str, Any]
    algorithm: Dict[str, Any]
    computed: Dict[str, Any]


class ComputeSummaryResponse(BaseModel):
    """Summary of compute configuration"""
    model: str
    gpus: str
    training_type: str
    lora: str
    micro_batch_size: int
    global_batch_size: int
    estimated_memory_per_gpu: str
    memory_utilization: str


class ComputeResponse(BaseModel):
    """Full compute calculation response"""
    config: ComputeConfigResponse
    yaml: str
    memory_estimate: MemoryBreakdownResponse
    zero_stage: int
    warnings: List[WarningResponse]
    recommendations: List[str]
    summary: ComputeSummaryResponse


class MemoryEstimateRequest(BaseModel):
    """Request for memory estimation only"""
    model_size: ModelSize = Field(default=ModelSize.B7)
    gpu_type: GPUType = Field(default=GPUType.A100_80G)
    num_gpus: int = Field(default=8, ge=1)
    batch_size: int = Field(default=4, ge=1)
    seq_len: int = Field(default=4096, ge=128)
    precision: Precision = Field(default=Precision.BF16)
    zero_stage: int = Field(default=2, ge=0, le=3)
    training_type: TrainingType = Field(default=TrainingType.GRPO)
    lora_enabled: bool = Field(default=False)
    lora_rank: int = Field(default=8, ge=1)


class MemoryEstimateResponse(BaseModel):
    """Response for memory estimation"""
    breakdown: MemoryBreakdownResponse
    recommended_zero_stage: int
    recommended_max_batch_size: int
    model_info: Dict[str, Any]
    gpu_info: Dict[str, Any]
    warnings: List[WarningResponse]
