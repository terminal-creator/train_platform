"""
Compute Calculator API Router
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from ..models.compute import (
    ComputeRequest,
    ComputeResponse,
    MemoryEstimateRequest,
    MemoryEstimateResponse,
    MemoryBreakdownResponse,
    WarningResponse,
    ComputeConfigResponse,
    ComputeSummaryResponse,
)
from ...core.compute_calculator import calculate_compute_config
from ...core.memory_estimator import estimate_memory

router = APIRouter(prefix="/compute", tags=["Compute Calculator"])


@router.post("/calculate", response_model=ComputeResponse)
async def calculate_config(request: ComputeRequest) -> ComputeResponse:
    """
    Calculate optimal verl configuration based on hardware and model specs.

    This endpoint analyzes your hardware setup and model requirements to recommend:
    - Optimal ZeRO stage
    - Micro batch size
    - Gradient accumulation steps
    - Tensor parallel size for rollout
    - And more...
    """
    try:
        result = calculate_compute_config(
            model_size=request.model_size.value,
            gpu_type=request.gpu_type.value,
            num_gpus=request.num_gpus,
            context_length=request.context_length,
            training_type=request.training_type.value,
            lora_enabled=request.lora_enabled,
            lora_rank=request.lora_rank,
            target_global_batch_size=request.target_global_batch_size,
        )

        return ComputeResponse(
            config=ComputeConfigResponse(**result["config"]),
            yaml=result["yaml"],
            memory_estimate=MemoryBreakdownResponse(**result["memory_estimate"]),
            zero_stage=result["zero_stage"],
            warnings=[WarningResponse(**w) for w in result["warnings"]],
            recommendations=result["recommendations"],
            summary=ComputeSummaryResponse(**result["summary"]),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory", response_model=MemoryEstimateResponse)
async def estimate_memory_usage(request: MemoryEstimateRequest) -> MemoryEstimateResponse:
    """
    Estimate GPU memory usage for a specific configuration.

    Returns detailed breakdown of memory usage including:
    - Model weights
    - Optimizer states
    - Gradients
    - Activations
    - KV Cache
    """
    try:
        result = estimate_memory(
            model_size=request.model_size.value,
            gpu_type=request.gpu_type.value,
            num_gpus=request.num_gpus,
            batch_size=request.batch_size,
            seq_len=request.seq_len,
            precision=request.precision.value,
            zero_stage=request.zero_stage,
            training_type=request.training_type.value,
            lora_enabled=request.lora_enabled,
            lora_rank=request.lora_rank,
        )

        return MemoryEstimateResponse(
            breakdown=MemoryBreakdownResponse(**result["breakdown"]),
            recommended_zero_stage=result["recommended_zero_stage"],
            recommended_max_batch_size=result["recommended_max_batch_size"],
            model_info=result["model_info"],
            gpu_info=result["gpu_info"],
            warnings=[WarningResponse(**w) for w in result["warnings"]],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gpu-types")
async def list_gpu_types() -> Dict[str, Any]:
    """
    List supported GPU types with their specifications.
    """
    return {
        "gpu_types": [
            # NVIDIA GPUs (CUDA backend)
            {"id": "A100-40G", "name": "NVIDIA A100 40GB", "memory_gb": 40, "fp16_tflops": 312, "backend": "cuda"},
            {"id": "A100-80G", "name": "NVIDIA A100 80GB", "memory_gb": 80, "fp16_tflops": 312, "backend": "cuda"},
            {"id": "H100-80G", "name": "NVIDIA H100 80GB", "memory_gb": 80, "fp16_tflops": 989, "backend": "cuda"},
            {"id": "H100-SXM", "name": "NVIDIA H100 SXM", "memory_gb": 80, "fp16_tflops": 989, "backend": "cuda"},
            {"id": "A800-80G", "name": "NVIDIA A800 80GB", "memory_gb": 80, "fp16_tflops": 312, "backend": "cuda"},
            {"id": "H800-80G", "name": "NVIDIA H800 80GB", "memory_gb": 80, "fp16_tflops": 989, "backend": "cuda"},
            {"id": "RTX4090", "name": "NVIDIA RTX 4090", "memory_gb": 24, "fp16_tflops": 82.6, "backend": "cuda"},
            {"id": "L40S", "name": "NVIDIA L40S", "memory_gb": 48, "fp16_tflops": 362, "backend": "cuda"},
            # Apple Silicon (MPS backend)
            {"id": "M1-Max-32G", "name": "Apple M1 Max 32GB", "memory_gb": 32, "fp16_tflops": 10.4, "backend": "mps"},
            {"id": "M1-Max-64G", "name": "Apple M1 Max 64GB", "memory_gb": 64, "fp16_tflops": 10.4, "backend": "mps"},
            {"id": "M2-Max-32G", "name": "Apple M2 Max 32GB", "memory_gb": 32, "fp16_tflops": 13.6, "backend": "mps"},
            {"id": "M2-Max-96G", "name": "Apple M2 Max 96GB", "memory_gb": 96, "fp16_tflops": 13.6, "backend": "mps"},
            {"id": "M2-Ultra-128G", "name": "Apple M2 Ultra 128GB", "memory_gb": 128, "fp16_tflops": 27.2, "backend": "mps"},
            {"id": "M2-Ultra-192G", "name": "Apple M2 Ultra 192GB", "memory_gb": 192, "fp16_tflops": 27.2, "backend": "mps"},
            {"id": "M3-Max-36G", "name": "Apple M3 Max 36GB", "memory_gb": 36, "fp16_tflops": 14.2, "backend": "mps"},
            {"id": "M3-Max-128G", "name": "Apple M3 Max 128GB", "memory_gb": 128, "fp16_tflops": 14.2, "backend": "mps"},
            {"id": "M4-Max-36G", "name": "Apple M4 Max 36GB", "memory_gb": 36, "fp16_tflops": 16.0, "backend": "mps"},
            {"id": "M4-Max-48G", "name": "Apple M4 Max 48GB", "memory_gb": 48, "fp16_tflops": 16.0, "backend": "mps"},
            {"id": "M4-Max-128G", "name": "Apple M4 Max 128GB", "memory_gb": 128, "fp16_tflops": 16.0, "backend": "mps"},
        ]
    }


@router.get("/model-sizes")
async def list_model_sizes() -> Dict[str, Any]:
    """
    List supported model sizes with their specifications.
    """
    return {
        "model_sizes": [
            {"id": "0.5B", "name": "0.5B", "params_billion": 0.5, "hidden_size": 896, "num_layers": 24},
            {"id": "1.5B", "name": "1.5B", "params_billion": 1.5, "hidden_size": 1536, "num_layers": 28},
            {"id": "3B", "name": "3B", "params_billion": 3, "hidden_size": 2048, "num_layers": 36},
            {"id": "7B", "name": "7B", "params_billion": 7, "hidden_size": 3584, "num_layers": 28},
            {"id": "14B", "name": "14B", "params_billion": 14, "hidden_size": 5120, "num_layers": 40},
            {"id": "32B", "name": "32B", "params_billion": 32, "hidden_size": 5120, "num_layers": 64},
            {"id": "72B", "name": "72B", "params_billion": 72, "hidden_size": 8192, "num_layers": 80},
        ]
    }


@router.get("/training-types")
async def list_training_types() -> Dict[str, Any]:
    """
    List supported training types with their descriptions.
    """
    return {
        "training_types": [
            {
                "id": "sft",
                "name": "Supervised Fine-Tuning",
                "description": "Standard supervised fine-tuning on instruction-response pairs",
                "needs_critic": False,
                "needs_reward": False,
            },
            {
                "id": "ppo",
                "name": "Proximal Policy Optimization",
                "description": "RLHF with PPO algorithm, requires critic and reward model",
                "needs_critic": True,
                "needs_reward": True,
            },
            {
                "id": "grpo",
                "name": "Group Relative Policy Optimization",
                "description": "Efficient RL without critic, good for math/code tasks",
                "needs_critic": False,
                "needs_reward": True,
            },
            {
                "id": "dpo",
                "name": "Direct Preference Optimization",
                "description": "Preference learning without explicit reward model",
                "needs_critic": False,
                "needs_reward": False,
            },
            {
                "id": "gspo",
                "name": "Group Self-Play Preference Optimization",
                "description": "Self-play preference learning, similar to GRPO",
                "needs_critic": False,
                "needs_reward": True,
            },
        ]
    }
