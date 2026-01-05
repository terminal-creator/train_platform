"""
Model Surgery API Router
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List

from ..models.surgery import (
    MergeRequest,
    MergeResponse,
    MergeScanRequest,
    MergeScanResponse,
    MergeScanResult,
    CheckpointSelectRequest,
    CheckpointSelectResponse,
    CheckpointMetricsResponse,
    OverfittingWarning,
    EMARequest,
    EMAResponse,
    SWARequest,
    SWAResponse,
)
from ...core.model_merger import ModelMerger, MergeConfig, MergeMethod, merge_models
from ...core.checkpoint_selector import (
    CheckpointSelector,
    SelectionCriteria,
    select_best_checkpoint,
)

router = APIRouter(prefix="/surgery", tags=["Model Surgery"])


@router.post("/merge", response_model=MergeResponse)
async def merge_models_endpoint(request: MergeRequest) -> MergeResponse:
    """
    Merge multiple models using specified method.

    Supported methods:
    - linear: Simple weighted average
    - slerp: Spherical linear interpolation (best for 2 models)
    - ties: TrIm, Elect Sign & Merge (good for multiple fine-tuned models)
    - dare: Drop And REscale (allows merging more models)
    - swa: Stochastic Weight Averaging (for checkpoints from same run)
    """
    try:
        result = merge_models(
            models=request.models,
            method=request.method.value,
            weights=request.weights,
            output_path=request.output_path,
            interpolation_t=request.interpolation_t,
            density=request.density,
            drop_rate=request.drop_rate,
            start_step=request.start_step,
        )

        return MergeResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/merge/scan", response_model=MergeScanResponse)
async def scan_merge_ratios(request: MergeScanRequest) -> MergeScanResponse:
    """
    Scan different merge ratios to find optimal configuration.

    This will merge models at various ratios and optionally evaluate each.
    """
    try:
        merger = ModelMerger()
        results = merger.scan_merge_ratios(
            model_a=request.model_a,
            model_b=request.model_b,
            method=MergeMethod(request.method.value),
            num_points=request.num_points,
            output_dir=request.output_dir,
        )

        scan_results = [MergeScanResult(**r) for r in results]

        # Find best ratio (by success for now, evaluation would need benchmark runner)
        successful_results = [r for r in scan_results if r.success]
        best_ratio = successful_results[len(successful_results) // 2].ratio if successful_results else 0.5

        return MergeScanResponse(
            results=scan_results,
            best_ratio=best_ratio,
            best_score=None,  # Would need evaluation
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/checkpoint/select", response_model=CheckpointSelectResponse)
async def select_checkpoint(request: CheckpointSelectRequest) -> CheckpointSelectResponse:
    """
    Intelligently select the best checkpoint based on criteria.

    Criteria options:
    - highest_reward: Select checkpoint with highest reward
    - highest_benchmark: Select checkpoint with best benchmark scores
    - lowest_kl: Select checkpoint with lowest KL divergence
    - balanced: Balance between reward, benchmark, and KL
    - custom: Use custom formula (e.g., "0.5*gsm8k + 0.3*math - 0.2*kl")
    """
    try:
        result = select_best_checkpoint(
            experiment_path=request.experiment_path,
            criteria=request.criteria.value,
            custom_formula=request.custom_formula,
            benchmark_weights=request.benchmark_weights,
        )

        if not result["success"]:
            return CheckpointSelectResponse(
                success=False,
                message=result.get("message", "Selection failed"),
            )

        return CheckpointSelectResponse(
            success=True,
            recommended=CheckpointMetricsResponse(**result["recommended"]),
            score=result["score"],
            criteria=result["criteria"],
            reasoning=result["reasoning"],
            alternatives=[CheckpointMetricsResponse(**a) for a in result["alternatives"]],
            timeline=[CheckpointMetricsResponse(**t) for t in result["timeline"]],
            overfitting_warnings=[OverfittingWarning(**w) for w in result["overfitting_warnings"]],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ema", response_model=EMAResponse)
async def extract_ema_weights(request: EMARequest) -> EMAResponse:
    """
    Extract EMA (Exponential Moving Average) weights from training run.

    EMA weights often have better generalization than final weights.
    """
    # This would need integration with training logs
    # For now, return placeholder
    return EMAResponse(
        success=False,
        output_path=None,
        message="EMA extraction requires training with EMA callback enabled",
        decay=request.decay,
    )


@router.post("/swa", response_model=SWAResponse)
async def swa_average(request: SWARequest) -> SWAResponse:
    """
    Perform Stochastic Weight Averaging on multiple checkpoints.

    SWA averages weights from multiple checkpoints, often improving generalization.
    """
    try:
        result = merge_models(
            models=request.checkpoint_paths,
            method="swa",
            output_path=request.output_path,
        )

        return SWAResponse(
            success=result["success"],
            output_path=result.get("output_path"),
            message=result["message"],
            num_checkpoints_averaged=result.get("metadata", {}).get("num_checkpoints", len(request.checkpoint_paths)),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/methods")
async def list_merge_methods() -> Dict[str, Any]:
    """
    List available merge methods with descriptions.
    """
    return {
        "methods": [
            {
                "id": "linear",
                "name": "Linear Merge",
                "description": "Simple weighted average of model weights",
                "min_models": 2,
                "max_models": None,
                "parameters": ["weights"],
            },
            {
                "id": "slerp",
                "name": "SLERP (Spherical Linear Interpolation)",
                "description": "Interpolates along the shortest path on a hypersphere",
                "min_models": 2,
                "max_models": 2,
                "parameters": ["interpolation_t"],
            },
            {
                "id": "ties",
                "name": "TIES (TrIm, Elect Sign & Merge)",
                "description": "Trims small weights, resolves sign conflicts, then merges",
                "min_models": 2,
                "max_models": None,
                "parameters": ["density"],
            },
            {
                "id": "dare",
                "name": "DARE (Drop And REscale)",
                "description": "Randomly drops delta weights and rescales",
                "min_models": 2,
                "max_models": None,
                "parameters": ["drop_rate"],
            },
            {
                "id": "swa",
                "name": "SWA (Stochastic Weight Averaging)",
                "description": "Averages weights from multiple checkpoints of same training run",
                "min_models": 2,
                "max_models": None,
                "parameters": ["start_step"],
            },
        ]
    }


@router.get("/selection-criteria")
async def list_selection_criteria() -> Dict[str, Any]:
    """
    List available checkpoint selection criteria.
    """
    return {
        "criteria": [
            {
                "id": "highest_reward",
                "name": "Highest Reward",
                "description": "Select checkpoint with highest average reward",
            },
            {
                "id": "highest_benchmark",
                "name": "Highest Benchmark",
                "description": "Select checkpoint with best benchmark scores",
            },
            {
                "id": "lowest_kl",
                "name": "Lowest KL Divergence",
                "description": "Select checkpoint closest to reference model",
            },
            {
                "id": "balanced",
                "name": "Balanced",
                "description": "Balance reward, benchmark, and KL divergence",
            },
            {
                "id": "custom",
                "name": "Custom Formula",
                "description": "Use custom formula to score checkpoints",
            },
        ]
    }


@router.get("/checkpoints/{job_id}")
async def list_job_checkpoints(job_id: str) -> Dict[str, Any]:
    """
    List available checkpoints for a training job.

    Returns all saved checkpoints with their metrics and paths.
    """
    from sqlmodel import Session, select
    from ...core.database import engine, Checkpoint

    with Session(engine) as session:
        statement = select(Checkpoint).where(
            Checkpoint.job_uuid == job_id
        ).order_by(Checkpoint.step)
        checkpoints = session.exec(statement).all()

        return {
            "job_id": job_id,
            "checkpoints": [
                {
                    "step": cp.step,
                    "path": cp.path,
                    "metrics": cp.metrics,
                    "eval_results": cp.eval_results,
                    "created_at": cp.created_at.isoformat() if cp.created_at else None,
                }
                for cp in checkpoints
            ],
            "total": len(checkpoints),
        }


@router.get("/model-info")
async def get_model_info(model_path: str) -> Dict[str, Any]:
    """
    Get information about a model.

    Returns model architecture details, parameter count, and layer information.
    """
    import os
    import json
    from pathlib import Path

    path = Path(model_path)

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

    result = {
        "path": str(path.absolute()),
        "exists": True,
        "is_directory": path.is_dir(),
    }

    # Check for config.json
    config_path = path / "config.json" if path.is_dir() else path.parent / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)

        result["config"] = {
            "model_type": config.get("model_type"),
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": config.get("num_attention_heads"),
            "intermediate_size": config.get("intermediate_size"),
            "vocab_size": config.get("vocab_size"),
            "max_position_embeddings": config.get("max_position_embeddings"),
        }

        # Estimate parameter count
        if config.get("hidden_size") and config.get("num_hidden_layers"):
            h = config["hidden_size"]
            l = config["num_hidden_layers"]
            v = config.get("vocab_size", 32000)
            i = config.get("intermediate_size", 4 * h)

            # Rough estimate for transformer
            embed_params = v * h * 2  # input + output embeddings
            layer_params = (4 * h * h + 3 * h * i) * l  # attention + MLP per layer
            total_params = embed_params + layer_params

            result["estimated_params"] = {
                "total": total_params,
                "total_billions": round(total_params / 1e9, 2),
            }

    # Check for model files
    if path.is_dir():
        model_files = []
        total_size = 0

        for f in path.iterdir():
            if f.suffix in [".safetensors", ".bin", ".pt"]:
                size = f.stat().st_size
                model_files.append({
                    "name": f.name,
                    "size_mb": round(size / 1024 / 1024, 2),
                })
                total_size += size

        result["model_files"] = model_files
        result["total_size_gb"] = round(total_size / 1024 / 1024 / 1024, 2)

    return result


@router.post("/merge/preview")
async def preview_merge(request: MergeRequest) -> Dict[str, Any]:
    """
    Preview a merge operation without actually performing it.

    Returns information about what would be merged and estimated output size.
    """
    import os
    from pathlib import Path

    result = {
        "method": request.method.value,
        "num_models": len(request.models),
        "weights": request.weights,
        "output_path": request.output_path,
        "valid": True,
        "issues": [],
    }

    # Validate model paths
    model_info = []
    total_size = 0

    for i, model_path in enumerate(request.models):
        path = Path(model_path)
        info = {
            "path": model_path,
            "exists": path.exists(),
            "index": i,
        }

        if path.exists():
            if path.is_dir():
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            else:
                size = path.stat().st_size
            info["size_gb"] = round(size / 1024 / 1024 / 1024, 2)
            total_size = max(total_size, size)  # Output size ~ largest input
        else:
            result["valid"] = False
            result["issues"].append(f"Model {i} not found: {model_path}")

        model_info.append(info)

    result["models"] = model_info
    result["estimated_output_size_gb"] = round(total_size / 1024 / 1024 / 1024, 2)

    # Method-specific validation
    if request.method.value == "slerp" and len(request.models) != 2:
        result["valid"] = False
        result["issues"].append("SLERP requires exactly 2 models")

    if request.weights and len(request.weights) != len(request.models):
        result["valid"] = False
        result["issues"].append(f"Weights count ({len(request.weights)}) doesn't match model count ({len(request.models)})")

    return result


@router.post("/checkpoint/compare")
async def compare_checkpoints(
    job_id: str,
    checkpoint_steps: List[int],
) -> Dict[str, Any]:
    """
    Compare multiple checkpoints from the same job.

    Returns metrics comparison and recommendations.
    """
    from sqlmodel import Session, select
    from ...core.database import engine, Checkpoint

    with Session(engine) as session:
        statement = select(Checkpoint).where(
            Checkpoint.job_uuid == job_id,
            Checkpoint.step.in_(checkpoint_steps)
        ).order_by(Checkpoint.step)
        checkpoints = session.exec(statement).all()

        if not checkpoints:
            raise HTTPException(
                status_code=404,
                detail=f"No checkpoints found for job {job_id} at steps {checkpoint_steps}"
            )

        comparison = []
        for cp in checkpoints:
            comparison.append({
                "step": cp.step,
                "path": cp.path,
                "metrics": cp.metrics,
                "eval_results": cp.eval_results,
            })

        # Find best checkpoint based on available metrics
        best_idx = 0
        best_score = float("-inf")

        for i, cp in enumerate(comparison):
            score = 0
            if cp["metrics"]:
                score += cp["metrics"].get("reward_mean", 0)
            if cp["eval_results"]:
                score += sum(cp["eval_results"].values()) / len(cp["eval_results"])

            if score > best_score:
                best_score = score
                best_idx = i

        return {
            "job_id": job_id,
            "checkpoints": comparison,
            "recommended_index": best_idx,
            "recommended_step": comparison[best_idx]["step"] if comparison else None,
            "total": len(comparison),
        }
