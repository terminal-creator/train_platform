"""
Monitoring API Router

Provides endpoints for:
- Push mode metrics reporting from training scripts
- GPU usage monitoring
- Gradient visualization
- System health checks
"""

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import json
import logging

from ..models.training import (
    GradientHeatmapResponse,
    GradientStats,
    EvaluationResult,
    EvaluationResultsResponse,
    ResourceUsage,
    ResourceUsageResponse,
)
from sqlmodel import Session
from ...core.database import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


# ============== Push Mode Models ==============

class MetricsReport(BaseModel):
    """Metrics report from training script"""
    job_uuid: str = Field(..., description="Job UUID")
    step: int = Field(..., description="Training step")
    epoch: Optional[int] = Field(0, description="Current epoch")

    # Standard metrics
    policy_loss: Optional[float] = None
    value_loss: Optional[float] = None
    total_loss: Optional[float] = None
    reward_mean: Optional[float] = None
    reward_std: Optional[float] = None
    kl_divergence: Optional[float] = None
    entropy: Optional[float] = None
    learning_rate: Optional[float] = None

    # Custom metrics
    extra_metrics: Dict[str, float] = Field(default_factory=dict)


class GpuUsageReport(BaseModel):
    """GPU usage report"""
    job_uuid: str
    gpu_index: int
    utilization: float = Field(..., ge=0, le=100, description="GPU utilization %")
    memory_used: float = Field(..., ge=0, description="Memory used in GB")
    memory_total: float = Field(..., ge=0, description="Total memory in GB")
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")


class StatusReport(BaseModel):
    """Status update from training script"""
    job_uuid: str
    status: str = Field(..., description="Status: running, completed, failed, paused")
    message: Optional[str] = None
    current_step: Optional[int] = None
    total_steps: Optional[int] = None


class LogReport(BaseModel):
    """Log entry from training script"""
    job_uuid: str
    level: str = Field("INFO", description="Log level: DEBUG, INFO, WARNING, ERROR")
    message: str
    timestamp: Optional[datetime] = None


class GradientStatsReport(BaseModel):
    """Gradient statistics from training callback"""
    layer_name: str
    norm: float
    mean: float
    std: float
    max_abs: float
    min_abs: float
    sparsity: float = 0.0


class GradientBatchReport(BaseModel):
    """Batch of gradient statistics"""
    job_uuid: str
    gradients: List[Dict[str, Any]]  # List of {step, stats: [...]}


class SampleDataReport(BaseModel):
    """Sample data for data review"""
    step: int
    index: int
    prompt: str
    response: str
    reward: float
    kl: float = 0.0
    log_prob: float = 0.0
    ref_log_prob: float = 0.0
    advantage: float = 0.0
    label: Optional[str] = None


class SampleBatchReport(BaseModel):
    """Batch of sample data"""
    job_uuid: str
    samples: List[SampleDataReport]


class CheckpointReport(BaseModel):
    """Checkpoint notification"""
    job_uuid: str
    step: int
    path: str
    eval_results: Optional[Dict[str, float]] = None


# ============== In-memory storage for samples and gradients ==============
# Note: In production, these should be stored in a database
_gradient_history: Dict[str, List[Dict[str, Any]]] = {}
_sample_history: Dict[str, List[Dict[str, Any]]] = {}


# ============== Push Mode Endpoints ==============

@router.post("/report")
async def report_metrics(report: MetricsReport) -> Dict[str, Any]:
    """
    Report training metrics from training script (Push mode).

    This endpoint allows training scripts to directly push metrics
    instead of relying on log parsing.

    Example usage in training script:
    ```python
    import requests

    def report_to_platform(job_uuid, step, metrics):
        requests.post(
            "http://platform:8000/api/v1/monitoring/report",
            json={
                "job_uuid": job_uuid,
                "step": step,
                **metrics
            }
        )
    ```
    """
    from .websocket import metrics_collector

    # Combine all metrics
    all_metrics = {
        'epoch': report.epoch,
        'policy_loss': report.policy_loss,
        'value_loss': report.value_loss,
        'total_loss': report.total_loss,
        'reward_mean': report.reward_mean,
        'reward_std': report.reward_std,
        'kl_divergence': report.kl_divergence,
        'entropy': report.entropy,
        'learning_rate': report.learning_rate,
        **report.extra_metrics,
    }

    # Remove None values
    all_metrics = {k: v for k, v in all_metrics.items() if v is not None}

    try:
        success = await metrics_collector.report_metrics(
            job_uuid=report.job_uuid,
            step=report.step,
            metrics=all_metrics,
        )

        if success:
            return {
                "success": True,
                "message": f"Metrics reported for step {report.step}",
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Job {report.job_uuid} not found"
            )
    except Exception as e:
        logger.error(f"Failed to report metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report/batch")
async def report_metrics_batch(reports: List[MetricsReport]) -> Dict[str, Any]:
    """
    Report multiple metrics at once (batch mode).

    Useful for buffered reporting to reduce HTTP overhead.
    """
    from .websocket import metrics_collector

    success_count = 0
    errors = []

    for report in reports:
        all_metrics = {
            'epoch': report.epoch,
            'policy_loss': report.policy_loss,
            'value_loss': report.value_loss,
            'total_loss': report.total_loss,
            'reward_mean': report.reward_mean,
            'reward_std': report.reward_std,
            'kl_divergence': report.kl_divergence,
            'entropy': report.entropy,
            **report.extra_metrics,
        }
        all_metrics = {k: v for k, v in all_metrics.items() if v is not None}

        try:
            success = await metrics_collector.report_metrics(
                job_uuid=report.job_uuid,
                step=report.step,
                metrics=all_metrics,
            )
            if success:
                success_count += 1
            else:
                errors.append(f"Job {report.job_uuid} not found")
        except Exception as e:
            errors.append(f"Step {report.step}: {str(e)}")

    return {
        "success": len(errors) == 0,
        "reported": success_count,
        "total": len(reports),
        "errors": errors if errors else None,
    }


@router.post("/gpu")
async def report_gpu_usage(report: GpuUsageReport) -> Dict[str, Any]:
    """
    Report GPU usage from training script.
    """
    from sqlmodel import Session
    from ...core.database import engine, GpuUsageRecord
    from .websocket import push_gpu_update

    try:
        with Session(engine) as session:
            record = GpuUsageRecord(
                job_uuid=report.job_uuid,
                gpu_index=report.gpu_index,
                utilization=report.utilization,
                memory_used=report.memory_used,
                memory_total=report.memory_total,
                temperature=report.temperature,
            )
            session.add(record)
            session.commit()

        # Push to WebSocket
        await push_gpu_update(report.job_uuid, [{
            "gpu_index": report.gpu_index,
            "utilization": report.utilization,
            "memory_used": report.memory_used,
            "memory_total": report.memory_total,
            "temperature": report.temperature,
        }])

        return {"success": True}
    except Exception as e:
        logger.error(f"Failed to report GPU usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gpu/batch")
async def report_gpu_usage_batch(reports: List[GpuUsageReport]) -> Dict[str, Any]:
    """
    Report GPU usage for multiple GPUs at once.
    """
    from sqlmodel import Session
    from ...core.database import engine, GpuUsageRecord
    from .websocket import push_gpu_update

    try:
        with Session(engine) as session:
            for report in reports:
                record = GpuUsageRecord(
                    job_uuid=report.job_uuid,
                    gpu_index=report.gpu_index,
                    utilization=report.utilization,
                    memory_used=report.memory_used,
                    memory_total=report.memory_total,
                    temperature=report.temperature,
                )
                session.add(record)
            session.commit()

        # Group by job_uuid and push
        by_job: Dict[str, List] = {}
        for report in reports:
            if report.job_uuid not in by_job:
                by_job[report.job_uuid] = []
            by_job[report.job_uuid].append({
                "gpu_index": report.gpu_index,
                "utilization": report.utilization,
                "memory_used": report.memory_used,
                "memory_total": report.memory_total,
                "temperature": report.temperature,
            })

        for job_uuid, gpu_data in by_job.items():
            await push_gpu_update(job_uuid, gpu_data)

        return {"success": True, "reported": len(reports)}
    except Exception as e:
        logger.error(f"Failed to report GPU usage batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/status")
async def report_status(report: StatusReport) -> Dict[str, Any]:
    """
    Report job status change from training script.
    """
    from sqlmodel import Session
    from ...core.database import engine, JobRepository, JobStatus
    from .websocket import push_status_update

    try:
        with Session(engine) as session:
            job_repo = JobRepository(session)
            job = job_repo.get_by_uuid(report.job_uuid)

            if not job:
                raise HTTPException(status_code=404, detail="Job not found")

            # Map status string to enum
            status_map = {
                'running': JobStatus.RUNNING,
                'completed': JobStatus.COMPLETED,
                'failed': JobStatus.FAILED,
                'paused': JobStatus.PAUSED,
                'pending': JobStatus.PENDING,
                'cancelled': JobStatus.CANCELLED,
            }

            if report.status.lower() in status_map:
                job.status = status_map[report.status.lower()]

            if report.current_step is not None:
                job.current_step = report.current_step
            if report.total_steps is not None:
                job.total_steps = report.total_steps

            if report.status.lower() == 'completed':
                job.completed_at = datetime.utcnow()

            job_repo.update(job)

        # Push to WebSocket
        await push_status_update(report.job_uuid, report.status, report.message)

        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to report status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/log")
async def report_log(report: LogReport) -> Dict[str, Any]:
    """
    Report a log entry from training script.
    """
    from sqlmodel import Session
    from ...core.database import engine, TrainingLog
    from .websocket import push_log_entry

    try:
        with Session(engine) as session:
            log_entry = TrainingLog(
                job_uuid=report.job_uuid,
                level=report.level,
                message=report.message,
                timestamp=report.timestamp or datetime.utcnow(),
            )
            session.add(log_entry)
            session.commit()

        # Push to WebSocket
        await push_log_entry(report.job_uuid, report.level, report.message)

        return {"success": True}
    except Exception as e:
        logger.error(f"Failed to report log: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gradients/batch")
async def report_gradients_batch(report: GradientBatchReport) -> Dict[str, Any]:
    """
    Report gradient statistics from training callback.

    This endpoint receives gradient statistics for multiple layers at each step.
    Data is stored in-memory for querying.
    """
    try:
        job_uuid = report.job_uuid

        if job_uuid not in _gradient_history:
            _gradient_history[job_uuid] = []

        for grad_entry in report.gradients:
            _gradient_history[job_uuid].append({
                "step": grad_entry.get("step", 0),
                "timestamp": datetime.utcnow().isoformat(),
                "stats": grad_entry.get("stats", []),
            })

        # Keep only last 1000 entries per job
        if len(_gradient_history[job_uuid]) > 1000:
            _gradient_history[job_uuid] = _gradient_history[job_uuid][-1000:]

        return {"success": True, "stored": len(report.gradients)}
    except Exception as e:
        logger.error(f"Failed to store gradient stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/samples/batch")
async def report_samples_batch(report: SampleBatchReport) -> Dict[str, Any]:
    """
    Report sample data from training callback.

    This endpoint receives samples collected during training for data review.
    Samples include prompts, responses, rewards, and other metrics.
    """
    try:
        job_uuid = report.job_uuid

        if job_uuid not in _sample_history:
            _sample_history[job_uuid] = []

        for sample in report.samples:
            _sample_history[job_uuid].append({
                "step": sample.step,
                "index": sample.index,
                "prompt": sample.prompt,
                "response": sample.response,
                "reward": sample.reward,
                "kl": sample.kl,
                "log_prob": sample.log_prob,
                "ref_log_prob": sample.ref_log_prob,
                "advantage": sample.advantage,
                "label": sample.label,
                "timestamp": datetime.utcnow().isoformat(),
            })

        # Keep only last 500 samples per job
        if len(_sample_history[job_uuid]) > 500:
            _sample_history[job_uuid] = _sample_history[job_uuid][-500:]

        return {"success": True, "stored": len(report.samples)}
    except Exception as e:
        logger.error(f"Failed to store samples: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/checkpoint")
async def report_checkpoint(report: CheckpointReport) -> Dict[str, Any]:
    """
    Report checkpoint save from training callback.
    """
    from sqlmodel import Session
    from ...core.database import engine, Checkpoint, CheckpointRepository

    try:
        with Session(engine) as session:
            checkpoint_repo = CheckpointRepository(session)

            checkpoint = Checkpoint(
                job_uuid=report.job_uuid,
                step=report.step,
                path=report.path,
                metrics=report.eval_results or {},
                eval_results=report.eval_results,
            )

            session.add(checkpoint)
            session.commit()

        logger.info(f"Checkpoint recorded: job={report.job_uuid} step={report.step}")
        return {"success": True, "step": report.step}
    except Exception as e:
        logger.error(f"Failed to record checkpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}/collected-gradients")
async def get_collected_gradients(
    job_id: str,
    start_step: Optional[int] = Query(None, description="Start step"),
    end_step: Optional[int] = Query(None, description="End step"),
    limit: int = Query(100, ge=1, le=1000, description="Max entries to return"),
) -> Dict[str, Any]:
    """
    Get collected gradient statistics for a job.

    Returns gradient data collected by the VerlCallback during training.
    """
    if job_id not in _gradient_history:
        return {"job_id": job_id, "gradients": [], "total": 0}

    gradients = _gradient_history[job_id]

    # Filter by step range
    if start_step is not None:
        gradients = [g for g in gradients if g["step"] >= start_step]
    if end_step is not None:
        gradients = [g for g in gradients if g["step"] <= end_step]

    # Sort by step and limit
    gradients = sorted(gradients, key=lambda x: x["step"])[-limit:]

    return {
        "job_id": job_id,
        "gradients": gradients,
        "total": len(gradients),
    }


@router.get("/{job_id}/collected-samples")
async def get_collected_samples(
    job_id: str,
    start_step: Optional[int] = Query(None, description="Start step"),
    end_step: Optional[int] = Query(None, description="End step"),
    min_reward: Optional[float] = Query(None, description="Minimum reward filter"),
    max_reward: Optional[float] = Query(None, description="Maximum reward filter"),
    label: Optional[str] = Query(None, description="Filter by label"),
    limit: int = Query(50, ge=1, le=500, description="Max samples to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
) -> Dict[str, Any]:
    """
    Get collected sample data for a job.

    Returns samples collected by the VerlCallback during training.
    Useful for data review and debugging.
    """
    if job_id not in _sample_history:
        return {"job_id": job_id, "samples": [], "total": 0}

    samples = _sample_history[job_id]

    # Apply filters
    if start_step is not None:
        samples = [s for s in samples if s["step"] >= start_step]
    if end_step is not None:
        samples = [s for s in samples if s["step"] <= end_step]
    if min_reward is not None:
        samples = [s for s in samples if s["reward"] >= min_reward]
    if max_reward is not None:
        samples = [s for s in samples if s["reward"] <= max_reward]
    if label is not None:
        samples = [s for s in samples if s.get("label") == label]

    total = len(samples)

    # Sort by step and paginate
    samples = sorted(samples, key=lambda x: (x["step"], x["index"]))
    samples = samples[offset:offset + limit]

    return {
        "job_id": job_id,
        "samples": samples,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{job_id}/sample-stats")
async def get_sample_stats(job_id: str) -> Dict[str, Any]:
    """
    Get statistics about collected samples for a job.

    Returns summary statistics including reward distribution,
    sample counts by step, and label distributions.
    """
    import statistics

    if job_id not in _sample_history or not _sample_history[job_id]:
        return {
            "job_id": job_id,
            "total_samples": 0,
            "step_range": [0, 0],
            "reward_stats": {},
            "label_distribution": {},
        }

    samples = _sample_history[job_id]

    # Calculate statistics
    rewards = [s["reward"] for s in samples]
    kls = [s["kl"] for s in samples if s["kl"] != 0]

    reward_stats = {
        "mean": statistics.mean(rewards) if rewards else 0,
        "std": statistics.stdev(rewards) if len(rewards) > 1 else 0,
        "min": min(rewards) if rewards else 0,
        "max": max(rewards) if rewards else 0,
    }

    kl_stats = {
        "mean": statistics.mean(kls) if kls else 0,
        "std": statistics.stdev(kls) if len(kls) > 1 else 0,
        "min": min(kls) if kls else 0,
        "max": max(kls) if kls else 0,
    }

    # Step range
    steps = [s["step"] for s in samples]

    # Label distribution
    label_counts = {}
    for s in samples:
        label = s.get("label") or "unlabeled"
        label_counts[label] = label_counts.get(label, 0) + 1

    return {
        "job_id": job_id,
        "total_samples": len(samples),
        "step_range": [min(steps), max(steps)] if steps else [0, 0],
        "reward_stats": reward_stats,
        "kl_stats": kl_stats,
        "label_distribution": label_counts,
    }


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Check monitoring system health.
    """
    from .websocket import metrics_collector, manager

    return {
        "status": "healthy",
        "metrics_collector_running": metrics_collector._running,
        "active_websocket_connections": len(manager.broadcast_connections),
        "jobs_being_monitored": len(manager.active_connections),
        "timestamp": datetime.utcnow().isoformat(),
    }


# ============== Original Monitoring Endpoints ==============

# WebSocket connections for real-time updates
_ws_connections: Dict[str, List[WebSocket]] = {}


@router.get("/{job_id}/gradient-heatmap", response_model=GradientHeatmapResponse)
async def get_gradient_heatmap(
    job_id: str,
    start_step: Optional[int] = Query(None, description="Start step"),
    end_step: Optional[int] = Query(None, description="End step"),
    layer_filter: Optional[str] = Query(None, description="Filter layers by name"),
    session: Session = Depends(get_session),
) -> GradientHeatmapResponse:
    """
    Get gradient heatmap data for visualization.

    Returns gradient magnitude data organized as layers x steps for heatmap rendering.
    For a 14B model like Qwen2.5-14B, there are typically 40 layers.
    """
    from ...core.database import JobRepository, MetricsRepository
    import random
    import math

    job_repo = JobRepository(session)
    metrics_repo = MetricsRepository(session)

    job = job_repo.get_by_uuid(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Get actual training steps from metrics
    db_metrics = metrics_repo.get_metrics(job_uuid=job_id)
    if db_metrics:
        all_steps = sorted(set(m.step for m in db_metrics))
        # Sample ~10 evenly spaced steps for the heatmap
        if len(all_steps) > 10:
            indices = [int(i * (len(all_steps) - 1) / 9) for i in range(10)]
            steps = [all_steps[i] for i in indices]
        else:
            steps = all_steps
    else:
        # Fallback: use job's total_steps or default
        max_step = job.total_steps or 100
        step_interval = max(1, max_step // 10)
        steps = list(range(1, max_step + 1, step_interval))

    if start_step:
        steps = [s for s in steps if s >= start_step]
    if end_step:
        steps = [s for s in steps if s <= end_step]

    # Generate realistic layers for a 14B model (40 layers)
    # For LoRA, typically only certain layers are fine-tuned
    num_layers = 40  # Qwen2.5-14B has 40 layers
    layers = []

    # Add embed_tokens
    layers.append("model.embed_tokens")

    # Add transformer layers (sample every 4 layers to avoid too many)
    for i in range(0, num_layers, 4):
        layers.append(f"model.layers.{i}.self_attn")
        layers.append(f"model.layers.{i}.mlp")

    # Add final layers
    layers.append("model.norm")
    layers.append("lm_head")

    if layer_filter:
        layers = [l for l in layers if layer_filter in l]

    # Generate demo heatmap data with realistic patterns
    # Earlier layers tend to have smaller gradients, later layers have larger gradients
    # Gradients should decrease over training as the model converges
    data = []
    for layer_idx, layer in enumerate(layers):
        layer_data = []
        # Later layers (higher index) tend to have larger gradient changes
        layer_factor = 0.5 + (layer_idx / len(layers)) * 0.5

        for step_idx, step in enumerate(steps):
            # Gradients decrease over training (model converging)
            progress = step_idx / max(1, len(steps) - 1)
            decay = math.exp(-progress * 0.5)

            # Add some noise and layer-specific variation
            base_value = layer_factor * decay * random.uniform(0.02, 0.08)
            layer_data.append(base_value)

        data.append(layer_data)

    return GradientHeatmapResponse(
        job_id=job_id,
        steps=steps,
        layers=layers,
        data=data,
    )


@router.get("/{job_id}/gradient-stats")
async def get_gradient_stats(
    job_id: str,
    step: Optional[int] = Query(None, description="Training step"),
) -> Dict[str, Any]:
    """
    Get detailed gradient statistics for a specific step.
    Shows gradient stats for representative layers across the model.
    """
    import random
    import math

    # Generate layers for a 14B model (40 layers)
    # Sample key layers: first, middle, and last few layers
    num_layers = 40
    sample_layers = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 39]  # Representative layers

    layers = ["model.embed_tokens"]

    for layer_idx in sample_layers:
        layers.extend([
            f"model.layers.{layer_idx}.self_attn.q_proj",
            f"model.layers.{layer_idx}.self_attn.k_proj",
            f"model.layers.{layer_idx}.self_attn.v_proj",
            f"model.layers.{layer_idx}.self_attn.o_proj",
            f"model.layers.{layer_idx}.mlp.gate_proj",
            f"model.layers.{layer_idx}.mlp.up_proj",
            f"model.layers.{layer_idx}.mlp.down_proj",
        ])

    layers.extend(["model.norm", "lm_head"])

    stats = []
    for layer_idx, layer in enumerate(layers):
        # Generate realistic gradient statistics
        # Later layers tend to have larger gradients
        layer_factor = 0.5 + (layer_idx / len(layers)) * 0.5

        # LoRA layers (q_proj, v_proj typically) have larger gradients
        is_lora_layer = any(x in layer for x in ['q_proj', 'v_proj'])
        lora_factor = 1.5 if is_lora_layer else 1.0

        base_std = layer_factor * lora_factor * random.uniform(0.02, 0.08)

        stats.append(GradientStats(
            layer_name=layer,
            mean=random.uniform(-0.001, 0.001) * layer_factor,
            std=base_std,
            max=base_std * random.uniform(5, 10),
            min=-base_std * random.uniform(5, 10),
        ))

    return {
        "job_id": job_id,
        "step": step,
        "stats": [s.model_dump() for s in stats],
    }


@router.get("/{job_id}/evaluations", response_model=EvaluationResultsResponse)
async def get_evaluations(
    job_id: str,
    benchmark: Optional[str] = Query(None, description="Filter by benchmark"),
) -> EvaluationResultsResponse:
    """
    Get evaluation results for checkpoints.
    """
    # Demo data
    results = [
        EvaluationResult(
            checkpoint_step=1000,
            benchmark="gsm8k",
            score=52.3,
            num_samples=1000,
            evaluated_at=datetime.utcnow(),
        ),
        EvaluationResult(
            checkpoint_step=2000,
            benchmark="gsm8k",
            score=58.1,
            num_samples=1000,
            evaluated_at=datetime.utcnow(),
        ),
        EvaluationResult(
            checkpoint_step=3000,
            benchmark="gsm8k",
            score=61.5,
            num_samples=1000,
            evaluated_at=datetime.utcnow(),
        ),
        EvaluationResult(
            checkpoint_step=1000,
            benchmark="math",
            score=15.8,
            num_samples=500,
            evaluated_at=datetime.utcnow(),
        ),
        EvaluationResult(
            checkpoint_step=2000,
            benchmark="math",
            score=19.2,
            num_samples=500,
            evaluated_at=datetime.utcnow(),
        ),
        EvaluationResult(
            checkpoint_step=3000,
            benchmark="math",
            score=22.5,
            num_samples=500,
            evaluated_at=datetime.utcnow(),
        ),
    ]

    if benchmark:
        results = [r for r in results if r.benchmark == benchmark]

    return EvaluationResultsResponse(
        job_id=job_id,
        results=results,
    )


@router.get("/{job_id}/gpu-usage")
async def get_gpu_usage(job_id: str) -> List[Dict[str, Any]]:
    """
    Get GPU usage history for a training job.
    """
    from sqlmodel import Session, select
    from ...core.database import engine, GpuUsageRecord

    with Session(engine) as session:
        statement = select(GpuUsageRecord).where(
            GpuUsageRecord.job_uuid == job_id
        ).order_by(GpuUsageRecord.timestamp.desc()).limit(100)
        records = session.exec(statement).all()

        if not records:
            # Return empty list if no records
            return []

        # Group by gpu_index, take latest for each
        gpu_data = {}
        for r in records:
            if r.gpu_index not in gpu_data:
                gpu_data[r.gpu_index] = {
                    "index": r.gpu_index,
                    "utilization": r.utilization,
                    "memory_used": r.memory_used,
                    "memory_total": r.memory_total,
                    "temperature": r.temperature,
                }

        return list(gpu_data.values())


@router.get("/{job_id}/resources", response_model=ResourceUsageResponse)
async def get_resource_usage(job_id: str) -> ResourceUsageResponse:
    """
    Get current resource usage for a training job.
    """
    import random

    # Demo data
    num_gpus = 8
    usage = ResourceUsage(
        gpu_utilization=[random.uniform(80, 100) for _ in range(num_gpus)],
        gpu_memory_used=[random.uniform(60, 75) for _ in range(num_gpus)],
        gpu_memory_total=[80.0] * num_gpus,
        cpu_utilization=random.uniform(20, 50),
        ram_used_gb=random.uniform(100, 200),
        disk_io_read_mbps=random.uniform(100, 500),
        disk_io_write_mbps=random.uniform(50, 200),
        network_recv_mbps=random.uniform(100, 1000),
        network_send_mbps=random.uniform(100, 1000),
    )

    return ResourceUsageResponse(
        job_id=job_id,
        timestamp=datetime.utcnow(),
        usage=usage,
    )


@router.websocket("/{job_id}/live")
async def websocket_live_metrics(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time metrics streaming.

    Phase 1.3 重构：
    - 从真实的指标文件读取数据（本地或 SSH）
    - 增量推送新增的指标
    - 推送训练状态（running, completed, failed）
    - 支持历史回放（playback=true）
    """
    from ...core.database import get_session, TrainingJob, JobStatus
    from ...core.metrics_reader import create_metrics_reader
    from ...core.ssh_runner import get_ssh_manager
    from sqlmodel import select

    await websocket.accept()

    # Track connection
    if job_id not in _ws_connections:
        _ws_connections[job_id] = []
    _ws_connections[job_id].append(websocket)

    try:
        # 1. 查询任务信息并获取配置（修复 P1: Session 泄漏）
        from ...core.database import engine, Session as DBSession

        with DBSession(engine) as session:
            statement = select(TrainingJob).where(TrainingJob.job_id == job_id)
            job = session.exec(statement).first()

            if not job:
                await websocket.send_json({
                    "error": "Job not found",
                    "job_id": job_id
                })
                return

            # 在 session 关闭前读取所有需要的数据
            run_mode = job.run_mode_config.get("mode", "local") if job.run_mode_config else "local"
            ssh_config = job.run_mode_config if run_mode == "ssh" else None

        # 2. 创建指标读取器（使用已读取的配置数据）
        if run_mode == "ssh" and ssh_config:
            metrics_dir = f"{ssh_config.get('ssh_working_dir', '~/verl_jobs')}/platform_metrics"
            ssh_manager = get_ssh_manager(
                host=ssh_config["ssh_host"],
                port=ssh_config.get("ssh_port", 22),
                username=ssh_config["ssh_username"],
                password=ssh_config.get("ssh_password"),
                key_path=ssh_config.get("ssh_key_path")
            )
            reader = create_metrics_reader(job_id, metrics_dir, run_mode="ssh", ssh_manager=ssh_manager)
        else:
            # 本地模式：默认指标目录
            metrics_dir = "./platform_metrics"
            reader = create_metrics_reader(job_id, metrics_dir, run_mode="local")

        logger.info(f"[WebSocket] Streaming metrics for {job_id} (mode: {run_mode})")

        # 3. 实时推送指标
        while True:
            # 读取新增的指标
            new_metrics, _ = reader.read_metrics_incremental(limit=10)

            # 读取状态
            status = reader.read_status()

            # 推送每个新增指标
            for metric in new_metrics:
                await websocket.send_json({
                    "type": "metric",
                    "data": metric
                })

            # 推送状态更新
            if status:
                await websocket.send_json({
                    "type": "status",
                    "data": status
                })

            # 如果任务已完成或失败，停止推送
            if status and status.get("status") in ["completed", "failed"]:
                logger.info(f"[WebSocket] Job {job_id} finished: {status.get('status')}")
                await websocket.send_json({
                    "type": "finished",
                    "status": status.get("status")
                })
                break

            # 等待 1 秒再读取
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        logger.info(f"[WebSocket] Client disconnected: {job_id}")
        _ws_connections[job_id].remove(websocket)
    except Exception as e:
        logger.error(f"[WebSocket] Error streaming metrics for {job_id}: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
        if websocket in _ws_connections.get(job_id, []):
            _ws_connections[job_id].remove(websocket)


@router.post("/{job_id}/alert-rules")
async def set_alert_rules(
    job_id: str,
    rules: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Set alert rules for a training job.

    Example rules:
    {
        "grad_explosion": {"condition": "grad_norm > 100", "action": "pause"},
        "kl_high": {"condition": "kl > 0.2", "action": "notify"}
    }
    """
    # Store rules (in real implementation, this would go to database)
    return {
        "job_id": job_id,
        "rules_set": len(rules),
        "message": "Alert rules configured",
    }


@router.get("/{job_id}/alerts")
async def get_alerts(
    job_id: str,
    severity: Optional[str] = Query(None, description="Filter by severity"),
) -> Dict[str, Any]:
    """
    Get triggered alerts for a training job.
    """
    # Demo data
    alerts = [
        {
            "id": "alert-001",
            "rule": "kl_high",
            "severity": "warning",
            "message": "KL divergence exceeded 0.15",
            "step": 3500,
            "timestamp": datetime.utcnow().isoformat(),
            "acknowledged": False,
        },
    ]

    if severity:
        alerts = [a for a in alerts if a["severity"] == severity]

    return {
        "job_id": job_id,
        "alerts": alerts,
        "total": len(alerts),
    }


@router.post("/{job_id}/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(job_id: str, alert_id: str) -> Dict[str, Any]:
    """
    Acknowledge an alert.
    """
    return {
        "job_id": job_id,
        "alert_id": alert_id,
        "acknowledged": True,
    }


# ============== Phase 1: Metrics Query Endpoints (from Database) ==============

@router.get("/{job_id}/metrics")
async def get_metrics_history(
    job_id: str,
    start_step: Optional[int] = Query(None, description="Start step (inclusive)"),
    end_step: Optional[int] = Query(None, description="End step (inclusive)"),
    limit: int = Query(1000, ge=1, le=10000, description="Max metrics to return"),
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    查询训练指标历史（Phase 1）

    从数据库查询指标数据，支持步骤范围过滤。

    为什么添加这个接口：
    - PlatformCallback 将指标持久化到数据库
    - 前端需要查询历史指标用于绘制图表
    - 支持范围查询，避免一次返回过多数据

    返回字段说明：
    - step: 训练步骤
    - epoch: 当前 epoch
    - loss: Loss 指标（actor_loss, critic_loss, total_loss）
    - reward: 奖励指标（mean, std, max, min）
    - kl: KL 散度（mean, max）
    - gradient: 梯度范数
    - performance: 性能指标（吞吐量、耗时等）
    - has_anomaly: 是否有异常
    """
    from ...core.database import JobRepository, MetricsRepository, TrainingMetric

    job_repo = JobRepository(session)
    metrics_repo = MetricsRepository(session)

    # 验证任务存在
    job = job_repo.get_by_uuid(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # 查询指标
    db_metrics = metrics_repo.get_metrics(
        job_uuid=job_id,
        start_step=start_step,
        end_step=end_step,
        limit=limit,
    )

    # 转换为前端友好的格式
    # 为什么：前端需要嵌套结构便于图表库使用
    metrics_list = []
    for m in db_metrics:
        metrics_list.append({
            "step": m.step,
            "epoch": m.epoch,
            "timestamp": m.timestamp.isoformat(),
            "loss": {
                "actor_loss": m.policy_loss,  # 映射回 actor_loss
                "critic_loss": m.value_loss,  # 映射回 critic_loss
                "total_loss": m.total_loss,
            },
            "reward": {
                "mean": m.reward_mean,
                "std": m.reward_std,
                "max": m.reward_max,
                "min": m.reward_min,
            },
            "kl": {
                "mean": m.kl_divergence,
                "max": m.kl_divergence_max,
            },
            "gradient": {
                "actor_norm": m.grad_norm_actor,
                "critic_norm": m.grad_norm_critic,
            },
            "performance": {
                "tokens_per_second": m.tokens_per_second,
                "step_time": m.step_time,
                "gpu_memory_allocated": m.gpu_memory_allocated_gib,
            },
            "has_anomaly": m.has_anomaly,
            "anomaly_type": m.anomaly_type,
            "anomaly_message": m.anomaly_message,
        })

    return {
        "job_id": job_id,
        "metrics": metrics_list,
        "total": len(metrics_list),
        "start_step": start_step,
        "end_step": end_step,
    }


@router.get("/{job_id}/metrics/anomalies")
async def get_anomalous_metrics(
    job_id: str,
    anomaly_type: Optional[str] = Query(None, description="Filter by anomaly type (nan, kl_explosion, loss_plateau)"),
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    查询异常指标（Phase 1）

    返回所有标记为异常的训练步骤，用于异常分析和告警。

    为什么需要这个接口：
    - PlatformCallback 自动检测异常并标记到数据库
    - 前端需要显示异常告警和异常步骤列表
    - 支持按异常类型筛选

    异常类型：
    - nan: NaN/Inf 检测
    - kl_explosion: KL 散度爆炸
    - loss_plateau: Loss 长期不下降
    """
    from ...core.database import JobRepository, TrainingMetric
    from sqlmodel import select

    job_repo = JobRepository(session)

    # 验证任务存在
    job = job_repo.get_by_uuid(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # 查询异常指标
    statement = select(TrainingMetric).where(
        TrainingMetric.job_uuid == job_id,
        TrainingMetric.has_anomaly == True
    )

    if anomaly_type:
        statement = statement.where(TrainingMetric.anomaly_type == anomaly_type)

    statement = statement.order_by(TrainingMetric.step)
    anomalies = session.exec(statement).all()

    # 转换为前端格式
    anomaly_list = []
    for a in anomalies:
        anomaly_list.append({
            "step": a.step,
            "epoch": a.epoch,
            "timestamp": a.timestamp.isoformat(),
            "anomaly_type": a.anomaly_type,
            "anomaly_message": a.anomaly_message,
            "metrics_snapshot": {
                "actor_loss": a.policy_loss,
                "reward_mean": a.reward_mean,
                "kl_mean": a.kl_divergence,
                "kl_max": a.kl_divergence_max,
            }
        })

    return {
        "job_id": job_id,
        "anomalies": anomaly_list,
        "total": len(anomaly_list),
        "anomaly_type": anomaly_type,
    }


@router.post("/{job_id}/metrics/sync")
async def sync_metrics_from_files(
    job_id: str,
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    手动触发指标同步（Phase 1）

    从 PlatformCallback 生成的 JSON Lines 文件同步指标到数据库。

    工作流程：
    1. 读取 {job_id}_metrics.jsonl 文件
    2. 解析新增的指标（增量同步）
    3. 批量插入数据库
    4. 同步异常信息

    为什么需要手动同步：
    - 训练过程中可能需要立即查看最新指标
    - 定时同步可能有延迟
    - 便于测试和调试

    使用场景：
    - 前端点击"刷新"按钮时调用
    - 训练完成后确保所有指标已同步
    """
    from ...core.database import JobRepository
    from ...core.metrics_persister import sync_metrics_for_job
    from pathlib import Path
    import os

    job_repo = JobRepository(session)

    # 验证任务存在
    job = job_repo.get_by_uuid(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # 获取指标文件目录
    # 为什么：PlatformCallback 将指标写入 platform_metrics 目录
    metrics_dir = Path(os.getenv("PLATFORM_METRICS_DIR", "./platform_metrics"))

    try:
        # 执行同步
        result = sync_metrics_for_job(
            job_uuid=job_id,
            job_id=job_id,  # 假设 job_id 用于文件名
            metrics_dir=metrics_dir,
            session=session,
        )

        return {
            "success": True,
            "job_id": job_id,
            "new_metrics_count": result["new_metrics_count"],
            "anomaly_synced": result["anomaly_synced"],
            "message": f"Synced {result['new_metrics_count']} new metrics",
        }

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Metrics file not found: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to sync metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to sync metrics: {str(e)}"
        )


@router.websocket("/{job_id}/playback")
async def websocket_metrics_playback(
    websocket: WebSocket,
    job_id: str,
    start_step: int = Query(0, description="Start step for playback"),
    end_step: Optional[int] = Query(None, description="End step for playback"),
    speed: float = Query(1.0, ge=0.1, le=10.0, description="Playback speed multiplier (1.0 = real-time)"),
):
    """
    历史指标回放 WebSocket (Phase 1.3)

    允许前端以指定速度回放历史训练指标，类似视频回放。

    为什么需要回放功能：
    - 分析训练过程：回看训练中的关键时刻
    - 调试问题：定位异常发生的步骤
    - 演示展示：向他人展示训练过程
    - 对比实验：同步回放多个实验的指标

    参数说明：
    - start_step: 从哪个 step 开始回放（默认 0）
    - end_step: 到哪个 step 结束（None 表示到最后）
    - speed: 播放速度（1.0 = 真实速度，2.0 = 2倍速，0.5 = 0.5倍速）

    消息格式：
    {
        "type": "metric",
        "data": {...},  # 指标数据
        "progress": 0.5  # 播放进度 (0-1)
    }
    """
    from ...core.database import get_session, MetricsRepository
    from sqlmodel import Session

    await websocket.accept()

    try:
        # 查询指标数据（修复 P1: Session 泄漏）
        from ...core.database import engine, Session as DBSession

        with DBSession(engine) as session:
            metrics_repo = MetricsRepository(session)

            # 获取指标列表
            db_metrics = metrics_repo.get_metrics(
                job_uuid=job_id,
                start_step=start_step,
                end_step=end_step,
                limit=10000,  # 回放模式允许更多数据
            )

            if not db_metrics:
                await websocket.send_json({
                    "type": "error",
                    "message": f"No metrics found for job {job_id}"
                })
                return

            # 在 session 关闭前将 ORM 对象转换为字典
            # 避免在 session 外访问 lazy-loaded 属性
            metrics_data = []
            for metric in db_metrics:
                metrics_data.append({
                    "step": metric.step,
                    "epoch": metric.epoch,
                    "timestamp": metric.timestamp,
                    "policy_loss": metric.policy_loss,
                    "value_loss": metric.value_loss,
                    "total_loss": metric.total_loss,
                    "reward_mean": metric.reward_mean,
                    "reward_std": metric.reward_std,
                    "reward_max": metric.reward_max,
                    "reward_min": metric.reward_min,
                    "kl_divergence": metric.kl_divergence,
                    "kl_divergence_max": metric.kl_divergence_max,
                    "grad_norm_actor": metric.grad_norm_actor,
                    "grad_norm_critic": metric.grad_norm_critic,
                    "tokens_per_second": metric.tokens_per_second,
                    "step_time": metric.step_time,
                    "gpu_memory_allocated_gib": metric.gpu_memory_allocated_gib,
                    "has_anomaly": metric.has_anomaly,
                    "anomaly_type": metric.anomaly_type,
                })

        total_metrics = len(metrics_data)
        logger.info(f"[Playback] Starting playback for {job_id}: {total_metrics} metrics, speed={speed}x")

        # 发送回放元信息
        await websocket.send_json({
            "type": "playback_info",
            "total_metrics": total_metrics,
            "start_step": start_step,
            "end_step": metrics_data[-1]["step"] if metrics_data else None,
            "speed": speed,
        })

        # 逐个推送指标
        for idx, metric in enumerate(metrics_data):
            # 转换为前端格式
            metric_data = {
                "step": metric["step"],
                "epoch": metric["epoch"],
                "timestamp": metric["timestamp"].isoformat(),
                "loss": {
                    "actor_loss": metric["policy_loss"],
                    "critic_loss": metric["value_loss"],
                    "total_loss": metric["total_loss"],
                },
                "reward": {
                    "mean": metric["reward_mean"],
                    "std": metric["reward_std"],
                    "max": metric["reward_max"],
                    "min": metric["reward_min"],
                },
                "kl": {
                    "mean": metric["kl_divergence"],
                    "max": metric["kl_divergence_max"],
                },
                "gradient": {
                    "actor_norm": metric["grad_norm_actor"],
                    "critic_norm": metric["grad_norm_critic"],
                },
                "performance": {
                    "tokens_per_second": metric["tokens_per_second"],
                    "step_time": metric["step_time"],
                    "gpu_memory_allocated": metric["gpu_memory_allocated_gib"],
                },
                "has_anomaly": metric["has_anomaly"],
                "anomaly_type": metric["anomaly_type"],
            }

            # 计算播放进度
            progress = (idx + 1) / total_metrics

            # 发送指标
            await websocket.send_json({
                "type": "metric",
                "data": metric_data,
                "progress": progress,
                "current_index": idx,
                "total": total_metrics,
            })

            # 根据速度延迟
            # 为什么：模拟真实训练的时间间隔
            # 假设每个 step 间隔 1 秒（可以根据 step_time 调整）
            base_interval = 0.5  # 基础间隔（秒）
            if metric["step_time"]:
                base_interval = min(metric["step_time"], 2.0)  # 最多 2 秒

            actual_interval = base_interval / speed
            await asyncio.sleep(actual_interval)

        # 回放完成
        logger.info(f"[Playback] Completed playback for {job_id}")
        await websocket.send_json({
            "type": "playback_complete",
            "total_metrics": total_metrics,
        })

    except WebSocketDisconnect:
        logger.info(f"[Playback] Client disconnected during playback: {job_id}")
    except Exception as e:
        logger.error(f"[Playback] Error during playback: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass


@router.get("/dashboard")
async def get_dashboard_summary() -> Dict[str, Any]:
    """
    Get summary data for the monitoring dashboard.
    Returns real data from database.
    """
    from sqlmodel import Session, select, func
    from ...core.database import engine, TrainingJob, JobStatus

    with Session(engine) as session:
        # 统计各状态任务数量
        def count_by_status(status: JobStatus) -> int:
            statement = select(func.count()).select_from(TrainingJob).where(
                TrainingJob.status == status
            )
            return session.exec(statement).one() or 0

        active_jobs = count_by_status(JobStatus.RUNNING)
        queued_jobs = count_by_status(JobStatus.QUEUED) + count_by_status(JobStatus.PENDING)
        completed_jobs = count_by_status(JobStatus.COMPLETED)
        failed_jobs = count_by_status(JobStatus.FAILED)

        # 计算总 GPU 时长 (小时)
        # TODO: 从实际训练记录计算
        total_gpu_hours = 0.0

        # 计算总训练 Token 数
        # TODO: 从实际训练记录计算
        total_training_tokens = 0

        # 暂无告警系统，返回空列表
        recent_alerts = []

        # 暂无实验评估数据，返回空列表
        top_performing_experiments = []

    return {
        "active_jobs": active_jobs,
        "queued_jobs": queued_jobs,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "total_gpu_hours": total_gpu_hours,
        "total_training_tokens": total_training_tokens,
        "recent_alerts": recent_alerts,
        "top_performing_experiments": top_performing_experiments,
    }


# ============== Phase 1.4: Diagnostics API ==============

@router.post("/{job_id}/diagnose")
async def diagnose_job(
    job_id: str,
    auto_mark_failed: bool = Query(True, description="Auto-mark job as failed if critical anomaly detected"),
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    诊断单个任务（Phase 1.4）

    执行所有异常检测并返回诊断结果。

    检测项：
    - NaN/Inf 检测
    - KL 散度爆炸
    - Loss 不下降
    - Reward 崩溃

    如果检测到严重异常（CRITICAL），且 auto_mark_failed=True，
    会自动将任务标记为失败。

    使用场景：
    - 手动诊断：用户点击"诊断"按钮
    - 自动诊断：定时任务定期调用
    """
    from ...core.diagnostics import DiagnosticService
    from ...core.database import JobRepository

    job_repo = JobRepository(session)
    job = job_repo.get_by_uuid(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    service = DiagnosticService(session)
    result = service.diagnose_job(job_id, auto_mark_failed=auto_mark_failed)

    return result


@router.get("/{job_id}/anomalies/detected")
async def get_detected_anomalies(
    job_id: str,
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    获取检测到的异常（Phase 1.4）

    返回当前检测到的所有异常，但不执行自动操作。

    与 /diagnose 的区别：
    - /diagnose: 执行检测并可能自动标记失败
    - /anomalies/detected: 仅返回检测结果，不执行操作
    """
    from ...core.diagnostics import AnomalyDetector

    detector = AnomalyDetector(session)
    anomalies = detector.detect_all(job_id)

    return {
        "job_id": job_id,
        "anomalies_count": len(anomalies),
        "anomalies": [a.to_dict() for a in anomalies],
    }


@router.post("/diagnose-all")
async def diagnose_all_running_jobs(
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    诊断所有运行中的任务（Phase 1.4）

    定期调用此接口（如每分钟一次）实现自动监控。

    工作流程：
    1. 查询所有运行中的任务
    2. 对每个任务执行异常检测
    3. 检测到严重异常时自动标记失败
    4. 返回诊断汇总

    使用场景：
    - 定时任务：cron job 每分钟调用
    - 手动触发：管理员手动执行全局诊断
    """
    from ...core.diagnostics import DiagnosticService

    service = DiagnosticService(session)
    result = service.diagnose_all_running_jobs()

    return result


@router.get("/{job_id}/health")
async def get_job_health_status(
    job_id: str,
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    获取任务健康状态（Phase 1.4）

    返回任务的整体健康评分和建议。

    健康评分：
    - 100: 完全健康，无异常
    - 80-99: 有轻微警告
    - 50-79: 有中等问题
    - 0-49: 有严重问题

    返回内容：
    - health_score: 健康评分 (0-100)
    - status: healthy, warning, critical
    - anomalies: 检测到的异常列表
    - suggestions: 诊断建议
    """
    from ...core.diagnostics import AnomalyDetector, AnomalySeverity
    from ...core.database import JobRepository

    job_repo = JobRepository(session)
    job = job_repo.get_by_uuid(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    detector = AnomalyDetector(session)
    anomalies = detector.detect_all(job_id)

    # 计算健康评分
    health_score = 100
    for anomaly in anomalies:
        if anomaly.severity == AnomalySeverity.CRITICAL:
            health_score -= 50
        elif anomaly.severity == AnomalySeverity.HIGH:
            health_score -= 30
        elif anomaly.severity == AnomalySeverity.MEDIUM:
            health_score -= 15
        elif anomaly.severity == AnomalySeverity.LOW:
            health_score -= 5

    health_score = max(0, health_score)

    # 判断状态
    if health_score >= 80:
        status = "healthy"
    elif health_score >= 50:
        status = "warning"
    else:
        status = "critical"

    # 收集建议
    suggestions = []
    for anomaly in anomalies:
        if anomaly.suggestion:
            suggestions.append(anomaly.suggestion)

    return {
        "job_id": job_id,
        "job_status": job.status.value,
        "health_score": health_score,
        "health_status": status,
        "anomalies_count": len(anomalies),
        "anomalies": [a.to_dict() for a in anomalies],
        "suggestions": suggestions,
    }
