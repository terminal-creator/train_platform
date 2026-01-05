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
    """
    await websocket.accept()

    # Track connection
    if job_id not in _ws_connections:
        _ws_connections[job_id] = []
    _ws_connections[job_id].append(websocket)

    try:
        while True:
            # In real implementation, this would stream actual metrics
            import random

            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "step": random.randint(0, 10000),
                "policy_loss": random.uniform(0.1, 1.0),
                "reward_mean": random.uniform(-0.5, 0.5),
                "kl_divergence": random.uniform(0.01, 0.1),
                "throughput_tokens_per_sec": random.uniform(1000, 2000),
            }

            await websocket.send_json(metrics)
            await asyncio.sleep(1)  # Send every second

    except WebSocketDisconnect:
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
