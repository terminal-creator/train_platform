"""
Demo模式API路由

提供Demo模式的控制和数据API
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .config import get_demo_settings, set_demo_mode, is_demo_mode
from .mock_data import (
    get_all_demo_jobs, get_demo_job,
    get_demo_metrics, get_realtime_metrics, get_metrics_summary,
    DEMO_DATASETS, get_demo_dataset, get_demo_dataset_samples,
    get_all_demo_eval_datasets,
    get_demo_checkpoints, get_best_checkpoint,
    get_gradient_heatmap, get_gradient_stats, get_gradient_health_report,
    get_demo_evaluation_results, get_evaluation_comparison,
    get_all_merge_results, get_all_swa_results,
    get_all_checkpoint_selections, get_all_rm_prompt_configs,
    get_all_demo_pipelines, get_demo_pipeline, get_pipeline_templates,
)

router = APIRouter(prefix="/demo", tags=["Demo Mode"])


# ============ 配置模型 ============

class DemoModeConfig(BaseModel):
    enabled: bool
    speed: float = 1.0
    start_stage: int = 1


class DemoModeResponse(BaseModel):
    enabled: bool
    speed: float
    start_stage: int
    message: str


# ============ Demo模式控制 ============

@router.get("/status", response_model=DemoModeResponse)
async def get_demo_status():
    settings = get_demo_settings()
    return DemoModeResponse(
        enabled=settings.enabled,
        speed=settings.speed,
        start_stage=settings.start_stage,
        message="Demo mode is active" if settings.enabled else "Demo mode is inactive",
    )


@router.post("/enable", response_model=DemoModeResponse)
async def enable_demo_mode(config: DemoModeConfig = None):
    if config:
        set_demo_mode(enabled=True, speed=config.speed, start_stage=config.start_stage)
    else:
        set_demo_mode(enabled=True)
    settings = get_demo_settings()
    return DemoModeResponse(
        enabled=True,
        speed=settings.speed,
        start_stage=settings.start_stage,
        message="Demo mode enabled successfully",
    )


@router.post("/disable", response_model=DemoModeResponse)
async def disable_demo_mode():
    set_demo_mode(enabled=False)
    return DemoModeResponse(enabled=False, speed=1.0, start_stage=1, message="Demo mode disabled")


# ============ Demo数据API ============

@router.get("/jobs")
async def list_demo_jobs(status: Optional[str] = None, algorithm: Optional[str] = None):
    jobs = get_all_demo_jobs()
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    if algorithm:
        jobs = [j for j in jobs if j["algorithm"] == algorithm]
    return {"jobs": jobs, "total": len(jobs)}


@router.get("/jobs/{job_id}")
async def get_demo_job_detail(job_id: str):
    job = get_demo_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/jobs/{job_id}/metrics")
async def get_demo_job_metrics(job_id: str, start_step: int = 0, limit: int = 1000):
    metrics = get_demo_metrics(job_id, start_step, limit)
    summary = get_metrics_summary(job_id)
    return {"metrics": metrics, "total": len(metrics), "summary": summary}


@router.get("/jobs/{job_id}/checkpoints")
async def get_demo_job_checkpoints(job_id: str):
    checkpoints = get_demo_checkpoints(job_id)
    best = get_best_checkpoint(job_id)
    return {"checkpoints": checkpoints, "total": len(checkpoints), "best_checkpoint": best}


@router.get("/jobs/{job_id}/gradient-heatmap")
async def get_demo_gradient_heatmap(job_id: str):
    heatmap = get_gradient_heatmap(job_id)
    return {"heatmap": heatmap}


@router.get("/jobs/{job_id}/gradient-stats")
async def get_demo_gradient_stats(job_id: str):
    stats = get_gradient_stats(job_id)
    health = get_gradient_health_report(job_id)
    return {"stats": stats, "health_report": health}


@router.get("/jobs/{job_id}/evaluations")
async def get_demo_job_evaluations(job_id: str):
    results = get_demo_evaluation_results(job_id)
    return {"evaluations": results, "total": len(results)}


# ============ 数据集 ============

@router.get("/datasets")
async def list_demo_datasets():
    return {"datasets": DEMO_DATASETS, "total": len(DEMO_DATASETS)}


@router.get("/datasets/{dataset_id}")
async def get_demo_dataset_detail(dataset_id: str):
    dataset = get_demo_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.get("/datasets/{dataset_id}/samples")
async def get_demo_dataset_sample_preview(dataset_id: str, limit: int = 10):
    samples = get_demo_dataset_samples(dataset_id, limit)
    return {"samples": samples, "total": len(samples)}


@router.get("/eval-datasets")
async def list_demo_eval_datasets():
    datasets = get_all_demo_eval_datasets()
    return {"datasets": datasets, "total": len(datasets)}


# ============ 评估对比 ============

@router.get("/evaluation-comparison")
async def get_demo_evaluation_comparison():
    comparison = get_evaluation_comparison()
    return {"comparison": comparison}


# ============ 模型手术 ============

@router.get("/surgery/merges")
async def list_demo_merges():
    merges = get_all_merge_results()
    return {"merges": merges, "total": len(merges)}


@router.get("/surgery/swa")
async def list_demo_swa():
    swa_results = get_all_swa_results()
    return {"swa_results": swa_results, "total": len(swa_results)}


@router.get("/surgery/checkpoint-selections")
async def list_demo_checkpoint_selections():
    selections = get_all_checkpoint_selections()
    return {"selections": selections, "total": len(selections)}


@router.get("/rm-prompts")
async def list_demo_rm_prompts():
    prompts = get_all_rm_prompt_configs()
    return {"prompts": prompts, "total": len(prompts)}


# ============ 流水线 ============

@router.get("/pipelines")
async def list_demo_pipelines():
    pipelines = get_all_demo_pipelines()
    return {"pipelines": pipelines, "total": len(pipelines)}


@router.get("/pipelines/{pipeline_id}")
async def get_demo_pipeline_detail(pipeline_id: str):
    pipeline = get_demo_pipeline(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return pipeline


@router.get("/pipeline-templates")
async def list_demo_pipeline_templates():
    templates = get_pipeline_templates()
    return {"templates": templates, "total": len(templates)}


# ============ 仪表板汇总 ============

@router.get("/dashboard")
async def get_demo_dashboard():
    jobs = get_all_demo_jobs()
    pipelines = get_all_demo_pipelines()
    running_jobs = [j for j in jobs if j["status"] == "running"]
    completed_jobs = [j for j in jobs if j["status"] == "completed"]

    return {
        "summary": {
            "total_jobs": len(jobs),
            "running_jobs": len(running_jobs),
            "completed_jobs": len(completed_jobs),
            "total_pipelines": len(pipelines),
            "active_pipelines": len([p for p in pipelines if p["status"] == "running"]),
        },
        "recent_jobs": jobs[:5],
        "active_pipeline": next((p for p in pipelines if p["status"] == "running"), None),
        "gpu_utilization": {
            "total_gpus": 8,
            "used_gpus": 8,
            "avg_utilization": 92.5,
            "avg_memory_used": 74.2,
        },
        "training_progress": {
            "current_job": running_jobs[0] if running_jobs else None,
            "recent_metrics": get_realtime_metrics(
                running_jobs[0]["id"], running_jobs[0]["current_step"]
            ) if running_jobs else None,
        },
    }
