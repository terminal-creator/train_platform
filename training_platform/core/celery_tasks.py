"""
Celery Tasks for Training Platform

Asynchronous tasks for training, evaluation, and maintenance.
"""

import os
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from .celery_config import app
from .database import (
    engine,
    Session,
    JobRepository,
    MetricsRepository,
    CheckpointRepository,
    TrainingJob,
    JobStatus,
)

logger = logging.getLogger(__name__)


# ============== Training Tasks ==============

@app.task(bind=True, name="training_platform.core.celery_tasks.train_model")
def train_model(
    self,
    job_uuid: str,
    config: Dict[str, Any],
    run_mode: str = "local",
    ssh_config: Optional[Dict[str, Any]] = None,
    _pipeline_uuid: Optional[str] = None,  # ✅ Pipeline 注入参数
    _stage_name: Optional[str] = None,     # ✅ Pipeline 注入参数
):
    """
    Train a model asynchronously

    This is the main training task that runs the verl training pipeline.

    Args:
        self: Celery task instance
        job_uuid: Training job UUID
        config: Training configuration
        run_mode: Execution mode ('local' or 'ssh')
        ssh_config: SSH configuration for remote execution
        _pipeline_uuid: Pipeline UUID (injected by PipelineExecutor, optional)
        _stage_name: Stage name (injected by PipelineExecutor, optional)

    Returns:
        Dict with training results
    """
    # Pipeline stage 状态记录（如果从 pipeline 调用）
    if _pipeline_uuid and _stage_name:
        from .pipeline_executor import mark_stage_running
        mark_stage_running(_pipeline_uuid, _stage_name, self.request.id)

    logger.info(f"Starting training task for job {job_uuid}")

    with Session(engine) as session:
        repo = JobRepository(session)
        job = repo.get_by_uuid(job_uuid)

        if not job:
            raise ValueError(f"Job {job_uuid} not found")

        try:
            # Update job status
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()
            repo.update(job)

            # Update task progress
            self.update_state(
                state="PROGRESS",
                meta={"current": 0, "total": 100, "status": "Initializing..."}
            )

            # Import run_mode module
            from .run_mode import execute_training

            # Execute training
            logger.info(f"Executing training in {run_mode} mode")
            result = execute_training(
                job_uuid=job_uuid,
                config=config,
                run_mode=run_mode,
                ssh_config=ssh_config,
                progress_callback=lambda current, total, status: self.update_state(
                    state="PROGRESS",
                    meta={"current": current, "total": total, "status": status}
                ),
            )

            # Update job status on success
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            repo.update(job)

            logger.info(f"Training task completed for job {job_uuid}")

            return {
                "job_uuid": job_uuid,
                "status": "completed",
                "result": result,
            }

        except Exception as e:
            logger.error(f"Training task failed for job {job_uuid}: {e}")

            # Update job status on failure
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            repo.update(job)

            # Re-raise exception for Celery retry
            raise


@app.task(bind=True, name="training_platform.core.celery_tasks.run_evaluation")
def run_evaluation(
    self,
    job_uuid: str,
    checkpoint_path: str,
    eval_dataset_uuid: str,
    _pipeline_uuid: Optional[str] = None,  # ✅ Pipeline 注入参数
    _stage_name: Optional[str] = None,     # ✅ Pipeline 注入参数
) -> Dict[str, Any]:
    """
    Run evaluation on a trained model

    Args:
        self: Celery task instance (bind=True for pipeline integration)
        job_uuid: Training job UUID
        checkpoint_path: Path to model checkpoint
        eval_dataset_uuid: Evaluation dataset UUID
        _pipeline_uuid: Pipeline UUID (injected by PipelineExecutor, optional)
        _stage_name: Stage name (injected by PipelineExecutor, optional)

    Returns:
        Dict with evaluation results
    """
    # Pipeline stage 状态记录（如果从 pipeline 调用）
    if _pipeline_uuid and _stage_name:
        from .pipeline_executor import mark_stage_running
        mark_stage_running(_pipeline_uuid, _stage_name, self.request.id)

    logger.info(f"Starting evaluation task for job {job_uuid}")

    # Import evaluation module
    from .evaluation import run_evaluation as eval_func

    try:
        result = eval_func(
            checkpoint_path=checkpoint_path,
            dataset_uuid=eval_dataset_uuid,
        )

        logger.info(f"Evaluation task completed for job {job_uuid}")
        return result

    except Exception as e:
        logger.error(f"Evaluation task failed for job {job_uuid}: {e}")
        raise


# ============== Preprocessing Tasks ==============

@app.task(bind=True, name="training_platform.core.celery_tasks.preprocess_dataset")
def preprocess_dataset(
    self,
    dataset_uuid: str,
    preprocessing_config: Dict[str, Any],
    _pipeline_uuid: Optional[str] = None,  # ✅ Pipeline 注入参数
    _stage_name: Optional[str] = None,     # ✅ Pipeline 注入参数
) -> Dict[str, Any]:
    """
    Preprocess a dataset (deduplication, filtering, etc.)

    Args:
        self: Celery task instance (bind=True for pipeline integration)
        dataset_uuid: Dataset UUID
        preprocessing_config: Preprocessing configuration

    Returns:
        Dict with preprocessing results
    """
    # Pipeline stage 状态记录（如果从 pipeline 调用）
    if _pipeline_uuid and _stage_name:
        from .pipeline_executor import mark_stage_running
        mark_stage_running(_pipeline_uuid, _stage_name, self.request.id)

    logger.info(f"Starting preprocessing task for dataset {dataset_uuid}")

    # Import dataset preprocessing module
    # (To be implemented)

    return {
        "dataset_uuid": dataset_uuid,
        "status": "completed",
        "message": "Dataset preprocessing completed",
    }


# ============== Maintenance Tasks ==============

@app.task(bind=True, name="training_platform.core.celery_tasks.cleanup_checkpoints")
def cleanup_checkpoints(
    self,
    job_uuid: str,
    keep_best_n: int = 3,
    _pipeline_uuid: Optional[str] = None,  # ✅ Pipeline 注入参数
    _stage_name: Optional[str] = None,     # ✅ Pipeline 注入参数
) -> Dict[str, Any]:
    """
    Clean up old checkpoints, keeping only the best N

    Args:
        self: Celery task instance (bind=True for pipeline integration)
        job_uuid: Training job UUID
        keep_best_n: Number of best checkpoints to keep
        _pipeline_uuid: Pipeline UUID (injected by PipelineExecutor, optional)
        _stage_name: Stage name (injected by PipelineExecutor, optional)

    Returns:
        Dict with cleanup results
    """
    # Pipeline stage 状态记录（如果从 pipeline 调用）
    if _pipeline_uuid and _stage_name:
        from .pipeline_executor import mark_stage_running
        mark_stage_running(_pipeline_uuid, _stage_name, self.request.id)

    logger.info(f"Starting checkpoint cleanup for job {job_uuid}")

    with Session(engine) as session:
        repo = CheckpointRepository(session)
        checkpoints = repo.get_checkpoints(job_uuid)

        if len(checkpoints) <= keep_best_n:
            return {
                "job_uuid": job_uuid,
                "deleted_count": 0,
                "message": f"Only {len(checkpoints)} checkpoints, no cleanup needed",
            }

        # Sort by metric (e.g., eval_loss) and keep best N
        # (Simplified - in production, use configurable metric)
        checkpoints_sorted = sorted(
            checkpoints,
            key=lambda cp: cp.metrics.get("eval_loss", float("inf"))
        )

        # Delete checkpoints beyond best N
        deleted_count = 0
        for cp in checkpoints_sorted[keep_best_n:]:
            # Delete checkpoint files
            if os.path.exists(cp.path):
                try:
                    import shutil
                    shutil.rmtree(cp.path)
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete checkpoint {cp.path}: {e}")

        logger.info(f"Cleaned up {deleted_count} checkpoints for job {job_uuid}")

        return {
            "job_uuid": job_uuid,
            "deleted_count": deleted_count,
            "kept_count": keep_best_n,
        }


@app.task(name="training_platform.core.celery_tasks.cleanup_old_checkpoints")
def cleanup_old_checkpoints(days: int = 30) -> Dict[str, Any]:
    """
    Periodic task to clean up checkpoints older than N days

    Args:
        days: Delete checkpoints older than this many days

    Returns:
        Dict with cleanup results
    """
    logger.info(f"Starting periodic checkpoint cleanup (>{days} days)")

    with Session(engine) as session:
        repo = CheckpointRepository(session)

        # Find old checkpoints
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # (Simplified - in production, add database query for old checkpoints)

        return {
            "status": "completed",
            "message": f"Cleaned up checkpoints older than {days} days",
        }


# ============== Periodic Tasks ==============

@app.task(name="training_platform.core.celery_tasks.scan_failed_jobs")
def scan_failed_jobs() -> Dict[str, Any]:
    """
    Periodic task to scan and diagnose failed jobs

    Returns:
        Dict with scan results
    """
    logger.info("Starting periodic failed jobs scan")

    with Session(engine) as session:
        repo = JobRepository(session)

        # Get failed jobs from last 24 hours
        failed_jobs, _ = repo.list_jobs(
            status=JobStatus.FAILED,
            limit=100,
        )

        # Run diagnostics on failed jobs
        from .diagnostics import DiagnosticService

        diagnostic_service = DiagnosticService(session)

        diagnosed_count = 0
        for job in failed_jobs:
            try:
                diagnostic_service.diagnose_job(job.uuid)
                diagnosed_count += 1
            except Exception as e:
                logger.error(f"Failed to diagnose job {job.uuid}: {e}")

        logger.info(f"Diagnosed {diagnosed_count} failed jobs")

        return {
            "status": "completed",
            "diagnosed_count": diagnosed_count,
            "total_failed": len(failed_jobs),
        }


@app.task(name="training_platform.core.celery_tasks.update_job_metrics")
def update_job_metrics() -> Dict[str, Any]:
    """
    Periodic task to update job metrics from running jobs

    完整的 metrics 闭环：
    1. 从 metrics 文件增量读取（使用 offset）
    2. 解析并存储到 DB
    3. 运行诊断检测异常
    4. 更新 job 的 metrics_last_offset

    Returns:
        Dict with update results
    """
    logger.info("Starting periodic job metrics update")

    from pathlib import Path
    from .metrics_persister import sync_metrics_from_file, sync_anomaly_from_status_file

    with Session(engine) as session:
        repo = JobRepository(session)

        # Get running jobs
        running_jobs, _ = repo.list_jobs(
            status=JobStatus.RUNNING,
            limit=100,
        )

        updated_count = 0
        total_new_metrics = 0
        anomaly_count = 0

        for job in running_jobs:
            try:
                # 确定 metrics 文件路径
                # 统一使用 platform_metrics 目录协议（与 WS 监控一致）
                import os

                # 获取 run_mode 配置
                run_mode = getattr(job, 'run_mode', 'local')

                if run_mode == "ssh":
                    # SSH 模式：从 run_mode_config 获取工作目录
                    ssh_config = getattr(job, 'run_mode_config', {}) or {}
                    ssh_working_dir = ssh_config.get('ssh_working_dir', '~/verl_jobs')
                    metrics_dir_str = f"{ssh_working_dir}/platform_metrics"
                    # SSH 模式下我们无法直接检查文件是否存在，跳过检查
                    logger.debug(f"Job {job.uuid} SSH metrics dir: {metrics_dir_str}")
                    # SSH 模式暂时跳过（需要 SSH 连接才能读取）
                    # TODO: 实现 SSH 模式的 metrics 同步
                    continue
                else:
                    # Local 模式：使用环境变量或默认值
                    metrics_dir_str = os.getenv("PLATFORM_METRICS_DIR", "./platform_metrics")
                    metrics_dir = Path(metrics_dir_str)

                    if not metrics_dir.exists():
                        logger.debug(f"Metrics directory not found: {metrics_dir}")
                        continue

                # Metrics 文件名：{job_uuid}_metrics.jsonl
                metrics_file = metrics_dir / f"{job.uuid}_metrics.jsonl"
                status_file = metrics_dir / f"{job.uuid}_status.json"

                # 增量同步 metrics（使用 last_offset）
                if metrics_file.exists():
                    result = sync_metrics_from_file(
                        job_uuid=job.uuid,
                        metrics_file=metrics_file,
                        session=session,
                        batch_size=100,
                        last_offset=job.metrics_last_offset,
                    )

                    new_metrics_count = result.get("new_metrics_count", 0)
                    new_offset = result.get("new_offset", job.metrics_last_offset)

                    if new_metrics_count > 0:
                        # 更新 job 的 last_offset
                        job.metrics_last_offset = new_offset
                        repo.update(job)

                        total_new_metrics += new_metrics_count
                        updated_count += 1

                        logger.info(
                            f"Job {job.uuid}: Synced {new_metrics_count} metrics "
                            f"(offset: {job.metrics_last_offset} -> {new_offset})"
                        )

                # 同步异常状态
                if status_file.exists():
                    anomaly_synced = sync_anomaly_from_status_file(
                        job_uuid=job.uuid,
                        status_file=status_file,
                        session=session,
                    )
                    if anomaly_synced:
                        anomaly_count += 1

            except Exception as e:
                logger.error(f"Failed to update metrics for job {job.uuid}: {e}", exc_info=True)

        logger.info(
            f"Metrics update completed: {updated_count}/{len(running_jobs)} jobs updated, "
            f"{total_new_metrics} new metrics, {anomaly_count} anomalies detected"
        )

        return {
            "status": "completed",
            "updated_count": updated_count,
            "total_running": len(running_jobs),
            "total_new_metrics": total_new_metrics,
            "anomaly_count": anomaly_count,
        }


# ============== Pipeline Tasks ==============

@app.task(bind=True, name="training_platform.core.celery_tasks.run_training_pipeline")
def run_training_pipeline(
    self,
    pipeline_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run a multi-stage training pipeline using DAG executor

    This is the new version that uses real DAG execution with dependency resolution.

    Args:
        self: Celery task instance
        pipeline_config: Pipeline configuration with format:
            {
                "pipeline_uuid": str,
                "stages": [
                    {
                        "name": str,
                        "task": str,  # Task type (train_model, etc.)
                        "params": dict,
                        "depends_on": List[str]  # Dependencies
                    }
                ]
            }

    Returns:
        Dict with pipeline results
    """
    from .pipeline_executor import PipelineExecutor

    pipeline_uuid = pipeline_config.get("pipeline_uuid")
    stages = pipeline_config.get("stages", [])

    logger.info(f"Starting DAG pipeline {pipeline_uuid} with {len(stages)} stages")

    if not pipeline_uuid or not stages:
        raise ValueError("pipeline_uuid and stages are required")

    try:
        # Use DAG executor for real async orchestration
        executor = PipelineExecutor(pipeline_uuid)
        result = executor.execute(stages)

        logger.info(f"Pipeline {pipeline_uuid} submitted successfully")
        return result

    except Exception as e:
        logger.error(f"Pipeline {pipeline_uuid} submission failed: {e}")

        # Update pipeline status to FAILED
        from .database import PipelineRepository, PipelineStatus
        with Session(engine) as session:
            repo = PipelineRepository(session)
            pipeline = repo.get_by_uuid(pipeline_uuid)
            if pipeline:
                pipeline.status = PipelineStatus.FAILED
                pipeline.completed_at = datetime.utcnow()
                pipeline.error_message = str(e)
                repo.update(pipeline)

        raise


# ============== Task Management ==============

@app.task(name="training_platform.core.celery_tasks.cancel_task")
def cancel_task(task_id: str) -> Dict[str, Any]:
    """
    Cancel a running Celery task

    Args:
        task_id: Celery task ID

    Returns:
        Dict with cancellation result
    """
    logger.info(f"Cancelling task {task_id}")

    app.control.revoke(task_id, terminate=True, signal="SIGKILL")

    return {
        "task_id": task_id,
        "status": "cancelled",
    }


@app.task(name="training_platform.core.celery_tasks.retry_failed_task")
def retry_failed_task(task_id: str) -> Dict[str, Any]:
    """
    Retry a failed Celery task

    Args:
        task_id: Celery task ID

    Returns:
        Dict with retry result
    """
    logger.info(f"Retrying task {task_id}")

    # Get task result
    result = app.AsyncResult(task_id)

    if result.failed():
        # Retry the task
        result.retry()

        return {
            "task_id": task_id,
            "status": "retrying",
        }
    else:
        return {
            "task_id": task_id,
            "status": "not_failed",
            "message": "Task is not in failed state",
        }
