"""
Training Jobs API Router

Provides CRUD operations for training jobs with:
- Persistent storage via SQLModel
- Ray Job Submission for execution
- Real-time updates via WebSocket
"""

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import uuid
import asyncio
import logging
import os

from sqlmodel import Session

from ..models.training import (
    TrainingJobCreate,
    TrainingJobUpdate,
    TrainingJobResponse,
    TrainingJobListResponse,
    TrainingJobLogsResponse,
    TrainingJobMetricsHistoryResponse,
    TrainingMetrics,
    JobStatus,
    TrainingAlgorithm,
)
from ...core.database import (
    get_session,
    init_db,
    TrainingJob as DBTrainingJob,
    TrainingMetric as DBTrainingMetric,
    TrainingLog as DBTrainingLog,
    Checkpoint as DBCheckpoint,
    JobRepository,
    MetricsRepository,
    LogRepository,
    CheckpointRepository,
    JobStatus as DBJobStatus,
    TrainingAlgorithm as DBTrainingAlgorithm,
)
from ...core.ray_runner import (
    get_default_runner,
    RayJobConfig,
)
from ...core.verl_adapter import (
    VerlTrainingConfig,
    VerlAlgorithm,
    create_ray_entrypoint,
)
from ...core.run_mode import (
    RunMode,
    get_current_runner,
    load_run_mode_config,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["Training Jobs"])

# Default directories for models and datasets
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "./models"))
DATASETS_DIR = Path(os.environ.get("DATASETS_DIR", "./datasets"))
REWARD_SCRIPTS_DIR = Path(os.environ.get("REWARD_SCRIPTS_DIR", "./reward_scripts"))


@router.get("/available-reward-scripts", response_model=List[Dict[str, Any]])
async def list_available_reward_scripts() -> List[Dict[str, Any]]:
    """
    List available reward scripts in the reward_scripts directory.

    Each script should implement the standard interface:
    - Input: JSON via stdin with prompts, responses, metadata
    - Output: JSON via stdout with rewards, details
    """
    scripts = []

    if not REWARD_SCRIPTS_DIR.exists():
        return scripts

    for item in REWARD_SCRIPTS_DIR.rglob("*.py"):
        if item.name.startswith('_') or item.name.startswith('.'):
            continue

        # Read first few lines to get description
        description = ""
        script_type = "rule"  # Default type
        try:
            with open(item, "r", encoding="utf-8") as f:
                content = f.read(2000)  # Read first 2000 chars

            # Extract docstring
            import re
            docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
            if docstring_match:
                description = docstring_match.group(1).strip().split('\n')[0]

            # Detect type from filename or content
            if "api" in item.name.lower() or "openai" in content.lower() or "anthropic" in content.lower():
                script_type = "api"
            elif "model" in item.name.lower() or "AutoModelForSequenceClassification" in content:
                script_type = "model"
            else:
                script_type = "rule"

        except Exception:
            pass

        # Get relative path from reward_scripts dir
        rel_path = item.relative_to(REWARD_SCRIPTS_DIR)

        scripts.append({
            "name": item.stem,
            "path": str(rel_path),
            "full_path": str(item.absolute()),
            "type": script_type,
            "description": description,
        })

    # Sort by name
    scripts.sort(key=lambda x: x["name"])
    return scripts


@router.get("/available-models", response_model=List[Dict[str, Any]])
async def list_available_models() -> List[Dict[str, Any]]:
    """
    List available models in the models directory.

    Scans the configured models directory for model folders.
    A valid model folder should contain config.json or pytorch_model files.
    """
    models = []

    if not MODELS_DIR.exists():
        return models

    for item in MODELS_DIR.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it looks like a valid model directory
            has_config = (item / "config.json").exists()
            has_model = any(
                f.name.startswith("pytorch_model") or
                f.name.startswith("model") or
                f.name.endswith(".safetensors")
                for f in item.iterdir() if f.is_file()
            )

            if has_config or has_model:
                # Get model size if possible
                size_bytes = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                size_gb = size_bytes / (1024 ** 3)

                models.append({
                    "name": item.name,
                    "path": str(item.absolute()),
                    "size_gb": round(size_gb, 2),
                    "has_config": has_config,
                })

    # Sort by name
    models.sort(key=lambda x: x["name"])
    return models


@router.get("/available-datasets", response_model=List[Dict[str, Any]])
async def list_available_datasets() -> List[Dict[str, Any]]:
    """
    List available datasets in the datasets directory.

    Scans for parquet and jsonl files which are common training data formats.
    """
    datasets = []

    if not DATASETS_DIR.exists():
        return datasets

    # Supported extensions
    extensions = {".parquet", ".jsonl", ".json", ".csv"}

    for item in DATASETS_DIR.rglob("*"):
        if item.is_file() and item.suffix.lower() in extensions and not item.name.startswith('.'):
            size_bytes = item.stat().st_size
            size_mb = size_bytes / (1024 ** 2)

            # Get relative path from datasets dir
            rel_path = item.relative_to(DATASETS_DIR)

            datasets.append({
                "name": str(rel_path),
                "path": str(item.absolute()),
                "format": item.suffix.lower().lstrip('.'),
                "size_mb": round(size_mb, 2),
            })

    # Sort by name
    datasets.sort(key=lambda x: x["name"])
    return datasets


@router.post("/dataset-preview", response_model=Dict[str, Any])
async def get_dataset_preview(request: Dict[str, str]) -> Dict[str, Any]:
    """
    Get preview and statistics for a dataset file.

    Supports parquet, jsonl, json, and csv formats.
    """
    file_path = Path(request.get("path", ""))

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {file_path}")

    # Security check: ensure path is within allowed directories
    try:
        file_path.resolve().relative_to(DATASETS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: path outside datasets directory")

    suffix = file_path.suffix.lower()
    preview_rows = 20
    result = {
        "preview": {"columns": [], "rows": []},
        "stats": {"total_rows": 0, "num_columns": 0, "columns": []}
    }

    try:
        if suffix == ".parquet":
            import pandas as pd
            df = pd.read_parquet(file_path)
            total_rows = len(df)
            result["preview"]["columns"] = list(df.columns)
            result["preview"]["rows"] = df.head(preview_rows).to_dict(orient="records")
            result["stats"]["total_rows"] = total_rows
            result["stats"]["num_columns"] = len(df.columns)
            result["stats"]["columns"] = [
                {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "non_null": int(df[col].notna().sum()),
                    "unique": int(df[col].nunique()),
                    "avg_length": round(df[col].astype(str).str.len().mean(), 1) if df[col].dtype == object else None
                }
                for col in df.columns
            ]

        elif suffix == ".jsonl":
            import json
            rows = []
            with open(file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if line.strip():
                        rows.append(json.loads(line))
                        if i >= 1000:  # Read max 1000 rows for stats
                            break

            total_rows = sum(1 for _ in open(file_path, "r", encoding="utf-8") if _.strip())
            if rows:
                result["preview"]["columns"] = list(rows[0].keys())
                result["preview"]["rows"] = rows[:preview_rows]
                result["stats"]["total_rows"] = total_rows
                result["stats"]["num_columns"] = len(rows[0].keys())

                # Calculate column stats from sample
                columns_stats = []
                for col in rows[0].keys():
                    values = [r.get(col) for r in rows if col in r]
                    non_null = sum(1 for v in values if v is not None)
                    unique = len(set(str(v) for v in values if v is not None))
                    str_values = [str(v) for v in values if v is not None]
                    avg_len = round(sum(len(s) for s in str_values) / len(str_values), 1) if str_values else 0
                    columns_stats.append({
                        "name": col,
                        "dtype": "object",
                        "non_null": non_null,
                        "unique": unique,
                        "avg_length": avg_len
                    })
                result["stats"]["columns"] = columns_stats

        elif suffix == ".json":
            import json
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list) and data:
                result["preview"]["columns"] = list(data[0].keys()) if isinstance(data[0], dict) else []
                result["preview"]["rows"] = data[:preview_rows]
                result["stats"]["total_rows"] = len(data)
                result["stats"]["num_columns"] = len(data[0].keys()) if isinstance(data[0], dict) else 0

        elif suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(file_path)
            total_rows = len(df)
            result["preview"]["columns"] = list(df.columns)
            result["preview"]["rows"] = df.head(preview_rows).to_dict(orient="records")
            result["stats"]["total_rows"] = total_rows
            result["stats"]["num_columns"] = len(df.columns)
            result["stats"]["columns"] = [
                {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "non_null": int(df[col].notna().sum()),
                    "unique": int(df[col].nunique()),
                    "avg_length": round(df[col].astype(str).str.len().mean(), 1) if df[col].dtype == object else None
                }
                for col in df.columns
            ]

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {suffix}")

    except Exception as e:
        logger.error(f"Error reading dataset {file_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")

    return result


def _db_job_to_response(job: DBTrainingJob, latest_metrics: DBTrainingMetric = None) -> TrainingJobResponse:
    """Convert database job to API response"""
    metrics = None
    if latest_metrics:
        metrics = TrainingMetrics(
            step=latest_metrics.step,
            timestamp=latest_metrics.timestamp,
            epoch=latest_metrics.epoch,
            policy_loss=latest_metrics.policy_loss,
            value_loss=latest_metrics.value_loss,
            total_loss=latest_metrics.total_loss,
            reward_mean=latest_metrics.reward_mean,
            reward_std=latest_metrics.reward_std,
            kl_divergence=latest_metrics.kl_divergence,
            entropy=latest_metrics.entropy,
            learning_rate=latest_metrics.learning_rate,
        )

    return TrainingJobResponse(
        id=job.uuid,
        name=job.name,
        description=job.description,
        status=JobStatus(job.status.value),
        algorithm=TrainingAlgorithm(job.algorithm.value),
        model_path=job.model_path,
        lora_enabled=job.lora_enabled,
        current_step=job.current_step,
        total_steps=job.total_steps,
        current_epoch=job.current_epoch,
        total_epochs=job.num_epochs,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        num_gpus=job.num_gpus,
        gpu_type=job.gpu_type,
        latest_metrics=metrics,
        output_path=job.output_path,
        checkpoint_paths=job.checkpoint_paths or [],
    )


@router.post("", response_model=TrainingJobResponse)
async def create_job(
    request: TrainingJobCreate,
    session: Session = Depends(get_session),
) -> TrainingJobResponse:
    """
    Create a new training job.

    The job will be stored in the database and queued for execution.
    """
    job_uuid = str(uuid.uuid4())[:8]

    # Create database record
    db_job = DBTrainingJob(
        uuid=job_uuid,
        name=request.name,
        description=request.description,
        status=DBJobStatus.PENDING,
        algorithm=DBTrainingAlgorithm(request.algorithm.value),
        model_path=request.model_path,
        lora_enabled=request.lora_enabled,
        lora_rank=request.lora_rank,
        train_data_path=request.train_data_path,
        eval_data_path=request.eval_data_path,
        num_gpus=request.num_gpus,
        gpu_type=request.gpu_type,
        learning_rate=request.learning_rate,
        batch_size=request.batch_size,
        num_epochs=request.num_epochs,
        max_steps=request.max_steps,
        total_steps=request.total_steps,
        context_length=request.context_length,
        kl_coef=request.kl_coef,
        rollout_n=request.rollout_n,
        config={
            "checkpoint_interval": request.checkpoint_interval,
            "eval_interval": request.eval_interval,
            "eval_benchmarks": request.eval_benchmarks,
            "config_overrides": request.config_overrides,
            "resume_from_checkpoint": request.resume_from_checkpoint,
            # GRPO Reward Function Config
            "reward_fn_type": request.reward_fn_type,
            "reward_fn_extract_answer": request.reward_fn_extract_answer,
            "reward_fn_compare_method": request.reward_fn_compare_method,
            "reward_fn_answer_key": request.reward_fn_answer_key,
            "reward_fn_custom_path": request.reward_fn_custom_path,
            # PPO Reward Model Config
            "reward_model_path": request.reward_model_path,
            "reward_model_enable_gc": request.reward_model_enable_gc,
            "reward_model_offload": request.reward_model_offload,
            "reward_model_micro_batch": request.reward_model_micro_batch,
            # Unified Reward Script Config
            "reward_script_path": request.reward_script_path,
            "reward_script_type": request.reward_script_type,
            "reward_script_metadata": request.reward_script_metadata,
        },
    )

    repo = JobRepository(session)
    db_job = repo.create(db_job)

    logger.info(f"Created job {job_uuid}: {request.name}")

    return _db_job_to_response(db_job)


@router.get("", response_model=TrainingJobListResponse)
async def list_jobs(
    status: Optional[JobStatus] = Query(None, description="Filter by status"),
    algorithm: Optional[TrainingAlgorithm] = Query(None, description="Filter by algorithm"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    session: Session = Depends(get_session),
) -> TrainingJobListResponse:
    """
    List all training jobs with optional filtering.
    """
    repo = JobRepository(session)
    metrics_repo = MetricsRepository(session)

    db_status = DBJobStatus(status.value) if status else None
    db_algorithm = DBTrainingAlgorithm(algorithm.value) if algorithm else None

    offset = (page - 1) * page_size
    jobs, total = repo.list_jobs(
        status=db_status,
        algorithm=db_algorithm,
        offset=offset,
        limit=page_size,
    )

    # Get latest metrics for each job
    responses = []
    for job in jobs:
        latest_metrics = metrics_repo.get_latest_metric(job.uuid)
        responses.append(_db_job_to_response(job, latest_metrics))

    return TrainingJobListResponse(
        jobs=responses,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{job_id}", response_model=TrainingJobResponse)
async def get_job(
    job_id: str,
    session: Session = Depends(get_session),
) -> TrainingJobResponse:
    """
    Get details of a specific training job.
    """
    repo = JobRepository(session)
    metrics_repo = MetricsRepository(session)

    job = repo.get_by_uuid(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    latest_metrics = metrics_repo.get_latest_metric(job_id)
    return _db_job_to_response(job, latest_metrics)


@router.patch("/{job_id}", response_model=TrainingJobResponse)
async def update_job(
    job_id: str,
    request: TrainingJobUpdate,
    session: Session = Depends(get_session),
) -> TrainingJobResponse:
    """
    Update a training job.

    For pending jobs: all fields can be updated.
    For running/completed jobs: only name and description can be updated.
    """
    repo = JobRepository(session)
    job = repo.get_by_uuid(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Name and description can always be updated
    if request.name is not None:
        job.name = request.name
    if request.description is not None:
        job.description = request.description
    if request.status is not None:
        job.status = DBJobStatus(request.status.value)

    # Other fields can only be updated for pending jobs
    if job.status == DBJobStatus.PENDING:
        if request.model_path is not None:
            job.model_path = request.model_path
        if request.algorithm is not None:
            job.algorithm = DBTrainingAlgorithm(request.algorithm.value)
        if request.lora_enabled is not None:
            job.lora_enabled = request.lora_enabled
        if request.lora_rank is not None:
            job.lora_rank = request.lora_rank
        if request.train_data_path is not None:
            job.train_data_path = request.train_data_path
        if request.eval_data_path is not None:
            job.eval_data_path = request.eval_data_path
        if request.num_gpus is not None:
            job.num_gpus = request.num_gpus
        if request.gpu_type is not None:
            job.gpu_type = request.gpu_type
        if request.learning_rate is not None:
            job.learning_rate = request.learning_rate
        if request.batch_size is not None:
            job.batch_size = request.batch_size
        if request.num_epochs is not None:
            job.num_epochs = request.num_epochs
        if request.max_steps is not None:
            job.max_steps = request.max_steps
        if request.context_length is not None:
            job.context_length = request.context_length
        if request.kl_coef is not None:
            job.kl_coef = request.kl_coef
        if request.rollout_n is not None:
            job.rollout_n = request.rollout_n

    job = repo.update(job)
    return _db_job_to_response(job)


@router.delete("/{job_id}")
async def delete_job(
    job_id: str,
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    Delete a training job.

    Note: This will not stop a running job. Use the stop endpoint first.
    """
    repo = JobRepository(session)

    if not repo.delete(job_id):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return {"message": f"Job {job_id} deleted"}


def _create_verl_config_from_job(job: DBTrainingJob) -> VerlTrainingConfig:
    """
    Create VerlTrainingConfig from database job record.

    Maps all database fields to verl configuration parameters.
    """
    # Map database algorithm to VerlAlgorithm
    algorithm_map = {
        "sft": VerlAlgorithm.SFT,
        "ppo": VerlAlgorithm.PPO,
        "grpo": VerlAlgorithm.GRPO,
        "dpo": VerlAlgorithm.DPO,
        "gspo": VerlAlgorithm.GSPO,
        "dapo": VerlAlgorithm.DAPO,
        "remax": VerlAlgorithm.REMAX,
        "rloo": VerlAlgorithm.RLOO,
    }
    algorithm = algorithm_map.get(job.algorithm.value.lower(), VerlAlgorithm.GRPO)

    # Extract config overrides from job.config
    config = job.config or {}
    checkpoint_interval = config.get("checkpoint_interval", 500)
    eval_interval = config.get("eval_interval", 100)

    # Extract reward configuration
    reward_fn_type = config.get("reward_fn_type", "math_verify")
    reward_fn_extract_answer = config.get("reward_fn_extract_answer", "boxed")
    reward_fn_compare_method = config.get("reward_fn_compare_method", "exact")
    reward_fn_answer_key = config.get("reward_fn_answer_key", "solution")
    reward_fn_custom_path = config.get("reward_fn_custom_path")
    reward_model_path = config.get("reward_model_path")
    reward_model_enable_gc = config.get("reward_model_enable_gc", True)
    reward_model_offload = config.get("reward_model_offload", False)
    reward_model_micro_batch = config.get("reward_model_micro_batch", 4)

    # Extract unified reward script configuration
    reward_script_path = config.get("reward_script_path")
    reward_script_type = config.get("reward_script_type", "rule")
    reward_script_metadata = config.get("reward_script_metadata", {})

    return VerlTrainingConfig(
        # Model
        model_path=job.model_path,

        # Algorithm
        algorithm=algorithm,

        # Data
        train_data_path=job.train_data_path or "",
        eval_data_path=job.eval_data_path,

        # Training params
        num_epochs=job.num_epochs or 3,
        max_steps=job.max_steps,
        learning_rate=job.learning_rate or 1e-6,
        batch_size=job.batch_size or 256,
        max_prompt_length=512,
        max_response_length=job.context_length or 1024,

        # RL specific
        kl_coef=job.kl_coef or 0.001,
        rollout_n=job.rollout_n or 5,

        # LoRA
        lora_enabled=job.lora_enabled or False,
        lora_rank=job.lora_rank or 8,

        # Resources
        num_gpus=job.num_gpus or 8,
        gpu_type=job.gpu_type or "A100-80G",

        # Checkpointing
        checkpoint_interval=checkpoint_interval,
        eval_interval=eval_interval,
        output_dir=f"./outputs/{job.uuid}",
        project_name="training_platform",
        experiment_name=job.name,

        # Resume from checkpoint
        resume_from_checkpoint=config.get("resume_from_checkpoint"),
        resume_mode="resume_path" if config.get("resume_from_checkpoint") else "auto",

        # GRPO Reward Function Config
        reward_fn_type=reward_fn_type,
        reward_fn_extract_answer=reward_fn_extract_answer,
        reward_fn_compare_method=reward_fn_compare_method,
        reward_fn_answer_key=reward_fn_answer_key,
        reward_fn_custom_path=reward_fn_custom_path,

        # PPO Reward Model Config
        reward_model_path=reward_model_path,
        reward_model_enable_gc=reward_model_enable_gc,
        reward_model_offload=reward_model_offload,
        reward_model_micro_batch=reward_model_micro_batch,

        # Unified Reward Script Config
        reward_script_path=reward_script_path,
        reward_script_type=reward_script_type,
        reward_script_metadata=reward_script_metadata,
    )


async def _run_training_job(job_id: str, session: Session):
    """Background task to run training via configured runner (Local or SSH)"""
    from .websocket import push_status_update, push_metrics_update

    repo = JobRepository(session)
    job = repo.get_by_uuid(job_id)

    if not job:
        logger.error(f"Job {job_id} not found for execution")
        return

    try:
        # Create VerlTrainingConfig from database job
        verl_config = _create_verl_config_from_job(job)

        # Get the current run mode configuration
        run_mode_config = load_run_mode_config()
        runner = get_current_runner()

        # Log the generated command for debugging
        entrypoint = create_ray_entrypoint(verl_config)
        logger.info(f"Job {job_id} entrypoint command:\n{entrypoint}")
        logger.info(f"Job {job_id} run mode: {run_mode_config.mode.value}")

        # Submit job using unified runner
        result = runner.submit_job(
            job_id=job.uuid,
            name=job.name,
            verl_config=verl_config,
            num_gpus=job.num_gpus,
        )

        if result.get("success"):
            # Store job reference (ray_job_id for local, pid for SSH)
            job.ray_job_id = result.get("ray_job_id") or result.get("pid")
            job.status = DBJobStatus.RUNNING
            job.started_at = datetime.utcnow()
            job.output_path = result.get("log_file") or f"./outputs/{job.uuid}"

            # Store run mode in config for later reference
            config = job.config or {}
            config["run_mode"] = run_mode_config.mode.value
            job.config = config

            repo.update(job)

            # Push WebSocket update
            mode_str = run_mode_config.mode.value
            await push_status_update(job_id, "running", f"Training started ({mode_str} mode)")

            logger.info(f"Job {job_id} submitted via {mode_str}: {job.ray_job_id}")

            # For SSH mode, start log streaming if needed
            if run_mode_config.mode == RunMode.SSH:
                _start_ssh_log_streaming(job_id, runner)

        else:
            job.status = DBJobStatus.FAILED
            repo.update(job)

            await push_status_update(job_id, "failed", result.get("error"))

            logger.error(f"Job {job_id} failed to submit: {result.get('error')}")

    except Exception as e:
        logger.error(f"Error running job {job_id}: {e}")
        job.status = DBJobStatus.FAILED
        repo.update(job)


def _start_ssh_log_streaming(job_id: str, runner):
    """Start SSH log streaming and push to WebSocket"""
    from .websocket import push_log_entry
    import asyncio

    def log_callback(log_line: str):
        # Push log to WebSocket (run in event loop)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(push_log_entry(job_id, log_line))
        except Exception as e:
            logger.debug(f"Failed to push log: {e}")

    runner.start_log_streaming(job_id, log_callback)


@router.post("/{job_id}/start")
async def start_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    Start a pending training job.
    """
    repo = JobRepository(session)
    job = repo.get_by_uuid(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status != DBJobStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not in PENDING status (current: {job.status.value})"
        )

    job.status = DBJobStatus.QUEUED
    repo.update(job)

    # Queue background task
    background_tasks.add_task(_run_training_job, job_id, session)

    return {"message": f"Job {job_id} queued for execution"}


@router.post("/{job_id}/stop")
async def stop_job(
    job_id: str,
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    Stop a running or pending training job.
    """
    repo = JobRepository(session)
    job = repo.get_by_uuid(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status not in [DBJobStatus.RUNNING, DBJobStatus.QUEUED, DBJobStatus.PENDING]:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} cannot be stopped (current: {job.status.value})"
        )

    # Determine run mode from job config
    config = job.config or {}
    job_run_mode = config.get("run_mode", "local")

    # Stop job using appropriate runner
    if job.ray_job_id:
        runner = get_current_runner()
        runner.stop_job(job_id, job.ray_job_id)

    job.status = DBJobStatus.CANCELLED
    job.completed_at = datetime.utcnow()
    repo.update(job)

    return {"message": f"Job {job_id} stopped"}


@router.post("/{job_id}/pause")
async def pause_job(
    job_id: str,
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    Pause a running training job.

    Note: Pause functionality depends on Ray/verl support.
    """
    repo = JobRepository(session)
    job = repo.get_by_uuid(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status != DBJobStatus.RUNNING:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not running (current: {job.status.value})"
        )

    job.status = DBJobStatus.PAUSED
    repo.update(job)

    return {"message": f"Job {job_id} paused"}


class ResumeRequest(BaseModel):
    """Request to resume a job from checkpoint"""
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Path to checkpoint (if None, will use latest checkpoint)"
    )
    checkpoint_step: Optional[int] = Field(
        default=None,
        description="Step number to resume from (alternative to checkpoint_path)"
    )


@router.post("/{job_id}/resume")
async def resume_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    request: Optional[ResumeRequest] = None,
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    Resume a paused or failed training job from checkpoint.

    If no checkpoint is specified, the system will automatically find the latest
    checkpoint in the job's output directory.

    The job will be resubmitted to Ray with resume_from_checkpoint set.
    """
    from sqlmodel import select
    from ...core.database import Checkpoint

    repo = JobRepository(session)
    job = repo.get_by_uuid(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Allow resuming paused, failed, or cancelled jobs
    if job.status not in [DBJobStatus.PAUSED, DBJobStatus.FAILED, DBJobStatus.CANCELLED]:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} cannot be resumed (current: {job.status.value}). Must be paused, failed, or cancelled."
        )

    # Determine checkpoint path
    checkpoint_path = None

    if request and request.checkpoint_path:
        # Use explicitly provided path
        checkpoint_path = request.checkpoint_path
    elif request and request.checkpoint_step:
        # Find checkpoint by step
        checkpoint_path = f"./outputs/{job_id}/global_step_{request.checkpoint_step}"
    else:
        # Find latest checkpoint from database
        statement = select(Checkpoint).where(
            Checkpoint.job_uuid == job_id
        ).order_by(Checkpoint.step.desc()).limit(1)
        latest_checkpoint = session.exec(statement).first()

        if latest_checkpoint:
            checkpoint_path = latest_checkpoint.path
        else:
            # Try to find from filesystem
            import os
            output_dir = f"./outputs/{job_id}"
            if os.path.exists(output_dir):
                # Look for global_step_* directories
                dirs = [d for d in os.listdir(output_dir) if d.startswith("global_step_")]
                if dirs:
                    # Get the latest by step number
                    latest = max(dirs, key=lambda x: int(x.split("_")[-1]))
                    checkpoint_path = os.path.join(output_dir, latest)

    # Update job config with resume path
    config = job.config or {}
    if checkpoint_path:
        config["resume_from_checkpoint"] = checkpoint_path
    job.config = config

    # Reset job status to queued
    job.status = DBJobStatus.QUEUED
    repo.update(job)

    # Queue background task to restart training
    background_tasks.add_task(_run_training_job, job_id, session)

    return {
        "message": f"Job {job_id} queued for resume",
        "resume_from_checkpoint": checkpoint_path,
    }


@router.get("/{job_id}/available-checkpoints")
async def get_available_checkpoints(
    job_id: str,
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    List available checkpoints for resuming a job.

    Returns both database-tracked checkpoints and filesystem checkpoints.
    """
    from sqlmodel import select
    from ...core.database import Checkpoint
    import os

    repo = JobRepository(session)
    job = repo.get_by_uuid(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Get checkpoints from database
    statement = select(Checkpoint).where(
        Checkpoint.job_uuid == job_id
    ).order_by(Checkpoint.step.desc())
    db_checkpoints = session.exec(statement).all()

    checkpoints = []
    for cp in db_checkpoints:
        checkpoints.append({
            "step": cp.step,
            "path": cp.path,
            "metrics": cp.metrics,
            "source": "database",
            "exists": os.path.exists(cp.path) if cp.path else False,
        })

    # Also scan filesystem for any untracked checkpoints
    output_dir = f"./outputs/{job_id}"
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            if item.startswith("global_step_"):
                step = int(item.split("_")[-1])
                path = os.path.join(output_dir, item)

                # Check if already in database
                if not any(cp["step"] == step for cp in checkpoints):
                    checkpoints.append({
                        "step": step,
                        "path": path,
                        "metrics": None,
                        "source": "filesystem",
                        "exists": True,
                    })

    # Sort by step descending
    checkpoints.sort(key=lambda x: x["step"], reverse=True)

    return {
        "job_id": job_id,
        "checkpoints": checkpoints,
        "total": len(checkpoints),
        "recommended": checkpoints[0] if checkpoints else None,
    }


@router.get("/{job_id}/logs", response_model=TrainingJobLogsResponse)
async def get_job_logs(
    job_id: str,
    lines: int = Query(100, ge=1, le=1000, description="Number of log lines"),
    offset: int = Query(0, ge=0, description="Offset from end"),
    session: Session = Depends(get_session),
) -> TrainingJobLogsResponse:
    """
    Get training logs for a job.
    """
    repo = JobRepository(session)
    log_repo = LogRepository(session)

    job = repo.get_by_uuid(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Try to get logs from runner if job is running
    logs = []
    if job.ray_job_id and job.status == DBJobStatus.RUNNING:
        runner = get_current_runner()
        runner_logs = runner.get_job_logs(job_id, job.ray_job_id, lines=lines)
        logs = runner_logs.split('\n') if runner_logs else []
    else:
        # Get from database
        db_logs, total = log_repo.get_logs(job_id, offset=offset, limit=lines)
        logs = [log.message for log in db_logs]

    return TrainingJobLogsResponse(
        job_id=job_id,
        logs=logs[-lines:],
        has_more=len(logs) > lines,
    )


@router.get("/{job_id}/metrics", response_model=TrainingJobMetricsHistoryResponse)
async def get_job_metrics(
    job_id: str,
    start_step: Optional[int] = Query(None, description="Start step"),
    end_step: Optional[int] = Query(None, description="End step"),
    session: Session = Depends(get_session),
) -> TrainingJobMetricsHistoryResponse:
    """
    Get training metrics history for a job.
    """
    repo = JobRepository(session)
    metrics_repo = MetricsRepository(session)

    job = repo.get_by_uuid(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    db_metrics = metrics_repo.get_metrics(
        job_uuid=job_id,
        start_step=start_step,
        end_step=end_step,
    )

    metrics = [
        TrainingMetrics(
            step=m.step,
            timestamp=m.timestamp,
            epoch=m.epoch,
            policy_loss=m.policy_loss,
            value_loss=m.value_loss,
            total_loss=m.total_loss,
            reward_mean=m.reward_mean,
            reward_std=m.reward_std,
            kl_divergence=m.kl_divergence,
            entropy=m.entropy,
            learning_rate=m.learning_rate,
        )
        for m in db_metrics
    ]

    return TrainingJobMetricsHistoryResponse(
        job_id=job_id,
        metrics=metrics,
        total_points=len(metrics),
    )


@router.get("/{job_id}/checkpoints")
async def get_job_checkpoints(
    job_id: str,
    session: Session = Depends(get_session),
) -> List[Dict[str, Any]]:
    """
    Get all checkpoints for a job.
    """
    repo = JobRepository(session)
    checkpoint_repo = CheckpointRepository(session)

    job = repo.get_by_uuid(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    checkpoints = checkpoint_repo.get_checkpoints(job_id)

    return [
        {
            "step": cp.step,
            "path": cp.path,
            "metrics": cp.metrics,
            "eval_results": cp.eval_results,
            "created_at": cp.created_at.isoformat(),
        }
        for cp in checkpoints
    ]


@router.get("/{job_id}/config")
async def get_job_config(
    job_id: str,
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    Get the verl configuration for a job.

    Returns both the computed configuration and the actual verl command.
    """
    repo = JobRepository(session)
    job = repo.get_by_uuid(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    from ...core.compute_calculator import calculate_compute_config

    # Get compute configuration
    config = calculate_compute_config(
        model_size=job.config.get("model_size", "7B") if job.config else "7B",
        gpu_type=job.gpu_type,
        num_gpus=job.num_gpus,
        context_length=job.context_length,
        training_type=job.algorithm.value,
        lora_enabled=job.lora_enabled,
        lora_rank=job.lora_rank,
    )

    # Generate verl training config and command
    verl_config = _create_verl_config_from_job(job)
    verl_command = verl_config.to_command()
    verl_args = verl_config.to_verl_command_args()

    return {
        "job_id": job_id,
        "config": config["config"],
        "yaml": config["yaml"],
        "verl_command": verl_command,
        "verl_args": verl_args,
    }


@router.get("/{job_id}/script")
async def get_job_script(
    job_id: str,
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    Generate a runnable shell script for the training job.

    This can be used to run training outside of the platform.
    """
    repo = JobRepository(session)
    job = repo.get_by_uuid(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    verl_config = _create_verl_config_from_job(job)
    script = verl_config.to_shell_script()

    return {
        "job_id": job_id,
        "script": script,
        "command": verl_config.to_command(),
    }
