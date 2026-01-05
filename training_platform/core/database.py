"""
Database Layer using SQLModel

Provides persistent storage for training jobs, metrics, and configurations.
Supports SQLite (default) and PostgreSQL.
"""

from sqlmodel import SQLModel, Field, Session, create_engine, select, JSON, Column
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import os
import json

# Database URL from environment or default to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./training_platform.db")


class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingAlgorithm(str, Enum):
    SFT = "sft"
    PPO = "ppo"
    GRPO = "grpo"
    DPO = "dpo"
    GSPO = "gspo"


# Evaluation enums
class EvalDatasetFormat(str, Enum):
    QA = "qa"                    # {"question": "...", "answer": "..."}
    DIALOGUE = "dialogue"        # {"messages": [...]}


class EvalCapability(str, Enum):
    MATH = "math"
    CODE = "code"
    REASONING = "reasoning"
    LANGUAGE = "language"
    CUSTOM = "custom"


class EvalMethod(str, Enum):
    EXACT_MATCH = "exact_match"
    CONTAINS = "contains"
    LLM_JUDGE = "llm_judge"


class EvalTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingJob(SQLModel, table=True):
    """Training job database model"""
    __tablename__ = "training_jobs"

    id: Optional[int] = Field(default=None, primary_key=True)
    uuid: str = Field(index=True, unique=True)
    name: str = Field(index=True)
    description: Optional[str] = None
    status: JobStatus = Field(default=JobStatus.PENDING, index=True)
    algorithm: TrainingAlgorithm = Field(index=True)

    # Model configuration
    model_path: str
    lora_enabled: bool = False
    lora_rank: int = 8

    # Data paths
    train_data_path: str
    eval_data_path: Optional[str] = None

    # Training parameters (stored as JSON)
    config: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))

    # Resource allocation
    num_gpus: int = 8
    gpu_type: str = "A100-80G"

    # Training hyperparameters
    learning_rate: float = 1e-6
    batch_size: int = 256
    num_epochs: int = 3
    max_steps: Optional[int] = None
    context_length: int = 4096
    kl_coef: float = 0.02
    rollout_n: int = 8

    # Progress tracking
    current_step: int = 0
    total_steps: Optional[int] = None
    current_epoch: int = 0

    # Ray job integration
    ray_job_id: Optional[str] = Field(default=None, index=True)

    # Output paths
    output_path: Optional[str] = None
    checkpoint_paths: List[str] = Field(default=[], sa_column=Column(JSON))

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TrainingMetric(SQLModel, table=True):
    """Training metrics history"""
    __tablename__ = "training_metrics"

    id: Optional[int] = Field(default=None, primary_key=True)
    job_uuid: str = Field(index=True, foreign_key="training_jobs.uuid")
    step: int = Field(index=True)
    epoch: int = 0

    # Loss metrics
    policy_loss: Optional[float] = None
    value_loss: Optional[float] = None
    total_loss: Optional[float] = None

    # RL metrics
    reward_mean: Optional[float] = None
    reward_std: Optional[float] = None
    kl_divergence: Optional[float] = None
    entropy: Optional[float] = None
    learning_rate: Optional[float] = None

    # Additional metrics (stored as JSON)
    extra_metrics: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TrainingLog(SQLModel, table=True):
    """Training logs"""
    __tablename__ = "training_logs"

    id: Optional[int] = Field(default=None, primary_key=True)
    job_uuid: str = Field(index=True, foreign_key="training_jobs.uuid")
    level: str = "INFO"
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Checkpoint(SQLModel, table=True):
    """Model checkpoints"""
    __tablename__ = "checkpoints"

    id: Optional[int] = Field(default=None, primary_key=True)
    job_uuid: str = Field(index=True, foreign_key="training_jobs.uuid")
    step: int
    path: str

    # Metrics at checkpoint
    metrics: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))

    # Evaluation results
    eval_results: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))

    created_at: datetime = Field(default_factory=datetime.utcnow)


class GpuUsageRecord(SQLModel, table=True):
    """GPU usage records for monitoring"""
    __tablename__ = "gpu_usage"

    id: Optional[int] = Field(default=None, primary_key=True)
    job_uuid: str = Field(index=True, foreign_key="training_jobs.uuid")
    gpu_index: int
    utilization: float
    memory_used: float
    memory_total: float
    temperature: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TrainingDataset(SQLModel, table=True):
    """Training dataset metadata with label field support"""
    __tablename__ = "training_datasets"

    id: Optional[int] = Field(default=None, primary_key=True)
    uuid: str = Field(index=True, unique=True)
    name: str = Field(index=True)
    description: Optional[str] = None

    # File info
    file_path: str
    file_format: str = "jsonl"  # jsonl, json, parquet
    file_size_mb: float = 0.0
    total_rows: int = 0
    columns: List[str] = Field(default=[], sa_column=Column(JSON))

    # Label field configuration
    label_fields: List[str] = Field(default=[], sa_column=Column(JSON))
    field_distributions: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    # {"tenant": {"珍酒": 180, "之了课堂": 102}, "best_model": {...}}

    # Loss computation field configuration
    prompt_field: str = "prompt"
    response_field: str = "response"

    created_at: datetime = Field(default_factory=datetime.utcnow)
    analyzed_at: Optional[datetime] = None


class EvalDataset(SQLModel, table=True):
    """Evaluation dataset for custom benchmarks"""
    __tablename__ = "eval_datasets"

    id: Optional[int] = Field(default=None, primary_key=True)
    uuid: str = Field(index=True, unique=True)
    name: str = Field(index=True)
    description: Optional[str] = None

    # Dataset properties
    format: EvalDatasetFormat
    capability: EvalCapability = Field(index=True)
    eval_method: EvalMethod = Field(default=EvalMethod.EXACT_MATCH)

    # File info
    file_path: str
    sample_count: int = 0

    # Label field for grouping results
    label_field: Optional[str] = None  # e.g., "difficulty", "category"
    available_labels: List[str] = Field(default=[], sa_column=Column(JSON))

    # Custom LLM judge prompt template
    judge_prompt_template: Optional[str] = None
    # Supports placeholders: {question}, {expected}, {actual}

    # LLM judge config (if method is llm_judge)
    judge_config: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class EvalTask(SQLModel, table=True):
    """Evaluation task for running evaluations"""
    __tablename__ = "eval_tasks"

    id: Optional[int] = Field(default=None, primary_key=True)
    uuid: str = Field(index=True, unique=True)
    name: Optional[str] = None  # Optional display name for comparisons

    # References
    job_uuid: Optional[str] = Field(default=None, index=True, foreign_key="training_jobs.uuid")
    checkpoint_id: Optional[int] = Field(default=None, foreign_key="checkpoints.id")
    dataset_uuid: str = Field(index=True, foreign_key="eval_datasets.uuid")

    # Model configuration
    model_type: str = "checkpoint"  # api, local_model, checkpoint
    inference_config: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    # inference_config examples:
    # checkpoint: {"checkpoint_path": "..."}
    # local_model: {"model_path": "..."}
    # api: {"base_url": "...", "model": "...", "api_key_env": "..."}

    # Status
    status: EvalTaskStatus = Field(default=EvalTaskStatus.PENDING)

    # Results
    score: Optional[float] = None  # 0-100
    correct_count: Optional[int] = None
    total_count: Optional[int] = None
    detailed_results: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))

    # Per-label results for grouped analysis
    label_results: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    # {"easy": {"correct": 10, "total": 15, "accuracy": 66.7}, "hard": {...}}

    # Sample-level results for comparison
    sample_results: List[Dict[str, Any]] = Field(default=[], sa_column=Column(JSON))
    # [{"id": "...", "input": "...", "expected": "...", "actual": "...", "correct": true, "label": "easy"}]

    # Error handling
    error_message: Optional[str] = None

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class EvalComparison(SQLModel, table=True):
    """Model comparison for A/B testing"""
    __tablename__ = "eval_comparisons"

    id: Optional[int] = Field(default=None, primary_key=True)
    uuid: str = Field(index=True, unique=True)
    name: str
    description: Optional[str] = None
    dataset_uuid: str = Field(index=True, foreign_key="eval_datasets.uuid")

    # Model A (baseline)
    model_a_task_uuid: str = Field(foreign_key="eval_tasks.uuid")
    model_a_name: str  # e.g., "训练前", "Baseline"

    # Model B (target)
    model_b_task_uuid: str = Field(foreign_key="eval_tasks.uuid")
    model_b_name: str  # e.g., "训练后", "Step 1000"

    # Comparison results
    comparison_results: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    # {
    #   "overall": {"model_a_accuracy": 65.5, "model_b_accuracy": 78.2, "delta": 12.7},
    #   "by_label": {"easy": {"delta": 5.0}, "hard": {"delta": 20.0}},
    #   "improved_count": 23, "degraded_count": 5, "unchanged_count": 72
    # }

    # Sample-level diffs
    sample_diffs: List[Dict[str, Any]] = Field(default=[], sa_column=Column(JSON))
    # [{"id": "...", "input": "...", "expected": "...",
    #   "model_a_output": "...", "model_a_correct": false,
    #   "model_b_output": "...", "model_b_correct": true,
    #   "change": "improved", "label": "easy"}]

    status: EvalTaskStatus = Field(default=EvalTaskStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=os.getenv("DB_ECHO", "false").lower() == "true",
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)


def init_db():
    """Initialize database tables"""
    SQLModel.metadata.create_all(engine)


def get_session():
    """Get database session"""
    with Session(engine) as session:
        yield session


class JobRepository:
    """Repository for training job operations"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, job: TrainingJob) -> TrainingJob:
        """Create a new training job"""
        self.session.add(job)
        self.session.commit()
        self.session.refresh(job)
        return job

    def get_by_uuid(self, uuid: str) -> Optional[TrainingJob]:
        """Get job by UUID"""
        statement = select(TrainingJob).where(TrainingJob.uuid == uuid)
        return self.session.exec(statement).first()

    def get_by_ray_job_id(self, ray_job_id: str) -> Optional[TrainingJob]:
        """Get job by Ray job ID"""
        statement = select(TrainingJob).where(TrainingJob.ray_job_id == ray_job_id)
        return self.session.exec(statement).first()

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        algorithm: Optional[TrainingAlgorithm] = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[List[TrainingJob], int]:
        """List jobs with filtering and pagination"""
        statement = select(TrainingJob)

        if status:
            statement = statement.where(TrainingJob.status == status)
        if algorithm:
            statement = statement.where(TrainingJob.algorithm == algorithm)

        # Get total count
        count_statement = select(TrainingJob)
        if status:
            count_statement = count_statement.where(TrainingJob.status == status)
        if algorithm:
            count_statement = count_statement.where(TrainingJob.algorithm == algorithm)
        total = len(self.session.exec(count_statement).all())

        # Order and paginate
        statement = statement.order_by(TrainingJob.created_at.desc())
        statement = statement.offset(offset).limit(limit)

        jobs = self.session.exec(statement).all()
        return list(jobs), total

    def update(self, job: TrainingJob) -> TrainingJob:
        """Update a training job"""
        job.updated_at = datetime.utcnow()
        self.session.add(job)
        self.session.commit()
        self.session.refresh(job)
        return job

    def delete(self, uuid: str) -> bool:
        """Delete a training job"""
        job = self.get_by_uuid(uuid)
        if job:
            self.session.delete(job)
            self.session.commit()
            return True
        return False


class MetricsRepository:
    """Repository for training metrics operations"""

    def __init__(self, session: Session):
        self.session = session

    def add_metric(self, metric: TrainingMetric) -> TrainingMetric:
        """Add a new metric record"""
        self.session.add(metric)
        self.session.commit()
        self.session.refresh(metric)
        return metric

    def add_metrics_batch(self, metrics: List[TrainingMetric]):
        """Add multiple metrics at once"""
        for metric in metrics:
            self.session.add(metric)
        self.session.commit()

    def get_metrics(
        self,
        job_uuid: str,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        limit: int = 1000,
    ) -> List[TrainingMetric]:
        """Get metrics for a job"""
        statement = select(TrainingMetric).where(TrainingMetric.job_uuid == job_uuid)

        if start_step is not None:
            statement = statement.where(TrainingMetric.step >= start_step)
        if end_step is not None:
            statement = statement.where(TrainingMetric.step <= end_step)

        statement = statement.order_by(TrainingMetric.step).limit(limit)
        return list(self.session.exec(statement).all())

    def get_latest_metric(self, job_uuid: str) -> Optional[TrainingMetric]:
        """Get latest metric for a job"""
        statement = (
            select(TrainingMetric)
            .where(TrainingMetric.job_uuid == job_uuid)
            .order_by(TrainingMetric.step.desc())
            .limit(1)
        )
        return self.session.exec(statement).first()


class LogRepository:
    """Repository for training logs"""

    def __init__(self, session: Session):
        self.session = session

    def add_log(self, log: TrainingLog) -> TrainingLog:
        """Add a log entry"""
        self.session.add(log)
        self.session.commit()
        self.session.refresh(log)
        return log

    def get_logs(
        self,
        job_uuid: str,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[List[TrainingLog], int]:
        """Get logs for a job"""
        count_statement = select(TrainingLog).where(TrainingLog.job_uuid == job_uuid)
        total = len(self.session.exec(count_statement).all())

        statement = (
            select(TrainingLog)
            .where(TrainingLog.job_uuid == job_uuid)
            .order_by(TrainingLog.timestamp.desc())
            .offset(offset)
            .limit(limit)
        )
        logs = list(self.session.exec(statement).all())
        return logs, total


class CheckpointRepository:
    """Repository for checkpoint operations"""

    def __init__(self, session: Session):
        self.session = session

    def add_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        """Add a checkpoint"""
        self.session.add(checkpoint)
        self.session.commit()
        self.session.refresh(checkpoint)
        return checkpoint

    def get_checkpoints(self, job_uuid: str) -> List[Checkpoint]:
        """Get all checkpoints for a job"""
        statement = (
            select(Checkpoint)
            .where(Checkpoint.job_uuid == job_uuid)
            .order_by(Checkpoint.step)
        )
        return list(self.session.exec(statement).all())

    def get_best_checkpoint(
        self,
        job_uuid: str,
        metric: str = "eval_loss",
        mode: str = "min",
    ) -> Optional[Checkpoint]:
        """Get best checkpoint by metric"""
        checkpoints = self.get_checkpoints(job_uuid)

        if not checkpoints:
            return None

        def get_metric_value(cp: Checkpoint) -> float:
            return cp.metrics.get(metric) or cp.eval_results.get(metric) or float('inf')

        if mode == "min":
            return min(checkpoints, key=get_metric_value)
        else:
            return max(checkpoints, key=get_metric_value)

    def get_by_id(self, checkpoint_id: int) -> Optional[Checkpoint]:
        """Get checkpoint by ID"""
        statement = select(Checkpoint).where(Checkpoint.id == checkpoint_id)
        return self.session.exec(statement).first()


class EvalDatasetRepository:
    """Repository for evaluation dataset operations"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, dataset: EvalDataset) -> EvalDataset:
        """Create a new evaluation dataset"""
        self.session.add(dataset)
        self.session.commit()
        self.session.refresh(dataset)
        return dataset

    def get_by_uuid(self, uuid: str) -> Optional[EvalDataset]:
        """Get dataset by UUID"""
        statement = select(EvalDataset).where(EvalDataset.uuid == uuid)
        return self.session.exec(statement).first()

    def list_datasets(
        self,
        capability: Optional[EvalCapability] = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[List[EvalDataset], int]:
        """List datasets with optional filtering"""
        statement = select(EvalDataset)
        if capability:
            statement = statement.where(EvalDataset.capability == capability)

        # Get total count
        count_statement = select(EvalDataset)
        if capability:
            count_statement = count_statement.where(EvalDataset.capability == capability)
        total = len(self.session.exec(count_statement).all())

        # Order and paginate
        statement = statement.order_by(EvalDataset.created_at.desc())
        statement = statement.offset(offset).limit(limit)

        return list(self.session.exec(statement).all()), total

    def update(self, dataset: EvalDataset) -> EvalDataset:
        """Update a dataset"""
        dataset.updated_at = datetime.utcnow()
        self.session.add(dataset)
        self.session.commit()
        self.session.refresh(dataset)
        return dataset

    def delete(self, uuid: str) -> bool:
        """Delete a dataset"""
        dataset = self.get_by_uuid(uuid)
        if dataset:
            self.session.delete(dataset)
            self.session.commit()
            return True
        return False


class EvalTaskRepository:
    """Repository for evaluation task operations"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, task: EvalTask) -> EvalTask:
        """Create a new evaluation task"""
        self.session.add(task)
        self.session.commit()
        self.session.refresh(task)
        return task

    def get_by_uuid(self, uuid: str) -> Optional[EvalTask]:
        """Get task by UUID"""
        statement = select(EvalTask).where(EvalTask.uuid == uuid)
        return self.session.exec(statement).first()

    def get_tasks_for_job(self, job_uuid: str) -> List[EvalTask]:
        """Get all tasks for a job"""
        statement = (
            select(EvalTask)
            .where(EvalTask.job_uuid == job_uuid)
            .order_by(EvalTask.created_at.desc())
        )
        return list(self.session.exec(statement).all())

    def get_tasks_for_checkpoint(self, checkpoint_id: int) -> List[EvalTask]:
        """Get all tasks for a checkpoint"""
        statement = (
            select(EvalTask)
            .where(EvalTask.checkpoint_id == checkpoint_id)
            .order_by(EvalTask.created_at.desc())
        )
        return list(self.session.exec(statement).all())

    def get_tasks_for_dataset(self, dataset_uuid: str) -> List[EvalTask]:
        """Get all tasks for a dataset"""
        statement = (
            select(EvalTask)
            .where(EvalTask.dataset_uuid == dataset_uuid)
            .order_by(EvalTask.created_at.desc())
        )
        return list(self.session.exec(statement).all())

    def list_tasks(
        self,
        status: Optional[EvalTaskStatus] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[List[EvalTask], int]:
        """List all tasks with optional filtering"""
        statement = select(EvalTask)
        if status:
            statement = statement.where(EvalTask.status == status)

        count_statement = select(EvalTask)
        if status:
            count_statement = count_statement.where(EvalTask.status == status)
        total = len(self.session.exec(count_statement).all())

        statement = statement.order_by(EvalTask.created_at.desc())
        statement = statement.offset(offset).limit(limit)

        return list(self.session.exec(statement).all()), total

    def update(self, task: EvalTask) -> EvalTask:
        """Update a task"""
        self.session.add(task)
        self.session.commit()
        self.session.refresh(task)
        return task


class TrainingDatasetRepository:
    """Repository for training dataset operations"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, dataset: TrainingDataset) -> TrainingDataset:
        """Create a new training dataset"""
        self.session.add(dataset)
        self.session.commit()
        self.session.refresh(dataset)
        return dataset

    def get_by_uuid(self, uuid: str) -> Optional[TrainingDataset]:
        """Get dataset by UUID"""
        statement = select(TrainingDataset).where(TrainingDataset.uuid == uuid)
        return self.session.exec(statement).first()

    def list_datasets(
        self,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[List[TrainingDataset], int]:
        """List datasets with pagination"""
        count_statement = select(TrainingDataset)
        total = len(self.session.exec(count_statement).all())

        statement = select(TrainingDataset).order_by(TrainingDataset.created_at.desc())
        statement = statement.offset(offset).limit(limit)

        return list(self.session.exec(statement).all()), total

    def update(self, dataset: TrainingDataset) -> TrainingDataset:
        """Update a dataset"""
        self.session.add(dataset)
        self.session.commit()
        self.session.refresh(dataset)
        return dataset

    def delete(self, uuid: str) -> bool:
        """Delete a dataset"""
        dataset = self.get_by_uuid(uuid)
        if dataset:
            self.session.delete(dataset)
            self.session.commit()
            return True
        return False


class EvalComparisonRepository:
    """Repository for evaluation comparison operations"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, comparison: EvalComparison) -> EvalComparison:
        """Create a new comparison"""
        self.session.add(comparison)
        self.session.commit()
        self.session.refresh(comparison)
        return comparison

    def get_by_uuid(self, uuid: str) -> Optional[EvalComparison]:
        """Get comparison by UUID"""
        statement = select(EvalComparison).where(EvalComparison.uuid == uuid)
        return self.session.exec(statement).first()

    def list_comparisons(
        self,
        dataset_uuid: Optional[str] = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[List[EvalComparison], int]:
        """List comparisons with optional filtering"""
        statement = select(EvalComparison)
        if dataset_uuid:
            statement = statement.where(EvalComparison.dataset_uuid == dataset_uuid)

        count_statement = select(EvalComparison)
        if dataset_uuid:
            count_statement = count_statement.where(EvalComparison.dataset_uuid == dataset_uuid)
        total = len(self.session.exec(count_statement).all())

        statement = statement.order_by(EvalComparison.created_at.desc())
        statement = statement.offset(offset).limit(limit)

        return list(self.session.exec(statement).all()), total

    def update(self, comparison: EvalComparison) -> EvalComparison:
        """Update a comparison"""
        self.session.add(comparison)
        self.session.commit()
        self.session.refresh(comparison)
        return comparison

    def delete(self, uuid: str) -> bool:
        """Delete a comparison"""
        comparison = self.get_by_uuid(uuid)
        if comparison:
            self.session.delete(comparison)
            self.session.commit()
            return True
        return False
