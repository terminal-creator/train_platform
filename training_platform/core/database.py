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


class ApiKey(SQLModel, table=True):
    """API Key for authentication"""
    __tablename__ = "api_keys"

    id: Optional[int] = Field(default=None, primary_key=True)
    key: str = Field(index=True, unique=True)  # The actual API key (hashed)
    name: str = Field(index=True)  # Friendly name
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    # Rate limiting
    rate_limit_per_minute: int = Field(default=60)  # Requests per minute
    # Scopes (optional)
    scopes: Optional[str] = None  # JSON string of allowed scopes


class TrainingJob(SQLModel, table=True):
    """Training job database model"""
    __tablename__ = "training_jobs"

    id: Optional[int] = Field(default=None, primary_key=True)
    uuid: str = Field(index=True, unique=True)
    name: str = Field(index=True)
    description: Optional[str] = None
    status: JobStatus = Field(default=JobStatus.PENDING, index=True)
    algorithm: TrainingAlgorithm = Field(index=True)

    # Phase 2: Recipe and data versioning
    recipe_id: Optional[str] = Field(default=None, index=True)  # Recipe used for this job
    dataset_version_hash: Optional[str] = Field(default=None, index=True)  # Dataset version snapshot

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

    # Metrics tracking
    metrics_last_offset: int = 0  # For incremental metrics file reading

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
    """Training metrics history (Phase 1 enhanced)"""
    __tablename__ = "training_metrics"

    id: Optional[int] = Field(default=None, primary_key=True)
    job_uuid: str = Field(index=True, foreign_key="training_jobs.uuid")
    step: int = Field(index=True)
    epoch: int = 0

    # Loss metrics (兼容旧版，policy_loss = actor_loss)
    policy_loss: Optional[float] = None  # actor_loss for PPO/GRPO
    value_loss: Optional[float] = None  # critic_loss for PPO/GRPO
    total_loss: Optional[float] = None

    # RL metrics
    reward_mean: Optional[float] = None
    reward_std: Optional[float] = None
    reward_max: Optional[float] = None  # Phase 1: 新增 max reward
    reward_min: Optional[float] = None  # Phase 1: 新增 min reward

    kl_divergence: Optional[float] = None  # KL mean
    kl_divergence_max: Optional[float] = None  # Phase 1: 新增 KL max (用于异常检测)
    entropy: Optional[float] = None
    learning_rate: Optional[float] = None

    # Gradient metrics (Phase 1: 新增梯度信息，用于调试)
    grad_norm_actor: Optional[float] = None  # actor 梯度范数
    grad_norm_critic: Optional[float] = None  # critic 梯度范数

    # Performance metrics (Phase 1: 新增性能指标)
    tokens_per_second: Optional[float] = None  # 吞吐量
    step_time: Optional[float] = None  # 单步耗时（秒）
    gpu_memory_allocated_gib: Optional[float] = None  # GPU 显存使用（GiB）

    # Anomaly detection (Phase 1: 新增异常标记)
    has_anomaly: bool = False  # 是否检测到异常
    anomaly_type: Optional[str] = None  # 异常类型: 'nan', 'kl_explosion', 'loss_plateau'
    anomaly_message: Optional[str] = None  # 异常详细信息

    # Additional metrics (stored as JSON)
    # Phase 1: 存储 PlatformCallback 的完整原始指标
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


class DatasetSyncStatus(str, Enum):
    """Dataset sync status to remote server"""
    NOT_SYNCED = "not_synced"
    SYNCING = "syncing"
    SYNCED = "synced"
    FAILED = "failed"


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

    # Pre-computed statistics (computed on upload)
    statistics: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    # {
    #   "format_type": "messages" | "prompt_response",
    #   "avg_turns": 1.5,
    #   "avg_prompt_chars": 100,
    #   "avg_response_chars": 200,
    #   "avg_total_chars": 300,
    #   "has_system_prompt": 85.0,  # percentage
    #   "prompt_length_distribution": {"0-50": 10, "50-100": 50, ...},
    #   "response_length_distribution": {...},
    #   "turns_distribution": {"1轮": 100, "2轮": 50, ...}
    # }

    # Pre-computed quality stats
    quality_stats: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    # {
    #   "quality_score": 85.5,
    #   "issues_found": 10,
    #   "issues": [{"issue_type": "空回复", "count": 5, "percentage": 0.5, "sample_indices": [1,2,3]}]
    # }

    # Loss computation field configuration
    prompt_field: str = "prompt"
    response_field: str = "response"

    # Remote sync info
    remote_path: Optional[str] = None  # Path on remote SSH server
    sync_status: DatasetSyncStatus = Field(default=DatasetSyncStatus.NOT_SYNCED)
    sync_error: Optional[str] = None
    synced_at: Optional[datetime] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    analyzed_at: Optional[datetime] = None


class DatasetVersion(SQLModel, table=True):
    """Dataset version snapshots for data lineage tracking (Phase 2)"""
    __tablename__ = "dataset_versions"

    id: Optional[int] = Field(default=None, primary_key=True)

    # Dataset identification
    dataset_name: str = Field(index=True)  # Dataset name (can have multiple versions)
    file_path: str  # Original file path

    # Version fingerprint
    file_hash: str = Field(index=True, unique=True)  # SHA256 hash - unique identifier
    hash_algorithm: str = "sha256"

    # File metadata
    file_size: int  # Size in bytes
    file_size_mb: float
    format: str  # jsonl, parquet, csv
    num_samples: Optional[int] = None  # Number of samples in dataset

    # Optional metadata
    description: Optional[str] = None
    tags: List[str] = Field(default=[], sa_column=Column(JSON))

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)  # When snapshot was created
    modified_at: str  # When the file was last modified (ISO format)


class PipelineStatus(str, Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Pipeline(SQLModel, table=True):
    """Training pipeline for multi-stage workflows (Phase 3)"""
    __tablename__ = "pipelines"

    id: Optional[int] = Field(default=None, primary_key=True)
    uuid: str = Field(index=True, unique=True)
    name: str = Field(index=True)
    description: Optional[str] = None

    # Pipeline configuration
    stages: List[Dict[str, Any]] = Field(default=[], sa_column=Column(JSON))
    # [{"name": "preprocess", "task": "preprocess_dataset", "params": {...}}, ...]

    dependencies: Dict[str, List[str]] = Field(default={}, sa_column=Column(JSON))
    # {"stage2": ["stage1"], "stage3": ["stage1", "stage2"]}

    # Execution status
    status: PipelineStatus = Field(default=PipelineStatus.PENDING, index=True)

    # Celery task tracking
    celery_task_id: Optional[str] = Field(default=None, index=True)
    stage_tasks: Dict[str, str] = Field(default={}, sa_column=Column(JSON))
    # {"stage1": "celery-task-id-1", "stage2": "celery-task-id-2"}

    # Results
    results: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    error_message: Optional[str] = None

    # Priority and retry
    priority: int = 5  # 1-10, higher = more important
    max_retries: int = 3
    retry_count: int = 0

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PipelineStageStatus(str, Enum):
    """Pipeline stage status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineStage(SQLModel, table=True):
    """Individual stage in a pipeline (Phase 3)"""
    __tablename__ = "pipeline_stages"

    id: Optional[int] = Field(default=None, primary_key=True)
    pipeline_uuid: str = Field(index=True, foreign_key="pipelines.uuid")

    # Stage identification
    stage_name: str
    stage_order: int  # Execution order

    # Task configuration
    task_name: str  # Celery task name
    task_params: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))

    # Dependencies
    depends_on: List[str] = Field(default=[], sa_column=Column(JSON))
    # List of stage names this stage depends on

    # Execution status
    status: PipelineStageStatus = Field(default=PipelineStageStatus.PENDING)
    celery_task_id: Optional[str] = None

    # Results
    result: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    error_message: Optional[str] = None

    # Retry configuration
    max_retries: int = 3
    retry_count: int = 0
    retry_delay: int = 60  # seconds

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


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


class ApiKeyRepository:
    """Repository for API key operations"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, api_key: ApiKey) -> ApiKey:
        """Create a new API key"""
        self.session.add(api_key)
        self.session.commit()
        self.session.refresh(api_key)
        return api_key

    def get_by_key(self, key: str) -> Optional[ApiKey]:
        """Get API key by key string"""
        statement = select(ApiKey).where(ApiKey.key == key)
        return self.session.exec(statement).first()

    def list_keys(self, active_only: bool = True) -> List[ApiKey]:
        """List all API keys"""
        statement = select(ApiKey)
        if active_only:
            statement = statement.where(ApiKey.is_active == True)
        return list(self.session.exec(statement).all())

    def update_last_used(self, api_key: ApiKey):
        """Update last_used_at timestamp"""
        api_key.last_used_at = datetime.utcnow()
        self.session.add(api_key)
        self.session.commit()

    def deactivate(self, key: str) -> bool:
        """Deactivate an API key"""
        api_key = self.get_by_key(key)
        if api_key:
            api_key.is_active = False
            self.session.add(api_key)
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


class DatasetVersionRepository:
    """Repository for dataset version operations (Phase 2)"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, version: DatasetVersion) -> DatasetVersion:
        """Create a new dataset version snapshot"""
        self.session.add(version)
        self.session.commit()
        self.session.refresh(version)
        return version

    def get_by_hash(self, file_hash: str) -> Optional[DatasetVersion]:
        """Get dataset version by hash"""
        statement = select(DatasetVersion).where(DatasetVersion.file_hash == file_hash)
        return self.session.exec(statement).first()

    def list_versions(
        self,
        dataset_name: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[List[DatasetVersion], int]:
        """List dataset versions with optional filtering"""
        statement = select(DatasetVersion)
        if dataset_name:
            statement = statement.where(DatasetVersion.dataset_name == dataset_name)

        count_statement = select(DatasetVersion)
        if dataset_name:
            count_statement = count_statement.where(DatasetVersion.dataset_name == dataset_name)
        total = len(self.session.exec(count_statement).all())

        statement = statement.order_by(DatasetVersion.created_at.desc())
        statement = statement.offset(offset).limit(limit)

        return list(self.session.exec(statement).all()), total

    def get_latest_version(self, dataset_name: str) -> Optional[DatasetVersion]:
        """Get the latest version of a dataset"""
        statement = (
            select(DatasetVersion)
            .where(DatasetVersion.dataset_name == dataset_name)
            .order_by(DatasetVersion.created_at.desc())
            .limit(1)
        )
        return self.session.exec(statement).first()

    def find_jobs_using_version(self, file_hash: str) -> List[TrainingJob]:
        """Find all training jobs that used this dataset version"""
        statement = (
            select(TrainingJob)
            .where(TrainingJob.dataset_version_hash == file_hash)
            .order_by(TrainingJob.created_at.desc())
        )
        return list(self.session.exec(statement).all())


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


class PipelineRepository:
    """Repository for pipeline operations (Phase 3)"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, pipeline: Pipeline) -> Pipeline:
        """Create a new pipeline"""
        self.session.add(pipeline)
        self.session.commit()
        self.session.refresh(pipeline)
        return pipeline

    def get_by_uuid(self, uuid: str) -> Optional[Pipeline]:
        """Get pipeline by UUID"""
        statement = select(Pipeline).where(Pipeline.uuid == uuid)
        return self.session.exec(statement).first()

    def list_pipelines(
        self,
        status: Optional[PipelineStatus] = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[List[Pipeline], int]:
        """List pipelines with optional filtering"""
        statement = select(Pipeline)
        if status:
            statement = statement.where(Pipeline.status == status)

        count_statement = select(Pipeline)
        if status:
            count_statement = count_statement.where(Pipeline.status == status)
        total = len(self.session.exec(count_statement).all())

        statement = statement.order_by(Pipeline.created_at.desc())
        statement = statement.offset(offset).limit(limit)

        return list(self.session.exec(statement).all()), total

    def update(self, pipeline: Pipeline) -> Pipeline:
        """Update a pipeline"""
        pipeline.updated_at = datetime.utcnow()
        self.session.add(pipeline)
        self.session.commit()
        self.session.refresh(pipeline)
        return pipeline

    def delete(self, uuid: str) -> bool:
        """Delete a pipeline"""
        pipeline = self.get_by_uuid(uuid)
        if pipeline:
            self.session.delete(pipeline)
            self.session.commit()
            return True
        return False

    def get_stages(self, pipeline_uuid: str) -> List[PipelineStage]:
        """Get all stages for a pipeline"""
        statement = (
            select(PipelineStage)
            .where(PipelineStage.pipeline_uuid == pipeline_uuid)
            .order_by(PipelineStage.stage_order)
        )
        return list(self.session.exec(statement).all())

    def create_stage(self, stage: PipelineStage) -> PipelineStage:
        """Create a new pipeline stage"""
        self.session.add(stage)
        self.session.commit()
        self.session.refresh(stage)
        return stage

    def update_stage(self, stage: PipelineStage) -> PipelineStage:
        """Update a pipeline stage"""
        self.session.add(stage)
        self.session.commit()
        self.session.refresh(stage)
        return stage

    def update_pipeline_status_atomic(
        self,
        pipeline_uuid: str,
        new_status: PipelineStatus,
        error_message: Optional[str] = None,
        allowed_current_statuses: Optional[List[PipelineStatus]] = None,
    ) -> bool:
        """
        原子性更新 pipeline 状态（并发安全）

        使用 SELECT FOR UPDATE 确保并发更新不会导致状态冲突。
        支持条件更新（仅在当前状态符合条件时才更新）。

        Args:
            pipeline_uuid: Pipeline UUID
            new_status: 目标状态
            error_message: 错误信息（可选）
            allowed_current_statuses: 允许的当前状态列表（可选）
                例如：[PipelineStatus.RUNNING] 表示只有当前是 RUNNING 时才能更新

        Returns:
            bool: 是否成功更新

        使用示例：
        ```python
        # 只有当 pipeline 是 RUNNING 状态时，才能标记为 FAILED
        success = repo.update_pipeline_status_atomic(
            pipeline_uuid="xxx",
            new_status=PipelineStatus.FAILED,
            error_message="Stage A failed",
            allowed_current_statuses=[PipelineStatus.RUNNING],
        )
        ```

        为什么需要这个方法：
        - 避免多个 stage 同时失败时的竞态问题
        - 避免重复将 pipeline 标记为 FAILED
        - 确保状态转换的有效性
        """
        from sqlalchemy import text

        try:
            # 使用 SELECT FOR UPDATE 锁定行
            statement = select(Pipeline).where(Pipeline.uuid == pipeline_uuid)
            # 添加 FOR UPDATE 锁
            statement = statement.with_for_update()

            pipeline = self.session.exec(statement).first()

            if not pipeline:
                return False

            # 检查当前状态是否允许更新
            if allowed_current_statuses:
                if pipeline.status not in allowed_current_statuses:
                    # 当前状态不允许更新，直接返回
                    return False

            # 更新状态
            pipeline.status = new_status
            pipeline.updated_at = datetime.utcnow()

            if new_status in [PipelineStatus.COMPLETED, PipelineStatus.FAILED]:
                pipeline.completed_at = datetime.utcnow()

            if error_message:
                pipeline.error_message = error_message

            self.session.add(pipeline)
            self.session.commit()
            self.session.refresh(pipeline)

            return True

        except Exception as e:
            self.session.rollback()
            import logging
            logging.getLogger(__name__).error(
                f"Failed to update pipeline status atomically: {e}",
                exc_info=True
            )
            return False
