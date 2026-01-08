"""
Pydantic models for Training Job API
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


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


class TrainingJobCreate(BaseModel):
    """Request to create a training job"""
    name: str = Field(..., min_length=1, max_length=100, description="Job name")
    description: Optional[str] = Field(default=None, description="Job description")

    # Model config
    model_path: str = Field(..., description="Path to base model")
    model_size: Optional[str] = Field(default=None, description="Model size for auto-config")

    # Training config
    algorithm: TrainingAlgorithm = Field(default=TrainingAlgorithm.GRPO, description="Training algorithm")
    lora_enabled: bool = Field(default=False, description="Enable LoRA")
    lora_rank: int = Field(default=8, ge=1, description="LoRA rank")

    # Data config
    train_data_path: str = Field(..., description="Path to training data")
    eval_data_path: Optional[str] = Field(default=None, description="Path to evaluation data")

    # Resource config
    num_gpus: int = Field(default=8, ge=1, description="Number of GPUs")
    gpu_type: Optional[str] = Field(default=None, description="GPU type")

    # Training hyperparameters
    learning_rate: Optional[float] = Field(default=None, description="Learning rate")
    batch_size: Optional[int] = Field(default=None, description="Batch size")
    num_epochs: int = Field(default=3, ge=1, description="Number of epochs")
    max_steps: Optional[int] = Field(default=None, description="Max training steps")
    total_steps: Optional[int] = Field(default=None, description="Total training steps (for progress display)")
    context_length: int = Field(default=4096, ge=128, description="Context length")

    # Algorithm-specific
    kl_coef: float = Field(default=0.02, ge=0, description="KL coefficient")
    rollout_n: int = Field(default=8, ge=1, description="Rollout samples per prompt")

    # GRPO Reward Function Configuration
    reward_fn_type: str = Field(default="math_verify", description="Reward function type: math_verify, format_check, custom")
    reward_fn_extract_answer: str = Field(default="boxed", description="Answer extraction method: boxed, last_number, json")
    reward_fn_compare_method: str = Field(default="exact", description="Comparison method: exact, numeric, fuzzy")
    reward_fn_answer_key: str = Field(default="solution", description="Key in dataset for ground truth answer")
    reward_fn_custom_path: Optional[str] = Field(default=None, description="Path to custom reward function script")

    # PPO Reward Model Configuration
    reward_model_path: Optional[str] = Field(default=None, description="Path to reward model (for PPO)")
    reward_model_enable_gc: bool = Field(default=True, description="Enable gradient checkpointing for reward model")
    reward_model_offload: bool = Field(default=False, description="Offload reward model params to CPU")
    reward_model_micro_batch: int = Field(default=4, ge=1, description="Micro batch size for reward model")

    # Unified Reward Script Configuration (for PPO/GRPO/GSPO)
    reward_script_path: Optional[str] = Field(
        default=None,
        description="Path to reward script (e.g., reward_scripts/rule_math.py)"
    )
    reward_script_type: str = Field(
        default="rule",
        description="Reward script type: rule, api, model"
    )
    reward_script_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata passed to reward script (e.g., api_key, model_path, checks)"
    )

    # Monitoring
    checkpoint_interval: int = Field(default=500, ge=1, description="Checkpoint interval")
    eval_interval: int = Field(default=1000, ge=1, description="Evaluation interval")
    eval_benchmarks: List[str] = Field(default=["gsm8k"], description="Benchmarks to evaluate")

    # Resume from checkpoint
    resume_from_checkpoint: Optional[str] = Field(
        default=None,
        description="Path to checkpoint directory (e.g., ./outputs/job-uuid/global_step_1000)"
    )

    # Advanced config
    config_overrides: Optional[Dict[str, Any]] = Field(default=None, description="Config overrides")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "qwen2.5-7b-grpo-math",
                "model_path": "Qwen/Qwen2.5-7B-Instruct",
                "algorithm": "grpo",
                "train_data_path": "/data/math_train.parquet",
                "num_gpus": 8,
                "num_epochs": 3,
            }
        }


class TrainingJobUpdate(BaseModel):
    """Request to update a training job (only for pending jobs)"""
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[JobStatus] = None

    # Model config
    model_path: Optional[str] = None
    algorithm: Optional[TrainingAlgorithm] = None
    lora_enabled: Optional[bool] = None
    lora_rank: Optional[int] = None

    # Data config
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None

    # Resource config
    num_gpus: Optional[int] = None
    gpu_type: Optional[str] = None

    # Training hyperparameters
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    num_epochs: Optional[int] = None
    max_steps: Optional[int] = None
    total_steps: Optional[int] = None
    context_length: Optional[int] = None

    # Algorithm-specific
    kl_coef: Optional[float] = None
    rollout_n: Optional[int] = None

    # GRPO Reward Function Configuration
    reward_fn_type: Optional[str] = None
    reward_fn_extract_answer: Optional[str] = None
    reward_fn_compare_method: Optional[str] = None
    reward_fn_answer_key: Optional[str] = None
    reward_fn_custom_path: Optional[str] = None

    # PPO Reward Model Configuration
    reward_model_path: Optional[str] = None
    reward_model_enable_gc: Optional[bool] = None
    reward_model_offload: Optional[bool] = None
    reward_model_micro_batch: Optional[int] = None

    # Unified Reward Script Configuration
    reward_script_path: Optional[str] = None
    reward_script_type: Optional[str] = None
    reward_script_metadata: Optional[Dict[str, Any]] = None


class TrainingMetrics(BaseModel):
    """Training metrics at a point in time"""
    step: int
    timestamp: datetime
    policy_loss: Optional[float] = None
    value_loss: Optional[float] = None
    total_loss: Optional[float] = None
    reward_mean: Optional[float] = None
    reward_std: Optional[float] = None
    kl_divergence: Optional[float] = None
    entropy: Optional[float] = None
    learning_rate: Optional[float] = None
    grad_norm: Optional[float] = None
    throughput_tokens_per_sec: Optional[float] = None
    epoch: Optional[int] = None


class TrainingJobResponse(BaseModel):
    """Response for training job"""
    id: str
    name: str
    description: Optional[str]
    status: JobStatus
    algorithm: TrainingAlgorithm
    model_path: str
    lora_enabled: bool

    # Progress
    current_step: int = 0
    total_steps: Optional[int] = None
    current_epoch: int = 0
    total_epochs: int

    # Timing
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Resources
    num_gpus: int
    gpu_type: Optional[str] = None

    # Latest metrics
    latest_metrics: Optional[TrainingMetrics] = None

    # Paths
    output_path: Optional[str] = None
    checkpoint_paths: List[str] = []


class TrainingJobListResponse(BaseModel):
    """Response for listing training jobs"""
    jobs: List[TrainingJobResponse]
    total: int
    page: int
    page_size: int


class TrainingJobLogsResponse(BaseModel):
    """Response for training job logs"""
    job_id: str
    logs: List[str]
    has_more: bool


class TrainingJobMetricsHistoryResponse(BaseModel):
    """Response for training metrics history"""
    job_id: str
    metrics: List[TrainingMetrics]
    total_points: int


class GradientStats(BaseModel):
    """Gradient statistics for a layer"""
    layer_name: str
    mean: float
    std: float
    max: float
    min: float


class GradientHeatmapResponse(BaseModel):
    """Response for gradient heatmap data"""
    job_id: str
    steps: List[int]
    layers: List[str]
    data: List[List[float]]  # 2D array: layers x steps


class EvaluationResult(BaseModel):
    """Evaluation result for a checkpoint"""
    checkpoint_step: int
    benchmark: str
    score: float
    num_samples: int
    evaluated_at: datetime


class EvaluationResultsResponse(BaseModel):
    """Response for evaluation results"""
    job_id: str
    results: List[EvaluationResult]


class ResourceUsage(BaseModel):
    """Resource usage statistics"""
    gpu_utilization: List[float]  # Per GPU
    gpu_memory_used: List[float]  # Per GPU in GB
    gpu_memory_total: List[float]  # Per GPU in GB
    cpu_utilization: float
    ram_used_gb: float
    disk_io_read_mbps: float
    disk_io_write_mbps: float
    network_recv_mbps: float
    network_send_mbps: float


class ResourceUsageResponse(BaseModel):
    """Response for resource usage"""
    job_id: str
    timestamp: datetime
    usage: ResourceUsage
