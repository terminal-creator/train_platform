"""
Pydantic models for Model Surgery API
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class MergeMethod(str, Enum):
    LINEAR = "linear"
    SLERP = "slerp"
    TIES = "ties"
    DARE = "dare"
    SWA = "swa"


class SelectionCriteria(str, Enum):
    HIGHEST_REWARD = "highest_reward"
    HIGHEST_BENCHMARK = "highest_benchmark"
    LOWEST_KL = "lowest_kl"
    BALANCED = "balanced"
    CUSTOM = "custom"


class MergeRequest(BaseModel):
    """Request for model merging"""
    models: List[str] = Field(..., min_length=2, description="List of model paths")
    method: MergeMethod = Field(default=MergeMethod.SLERP, description="Merge method")
    weights: Optional[List[float]] = Field(default=None, description="Weights for each model")
    output_path: Optional[str] = Field(default=None, description="Output path for merged model")

    # Method-specific parameters
    interpolation_t: float = Field(default=0.5, ge=0.0, le=1.0, description="SLERP interpolation t")
    density: float = Field(default=0.5, ge=0.0, le=1.0, description="TIES density")
    drop_rate: float = Field(default=0.9, ge=0.0, le=1.0, description="DARE drop rate")
    start_step: Optional[int] = Field(default=None, description="SWA start step")

    class Config:
        json_schema_extra = {
            "example": {
                "models": ["/path/to/base_model", "/path/to/finetuned_model"],
                "method": "slerp",
                "interpolation_t": 0.7,
                "output_path": "/path/to/output",
            }
        }


class MergeResponse(BaseModel):
    """Response for model merging"""
    success: bool
    output_path: Optional[str]
    method: str
    models_merged: List[str]
    weights_used: List[float]
    message: str
    metadata: Dict[str, Any]


class MergeScanRequest(BaseModel):
    """Request for scanning merge ratios"""
    model_a: str = Field(..., description="First model path")
    model_b: str = Field(..., description="Second model path")
    method: MergeMethod = Field(default=MergeMethod.SLERP, description="Merge method")
    num_points: int = Field(default=9, ge=3, le=20, description="Number of ratios to test")
    output_dir: Optional[str] = Field(default=None, description="Output directory")
    eval_dataset: Optional[str] = Field(default=None, description="Dataset for evaluation")


class MergeScanResult(BaseModel):
    """Single result from merge scan"""
    ratio: float
    success: bool
    output_path: Optional[str]
    message: str
    eval_score: Optional[float] = None


class MergeScanResponse(BaseModel):
    """Response for merge scan"""
    results: List[MergeScanResult]
    best_ratio: float
    best_score: Optional[float]


class CheckpointMetricsResponse(BaseModel):
    """Metrics for a checkpoint"""
    step: int
    path: str
    reward_mean: Optional[float]
    reward_std: Optional[float]
    kl_divergence: Optional[float]
    policy_loss: Optional[float]
    value_loss: Optional[float]
    entropy: Optional[float]
    benchmarks: Dict[str, Optional[float]]


class CheckpointSelectRequest(BaseModel):
    """Request for checkpoint selection"""
    experiment_path: str = Field(..., description="Path to experiment directory")
    criteria: SelectionCriteria = Field(default=SelectionCriteria.BALANCED, description="Selection criteria")
    custom_formula: Optional[str] = Field(default=None, description="Custom formula for CUSTOM criteria")
    benchmark_weights: Optional[Dict[str, float]] = Field(default=None, description="Weights for benchmarks")

    class Config:
        json_schema_extra = {
            "example": {
                "experiment_path": "/experiments/grpo-math-v1",
                "criteria": "balanced",
                "benchmark_weights": {"gsm8k": 0.5, "math": 0.5},
            }
        }


class OverfittingWarning(BaseModel):
    """Overfitting warning"""
    step: int
    type: str
    message: str
    severity: str


class CheckpointSelectResponse(BaseModel):
    """Response for checkpoint selection"""
    success: bool
    message: Optional[str] = None
    recommended: Optional[CheckpointMetricsResponse] = None
    score: Optional[float] = None
    criteria: Optional[str] = None
    reasoning: Optional[str] = None
    alternatives: Optional[List[CheckpointMetricsResponse]] = None
    timeline: Optional[List[CheckpointMetricsResponse]] = None
    overfitting_warnings: Optional[List[OverfittingWarning]] = None


class EMARequest(BaseModel):
    """Request for EMA weight extraction"""
    training_run_id: str = Field(..., description="Training run ID")
    decay: float = Field(default=0.999, ge=0.9, le=0.9999, description="EMA decay")
    save_path: Optional[str] = Field(default=None, description="Save path")


class EMAResponse(BaseModel):
    """Response for EMA extraction"""
    success: bool
    output_path: Optional[str]
    message: str
    decay: float


class SWARequest(BaseModel):
    """Request for SWA averaging"""
    checkpoint_paths: List[str] = Field(..., min_length=2, description="List of checkpoint paths")
    output_path: Optional[str] = Field(default=None, description="Output path")


class SWAResponse(BaseModel):
    """Response for SWA"""
    success: bool
    output_path: Optional[str]
    message: str
    num_checkpoints_averaged: int
