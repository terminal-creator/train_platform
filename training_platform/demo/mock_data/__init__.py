"""
Mock数据模块 - 提供高质量的演示数据

为演示模式提供完整的Mock数据，讲述一个完整的数学推理能力训练故事。
"""

from .jobs import (
    DEMO_JOBS, DEMO_JOB_UUIDS, BASE_TIME,
    get_demo_job, get_all_demo_jobs
)
from .metrics import get_demo_metrics, get_realtime_metrics, get_metrics_summary
from .datasets import (
    DEMO_DATASETS, DEMO_EVAL_DATASETS, DEMO_DATASET_SAMPLES,
    get_demo_dataset, get_demo_dataset_samples,
    get_all_demo_datasets, get_all_demo_eval_datasets,
)
from .checkpoints import DEMO_CHECKPOINTS, get_demo_checkpoints, get_best_checkpoint
from .gradients import get_gradient_heatmap, get_gradient_stats, get_gradient_health_report
from .evaluations import (
    DEMO_EVALUATIONS, get_demo_evaluation_results,
    get_evaluation_comparison, get_all_demo_evaluations,
)
from .surgery import (
    get_merge_result, get_checkpoint_selection,
    get_all_merge_results, get_all_swa_results,
    get_all_checkpoint_selections, get_all_rm_prompt_configs,
)
from .pipelines import (
    DEMO_PIPELINES, get_demo_pipeline,
    get_all_demo_pipelines, get_pipeline_templates,
)
from .compute import (
    get_demo_compute_result, get_gpu_types, get_model_sizes,
    DEMO_GPU_TYPES, DEMO_MODEL_SIZES,
)

__all__ = [
    # Jobs
    'DEMO_JOBS', 'DEMO_JOB_UUIDS', 'BASE_TIME',
    'get_demo_job', 'get_all_demo_jobs',
    # Metrics
    'get_demo_metrics', 'get_realtime_metrics', 'get_metrics_summary',
    # Datasets
    'DEMO_DATASETS', 'DEMO_EVAL_DATASETS', 'DEMO_DATASET_SAMPLES',
    'get_demo_dataset', 'get_demo_dataset_samples',
    'get_all_demo_datasets', 'get_all_demo_eval_datasets',
    # Checkpoints
    'DEMO_CHECKPOINTS', 'get_demo_checkpoints', 'get_best_checkpoint',
    # Gradients
    'get_gradient_heatmap', 'get_gradient_stats', 'get_gradient_health_report',
    # Evaluations
    'DEMO_EVALUATIONS', 'get_demo_evaluation_results',
    'get_evaluation_comparison', 'get_all_demo_evaluations',
    # Surgery
    'get_merge_result', 'get_checkpoint_selection',
    'get_all_merge_results', 'get_all_swa_results',
    'get_all_checkpoint_selections', 'get_all_rm_prompt_configs',
    # Pipelines
    'DEMO_PIPELINES', 'get_demo_pipeline',
    'get_all_demo_pipelines', 'get_pipeline_templates',
    # Compute
    'get_demo_compute_result', 'get_gpu_types', 'get_model_sizes',
    'DEMO_GPU_TYPES', 'DEMO_MODEL_SIZES',
]
