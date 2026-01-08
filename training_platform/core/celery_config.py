"""
Celery Configuration for Training Platform

Provides asynchronous task queue for long-running operations.
"""

import os
from celery import Celery
from kombu import Queue, Exchange

# Redis connection URL
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6381/0")

# Create Celery app
app = Celery(
    "training_platform",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

# Celery configuration
app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task execution settings
    task_track_started=True,
    task_time_limit=24 * 3600,  # 24 hours
    task_soft_time_limit=23 * 3600,  # 23 hours
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Result backend settings
    result_expires=7 * 24 * 3600,  # 7 days
    result_backend_transport_options={
        "master_name": "mymaster",
        "visibility_timeout": 3600,
    },

    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=True,

    # Queue settings
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",

    # Define queues with priorities
    task_queues=(
        Queue("default", Exchange("default"), routing_key="default", priority=5),
        Queue("training", Exchange("training"), routing_key="training.*", priority=10),
        Queue("evaluation", Exchange("evaluation"), routing_key="evaluation.*", priority=7),
        Queue("preprocessing", Exchange("preprocessing"), routing_key="preprocessing.*", priority=3),
        Queue("maintenance", Exchange("maintenance"), routing_key="maintenance.*", priority=1),
    ),

    # Task routing
    task_routes={
        "training_platform.core.celery_tasks.train_model": {
            "queue": "training",
            "routing_key": "training.main",
        },
        "training_platform.core.celery_tasks.run_evaluation": {
            "queue": "evaluation",
            "routing_key": "evaluation.run",
        },
        "training_platform.core.celery_tasks.preprocess_dataset": {
            "queue": "preprocessing",
            "routing_key": "preprocessing.data",
        },
        "training_platform.core.celery_tasks.cleanup_checkpoints": {
            "queue": "maintenance",
            "routing_key": "maintenance.cleanup",
        },
    },

    # Retry settings
    task_autoretry_for=(Exception,),
    task_retry_kwargs={"max_retries": 3, "countdown": 60},
    task_retry_backoff=True,
    task_retry_backoff_max=600,
    task_retry_jitter=True,

    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,

    # Beat scheduler (for periodic tasks)
    beat_schedule={
        "scan-failed-jobs": {
            "task": "training_platform.core.celery_tasks.scan_failed_jobs",
            "schedule": 300.0,  # Every 5 minutes
        },
        "cleanup-old-checkpoints": {
            "task": "training_platform.core.celery_tasks.cleanup_old_checkpoints",
            "schedule": 3600.0,  # Every hour
        },
        "update-job-metrics": {
            "task": "training_platform.core.celery_tasks.update_job_metrics",
            "schedule": 60.0,  # Every minute
        },
    },
)

# Auto-discover tasks
app.autodiscover_tasks(["training_platform.core"])

# Import modules to register their tasks
from . import pipeline_executor  # noqa: F401
from . import celery_tasks  # noqa: F401


# ============== Celery Signals for Pipeline Stage Tracking ==============

from celery.signals import task_prerun, task_success, task_failure


@task_prerun.connect
def track_pipeline_stage_start(sender=None, task_id=None, task=None, args=None, kwargs=None, **extra):
    """
    在 task 开始执行前自动记录 stage 状态

    通过 Celery signals 自动捕获 task 开始事件，检查 kwargs 中是否有：
    - _pipeline_uuid: Pipeline UUID
    - _stage_name: Stage name

    如果有，则调用 mark_stage_running 记录状态
    """
    if not kwargs:
        return

    pipeline_uuid = kwargs.get('_pipeline_uuid')
    stage_name = kwargs.get('_stage_name')

    if pipeline_uuid and stage_name:
        # Import here to avoid circular dependency
        from .pipeline_executor import mark_stage_running
        mark_stage_running(pipeline_uuid, stage_name, task_id)


if __name__ == "__main__":
    app.start()
