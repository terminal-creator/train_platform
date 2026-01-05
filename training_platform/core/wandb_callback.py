"""
WandB Integration Callback for Training Platform

Provides dual logging to both our platform and Weights & Biases.
This allows users to leverage W&B's visualization while also
populating our platform's monitoring dashboard.

Usage:
    from training_platform.core.wandb_callback import DualLogger

    logger = DualLogger(
        job_uuid="your-job-uuid",
        platform_url="http://localhost:8000",
        wandb_project="my-project",
        wandb_run_name="experiment-1",
    )

    # During training
    logger.log_metrics(step=100, metrics={"loss": 0.5, "reward": 0.8})
    logger.log_gpu_usage(gpu_index=0, utilization=95.0, memory_used=70.0, memory_total=80.0)

    # When done
    logger.finish()
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class PlatformConfig:
    """Configuration for Training Platform API"""
    job_uuid: str
    base_url: str = "http://localhost:8000"
    api_prefix: str = "/api/v1/monitoring"
    enabled: bool = True
    batch_size: int = 10  # Batch metrics before sending
    timeout: float = 5.0


@dataclass
class WandBConfig:
    """Configuration for Weights & Biases"""
    project: str
    run_name: Optional[str] = None
    entity: Optional[str] = None  # W&B team/user
    config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    enabled: bool = True
    log_model: bool = False  # Whether to log model checkpoints


class PlatformLogger:
    """
    Logger for Training Platform's monitoring API.

    Sends metrics via HTTP to our platform's push mode endpoints.
    """

    def __init__(self, config: PlatformConfig):
        self.config = config
        self._metrics_buffer: List[Dict] = []
        self._session = requests.Session()

    def _get_url(self, endpoint: str) -> str:
        return f"{self.config.base_url}{self.config.api_prefix}/{endpoint}"

    def log_metrics(self, step: int, metrics: Dict[str, Any], epoch: int = 0):
        """Log training metrics to platform"""
        if not self.config.enabled:
            return

        report = {
            "job_uuid": self.config.job_uuid,
            "step": step,
            "epoch": epoch,
            **self._normalize_metrics(metrics),
        }

        self._metrics_buffer.append(report)

        # Flush if buffer is full
        if len(self._metrics_buffer) >= self.config.batch_size:
            self.flush()

    def _normalize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize metric names to platform standard"""
        mapping = {
            # Common verl metric names
            "train/loss": "total_loss",
            "train/policy_loss": "policy_loss",
            "train/value_loss": "value_loss",
            "train/reward": "reward_mean",
            "train/reward_mean": "reward_mean",
            "train/reward_std": "reward_std",
            "train/kl": "kl_divergence",
            "train/kl_divergence": "kl_divergence",
            "train/entropy": "entropy",
            "critic/loss": "value_loss",
            "actor/loss": "policy_loss",
            # W&B common names
            "loss": "total_loss",
            "reward": "reward_mean",
            "kl": "kl_divergence",
        }

        normalized = {}
        extra_metrics = {}

        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue

            std_key = mapping.get(key)
            if std_key:
                normalized[std_key] = value
            else:
                extra_metrics[key] = value

        if extra_metrics:
            normalized["extra_metrics"] = extra_metrics

        return normalized

    def log_gpu_usage(
        self,
        gpu_index: int,
        utilization: float,
        memory_used: float,
        memory_total: float,
        temperature: Optional[float] = None,
    ):
        """Log GPU usage to platform"""
        if not self.config.enabled:
            return

        try:
            self._session.post(
                self._get_url("gpu"),
                json={
                    "job_uuid": self.config.job_uuid,
                    "gpu_index": gpu_index,
                    "utilization": utilization,
                    "memory_used": memory_used,
                    "memory_total": memory_total,
                    "temperature": temperature,
                },
                timeout=self.config.timeout,
            )
        except Exception as e:
            logger.warning(f"Failed to log GPU usage to platform: {e}")

    def log_status(self, status: str, message: Optional[str] = None):
        """Log job status change"""
        if not self.config.enabled:
            return

        try:
            self._session.post(
                self._get_url("status"),
                json={
                    "job_uuid": self.config.job_uuid,
                    "status": status,
                    "message": message,
                },
                timeout=self.config.timeout,
            )
        except Exception as e:
            logger.warning(f"Failed to log status to platform: {e}")

    def log_message(self, level: str, message: str):
        """Log a message to platform"""
        if not self.config.enabled:
            return

        try:
            self._session.post(
                self._get_url("log"),
                json={
                    "job_uuid": self.config.job_uuid,
                    "level": level,
                    "message": message,
                },
                timeout=self.config.timeout,
            )
        except Exception as e:
            logger.warning(f"Failed to log message to platform: {e}")

    def flush(self):
        """Flush buffered metrics to platform"""
        if not self._metrics_buffer:
            return

        try:
            self._session.post(
                self._get_url("report/batch"),
                json=self._metrics_buffer,
                timeout=self.config.timeout,
            )
            self._metrics_buffer = []
        except Exception as e:
            logger.warning(f"Failed to flush metrics to platform: {e}")

    def finish(self):
        """Finish logging and flush remaining metrics"""
        self.flush()
        self._session.close()


class WandBLogger:
    """
    Logger for Weights & Biases.

    Wraps the wandb API for consistent interface.
    """

    def __init__(self, config: WandBConfig):
        self.config = config
        self._run = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazily initialize wandb run"""
        if self._initialized or not self.config.enabled:
            return

        try:
            import wandb

            self._run = wandb.init(
                project=self.config.project,
                name=self.config.run_name,
                entity=self.config.entity,
                config=self.config.config,
                tags=self.config.tags,
                notes=self.config.notes,
                reinit=True,
            )
            self._initialized = True
            logger.info(f"W&B run initialized: {self._run.url}")
        except ImportError:
            logger.warning("wandb not installed. Run: pip install wandb")
            self.config.enabled = False
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            self.config.enabled = False

    def log_metrics(self, step: int, metrics: Dict[str, Any], **kwargs):
        """Log metrics to W&B"""
        if not self.config.enabled:
            return

        self._ensure_initialized()

        try:
            import wandb
            wandb.log(metrics, step=step, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to log metrics to wandb: {e}")

    def log_config(self, config: Dict[str, Any]):
        """Update run config"""
        if not self.config.enabled:
            return

        self._ensure_initialized()

        try:
            import wandb
            wandb.config.update(config)
        except Exception as e:
            logger.warning(f"Failed to update wandb config: {e}")

    def log_artifact(
        self,
        name: str,
        artifact_type: str,
        path: str,
        metadata: Optional[Dict] = None,
    ):
        """Log an artifact (model checkpoint, dataset, etc.)"""
        if not self.config.enabled or not self.config.log_model:
            return

        self._ensure_initialized()

        try:
            import wandb

            artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata)
            artifact.add_dir(path)
            self._run.log_artifact(artifact)
        except Exception as e:
            logger.warning(f"Failed to log artifact to wandb: {e}")

    def log_summary(self, summary: Dict[str, Any]):
        """Log summary metrics (final results)"""
        if not self.config.enabled:
            return

        self._ensure_initialized()

        try:
            import wandb
            for key, value in summary.items():
                wandb.run.summary[key] = value
        except Exception as e:
            logger.warning(f"Failed to log summary to wandb: {e}")

    def finish(self):
        """Finish the wandb run"""
        if not self._initialized:
            return

        try:
            import wandb
            wandb.finish()
        except Exception as e:
            logger.warning(f"Failed to finish wandb run: {e}")


class DualLogger:
    """
    Combined logger that sends metrics to both Training Platform and W&B.

    This is the main class to use for training scripts.

    Example:
        logger = DualLogger(
            job_uuid="abc123",
            platform_url="http://localhost:8000",
            wandb_project="my-project",
            wandb_config={"model": "Qwen2.5-7B", "algorithm": "GRPO"},
        )

        for step in range(1000):
            metrics = train_step()
            logger.log_metrics(step, metrics)

        logger.finish()
    """

    def __init__(
        self,
        # Platform config
        job_uuid: str,
        platform_url: str = "http://localhost:8000",
        platform_enabled: bool = True,
        # W&B config
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        wandb_tags: Optional[List[str]] = None,
        wandb_enabled: bool = True,
    ):
        # Initialize platform logger
        self.platform = PlatformLogger(
            PlatformConfig(
                job_uuid=job_uuid,
                base_url=platform_url,
                enabled=platform_enabled,
            )
        )

        # Initialize W&B logger if project specified
        self.wandb = WandBLogger(
            WandBConfig(
                project=wandb_project or "training-platform",
                run_name=wandb_run_name or job_uuid,
                entity=wandb_entity,
                config=wandb_config or {},
                tags=wandb_tags or [],
                enabled=wandb_enabled and wandb_project is not None,
            )
        )

        self.job_uuid = job_uuid
        self._step = 0

    def log_metrics(
        self,
        step: int,
        metrics: Dict[str, Any],
        epoch: int = 0,
        commit: bool = True,
    ):
        """
        Log training metrics to both platform and W&B.

        Args:
            step: Training step
            metrics: Dictionary of metric name -> value
            epoch: Current epoch
            commit: Whether to commit W&B logs immediately
        """
        self._step = step

        # Log to platform
        self.platform.log_metrics(step, metrics, epoch)

        # Log to W&B
        self.wandb.log_metrics(step, metrics, commit=commit)

    def log_gpu_usage(
        self,
        gpu_index: int,
        utilization: float,
        memory_used: float,
        memory_total: float,
        temperature: Optional[float] = None,
    ):
        """Log GPU usage (platform only)"""
        self.platform.log_gpu_usage(
            gpu_index, utilization, memory_used, memory_total, temperature
        )

        # Also log to W&B as metrics
        self.wandb.log_metrics(
            self._step,
            {
                f"gpu/{gpu_index}/utilization": utilization,
                f"gpu/{gpu_index}/memory_used_gb": memory_used,
                f"gpu/{gpu_index}/memory_pct": (memory_used / memory_total) * 100,
            },
            commit=False,
        )

    def log_status(self, status: str, message: Optional[str] = None):
        """Log job status change"""
        self.platform.log_status(status, message)

        # Log as W&B alert for important statuses
        if status in ("failed", "completed"):
            try:
                import wandb
                if status == "failed":
                    wandb.alert(
                        title=f"Training {status}",
                        text=message or f"Job {self.job_uuid} {status}",
                        level=wandb.AlertLevel.ERROR if status == "failed" else wandb.AlertLevel.INFO,
                    )
            except Exception:
                pass

    def log_checkpoint(
        self,
        step: int,
        checkpoint_path: str,
        eval_results: Optional[Dict[str, float]] = None,
    ):
        """Log a model checkpoint"""
        # Log to platform
        self.platform.log_message(
            "INFO",
            f"Checkpoint saved at step {step}: {checkpoint_path}"
        )

        # Log to W&B
        if eval_results:
            self.wandb.log_metrics(step, {f"eval/{k}": v for k, v in eval_results.items()})

        self.wandb.log_artifact(
            name=f"checkpoint-{step}",
            artifact_type="model",
            path=checkpoint_path,
            metadata={"step": step, "eval_results": eval_results},
        )

    def log_config(self, config: Dict[str, Any]):
        """Log training configuration"""
        self.wandb.log_config(config)

    def log_summary(self, summary: Dict[str, Any]):
        """Log final summary (best results, etc.)"""
        self.wandb.log_summary(summary)

    def finish(self, status: str = "completed"):
        """Finish logging session"""
        self.platform.log_status(status)
        self.platform.finish()
        self.wandb.finish()


# ============== verl Integration ==============


def create_verl_callback(
    job_uuid: str,
    platform_url: str = "http://localhost:8000",
    wandb_project: Optional[str] = None,
    wandb_config: Optional[Dict[str, Any]] = None,
):
    """
    Create a callback function compatible with verl training loop.

    Usage in verl config:
        trainer:
          callbacks:
            - training_platform.core.wandb_callback.create_verl_callback

    Or in code:
        from training_platform.core.wandb_callback import create_verl_callback

        callback = create_verl_callback(
            job_uuid="abc123",
            wandb_project="my-project",
        )

        # In training loop
        callback.on_train_step(step, metrics)
    """

    class VerlCallback:
        def __init__(self):
            self.logger = DualLogger(
                job_uuid=job_uuid,
                platform_url=platform_url,
                wandb_project=wandb_project,
                wandb_config=wandb_config,
            )

        def on_train_start(self, config: Dict[str, Any]):
            """Called at the start of training"""
            self.logger.log_config(config)
            self.logger.log_status("running", "Training started")

        def on_train_step(
            self,
            step: int,
            metrics: Dict[str, Any],
            epoch: int = 0,
        ):
            """Called after each training step"""
            self.logger.log_metrics(step, metrics, epoch)

        def on_eval_step(
            self,
            step: int,
            eval_results: Dict[str, float],
        ):
            """Called after evaluation"""
            self.logger.log_metrics(
                step,
                {f"eval/{k}": v for k, v in eval_results.items()},
            )

        def on_save_checkpoint(
            self,
            step: int,
            checkpoint_path: str,
            eval_results: Optional[Dict[str, float]] = None,
        ):
            """Called when saving a checkpoint"""
            self.logger.log_checkpoint(step, checkpoint_path, eval_results)

        def on_train_end(self, summary: Optional[Dict[str, Any]] = None):
            """Called at the end of training"""
            if summary:
                self.logger.log_summary(summary)
            self.logger.finish("completed")

        def on_train_error(self, error: Exception):
            """Called when training fails"""
            self.logger.log_status("failed", str(error))
            self.logger.finish("failed")

    return VerlCallback()


# ============== Convenience Functions ==============


def init_logging(
    job_uuid: str,
    platform_url: str = "http://localhost:8000",
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_config: Optional[Dict[str, Any]] = None,
) -> DualLogger:
    """
    Convenience function to initialize dual logging.

    Example:
        from training_platform.core.wandb_callback import init_logging

        logger = init_logging(
            job_uuid=os.environ.get("JOB_UUID", "local-test"),
            wandb_project="my-project",
            wandb_config={"model": "Qwen2.5-7B"},
        )

        for step, batch in enumerate(dataloader):
            loss = train_step(batch)
            logger.log_metrics(step, {"loss": loss})

        logger.finish()
    """
    return DualLogger(
        job_uuid=job_uuid,
        platform_url=platform_url,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_config=wandb_config,
    )
